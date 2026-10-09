"""Algebraic C-grid generators for the generic curvilinear operator layer.

A C-grid wraps AROUND the body (i-index) and stretches OUTWARD in the surface
normal (j-index: j=0 = wall, j=nj-1 = far field).  The two ends of the i-sweep
form the OPEN TAIL / WAKE CUT that is clamped to the uniform free stream; unlike
an O-grid the cut is NOT periodic, so the wake is resolved on the downstream side.

Two bodies:
  * cylinder  : full circle wrapped with the cut on the downstream axis (+x).  The
                wall is the ENTIRE j=0 circle (full pressure integration -> good
                drag), and the wake trails downstream to the far field.
  * naca      : 4-digit airfoil (0012 symmetric / 2412 cambered).  The j=0 line is
                the airfoil surface (upper TE -> LE -> lower TE) plus the two halves
                of the trailing-edge wake centreline; only the airfoil arc is wall.

Every generator returns (X, Y, wall, recv, Lref) where X,Y are the (ni+1, nj+1)
node arrays, wall/recv are (ni, nj) cell masks, and Lref is the reference length
(chord or diameter) for force normalisation.
"""
import numpy as np


# --------------------------------------------------------------------------- #
# NACA 4-digit geometry
# --------------------------------------------------------------------------- #
def _naca_yt(x, t):
    """Half-thickness distribution (NACA 4-digit, SHARP trailing edge).

    The standard last coefficient 0.1015 gives yt(1)=0.00126*t (a blunt TE).  For a
    C-grid the TE must COINCIDE with the wake centreline (y=0); using 0.1036 makes
    yt(1)=0 (sharp TE) so the airfoil-closure point equals the wake start and no
    degenerate sliver cells appear at the junction."""
    return 5.0 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2
                      + 0.2843 * x ** 3 - 0.1036 * x ** 4)


def _naca_camber(x, m, p):
    """Camber line yc and its slope dyc/dx (NACA 4-digit)."""
    if p <= 0.0 or m <= 0.0:
        return np.zeros_like(x), np.zeros_like(x)
    yc = np.where(x < p,
                  m / (p * p) * (2.0 * p * x - x * x),
                  m / ((1.0 - p) ** 2) * ((1.0 - 2.0 * p) + 2.0 * p * x - x * x))
    dyc = np.where(x < p,
                   2.0 * m / (p * p) * (p - x),
                   2.0 * m / ((1.0 - p) ** 2) * (p - x))
    return yc, dyc


def naca_surface(m, p, t, n=240):
    """Ordered airfoil surface points.

    Returns (surf, up, lo):
      surf : (2*n-1, 2) full ordered curve  upper TE -> LE -> lower TE
      up   : (n, 2)      upper surface TE -> LE
      lo   : (n, 2)      lower surface LE -> TE
    Cosine clustering in x gives smooth LE/TE resolution.
    """
    beta = np.linspace(0.0, np.pi, n)
    x = 0.5 * (1.0 - np.cos(beta))            # 0..1, cosine spaced
    yt = _naca_yt(x, t)
    yc, dyc = _naca_camber(x, m, p)
    th = np.arctan(dyc)
    xu = x - yt * np.sin(th); yu = yc + yt * np.cos(th)
    xl = x + yt * np.sin(th); yl = yc - yt * np.cos(th)
    up = np.column_stack([xu[::-1], yu[::-1]])        # TE -> LE (x decreasing)
    lo = np.column_stack([xl, yl])                     # LE -> TE (x increasing)
    surf = np.vstack([up, lo[1:]])                     # drop duplicated LE
    return surf, up, lo


# --------------------------------------------------------------------------- #
# shared helpers
# --------------------------------------------------------------------------- #
def _stretch_r(n_nodes, Rb, Rf, beta=3.0):
    """Algebraic radial stretching clustered at the wall (j=0): r(0)=Rb, r(1)=Rf,
    dr/dr smallest at the wall.  f(eta)=(exp(beta*eta)-1)/(exp(beta)-1).
    n_nodes = number of node LAYERS (nj_cells + 1)."""
    eta = np.linspace(0.0, 1.0, n_nodes)
    f = (np.exp(beta * eta) - 1.0) / (np.exp(beta) - 1.0)
    # r(j) measured FROM the wall; map to absolute coordinate via the body radius
    return Rb + (Rf - Rb) * f


def _geom_stretch(h_wall, Rf, n_nodes, g_lo=1.001, g_hi=6.0):
    """Near-wall GEOMETRIC radial stretching for y+~1 grids.  r[0]=0 (wall),
    r[1]=h_wall (first off-wall layer = target wall spacing), and the growth
    ratio g = r[j+1]/r[j] is held CONSTANT, solved by bisection so that the
    far-field node r[n_nodes-1] = Rf.  n_nodes = nj_cells + 1.  This gives a
    controlled first spacing (y+ = h_wall*u_tau/nu) with a smooth, monotone
    growth out to the far field -- the standard recipe for resolving a turbulent
    boundary layer without the wall-clustering blow-up of pure algebraic maps."""
    N = n_nodes - 1  # number of intervals

    def F(g):
        if abs(g - 1.0) < 1e-12:
            return h_wall * N - Rf
        return h_wall * (g ** N - 1.0) / (g - 1.0) - Rf

    if F(g_lo) > 0.0:
        g = 1.0 + 1e-9                      # even the smallest ratio overshoots
    else:
        lo, hi = g_lo, g_hi
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if F(mid) > 0.0:
                hi = mid
            else:
                lo = mid
        g = 0.5 * (lo + hi)
    j = np.arange(n_nodes)
    r = h_wall * (g ** j - 1.0) / (g - 1.0)
    return r



def _resample_curvature(curve, n, beta=1.0):
    """Resample `curve` to n+1 points whose arc-length DENSITY is proportional to
    (1 + beta*|kappa|*Ltot): more points where the surface curves sharply (LE,
    concave cove), fewer along the flat flanks.  This keeps the EXACT geometry --
    only the node distribution changes -- and is the key to a fold-free algebraic
    C-grid on the real GA(W)-2 slat/flap: uniform arc-length resampling wastes
    nodes on the flat flanks and under-resolves the tight LE/cove by ~15x, which
    lets the normal extrusion cross there and pinch sliver cells.  beta=0 recovers
    uniform spacing."""
    P = np.asarray(curve, float)
    x, y = P[:, 0], P[:, 1]
    d = np.hypot(np.diff(x), np.diff(y))
    s = np.concatenate([[0.0], np.cumsum(d)])
    stot = s[-1]
    if stot <= 0.0 or len(P) < 5 or n < 2:
        return _resample_uniform(P, n)
    dx = np.gradient(x); dy = np.gradient(y)
    ddx = np.gradient(dx); ddy = np.gradient(dy)
    denom = (dx * dx + dy * dy) ** 1.5
    denom[denom < 1e-14] = 1e-14
    kap = np.abs(dx * ddy - dy * ddx) / denom
    w = np.maximum(1.0 + beta * kap * stot, 1e-6)
    W = np.concatenate([[0.0], np.cumsum(0.5 * (w[:-1] + w[1:]) * d)])
    if W[-1] <= 0.0:
        return _resample_uniform(P, n)
    tgt = np.linspace(0.0, W[-1], n + 1)
    si = np.interp(tgt, W, s)
    return np.column_stack([np.interp(si, s, x), np.interp(si, s, y)])


def _resample_uniform(curve, n):
    """Resample an open (ni+1, 2) curve to exactly n+1 points equally spaced in
    arc length so the i-index has a near-uniform spacing."""
    d = np.sqrt(np.sum(np.diff(curve, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    snew = np.linspace(s[0], s[-1], n + 1)
    out = np.empty((n + 1, 2))
    out[:, 0] = np.interp(snew, s, curve[:, 0])
    out[:, 1] = np.interp(snew, s, curve[:, 1])
    return out


def _wake_stretch(n, Lw, h0=0.012):
    """Wake centreline stations from the TE (s=0, x=1) to the far cut (s=Lw) with
    GEOMETRIC spacing: the first station spacing is fixed to ~h0 (the fine airfoil
    arc spacing) and each subsequent spacing grows by a constant ratio so the whole
    wake is filled exactly to Lw.  This keeps the TE-junction cells isotropic
    (a plain exponential map with Lw=12 gives a near-TE spacing of ~0.18 -- 20x the
    arc spacing -- which collapses the junction cells and spikes the diffusion
    coefficient); a geometric map starts at ~0.012 and grows smoothly to the far
    field.  The ratio r is solved from  h0*(r^(n+1)-1)/(r-1) = Lw  by Newton."""
    r = 1.25
    for _ in range(60):
        rn = r ** (n + 1)
        f = h0 * (rn - 1.0) / (r - 1.0) - Lw
        fp = h0 * (((n + 1) * r ** n * (r - 1.0) - (rn - 1.0)) / (r - 1.0) ** 2)
        if not np.isfinite(fp) or fp == 0.0:
            break
        r = r - f / fp
        if r <= 1.0001:
            r = 1.05
            break
    k = np.arange(n + 1)
    return np.cumsum(h0 * r ** k)


def _elliptic_smooth(X, Y, n_iter=400, omega=1.6):
    """Relax the interior grid coordinates with a 5-point Laplacian (SOR) while
    holding the four physical boundaries fixed (j=0 wall, j=nj-1 far field, the two
    i=0 / i=ni wake cuts).  The algebraic C-grid near a sharp TE has small twists
    (the surface normal leans slightly downstream, so at large radius the off-body
    node x overshoots the wake line); a few hundred Laplacian iterations diffuse
    those kinks out and make the map X(i,j),Y(i,j) smooth.  The wall line (j=0) is
    preserved exactly so the airfoil shape and the force integral are untouched.

    IMPORTANT: this is RED-BLACK (Gauss-Seidel) SOR, the vectorised form of the
    in-place update.  A plain Jacobi update with over-relaxation (compute the whole
    new field from the old one, then over-relax) DIVERGES once omega > 2/(1+rho_J),
    which at the high aspect ratio of this grid is only just above 1 -- that was the
    earlier `J ~ 1e92` blow-up.  Red-black Gauss-Seidel is a contraction for every
    0 < omega < 2, so it converges (and 1<omega<2 accelerates it)."""
    ni, nj = X.shape[0] - 1, X.shape[1] - 1
    I, J = np.meshgrid(np.arange(1, ni), np.arange(1, nj), indexing="ij")
    iu = I.reshape(-1); ju = J.reshape(-1)
    even = ((I + J) % 2 == 0).reshape(-1)
    ri, rj = iu[even], ju[even]          # red point (i, j) coordinates
    bi, bj = iu[~even], ju[~even]        # black point (i, j) coordinates
    for _ in range(n_iter):
        # red pass: red points are not 4-adjacent, so all neighbours are old
        ax = 0.25 * (X[ri - 1, rj] + X[ri + 1, rj] + X[ri, rj - 1] + X[ri, rj + 1])
        ay = 0.25 * (Y[ri - 1, rj] + Y[ri + 1, rj] + Y[ri, rj - 1] + Y[ri, rj + 1])
        X[ri, rj] = (1.0 - omega) * X[ri, rj] + omega * ax
        Y[ri, rj] = (1.0 - omega) * Y[ri, rj] + omega * ay
        # black pass: red neighbours are already updated this sweep (Gauss-Seidel)
        ax = 0.25 * (X[bi - 1, bj] + X[bi + 1, bj] + X[bi, bj - 1] + X[bi, bj + 1])
        ay = 0.25 * (Y[bi - 1, bj] + Y[bi + 1, bj] + Y[bi, bj - 1] + Y[bi, bj + 1])
        X[bi, bj] = (1.0 - omega) * X[bi, bj] + omega * ax
        Y[bi, bj] = (1.0 - omega) * Y[bi, bj] + omega * ay
    return X, Y


def _winslow_smooth(X, Y, iters=1000, omega=1.0):
    """Winslow/Thompson elliptic grid smoothing: relax the interior nodes so the
    mapping (i,j)->(x,y) satisfies the harmonic equations IN THE COMPUTATIONAL
    domain,

        alpha * x_xi_xi - 2 beta * x_xi_eta + gamma * x_eta_eta = 0  (same for y)

    with metric coefficients alpha = |r_eta|^2, beta = r_xi . r_eta,
    gamma = |r_xi|^2 (r = (x,y)).  This is the correct elliptic operator on a
    strongly anisotropic body-fitted grid -- a plain physical-space Laplacian
    (x <- average of neighbours) smooths in the WRONG metric and cannot untangle
    a fold; Winslow can, because it weights the eta-direction by alpha=|r_eta|^2
    and the xi-direction by gamma=|r_xi|^2, i.e. it preserves the near-wall
    clustering while relaxing the distortion.

    Boundaries are held EXACTLY fixed: j=0 (the body wall -> airfoil shape and
    force integral untouched), j=nj-1 (far field) and the two wake-cut columns
    i=0 / i=ni.  On the 30P30N main element (which has a cove + sharp TE where
    the algebraic normal extrusion self-intersects) this removes every folded
    (J->0) cell and restores div(uniform)==0 to machine precision.

    Uses Jacobi updates with under-relaxation omega<=1 (over-relaxation >1
    DIVERGES on this stiff high-aspect-ratio system), vectorised over the whole
    interior each sweep."""
    X = X.copy(); Y = Y.copy()
    for _ in range(int(iters)):
        xe = X[2:, 1:-1] - X[:-2, 1:-1]
        ye = Y[2:, 1:-1] - Y[:-2, 1:-1]
        xn = X[1:-1, 2:] - X[1:-1, :-2]
        yn = Y[1:-1, 2:] - Y[1:-1, :-2]
        alpha = xn * xn + yn * yn
        gamma = xe * xe + ye * ye
        beta = xe * xn + ye * yn
        xe_n = 0.25 * (X[2:, 2:] - X[2:, :-2] - X[:-2, 2:] + X[:-2, :-2])
        ye_n = 0.25 * (Y[2:, 2:] - Y[2:, :-2] - Y[:-2, 2:] + Y[:-2, :-2])
        denom = 2.0 * (alpha + gamma)
        xnew = (alpha * (X[2:, 1:-1] + X[:-2, 1:-1])
                + gamma * (X[1:-1, 2:] + X[1:-1, :-2]) - 2.0 * beta * xe_n) / denom
        ynew = (alpha * (Y[2:, 1:-1] + Y[:-2, 1:-1])
                + gamma * (Y[1:-1, 2:] + Y[1:-1, :-2]) - 2.0 * beta * ye_n) / denom
        X[1:-1, 1:-1] = (1.0 - omega) * X[1:-1, 1:-1] + omega * xnew
        Y[1:-1, 1:-1] = (1.0 - omega) * Y[1:-1, 1:-1] + omega * ynew
    return X, Y


def _round_le(arc, win=0.06, niters=30):
    """Round the leading edge (the min-x point of the ordered arc) by LOCALISED
    Laplacian fairing so a sharp / zero-thickness LE gains finite thickness and
    the nose cells are no longer slivers.  The window is centred on the LE and
    tapered to zero at its edges, so the chord, TE and the rest of the contour are
    preserved exactly -- only the tip is faired.  A finite LE radius is physical
    (real high-lift sections are not mathematically sharp), and the radius implied
    here (~win * chord) is a fraction of a percent of chord, far below any
    aerodynamic sensitivity.  Without it the algebraic normal extrusion at a sharp
    LE pinches the upper/lower nose cells into ~1e-9-area slivers that blow the
    explicit solver up -- this is the dominant instability of the slat/flap."""
    a = np.array(arc, float)
    n = a.shape[0]
    ds = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(ds)]); stot = s[-1]
    iLE = int(np.argmin(a[:, 0]))
    sLE = s[iLE]
    half = max(2, int(round(win * n)))
    lo = max(1, iLE - half); hi = min(n - 2, iLE + half)
    idx = np.arange(lo, hi + 1)
    if len(idx) < 3:
        return a
    w = 1.0 - np.abs(s[idx] - sLE) / (win * stot)   # 1 at LE, 0 at window edge
    w = np.clip(w, 0.0, 1.0)
    for _ in range(niters):
        a2 = a.copy()
        lap = 0.5 * (a[idx - 1] + a[idx + 1]) - a[idx]
        a2[idx] = a[idx] + 0.5 * lap * w[:, None]
        a = a2
    return a


def _round_te(arc, thick=0.012, win=0.10, niters=25):
    """Blunt a (near) zero-thickness trailing edge in the chord-1 frame (TE at
    (1,0), chord along +x).  The two TE endpoints arc[0]/arc[-1] are separated to a
    finite thickness `thick` (fraction of chord) and the nearby contour is
    Laplacian-faired toward them (tapered window `win`).  A finite TE radius is
    physical (real high-lift sections are not mathematically sharp) and removes
    the wake-cut cell pinch at the TE junction -- the tiny cell where the airfoil
    arc meets the wake centreline -- that seeds the explicit-solver blow-up on the
    small slat/flap (the slat/flap TE is sharp; the reference main is a blunt NACA
    TE, which is why main survives).  Only the tip region is touched; the rest of
    the contour (and thus the forces) is preserved."""
    a = np.array(arc, float)
    n = a.shape[0]
    a[0] = np.array([1.0, -0.5 * thick])
    a[-1] = np.array([1.0, 0.5 * thick])
    ds = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(ds)]); stot = s[-1]
    half = max(2, int(round(win * n)))
    for _ in range(niters):
        a2 = a.copy()
        for end in ("lo", "hi"):
            if end == "lo":
                idx = np.arange(1, min(half + 1, n - 1))
                sc = s[idx]
            else:
                idx = np.arange(max(1, n - 1 - half), n - 1)
                sc = stot - s[idx]
            if len(idx) < 2:
                continue
            w = np.clip(1.0 - sc / (win * stot), 0.0, 1.0)
            lap = 0.5 * (a[idx - 1] + a[idx + 1]) - a[idx]
            a2[idx] = a[idx] + 0.5 * lap * w[:, None]
        a = a2
    return a


def _fair_arc(arc, iters=6, keep_le=True):
    """Mild global fairing of the ordered wall arc by a few Laplacian passes so
    small KINKS (tangent discontinuities) in the digitised contour no longer make
    the surface normal jump -> which pinches the first off-wall layer into a
    sliver.  The GA(W)-2 slat/flap contours carry such kinks on the lower surface
    near the TE (a slope change of ~30-40 deg over one node) where the raw data
    switches from cove to aft-flank; fairing rounds them over a couple of nodes.
    The TE endpoints (arc[0]==arc[-1]) are held fixed so the wake still attaches to
    the exact TE; the shape change is O(iters * node spacing^2 * curvature) and is
    a small fraction of a percent of chord -- far below aerodynamic sensitivity.
    iters<=0 returns the arc unchanged."""
    a = np.array(arc, float)
    M = a.shape[0]
    if iters is None or iters <= 0 or M < 5:
        return a
    p0, p1 = a[0].copy(), a[-1].copy()
    for _ in range(int(iters)):
        a[1:-1] = 0.5 * (a[:-2] + a[2:])
        a[0], a[-1] = p0, p1
    return a


def _fair_arc_windowed(arc, centers, half=5, iters=8, relax=0.5):
    """LOCALISED (windowed) Laplacian fairing over one or more windows.  `centers`
    is an int, a (center, half) pair, or a list of those.  Inside each window
    [center-half, center+half] the wall arc nodes are relaxed with a raised-cosine
    weight (1 at centre -> 0 at edge, C1 transition, no new kink); windows are
    summed so overlapping bands are covered.  Everything outside the windows -- the
    lift-generating camber / thickness -- is left EXACTLY as the real GA(W)-2
    contour, so forces are preserved.  This targets the slat/flap/main near-wall
    PINCH BANDS (local thin-strip cells, J~1e-7) that the GLOBAL _fair_arc also
    removes but only by smearing the whole contour and killing the lift.  TE
    endpoints held fixed.  iters<=0 returns the arc unchanged."""
    a = np.array(arc, float)
    M = a.shape[0]
    if iters is None or iters <= 0 or M < 5:
        return a
    p0, p1 = a[0].copy(), a[-1].copy()
    w = np.zeros(M)
    if isinstance(centers, (list, tuple)) and len(centers) > 0 \
            and isinstance(centers[0], (list, tuple)):
        pairs = [(int(c), int(h)) for c, h in centers]
    elif isinstance(centers, (list, tuple)):
        pairs = [(int(c), int(half)) for c in centers]
    else:
        pairs = [(int(centers), int(half))]
    for (c, h) in pairs:
        if c < 0:
            continue
        for k in range(M):
            d = abs(k - c)
            if d <= h:
                w[k] = max(w[k], 0.5 * (1.0 + np.cos(np.pi * d / h)))
    for _ in range(int(iters)):
        lap = 0.5 * (a[:-2] + a[2:]) - a[1:-1]
        a[1:-1] += relax * w[1:-1, None] * lap
        a[0], a[-1] = p0, p1
    return a


def _fill_te_bridge(arc, frac=0.16):
    """Replace the cove-floor / sharp-trailing-edge notch of an ordered wall arc
    (localised chord-1 frame, TE at index 0, going TE -> LE -> TE) with a smooth
    bridge from the TE (arc[0]) to the arc point at fractional arc-length `frac`.
    This removes the re-entrant corner / cove that makes algebraic normal
    extrusion self-intersect (a single-block C-grid cannot resolve a concave
    cove -> the extruded layers cross -> sliver / inverted cells -> the explicit
    solver diverges).  The bridge is a straight chord (optionally a tiny outward
    bulge); the rest of the (exact) contour is untouched, so only the small cove
    region is faired.  Returns the (possibly) modified arc."""
    a = np.array(arc, float)
    ds = np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    stot = s[-1]
    starget = frac * stot
    k_low = int(np.searchsorted(s, starget))
    k_low = max(k_low, 3)
    if k_low >= len(a) - 1:
        return a
    p0 = a[0]; p1 = a[k_low]
    for k in range(1, k_low):
        t = k / k_low
        a[k] = (1.0 - t) * p0 + t * p1
    return a


def _recluster_wall(X, Y, r_target, Rf):
    """Re-impose a geometric near-wall clustering on a (ni+1, nj+1) grid that has
    ALREADY been made fold-free by Winslow but whose interior spacing has been
    flattened by the boundary-determined elliptic solve.

    For every streamline i (constant i, j=0..nj) the Winslow-smoothed line is a
    smooth 1-1 curve from the wall (j=0) out to the far field (j=nj).  We walk
    that curve and RE-POSITION each interior node at the ABSOLUTE radius
    `r_target[j]` measured from the wall point -- i.e. we invert the (monotonised)
    radius-vs-arclength relation of the smooth line and drop the node at the
    target radius.  Because every streamline is placed at the SAME absolute radii,
    adjacent lines (which are non-crossing on the smooth map) stay non-crossing,
    so the untangling is preserved; and the first off-wall radius is exactly
    r_target[1] -> uniform y+.  j=0 (radius 0) and j=nj (radius Rf) are pinned to
    the wall and far-field points, so the airfoil shape and far field are intact.

    This is the practical equivalent of a control-function (Hilgenstock) elliptic
    generator done as two robust decoupled steps: (1) Winslow untangles, (2)
    radius re-clustering restores the wall layer.

    r_target : radial stations.  Either a 1D (nj+1,) array applied to EVERY
               streamline (the historical behaviour), OR a 2D (ni+1, nj+1) array
               giving a PER-STREAMLINE station set.  The 2D form is used to
               decouple the wake-cut spacing from the airfoil y+ spacing (the
               30P30N stability fix): the airfoil keeps a tiny y+50 first layer
               while the non-wall wake cut gets a thicker isotropic first layer.
    Rf       : far-field radial extent (r_target[-1] == Rf on every streamline)."""
    """Re-impose a geometric near-wall clustering on a (ni+1, nj+1) grid that has
    ALREADY been made fold-free by Winslow but whose interior spacing has been
    flattened by the boundary-determined elliptic solve.

    For every streamline i (constant i, j=0..nj) the Winslow-smoothed line is a
    smooth 1-1 curve from the wall (j=0) out to the far field (j=nj).  We walk
    that curve and RE-POSITION each interior node at the ABSOLUTE radius
    `r_target[j]` measured from the wall point -- i.e. we invert the (monotonised)
    radius-vs-arclength relation of the smooth line and drop the node at the
    target radius.  Because every streamline is placed at the SAME absolute radii,
    adjacent lines (which are non-crossing on the smooth map) stay non-crossing,
    so the untangling is preserved; and the first off-wall radius is exactly
    r_target[1] -> uniform y+.  j=0 (radius 0) and j=nj (radius Rf) are pinned to
    the wall and far-field points, so the airfoil shape and far field are intact.

    This is the practical equivalent of a control-function (Hilgenstock) elliptic
    generator done as two robust decoupled steps: (1) Winslow untangles, (2)
    radius re-clustering restores the wall layer.

    r_target : radial stations.  Either a 1D (nj+1,) array applied to EVERY
               streamline (the historical behaviour), OR a 2D (ni+1, nj+1) array
               giving a PER-STREAMLINE station set.  The 2D form is used to
               decouple the wake-cut spacing from the airfoil y+ spacing (the
               30P30N stability fix): the airfoil keeps a tiny y+50 first layer
               while the non-wall wake cut gets a thicker isotropic first layer.
    Rf       : far-field radial extent (r_target[-1] == Rf on every streamline)."""
    X = X.copy(); Y = Y.copy()
    ni, nj = X.shape[0] - 1, X.shape[1] - 1
    rt = np.asarray(r_target, float)
    per_col = (rt.ndim == 2)            # 2D -> per-streamline stations
    for i in range(ni + 1):
        rti = rt[i] if per_col else rt
        xs = X[i, :]; ys = Y[i, :]
        wx, wy = xs[0], ys[0]                       # wall point
        ds = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
        s = np.concatenate([[0.0], np.cumsum(ds)])  # arclength from wall
        rho = np.hypot(xs - wx, ys - wy)            # radius from wall (monotonised)
        rho_m = np.maximum.accumulate(rho)          # force non-decreasing
        # positions along the line where the radius equals each target station
        tgt = np.clip(rti, rho_m[0], rho_m[-1])
        si = np.interp(tgt, rho_m, s)               # arclength at target radius
        X[i, :] = np.interp(si, s, xs)
        Y[i, :] = np.interp(si, s, ys)
    return X, Y


def _curve_normals(curve):
    """Outward normals of an open curve via central differences of the tangent,
    rotated +90 deg.  Sign convention handled by the caller per segment."""
    t = np.gradient(curve, axis=0)                    # tangent
    n = np.column_stack([t[:, 1], -t[:, 0]])          # rotate -90? then normalise
    ln = np.hypot(n[:, 0], n[:, 1])
    n = n / ln[:, None]
    return n


def _smooth_normals(n, niters=30, ends="reflect"):
    """Iteratively average an (M,2) unit-normal field with its neighbours and
    re-normalise.  This removes the sharp FLIP between adjacent outward normals
    that occurs at high-curvature surface points (e.g. the lower-surface cove of a
    GA(W)-2 slat/flap): there the true geometric normals of two nearby wall nodes
    point in nearly OPPOSITE directions, so algebraic normal extrusion crosses the
    two layers and pinches the quad into a ~1e-9-area sliver that blows the explicit
    solver up.  Smoothing the *direction field* (NOT the wall itself -- j=0 stays
    exact) makes adjacent normals consistent so the off-wall layers never cross.
    Endpoints are held (reflective) so the LE/TE normals are preserved; the
    smoothing is localised to the sharp turn and leaves mildly-curved regions
    essentially normal to the wall."""
    n = np.array(n, float)
    M = n.shape[0]
    for _ in range(niters):
        ns = n.copy()
        ns[1:-1] = 0.5 * (n[:-2] + n[2:])            # central average (Laplacian)
        # endpoints: reflect neighbour so the turn is rounded, not flattened
        if ends == "reflect":
            if M >= 2:
                ns[0] = n[1]
                ns[-1] = n[-2]
        # re-normalise every node (keep unit length; only direction is smoothed)
        ln = np.hypot(ns[:, 0], ns[:, 1])
        ln[ln < 1e-12] = 1.0
        ns /= ln[:, None]
        n = ns
    return n


def _arc_normals(arc, smooth_iters=0):
    """Outward-unit-normal DIRECTION field of an ordered wall arc.

    The robust trick: low-pass the *curve positions* first (a plain central
    average -- positions are well conditioned and never anti-align), THEN take the
    tangent / normal.  Averaging the normal *vectors* directly fails at a ~180 deg
    flip (two opposite unit vectors average to ~0), which is exactly what happens
    at the tight lower-surface cove of a GA(W)-2 slat/flap; smoothing the curve
    instead makes the tangent rotate smoothly through the kink, so the extruded
    off-wall layers never cross.  smooth_iters=0 recovers the raw central-difference
    normals.  NOTE: this only smooths the DIRECTION field; the wall (j=0) keeps its
    exact coordinates because the caller adds r[j]*n to the untouched arc."""
    A = np.asarray(arc, float).copy()
    for _ in range(int(smooth_iters)):
        if A.shape[0] < 3:
            break
        A[1:-1] = 0.5 * (A[:-2] + A[2:])
    t = np.gradient(A, axis=0)
    n = np.column_stack([t[:, 1], -t[:, 0]])
    ln = np.hypot(n[:, 0], n[:, 1])
    ln[ln < 1e-12] = 1.0
    return n / ln[:, None]


# --------------------------------------------------------------------------- #
# Cylinder C-grid
# --------------------------------------------------------------------------- #
def make_cylinder_cgrid(Rb=1.0, Rf=20.0, ni=181, nj=71, beta=3.0,
                        cut_deg=2.0, centre=(0.0, 0.0)):
    """C-grid around a cylinder of radius Rb with the open cut on the downstream
    (+x) axis.  The j=0 line is the FULL circle (wall), so the pressure drag is
    integrated over the whole body; the wake trails downstream and is resolved up
    to the far field (the two cut columns are clamped to free stream at x>+Rf).

    i sweeps the body angle phi from -cut (just below +x) clockwise around the
    front (-x) to +cut (just above +x); both ends meet the downstream axis and
    extrude along +x as the wake cut.
    """
    cx, cy = centre
    cut = np.deg2rad(cut_deg)
    # body angle phi in (-pi, pi], cut at phi=0 (+x downstream)
    phi = -cut - np.linspace(0.0, 2.0 * np.pi - 2.0 * cut, ni + 1)
    bx = cx + Rb * np.cos(phi)
    by = cy + Rb * np.sin(phi)
    body = np.column_stack([bx, by])                  # (ni+1, 2) j=0 wall line
    # radial outward normal at every body point
    nrm = np.column_stack([np.cos(phi), np.sin(phi)])  # points away from centre
    r = _stretch_r(nj + 1, 0.0, Rf, beta=beta)        # stretch measured from wall
    X = np.zeros((ni + 1, nj + 1)); Y = np.zeros((ni + 1, nj + 1))
    for j in range(nj + 1):
        X[:, j] = body[:, 0] + r[j] * nrm[:, 0]
        Y[:, j] = body[:, 1] + r[j] * nrm[:, 1]
    # wall = all j=0 (full circle); recv = far field + the two cut columns
    wall = np.zeros((ni, nj), dtype=bool); wall[:, 0] = True
    recv = np.zeros((ni, nj), dtype=bool)
    recv[:, nj - 1] = True
    recv[0, :] = True; recv[ni - 1, :] = True
    return X, Y, wall, recv, 2.0 * Rb


# --------------------------------------------------------------------------- #
# NACA C-grid
# --------------------------------------------------------------------------- #
def make_naca_cgrid(m=0.0, p=0.0, t=0.12, ni=281, nj=81, Rf=20.0, beta=3.0,
                    Lw=None, n_surf=200, centre=(0.0, 0.0), h_wall_abs=None):
    """C-grid around a NACA 4-digit airfoil (chord c=1, LE near origin, TE at x=1).

    The j=0 line is built from three independent segments so every normal is exact:

        [upper-wake: wake_end -> TE]  [airfoil: TE -> LE -> TE]  [lower-wake: TE -> wake_end]

    Airfoil normals are the true outward surface normals (from the surface tangent);
    the two wake halves use the constant normals (+y upper, -y lower).  Only the
    airfoil arc (n_wake <= i < n_wake+n_arc) is no-slip wall; the wake cut and the
    far field are free stream.

    m,p,t : camber slope/position + thickness  (0012 -> 0,0,0.12 ; 2412 -> 0.02,0.4,0.12)
    """
    cx, cy = centre
    _, up, lo = naca_surface(m, p, t, n_surf)
    te = np.array([1.0, 0.0]); le = np.array([0.0, 0.0])
    if Lw is None:
        Lw = Rf
    wake_end = np.array([1.0 + Lw, 0.0])
    # cell allocation: wake halves ~16% each, airfoil the rest
    n_wake = max(8, int(round(0.16 * ni)))
    n_arc = ni - 2 * n_wake
    if n_arc < 24:
        n_wake = max(5, ni // 12); n_arc = ni - 2 * n_wake
    # airfoil arc TE -> LE -> TE (up is TE->LE, lo is LE->TE)
    arc0 = np.vstack([up, lo[1:]])
    arc = _resample_uniform(arc0, n_arc)              # (n_arc+1, 2)
    # airfoil outward normals
    t_arc = np.gradient(arc, axis=0)
    n_arc_v = np.column_stack([t_arc[:, 1], -t_arc[:, 0]])
    ln = np.hypot(n_arc_v[:, 0], n_arc_v[:, 1]); n_arc_v /= ln[:, None]
    for k in range(n_arc_v.shape[0]):                 # orient away from chord mid
        if np.dot(n_arc_v[k], arc[k] - np.array([0.5, 0.0])) < 0.0:
            n_arc_v[k] = -n_arc_v[k]
    # Localised clamp of the downstream (+x) normal component, restricted to the
    # AFT airfoil (x > x_te_clamp) so the first off-wall layer never bulges past
    # the trailing edge (x=1) and folds the i-sweep (the classic C-grid "TE kink").
    # Only the aft region is touched -- the leading edge (where the true normal is
    # already ~(-1,0), upstream) is left EXACTLY as the geometric normal, which
    # avoids the zero-area collapse a global clamp would cause at the LE.  Near the
    # TE the clamped normals become +/-y, matching the wake-centreline normals.
    x_te_clamp = 0.30
    aft = arc[:, 0] > x_te_clamp
    n_arc_v[aft, 0] = np.maximum(n_arc_v[aft, 0], 0.0)
    ln = np.hypot(n_arc_v[aft, 0], n_arc_v[aft, 1])
    n_arc_v[aft, 0] /= ln; n_arc_v[aft, 1] /= ln
    # wake halves (kept as separate segments -> clean normals)
    # wake halves: STRETCHED (dense near TE) so the near-TE spacing matches the
    # fine airfoil spacing and the TE-junction cells stay well shaped.
    wpos = _wake_stretch(n_wake, Lw)                # TE(s=0, x=1) -> far (x=1+Lw)
    uw = np.column_stack([1.0 + wpos[::-1], np.zeros(n_wake + 1)])  # far -> TE
    lw = np.column_stack([1.0 + wpos, np.zeros(n_wake + 1)])        # TE -> far
    # assemble full j=0 curve (n_wake + n_arc+1 + n_wake = ni+1 nodes)
    curve = np.vstack([uw[:-1], arc, lw[1:]]) + np.array([cx, cy])
    nrm = np.zeros_like(curve)
    nrm[:n_wake] = np.array([0.0, 1.0])               # upper wake
    nrm[n_wake:n_wake + n_arc + 1] = n_arc_v          # airfoil (true surface normal)
    nrm[n_wake + n_arc + 1:] = np.array([0.0, -1.0])  # lower wake
    # algebraic wall-normal stretching
    if h_wall_abs is None:
        r = _stretch_r(nj + 1, 0.0, Rf, beta=beta)
    else:
        # wall-function grid: geometric stretch with first off-wall layer =
        # h_wall_abs (absolute; NACA chord = 1 so no scaling needed).  Matches the
        # make_contour_cgrid wall-function path used by the 30P30N production rig.
        r = _geom_stretch(h_wall_abs, Rf, nj + 1)
    X = np.zeros((ni + 1, nj + 1)); Y = np.zeros((ni + 1, nj + 1))
    for j in range(nj + 1):
        X[:, j] = curve[:, 0] + r[j] * nrm[:, 0]
        Y[:, j] = curve[:, 1] + r[j] * nrm[:, 1]
    # NOTE: elliptic (Laplacian SOR) smoothing of the interior is intentionally NOT
    # applied.  On this highly-anisotropic C-grid (fine near-wall / wake spacing vs
    # Rf=20 far field) SOR collapses cells at the leading edge and in the far field
    # (Jmin drops ~50x, Jmax ~5x) and only worsens the map.  With the surface-normal
    # clamp above the algebraic grid is already valid (J>0 everywhere, div(uniform)
    # == 0 to machine precision) so no post-smoothing is needed.  _elliptic_smooth is
    # kept for reference / other bodies.
    # wall = airfoil arc cells only
    wall = np.zeros((ni, nj), dtype=bool)
    wall[n_wake:n_wake + n_arc, 0] = True
    recv = np.zeros((ni, nj), dtype=bool)
    recv[:, nj - 1] = True
    recv[0, :] = True; recv[ni - 1, :] = True
    return X, Y, wall, recv, 1.0


# --------------------------------------------------------------------------- #
# Three-element (high-lift) airfoil: a SLAT + MAIN + FLAP, each its own C-grid
# --------------------------------------------------------------------------- #
def _transform_cgrid(X, Y, theta_deg, tx, ty, scale=1.0):
    """Scale by `scale`, rotate the (ni+1,nj+1) node arrays by theta_deg about the
    local origin, then translate by (tx,ty).  make_naca_cgrid always builds a
    chord=1 airfoil, so the element chord is applied here as `scale`.  wall/recv
    are cell masks (index-only) so they pass through unchanged -- only X,Y move."""
    th = np.deg2rad(theta_deg)
    c, s = np.cos(th), np.sin(th)
    Xr = scale * (c * X - s * Y) + tx
    Yr = scale * (s * X + c * Y) + ty
    return Xr, Yr


def make_three_element_cgrid(slat=(0.30, 0.0, 0.0, 0.12, -20.0, -0.2379, 0.1284),
                              main=(1.0, 0.02, 0.4, 0.12, 0.0, 0.0, 0.0),
                              flap=(0.50, 0.0, 0.0, 0.12, -35.0, 0.95, -0.04),
                              ni_s=121, nj_s=51, ni_m=181, nj_m=61,
                              ni_f=121, nj_f=51, Rf_base=18.0, beta=3.0,
                              Lw_m=None, n_surf=200):
    """Generate a three-element high-lift configuration as THREE independent
    body-fitted C-grids (slat / main / flap), each solved separately and coupled
    later by overset (see tensorlbm.overset).  Returns a dict:

        {
          'elements': [  # ordered main, slat, flap (index 0 = MAIN = reference)
              {'name', 'X','Y','wall','recv','Lref','ni','nj','theta','t'},
              ...
          ],
          'Lref',          # reference length = main chord (for force normalisation)
          'placement',     # echo of the input placement tuples
        }

    Each element tuple is (chord, m, p, t_thick, theta_deg, tx, ty):
      * a NACA (m,p,t_thick) airfoil of the given chord is built in LOCAL coords
        (LE at (0,0), TE at (chord,0), C-grid wake trailing +x),
      * then rotated by theta_deg (negative = nose-down / trailing-edge-up deflected
        downward, the usual high-lift setting) and translated by (tx,ty) so the
        element lands at its slot.  The far field (Rf) of every element is large
        enough that the three grids OVERLAP -- adjacent elements cover each other's
        outer zones, giving the overset layer a donor region to interpolate from.

    Default placement is a sensible static high-lift rig (gap + overlap between
    slat/main and main/flap).  Slat sits ahead-and-above the main LE, flap
    behind-and-below the main TE, both deflected downward to generate the slot
    flow that produces the high-lift increment.  AoA is applied later as a rotated
    free stream (NOT baked into the geometry), so the chord axis stays horizontal.
    """
    # Per-element Rf is set to Rf_base * chord so that EVERY element's grid has
    # the SAME local radial extent (Rf_local = Rf_base), hence the SAME near-wall
    # cell size AFTER the chord scaling in _transform_cgrid.  This keeps the
    # tiny slat/flap on the SAME absolute near-wall scale as the reference main,
    # so their discrete divergence / wake-junction cells are well shaped and they
    # stay stable (the old single Rf=18 gave Rf_local=18/chord ~ 60 for a small
    # slat -> ~1e-5-area junction cells -> div(uniform)~1e2 -> divergence).
    # The (absolute) far field of each sub-element is Rf_base*chord, which still
    # overlaps the main (Rf=18) so the overset layer has a donor region.
    cfgs = {
        "main": dict(chord=main[0], m=main[1], p=main[2], t=main[3],
                     theta=main[4], tx=main[5], ty=main[6], ni=ni_m, nj=nj_m,
                     Rf=Rf_base * main[0], Lw=Lw_m),
        "slat": dict(chord=slat[0], m=slat[1], p=slat[2], t=slat[3],
                     theta=slat[4], tx=slat[5], ty=slat[6], ni=ni_s, nj=nj_s,
                     Rf=Rf_base * slat[0], Lw=None),
        "flap": dict(chord=flap[0], m=flap[1], p=flap[2], t=flap[3],
                     theta=flap[4], tx=flap[5], ty=flap[6], ni=ni_f, nj=nj_f,
                     Rf=Rf_base * flap[0], Lw=None),
    }
    order = ["main", "slat", "flap"]
    elements = []
    for name in order:
        c = cfgs[name]
        # Rf is ABSOLUTE radial extent; make_naca_cgrid builds in local chord-1
        # coords and _transform_cgrid scales by chord, so the local Rf must be
        # Rf/chord to keep every element's near-wall cell size (and thus its
        # discrete-divergence / junction quality) on the same absolute scale as
        # the reference main -- otherwise the tiny slat/flap get ~1e-5-area
        # junction cells whose div(uniform) reaches ~1e2 and destabilises them.
        Rf_local = c["Rf"] / c["chord"]
        X, Y, wall, recv, Lref = make_naca_cgrid(
            m=c["m"], p=c["p"], t=c["t"], ni=c["ni"], nj=c["nj"],
            Rf=Rf_local, beta=beta, Lw=c["Lw"], n_surf=n_surf)
        X, Y = _transform_cgrid(X, Y, c["theta"], c["tx"], c["ty"],
                                scale=c["chord"])
        elements.append(dict(name=name, X=X, Y=Y, wall=wall, recv=recv,
                             Lref=c["chord"], ni=c["ni"], nj=c["nj"],
                             theta=c["theta"], t=(c["tx"], c["ty"])))
    return dict(elements=elements, Lref=cfgs["main"]["chord"],
                placement=dict(slat=slat, main=main, flap=flap))


# --------------------------------------------------------------------------- #
# Real-airfoil C-grid: wrap a C-grid around an ARBITRARY closed contour
# (e.g. the genuine 30P30N GA(W)-2 element shapes imported from coordinate
# files) instead of a NACA 4-digit section.  The chord is auto-detected and the
# contour is localised (LE->origin, chord->+x, chord-normalised) so the standard
# C-grid machinery in make_naca_cgrid (which assumes the wake trails +x) applies;
# the finished grid is then rigidly mapped back to its absolute pose.  This path
# is fully independent of the NACA path so the validated rig is untouched.
# --------------------------------------------------------------------------- #
def _detect_chord(contour):
    """Return (LE, TE, c, ang) for a closed airfoil contour (M,2).

    The chord direction is the contour's principal axis (PCA): for a thin body
    the first eigenvector IS the chord, which is robust even for cambered /
    deflected sections whose cove makes a naive max-separation or min/max-x test
    pick the wrong pair and yield a degenerate (J->0) grid.  LE / TE are the
    contour vertices at the two extremes of the chord-wise projection."""
    P = np.asarray(contour, float)
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[0]])                    # close the loop
    mu = P.mean(0)
    cov = (P - mu).T @ (P - mu) / len(P)
    w, V = np.linalg.eigh(cov)
    pc1 = V[:, int(np.argmax(w))]                   # chord direction (unit)
    if pc1[0] < 0.0:
        pc1 = -pc1                                  # eigenvector sign is arbitrary:
        #                                             force chord to point +x so
        #                                             LE is left, TE is right
    proj = P @ pc1
    ile = int(np.argmin(proj)); ite = int(np.argmax(proj))
    LE, TE = P[ile], P[ite]
    c = float(np.hypot(TE[0] - LE[0], TE[1] - LE[1]))
    ang = np.arctan2(TE[1] - LE[1], TE[0] - LE[0])
    return LE, TE, c, ang


def _order_contour_loop(contour, n_arc, curv_beta=0.0):
    """Order a closed airfoil contour as a clean TE -> LE -> TE wall arc of
    n_arc+1 points (original frame).  Instead of splitting upper/lower by the
    chord line (which fails for cambered/coved sections that cross the chord
    line off the ends), this traverses the actual topological loop: resample by
    arc length, find TE as the chord-wise extreme, then walk the loop from TE
    around through LE and back to TE.  Robust for any simple closed airfoil."""
    P = np.asarray(contour, float)
    if not np.allclose(P[0], P[-1]):
        P = np.vstack([P, P[0]])                    # close the loop
    d = np.sqrt(np.sum(np.diff(P, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(d)])
    sd = np.linspace(0.0, s[-1], 4 * n_arc + 1)
    Pd = np.column_stack([np.interp(sd, s, P[:, 0]), np.interp(sd, s, P[:, 1])])
    # chord direction (PCA) to locate TE
    mu = Pd.mean(0)
    cov = (Pd - mu).T @ (Pd - mu) / len(Pd)
    w, V = np.linalg.eigh(cov)
    pc1 = V[:, int(np.argmax(w))]
    if pc1[0] < 0.0:
        pc1 = -pc1                                  # chord points +x -> TE is right
    proj = Pd @ pc1
    ite = int(np.argmax(proj))
    Pd = np.roll(Pd, -ite, axis=0)                  # TE at index 0
    proj2 = Pd @ pc1
    ile2 = int(np.argmin(proj2))
    loop = Pd[:ile2 + 1]                            # TE -> LE (surface A)
    rest = np.vstack([Pd[ile2:], Pd[0:1]])           # LE -> TE (surface B, wraps)
    full = np.vstack([loop[:-1], rest])              # TE .. (pre-LE), LE .. TE
    if curv_beta and curv_beta > 0.0:
        return _resample_curvature(full, n_arc, beta=curv_beta)
    return _resample_uniform(full, n_arc)


def make_contour_cgrid(contour, ni=281, nj=81, Rf_base=18.0, beta=3.0,
                       Lw=None, h_wall=None, n_winslow=1000, te_bridge=0.16,
                       n_norm_smooth=0, round_le_win=0.0, curv_beta=0.0,
                       norm_arc_smooth=0, arc_fair=0,
                       arc_fair_win=None, arc_fair_win_half=5,
                       arc_fair_win_relax=0.5,
                       n_wake_override=None,
                       h_wake=0.012, h_wake_junc=None, n_wake_ramp=15,
                       h_wake_far=None, round_te_thick=0.0, round_te_win=0.10):
    """C-grid around an arbitrary closed airfoil contour given in ABSOLUTE
    coordinates (ndarray (M,2) ordered around the loop; the whole loop is the
    no-slip wall).  Auto-detects the chord, localises the contour to a chord-1
    frame so the standard wake-trails-+x C-grid machine applies, then rigidly
    maps the finished grid back to its absolute pose.  Returns
    (X, Y, wall, recv, Lref) with Lref = the absolute chord length."""
    C = np.asarray(contour, float)
    if np.allclose(C[0], C[-1]):
        C = C[:-1]                                 # drop repeated closing point
    LE, TE, c, ang = _detect_chord(C)
    # order the wall arc TE->LE->TE by topological loop traversal (robust for
    # cambered/coved sections), then localise to a chord-1 frame (chord -> +x)
    # so the standard wake-trails-+x C-grid machine applies.
    n_wake = max(8, int(round(0.16 * ni)))
    if n_wake_override is not None:
        n_wake = int(n_wake_override)
    n_arc = ni - 2 * n_wake
    if n_arc < 24:
        n_wake = max(5, ni // 12); n_arc = ni - 2 * n_wake
    arc = _order_contour_loop(C, n_arc, curv_beta=curv_beta)
    # Fair the cove / sharp-trailing-edge notch (a single-block C-grid cannot
    # resolve a concave cove: algebraic normal extrusion self-intersects there,
    # producing sliver / inverted cells that blow the explicit solver up).  The
    # bridge only replaces the short TE->(frac*chord) segment; the exact contour
    # is preserved everywhere else.  te_bridge<=0 disables.
    if te_bridge and te_bridge > 0.0:
        arc = _fill_te_bridge(arc, frac=te_bridge)
    # Round a sharp leading edge (localised fairing around the min-x point) so the
    # algebraic normal extrusion does not pinch the nose cells into slivers.
    if round_le_win and round_le_win > 0.0:
        arc = _round_le(arc, win=round_le_win)
    # localise with an explicit R(-ang) (column form to avoid row/transpose
    # sign confusion): maps LE->(0,0), TE->(1,0), chord along +x.
    ca, sa = np.cos(ang), np.sin(ang)
    Rneg = np.array([[ca, sa], [-sa, ca]])          # = R(-ang)
    arc = (Rneg @ (arc - LE).T).T / c
    # Blunt a sharp trailing edge (chord-1 frame, TE at (1,0)) so the wake-cut
    # cell at the TE junction is not pinched to a sliver.  See _round_te.
    if round_te_thick and round_te_thick > 0.0:
        arc = _round_te(arc, thick=round_te_thick, win=round_te_win)
    # Mild fairing of small wall kinks (tangent discontinuities) in the digitised
    # contour; see _fair_arc.  Applied after localisation so the node spacing is
    # the chord-1 spacing.  If arc_fair_win is set (a (center, half) pair), ONLY
    # the windowed neighbourhood is faired -- this removes the slat/flap near-wall
    # PINCH while leaving the lift-generating contour untouched (global fairing
    # smears the whole element and kills the lift).
    if arc_fair and arc_fair > 0:
        if arc_fair_win is not None:
            arc = _fair_arc_windowed(arc, arc_fair_win, half=arc_fair_win_half,
                                     iters=int(arc_fair), relax=arc_fair_win_relax)
        else:
            arc = _fair_arc(arc, iters=arc_fair)
    # true outward surface normals (same recipe as make_naca_cgrid)
    n_arc_v = _arc_normals(arc, smooth_iters=norm_arc_smooth)
    # ---- orient the whole normal field OUTWARD with ONE global sign ------------
    # For a simple closed loop, the tangent-rotated field (t_y, -t_x) points either
    # uniformly outward or uniformly inward; the sign is fixed by the loop's
    # orientation, i.e. the sign of its signed area.  This is exact for arbitrarily
    # thin / cambered sections and needs no interior reference point -- the old
    # mid-chord rule dot(n, arc-[0.5,0]) silently flipped the lower-trailing-edge
    # normals of the cambered slat/flap to point UP, so the TE junction crossed the
    # adjacent wake node and pinched a ~1e-9 sliver that blew the solver up.
    # (I verified numerically that (t_y,-t_x) is INWARD here, and sarea<0 for all
    # three elements, so "flip when sarea<0" yields outward normals.)
    xa, ya = arc[:, 0], arc[:, 1]
    sarea = 0.5 * np.sum(xa[:-1] * ya[1:] - xa[1:] * ya[:-1])
    if sarea < 0.0:
        n_arc_v = -n_arc_v
    # Smooth the NORMAL DIRECTION field (wall geometry untouched) so adjacent
    # outward normals no longer flip ~180 deg at a sharp cove/curvature and pinch
    # the off-wall layers into slivers.  Only matters for the real GA(W)-2 slat/
    # flap whose lower-surface turn is tighter than the first off-wall spacing.
    if n_norm_smooth and n_norm_smooth > 0:
        n_arc_v = _smooth_normals(n_arc_v, niters=n_norm_smooth)
    # Localised clamp of the downstream (+x) normal component over the AFT airfoil
    # (x > 0.30) so the first off-wall layer never bulges past the TE (x=1) and
    # folds the i-sweep -- the classic C-grid "TE kink".  Same device as
    # make_naca_cgrid; only the aft region is touched, and near the TE the clamped
    # normals become +-y, matching the wake-centreline normals.
    x_te_clamp = 0.30
    aft = arc[:, 0] > x_te_clamp
    n_arc_v[aft, 0] = np.maximum(n_arc_v[aft, 0], 0.0)
    ln = np.hypot(n_arc_v[aft, 0], n_arc_v[aft, 1]); ln[ln < 1e-12] = 1.0
    n_arc_v[aft, 0] /= ln; n_arc_v[aft, 1] /= ln
    # TE-junction handling.  The two TE nodes (arc[0] and arc[n_arc]) meet the two
    # wake halves, which are cut along the chord line and therefore have normals
    # (0, +-1).  WHICH sign each half takes must follow the OUTWARD normal of the
    # arc end it attaches to -- the previous code hard-coded uw=(0,+1)/lw=(0,-1),
    # which is only correct when the loop ordering happens to start on the UPPER
    # TE.  For the cambered slat/flap (_order_contour_loop starts on the LOWER TE)
    # that assignment is reversed, so the arc end and its wake half pointed in
    # opposite directions and the first off-wall quad collapsed into a ~1e-9 sliver
    # -> the explicit solver blew up at that cell.  Derive the sign from arc[0].
    nu = np.array([0.0, 1.0 if n_arc_v[0, 1] >= 0.0 else -1.0])
    nl = -nu
    n_blend = max(4, n_arc // 12)
    for kk in range(n_blend):
        w = (kk + 1) / n_blend                       # 0..1, ramp to wake normal
        n_arc_v[kk] = (1.0 - w) * n_arc_v[kk] + w * nu
        n_arc_v[n_arc - kk] = (1.0 - w) * n_arc_v[n_arc - kk] + w * nl
    ln = np.hypot(n_arc_v[:, 0], n_arc_v[:, 1]); ln[ln < 1e-12] = 1.0
    n_arc_v /= ln[:, None]
    if Lw is None:
        Lw = Rf_base
    wpos = _wake_stretch(n_wake, Lw)
    uw = np.column_stack([1.0 + wpos[::-1], np.zeros(n_wake + 1)])
    lw = np.column_stack([1.0 + wpos, np.zeros(n_wake + 1)])
    curve = np.vstack([uw[:-1], arc, lw[1:]])
    nrm = np.zeros_like(curve)
    nrm[:n_wake] = nu
    nrm[n_wake:n_wake + n_arc + 1] = n_arc_v
    nrm[n_wake + n_arc + 1:] = nl
    if h_wall is None:
        r = _stretch_r(nj + 1, 0.0, Rf_base, beta=beta)
    else:
        r = _geom_stretch(h_wall, Rf_base, nj + 1)
    X = np.zeros((ni + 1, nj + 1)); Y = np.zeros((ni + 1, nj + 1))
    for j in range(nj + 1):
        X[:, j] = curve[:, 0] + r[j] * nrm[:, 0]
        Y[:, j] = curve[:, 1] + r[j] * nrm[:, 1]
    # Elliptic (Winslow) smoothing in the localised chord-1 frame: the algebraic
    # normal extrusion self-intersects (J->0 folds) at the real section's cove and
    # sharp trailing edge; Winslow relaxation untangles them while holding the wall
    # (j=0), far field and wake cuts fixed, so the airfoil shape / forces are intact.
    if n_winslow and n_winslow > 0:
        X, Y = _winslow_smooth(X, Y, iters=n_winslow)
        # Winslow flattens the interior and destroys the near-wall clustering (the
        # solve is boundary-determined), so re-impose the geometric spacing by
        # resampling each fold-free streamline along its own arc length.  j=0
        # (wall) and j=nj (far field) stay pinned -> shape & clustering restored.
        #
        # WAKE-CUT DECOUPLING (30P30N stability fix): the two wake-cut halves are
        # NOT no-slip walls -- they are the clamped free-stream cut.  They must not
        # inherit the airfoil y+50 first-off-wall spacing: that pins the thin
        # direction to ~2e-4 while the streamwise wake spacing grows to >1
        # downstream, producing AR~1000-7000 slivers that blow the collocated
        # projection.  Give the wake cut its OWN isotropic first-off-centreline
        # spacing (~the wake streamwise growth first step h0=0.012 from
        # _wake_stretch) stretched monotonically out to the far field.  The wake
        # cut must stay THICK everywhere (never revert to a y+50 layer) -- a thin
        # clamped free-stream line sitting right on the developing wake seeds a
        # strong free-stream/wake mismatch that diverges.  The airfoil surface
        # keeps its y+50 layer.  h_wake=None restores the old behaviour.
        r2d = np.broadcast_to(r, (ni + 1, nj + 1)).copy()
        if h_wake is not None:
            r_wake = _geom_stretch(h_wake, Rf_base, nj + 1)
            wake_lo = np.arange(0, n_wake)
            wake_hi = np.arange(n_wake + n_arc + 1, ni + 1)
            if h_wake_junc is None:
                # legacy: constant isotropic wake first-off-layer everywhere
                r2d[wake_lo, :] = r_wake
                r2d[wake_hi, :] = r_wake
            else:
                # STREAMWISE-GROWING WAKE FIRST-OFF-LAYER (30P30N sliver fix):
                # the wake cut is NOT a wall, so its first-off-centreline spacing
                # must stay thick -- but it must ALSO grow with streamwise distance
                # from the TE, because the streamwise wake spacing balloons from
                # ~h0 at the TE to ~Rf at the far field.  A CONSTANT thick layer
                # (h_wake) gives AR ~ Rf/h_wake ~ 1500 at the far-end wake columns
                # (i=0 / i=ni), which is exactly the sliver that blew the slat/flap
                # as single blocks.  Instead stretch the first-off-layer
                # geometrically from h_wake_junc (thin, at the TE junction -- matches
                # the airfoil y+50 to avoid a 40x single-column jump) out to
                # h_wake_far (thick, at the far field -- bounds the far-end AR).
                # This removes BOTH the TE-junction jump and the far-end sliver.
                h_far = h_wake_far if h_wake_far is not None else h_wake
                Dmax = max(1, n_wake - 1)
                for i in range(ni + 1):
                    if i < n_wake:
                        d = (n_wake - 1) - i          # 0 at upper junction
                    elif i > n_wake + n_arc:
                        d = i - (n_wake + n_arc + 1)   # 0 at lower junction
                    else:
                        continue                       # airfoil arc keeps r
                    h1 = h_wake_junc * (h_far / h_wake_junc) ** (d / Dmax)
                    r2d[i] = _geom_stretch(h1, Rf_base, nj + 1)
        X, Y = _recluster_wall(X, Y, r2d, Rf_base)
    wall = np.zeros((ni, nj), dtype=bool)
    wall[n_wake:n_wake + n_arc, 0] = True
    recv = np.zeros((ni, nj), dtype=bool)
    recv[:, nj - 1] = True
    recv[0, :] = True; recv[ni - 1, :] = True
    # inverse-localise: rotate by +ang, scale by c, translate LE back
    Xr, Yr = _transform_cgrid(X, Y, np.rad2deg(ang), LE[0], LE[1], scale=c)
    return Xr, Yr, wall, recv, c


def _load_30p30n(path=None):
    """Load the real 30P30N element contours (normalised coordinates) from the
    airfoil_data directory.  Returns dict name(lower)->(M,2) absolute array.

    Uses gh_{Slat,Main,Flap}.dat -- the deployed 30P30N geometry as used by the
    Wolf-Dynamics / linuxguy123 30P-30N validation case, which is the form most
    literature validates against.  IMPORTANT: the slat/flap CHORD-LINE angles
    read ~+49 / -35 deg because the GA(W)-2 sections carry camber + cove; the
    actual hinge DEFLECTION is 30 / 30 deg (nose-down), which is the canonical
    NASA rigging.  The chord-line angle is NOT the deflection -- do not re-rotate
    the sections to "30 deg chord-line", that destroys the real section pose.

    (precise_{Slat,Main,Flap}.dat is a deprecated re-deployment attempt and is
    NOT used here.)"""
    import os
    if path is None:
        path = os.path.join(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))), "airfoil_data")
    out = {}
    for name in ("Slat", "Main", "Flap"):
        out[name.lower()] = np.loadtxt(os.path.join(path, f"gh_{name}.dat"))
    return out


def make_three_element_cgrid_real(ni_s=121, nj_s=51, ni_m=181, nj_m=61,
                                  ni_f=121, nj_f=51, Rf_base=18.0, beta=3.0,
                                  contours=None, h_wall=None, h_wall_abs=None,
                                  n_winslow=1000, te_bridge=0.16,
                                  n_norm_smooth=0, round_le_win=0.0,
                                  curv_beta=0.0, norm_arc_smooth=0, arc_fair=0,
                                  arc_fair_win=None, arc_fair_win_half=5,
                                  arc_fair_win_relax=0.5,
                                  per_element=None, h_wake_junc=2.0e-3,
                                  n_wake_ramp=15, h_wake_far=0.1,
                                  te_thick=0.012):
    """Three-element 30P30N high-lift config built from the REAL GA(W)-2 element
    contours (not NACA approximations).  Each element's C-grid is wrapped around
    its actual surface shape and posed exactly as in the coordinate data, so the
    slat/main/flap are the genuine deployed geometry (slat chord ~15%c, flap
    ~30%c, slot gaps ~3%/1.3%c, deflections baked into the contours).  Returns
    the SAME dict structure as make_three_element_cgrid so the overset driver is
    completely unchanged.

    Wall spacing: h_wall (scalar) is applied in each element's OWN normalised
    (chord-1) frame, so larger elements get a proportionally thicker first layer.
    h_wall_abs (scalar, ABSOLUTE first-off-wall spacing) is preferred for wall
    functions: it is divided by each element's chord so every element's first
    cell has the SAME absolute y+ (=> uniform wall-function quality)."""
    if contours is None:
        contours = _load_30p30n()
    cfgs = {
        "main": dict(contour=contours["main"], ni=ni_m, nj=nj_m, Rf=Rf_base),
        "slat": dict(contour=contours["slat"], ni=ni_s, nj=nj_s, Rf=Rf_base),
        "flap": dict(contour=contours["flap"], ni=ni_f, nj=nj_f, Rf=Rf_base),
    }
    # per-element overrides: dict name->dict(n_winslow=, te_bridge=, h_wall_abs=)
    if per_element is None:
        per_element = {}
    order = ["main", "slat", "flap"]
    elements = []
    for name in order:
        c = cfgs[name]
        hw = h_wall
        pe = per_element.get(name, {})
        nw = pe.get("n_winslow", n_winslow)
        tb = pe.get("te_bridge", te_bridge)
        nns = pe.get("n_norm_smooth", n_norm_smooth)
        rl = pe.get("round_le_win", round_le_win)
        cb = pe.get("curv_beta", curv_beta)
        nas = pe.get("norm_arc_smooth", norm_arc_smooth)
        af = pe.get("arc_fair", arc_fair)
        af_win = pe.get("arc_fair_win", arc_fair_win)
        # Blunt the sharp slat/flap TE (main is already a blunt NACA TE).  The
        # sharp TE pinches the wake-cut cell at the TE junction into a sliver that
        # seeds the explicit-solver blow-up; a finite TE radius removes it.
        rt = pe.get("round_te_thick", te_thick if name != "main" else 0.0)
        if hw is None and h_wall_abs is not None:
            # scale absolute spacing by this element's chord -> uniform y+
            LE, TE, ch, ang = _detect_chord(np.asarray(c["contour"], float))
            hw = pe.get("h_wall_abs", h_wall_abs) / max(ch, 1e-6)
        X, Y, wall, recv, Lref = make_contour_cgrid(
            c["contour"], ni=c["ni"], nj=c["nj"], Rf_base=c["Rf"], beta=beta,
            h_wall=hw, n_winslow=nw, te_bridge=tb, n_norm_smooth=nns,
            round_le_win=rl, curv_beta=cb, norm_arc_smooth=nas, arc_fair=af,
            arc_fair_win=af_win, arc_fair_win_half=arc_fair_win_half,
            arc_fair_win_relax=arc_fair_win_relax,
            round_te_thick=rt,
            h_wake_junc=pe.get("h_wake_junc", h_wake_junc),
            n_wake_ramp=pe.get("n_wake_ramp", n_wake_ramp),
            h_wake_far=pe.get("h_wake_far", h_wake_far))
        elements.append(dict(name=name, X=X, Y=Y, wall=wall, recv=recv,
                             Lref=Lref, ni=c["ni"], nj=c["nj"],
                             theta=0.0, t=(0.0, 0.0)))
    return dict(elements=elements, Lref=max(e["Lref"] for e in elements),
                placement=dict(real=True))


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(X, Y, title, fname):
        ni = X.shape[0] - 1
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(X[:, 0], Y[:, 0], "k-", lw=1.5, label="wall j=0")
        for j in (nj // 4, nj // 2, 3 * nj // 4, nj - 1):
            ax.plot(X[:, j], Y[:, j], "b-", lw=0.4, alpha=0.6)
        for i in (0, ni // 4, ni // 2, 3 * ni // 4, ni):
            ax.plot(X[i], Y[i], "r-", lw=0.4, alpha=0.6)
        ax.set_aspect("equal"); ax.set_title(title); ax.legend(loc="upper right")
        fig.savefig(fname, dpi=120); plt.close()
        print("wrote", fname)

    nj = 61
    Xc, Yc, wc, rc, Lc = make_cylinder_cgrid(ni=121, nj=nj)
    plot(Xc, Yc, "cylinder C-grid", "/workspace/cgrid_cyl.png")
    Xn, Yn, wn, rn, Ln = make_naca_cgrid(ni=161, nj=nj)
    plot(Xn, Yn, "NACA0012 C-grid", "/workspace/cgrid_naca.png")
    print("Lref cyl", Lc, "Lref naca", Ln)
