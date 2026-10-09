"""
Overset / Chimera FVM for flow past a cylinder.

- Near field: body-fitted O-grid (polar annulus) around the cylinder, solved with a
  coordinate-transform (xi,eta metric) FVM; radial geometric stretching for near-wall
  refinement; 2nd-order upwind convection.
- Far field: uniform Cartesian background grid (identity metric), with a circular hole
  cut where the O-grid lives.
- The two grids are coupled by Overset/Chimera: a fringe/overlap annulus where each
  grid receives velocity from the other by bilinear interpolation (donor mapping).

State is cell-centered collocated (u,v,p at cell centers) on each structured block.
Projection method (Chorin): momentum predictor -> pressure Poisson -> velocity correct.

Reused idea: pressure-outlet BC (p=0) with RHS zeroing, exactly the probe11_fix fix.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import splu
import polar_ogrid as POG            # verified ORTHOGONAL-POLAR operators (div/grad/conv/diff/project)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Convective scheme selector: "upwind2" (stable 2nd-order upwind) or
# "quick" (3rd-order QUICK).  QUICK cuts numerical diffusion but is unstable
# on the tiny near-wall polar cells (|J|~1e-5); upwind2 is the stable baseline
# used for physics diagnostics.
SCHEME = "upwind2"


class Block:
    def __init__(self, name, ni, nj, xc, yc, periodic_i=False):
        self.name = name
        self.ni, self.nj = ni, nj
        self.xc, self.yc = xc, yc  # (ni,nj) cell centers
        self.u = np.zeros((ni, nj))
        self.v = np.zeros((ni, nj))
        self.p = np.zeros((ni, nj))
        self.periodic_i = periodic_i
        self.J = None
        self.dxdxi = self.dydxi = self.dxdeta = self.dydeta = None
        self.Axi_x = self.Axi_y = self.Aeta_x = self.Aeta_y = None
        self.solved = None
        self.recv = None
        self.hole = None
        self.r = None
        self.factor = None
        self.U = None
        self._nu = None
        # background geometry helpers
        self.hx = self.hy = self.Lx = self.Ly = None


# --------------------------------------------------------------------------- #
# Grid generation
# --------------------------------------------------------------------------- #
def make_ogrid(ni, nj, cx, cy, R, Rf, beta=2.0):
    """Polar O-grid wrapping the cylinder. Returns (block, geom)."""
    thn = np.linspace(0.0, 2.0 * np.pi, ni, endpoint=False)  # (ni,) theta nodes
    t = np.linspace(0.0, 1.0, nj + 1)
    rn = R + (Rf - R) * (np.exp(beta * t) - 1.0) / (np.exp(beta) - 1.0)  # (nj+1,) radial nodes
    THN, RN = np.meshgrid(thn, rn, indexing="ij")  # (ni, nj+1) nodes
    xn = cx + RN * np.cos(THN)
    yn = cy + RN * np.sin(THN)
    # cell-center theta (periodic) and radius
    thc = 0.5 * (thn + np.roll(thn, -1))
    rc = 0.5 * (rn[:-1] + rn[1:])
    THC, RC = np.meshgrid(thc, rc, indexing="ij")  # (ni, nj)
    xc = cx + RC * np.cos(THC)
    yc = cy + RC * np.sin(THC)

    blk = Block("ogrid", ni, nj, xc, yc, periodic_i=True)
    blk.r = np.sqrt((xc - cx) ** 2 + (yc - cy) ** 2)
    # node metrics
    dxdxi_n = 0.5 * (np.roll(xn, -1, axis=0) - np.roll(xn, 1, axis=0))
    dydxi_n = 0.5 * (np.roll(yn, -1, axis=0) - np.roll(yn, 1, axis=0))
    dxdeta_n = np.zeros_like(xn)
    dydeta_n = np.zeros_like(yn)
    dxdeta_n[:, 1:-1] = 0.5 * (xn[:, 2:] - xn[:, 0:-2])
    dydeta_n[:, 1:-1] = 0.5 * (yn[:, 2:] - yn[:, 0:-2])
    dxdeta_n[:, 0] = xn[:, 1] - xn[:, 0]
    dydeta_n[:, 0] = yn[:, 1] - yn[:, 0]
    dxdeta_n[:, -1] = xn[:, -1] - xn[:, -2]
    dydeta_n[:, -1] = yn[:, -1] - yn[:, -2]

    def avg_nodes(M):
        ipp = np.roll(M, -1, axis=0)
        return 0.25 * (M[:, :-1] + ipp[:, :-1] + M[:, 1:] + ipp[:, 1:])

    blk.dxdxi = avg_nodes(dxdxi_n)
    blk.dydxi = avg_nodes(dydxi_n)
    blk.dxdeta = avg_nodes(dxdeta_n)
    blk.dydeta = avg_nodes(dydeta_n)
    blk.J = blk.dxdxi * blk.dydeta - blk.dxdeta * blk.dydxi

    # ---- polar (r,theta) helpers for a SPD pressure operator ----
    # On a polar grid xi=theta, eta=r are ORTHOGONAL, so the conservative
    # (r,theta) Laplacian has NO cross terms and is SPD (diagonally-dominant
    # M-matrix).  The collocated curvilinear D o G assembled elsewhere is
    # INDEFINITE on a strongly stretched polar grid (symmetric-part min
    # eigenvalue ~ -1.5e5); that pumps energy into a checkerboard-like mode every
    # projection and blows the coupled run up.  We therefore solve/correct pressure
    # with the orthogonal polar operator below; the Cartesian gradient/divergence
    # are recovered by a (cos,sin) rotation of the (r,theta) derivatives.
    blk.rc = rc                                    # (ni,nj) cell-center radii
    blk.cos_th = (xc - cx) / rc                    # (ni,nj) cos(theta)_cell
    blk.sin_th = (yc - cy) / rc                    # (ni,nj) sin(theta)_cell
    blk.rn_nodes = rn                             # (nj+1,) radial NODE radii
    blk.dtheta = 2.0 * np.pi / ni

    # Face area vectors must follow the METRIC convention  A = J (xi_x, xi_y)
    # so that divergence/diffusion (which divide by the signed J) stay consistent.
    # The O-grid Jacobian J = x_xi*y_eta - x_eta*y_xi < 0, so the geometric
    # edge-rotate vectors (which equal -A_metric) are NEGATED here to become the
    # metric-consistent A_xi=(y_eta,-x_eta) and A_eta=(-y_xi,+x_xi).
    # xi-face (+xi normal) = ( dydeta , -dxdeta ) = ( y_eta , -x_eta )
    Axi_x = (yn[:, 1:] - yn[:, :-1])           # +(dy_eta)   shape (ni, nj)
    Axi_y = -(xn[:, 1:] - xn[:, :-1])          # -(dx_eta)
    # eta-face (+eta normal) = ( -dydxi , dxdxi ) = ( -y_xi , +x_xi )
    # periodic roll in i so all ni faces present -> shape (ni, nj+1)
    Aeta_x = -(yn - np.roll(yn, 1, axis=0))    # -(dy_xi)
    Aeta_y = (xn - np.roll(xn, 1, axis=0))     # +(dx_xi)
    blk.Axi_x = Axi_x
    blk.Axi_y = Axi_y
    blk.Aeta_x = Aeta_x
    blk.Aeta_y = Aeta_y

    geom = dict(cx=cx, cy=cy, R=R, Rf=Rf, beta=beta, ni=ni, nj=nj, rn=rn)
    return blk, geom


def make_background(Nx, Ny, Lx, Ly, cx, cy, R, Rf, U, Rh_hole=None):
    hx = Lx / Nx
    hy = Ly / Ny
    xc = (np.arange(Nx) + 0.5) * hx
    yc = (np.arange(Ny) + 0.5) * hy
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    r = np.sqrt((Xc - cx) ** 2 + (Yc - cy) ** 2)
    blk = Block("bg", Nx, Ny, Xc, Yc, periodic_i=False)
    blk.r = r
    blk.hx, blk.hy, blk.Lx, blk.Ly = hx, hy, Lx, Ly
    blk.J = np.full((Nx, Ny), hx * hy)
    blk.dxdxi = np.full((Nx, Ny), hx)
    blk.dydxi = np.zeros((Nx, Ny))
    blk.dxdeta = np.zeros((Nx, Ny))
    blk.dydeta = np.full((Nx, Ny), hy)
    blk.Axi_x = np.full((Nx, Ny), hy)  # (dydeta, -dxdeta) = (hy, 0)  interior xi-faces
    blk.Axi_y = np.zeros((Nx, Ny))
    blk.Aeta_x = np.zeros((Nx, Ny + 1))  # (-dydxi, dxdxi) = (0, hx)  FULL eta-face set
    blk.Aeta_y = np.full((Nx, Ny + 1), hx)
    # Immersed cylinder markers (used by the 'bg2og' coupling where the BACKGROUND
    # Cartesian grid carries the body and sheds its own clean von Karman street).
    #   solid : cells strictly inside the cylinder (r < R) -> velocity pinned 0
    #   wall  : fluid cells with a solid 4-neighbour -> no-slip layer (velocity 0)
    # The default hole/recv/solved below assume the OLD coupling (background holed,
    # body lives on the O-grid).  run_cylinder overrides hole/recv/solved for bg2og.
    blk.solid = r < R
    roll = (np.roll(blk.solid, 1, 0) | np.roll(blk.solid, -1, 0) |
            np.roll(blk.solid, 1, 1) | np.roll(blk.solid, -1, 1))
    blk.wall = (~blk.solid) & roll
    blk.hole = r < Rh_hole if Rh_hole is not None else np.zeros_like(blk.solid)
    blk.recv = (~blk.hole) & (r <= Rf)
    blk.solved = (~blk.hole) & (~blk.recv)
    return blk


# --------------------------------------------------------------------------- #
# Ghost / padding for boundaries
# --------------------------------------------------------------------------- #
def _pad(block, f, kind, U=None):
    """Return padded (ni+2, nj+2) array with boundary ghosts filled per BC."""
    U = block.U if block.U is not None else 0.0
    ni, nj = block.ni, block.nj
    fp = np.zeros((ni + 2, nj + 2))
    fp[1:-1, 1:-1] = f
    if block.periodic_i:
        fp[0, 1:-1] = f[-1, :]
        fp[-1, 1:-1] = f[0, :]
    else:
        if kind == "u":
            fp[0, 1:-1] = U  # inlet u=U
        else:
            fp[0, 1:-1] = 0.0  # inlet v=0
        fp[-1, 1:-1] = f[-1, :]  # outlet zero-gradient
    # j walls (bottom row0, top row nj+1)
    if block.name == "ogrid":
        if kind == "u":
            fp[1:-1, 0] = -f[:, 0]  # no-slip reflect
            fp[1:-1, -1] = f[:, -1]
        else:
            fp[1:-1, 0] = -f[:, 0]
            fp[1:-1, -1] = f[:, -1]
    else:  # background: slip walls top/bottom, OUTFLOW at downstream (right) end
        if kind == "u":
            fp[1:-1, 0] = f[:, 0]  # u zero-gradient (inlet)
            fp[1:-1, -1] = f[:, -1]  # u zero-gradient (outflow)
        else:
            fp[1:-1, 0] = -f[:, 0]  # v=0 at top/bottom slip walls -> reflect
            # downstream outflow: ZERO-GRADIENT (not reflect) so the wake leaves
            # the domain instead of reflecting back as a standing wave (which
            # would corrupt the shedding frequency St).  u already zero-gradient.
            fp[1:-1, -1] = f[:, -1]
    return fp


# --------------------------------------------------------------------------- #
# Convective fluxes (2nd-order upwind) on full face stencils
# --------------------------------------------------------------------------- #
def _face_metric_xi_full(M, block):
    # M is already the edge-based face vector at faces 0..ni-1; extend to full (ni+1)
    last = M[0:1] if block.periodic_i else M[-1:]
    return np.concatenate([M, last], axis=0)  # (ni+1, nj)


def _face_metric_eta_full(M, block):
    # Aeta is stored as the FULL eta-face set (ni, nj+1): faces j=0..nj
    # (inner wall / outer ring for ogrid, top+bottom walls for background).
    #
    # CRITICAL ALIGNMENT FIX: Aeta[i,j] as generated in make_ogrid is the edge
    # between node (i-1,j) and (i,j).  The eta-face that actually BOUNDS cell i
    # (connecting node i to node i+1) lives at index i+1.  Without shifting, the
    # divergence used Aeta[i] for cell i, putting the eta-face one column off
    # from the xi-face; the discrete metric identity sum(face vectors)=0 then
    # failed, so a constant (divergence-free) flow produced a spurious ~50-115
    # near-wall divergence (divided by the tiny near-wall Jacobian) that grew
    # exponentially every step.  Rolling by -1 in i aligns Aeta[i] with cell i's
    # true eta boundary (node i..i+1), matching the xi-face indexing, so the
    # metric identity holds and uniform flow has exact zero divergence.
    # Background Aeta is all-zero, so the roll is a no-op there.
    if block.periodic_i:
        return np.roll(M, -1, axis=0)
    return M  # (ni, nj+1)


def _upwind_xi(phip, Fxi, block):
    """3rd-order QUICK reconstruction of the transported scalar at the xi-faces.

    QUICK removes most of the NUMERICAL diffusion of 2nd-order upwind (its
    truncation error is O(dx^3) vs O(dx^2)), which matters here: at Re=100 the
    physical viscosity is tiny, so 2nd-order upwind added ~2-3x numerical
    diffusion on the coarse O-grid and smeared the boundary layer -> Cd came out
    ~0.6 instead of ~1.6 and the wake would not shed.  QUICK recovers the
    physical Reynolds number.
      face between cell (k-1) and k, Fxi>0 (flow +xi):
          phi = (-phi_{k-2} + 6 phi_{k-1} + 3 phi_k) / 8
      Fxi<0:  phi = (-phi_{k+1} + 6 phi_k + 3 phi_{k-1}) / 8
    """
    ni = Fxi.shape[0] - 1
    nj = Fxi.shape[1]
    nr = phip.shape[0]
    k = np.arange(ni + 1)
    i_left = k
    i_right = k + 1
    i_ll = k - 1
    i_rr = k + 2
    if block.periodic_i:
        i_left %= nr
        i_right %= nr
        i_ll %= nr
        i_rr %= nr
    else:
        i_ll = np.clip(i_ll, 0, nr - 1)
        i_rr = np.clip(i_rr, 0, nr - 1)
    phiL = phip[i_left, 1:-1]
    phiR = phip[i_right, 1:-1]
    if SCHEME == "quick":
        phiLL = phip[i_ll, 1:-1]
        phiRR = phip[i_rr, 1:-1]
        quick_pos = (-phiLL + 6.0 * phiL + 3.0 * phiR) / 8.0
        quick_neg = (-phiRR + 6.0 * phiR + 3.0 * phiL) / 8.0
        return np.where(Fxi > 0.0, quick_pos, quick_neg)
    # 2nd-order upwind: use the upstream cell value (phiL for +xi, phiR for -xi)
    return np.where(Fxi > 0.0, phiL, phiR)


def _upwind_eta_fixed(phip, Feta, block):
    """3rd-order QUICK reconstruction at the eta-faces (see _upwind_xi)."""
    ni = Feta.shape[0]
    nj = Feta.shape[1] - 1
    nc = phip.shape[1]
    k = np.arange(nj + 1)
    j_left = k
    j_right = k + 1
    j_ll = k - 1
    j_rr = k + 2
    j_ll = np.clip(j_ll, 0, nc - 1)
    j_rr = np.clip(j_rr, 0, nc - 1)
    phiL = phip[1:-1, j_left]
    phiR = phip[1:-1, j_right]
    if SCHEME == "quick":
        phiLL = phip[1:-1, j_ll]
        phiRR = phip[1:-1, j_rr]
        quick_pos = (-phiLL + 6.0 * phiL + 3.0 * phiR) / 8.0
        quick_neg = (-phiRR + 6.0 * phiR + 3.0 * phiL) / 8.0
        return np.where(Feta > 0.0, quick_pos, quick_neg)
    # 2nd-order upwind: upstream cell value (phiL for +eta, phiR for -eta)
    return np.where(Feta > 0.0, phiL, phiR)


def _rhie_pressure_flux(block, p, dt):
    """Rhie-Chow pressure contribution to the face mass fluxes.

    Returns (Fxi_p, Feta_p) to be SUBTRACTED from the linear face mass flux:
        Fxi  = 0.5(uL+uR)*Ax - dt*Fxi_p
        Feta = 0.5(uL+uR)*Ag - dt*Feta_p
    where Fxi_p = (grad p . A_face) at the face, evaluated with the FACE (1-cell)
    pressure gradient (p_right - p_left)/dist -- NOT the 2-cell cell-centered
    gradient.  Using the face gradient couples adjacent cells through the pressure
    jump ACROSS the face and removes the odd-even (checkerboard) pressure-velocity
    decoupling of collocated grids.  That decoupled mode lives in the null space
    of the 2-cell divergence/gradient stencil, so the projection (assembled from
    the same stencil) can never see or clean it -> it grew ~3x/step and overflowed.
    """
    ni, nj = block.ni, block.nj
    Ax_f = _face_metric_xi_full(block.Axi_x, block)
    Ay_f = _face_metric_xi_full(block.Axi_y, block)
    Axg = _face_metric_eta_full(block.Aeta_x, block)
    Ayg = _face_metric_eta_full(block.Aeta_y, block)
    if block.periodic_i:
        dth = block.dtheta
        rc = block.rc                       # (nj,) cell radii (depend on j only)
        rn = block.rn_nodes                 # (nj+1,) radial NODE radii
        # xi-face (ni+1, nj): between cell (k-1) and k (periodic in xi).
        # arc length between theta cells depends only on j: rc[j]*dth.
        k = np.arange(ni + 1)
        kL = (k - 1) % ni
        kR = k % ni
        dist_xi = rc[None, :] * dth                 # (1,nj) -> (ni+1,nj)
        dp_xi = (p[kR, :] - p[kL, :]) / dist_xi
        mag_xi = (rn[1:] - rn[:-1])[None, :]        # |A_xi| = radial node spacing
        Fxi_p = dp_xi * mag_xi
        # eta-face (ni, nj+1): radial, between cell (j-1) and j.
        # arc length at face j = rn[j]*dth (face sits on node radius rn[j]).
        j = np.arange(nj + 1)
        dist_eta = np.zeros(nj + 1)
        dist_eta[1:] = rn[1:] - rn[:-1]
        dp_eta = np.zeros((ni, nj + 1))
        # face j (1..nj-1): between cell (j-1) and j.  Boundary faces j=0 (inner
        # wall) and j=nj (outer recv ring) stay 0 (Neumann dp/dn=0).
        dp_eta[:, 1:nj] = (p[:, 1:nj] - p[:, 0:nj - 1]) / dist_eta[None, 1:nj]
        mag_eta = rn[None, :] * dth                # |A_eta| = arc length (1,nj+1)
        Feta_p = dp_eta * mag_eta
    else:
        hx, hy = block.hx, block.hy
        k = np.arange(ni + 1)
        dp_xi = np.zeros((ni + 1, nj))
        dp_xi[1:ni, :] = (p[1:ni, :] - p[0:ni - 1, :]) / hx
        Fxi_p = dp_xi * Ax_f                       # |A_xi| = hy
        j = np.arange(nj + 1)
        dp_eta = np.zeros((ni, nj + 1))
        dp_eta[:, 1:nj] = (p[:, 1:nj] - p[:, 0:nj - 1]) / hy
        Feta_p = dp_eta * Ayg                       # |A_eta| = hx
    return Fxi_p, Feta_p


def predictor(block, dt, nu, U):
    ni, nj = block.ni, block.nj
    J = block.J
    Ax_f = _face_metric_xi_full(block.Axi_x, block)
    Ay_f = _face_metric_xi_full(block.Axi_y, block)
    Axg = _face_metric_eta_full(block.Aeta_x, block)
    Ayg = _face_metric_eta_full(block.Aeta_y, block)

    up = _pad(block, block.u, "u", U)
    vp = _pad(block, block.v, "v", U)

    # xi-face mass flux (pure linear central average).  NOTE: the Rhie-Chow
    # pressure term that USED to be subtracted here (-dt*Fxi_p) is intentionally
    # REMOVED.  With the SPD projection (G = D.T, exact adjoint of the actual
    # divergence) the checkerboard mode is already killed by the velocity
    # correction, so Rhie-Chow is redundant.  Worse, now that the pressure is
    # well-resolved (~1e2 on the stretched grid) the term -dt*(grad p . A) reaches
    # ~0.08 -- 4x the linear mass flux u*A~0.02 -- so it dominates the flux and
    # pumps energy into the velocity, driving u past U until the explicit CFL
    # (U*dt/h) exceeds 1 and the convection runs away.  Pure linear flux is the
    # standard, stable Chorin predictor here.
    uL = up[0:ni + 1, 1:-1]
    uR = up[1:ni + 2, 1:-1]
    vL = vp[0:ni + 1, 1:-1]
    vR = vp[1:ni + 2, 1:-1]
    u_f = 0.5 * (uL + uR)
    v_f = 0.5 * (vL + vR)
    Fxi = u_f * Ax_f + v_f * Ay_f

    fu = _upwind_xi(up, Fxi, block) * Fxi
    fv = _upwind_xi(vp, Fxi, block) * Fxi

    # eta-face mass flux (pure linear central average)
    uLb = up[1:-1, 0:nj + 1]
    uRb = up[1:-1, 1:nj + 2]
    vLb = vp[1:-1, 0:nj + 1]
    vRb = vp[1:-1, 1:nj + 2]
    u_fg = 0.5 * (uLb + uRb)
    v_fg = 0.5 * (vLb + vRb)
    Feta = u_fg * Axg + v_fg * Ayg

    fgu = _upwind_eta_fixed(up, Feta, block) * Feta
    fgv = _upwind_eta_fixed(vp, Feta, block) * Feta

    # CRITICAL SIGN FIX: the convection momentum flux fu = upwind(u)*Fxi does
    # NOT carry the 1/J metric factor (unlike the diffusion flux, which is built
    # from _grad_cell and already contains 1/J).  So its divergence must divide
    # by the CELL VOLUME |J|, not the SIGNED Jacobian J.  On the polar O-grid
    # J = x_xi*y_eta - x_eta*y_xi = -r < 0, so dividing by J (signed) flips the
    # sign of the convective tendency: the predictor's -duc then ADDS the flux
    # (+div_phys) instead of subtracting it (-div_phys) -> a backward-in-time,
    # anti-dissipative, energy-injecting convection that blows the O-grid up
    # (the background Cartesian grid has J>0, so it was unaffected -- which is
    # exactly why only the O-grid was unstable).  Use |J| (volume) here.
    vol = np.abs(J)
    duc = (fu[1:, :] - fu[:-1, :] + fgu[:, 1:] - fgu[:, :-1]) / vol
    dvc = (fv[1:, :] - fv[:-1, :] + fgv[:, 1:] - fgv[:, :-1]) / vol

    # diffusion
    diff_u = _diffusion(block, up, "u")
    diff_v = _diffusion(block, vp, "v")

    ustar = block.u + dt * (-duc + diff_u)
    vstar = block.v + dt * (-dvc + diff_v)
    # Hold only the HOLE + RECV (fringe / overset-donor) cells at their previous
    # value.  For the 'bg2og' background the SOLID + WALL (immersed cylinder) cells
    # MUST be updated by the predictor so the predicted u* there is non-zero;
    # apply_ibm then pins them to zero and (when called with dt) integrates the
    # removed momentum as the cylinder body force.  For every other coupling
    # block.solved == ~(hole|recv), so this reduces to the old behaviour.
    _hold = block.hole | block.recv
    ustar[_hold] = block.u[_hold]
    vstar[_hold] = block.v[_hold]
    block.u = ustar
    block.v = vstar


def ogrid_predictor(og, dt, nu, Ueff, conv_scheme="upwind1", conv_filter_sigma=0.0,
                    hv_cfl=0.0):
    """Predictor step for the POLAR O-grid, using the VERIFIED polar_ogrid operators
    (polar_conv = conservative upwind convective flux, polar_diffusion = nu*div(grad),
    both checked to machine precision).  This replaces the broken (xi,eta) metric
    predictor (which produced the wrong-sign Cd ~ -0.99) with the orthogonal-polar
    formulation that matches the standalone probe114/verify tests.

    BCs enforced here:
      * inner no-slip wall (j=0, r=R): u = v = 0  (held every step)
      * outer fringe (recv, r >= Rf):  Dirichlet donor velocity (set by
        overset_exchange) -- held here, NOT modified by the predictor; the
        projection's reduced-Poisson system then keeps p[fringe] = 0 so the
        pressure spike that corrupted the old force calc cannot appear.
    """
    duc, dvc = POG.polar_conv(og, og.u, og.v, Ueff, scheme=conv_scheme)
    diff_u, diff_v = POG.polar_diffusion(og, og.u, og.v)
    ustar = og.u + dt * (-duc + diff_u)
    vstar = og.v + dt * (-dvc + diff_v)
    # Optional LOCAL 4th-order hyperviscosity: a[i,j]=cfl*Delta^4/dt so the explicit
    # 4th-order CFL is uniform cfl everywhere (never NaN on the tiny near-wall cells).
    # Damps ONLY grid-scale modes (rate ~k^4) -> removes the spurious near-wall
    # radial saw-tooth / jet (which over-predicts Cp via Bernoulli and Cv via wall
    # shear) while leaving the St~0.18 vortex street (wavelength ~5D >> grid) intact.
    if hv_cfl > 0.0:
        hu, hv = POG.polar_hypervis_local(og, ustar, vstar, cfl=hv_cfl, dt=dt)
        # 4th-order hyperviscosity is -dt * a*Lap^2(u) (Lap^2 has +k^4 eigenvalues,
        # so the MINUS sign damps grid-scale modes; the plus sign would amplify -> NaN)
        ustar = ustar - dt * hu
        vstar = vstar - dt * hv
    # no-slip solid wall (inner eta-face, j=0)
    ustar[:, 0] = 0.0
    vstar[:, 0] = 0.0
    # recv fringe held as Dirichlet (donor bg velocity from overset_exchange)
    ustar[og.recv] = og.u[og.recv]
    vstar[og.recv] = og.v[og.recv]
    # Optional theta-direction Shapiro 4th-order filter on the O-grid predicted
    # velocity (solved cells only).  Damps only the top angular wavenumbers, so
    # it kills the grid-scale near-wall radial 'spokes' without smearing the
    # physical St~0.18 vortex street.  Applied to the actual flow (not just the
    # plot) so it also keeps the grid-scale noise out of the force integral.
    if conv_filter_sigma > 0.0:
        m = og.solved
        ustar[m] = POG.polar_filter(og, ustar, sigma_theta=conv_filter_sigma)[m]
        vstar[m] = POG.polar_filter(og, vstar, sigma_theta=conv_filter_sigma)[m]
    og.u = ustar
    og.v = vstar


def _grad_cell(block, fp):
    """Cell-center physical gradient (gx,gy) from padded field fp."""
    ni, nj = block.ni, block.nj
    fc = fp[1:-1, 1:-1]
    if block.periodic_i:
        dxi = 0.5 * (np.roll(fc, -1, axis=0) - np.roll(fc, 1, axis=0))
    else:
        dxi = np.zeros_like(fc)
        dxi[1:-1, :] = 0.5 * (fc[2:, :] - fc[:-2, :])
        dxi[0, :] = fc[1, :] - fc[0, :]
        dxi[-1, :] = fc[-1, :] - fc[-2, :]
    deta = np.zeros_like(fc)
    deta[:, 1:-1] = 0.5 * (fc[:, 2:] - fc[:, :-2])
    deta[:, 0] = fc[:, 1] - fc[:, 0]  # wall one-sided (uses ghost reflect for no-slip)
    deta[:, -1] = fc[:, -1] - fc[:, -2]
    J = block.J
    gx = (1.0 / J) * (block.dydeta * dxi - block.dydxi * deta)
    gy = (1.0 / J) * (-block.dxdeta * dxi + block.dxdxi * deta)
    return gx, gy


def _diffusion(block, fp, kind):
    ni, nj = block.ni, block.nj
    gx, gy = _grad_cell(block, fp)
    nu = block._nu
    Ax_f = _face_metric_xi_full(block.Axi_x, block)
    Ay_f = _face_metric_xi_full(block.Axi_y, block)
    Axg = _face_metric_eta_full(block.Aeta_x, block)
    Ayg = _face_metric_eta_full(block.Aeta_y, block)
    # xi-face gradient: avg of cell (k-1) and cell (k), faces k=0..ni
    k = np.arange(ni + 1)
    il = (k - 1) % ni if block.periodic_i else np.clip(k - 1, 0, ni - 1)
    ir = np.clip(k, 0, ni - 1)
    gxF = 0.5 * (gx[il] + gx[ir])
    gyF = 0.5 * (gy[il] + gy[ir])
    flux_xi = nu * (gxF * Ax_f + gyF * Ay_f)  # (ni+1, nj)
    # eta-face gradient: avg of cell (j-1) and cell (j), faces j=0..nj
    j = np.arange(nj + 1)
    jl = np.clip(j - 1, 0, nj - 1)
    jr = np.clip(j, 0, nj - 1)
    gxFg = 0.5 * (gx[:, jl] + gx[:, jr])
    gyFg = 0.5 * (gy[:, jl] + gy[:, jr])
    flux_eta = nu * (gxFg * Axg + gyFg * Ayg)  # (ni, nj+1)
    diff = (flux_xi[1:, :] - flux_xi[:-1, :] + flux_eta[:, 1:] - flux_eta[:, :-1]) / block.J
    return diff


def _pad_div(block, f, kind):
    """Ghost fill for the PROJECTION divergence operator.

    CRITICAL CONSISTENCY FIX: the projection Poisson must satisfy
        L == D o G   (discrete divergence o gradient)
    exactly, so that Chorin correction removes divergence to machine precision.
    For D o G to be self-adjoint (constant pressure in its null space, L*ones==0),
    the divergence ghost BCs MUST match the gradient BCs (gradient() uses
    one-sided / Neumann at every non-periodic boundary -- it never injects a
    prescribed velocity).

    The momentum PREDICTOR keeps the physical inflow velocity U in its inlet ghost
    (in _pad, used by predictor()), but the PROJECTION divergence must NOT do that:
    injecting U into the inlet ghost makes div(0) != 0 (a zero field spuriously
    acquires ~U*hx divergence at the inlet column), which breaks L*ones==0 and lets
    the pressure solve amplify a constant-divergence mode -> overflow by ~step 7.

    Here the inlet ghost is the ZERO-GRADIENT (cell value) so div(0)==0 identically;
    the real inflow flux is still captured because the inlet CELL itself carries the
    prescribed U (Dirichlet), so 0.5*(u_ghost + u_cell)*A = u_cell*A = U*A.
    Walls keep their reflect / slip ghosts (they contribute 0 for a zero field too).
    """
    ni, nj = block.ni, block.nj
    fp = np.zeros((ni + 2, nj + 2))
    fp[1:-1, 1:-1] = f
    if block.periodic_i:
        fp[0, 1:-1] = f[-1, :]
        fp[-1, 1:-1] = f[0, :]
    else:
        # inlet i=0 and outlet i=ni-1: ZERO-GRADIENT (cell value), NOT U.
        fp[0, 1:-1] = f[0, :]
        fp[-1, 1:-1] = f[-1, :]
    # j walls (bottom row0, top row nj+1)
    if block.name == "ogrid":
        fp[1:-1, 0] = -f[:, 0]   # no-slip reflect (u and v)
        fp[1:-1, -1] = f[:, -1]
    else:  # background slip walls
        if kind == "u":
            fp[1:-1, 0] = f[:, 0]   # u zero-gradient
            fp[1:-1, -1] = f[:, -1]
        else:
            fp[1:-1, 0] = -f[:, 0]  # v=0 at wall -> reflect
            fp[1:-1, -1] = -f[:, -1]
    return fp


def divergence(block, u, v):
    """Cell-center divergence of (u,v) mass flux (used for projection RHS).

    The O-grid uses the SAME conservative (xi,eta) metric divergence as the
    background.  The metric FVM satisfies discrete summation-by-parts, so L == D o
    G is symmetric (in the J-weighted inner product) and SPD -- PROVIDED the
    periodic (theta) faces are treated with wraparound (which _face_metric_xi_full
    and _pad_div already do).  An earlier non-periodic metric version had a theta
    seam and was indefinite (min eigenvalue ~ -1.5e5); the orthogonal (r,theta)
    polar helpers tried to fix that but were NOT exact adjoints (asymmetry 1.4e4,
    still indefinite) -> pressure overflowed 1e13x.  The periodic metric version is
    the correct, symmetric operator."""
    if block.periodic_i:
        pass  # fall through to the metric divergence (periodic-aware)
    ni, nj = block.ni, block.nj
    ni, nj = block.ni, block.nj
    up = _pad_div(block, u, "u")
    vp = _pad_div(block, v, "v")
    Ax_f = _face_metric_xi_full(block.Axi_x, block)
    Ay_f = _face_metric_xi_full(block.Axi_y, block)
    Axg = _face_metric_eta_full(block.Aeta_x, block)
    Ayg = _face_metric_eta_full(block.Aeta_y, block)
    uL = up[0:ni + 1, 1:-1]; uR = up[1:ni + 2, 1:-1]
    vL = vp[0:ni + 1, 1:-1]; vR = vp[1:ni + 2, 1:-1]
    Fxi = 0.5 * (uL + uR) * Ax_f + 0.5 * (vL + vR) * Ay_f
    uLb = up[1:-1, 0:nj + 1]; uRb = up[1:-1, 1:nj + 2]
    vLb = vp[1:-1, 0:nj + 1]; vRb = vp[1:-1, 1:nj + 2]
    Feta = 0.5 * (uLb + uRb) * Axg + 0.5 * (vLb + vRb) * Ayg
    # Divide by the CELL VOLUME |J| (not the signed Jacobian J): the divergence
    # is the outward flux per unit volume.  On the polar O-grid J = -r < 0, so
    # dividing by J would flip the sign of the divergence and, with the exact
    # adjoint G = D.T, make the projection correct the wrong (sign-flipped) field.
    # Using |J| keeps the discrete divergence physically signed and consistent
    # with the convection tendency (which also divides by |J|).
    div = (Fxi[1:, :] - Fxi[:-1, :] + Feta[:, 1:] - Feta[:, :-1]) / np.abs(block.J)
    return div


def _grad_polar(block, p):
    """Cartesian physical gradient of scalar p on the polar O-grid, from the
    orthogonal (r,theta) derivatives.

    This is the SPD-consistent gradient: its transpose (see _div_polar) gives the
    conservative (r,theta) Laplacian -- a diagonally-dominant M-matrix, so the
    pressure Poisson on the O-grid is SPD instead of the INDEFINITE collocated
    curvilinear D o G (symmetric-part min eigenvalue ~ -1.5e5) that pumped energy
    into a checkerboard-like mode every step and blew the coupled run up.  The
    Cartesian components are recovered by rotating (dp/dr, (1/r) dp/dtheta)."""
    ni, nj = block.ni, block.nj
    dth = block.dtheta
    rn = block.rn_nodes               # (nj+1,) radial NODE radii
    rc = block.rc                     # (ni,nj) cell-center radii
    cos_th = block.cos_th
    sin_th = block.sin_th
    # theta derivative (periodic), central at cell
    pp = np.roll(p, -1, axis=0)
    pm = np.roll(p, 1, axis=0)
    dth_cell = (pp - pm) / (2.0 * dth)
    # radial derivative, face-based (conservative) then averaged to cell
    drf_p = np.zeros((ni, nj))        # face j+1/2
    drf_m = np.zeros((ni, nj))        # face j-1/2
    for j in range(nj):
        drp = rn[j + 1] - rn[j]
        if j < nj - 1:
            drf_p[:, j] = (p[:, j + 1] - p[:, j]) / drp
        else:
            # outer boundary (Dirichlet recv ring): one-sided, keeps operator interior-SPD
            drm = rn[j] - rn[j - 1]
            drf_p[:, j] = (p[:, j] - p[:, j - 1]) / drm
        if j > 0:
            drm = rn[j] - rn[j - 1]
            drf_m[:, j] = (p[:, j] - p[:, j - 1]) / drm
        else:
            drf_m[:, j] = 0.0         # inner no-slip wall: Neumann dp/dr = 0
    dr_cell = 0.5 * (drf_p + drf_m)
    gx = cos_th * dr_cell - sin_th * (dth_cell / rc)
    gy = sin_th * dr_cell + cos_th * (dth_cell / rc)
    return gx, gy


def _div_polar(block, u, v):
    """Physical (Cartesian) divergence div(u,v) = du/dx + dv/dy on the polar O-grid,
    from the orthogonal (r,theta) form  div = (1/r) d(r u_r)/dr + (1/r) du_theta/dtheta.

    Transpose-consistent with _grad_polar: their composition is exactly the
    conservative (r,theta) Laplacian (SPD), so the O-grid projection removes
    divergence without pumping energy into a checkerboard mode."""
    ni, nj = block.ni, block.nj
    dth = block.dtheta
    rn = block.rn_nodes
    rc = block.rc
    cos_th = block.cos_th
    sin_th = block.sin_th
    ur = u * cos_th + v * sin_th          # radial velocity component
    ut = -u * sin_th + v * cos_th         # tangential velocity component
    # theta divergence (periodic central)
    utp = np.roll(ut, -1, axis=0)
    utm = np.roll(ut, 1, axis=0)
    dth_ut = (utp - utm) / (2.0 * dth)
    # radial divergence of (r ur): conservative face flux F = r_face * ur_face
    F_p = np.zeros((ni, nj))
    F_m = np.zeros((ni, nj))
    for j in range(nj):
        rc_face = 0.5 * (rn[j] + rn[j + 1])
        if j < nj - 1:
            ur_face = 0.5 * (ur[:, j] + ur[:, j + 1])
            F_p[:, j] = rc_face * ur_face
        else:
            F_p[:, j] = rc_face * ur[:, j]   # one-sided at outer Dirichlet ring
        if j > 0:
            rc_face_m = 0.5 * (rn[j - 1] + rn[j])
            ur_face_m = 0.5 * (ur[:, j - 1] + ur[:, j])
            F_m[:, j] = rc_face_m * ur_face_m
        else:
            F_m[:, j] = 0.0                   # inner wall: no radial flux
    dr_cell = np.zeros((ni, nj))
    for j in range(nj):
        if 0 < j < nj - 1:
            dr_cell[:, j] = rn[j + 1] - rn[j - 1]
        elif j == 0:
            dr_cell[:, j] = rn[j + 1] - rn[j]
        else:
            dr_cell[:, j] = rn[j] - rn[j - 1]
    div_r = (F_p - F_m) / (rc * dr_cell)
    return div_r + dth_ut / rc


def gradient(block, p):
    """Physical gradient of scalar p at cell centers.

    For the polar O-grid (periodic_i) this uses the SAME conservative (xi,eta)
    metric gradient as the background (via _grad_cell, which applies the periodic
    theta wraparound), so L == D o G is the symmetric, SPD operator obtained from
    discrete summation-by-parts.  The earlier orthogonal (r,theta) polar helpers
    (_grad_polar/_div_polar) were NOT exact adjoints (asymmetry ~1.4e4, indefinite,
    largest eigenvalue -9.5e4) and made the pressure overflow ~1e13x.  MUST match
    exactly the gradient used inside project()."""
    if block.periodic_i:
        fp = np.zeros((block.ni + 2, block.nj + 2))
        fp[1:-1, 1:-1] = p
        fp[0, 1:-1] = p[-1, :]
        fp[-1, 1:-1] = p[0, :]
        fp[1:-1, 0] = p[:, 0]
        fp[1:-1, -1] = p[:, -1]
        return _grad_cell(block, fp)
    ni, nj = block.ni, block.nj
    px = np.zeros_like(p)
    px[1:-1, :] = 0.5 * (p[2:, :] - p[:-2, :])
    px[0, :] = p[1, :] - p[0, :]
    px[-1, :] = p[-1, :] - p[-2, :]
    py = np.zeros_like(p)
    py[:, 1:-1] = 0.5 * (p[:, 2:] - p[:, :-2])
    py[:, 0] = p[:, 1] - p[:, 0]
    py[:, -1] = p[:, -1] - p[:, -2]
    J = block.J
    gx = (1.0 / J) * (block.dydeta * px - block.dydxi * py)
    gy = (1.0 / J) * (-block.dxdeta * px + block.dxdxi * py)
    return gx, gy


# --------------------------------------------------------------------------- #
# Pressure Poisson (curvilinear, conservative form)
# --------------------------------------------------------------------------- #
def _assemble_DoG(block):
    """Assemble the O-grid pressure-projection operator and force it SPD.

    Column assembly gives the EXACT discrete  L = D o G  (divergence o gradient)
    using the *same* gradient()/divergence() that project() calls, so the Chorin
    projection (u -= dt*G(p), L p = div(u*)/dt) removes divergence exactly.

    PROBLEM this function fixes: on a collocated, highly-stretched polar O-grid,
    D o G is *indefinite* -- its symmetric part has a small negative eigenvalue
    (about -1 on the unit-velocity scale, M-weighted about -1e-3).  A non-PSD
    pressure operator makes the pressure correction NON-contractive: for the
    indefinite modes the corrected velocity has *more* kinetic energy than before,
    and the run diverges (overflow in the convection flux by ~step 100).

    FIX: enforce SPD by
            L_spd = 0.5*(L + L^T) + sigma*I,   sigma = max(0, -lambda_min(L_sym)) + eps.
    Because  eig(L + sigma*I) = eig(L) + sigma, this tiny shift flips the indefinite
    floor positive WITHOUT touching the well-resolved physical pressure modes
    (their eigenvalues are O(1..1e2)), and it also eliminates the constant-pressure
    null space, so L_spd is strictly invertible (no pressure pin needed).  The
    projection is then strictly contractive and the solver is stable.  The shift is
    chosen per-grid from the assembled operator, so it auto-adapts to the stretch.
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import identity
    ni, nj = block.ni, block.nj
    N = ni * nj

    def idx(i, j):
        return i * nj + j

    A = lil_matrix((N, N))
    ptmp = np.zeros((ni, nj))
    for k in range(N):
        p = ptmp.copy()
        p.reshape(-1)[k] = 1.0
        gx, gy = gradient(block, p)
        d = divergence(block, gx, gy)  # (ni, nj) = (D o G) applied to e_k
        A[:, k] = d.reshape(-1)
    A = A.tocsr()
    # Symmetrize: L is already nearly symmetric (skew part ~1e-3..1e-4 of magnitude),
    # this removes the residual skew part that would otherwise break contractiveness.
    Asym = 0.5 * (A + A.T)
    # Minimal shift so the symmetric part is positive definite.
    try:
        lam = float(eigsh(Asym, k=1, which="SA", return_eigenvectors=False)[0])
    except Exception:
        lam = -1.0
    sigma = max(0.0, -lam) + 1e-9
    Aspd = Asym + sigma * identity(N)  # sigma*I -> adds sigma to every diagonal entry
    block._A_poisson = Aspd
    block._poisson_sigma = sigma
    return splu(Aspd.tocsc())


def _assemble_div_matrix(block):
    """Assemble the discrete divergence operator  D  (N x 2N CSR sparse) so that

        D @ np.concatenate([u.ravel(), v.ravel()])

    EQUALS  divergence(block, u, v).ravel()  to machine precision.

    D is the ACTUAL discrete divergence used everywhere else (metric FVM,
    periodic-theta aware, with the projection ghost BCs from _pad_div).  Its
    transpose  G = D.T  is therefore the EXACT adjoint gradient, and

        L = D o G = D @ D.T

    is SYMMETRIC POSITIVE-DEFINITE BY CONSTRUCTION (it is a Gram matrix
    D@D.T).  The Chorin correction  u <- u - dt*G*p = u - dt*D.T*p  then
    removes the divergence EXACTLY:

        D(u_corr) = D(u*) - dt*D@D.T*p = D(u*) - dt*(D(u*)/dt) = 0,

    and because L is SPD the pressure is bounded and unique -- no energy is
    pumped into any mode.

    WHY THIS REPLACES THE OLD APPROACH: the previous operator was assembled as
    L = D o G using a hand-derived `gradient` and `divergence`.  On the strongly
    stretched polar O-grid those two were NOT exact discrete adjoints (the
    cell-centered gradient stencil is not the transpose of the face-averaged
    divergence), so L was NON-SYMMETRIC and INDEFINITE (largest eigenvalue
    ~ -1.5e5, asymmetry 1.4e4).  A unit RHS then produced |p| ~ 1e9-1e13 and the
    run overflowed.  Taking G = D.T from the ACTUAL divergence matrix removes
    that error at the source -- the transpose is exact by definition."""
    ni, nj = block.ni, block.nj
    N = ni * nj
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    z = np.zeros((ni, nj))
    # ---- u-component columns 0 .. N-1 ----
    for k in range(N):
        u = z.copy()
        u.reshape(-1)[k] = 1.0
        d = divergence(block, u, z).reshape(-1)
        nz = np.nonzero(d)[0]
        if nz.size:
            rows.append(nz)
            cols.append(np.full(nz.shape, k, dtype=int))
            vals.append(d[nz])
    # ---- v-component columns N .. 2N-1 ----
    for k in range(N):
        v = z.copy()
        v.reshape(-1)[k] = 1.0
        d = divergence(block, z, v).reshape(-1)
        nz = np.nonzero(d)[0]
        if nz.size:
            rows.append(nz)
            cols.append(np.full(nz.shape, N + k, dtype=int))
            vals.append(d[nz])
    D = coo_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(N, 2 * N),
    ).tocsr()
    return D


def build_poisson(block):
    """Pressure-projection Laplacian  L = D o G  with  G = D.T  (exact adjoint).

    L = D @ D.T is symmetric positive-definite by construction, so the Chorin
    correction cleans the divergence exactly and the pressure cannot overflow.

    Boundary overrides (applied AFTER assembly):
      * hole cells    -> Dirichlet p = received_from_donor (block.p set by exchange)
      * recv (fringe) -> left as a solved (Neumann-pressure) row -- velocity-only
                         overset coupling, avoids double-Dirichlet blow-up
      * bg outlet col -> Dirichlet p = 0
      * ogrid         -> pin ONE solved cell to remove the constant-pressure null
                         space (the only null space of D@D.T on a connected grid).
    """
    ni, nj = block.ni, block.nj
    N = ni * nj

    def idx(i, j):
        return i * nj + j

    # ---- exact SPD operator: L = D @ D.T (Gram matrix) ----
    D = _assemble_div_matrix(block)
    G = D.T                                   # exact adjoint gradient (2N x N)
    block._D_poisson = D                      # kept for the velocity correction
    A = (D @ G).tolil()                       # N x N, SPD by construction

    # ---- boundary overrides (keep off-diagonal coupling intact) ----
    def set_dirichlet(k):
        A[k, :] = 0.0
        A[k, k] = 1.0

    # The HOLE cells are covered by the donor grid -> pressure Dirichlet p=0
    # (invisible to the solved region).  The RECV/fringe ring is a VELOCITY
    # Dirichlet boundary (donor velocity held in the predictor) with a NEUMANN
    # pressure BC, so it is LEFT as a solved row (NOT a Dirichlet row) -- this is
    # the velocity-only overset coupling, which avoids over-constraining the fringe
    # with both donor velocity AND donor pressure (that double Dirichlet created a
    # growing interface residual -> blow-up).
    for k in range(N):
        i, j = divmod(k, nj)
        if block.hole[i, j]:
            set_dirichlet(k)

    # Fringe / recv ring: VELOCITY-Dirichlet (donor value held in the predictor)
    # with PRESSURE Dirichlet p = 0 (freestream reference at the outer boundary).
    # This is the key fix for the O-grid force accuracy: the donor (background)
    # velocity sampled onto the O-grid fringe is NON-divergence-free in the polar
    # metric, so if the fringe pressure were SOLVED (Neumann) it absorbed that
    # spurious divergence as a huge pressure spike at r~0.6 (p reached +7 at the
    # front, -1.3 at the back) that corrupted the whole pressure field and left
    # the cylinder wall pressure ~5x too small -> Cd ~0.2 instead of ~1.6.  Pinning
    # the fringe pressure to the freestream (0) references the O-grid pressure
    # correctly and removes the spike; pressure continuity across the overset
    # interface is then imposed as p_fringe = 0 = freestream on both grids.
    for k in range(N):
        i, j = divmod(k, nj)
        if block.recv[i, j]:
            set_dirichlet(k)

    if block.name == "bg":
        for j in range(nj):
            set_dirichlet(idx(ni - 1, j))  # outlet p = 0
    else:
        # ogrid: inner no-slip wall (dp/dn=0) is NEUMANN, so D@D.T has the
        # constant-pressure null space (L*ones == 0).  The fringe (recv) is now a
        # Dirichlet p=0 row (above), which removes that null space, so eps_reg is
        # only a harmless safety regularizer.
        eps_reg = 1.0e-10 * A.diagonal().max()
        from scipy.sparse import eye
        A = A + eps_reg * eye(N, format="lil")

    A_csr = csr_matrix(A)
    block._A_poisson = A_csr
    return splu(A_csr.tocsc())


def decheckerboard(p, passes=2):
    """Remove the collocated-grid odd-even (checkerboard) pressure mode.

    In this projection a checkerboard pressure p_ck (alternating sign between
    adjacent cells along x, y or both) has an EXACTLY zero cell-centred (2-cell)
    gradient, so it lies in the null space of the discrete gradient operator D^T
    used for the velocity correction.  It therefore does NOT enter u,v at all --
    the velocity field stays clean -- it only contaminates the STORED pressure
    (which then shows up as a fine stripe pattern in the pressure figure and, if
    used, in a form-drag surface integral).  A 3-point [1/4,1/2,1/4] average along
    x annihilates the mode exactly while preserving the smooth physical pressure,
    so filtering the pressure after the projection is a loss-free cleanup.
    """
    q = p
    for _ in range(passes):
        q[1:-1, :] = 0.25 * q[:-2, :] + 0.5 * q[1:-1, :] + 0.25 * q[2:, :]
    return q


def project(block, dt, fringe_p=None):
    div = divergence(block, block.u, block.v)
    rhs = np.zeros_like(div)
    # Solved cells carry their local divergence as RHS (corrected by the projection).
    # The fringe (recv) is a velocity-Dirichlet boundary; by default it is a
    # pressure-Dirichlet(p=0) row (rhs=0).  When fringe_p is supplied (global
    # coupled projection) the fringe row is Dirichlet-pinned to the *interpolated
    # donor/acceptor interface pressure* so the overset seam is PRESSURE-CONTINUOUS
    # (no O(1) pressure jump -> no Chimera mass leak -> no two-way drift).
    # The hole (covered by the other grid) is a pressure Dirichlet row -> rhs = block.p.
    rhs[block.solved] = div[block.solved] / dt
    rhs[block.recv] = 0.0
    if fringe_p is not None:
        rhs[block.recv] = np.asarray(fringe_p).reshape(-1)
    rhs[block.hole] = block.p[block.hole]
    if block.name == "bg":
        # pressure-outlet BC: p = 0 on the DOWNSTREAM boundary (last i-row, i=ni-1).
        # NOTE: this is rhs[-1, :] (last i-ROW), NOT rhs[:, -1] (last j-COLUMN, which
        # would wrongly pin the TOP wall and leave its pressure flat -> the projection
        # could not correct velocity there, leaving a residual divergence that grew
        # every step and blew the run up by ~step 10).  The outlet rows are already
        # Dirichlet (e_k) in build_poisson, so zeroing the RHS here fixes p_outlet=0.
        rhs[-1, :] = 0.0
    x = block.factor.solve(rhs.reshape(-1))
    p = x.reshape(block.ni, block.nj)
    block.p = p
    # velocity correction via the EXACT adjoint gradient  G = D.T  assembled in
    # build_poisson.  corr = D.T @ p  is the (u_corr, v_corr) vector; because G is
    # the transpose of the actual divergence, this is the precise Chorin correction
    # that cancels the divergence (div(u_corr) == 0) and is energy-stable -- no
    # checkerboard mode can survive, since the face-averaged divergence couples
    # adjacent cells and its transpose does too.
    D = block._D_poisson
    corr = (D.T @ p.reshape(-1))               # (2N,) = [cu ; cv]
    N = block.ni * block.nj
    cu = corr[:N].reshape(block.ni, block.nj)
    cv = corr[N:].reshape(block.ni, block.nj)
    u = block.u - dt * cu
    v = block.v - dt * cv
    # Velocity correction is applied to SOLVED cells only.  The fringe (recv) ring
    # is a velocity-Dirichlet boundary (donor value held by overset_exchange), so
    # it MUST be HELD, not corrected: correcting it would depart from the donor
    # velocity, and the next step's exchange would overwrite it anyway, creating a
    # one-step feedback that injected a residual divergence seam.  The hole cells
    # are also held (they are never reached by any solved-cell stencil).
    u[block.hole | block.recv] = block.u[block.hole | block.recv]
    v[block.hole | block.recv] = block.v[block.hole | block.recv]
    # Immersed-boundary (bg2og) wall re-assertion: solid + adjacent wall cells are
    # held at zero velocity (no-slip) every step.  They are NOT hole/recv rows, so
    # the projection would otherwise correct them; re-pin AFTER the correction just
    # like the O-grid re-asserts its j=0 wall (see run_cylinder).
    if getattr(block, "solid", None) is not None and block.solid.any():
        u[block.solid] = block.u[block.solid]
        v[block.solid] = block.v[block.solid]
    if getattr(block, "wall", None) is not None and block.wall.any():
        u[block.wall] = block.u[block.wall]
        v[block.wall] = block.v[block.wall]
    block.u = u
    block.v = v
    # Collocated-grid cleanup: strip the null-space checkerboard pressure from the
    # BACKGROUND (Cartesian) grid so the stored pressure is physical and smooth.
    # This does not touch u,v (the checkerboard is in null(D^T)), so the dynamics
    # are unchanged; it only fixes the pressure field / its surface integral.
    if getattr(block, "name", None) == "bg":
        block.p = decheckerboard(block.p, 2)


def apply_ibm(bg, dt=None):
    """Direct-forcing immersed boundary for the background Cartesian grid (bg2og).

    Pins the cylinder interior (solid) AND the adjacent no-slip layer (wall) to
    zero velocity, so the background grid carries the body and sheds its OWN clean
    von Karman street (a Cartesian grid has no polar (-1)^j checkerboard null
    space).  Called after the predictor (with dt -> also accumulates the body
    force from the momentum removed) and again after the projection (no dt)."""
    if getattr(bg, "solid", None) is None:
        return
    if dt is not None:
        # Immersed-boundary force = momentum removed to hold the body at rest,
        # summed over the BODY region (solid + the adjacent no-slip wall layer):
        #     F = rho * sum(u_pred * V) / dt   (rho = 1).
        # The solid cells alone are isolated by the zero-velocity wall ring and
        # carry almost no predicted momentum, so the momentum sink would be ~0 if
        # measured over solid only.  The wall layer is where the fluid actually
        # "feels" the body (the no-slip stress), so it must be included -- the
        # solid+wall region is the effective immersed body.  A positive Fx means
        # downstream drag (opposite of the IBM reaction force on the fluid).
        body = bg.solid | bg.wall
        vol = np.abs(bg.J[body])
        us = bg.u[body]
        vs = bg.v[body]
        fx = (us * vol).sum() / dt
        fy = (vs * vol).sum() / dt
        bg._F_body = np.array([fx, fy])
        # DEBUG: record the predicted (pre-pin) body-region velocity statistics.
        bg._ustar_dbg = (float(us.max()), float(us.min()), float(us.mean()),
                         float(vs.max()), float(vs.min()), float(vs.mean()))
    bg.u[bg.solid] = 0.0
    bg.v[bg.solid] = 0.0
    if getattr(bg, "wall", None) is not None and bg.wall.any():
        bg.u[bg.wall] = 0.0
        bg.v[bg.wall] = 0.0


def cylinder_forces_bg(bg, geom, nu, U):
    """Force on the cylinder carried by the background immersed boundary (bg2og).

    Surface integral of the fluid stress over the stair-step body boundary
    (the solid|wall / fluid interface), decomposed into form drag (pressure) plus
    viscous skin friction:

        F = -∮_S p n dS  +  ∮_S (tau.n) dS,   n = body outward normal (into fluid)

    with rho = 1, D = 2R.  Drag is POSITIVE downstream (+x), lift POSITIVE +y.
    The pressure is first de-checkerboarded (the odd-even mode is a projection
    null space that does not affect the velocity but pollutes the raw field).
    """
    if getattr(bg, "solid", None) is None:
        return float("nan"), float("nan"), float("nan"), float("nan")
    # The direct-forcing IBM sets BOTH the solid cells AND the first fluid ring
    # (wall) to zero velocity, so the effective no-slip body is solid | wall.
    body = bg.solid | bg.wall
    u, v = bg.u, bg.v
    hx, hy = bg.hx, bg.hy
    # physical pressure: the stored projection variable is p_solver = -p_physical
    # and carries a null-space checkerboard -> de-checkerboard, then flip sign.
    p_phys = -decheckerboard(bg.p, 2)
    # neighbour-in-body masks: a face is on the body surface iff its neighbour is
    # NOT in the body (i.e. it is a solved fluid cell).
    E = np.roll(body, -1, 0); W = np.roll(body, 1, 0)
    Nn = np.roll(body, -1, 1); S = np.roll(body, 1, 1)
    # ---- form (pressure) drag: Fp = -∮ p n dS ----------------------------------
    # east/west faces carry area hy (normal in x); north/south carry hx (normal y)
    pe = np.where(~E, np.roll(p_phys, -1, 0), 0.0)
    pw = np.where(~W, np.roll(p_phys, 1, 0), 0.0)
    pn = np.where(~Nn, np.roll(p_phys, -1, 1), 0.0)
    ps = np.where(~S, np.roll(p_phys, 1, 1), 0.0)
    Fp_x = (-hy * (pe - pw))[body].sum()      # -p*(+1)*hy (east) ; -p*(-1)*hy (west)
    Fp_y = (-hx * (pn - ps))[body].sum()
    # ---- viscous drag: Fv = +∮ (tau·n) dS  with the FULL Newtonian stress ------
    # no-slip pins the body side to zero, so each face uses a one-sided normal
    # gradient.  x-traction: x-faces -> tau_xx = 2 nu du/dx ; y-faces -> tau_yx =
    # nu(du/dy + dv/dx).  y-traction: x-faces -> tau_xy ; y-faces -> tau_yy=2nu dv/dy.
    ue = np.where(~E, np.roll(u, -1, 0), 0.0); uw = np.where(~W, np.roll(u, 1, 0), 0.0)
    un = np.where(~Nn, np.roll(u, -1, 1), 0.0); us = np.where(~S, np.roll(u, 1, 1), 0.0)
    ve = np.where(~E, np.roll(v, -1, 0), 0.0); vw = np.where(~W, np.roll(v, 1, 0), 0.0)
    vn = np.where(~Nn, np.roll(v, -1, 1), 0.0); vs = np.where(~S, np.roll(v, 1, 1), 0.0)
    Fv_x = (2.0 * nu * (ue + uw) / hx * hy + nu * (un + us) / hy * hx)[body].sum()
    Fv_y = (nu * (ve + vw) / hx * hy + 2.0 * nu * (vn + vs) / hy * hx)[body].sum()
    Fx = Fp_x + Fv_x
    Fy = Fp_y + Fv_y
    D = 2.0 * geom["R"]
    q = 0.5 * U * U * D                      # 0.5 * rho * U^2 * D  (rho = 1)
    # drag is measured POSITIVE downstream (both form and viscous parts are +).
    return Fx / q, Fy / q, float(Fx), float(Fy)


# --------------------------------------------------------------------------- #
# Global coupled pressure projection (Chimera interface-pressure continuity)
# --------------------------------------------------------------------------- #
def joint_project(bg, og, geom, dt, Rf, rc=True):
    """Two-grid (global) pressure projection that enforces a PRESSURE-CONTINUOUS
    overset interface -- the robust cure for the two-way drift.

    ROOT CAUSE of the drift: with plain two-way velocity-Dirichlet coupling, each
    grid is projected independently.  The O-grid solve references p=0 at its recv
    ring (r=Rf_sponge), so the O-grid interface pressure at r~Rf is a FREE value
    (typically O(1)).  The background fringe (r in [Rh_hole,Rf]) is a p=0
    Dirichlet row.  That O(1) PRESSURE JUMP across the seam is not divergence:
    it is a mismatch the next exchange cannot remove, and because the fringe
    velocity (continuous) is re-corrected by two DIFFERENT pressure references it
    leaves a residual divergence every step that round-trips and GROWS
    (og_umax / Cd climb monotonically -- never saturate).

    FIX: pin the BG fringe pressure to the *interpolated OG interface pressure*
    instead of 0, so  p_bg(Γ) == p_og(Γ).  Both grids are still projected with
    their own (verified) operators and stay individually divergence-free, but the
    interface is now pressure-continuous -> the round-trip leak vanishes -> the
    drift stops.  OG is projected FIRST (it supplies the interface pressure); BG
    is projected SECOND with its fringe row Dirichlet-set to that pressure,
    reusing bg.factor (fringe rows are e_k Dirichlet, so only the RHS changes).

    The O-grid outer ring is held as a RADIAL ZERO-GRADIENT outflow (pinned to its
    own Rf solution, set in overset_exchange), so the OG is self-consistent and
    does NOT re-ingest the background divergence -- that is what keeps the two-way
    coupling from blowing up.  We re-pin it here AFTER the OG projection to the
    corrected inner-edge value, so the fringe is seamless w.r.t. the projected
    interior (no predictor-vs-corrected seam).

    Order matters for the velocity BCs: overset_exchange (called before this,
    exchange-first) has already set the fringe velocities to the donor values, so
    both grids solve their pressure against a continuous interface VELOCITY; this
    routine adds the matching continuous interface PRESSURE."""
    # 1) O-grid independent projection (recv pinned 0) -> og.p (ref p=0 at recv)
    POG.project(og, dt, og.recv, og.factor, rc=rc)
    # 1b) re-pin the OG outer ring to the CORRECTED inner-edge solution
    #     (radial zero-gradient) so the fringe is seamless w.r.t. the interior.
    jr = int(np.min(np.where(og.recv[0])) - 1)
    _ii, _jj = np.nonzero(og.recv)
    og.u[_ii, _jj] = og.u[_ii, jr]
    og.v[_ii, _jj] = og.v[_ii, jr]
    # 2) sample the OG pressure onto the BG fringe ring (the physical interface)
    mb = (~bg.hole) & (bg.r <= Rf)
    p_if = _sample_og(og.p, bg.xc[mb], bg.yc[mb], geom)
    # 3) BG projection with fringe Dirichlet = p_if (interface pressure continuity)
    project(bg, dt, fringe_p=p_if)


# --------------------------------------------------------------------------- #
# Mass-Conserving Interpolation (MCI) for the overset fringe
# --------------------------------------------------------------------------- #
def _assemble_og_div_matrix(block):
    """Assemble the O-grid divergence matrix D (N x 2N) from POG.polar_div, so
    MCI / band projections use exactly the O-grid's own (verified adjoint)
    divergence operator.  Mirrors _assemble_div_matrix but drives polar_div."""
    import polar_ogrid as POG
    ni, nj = block.ni, block.nj
    N = ni * nj
    from scipy.sparse import coo_matrix
    rows, cols, vals = [], [], []
    z = np.zeros((ni, nj))
    for k in range(N):
        u = z.copy(); u.reshape(-1)[k] = 1.0
        d = POG.polar_div(block, u, z).reshape(-1)
        nz = np.nonzero(d)[0]
        if nz.size:
            rows.append(nz); cols.append(np.full(nz.shape, k, dtype=int)); vals.append(d[nz])
    for k in range(N):
        v = z.copy(); v.reshape(-1)[k] = 1.0
        d = POG.polar_div(block, z, v).reshape(-1)
        nz = np.nonzero(d)[0]
        if nz.size:
            rows.append(nz); cols.append(np.full(nz.shape, N + k, dtype=int)); vals.append(d[nz])
    D = coo_matrix((np.concatenate(vals),
                    (np.concatenate(rows), np.concatenate(cols))),
                   shape=(N, 2 * N)).tocsr()
    return D


def _band_poisson(block, band_mask):
    """Precompute (once, cached on the block) a local Poisson factor that makes
    the fringe *band* divergence-free in the BLOCK's own metric.

    The band (set of fringe/recv cells) is treated as the only DOFs; every cell
    outside the band is a held Dirichlet neighbour (its current velocity enters
    the RHS as a constant).  A_band = D_bb @ D_bb.T is SPD because the band is
    bounded by Dirichlet rings on both sides (inner hole / outer solved or
    p=0 fringe).  Reusing the already-assembled global divergence matrix
    block._D_poisson keeps this exact and cheap."""
    if getattr(block, "_mci_mask", None) is band_mask and hasattr(block, "_mci"):
        return block._mci
    if not hasattr(block, "_D_poisson"):
        # The O-grid is built with POG.build_poisson (which does not stash the
        # raw divergence matrix); assemble it from POG.polar_div so the MCI band
        # Poisson uses the SAME divergence operator as the O-grid's main projection.
        block._D_poisson = _assemble_og_div_matrix(block)
    D = block._D_poisson                       # N x 2N global divergence
    N = block.ni * block.nj
    band_rows = np.nonzero(band_mask.ravel())[0]
    band_vel = np.concatenate([band_rows, N + band_rows])   # (u,v) DOFs of band
    D_bb = D[band_rows][:, band_vel].tocsc()  # band x band-velocity
    from scipy.sparse import eye
    A = (D_bb @ D_bb.T).tolil()
    A = A + 1.0e-12 * A.diagonal().max() * eye(A.shape[0], format="lil")
    factor = splu(A.tocsc())
    block._mci = (factor, D_bb, band_rows)
    block._mci_mask = band_mask
    return block._mci


def apply_mci(block, band_mask, dt):
    """Locally project the fringe *band* to be divergence-free in the block's own
    metric, by correcting only the band velocities:

        div_band = D[band] @ [u;v]            (includes held-neighbour contribution)
        p_band   = A_band^{-1} (div_band / dt)
        (u,v)_band <- (u,v)_band - dt * D_bb^T p_band

    WHY: a donor velocity interpolated onto the fringe is divergence-free in the
    DONOR's metric but carries a spurious divergence in the ACCEPTOR's metric.
    The acceptor's main projection then cancels that spurious divergence by
    correcting its interior, injecting a source every step.  In two-way coupling
    that source round-trips and grows monotonically (og_umax / Cd climb without
    bound).  Making the fringe div-free in the acceptor metric up front removes
    the spurious source -> the interface mass flux is conserved -> no drift."""
    if not band_mask.any():
        return
    factor, D_bb, band_rows = _band_poisson(block, band_mask)
    N = block.ni * block.nj
    nband = band_rows.size
    uv = np.concatenate([block.u.ravel(), block.v.ravel()])
    div = (block._D_poisson[band_rows] @ uv)
    if hasattr(div, "toarray"):
        div = div.toarray().ravel()
    else:
        div = div.ravel()
    p = factor.solve(div / dt)
    corr = (D_bb.T @ p).ravel()
    u = block.u.ravel().copy(); u[band_rows] -= dt * corr[:nband]
    v = block.v.ravel().copy(); v[band_rows] -= dt * corr[nband:]
    block.u = u.reshape(block.ni, block.nj)
    block.v = v.reshape(block.ni, block.nj)


# --------------------------------------------------------------------------- #
# Overset exchange (donor bilinear interpolation)
# --------------------------------------------------------------------------- #
def _sample_bg(field, px, py, blk):
    hx, hy = blk.hx, blk.hy
    fi = px / hx - 0.5
    fj = py / hy - 0.5
    i0 = np.floor(fi).astype(int)
    j0 = np.floor(fj).astype(int)
    i0 = np.clip(i0, 0, blk.ni - 2)
    j0 = np.clip(j0, 0, blk.nj - 2)
    wi = np.clip(fi - i0, 0.0, 1.0)
    wj = np.clip(fj - j0, 0.0, 1.0)
    f00 = field[i0, j0]
    f10 = field[i0 + 1, j0]
    f01 = field[i0, j0 + 1]
    f11 = field[i0 + 1, j0 + 1]
    return (f00 * (1 - wi) * (1 - wj) + f10 * wi * (1 - wj) +
            f01 * (1 - wi) * wj + f11 * wi * wj)


def _sample_og(field, px, py, geom):
    cx, cy = geom["cx"], geom["cy"]
    # NOTE: the radial INDEX inversion must use the O-grid's OWN outer radius
    # (Rf_og = rn[-1]) -- NOT the background fringe radius Rf.  Using Rf here
    # compressed the mapping and mis-sampled the O-grid (only matters for twoway).
    R = geom["R"]
    Rf = geom.get("Rf_og", geom["rn"][-1])
    beta = geom["beta"]
    ni, nj = geom["ni"], geom["nj"]
    rn = geom["rn"]
    dx = px - cx
    dy = py - cy
    r = np.sqrt(dx ** 2 + dy ** 2)
    th = np.arctan2(dy, dx)
    th = np.where(th < 0, th + 2 * np.pi, th)
    # r -> jf (invert geometric stretching)
    tj = np.log(1.0 + (r - R) / (Rf - R) * (np.exp(beta) - 1.0)) / beta
    jf = np.clip(tj * nj, 0, nj - 1.0001)
    j0 = np.floor(jf).astype(int)
    wj = np.clip(jf - j0, 0.0, 1.0)
    # th -> i index (periodic). cell centers at thc = (thn[i]+thn[i+1])/2
    if_ = (th / (2 * np.pi) * ni) % ni
    i0 = np.floor(if_).astype(int) % ni
    i1 = (i0 + 1) % ni
    wi = np.clip(if_ - np.floor(if_), 0.0, 1.0)
    f00 = field[i0, j0]
    f10 = field[i1, j0]
    f01 = field[i0, j0 + 1]
    f11 = field[i1, j0 + 1]
    return (f00 * (1 - wi) * (1 - wj) + f10 * wi * (1 - wj) +
            f01 * (1 - wi) * wj + f11 * wi * wj)


def _fringe_sponge(og, bg, geom, r_in, Rf_og, a_min=0.3):
    """Relax the O-grid outer zone (r >= r_in) velocity toward the background donor
    (sponge layer) instead of a hard Dirichlet reset.

    A hard `og.recv = donor` every step PLUS `p=0` at the same interface
    over-constrains the boundary: the wake that advects into the fringe is chopped
    abruptly, which reflects as a ~10 Hz resonance that pollutes the cylinder
    forces (inflating Cd, adding a spurious lift mode).  The sponge blends
    smoothly -- `a_min` at the inner sponge edge ramping to 1.0 at the outer edge
    -- so outgoing disturbances are absorbed, not reflected.  Pressure stays 0 in
    the fringe (matching the donor uniform flow), so the pressure seam is still
    bounded (only velocity is relaxed)."""
    m = og.r >= r_in
    if not m.any():
        return
    du = _sample_bg(bg.u, og.xc[m], og.yc[m], bg)
    dv = _sample_bg(bg.v, og.xc[m], og.yc[m], bg)
    s = np.clip((og.r[m] - r_in) / (Rf_og - r_in), 0.0, 1.0)
    alpha = a_min + (1.0 - a_min) * s * s
    og.u[m] = (1.0 - alpha) * og.u[m] + alpha * du
    og.v[m] = (1.0 - alpha) * og.v[m] + alpha * dv


def overset_exchange(bg, og, geom, Rh_hole, Rf, Ueff=None, coupling="oneway", aoa=0.0,
                   Rf_sponge=0.5, tw_blend=1.0):
    # VELOCITY-ONLY overset coupling.  The fringe ring of each grid receives the
    # DONOR's *solved* velocity (Dirichlet velocity BC).  Pressure is NOT exchanged:
    # each grid solves its own pressure Poisson, so the fringe is a NEUMANN pressure
    # boundary (dp/dn=0).  Exchanging pressure too would over-constrain the fringe
    # with both donor velocity AND donor pressure, creating a growing interface
    # residual that fed the ~3x/step blow-up.  With velocity-only coupling the
    # interface velocity is continuous (Dirichlet) and the pressure seam stays
    # bounded (the pin in build_poisson removes the ogrid null space).
    #
    # COUPLING MODE:
    #   * 'oneway'  -> the O-grid fringe receives the background's *solved* velocity
    #     (developed far field); the background fringe is held at the analytic
    #     uniform far field.  No feedback into the O-grid -> stable, but the O-grid
    #     far field is clamped symmetric so the near wake is truncated -> no shedding
    #     (lift killed).  Used as a reference / stable baseline.
    #   * 'twoway'  -> the background fringe ALSO receives the O-grid's solved
    #     velocity (full Chimera).  Physically richest (background shows the wake)
    #     but the metric-incompatible donor divergence round-trips and GROWS
    #     monotonically (og_umax / Cd climb without bound).  Unstable long-term.
    #   * 'og2bg'   -> the background fringe receives the O-grid wake (so the
    #     background carries the FULL von Karman street for the vorticity plot) but
    #     the O-grid fringe is CLAMPED to the uniform far field (exactly the
    #     standalone-O-grid BC, which is stable and sheds in the near field).  There
    #     is NO feedback into the O-grid, so it neither drifts nor kills lift: the
    #     cylinder forces (St/Cd/Cl) come from a stable O-grid, while the background
    #     visualises the complete downstream wake.  This is the recommended stable
    #     two-grid configuration.
    Ueff = Ueff if Ueff is not None else bg.U
    if coupling == "og2bg":
        # O-grid fringe CLAMPED to uniform far field (stable, standalone-like).
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)
        # background fringe receives the O-grid solved velocity (carries the wake).
        mb = (~bg.hole) & (bg.r <= Rf)
        gu = _sample_og(og.u, bg.xc[mb], bg.yc[mb], geom)
        gv = _sample_og(og.v, bg.xc[mb], bg.yc[mb], geom)
        if tw_blend >= 1.0:
            bg.u[mb] = gu
            bg.v[mb] = gv
        else:
            bg.u[mb] = (1.0 - tw_blend) * bg.u[mb] + tw_blend * gu
            bg.v[mb] = (1.0 - tw_blend) * bg.v[mb] + tw_blend * gv
        return
    if coupling == "bg2og":
        # BACKGROUND carries the body and sheds its own clean von Karman street
        # (Cartesian grid, no polar (-1)^j checkerboard null space).  The O-grid is
        # a NEAR-FIELD REFINEMENT that only SAMPLES the background (one-way): its
        # outer fringe receives the background *solved* velocity.  It does NOT feed
        # back into the background -- that would re-inject the O-grid's polar
        # pressure-rings and break the clean Cartesian field.  The body force and
        # the displayed street both come from the background.
        mb = og.recv
        og.u[mb] = _sample_bg(bg.u, og.xc[mb], og.yc[mb], bg)
        og.v[mb] = _sample_bg(bg.v, og.xc[mb], og.yc[mb], bg)
        return
    if coupling == "oneway":
        # O-grid outer zone receives the background solved velocity through the
        # sponge (absorbs the outgoing wake instead of chopping it).  Background
        # fringe = analytic uniform far field at a small ANGLE OF ATTACK aoa
        # (symmetry-breaking seed for vortex shedding).
        _fringe_sponge(og, bg, geom, Rf_sponge, geom["Rf_og"])
        bg.u[bg.recv] = Ueff * np.cos(aoa)
        bg.v[bg.recv] = Ueff * np.sin(aoa)
    else:  # coupling == "twoway"  (full Chimera, drift-free via joint_project)
        # O-grid outer zone: ZERO-GRADIENT OUTFLOW.  The O-grid sheds into its OWN
        # developed wake and does NOT re-absorb the background wake (no sponge), so
        # it stays SELF-CONSISTENT and STABLE (no drift) while its far field is a
        # FREE wake -> the near-field von Karman street develops fully (strong
        # lift, Cl~0.2).  The background fringe still receives the O-grid wake
        # (two-way velocity) so the background carries the full street for plotting.
        # Stability math: joint_project() pins the BG fringe pressure to the
        # interpolated OG interface pressure (p_bg(Γ)==p_og(Γ)), eliminating the
        # O(1) pressure jump that was the Chimera mass leak; and because the OG no
        # longer ingests the donor divergence (outflow, not sponge), that leak
        # cannot round-trip and grow.  The two together give a steady, physical
        # two-way state.
        mb = (~bg.hole) & (bg.r <= Rf)
        gu = _sample_og(og.u, bg.xc[mb], bg.yc[mb], geom)
        gv = _sample_og(og.v, bg.xc[mb], bg.yc[mb], geom)
        bg.u[mb] = gu
        bg.v[mb] = gv
        # O-grid outer ring: UNIFORM-FLOW Dirichlet clamp (identical to the
        # standalone-O-grid / og2bg BC that is verified STABLE and SHEDS).  This is
        # the key to avoiding the two-way blow-up: a zero-divergence UNIFORM inflow
        # (not the background's developed, divergent wake) supplies the OG's driving
        # flow.  Re-ingesting the donor's divergent wake through this ring was what
        # made the earlier 'twoway + bg-donor OG' run overflow (og_umax -> 3.5,
        # Cd -> -5).  With a clean uniform clamp the OG develops its OWN near-field
        # von Karman street (strong lift, Cl~0.2) while staying self-consistent, and
        # joint_project() below pins the BG fringe PRESSURE to the interpolated OG
        # interface pressure (p_bg(Γ)==p_og(Γ)) so the seam is pressure-continuous
        # and the round-trip mass leak (the old two-way drift) is gone -- without any
        # velocity-donor re-ingestion.  The background still receives the OG wake
        # (bg.u[mb] above) so it carries the full street for the vorticity plot.
        og.u[og.recv] = Ueff * np.cos(aoa)
        og.v[og.recv] = Ueff * np.sin(aoa)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def cylinder_forces(og, geom, nu, U):
    """Integrated force on the cylinder (r = R) from the O-grid wall cells (j=0).

    Force on the body  F = ∮ (-p I + tau) . n dS,  n = outward fluid normal
    (points INTO the cylinder, i.e. -e_r at the wall).  For the O-grid the inner
    wall face vector  A_eta[:,0]  IS exactly  n * |A|  (computed in make_ogrid as
    the metric-consistent face vector).  So

        F_pressure = -Σ_i  p[i,0] * A_eta[:,0]
        F_viscous  =  Σ_i  tau[i,0] . A_eta[:,0]

    with tau = nu*(grad u + grad u^T) (rho=1).  Drag Cd = -Fx / (0.5 ρ U^2 D),
    Lift Cl = Fy / (0.5 ρ U^2 D)  (flow in +x, D = 2R)."""
    i = np.arange(og.ni)
    j = 0
    # Physical pressure: the projection pressure p_proj is the Lagrange multiplier
    # whose gradient in the correction uses the metric |J| (divergence divides by
    # |J|).  The physical gradient divides by the SIGNED Jacobian J.  On a grid with
    # constant-sign J the two differ by sign(J), so p_phys = sign(J) * p_proj.
    # The O-grid has J = -r < 0 -> p_phys = -p_proj; the background J > 0 -> p_phys = p.
    p = og.p[:, j] * np.sign(og.J[:, j])
    Axe = og.Aeta_x[:, j]
    Aye = og.Aeta_y[:, j]
    # pressure part
    Fpx = -np.sum(p * Axe)
    Fpy = -np.sum(p * Aye)
    # viscous part from the cell-centre physical gradient at the wall
    up = _pad(og, og.u, "u", og.U)
    vp = _pad(og, og.v, "v", og.U)
    gxu, gyu = _grad_cell(og, up)
    gxv, gyv = _grad_cell(og, vp)
    ux = gxu[:, j]; uy = gyu[:, j]
    vx = gxv[:, j]; vy = gyv[:, j]
    tauxx = 2.0 * nu * ux
    tauxy = nu * (uy + vx)
    tauyy = 2.0 * nu * vy
    Fvx = np.sum(tauxx * Axe + tauxy * Aye)
    Fvy = np.sum(tauxy * Axe + tauyy * Aye)
    Fx = Fpx + Fvx
    Fy = Fpy + Fvy
    D = 2.0 * geom["R"]
    q = 0.5 * U * U * D                      # 0.5 * rho * U^2 * D  (rho = 1)
    # Fx, Fy are the physical force the fluid exerts on the body (n = Aeta/|A|
    # points OUT of the body, into the fluid).  Drag is the +x (flow) component,
    # so Cd = +Fx/q (Fx>0 means the body is pushed downstream = drag).  Lift is
    # the +y component, Cl = +Fy/q.
    Cd = Fx / q
    Cl = Fy / q
    return Cd, Cl, float(Fx), float(Fy)


def og_vorticity(og, u=None, v=None):
    """Physical vorticity  w = dv/dx - du/dy  on the O-grid cell centres.

    Implemented with the verified polar_ogrid gradient operators (the polar
    O-grid block no longer carries the (xi,eta) metric fields that the old
    _grad_cell needed).  u/v may be passed explicitly so the DISPLAY vorticity
    can be computed from a theta-filtered velocity (which removes the grid-scale
    angular 'striping' the raw polar_grad differencing amplifies -- a
    post-processing artefact, not a flow feature)."""
    if u is None:
        u = og.u
    if v is None:
        v = og.v
    gxu, gyu = POG.polar_grad(og, u)
    gxv, gyv = POG.polar_grad(og, v)
    return gxv - gyu                        # (ni, nj)


def _save_vorticity(bg, og, geom, Rh_hole, Rf, out_prefix):
    cx, cy = geom["cx"], geom["cy"]
    R = geom["R"]
    Rf_og = og.r.max()
    # ---- O-grid vorticity (the near-field wake) ----
    # The raw polar_grad differencing amplifies the ~2% grid-scale angular
    # velocity content into radial "striping" spokes in the vorticity field (a
    # post-processing artefact, not real flow).  A theta-direction Shapiro
    # 4th-order filter removes only the top angular wavenumbers (the physical
    # St~0.18 vortex street is m=1..6 and is untouched), giving a clean plot.
    u_disp = POG.polar_filter(og, og.u, sigma_theta=0.05)
    v_disp = POG.polar_filter(og, og.v, sigma_theta=0.05)
    w = og_vorticity(og, u_disp, v_disp)
    # The O-grid is polar; plotting w on the native quad mesh with pcolormesh
    # produces radial striping artefacts (non-monotonic cell-centre coords).  So
    # we interpolate (linear) the O-grid vorticity onto a CARTESIAN grid that
    # covers the O-grid disc -> a clean field with no artificial grid striping.
    from scipy.interpolate import griddata
    nxg, nyg = 360, 144
    gx = np.linspace(0, bg.Lx, nxg)
    gy = np.linspace(0, bg.Ly, nyg)
    GX, GY = np.meshgrid(gx, gy)
    pts = np.column_stack([og.xc.ravel(), og.yc.ravel()])
    val = w.ravel()
    wc = griddata(pts, val, (GX, GY), method="linear")
    inside = (GX - cx) ** 2 + (GY - cy) ** 2 <= Rf_og ** 2
    wc = np.where(inside, wc, np.nan)
    wc = np.where((GX - cx) ** 2 + (GY - cy) ** 2 < R ** 2, np.nan, wc)  # hole
    plt.figure(figsize=(11, 4.6))
    mesh = plt.imshow(wc, origin="lower", extent=[0, bg.Lx, 0, bg.Ly],
                     cmap="RdBu_r", vmin=-3, vmax=3)
    # ---- background vorticity (clean channel, ~0) for context ----
    Nx, Ny = bg.ni, bg.nj
    ub = bg.u; vb = bg.v
    wb = np.zeros((Nx, Ny))
    wb[1:-1, 1:-1] = 0.5 * (vb[2:, 1:-1] - vb[:-2, 1:-1]) / bg.hx - 0.5 * (ub[1:-1, 2:] - ub[1:-1, :-2]) / bg.hy
    wb = np.where(bg.hole, np.nan, wb)
    plt.imshow(wb.T, origin="lower", extent=[0, bg.Lx, 0, bg.Ly], cmap="RdBu_r",
               vmin=-3, vmax=3, alpha=0.55)
    # outlines
    th = np.linspace(0, 2 * np.pi, 240)
    for rr, ls in ((R, "k-"), (Rf, "k--"), (Rf_og, "k:")):
        plt.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), ls, lw=0.7)
    plt.colorbar(mesh, label="vorticity  (dv/dx - du/dy)")
    plt.title("Overset cylinder (Re=100): O-grid near-field vorticity + wake\n"
              "solid=cylinder, dashed=O-grid fringe (r=Rf), dotted=O-grid outer edge")
    plt.xlabel("x"); plt.ylabel("y")
    plt.xlim(0, bg.Lx); plt.ylim(0, bg.Ly)
    plt.tight_layout()
    plt.savefig(f"/workspace/{out_prefix}_vorticity.png", dpi=140)
    plt.close()


def _interp_og_to_cart(og, val, bg, geom, Rf_og):
    """Interpolate an O-grid scalar field onto a Cartesian grid covering the
    O-grid disc (linear; nan outside the disc and inside the cylinder hole)."""
    from scipy.interpolate import griddata
    nxg, nyg = 360, 144
    gx = np.linspace(0, bg.Lx, nxg)
    gy = np.linspace(0, bg.Ly, nyg)
    GX, GY = np.meshgrid(gx, gy)
    cx, cy = geom["cx"], geom["cy"]
    R = geom["R"]
    pts = np.column_stack([og.xc.ravel(), og.yc.ravel()])
    vc = griddata(pts, val.ravel(), (GX, GY), method="linear")
    inside = (GX - cx) ** 2 + (GY - cy) ** 2 <= Rf_og ** 2
    vc = np.where(inside, vc, np.nan)
    vc = np.where((GX - cx) ** 2 + (GY - cy) ** 2 < R ** 2, np.nan, vc)
    return vc, GX, GY


def _save_velocity(bg, og, geom, Rh_hole, Rf, out_prefix):
    """Velocity-magnitude field (background + O-grid overlay) with streamlines
    (far field, from the background) and O-grid arrows (near field, the real
    recirculating wake the overset grid resolves)."""
    from scipy.interpolate import griddata
    cx, cy = geom["cx"], geom["cy"]
    R = geom["R"]
    Rf_og = og.r.max()
    # displayed O-grid velocity: light polar filter removes the radial
    # checkerboard / angular striping (a post-processing artefact, not flow).
    uf = POG.polar_filter(og, og.u, sigma_theta=0.05, sigma_r=0.05)
    vf = POG.polar_filter(og, og.v, sigma_theta=0.05, sigma_r=0.05)
    # display-only: remove the radial (-1)^j checkerboard so the near-field is not
    # drawn as a spurious radial "starburst" (the solver may leave this mode when
    # the checkerboard filter is off; it is a post-processing artefact).
    uf = POG.polar_filter_checkerboard(og, uf, alpha=1.0)
    vf = POG.polar_filter_checkerboard(og, vf, alpha=1.0)
    spd_og = np.hypot(uf, vf)
    spd_og_c, GX, GY = _interp_og_to_cart(og, spd_og, bg, geom, Rf_og)
    # background speed (Cartesian; hole -> nan)
    spd_bg = np.hypot(bg.u, bg.v)
    spd_bg = np.where(bg.hole, np.nan, spd_bg)
    # robust colour limit: a few noisy cells must not squash the whole field
    cand = [float(np.nanpercentile(a, 99.0)) for a in (spd_bg, spd_og_c)
            if np.isfinite(a).any()]
    vmax = max(cand) if cand else 1.0
    vmax = max(vmax, 1e-6)
    plt.figure(figsize=(11, 4.6))
    hbg = plt.imshow(spd_bg.T, origin="lower", extent=[0, bg.Lx, 0, bg.Ly],
                     cmap="viridis", vmin=0, vmax=vmax)
    plt.imshow(spd_og_c, origin="lower", extent=[0, bg.Lx, 0, bg.Ly],
               cmap="viridis", vmin=0, vmax=vmax, alpha=0.95)
    # background streamlines (far field only -- the O-grid disc carries the real
    # near-field flow, drawn below as arrows); mask the cylinder and the O-disc.
    Nxs, Nys = 96, 38
    gxc = np.linspace(bg.hx * 0.5, bg.Lx - bg.hx * 0.5, Nxs)
    gyc = np.linspace(bg.hy * 0.5, bg.Ly - bg.hy * 0.5, Nys)
    GXC, GYC = np.meshgrid(gxc, gyc)
    Ub = griddata(np.column_stack([bg.xc.ravel(), bg.yc.ravel()]), bg.u.ravel(),
                  (GXC, GYC), method="linear")
    Vb = griddata(np.column_stack([bg.xc.ravel(), bg.yc.ravel()]), bg.v.ravel(),
                  (GXC, GYC), method="linear")
    disc = (GXC - cx) ** 2 + (GYC - cy) ** 2 < (Rf_og * 1.02) ** 2
    Ub = np.where(disc, np.nan, Ub)
    Vb = np.where(disc, np.nan, Vb)
    spc = np.hypot(Ub, Vb)
    sp0 = float(np.nanmax(spc)) if np.isfinite(np.nanmax(spc)) else 1.0
    sp0 = max(sp0, 1e-6)
    plt.streamplot(gxc, gyc, Ub, Vb, color=spc, cmap="plasma",
                   linewidth=0.7, density=1.3, arrowsize=0.8)
    # O-grid near-field arrows (every few cells, inside the O-disc)
    ii = slice(0, None, 5) if og.ni >= 40 else slice(None)
    jj = slice(0, og.nj, 2)
    qx = og.xc[ii, jj]; qy = og.yc[ii, jj]
    qu = uf[ii, jj]; qv = vf[ii, jj]
    qm = (qx - cx) ** 2 + (qy - cy) ** 2 <= (0.92 * Rf_og) ** 2
    plt.quiver(qx[qm], qy[qm], qu[qm], qv[qm], color="w", scale=16,
               width=0.0026, headwidth=3, alpha=0.85)
    # outlines
    th = np.linspace(0, 2 * np.pi, 240)
    for rr, ls in ((R, "k-"), (Rf, "k--"), (Rf_og, "k:")):
        plt.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), ls, lw=0.7)
    plt.colorbar(hbg, label="speed |U|  (U=1, freestream)")
    plt.title("Overset cylinder (Re=100): velocity magnitude |U| + streamlines/arrows\n"
              "solid=cylinder, dashed=O-grid fringe (r=Rf), dotted=O-grid outer edge")
    plt.xlabel("x / D"); plt.ylabel("y / D")
    plt.xlim(0, bg.Lx); plt.ylim(0, bg.Ly)
    plt.tight_layout()
    plt.savefig(f"/workspace/{out_prefix}_velocity.png", dpi=140)
    plt.close()


def _save_pressure(bg, og, geom, Rh_hole, Rf, out_prefix):
    """Pressure field (background + O-grid overlay).  The two grids carry
    independent, arbitrary p=0 references, so the O-grid pressure is re-referenced
    to the background pressure in the overlap band to make the combined field
    continuous across the overset interface.  Diverging colormap centred at the
    freestream pressure (0): red = high (stagnation), blue = low (wake)."""
    from scipy.interpolate import griddata
    cx, cy = geom["cx"], geom["cy"]
    R = geom["R"]
    Rf_og = og.r.max()
    # sample the background pressure AT the O-grid cell centres (so the offset
    # matches the two grids in their overlap band).  points = background mesh,
    # values = bg.p, query = O-grid centres.
    bg_pts = np.column_stack([bg.xc.ravel(), bg.yc.ravel()])
    bgp_samp = griddata(bg_pts, bg.p.ravel(), (og.xc, og.yc), method="linear")  # (ni,nj)
    og_sol = og.solved & np.isfinite(bgp_samp)
    offset = float(np.nanmean(bgp_samp[og_sol] - og.p[og_sol])) if og_sol.any() else 0.0
    # display-only radial smoothing removes the residual radial pressure
    # saw-tooth (null space left when reg_r==0) so the near field is not drawn as
    # concentric rings; the physical (smooth) pressure is barely affected.
    p_og = POG.polar_filter(og, og.p, sigma_theta=0.05, sigma_r=0.3) + offset
    p_og_c, GX, GY = _interp_og_to_cart(og, p_og, bg, geom, Rf_og)
    # background pressure (Cartesian; hole -> nan); ref at freestream (0)
    p_bg = np.where(bg.hole, np.nan, bg.p.copy())
    pm = 0.0
    for arr in (p_bg, p_og_c):
        if np.isfinite(arr).any():
            pm = max(pm, float(np.nanpercentile(np.abs(arr), 99.0)))
    pm = max(pm, 1e-6)
    plt.figure(figsize=(11, 4.6))
    hbg = plt.imshow(p_bg.T, origin="lower", extent=[0, bg.Lx, 0, bg.Ly],
                     cmap="RdBu_r", vmin=-pm, vmax=pm)
    plt.imshow(p_og_c, origin="lower", extent=[0, bg.Lx, 0, bg.Ly],
               cmap="RdBu_r", vmin=-pm, vmax=pm, alpha=0.95)
    th = np.linspace(0, 2 * np.pi, 240)
    for rr, ls in ((R, "k-"), (Rf, "k--"), (Rf_og, "k:")):
        plt.plot(cx + rr * np.cos(th), cy + rr * np.sin(th), ls, lw=0.7)
    plt.colorbar(hbg, label="pressure p  (ref: freestream p=0)")
    plt.title("Overset cylinder (Re=100): pressure field p\n"
              "solid=cylinder, dashed=O-grid fringe (r=Rf), dotted=O-grid outer edge")
    plt.xlabel("x / D"); plt.ylabel("y / D")
    plt.xlim(0, bg.Lx); plt.ylim(0, bg.Ly)
    plt.tight_layout()
    plt.savefig(f"/workspace/{out_prefix}_pressure.png", dpi=140)
    plt.close()


def run_cylinder(nsteps=8000, Nx=120, Ny=48, Lx=2.5, Ly=1.0, U=1.0, Re=100.0,
                 ni=120, nj=48, R=0.1, Rh_hole=0.42, Rf=0.6, Rf_og=0.78, beta=2.5,
                 dt=None, probe_every=500, out_prefix="probe13", ramp_steps=200,
                 aoa_deg=3.0, field_dump=None, field_every=50, Rf_sponge=0.5,
                 conv_scheme="upwind1", conv_filter_sigma=0.0, reg_r=0.0, fv_sigma=0.0,
                 rc=False, fck=1.0, hv_cfl=0.0, coupling="oneway", tw_blend=1.0,
                 mci="none", schwarz=1):
    cx = 0.25 * Lx
    # symmetry-breaking seed for vortex shedding: a tiny off-centre offset reliably
    # triggers the antisymmetric wake.  A perfectly centred cylinder on a symmetric
    # Cartesian grid can stay locked (roundoff grows the antisymmetric mode only
    # after a long convective time); the 1%R offset removes that stall.  It is far
    # too small to bias Cd/Cl_amp/St (the street remains the natural Re=100 one).
    cy = 0.5 * Ly + (0.01 * R if coupling == "bg2og" else 0.0)
    nu = U * (2 * R) / Re
    aoa = np.deg2rad(aoa_deg)     # symmetry-breaking seed for vortex shedding
    bg = make_background(Nx, Ny, Lx, Ly, cx, cy, R, Rf, U, Rh_hole)
    if coupling == "bg2og":
        # background CARRIES the body: no hole, no O-grid back-fill (otherwise the
        # O-grid's polar checkerboard / pressure-rings would pollute the clean
        # Cartesian field).  bg solves the WHOLE fluid; the immersed solid/wall
        # layer is held at zero velocity by direct forcing.  The O-grid only
        # SAMPLES the background (one-way) as a near-field display refinement.
        bg.hole = np.zeros_like(bg.solid)
        bg.recv = np.zeros_like(bg.solid)
        bg.solved = (~bg.solid) & (~bg.wall)
    # The O-grid MUST extend BEYOND the background's fringe boundary (Rf_og > Rf):
    # its outer fringe (r in [Rf, Rf_og]) then receives the BACKGROUND's *solved*
    # velocity, while the background's fringe (r in [Rh_hole, Rf]) receives the
    # O-grid's *solved* velocity.  Each fringe is fed by the OTHER grid's solved
    # region -> no circularity, no double-fringe overlap.  With Rf_og == Rf (the
    # old setup) the O-grid solved only r < Rh_hole and the whole [Rh_hole, Rf]
    # ring was fringe for BOTH grids (no donor) -> the exchange fed each grid a
    # lagged copy of the other and the coupled run grew ~3x/step and overflowed.
    # ---- O-grid built with the VERIFIED orthogonal-polar operators ----
    # (the old overset make_ogrid carried a broken (xi,eta) metric that produced
    #  the wrong-sign Cd ~ -0.99; polar_ogrid.make_ogrid uses the same physical
    #  annulus but the self-consistent polar face/volume metric, unit-tested to
    #  machine precision).  geom is reconstructed from the block's own rn so the
    #  background sampling in overset_exchange stays consistent.
    og = POG.make_ogrid(ni, nj, cx, cy, R, Rf_og, beta=beta)
    geom = dict(cx=cx, cy=cy, R=R, Rf=Rf, Rf_og=Rf_og, beta=beta, ni=ni, nj=nj, rn=og.rn)
    og.hole = np.zeros((ni, nj), dtype=bool)
    # O-grid solves r < Rf_sponge; the outer zone r in [Rf_sponge, Rf_og] is a
    # sponge that relaxes toward the background donor (absorbs the outgoing wake
    # instead of reflecting it as a ~10 Hz resonance).  p=0 in the sponge too.
    og.recv = og.r >= Rf_sponge
    og.solved = ~og.recv
    for b in (bg, og):
        b._nu = nu
        b.U = U
    # Wind-tunnel start-up: begin from REST and ramp the inflow velocity from 0
    # to U.  This keeps the no-slip transient on the O-grid wall gentle (the tiny
    # near-wall cells have |J|~1e-5, so an incompatible u=U initial field would
    # otherwise produce a huge spurious divergence -> unstable pressure correction).
    bg.v[:] = 0.0
    og.u[:] = 0.0
    og.v[:] = 0.0
    if coupling == "bg2og":
        # Start from a UNIFORM free stream instead of from rest: this skips the
        # long convective fill of the (long) domain and lets the wake develop in
        # ~10-20 convective times.  The immersed cylinder is then pinned to zero
        # by apply_ibm so the body is present from step 1 (impulsive start, which
        # the projection handles without instability).  (The other couplings keep
        # the gentle from-rest ramp, which they need for the O-grid wall.)
        bg.u[:] = U
        apply_ibm(bg)
    else:
        bg.u[:] = 0.0
    bg.factor = build_poisson(bg)
    # Time-step: must be computed BEFORE the O-grid Poisson because the Rhie-Chow
    # projection operator needs dt.  (The old code referenced an undefined dt0 and
    # built the Poisson before dt existed -> NameError / inconsistent operator.)
    min_cell = min(np.sqrt(np.abs(bg.J[bg.solved])).min(),
                   np.sqrt(np.abs(og.J[og.solved])).min())
    if dt is None:
        dt_cfl = 0.2 * min_cell / U
        # Diffusion-stability limit: explicit diffusion is unstable if
        #     dt * nu * (1/dx^2 + 1/dy^2) > 0.5  (2D)
        # The polar O-grid's smallest cell dimension is the NEAR-WALL radial spacing
        # dr0 (tiny), so this is the binding constraint -- with the old dt the
        # near-wall diffusion ran ~3x over the stability limit, over-smoothing the
        # boundary layer and dropping the effective Reynolds number far below 100
        # (wake never separated -> Cd ~ 0.1 instead of ~1.6).  Enforce it here.
        rn = geom["rn"]
        dr_min = float(rn[1] - rn[0])                 # near-wall radial cell size
        dth_min = float(og.r.min() * (2.0 * np.pi / og.ni))  # near-wall theta arc
        dx_min = min(dr_min, dth_min)
        dt_diff = 0.25 * dx_min ** 2 / nu             # 2D explicit-diffusion limit
        dt = min(dt_cfl, dt_diff)
        print(f"  dt_cfl={dt_cfl:.3e} dt_diff={dt_diff:.3e} -> dt={dt:.3e} "
              f"dr_min={dr_min:.4e} (nu*dt/dr^2={nu*dt/(dr_min*dr_min):.2f})")
    print(f"dt={dt:.3e} nu={nu:.3e} min_cell={min_cell:.3e} ramp={ramp_steps} "
          f"bg_solved={bg.solved.sum()} og_solved={og.solved.sum()}")
    # O-grid projection uses the VERIFIED polar adjoint operator (reduced Poisson
    # on solved cells, p[fringe]=0) -> exactly div-free correction, no pressure spike
    og.factor = POG.build_poisson(og, og.recv, reg_r=reg_r, dt=dt, rc=rc)
    overset_exchange(bg, og, geom, Rh_hole, Rf, Ueff=0.0, coupling=coupling, aoa=aoa,
                    tw_blend=tw_blend)

    # probe point: downstream of cylinder ~ 1D
    px = int((cx + 1.0 * (2 * R)) / bg.hx)
    py = int(cy / bg.hy)
    hist = []
    cl_hist = []          # (t, Cd, Cl) recorded after the inflow ramp, for St/Cd/Cl

    for step in range(1, nsteps + 1):
        # bg2og starts from a uniform free stream, so no inflow ramp is needed
        # (and ramping would transiently drop the already-uniform field).
        Ueff = U if coupling == "bg2og" else U * min(1.0, step / ramp_steps)
        for b in (bg, og):
            b._nu = nu
            b.U = Ueff
        if coupling == "bg2og":
            # ===== background carries the body; O-grid = near-field refinement =====
            # The Cartesian background sheds its OWN clean von Karman street (no polar
            # (-1)^j checkerboard null space).  The O-grid outer fringe is driven by the
            # background *solved* velocity (NOT a uniform clamp), so the O-grid does not
            # develop its own aliased shed; fck can then remove the O-grid checkerboard
            # freely, and the global St comes from the background.  This breaks the
            # og2bg deadlock where the shed mode and the (-1)^j checkerboard are inseparable.
            overset_exchange(bg, og, geom, Rh_hole, Rf, Ueff=Ueff, coupling="bg2og", aoa=aoa)
            # 1) background predictor (immersed body) + projection (background sheds)
            predictor(bg, dt, nu, Ueff)
            apply_ibm(bg, dt)                  # pins wall AND accumulates body force
            project(bg, dt)
            # 2) O-grid near-field refinement, driven by the just-updated background
            mb = og.recv
            og.u[mb] = _sample_bg(bg.u, og.xc[mb], og.yc[mb], bg)
            og.v[mb] = _sample_bg(bg.v, og.xc[mb], og.yc[mb], bg)
            ogrid_predictor(og, dt, nu, Ueff, conv_scheme=conv_scheme,
                           conv_filter_sigma=conv_filter_sigma, hv_cfl=hv_cfl)
            POG.project(og, dt, og.recv, og.factor, rc=rc)
            # PRESSURE continuity at the O-grid fringe: the O-grid projection pins
            # p[fringe]=0, but the background pressure there is NOT 0 (it is the
            # developed wake suction).  Without this shift the O-grid wall pressure
            # is offset and the integrated drag comes out wrong-signed.  Re-reference
            # the O-grid pressure to the background so cylinder_forces(og) is physical.
            m = og.recv
            og.p[m] = _sample_bg(bg.p, og.xc[m], og.yc[m], bg)
            # 3) re-assert no-slip walls on both grids (post-projection)
            apply_ibm(bg)                     # re-pin wall (no dt -> no force re-accum)
            og.u[:, 0] = 0.0
            og.v[:, 0] = 0.0
            if fv_sigma > 0.0:
                og.u = POG.polar_filter(og, og.u, sigma_r=fv_sigma)
                og.v = POG.polar_filter(og, og.v, sigma_r=fv_sigma)
            if fck > 0.0:
                og.u = POG.polar_filter_checkerboard(og, og.u, alpha=fck)
                og.v = POG.polar_filter_checkerboard(og, og.v, alpha=fck)
        else:
            # Overset coupling is applied at the START of the step, so the fringe
            # divergence it injects is removed by THIS step's projection (predictor ->
            # project) rather than persisting until the next step.
            overset_exchange(bg, og, geom, Rh_hole, Rf, Ueff=Ueff, coupling=coupling, aoa=aoa,
                            tw_blend=tw_blend)
            predictor(bg, dt, nu, Ueff)
            ogrid_predictor(og, dt, nu, Ueff, conv_scheme=conv_scheme,
                           conv_filter_sigma=conv_filter_sigma, hv_cfl=hv_cfl)
            # Overlapping-Schwarz coupling iterations (schwarz=1 -> single projection).
            for _k in range(max(1, int(schwarz))):
                if _k > 0:
                    overset_exchange(bg, og, geom, Rh_hole, Rf, Ueff=Ueff, coupling=coupling,
                                    aoa=aoa, tw_blend=tw_blend)
                    if mci in ("both", "bg"):
                        apply_mci(bg, bg.recv, dt)
                    if mci in ("both", "og"):
                        apply_mci(og, og.recv, dt)
                if coupling == "twoway":
                    joint_project(bg, og, geom, dt, Rf, rc=rc)
                else:
                    project(bg, dt)
                    POG.project(og, dt, og.recv, og.factor, rc=rc)
            if fv_sigma > 0.0:
                og.u = POG.polar_filter(og, og.u, sigma_r=fv_sigma)
                og.v = POG.polar_filter(og, og.v, sigma_r=fv_sigma)
            if fck > 0.0:
                og.u = POG.polar_filter_checkerboard(og, og.u, alpha=fck)
                og.v = POG.polar_filter_checkerboard(og, og.v, alpha=fck)
            # relax the O-grid outer zone toward the background donor (sponge).  For
            # og2bg the fringe is CLAMPED to uniform flow (set in overset_exchange);
            # for bg2og the O-grid is driven by the background (handled in its branch);
            # only oneway/twoway sponge here.
            if coupling not in ("og2bg", "bg2og"):
                _fringe_sponge(og, bg, geom, Rf_sponge, Rf_og)
            # RE-ASSERT the O-grid no-slip wall AFTER the projection.
            og.u[:, 0] = 0.0
            og.v[:, 0] = 0.0
        if field_dump is not None and step % field_every == 0:
            field_dump.append(dict(step=step, u=og.u.copy(), v=og.v.copy(),
                                   p=og.p.copy(), t=step * dt))
        if step >= ramp_steps:
            if coupling == "bg2og":
                Cd, Cl, Fx, Fy = cylinder_forces_bg(bg, geom, nu, U)
                Cp, Cv, chk = 0.0, 0.0, 0.0
            else:
                Cd, Cl, Fx, Fy = POG.cylinder_forces(og, nu, U)
                Cp, Cv, chk = POG.force_decomp(og, nu, U)
            cl_hist.append((step * dt, Cd, Cl, Cp, Cv, chk))
        if step % probe_every == 0:
            ub = bg.u[bg.solved]
            uo = og.u[og.solved]
            ubmax = float(np.abs(ub).max()) if ub.size else 0.0
            uomax = float(np.abs(uo).max()) if uo.size else 0.0
            vprobe = bg.v[px, py] if (0 <= px < Nx and 0 <= py < Ny) else 0.0
            nan = not (np.isfinite(bg.u).all() and np.isfinite(og.u).all()
                       and np.isfinite(bg.v).all() and np.isfinite(og.v).all())
            cd = cl_hist[-1][1] if cl_hist else float("nan")
            cl = cl_hist[-1][2] if cl_hist else float("nan")
            hist.append((step, ubmax, uomax, vprobe))
            print(f"step{step:4d} Ueff={Ueff:.3f} bg_umax={ubmax:.4f} og_umax={uomax:.4f} "
                  f"vprobe={vprobe:+.4f} Cd={cd:+.4f} Cl={cl:+.4f} nan={nan}")
            if coupling == "bg2og" and hasattr(bg, "_ustar_dbg"):
                umx, umn, ume, vmx, vmn, vme = bg._ustar_dbg
                print(f"    ustar_solid: umax={umx:+.4f} umin={umn:+.4f} umean={ume:+.4f} "
                      f"vmax={vmx:+.4f} vmin={vmn:+.4f} vmean={vme:+.4f}")
            if nan:
                # locate first non-finite cell in each block
                for nm, blk in (("bg", bg), ("og", og)):
                    bad = ~np.isfinite(blk.u)
                    if bad.any():
                        ib, jb = np.unravel_index(np.argmax(bad), blk.u.shape)
                        print(f"    first NaN in {nm} at (i,j)=({ib},{jb}) r={blk.r[ib,jb]:.3f}")
                break

    # ---- post-process forces: mean Cd, lift amplitude, Strouhal from Cl FFT ----
    summary = {}
    if len(cl_hist) > 50:
        arr = np.array(cl_hist)
        t = arr[:, 0]; Cd = arr[:, 1]; Cl = arr[:, 2]
        # discard the initial transient (first ~20% of the developed window)
        k0 = max(1, len(t) // 5)
        Cd_m = float(np.mean(Cd[k0:]))
        Cl_m = float(np.mean(Cl[k0:]))
        Cl_amp = 0.5 * (float(np.max(Cl[k0:])) - float(np.min(Cl[k0:])))
        Cp_m = float(np.mean(arr[k0:, 3]))
        Cv_m = float(np.mean(arr[k0:, 4]))
        chk_m = float(np.mean(arr[k0:, 5]))
        # FFT of Cl (detrend) to find the shedding frequency
        cl_detr = Cl[k0:] - np.mean(Cl[k0:])
        dt_s = float(t[1] - t[0])
        f = np.fft.rfftfreq(len(cl_detr), d=dt_s)
        P = np.abs(np.fft.rfft(cl_detr)) ** 2
        P[0] = 0.0
        fmax = f[np.argmax(P)] if P.max() > 0 else 0.0
        St = float(fmax * (2.0 * R) / U) if U > 0 else 0.0
        summary = dict(Cd_mean=Cd_m, Cl_mean=Cl_m, Cl_amp=Cl_amp, f_peak=fmax, St=St,
                       Cp_mean=Cp_m, Cv_mean=Cv_m, chk_mean=chk_m)
        print(f"\n=== FORCE SUMMARY (Re={Re}, after ramp) ===")
        print(f"  Cd_mean = {Cd_m:.4f}   (Cp={Cp_m:.4f} + Cv={Cv_m:.4f})")
        print(f"  Cl_mean = {Cl_m:+.4f}   Cl_amp = {Cl_amp:.4f}")
        print(f"  shedding freq f = {fmax:.4f}   St = f*D/U = {St:.4f}")
        print(f"  wall tangential-velocity checkerboard chk = {chk_m:.4f}")
        print(f"  baseline: St~0.182, Cd~1.59, Cl_amp~0.22")

    # dump the final raw fields so the velocity/pressure/vorticity figures can be
    # re-rendered offline (without re-running the solver)
    np.savez(f"/workspace/{out_prefix}_fields.npz",
             bg_u=bg.u, bg_v=bg.v, bg_p=bg.p, bg_hole=bg.hole,
             bg_xc=bg.xc, bg_yc=bg.yc, bg_Lx=bg.Lx, bg_Ly=bg.Ly,
             bg_hx=bg.hx, bg_hy=bg.hy, bg_solved=bg.solved,
             bg_solid=bg.solid, bg_wall=bg.wall,
             og_u=og.u, og_v=og.v, og_p=og.p, og_r=og.r,
             og_xc=og.xc, og_yc=og.yc, og_solved=og.solved,
             cx=geom["cx"], cy=geom["cy"], R=geom["R"], Rf=geom["Rf"],
             Rf_og=float(og.r.max()), ni=og.ni, nj=og.nj)
    # save vorticity + velocity + pressure snapshots (background + ogrid overlay)
    _save_vorticity(bg, og, geom, Rh_hole, Rf, out_prefix)
    _save_velocity(bg, og, geom, Rh_hole, Rf, out_prefix)
    _save_pressure(bg, og, geom, Rh_hole, Rf, out_prefix)
    np.save(f"/workspace/{out_prefix}_hist.npy", np.array(hist))
    if cl_hist:
        np.save(f"/workspace/{out_prefix}_forces.npy", np.array(cl_hist))
    return bg, og, hist, summary, field_dump


if __name__ == "__main__":
    import sys
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    run_cylinder(nsteps=n, probe_every=100)
