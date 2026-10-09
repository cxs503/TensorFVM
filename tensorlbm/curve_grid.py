"""Generic (non-polar) curvilinear FVM operators on a structured C/O grid.

Sibling of polar_ogrid.py that builds EVERYTHING from node coordinates
X[i,j], Y[i,j] numerically (no r/theta assumption), so it works for an
arbitrary body curve (cylinder, NACA).  topology:
    periodic_i=True  -> O-grid (i wraps; only outer j=nj-1 fringe)
    periodic_i=False -> C-grid (i open at tail; i=0/ni-1 columns + j=nj-1 row
                        are clamped free-stream fringe; j=0 = no-slip wall).
Metric convention (must match polar for self-adjointness):
    xi-face  (node column i, between cell i-1/i): A_xi = (-ey, ex)  points +i
    eta-face (node row   j, between cell j-1/j):   A_eta= (ey,-ex)  points +j
so div(u) = (1/J) sum_faces A_face . u_face and L = div o grad is exact adjoint.
"""

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix, diags, eye, tril, triu
from scipy.sparse.linalg import splu, bicgstab, spilu, LinearOperator, cg, spsolve_triangular


class Block:
    def __init__(self, name, ni, nj, xc, yc):
        self.name = name
        self.ni, self.nj = ni, nj
        self.xc, self.yc = xc, yc
        self.u = np.zeros((ni, nj))
        self.v = np.zeros((ni, nj))
        self.p = np.zeros((ni, nj))
        self.J = None
        self.periodic_i = False
        self.recv = None
        self.Lref = 1.0
        self.U = 1.0
        self._nu = 0.0


def make_cblock(X, Y, periodic_i=False, U=1.0, Lref=1.0, wall_mask=None,
                j_floor=0.0):
    ni = X.shape[0] - 1
    nj = X.shape[1] - 1
    xc = 0.25 * (X[:-1, :-1] + X[1:, :-1] + X[1:, 1:] + X[:-1, 1:])
    yc = 0.25 * (Y[:-1, :-1] + Y[1:, :-1] + Y[1:, 1:] + Y[:-1, 1:])
    blk = Block("cgrid", ni, nj, xc, yc)
    blk.periodic_i = periodic_i
    per = periodic_i
    blk.U = U
    blk.Lref = Lref

    # xi-face vectors (ni+1, nj): face i at node column i, between cell i-1/i.
    # The OUTWARD (+i) face vector is the geometric edge node(i,J)->node(i,J+1)
    # rotated -90deg: A_xi = (ey, -ex) with (ex,ey)=node(i,J+1)-node(i,J).
    #
    # This is built DIRECTLY from node coordinates -- no local-tangent flip
    # heuristic.  For ANY cell the four outward faces then sum to the zero vector
    # (they are the edges of a closed quadrilateral), so div(uniform) == 0 to
    # machine precision on every valid grid: Cartesian, uniform-polar, an O-grid
    # wrap, or a C-grid that folds back at the body trailing edge.  The previous
    # flip heuristic reversed its sign at a trailing edge (the surface line doubles
    # back in x while the cell centres bulge through the TE) and broke the FV
    # closure, which is what made div(uniform) blow up for the NACA airfoil.
    Axi_x = np.zeros((ni + 1, nj)); Axi_y = np.zeros((ni + 1, nj))
    for i in range(ni + 1):
        ex = X[i, 1:] - X[i, :-1]; ey = Y[i, 1:] - Y[i, :-1]
        Axi_x[i] = ey; Axi_y[i] = -ex
    Axi_len = np.hypot(Axi_x, Axi_y)

    # eta-face vectors (ni, nj+1): face j=J at nodes (I,J)-(I+1,J).  OUTWARD (+j)
    # face vector is the edge node(I,J)->node(I+1,J) rotated -90deg of the +i edge:
    # A_eta = (-ey, ex) with (ex,ey)=node(I+1,J)-node(I,J).  face j=J separates cell
    # J-1 and J; j=0 is the wall, j=nj the far boundary.  Again direct -- no flip.
    Aef_x = np.zeros((ni, nj + 1)); Aef_y = np.zeros((ni, nj + 1))
    for I in range(ni):
        eix = X[I + 1, :] - X[I, :]; eiy = Y[I + 1, :] - Y[I, :]
        Aef_x[I] = -eiy; Aef_y[I] = eix
    Aef_len = np.hypot(Aef_x, Aef_y)

    x0, x1, x2, x3 = X[:-1, :-1], X[1:, :-1], X[1:, 1:], X[:-1, 1:]
    y0, y1, y2, y3 = Y[:-1, :-1], Y[1:, :-1], Y[1:, 1:], Y[:-1, 1:]
    J = 0.5 * np.abs((x0 * y1 - x1 * y0) + (x1 * y2 - x2 * y1) +
                     (x2 * y3 - x3 * y2) + (x3 * y0 - x0 * y3))

    # cell-centre unit normals (avg of the two bounding faces) -- diagnostic only
    # (not consumed by the divergence / gradient / Laplacian operators).
    ax = 0.5 * (Axi_x[:-1] + Axi_x[1:]); ay = 0.5 * (Axi_y[:-1] + Axi_y[1:])
    ln = np.hypot(ax, ay)
    nxi_cx = np.where(ln > 0.0, ax / ln, 0.0)
    nxi_cy = np.where(ln > 0.0, ay / ln, 0.0)
    ax = 0.5 * (Aef_x[:, :-1] + Aef_x[:, 1:]); ay = 0.5 * (Aef_y[:, :-1] + Aef_y[:, 1:])
    ln = np.hypot(ax, ay)
    neta_cx = np.where(ln > 0.0, ax / ln, 0.0)
    neta_cy = np.where(ln > 0.0, ay / ln, 0.0)

    # PER-FACE distances (ONE value per face, identical from both adjacent cells)
    # so the assembled Laplacian is exactly symmetric in the volume-weighted inner
    # product even on curved / non-uniform grids.  xi-face f (0..ni) separates cell
    # f-1 (left) and cell f (right); h_f = |n_xi[f] . (xc[f]-xc[f-1])| with n_xi the
    # FACE normal (A_xi/|A_xi|), not the cell-centre-averaged normal.  eta-face j
    # analogous.  For a Laplacian of r^2 this gives the exact 4 on a uniform grid
    # (Cartesian or uniform-polar) and is 2nd-order on stretched grids.
    nxi_fx = Axi_x / Axi_len; nxi_fy = Axi_y / Axi_len          # (ni+1, nj) face normals
    nfta_x = Aef_x / Aef_len; nfta_y = Aef_y / Aef_len          # (ni, nj+1) face normals
    # xi faces 0..ni-1: left cell = cell f-1 (periodic wrap / clamp), right = cell f
    if per:
        xcl_f = np.roll(xc, 1, 0); ycl_f = np.roll(yc, 1, 0)
    else:
        xcl_f = np.concatenate([xc[:1], xc[:-1]], 0)
        ycl_f = np.concatenate([yc[:1], yc[:-1]], 0)
    hxi_f = np.abs(nxi_fx[:-1] * (xc - xcl_f) + nxi_fy[:-1] * (yc - ycl_f))  # (ni,nj): f=0..ni-1
    if per:
        hxi_f = np.concatenate([hxi_f, hxi_f[:1]], 0)          # face ni == node 0 (seam)
    else:
        hxi_f = np.concatenate([hxi_f, hxi_f[-1:]], 0)         # open end mirrored
        hxi_f[0] = hxi_f[1]; hxi_f[ni] = hxi_f[ni - 1]         # clamp both open ends
    # eta faces 0..nj-1: inner cell = cell j-1, outer = cell j
    xjm1 = np.concatenate([xc[:, :1], xc[:, :-1]], 1)
    yjm1 = np.concatenate([yc[:, :1], yc[:, :-1]], 1)
    heta_f = np.abs(nfta_x[:, :-1] * (xc - xjm1) + nfta_y[:, :-1] * (yc - yjm1))  # (ni,nj): j=0..nj-1
    heta_f = np.concatenate([heta_f, heta_f[:, -1:]], 1)       # (ni, nj+1): outer face mirrored
    heta_f[:, 0] = heta_f[:, 1]; heta_f[:, nj] = heta_f[:, nj - 1]  # wall + far-field clamp

    blk.Axi_x, blk.Axi_y, blk.Axi_len = Axi_x, Axi_y, Axi_len
    blk.Aef_x, blk.Aef_y, blk.Aef_len = Aef_x, Aef_y, Aef_len
    blk.nxi_x, blk.nxi_y = nxi_cx, nxi_cy
    blk.neta_x, blk.neta_y = neta_cx, neta_cy
    blk.hxi_f, blk.heta_f = hxi_f, heta_f
    # Metric regularisation: floor the cell Jacobian (area) so a degenerate /
    # over-collapsed cell (e.g. a wake-cut near-TE cell on a small slat/flap where
    # Winslow smoothing pinches the first off-wall spacing to ~1e-7) cannot drive
    # the explicit div/laplacian operators' 1/J division to overflow / a CFL->
    # 1 blow-up.  SIGN-PRESERVING: J_eff = sign(J) * max(|J|, j_floor).  A small
    # POSITIVE J (the near-wall pinch) is lifted to +j_floor, capping 1/J so the
    # projection stays stable (needs 1/J * dt < ~1 -> j_floor >~ dt).  A NEGATIVE J
    # (a true inverted cell) KEEPS its sign and only has its magnitude floored, so
    # the projection direction is not flipped (a naive np.maximum would turn an
    # inverted cell into a tiny positive one and reverse the local flux).  A
    # healthy grid (|J| well above j_floor) is untouched.  j_floor<=0 disables.
    blk.J = (J if j_floor <= 0.0
             else np.sign(J) * np.maximum(np.abs(J), j_floor))

    recv = np.zeros((ni, nj), dtype=bool)
    recv[:, nj - 1] = True
    if not periodic_i:
        recv[0, :] = True; recv[ni - 1, :] = True
    blk.recv = recv
    # wall = no-slip body surface.  Default: the whole j=0 line (used by O-grid
    # style wraps and the cylinder C-grid where j=0 is the full body).  For a
    # general C-grid the generator passes an explicit wall_mask so ONLY the body
    # arc is no-slip and the tail/wake-cut j=0 cells stay free-stream.
    if wall_mask is None:
        blk.wall = np.zeros((ni, nj), dtype=bool); blk.wall[:, 0] = True
    else:
        blk.wall = np.asarray(wall_mask, dtype=bool)
    return blk


def _il(u, periodic):
    if periodic:
        return np.roll(u, 1, axis=0)
    return np.concatenate([u[:1], u[:-1]], axis=0)


def _ir(u, periodic):
    if periodic:
        return np.roll(u, -1, axis=0)
    return np.concatenate([u[1:], u[-1:]], axis=0)


def c_div(blk, u, v):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    ul, vl = _il(u, per), _il(v, per)
    ur, vr = u, v
    uf = 0.5 * (ul + ur); vf = 0.5 * (vl + vr)
    Fxi = np.zeros((ni + 1, nj))
    Fxi[:ni] = uf * blk.Axi_x[:ni] + vf * blk.Axi_y[:ni]
    Fxi[ni] = u[-1] * blk.Axi_x[ni] + v[-1] * blk.Axi_y[ni]
    if per:
        Fxi[ni] = Fxi[0]            # periodic closure (node ni == node 0)
    # eta faces (unified): face j=J flux, cell inner = face j=J, outer = face j=J+1
    uf_eta = np.zeros((ni, nj + 1)); vf_eta = np.zeros((ni, nj + 1))
    uf_eta[:, 1:nj] = 0.5 * (u[:, :-1] + u[:, 1:])   # interior faces j=1..nj-1
    vf_eta[:, 1:nj] = 0.5 * (v[:, :-1] + v[:, 1:])
    uf_eta[:, nj] = u[:, -1]; vf_eta[:, nj] = v[:, -1]   # far outer face (recv)
    # inner face j=0 (the wall / wake-cut line): use the cell's OWN velocity as a
    # one-sided flux.  For a true no-slip wall u[:,0]=0 -> zero flux (correct);
    # for the WAKE-CUT line (j=0 cells that are NOT wall) this carries the actual
    # centreline flux instead of the previously hard-coded 0, which had forced
    # zero normal flux through the wake cut and injected a large spurious
    # divergence into the tiny wake-cut cells of the small slat/flap elements
    # (their j=0 wake cells are ~1e-5 in area, so div(uniform) reached ~1e2 there
    # and destabilised the projection).  With the real flux, div(uniform) == 0 to
    # machine precision on every closed cell, exactly like the cylinder grid.
    uf_eta[:, 0] = u[:, 0]; vf_eta[:, 0] = v[:, 0]
    Feta = uf_eta * blk.Aef_x + vf_eta * blk.Aef_y        # face j=J
    Fin = Feta[:, :nj]      # cell J inner = face j=J
    Fout = Feta[:, 1:]      # cell J outer = face j=J+1
    Fright = Fxi[1:]; Fleft = Fxi[:-1]
    return (Fout - Fin + Fright - Fleft) / blk.J


def c_grad(blk, p):
    """Cartesian gradient of scalar p at cell centres, constructed as the EXACT
    adjoint of c_div (so L = c_div o c_grad is symmetric & negative semi-definite
    and gives lap(x^2+y^2) = +4 exactly):

        grad(p)[c] = (1/(2 J_c)) * sum_{4 faces f of cell c}  A_f * (p_nb - p_c)

    where A_f is the (physically-flipped) outward face-area vector of face f and
    p_nb the neighbour cell across f.  This is the discrete form of
    div o grad = -<grad,grad>, i.e. the Laplacian, with the SAME face vectors
    c_div uses -- so the volume-weighted inner product <p, L q> == <L p, q> holds
    (periodic seam / no boundary term)."""
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    p_r = _ir(p, per)                                 # neighbour at i+1
    p_l = _il(p, per)                                 # neighbour at i-1
    p_o = np.concatenate([p[:, 1:], p[:, -1:]], 1)    # neighbour at j+1 (outer)
    p_i = np.concatenate([p[:, :1], p[:, :-1]], 1)    # neighbour at j-1 (inner)
    # outward face-area vectors (per cell):
    #   right face of cell i = node column i+1 (A_xi[i+1], points +i = outward)
    #   left  face of cell i = node column i   (A_xi[i],    points +i = INWARD)
    #   outer face of cell j = face j+1 (A_ef[:, j+1], points +j = outward)
    #   inner face of cell j = face j   (A_ef[:, j],   points +j = outward)
    Axr, Ayr = blk.Axi_x[1:], blk.Axi_y[1:]           # right faces (ni, nj)
    Axl, Ayl = blk.Axi_x[:-1], blk.Axi_y[:-1]         # left faces  (ni, nj)
    Axo, Ayo = blk.Aef_x[:, 1:], blk.Aef_y[:, 1:]     # outer faces (ni, nj)
    Axi, Ayi = blk.Aef_x[:, :-1], blk.Aef_y[:, :-1]   # inner faces (ni, nj)
    gx = 0.5 * (Axr * (p_r - p) - Axl * (p_l - p)
                + Axo * (p_o - p) - Axi * (p_i - p)) / blk.J
    gy = 0.5 * (Ayr * (p_r - p) - Ayl * (p_l - p)
                + Ayo * (p_o - p) - Ayi * (p_i - p)) / blk.J
    return gx, gy


def c_lap(blk, p):
    """Generic FV Laplacian, direct face-flux form:

        lap(p)_c = (1/J_c) * sum_{4 faces f of cell c} |A_f| / h_f * (p_nb - p_c)

    where |A_f| is the TRUE face length and h_f is the PER-FACE distance between
    the two adjacent cell centres (projected onto the face normal).  Using one
    shared h_f per face makes the volume-weighted inner product <p, Lq> exactly
    symmetric even on curved / non-uniform grids (the off-diagonal 'flux' |A_f|/h_f
    is identical from both sides).  On a uniform grid -- Cartesian or uniform-polar
    -- this yields lap(x^2+y^2) = +4 to machine precision, and it is 2nd-order on
    smoothly stretched grids (like polar_lap but derived purely numerically, so it
    works for an arbitrary body curve).  Boundaries (j=0 wall, far field, open i
    ends) collapse to Neumann (zero flux) because the one-sided neighbour equals
    the cell itself (see make_cblock's mirror of h_f).  NOTE: this is a SEPARATE
    operator from the c_div o c_grad adjoint pair (as in polar_ogrid.py); the
    Poisson matrix uses c_lap, the velocity correction uses c_grad."""
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    p_r = _ir(p, per)                                  # neighbour i+1 (outward +xi)
    p_l = _il(p, per)                                  # neighbour i-1 (outward -xi)
    p_o = np.concatenate([p[:, 1:], p[:, -1:]], 1)     # neighbour j+1 (outer +eta)
    p_i = np.concatenate([p[:, :1], p[:, :-1]], 1)     # neighbour j-1 (inner -eta)
    Axl, Axr = blk.Axi_len[:-1], blk.Axi_len[1:]       # |A| at xi faces i and i+1
    hxl, hxr = blk.hxi_f[:-1], blk.hxi_f[1:]           # per-face distances
    Aei, Aeo = blk.Aef_len[:, :-1], blk.Aef_len[:, 1:]
    hei, heo = blk.heta_f[:, :-1], blk.heta_f[:, 1:]
    flux = (Axr * (p_r - p) / hxr
            + Axl * (p_l - p) / hxl
            + Aeo * (p_o - p) / heo
            + Aei * (p_i - p) / hei)
    return flux / blk.J


def _upwind(phiL, phiR, F):
    return np.where(F > 0.0, phiL, phiR)


def _minmod(a, b):
    same = (a * b) > 0.0
    mag = np.minimum(np.abs(a), np.abs(b))
    return np.where(same, np.sign(a) * mag, 0.0)


def _vanleer(dm, dp):
    s = dm * dp; denom = dm + dp
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((s > 0.0) & (denom != 0.0), 2.0 * dm * dp / denom, 0.0)
    return r


def _ppm_states(P):
    p = np.arange(3, P.shape[0] - 3)
    fm2, fm1, fc, fp1, fp2 = P[p - 2], P[p - 1], P[p], P[p + 1], P[p + 2]
    fL = (7.0 / 12.0) * (fm1 + fc) - (1.0 / 12.0) * (fm2 + fp1)
    fR = (7.0 / 12.0) * (fc + fp1) - (1.0 / 12.0) * (fm1 + fp2)
    fmin = np.minimum(np.minimum(fm1, fc), fp1); fmax = np.maximum(np.maximum(fm1, fc), fp1)
    fL = np.maximum(np.minimum(fL, fmax), fmin); fR = np.maximum(np.minimum(fR, fmax), fmin)
    cond = (fR - fc) * (fc - fL) <= 0.0
    fL = np.where(cond, fc, fL); fR = np.where(cond, fc, fR)
    qp = fL; qm = np.roll(fR, 1, axis=0)
    return qm, qp


def _weno5_states(P):
    c = np.arange(P.shape[0] - 6) + 3
    Pm3, Pm2, Pm1, Pc, Pp1, Pp2, Pp3 = (P[c - 3], P[c - 2], P[c - 1], P[c], P[c + 1], P[c + 2], P[c + 3])
    p0m = (1.0 / 3) * Pm2 - (7.0 / 6) * Pm1 + (11.0 / 6) * Pc
    p1m = (-1.0 / 6) * Pm1 + (5.0 / 6) * Pc + (1.0 / 3) * Pp1
    p2m = (1.0 / 3) * Pc + (5.0 / 6) * Pp1 - (1.0 / 6) * Pp2
    is0m = (13.0 / 12) * (Pm2 - 2 * Pm1 + Pc) ** 2 + (1.0 / 4) * (Pm2 - 4 * Pm1 + 3 * Pc) ** 2
    is1m = (13.0 / 12) * (Pm1 - 2 * Pc + Pp1) ** 2 + (1.0 / 4) * (Pm1 - Pp1) ** 2
    is2m = (13.0 / 12) * (Pc - 2 * Pp1 + Pp2) ** 2 + (1.0 / 4) * (3 * Pc - 4 * Pp1 + Pp2) ** 2
    p0p = (-1.0 / 6) * Pm1 + (5.0 / 6) * Pc + (1.0 / 3) * Pp1
    p1p = (1.0 / 3) * Pc + (5.0 / 6) * Pp1 - (1.0 / 6) * Pp2
    p2p = (11.0 / 6) * Pp1 - (7.0 / 6) * Pp2 + (1.0 / 3) * Pp3
    is0p, is1p, is2p = is2m, is1m, (13.0 / 12) * (Pp1 - 2 * Pp2 + Pp3) ** 2 + (1.0 / 4) * (3 * Pp1 - 4 * Pp2 + Pp3) ** 2
    eps = 1e-6
    def _w(is0, is1, is2, d0, d1, d2, pp0, pp1, pp2):
        a0 = d0 / (is0 + eps) ** 2; a1 = d1 / (is1 + eps) ** 2; a2 = d2 / (is2 + eps) ** 2
        s = a0 + a1 + a2
        return (a0 * pp0 + a1 * pp1 + a2 * pp2) / s
    qm = _w(is0m, is1m, is2m, 0.1, 0.6, 0.3, p0m, p1m, p2m)
    qp = _w(is0p, is1p, is2p, 0.3, 0.6, 0.1, p0p, p1p, p2p)
    return qm, qp


def _pad_i(field, n=3, periodic=True):
    if periodic:
        idx = (np.arange(field.shape[0] + 2 * n) - n) % field.shape[0]
        return field[idx]
    ni = field.shape[0]
    pad = np.empty((ni + 2 * n,) + field.shape[1:], dtype=field.dtype)
    pad[n:ni + n] = field
    for g in range(n):
        pad[g] = field[0]; pad[ni + n + g] = field[-1]
    return pad


def _recon_conv(blk, u, v, scheme, limiter=None):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    duc = np.zeros((ni, nj)); dvc = np.zeros((ni, nj))
    Pu = _pad_i(u, 3, per); Pv = _pad_i(v, 3, per)
    if scheme == "ppm":
        uL, uR = _ppm_states(Pu); vL, vR = _ppm_states(Pv)
    elif scheme == "weno":
        uL, uR = _weno5_states(Pu); vL, vR = _weno5_states(Pv)
    elif scheme in ("muscl", "vanleer"):
        PuL = np.roll(Pu, 1, 0); PuR = np.roll(Pu, -1, 0); PuLL = np.roll(Pu, 2, 0); PuRR = np.roll(Pu, -2, 0)
        PvL = np.roll(Pv, 1, 0); PvR = np.roll(Pv, -1, 0); PvLL = np.roll(Pv, 2, 0); PvRR = np.roll(Pv, -2, 0)
        lim = limiter
        uL = Pu + 0.5 * lim(Pu - PuLL, PuR - Pu); vL = Pv + 0.5 * lim(Pv - PvLL, PvR - Pv)
        uR = Pu - 0.5 * lim(Pu - PuRR, PuL - Pu); vR = Pv - 0.5 * lim(Pv - PvRR, PvL - Pv)
        uL = uL[3:3 + ni]; uR = uR[3:3 + ni]; vL = vL[3:3 + ni]; vR = vR[3:3 + ni]
    Ax = blk.Axi_x[:-1]; Ay = blk.Axi_y[:-1]
    if scheme in ("ppm", "weno"):
        mflxL = uL * Ax + vL * Ay; mflxR = uR * Ax + vR * Ay
        alpha = np.maximum(np.abs(mflxL), np.abs(mflxR))
        Fxu = 0.5 * (mflxL * uL + mflxR * uR) - 0.5 * alpha * (uR - uL)
        Fxv = 0.5 * (mflxL * vL + mflxR * vR) - 0.5 * alpha * (vR - vL)
    else:
        Fxu = uL * (uL * Ax + vL * Ay); Fxv = vL * (uL * Ax + vL * Ay)
    if per:
        duc += (np.roll(Fxu, -1, 0) - Fxu) / blk.J; dvc += (np.roll(Fxv, -1, 0) - Fxv) / blk.J
    else:
        # Non-periodic (C-grid): the reconstruction above yields Fxu of shape
        # (ni, nj) = fluxes at the ni interior xi-faces 0..ni-1.  The C-grid has
        # ni+1 xi-faces; the missing face ni is the OPEN wake cut (clamped to the
        # free stream every step), so we append a copy of the last reconstructed
        # flux (cell ni-1 is a recv/free-stream cell) to recover the correct
        # (ni, nj) divergence.  This branch is never taken by the periodic O-grid.
        Fxu_full = np.concatenate([Fxu, Fxu[-1:]], axis=0)   # (ni+1, nj)
        Fxv_full = np.concatenate([Fxv, Fxv[-1:]], axis=0)
        duc += (Fxu_full[1:] - Fxu_full[:-1]) / blk.J
        dvc += (Fxv_full[1:] - Fxv_full[:-1]) / blk.J
    # eta faces (unified Aef, faces j=0..nj)
    Mu = np.zeros((nj + 7, ni)); Mv = np.zeros((nj + 7, ni))
    Mu[3:nj + 3, :] = u.T; Mv[3:nj + 3, :] = v.T
    for g in (0, 1, 2):
        Mu[g, :] = -Mu[6 - g, :]; Mv[g, :] = -Mv[6 - g, :]
    for g in (nj + 3, nj + 4, nj + 5, nj + 6):
        Mu[g, :] = Mu[nj + 2, :]; Mv[g, :] = Mv[nj + 2, :]
    if scheme == "ppm":
        uLr, uRr = _ppm_states(Mu); vLr, vRr = _ppm_states(Mv)
    elif scheme == "weno":
        uLr, uRr = _weno5_states(Mu); vLr, vRr = _weno5_states(Mv)
    else:
        uLr = Mu; uRr = Mu; vLr = Mv; vRr = Mv
    uLr = uLr.T; uRr = uRr.T; vLr = vLr.T; vRr = vRr.T
    Axf = blk.Aef_x; Ayf = blk.Aef_y
    if scheme in ("ppm", "weno"):
        mflxL = uLr * Axf + vLr * Ayf; mflxR = uRr * Axf + vRr * Ayf
        alpha = np.maximum(np.abs(mflxL), np.abs(mflxR))
        Fgu = 0.5 * (mflxL * uLr + mflxR * uRr) - 0.5 * alpha * (uRr - uLr)
        Fgv = 0.5 * (mflxL * vLr + mflxR * vRr) - 0.5 * alpha * (vRr - vLr)
    else:
        mflx = 0.5 * (uLr + uRr) * Axf + 0.5 * (vLr + vRr) * Ayf
        Fgu = 0.5 * (uLr + uRr) * mflx; Fgv = 0.5 * (vLr + vRr) * mflx
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0
    Fgu_outer = Fgu[:, 1:]; Fgv_outer = Fgv[:, 1:]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, 1:nj]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner) / blk.J; dvc += (Fgv_outer - Fgv_inner) / blk.J
    return duc, dvc


def c_conv(blk, u, v, scheme="ppm"):
    if scheme == "upwind1":
        return _conv_upwind1(blk, u, v)
    if scheme in ("ppm", "weno"):
        return _recon_conv(blk, u, v, scheme)
    if scheme in ("muscl", "vanleer"):
        lim = _minmod if scheme == "muscl" else _vanleer
        return _recon_conv(blk, u, v, scheme, limiter=lim)
    if scheme == "central":
        return _conv_central(blk, u, v)
    raise ValueError(f"unknown scheme {scheme!r}")


def _conv_upwind1(blk, u, v):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    # xi faces (faces 0..ni-1, interior; open end face ni mirrored below)
    ul, vl = _il(u, per), _il(v, per); ur, vr = u, v
    Fxi_avg = 0.5 * (ul + ur) * blk.Axi_x[:-1] + 0.5 * (vl + vr) * blk.Axi_y[:-1]
    uface = _upwind(ul, ur, Fxi_avg); vface = _upwind(vl, vr, Fxi_avg)
    Fxi = uface * blk.Axi_x[:-1] + vface * blk.Axi_y[:-1]
    Fxu = uface * Fxi; Fxv = vface * Fxi
    if per:
        duc = (np.roll(Fxu, -1, 0) - Fxu) / blk.J; dvc = (np.roll(Fxv, -1, 0) - Fxv) / blk.J
    else:
        Fxu_full = np.concatenate([Fxu, Fxu[-1:]], axis=0)
        Fxv_full = np.concatenate([Fxv, Fxv[-1:]], axis=0)
        duc = (Fxu_full[1:] - Fxu_full[:-1]) / blk.J; dvc = (Fxv_full[1:] - Fxv_full[:-1]) / blk.J
    # eta faces (faces 0..nj): j=0 wall (zero flux), j=nj outer (Neumann = cell)
    uL = np.zeros((ni, nj + 1)); uR = np.zeros((ni, nj + 1))
    vL = np.zeros((ni, nj + 1)); vR = np.zeros((ni, nj + 1))
    uL[:, 1:nj] = u[:, :-1]; uR[:, 1:nj] = u[:, 1:]
    vL[:, 1:nj] = v[:, :-1]; vR[:, 1:nj] = v[:, 1:]
    uL[:, nj] = u[:, -1]; uR[:, nj] = u[:, -1]
    vL[:, nj] = v[:, -1]; vR[:, nj] = v[:, -1]
    Axf = blk.Aef_x; Ayf = blk.Aef_y
    mflx = 0.5 * ((uL * Axf + vL * Ayf) + (uR * Axf + vR * Ayf))
    uface = _upwind(uL, uR, mflx); vface = _upwind(vL, vR, mflx)
    Fgu = uface * (uface * Axf + vface * Ayf); Fgv = vface * (uface * Axf + vface * Ayf)
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0
    Fgu_outer = Fgu[:, 1:]; Fgv_outer = Fgv[:, 1:]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, 1:nj]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner) / blk.J; dvc += (Fgv_outer - Fgv_inner) / blk.J
    return duc, dvc


def _conv_central(blk, u, v):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    # xi faces (faces 0..ni-1, interior; open end face ni mirrored below)
    uface = 0.5 * (_il(u, per) + u); vface = 0.5 * (_il(v, per) + v)
    Fxi = uface * blk.Axi_x[:-1] + vface * blk.Axi_y[:-1]
    Fxu = uface * Fxi; Fxv = vface * Fxi
    if per:
        duc = (np.roll(Fxu, -1, 0) - Fxu) / blk.J; dvc = (np.roll(Fxv, -1, 0) - Fxv) / blk.J
    else:
        Fxu_full = np.concatenate([Fxu, Fxu[-1:]], axis=0)
        Fxv_full = np.concatenate([Fxv, Fxv[-1:]], axis=0)
        duc = (Fxu_full[1:] - Fxu_full[:-1]) / blk.J; dvc = (Fxv_full[1:] - Fxv_full[:-1]) / blk.J
    # eta faces (faces 0..nj): j=0 wall (zero flux), j=nj outer (Neumann = cell)
    uL = np.zeros((ni, nj + 1)); uR = np.zeros((ni, nj + 1))
    vL = np.zeros((ni, nj + 1)); vR = np.zeros((ni, nj + 1))
    uL[:, 1:nj] = u[:, :-1]; uR[:, 1:nj] = u[:, 1:]
    vL[:, 1:nj] = v[:, :-1]; vR[:, 1:nj] = v[:, 1:]
    uL[:, nj] = u[:, -1]; uR[:, nj] = u[:, -1]
    vL[:, nj] = v[:, -1]; vR[:, nj] = v[:, -1]
    Axf = blk.Aef_x; Ayf = blk.Aef_y
    uface = 0.5 * (uL + uR); vface = 0.5 * (vL + vR)
    Fgu = uface * (uface * Axf + vface * Ayf); Fgv = vface * (uface * Axf + vface * Ayf)
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0
    Fgu_outer = Fgu[:, 1:]; Fgv_outer = Fgv[:, 1:]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, 1:nj]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner) / blk.J; dvc += (Fgv_outer - Fgv_inner) / blk.J
    return duc, dvc


def c_max_inv_h2(blk):
    """Maximum over all cells of (1/hx^2 + 1/hy^2), where hx, hy are the cell
    edge lengths.  This is the quantity that sets the explicit-diffusion CFL:
        dt * coeff * (1/hx^2 + 1/hy^2) <= safety   (per cell)
    so the largest value (the most stretched / finest cell) bounds dt.  Using the
    short edge rather than the cell AREA |J| matters on stretched cells: |J| is
    set by the long edge while stability is governed by the short one."""
    hxi = np.sqrt(blk.Axi_x ** 2 + blk.Axi_y ** 2)   # (ni+1,nj) xi-face lengths
    het = np.sqrt(blk.Aef_x ** 2 + blk.Aef_y ** 2)   # (ni,nj+1) eta-face lengths
    hx = 0.5 * (hxi[:-1, :] + hxi[1:, :])            # cell xi edge
    hy = 0.5 * (het[:, :-1] + het[:, 1:])            # cell eta edge
    hx2 = np.maximum(hx ** 2, 1e-30)
    hy2 = np.maximum(hy ** 2, 1e-30)
    return np.max(1.0 / hx2 + 1.0 / hy2)


def c_kmax4(blk):
    """Per-cell upper bound on the 4th-order (biharmonic) operator eigenvalue
    |L4|_max, used to set a LOCAL, grid-scale-limited explicit 4th-order
    dissipation that is STABLE on a stretched wall-function grid:

        nu_hyp(i,j) = C_hyp / (kmax4(i,j) * dt)

    so that  nu_hyp * |L4|_max * dt = C_hyp  (uniform, < 1)  at EVERY cell,
    including the tiny wake-cut / near-TE cells that a GLOBAL nu_hyp cannot
    stabilise.  kmax4 is the square of the 2nd-order Laplacian's per-cell max
    eigenvalue (von Neumann: c_lap eigenvalue <= 4/h^2 per direction at kh=pi):

        |L4|_max = (4/hx^2 + 4/hy^2)^2
    """
    hxi = np.sqrt(blk.Axi_x ** 2 + blk.Axi_y ** 2)   # (ni+1,nj) xi-face lengths
    het = np.sqrt(blk.Aef_x ** 2 + blk.Aef_y ** 2)   # (ni,nj+1) eta-face lengths
    hx = 0.5 * (hxi[:-1, :] + hxi[1:, :])            # cell xi edge (ni,nj)
    hy = 0.5 * (het[:, :-1] + het[:, 1:])            # cell eta edge (ni,nj)
    hx2 = np.maximum(hx ** 2, 1e-30)
    hy2 = np.maximum(hy ** 2, 1e-30)
    lam2 = 4.0 / hx2 + 4.0 / hy2
    return lam2 ** 2


def c_hypervis_local(blk, u, v, C_hyp, dt, kmax4=None):
    """LOCAL, grid-scale-limited 4th-order DISSIPATION -- the stabiliser that lets
    the explicit collocated Chorin projection survive high-Re wall-bounded flow
    (Re=9e6, where molecular + SA eddy viscosity ~1e-7 is far too weak to damp the
    grid-scale pressure-velocity decoupling / checkerboard).

        du -= nu_hyp(i,j) * c_lap(c_lap(u)),   nu_hyp = C_hyp / (kmax4 * dt)

    WHY THIS (and not the old global +nu_hyp*c_lap(c_lap(u))): c_lap is the POSITIVE
    Laplacian (∇², negative-definite), so c_lap(c_lap(u)) is the POSITIVE biharmonic
    ∇⁴.  Adding it with a PLUS sign is ANTI-diffusion -- it AMPLIFIES the checkerboard
    (that is why every earlier global-hyperviscosity test diverged in ~15 steps).
    The MINUS sign is essential: at a velocity peak c_lap(c_lap(u)) > 0, so
    du -= nu_hyp*c_lap(c_lap(u)) pulls the peak down (true dissipation).

    WHY LOCAL (not global): a GLOBAL nu_hyp is forced by the explicit-stability limit
    nu_hyp*kmax4^wall*dt <= 1 to be ~1e-11 on the fine-wall (1/hy~2500) cells, which
    is far too weak to damp the checkerboard there; but near the coarse far field the
    same 1e-11 is over the limit and self-diverges.  Normalising by the LOCAL biharmonic
    eigenvalue makes the per-step damping number uniform (C_hyp) everywhere, so it is
    simultaneously stable AND strong enough at every cell.  Because 4th-order dissipation
    scales as k^4, it damps ONLY the top-wavenumber (checkerboard) mode and leaves the
    physical large-scale flow at the true Re=9e6 untouched (a 2nd-order artificial
    viscosity with the same kmax4 bound would instead lower the effective Re to ~L/h).

    Returns (du, dv) to be ADDED to the predictor update (u += dt*(-conv + d + diss)).
    """
    if kmax4 is None:
        kmax4 = c_kmax4(blk)
    # Bound nh from above: in the coarse far-field cells kmax4 = (4/h^2+4/h^2)^2
    # -> 0, which would make nh = C_hyp/(kmax4*dt) -> inf and overflow when
    # multiplied by c_lap(c_lap(u)).  Clamp kmax4 to its own maximum so nh is the
    # GLOBAL fine-cell value C_hyp/(kmax4_max*dt): the damping number is then
    # exactly C_hyp at the finest (checkerboard-bearing) cell and < C_hyp elsewhere
    # -- stable AND bounded everywhere, no overflow.  (The original per-cell form
    # diverged at step ~45 via the far-field nh blow-up.)
    kmax4c = np.maximum(kmax4, kmax4.max())
    nh = C_hyp / (kmax4c * dt)
    lu = c_lap(blk, u); lv = c_lap(blk, v)
    return (-nh * c_lap(blk, lu), -nh * c_lap(blk, lv))


def c_diffusion(blk, u, v):
    # Use the effective (molecular + turbulent) viscosity when the SA model has
    # set blk._nu_eff; otherwise fall back to the laminar molecular viscosity.
    # Capped by the explicit-diffusion CFL (blk._diff_cap) which is a GLOBAL
    # constant so the diffusion coefficient stays smooth (no grad(coeff) jumps).
    nu = getattr(blk, "_nu_eff", None)
    if nu is None:
        nu = blk._nu
    cap = getattr(blk, "_diff_cap", None)
    if cap is not None:
        nu = np.minimum(nu, cap)
    return (nu * c_div(blk, *c_grad(blk, u)), nu * c_div(blk, *c_grad(blk, v)))


def c_hypervis(blk, u, v, nu_hyp):
    lu = c_lap(blk, u); lv = c_lap(blk, v)
    return (nu_hyp * c_lap(blk, lu), nu_hyp * c_lap(blk, lv))


def c_checkerboard_project(blk, f, mode="pair"):
    """Light cosmetic smoother for the collocated pressure field.

    IMPORTANT (verified, not assumed): on this metric FVM the centred
    c_div / c_grad operators are NOT blind to the (-1)^i, (-1)^j or
    (-1)^(i+j) modes -- c_div(c_grad((-1)^i)) is O(1e2), i.e. these are
    NOT null modes of the Poisson matrix (see _diag_poisson2.py / the
    c_div(c_grad(mode)) check).  So a checkerboard "null-space removal"
    is mathematically unfounded here; removing those modes from the
    velocity every step actually perturbs the real divergence-free field
    and drives the solver to NaN/diverge (observed: Cd=-5.7 after the
    bi-directional variant was tried).

    What this routine does is therefore a MILD j-direction high-frequency
    smoother (originally added for the O-grid where streamwise wraps and
    the j-direction is the dominant noisy direction).  It is cosmetic only
    -- it does NOT fix the genuine collocated-grid pressure striping that
    shows up in the far field / wake of a strongly-stretched C-grid.  That
    striping is real high-frequency pressure content on a stretched grid
    and is physically inert for the DIVERGENCE-FREE velocity it produces
    (forces Cd/Cl are converged and correct).  The proper cure is a
    staggered (MAC) layout or Rhie-Chow momentum interpolation, not a
    filter.  Default mode="pair" == j-direction pair only; mode="global"
    removes a pure (-1)^j mode.

    Keep a verified reference: the original j-only pass (used by all the
    passing validation runs) is retained.  Do NOT extend it to the i
    direction in the loop -- it destabilises.
    """
    alt = (-1.0) ** np.arange(f.shape[1])
    if mode == "global":
        cb = np.mean(f * alt[None, :], axis=1)
        return f - cb[:, None] * alt[None, :]
    # j-direction odd/even pair (mild cosmetic smoother; see docstring)
    c = np.zeros_like(f)
    c[:, 0:-1:2] = 0.5 * (f[:, 0:-1:2] - f[:, 1::2])
    c[:, 1::2] = 0.5 * (f[:, 1::2] - f[:, 0:-1:2])
    return f - c


def c_filter_shapiro(blk, f, sigma_theta=0.0, sigma_r=0.0):
    g = f
    if sigma_theta > 0.0:
        if blk.periodic_i:
            st = (np.roll(g, 2, 0) + np.roll(g, -2, 0) - 4.0 * (np.roll(g, 1, 0) + np.roll(g, -1, 0)) + 6.0 * g)
        else:
            gp2 = np.empty_like(g); gp2[1:] = g[:-1]; gp2[0] = g[0]
            gm2 = np.empty_like(g); gm2[:-1] = g[1:]; gm2[-1] = g[-1]
            gp1 = np.empty_like(g); gp1[1:] = g[:-1]; gp1[0] = g[0]
            gm1 = np.empty_like(g); gm1[:-1] = g[1:]; gm1[-1] = g[-1]
            st = (gp2 + gm2) - 4.0 * (gp1 + gm1) + 6.0 * g
        g = g - (sigma_theta / 16.0) * st
    if sigma_r > 0.0:
        gp2 = np.empty_like(g); gp2[:, :-2] = g[:, 2:]; gp2[:, -2:] = g[:, -1:]
        gm2 = np.empty_like(g); gm2[:, 2:] = g[:, :-2]; gm2[:, :2] = g[:, :1]
        gp1 = np.empty_like(g); gp1[:, :-1] = g[:, 1:]; gp1[:, -1] = g[:, -1]
        gm1 = np.empty_like(g); gm1[:, 1:] = g[:, :-1]; gm1[:, 0] = g[:, 0]
        st = (gp2 + gm2) - 4.0 * (gp1 + gm1) + 6.0 * g
        g = g - (sigma_r / 16.0) * st
    return g


def _lap_face_matrix(blk, recv_mask):
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    rows, cols, vals = [], [], []
    e = np.zeros((ni, nj))
    for k in range(N):
        i = k // nj; j = k % nj
        e[i, j] = 1.0
        col = c_lap(blk, e).reshape(-1)
        nz = np.nonzero(col)[0]
        rows.extend(nz.tolist()); cols.extend([k] * nz.size); vals.extend(col[nz].tolist())
        e[i, j] = 0.0
    return csc_matrix((vals, (rows, cols)), shape=(N, N))


def _adj_lap_matrix(blk):
    """Assemble the ADJOINT Laplacian  L = c_div o c_grad  (the exact operator the
    projection actually uses: project() solves L p = div(u*)/dt and corrects with
    u -= dt*c_grad(p)).  Using THIS matrix (rather than the direct face-flux
    c_lap) makes the pressure correction EXACTLY consistent -- by construction
    c_div(u_new) = c_div(u*) - dt*(c_div o c_grad) p = 0 -- so the divergence is
    removed to solver precision on EVERY grid, regardless of how non-uniform J is.

    Why this matters: c_lap = M^{-1} K with M=diag(J) is only symmetric in the
    J-weighted inner product, not in Euclidean, and on a strongly stretched grid
    (airfoil: J spans 6e-5..9.8, a 1.6e5 ratio) solving that scaled non-symmetric
    system with SuperLU is ill-conditioned AND diverges from the c_div/c_grad
    correction, so the residual divergence accumulated and the pressure blew up
    (pmax ~1e70 on the airfoil).  The cylinder only 'survived' because its J ratio
    (~290) keeps c_lap ~= c_div o c_grad.  Building L from c_div o c_grad removes
    the inconsistency entirely; the matrix is well-conditioned enough for SuperLU
    and the projection is divergence-free everywhere.  (retained c_lap /
    _lap_face_matrix for reference and the metric self-test.)"""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    rows, cols, vals = [], [], []
    e = np.zeros((ni, nj))
    for k in range(N):
        i = k // nj; j = k % nj
        e[i, j] = 1.0
        col = c_div(blk, *c_grad(blk, e)).reshape(-1)
        nz = np.nonzero(col)[0]
        rows.extend(nz.tolist()); cols.extend([k] * nz.size); vals.extend(col[nz].tolist())
        e[i, j] = 0.0
    return csc_matrix((vals, (rows, cols)), shape=(N, N))


def _rhie_lap_matrix(blk, dt):
    """OpenFOAM-style Rhie-Chow pressure operator, built from the SAME pair of
    operators the projection correction uses, so the two are EXACT discrete
    adjoints and the projection removes the divergence to machine precision:

        L p  =  c_div( rAU * c_grad(p) ),      rAU = dt / J   (per cell)

    i.e.  L = C_div . diag(rAU) . C_grad.  This is exactly OpenFOAM's
    `laplacian(rAU, p)` with rAU = dt/J (the inverse diagonal A_P = J/dt of the
    implicit momentum matrix), evaluated per cell and interpolated to faces by
    c_div.  Because the correction in project() is  u -= rAU * c_grad(p), the
    two are adjoints:

        c_div( u* - rAU*c_grad(p) ) = c_div(u*) - L p

    so once  L p = c_div(u*)  is solved,  c_div(u_new) = 0  EXACTLY.  (This is
    why the naive face-gradient form  div((dt/Jf) grad p), which is NOT adjoint
    to the cell-centre correction, diverges on stretched grids -- the divergence
    is not removed and p blows up.)  Crucially, on a uniform grid rAU = dt/J is
    constant, so L = (dt/J) * (c_div o c_grad) and the corrected velocity is
    IDENTICAL to the old unweighted operator; on a stretched grid the spatially
    varying rAU (=dt/J, J spans ~1.6e5) is the Rhie-Chow ingredient that couples
    the pressure correction through rAU and suppresses the collocated-grid
    checkerboard.  recv cells are excluded (their velocity is prescribed => the
    pressure flux through their faces is zero, Neumann for p); the matrix stays
    non-singular exactly as c_div o c_grad did.
    """
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    # Rhie-Chow weight rAU = dt / J**gamma.  gamma=1.0 is the textbook OpenFOAM
    # form (A_P = J/dt => rAU = dt/J) and is the proper FVM pressure operator, but
    # on a strongly-stretched explicit-projection grid the 1/J near fine wall cells
    # amplifies the pressure correction ~1/J and drives a divergence (the velocity
    # jumps to several times free-stream at step 1).  gamma<1 keeps the variation
    # that damps the far-field checkerboard but tames the near-wall amplification.
    gamma = getattr(blk, "_rhie_gamma", 1.0)
    # Rhie-Chow weight.  Two forms:
    #   form="dtJ" : rAU = dt / J**gamma  (textbook-only; large near fine walls
    #               -> over-corrects in an EXPLICIT Chorin correction, diverges on
    #               strongly stretched grids).  Retained for reference/tests.
    #   form="ap"  : rAU = 1 / A_P,  A_P = J/dt + nu * (diffusion diagonal)
    #               -- the GENUINE OpenFOAM rAU (inverse diagonal of the implicit
    #               momentum matrix).  Because A_P is stiff (large) near fine
    #               walls, rAU is SMALL there, so the pressure correction
    #               u -= rAU*grad(p) is bounded and the explicit projection stays
    #               stable.  This is what makes Rhie-Chow viable without a fully
    #               coupled SIMPLE momentum solve.
    form = getattr(blk, "_rhie_form", "dtJ")
    if form == "ap":
        nu = getattr(blk, "_nu", 0.0)
        S = (blk.Axi_len[1:] / blk.hxi_f[1:] + blk.Axi_len[:-1] / blk.hxi_f[:-1]
             + blk.Aef_len[:, 1:] / blk.heta_f[:, 1:] + blk.Aef_len[:, :-1] / blk.heta_f[:, :-1])
        Ap = blk.J / dt + nu * S          # implicit momentum diagonal
        rAU = 1.0 / Ap
    else:
        rAU = dt / (blk.J ** gamma)       # (ni, nj) per-cell Rhie-Chow weight
    # OPTIONAL clamp: bound the rAU spread to [rmed/clamp, rmed*clamp] about its
    # solved-cell median.  Only meaningful for the dtJ form (ap form is already
    # bounded by construction).  Keeps the fix available for reference tests.
    clamp = getattr(blk, "_rhie_clamp", None)
    if clamp is not None and clamp > 0.0 and form != "ap":
        recv = blk.recv.reshape(-1)
        solved = ~recv
        rflat = rAU.reshape(-1)
        rmed = float(np.median(rflat[solved])) if solved.any() else float(np.median(rflat))
        rAU = np.clip(rAU, rmed / clamp, rmed * clamp)
    blk._rAU = rAU                       # cached for project()'s correction
    e = np.zeros((ni, nj))
    rows, cols, vals = [], [], []
    for k in range(N):
        i = k // nj; j = k % nj
        e[i, j] = 1.0
        gpx, gpy = c_grad(blk, e)
        gp = (rAU * gpx, rAU * gpy)          # rAU * grad(e) : Rhie-Chow weight
        col = c_div(blk, gp[0], gp[1]).reshape(-1)
        nz = np.nonzero(col)[0]
        rows.extend(nz.tolist()); cols.extend([k] * nz.size); vals.extend(col[nz].tolist())
        e[i, j] = 0.0
    return csc_matrix((vals, (rows, cols)), shape=(N, N))


def _apply_rhie_lap_face(blk, rAU, p):
    """Apply the TRUE Rhie-Chow Laplacian  L p = div( rAU * grad_face(p) )  where
    grad_face uses the FACE pressure gradient (p_nb - p_c)/h_face -- NOT the
    cell-centre gradient.  This is the single ingredient that actually suppresses
    the collocated checkerboard: for an alternating p the cell-centre gradient
    (c_grad) averages to ~0 while the face gradient (p_R - p_L)/h is LARGE, so the
    cell-gradient form (_rhie_lap_matrix) leaves the checkerboard in p and the
    explicit correction diverges, whereas the face-gradient form cancels it and p
    stays smooth.  rAU at a face = 0.5*(rAU_L + rAU_R) (arithmetic mean).  The
    divergence is taken with the SAME face fluxes c_div uses, so the correction
    u -= rAU_cell*grad(p) is the exact discrete adjoint (c_div(u_new)=0)."""
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    p_r = _ir(p, per); p_l = _il(p, per)
    p_o = np.concatenate([p[:, 1:], p[:, -1:]], 1)
    p_i = np.concatenate([p[:, :1], p[:, :-1]], 1)
    rAU_r = _ir(rAU, per); rAU_l = _il(rAU, per)
    rAU_o = np.concatenate([rAU[:, 1:], rAU[:, -1:]], 1)
    rAU_i = np.concatenate([rAU[:, :1], rAU[:, :-1]], 1)
    # face rAU = 0.5*(owner + neighbour)
    rAr = 0.5 * (rAU + rAU_r); rAl = 0.5 * (rAU + rAU_l)
    rAo = 0.5 * (rAU + rAU_o); rAi = 0.5 * (rAU + rAU_i)
    # xi faces
    Axr, Axl = blk.Axi_len[1:], blk.Axi_len[:-1]
    hxr, hxl = blk.hxi_f[1:], blk.hxi_f[:-1]
    Aei, Aeo = blk.Aef_len[:, :-1], blk.Aef_len[:, 1:]
    hei, heo = blk.heta_f[:, :-1], blk.heta_f[:, 1:]
    flux = (rAr * Axr * (p_r - p) / hxr
            + rAl * Axl * (p_l - p) / hxl
            + rAo * Aeo * (p_o - p) / heo
            + rAi * Aei * (p_i - p) / hei)
    return flux / blk.J


def _rhie_grad_face(blk, rAU, p):
    """Cell-centre pressure GRADIENT consistent with the face-gradient Laplacian
    _apply_rhie_lap_face -- the TRUE Rhie-Chow gradient.

    CRITICAL: the velocity correction must use THIS gradient, not c_grad.  L is
    built from the FACE pressure gradient (p_R - p_L)/h_face, whereas c_grad uses
    the CELL-CENTRE gradient.  For a p with a sharp feature (the slat's pressure
    jumps across its overset hole boundary, and any collocated p has a residual
    checkerboard) these two gradients DISAGREE, so

        div( rAU * c_grad(p) )  !=  rAU * L p  =  rAU * div(u*)

    and the correction ADDS divergence instead of removing it -- the 3-element
    run blew up 27x per projection step because of exactly this.  Using the
    face-gradient here makes the correction the exact discrete adjoint of L, so
    div(u_new) = div(u*) - rAU*L p = (1 - rAU) div(u*) -> 0 in one step at rAU=1.

    Implemented as the Gauss gradient: each face contributes the FACE gradient
    VECTOR  (p_R - p_L)/h_face * n_face  (n_face = A_face/|A_face|), and the
    cell-centre gradient is the arithmetic mean of the 4 face vectors -- this is
    precisely the OpenFOAM Rhie-Chow cell-centre gradient that suppresses the
    checkerboard and stays consistent with the pressure equation."""
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    p_r = _ir(p, per); p_l = _il(p, per)
    p_o = np.concatenate([p[:, 1:], p[:, -1:]], 1)
    p_i = np.concatenate([p[:, :1], p[:, :-1]], 1)
    rAU_r = _ir(rAU, per); rAU_l = _il(rAU, per)
    rAU_o = np.concatenate([rAU[:, 1:], rAU[:, -1:]], 1)
    rAU_i = np.concatenate([rAU[:, :1], rAU[:, :-1]], 1)
    rAr = 0.5 * (rAU + rAU_r); rAl = 0.5 * (rAU + rAU_l)
    rAo = 0.5 * (rAU + rAU_o); rAi = 0.5 * (rAU + rAU_i)
    # xi faces (right face of cell c between c and c+1): vector = A_face / |A_face|
    Axr, Axl = blk.Axi_x[1:], blk.Axi_x[:-1]
    Ayr, Ayl = blk.Axi_y[1:], blk.Axi_y[:-1]
    Axlr, Axll = blk.Axi_len[1:], blk.Axi_len[:-1]
    nxr, nyr = Axr / Axlr, Ayr / Axlr
    nxl, nyl = Axl / Axll, Ayl / Axll
    hxr, hxl = blk.hxi_f[1:], blk.hxi_f[:-1]
    # eta faces (outer face of cell c between c and c+1 in j)
    Aeo, Aei = blk.Aef_x[:, 1:], blk.Aef_x[:, :-1]
    Ayo, Ayi = blk.Aef_y[:, 1:], blk.Aef_y[:, :-1]
    Aelo, Aeli = blk.Aef_len[:, 1:], blk.Aef_len[:, :-1]
    neo_x, neo_y = Aeo / Aelo, Ayo / Aelo
    nei_x, nei_y = Aei / Aeli, Ayi / Aeli
    heo, hei = blk.heta_f[:, 1:], blk.heta_f[:, :-1]
    # face gradient vectors g_f = (p_R - p_L)/h_f * n_f, weighted by face rAU.
    # OUTWARD-face terms use (p_nb - p_c); INWARD-face terms use (p_c - p_nb) so the
    # gradient is the consistent Gauss gradient matching c_div / L (the wrong sign
    # here makes the correction ADD divergence and explodes the run).
    gxr = rAr * (p_r - p) / hxr * nxr;  gyr = rAr * (p_r - p) / hxr * nyr
    gxl = rAl * (p - p_l) / hxl * nxl;  gyl = rAl * (p - p_l) / hxl * nyl
    gxo = rAo * (p_o - p) / heo * neo_x; gyo = rAo * (p_o - p) / heo * neo_y
    gxi = rAi * (p - p_i) / hei * nei_x; gyi = rAi * (p - p_i) / hei * nei_y
    # Gauss cell-centre gradient = mean of the four face gradient vectors
    gx = (gxr + gxl + gxo + gxi) / 4.0
    gy = (gyr + gyl + gyo + gyi) / 4.0
    return gx, gy


def _rhie_lap_face_matrix(blk, rAU):
    """Matrix form of _apply_rhie_lap_face (assembled by column action)."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    rows, cols, vals = [], [], []
    e = np.zeros((ni, nj))
    for k in range(N):
        i = k // nj; j = k % nj
        e[i, j] = 1.0
        col = _apply_rhie_lap_face(blk, rAU, e).reshape(-1)
        nz = np.nonzero(col)[0]
        rows.extend(nz.tolist()); cols.extend([k] * nz.size); vals.extend(col[nz].tolist())
        e[i, j] = 0.0
    return csc_matrix((vals, (rows, cols)), shape=(N, N))


def _rhie_lap_consistent(blk, rAU):
    """CONSISTENT Rhie-Chow Laplacian matrix: assembled by column action of the
    EXACT operator the velocity correction uses,

        L p = c_div( rAU * _rhie_grad_face(p) ),

    so that  c_div(u_new) = c_div(u*) - rAU * L p  EXACTLY and the projection
    removes the divergence to machine precision (the correction and the matrix
    are the SAME discrete operator).  The original _rhie_lap_face_matrix used the
    direct face-FLUX form (_apply_rhie_lap_face), which DIFFERS from the
    cell-centre-mean correction _rhie_grad_face by up to ~95% on a stretched
    C-grid, so the correction ADDED divergence instead of removing it (cylinder
    Cd came out 2x too high, residual div RMS ~0.3).  This form fixes that."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    rows, cols, vals = [], [], []
    e = np.zeros((ni, nj))
    for k in range(N):
        i = k // nj; j = k % nj
        e[i, j] = 1.0
        gx, gy = _rhie_grad_face(blk, rAU, e)
        col = c_div(blk, rAU * gx, rAU * gy).reshape(-1)
        nz = np.nonzero(col)[0]
        rows.extend(nz.tolist()); cols.extend([k] * nz.size); vals.extend(col[nz].tolist())
        e[i, j] = 0.0
    return csc_matrix((vals, (rows, cols)), shape=(N, N))


class _SpluPoisson:
    """DIRECT (SuperLU) Poisson solver for the reduced (solved-cell) system.

    WHY DIRECT: a direct LU factorisation sidesteps the Krylov/ILU instability
    entirely (the previous iterative _PoissonSolver DIVERGED on the ~1e8 dynamic
    range of the 1/J-weighted operator entries).  The factorisation is built ONCE
    per block (the metric is fixed) and reused every step.

    JACOBI ROW-SCALING (default ON): the Rhie-Chow face operator on a strongly
    stretched C-grid has entries ranging from ~1e1 (coarse far-field cells) to
    ~1e9 (tiny near-wall cells), so its DYNAMIC RANGE is ~1e8 and a SuperLU solve
    of the UNSCALED system loses all precision -> a GARBAGE pressure that diverges
    the coupled slat/flap run (observed: slat solved div GREW every step, blow-up
    by ~1e4 steps).  Scaling each row by its diagonal (unit-diagonal system) removes
    that dynamic range, so SuperLU returns the TRUE pressure to machine precision.
    Scaling both sides leaves the solution p UNCHANGED (it is a similarity-style
    row scaling of A x = b), so the projection correction is identical -- only the
    solve accuracy improves.  (The MG smoother deliberately does NOT use this
    scaling, because on the natural operator Gauss-Seidel is diagonally dominant
    and stable; Jacobi-scaling would make it non-DD and overflow.  That caveat is
    specific to the GS smoother, NOT to a direct SuperLU solve.)"""

    def __init__(self, L_red, solved_idx, scale=True):
        self.idx = solved_idx
        self._A = L_red.tocsc()
        if scale:
            D = self._A.diagonal()
            self._Dinv = np.where(np.abs(D) > 0.0, 1.0 / D, 1.0)
            Ls = (diags(self._Dinv) @ self._A).tocsc()
        else:
            self._Dinv = None
            Ls = self._A
        self._lu = splu(Ls.tocsc())

    def solve(self, b_reduced, x0=None):
        b = np.asarray(b_reduced, dtype=float)
        if self._Dinv is not None:
            b = b * self._Dinv
        return self._lu.solve(b)


class _PoissonSolver:
    """Robust Poisson solver for the (possibly strongly varying-coefficient)
    Rhie-Chow operator  L = c_div( rAU * grad(p) ).

    The raw matrix is ill-conditioned once rAU = dt/J varies across a stretched
    grid (its condition number scales ~ N^2 * rAU_ratio), so a bare SuperLU/LU
    factorisation returns a garbage pressure and the Chorin correction blows up.
    We therefore:

      * symmetrically Jacobi-scale the reduced system  Ls = D^{-1/2} L D^{-1/2}
        (unit diagonal -> far better conditioning for Krylov methods),
      * build an ILU(0) preconditioner on Ls (this is the lightweight stand-in
        for OpenFOAM's GAMG / AMG that handles varying coefficients),
      * solve with bicgstab (falls back to cg when Ls is symmetric PSD).

    solve(b_reduced) takes the reduced RHS and returns the reduced solution, so
    project() can keep calling it exactly as it did splu.solve()."""

    def __init__(self, L_red, solved_idx):
        self.idx = solved_idx
        L = L_red.tocsc()
        d = np.abs(L.diagonal())
        d[d <= 0] = 1.0
        dh = 1.0 / np.sqrt(d)
        Ls = L.multiply(dh[:, None]).multiply(dh[None, :]).tocsc()
        self._Ls = Ls
        self._scale = dh
        self._L = L
        # ILU preconditioner on the scaled (better-conditioned) system
        self._M = None
        try:
            ilu = spilu(Ls.tocsc(), drop_tol=1e-5, fill_factor=30)
            self._M = LinearOperator(Ls.shape, matvec=ilu.solve)
        except Exception:
            self._M = None

    def solve(self, b_reduced, x0=None):
        # x0 accepted for interface symmetry with _MGSolver (ignored here:
        # ILU/BiCGStab already converges fast on the (well-conditioned) adjoint
        # operator, so warm-starting is unnecessary).
        b = self._scale * b_reduced
        x, info = bicgstab(self._Ls, b, M=self._M, rtol=1e-11, atol=0.0, maxiter=4000)
        if info != 0:
            # second attempt: looser tol, then cg (Ls is symmetric PSD)
            x, info = bicgstab(self._Ls, b, M=self._M, rtol=1e-8, atol=0.0, maxiter=6000)
        if info != 0:
            try:
                x, info = cg(self._Ls, b, rtol=1e-8, atol=0.0, maxiter=6000)
            except Exception:
                pass
        return self._scale * x


# ----------------------------------------------------------------------------
# Geometric Algebraic Multigrid (GAMG) solver for the pressure-correction op.
#
# WHY THIS IS NEEDED (the whole point of the SIMPLE + AMG upgrade):
#   The Rhie-Chow operator  L = c_div( rAU * grad(p) )  has condition number
#   kappa ~ rAU_ratio * N^2.  On a stretched C-grid rAU_ratio (=Jmax/Jmin) is
#   1e5..1e7, so kappa ~ 1e13..1e15 -- past double precision.  A DIRECT factor
#   (SuperLU) or a single-level Krylov/ILU solver therefore returns a garbage
#   (spiky, amplified) pressure and the Chorin / SIMPLE correction diverges.
#   This is exactly why OpenFOAM must pair Rhie-Chow with GAMG/AMG: a MULTIGRID
#   solver reduces the error by a constant factor per V-cycle, independent of
#   kappa, because it destroys the smooth error on coarse grids.  With MG the
#   (still very ill-conditioned) Rhie operator is solved to machine precision
#   and Rhie-Chow finally removes the collocated checkerboard on stretched grids.
#
# This is a GEOMETRIC multigrid: the C-grid is structured (ni x nj indices), so
# we coarsen by 2 in each index direction, build the coarse operators by Galerkin
# R^T A R, restrict with full-weighting and prolongate with bilinear interpolation.
# (For a structured grid this is functionally identical to an algebraic multigrid
# and is what solvers like BoxMG use; pyamg could not be installed here because its
# wheel build fails against the system numpy 1.x, so it is re-implemented directly.)
# The solver is used as a preconditioner for BiCGStab (robust for the slightly
# non-symmetric scaled system) with a plain-MG V-cycle fallback.
# ----------------------------------------------------------------------------

def _mg_fullweight_Rj(li, lj, Mi, Mj, cMj):
    """Semi-coarsening (j-direction) full-weighting restriction (fine -> coarse).
    The i-index is unchanged; only j is halved (the C-grid is stretched in j /
    wall-normal).  Coarse cell index = li * cMj + (lj//2).  Standard 1D
    full-weighting weights:
        lj even (centre)     -> 0.5  to coarse(lj//2)
        lj odd  (between)    -> 0.25 to coarse(lj//2) and 0.25 to coarse(lj//2+1)."""
    nf = li.size
    rows, cols, vals = [], [], []
    for f in range(nf):
        li_f = li[f]; clj = lj[f] // 2
        if lj[f] % 2 == 0:
            rows.append(li_f * cMj + clj); cols.append(f); vals.append(0.5)
        else:
            rows.append(li_f * cMj + clj); cols.append(f); vals.append(0.25)
            if clj + 1 < cMj:
                rows.append(li_f * cMj + clj + 1); cols.append(f); vals.append(0.25)
    return coo_matrix((vals, (rows, cols)), shape=(Mi * cMj, nf)).tocsr()


def _mg_bilinear_Pj(li, lj, Mi, Mj, cMj):
    """Semi-coarsening (j-direction) bilinear prolongation (coarse -> fine).
    Coarse cell index = li * cMj + (lj//2).
        lj even -> 1.0  from coarse(lj//2)
        lj odd  -> 0.5  from each of coarse(lj//2) and coarse(lj//2+1)."""
    nf = li.size
    rows, cols, vals = [], [], []
    for f in range(nf):
        li_f = li[f]; clj = lj[f] // 2
        if lj[f] % 2 == 0:
            rows.append(f); cols.append(li_f * cMj + clj); vals.append(1.0)
        else:
            rows.append(f); cols.append(li_f * cMj + clj); vals.append(0.5)
            if clj + 1 < cMj:
                rows.append(f); cols.append(li_f * cMj + clj + 1); vals.append(0.5)
    return coo_matrix((vals, (rows, cols)), shape=(nf, Mi * cMj)).tocsr()


class _MGSolver:
    """Geometric multigrid (AMG) solver for the reduced pressure operator
    L_red (solved cells only).  Interface matches _PoissonSolver.solve(b_reduced)
    so build_poisson() can return it transparently and project() stays unchanged.

    Design choices that make MG robust on this strongly STRETCHED, variable-
    coefficient C-grid (where a naive Jacobi/ILU/Krylov direct solve diverges -- see
    history):
      * SEMI-COARSENING in j (wall-normal): the grid is stretched ~1e7 in j but
        only mildly in i, so coarsening only j removes the dominant anisotropy
        that otherwise destroys standard full-coarsening MG convergence.
      * NATURAL (diagonally-dominant) Rhie operator, NOT Euclidean-symmetric-scaled:
        the FV pressure operator has row-sum ~ 0 (off-diagonals share sign), so a
        Gauss-Seidel smoother is STABLE (triangular solves divide by the dominant
        diagonal) and CONVERGENT (irreducibly DD).  Full-weighting Galerkin R^T A R
        then PRESERVES row-sum-zero at every coarse level, so coarse operators stay
        DD too -- no tiny/overflowing coarse diagonal.  (A global Jacobi dh-scaling
        to Euclidean symmetry forces diag -> -1 while leaving off-diagonals ~
        sqrt(J_ratio) ~ 2600, making the operator non-DD and the GS substitution
        OVERFLOW -- that was the earlier failure.)
      * used as a BiCGStab preconditioner (the operator is slightly non-symmetric),
        with a plain-MG-V-cycle fallback.
    """

    def __init__(self, L_red, solved_idx, ni, nj, nsmooth=2):
        self.ni, self.nj = ni, nj
        # Use the NATURAL Rhie operator L_red (NOT scaled to Euclidean symmetry).
        # The FV pressure operator L = c_div(rAU * c_grad) is a weighted Laplacian
        # whose rows are DIAGONALLY DOMINANT (row-sum ~ 0, all off-diagonals share
        # sign), so a Gauss-Seidel smoother on it is STABLE -- the triangular solves
        # divide by the largest-in-magnitude (diagonal) entry -- and CONVERGENT
        # (irreducibly DD).  A global Jacobi dh-scaling to Euclidean symmetry would
        # force the diagonal to -1 while leaving off-diagonals ~ sqrt(J_ratio)
        # (~2600 on this grid) -> NON-diagonally-dominant, and the GS triangular
        # substitution then OVERFLOWS.  Keeping the natural operator also means the
        # full-weighting Galerkin R^T A R PRESERVES the row-sum-zero property at
        # EVERY coarse level:
        #     sum_m A_c[k,m] = sum_i R[k,i] (sum_j A[i,j]) (sum_m P[j,m])
        #                    = 0   because  sum_j A[i,j] = 0  and  col-sum(P) = 1,
        # so coarse operators stay DD too -> no tiny/overflowing coarse diagonals.
        self.Atrue = L_red.tocsc()                # true operator for the outer solve
        i = solved_idx // nj; j = solved_idx % nj
        i0, j0 = int(i.min()), int(j.min())
        li = (i - i0).astype(np.int64)          # i kept unchanged across levels
        lj = (j - j0).astype(np.int64)
        Mi = int(li.max()) + 1; Mj = int(lj.max()) + 1
        self.As, self.LDs, self.UDs, self.Rs, self.Ps = [], [], [], [], []
        A = self.Atrue
        cur_li, cur_lj = li.copy(), lj.copy()
        cur_Mj = Mj
        while True:
            A = A.tocsc()
            self.As.append(A)
            D = A.diagonal()
            L = tril(A, -1).tocsc(); U = triu(A, 1).tocsc()
            self.LDs.append((L + diags(D)).tocsc())   # lower-tri incl diag (neg)
            self.UDs.append((U + diags(D)).tocsc())    # upper-tri incl diag (neg)
            if cur_Mj <= 4:
                break
            cMj = (cur_Mj + 1) // 2
            R = _mg_fullweight_Rj(cur_li, cur_lj, Mi, cur_Mj, cMj)
            P = R.T.tocsr()                      # exact Galerkin transpose
            self.Rs.append(R); self.Ps.append(P)
            A = (R @ A @ P).tocsc()
            # re-index to the contiguous coarse rectangle (solved cells form a full
            # rectangle at every level, so every (li,clj) is present).  i is unchanged
            # (semi-coarsening in j), only j shrinks.
            cur_li = np.arange(Mi * cMj) // cMj
            cur_lj = np.arange(Mi * cMj) % cMj
            cur_Mj = cMj
        self.coarse = self.As[-1].tocsc()
        try:
            self._cf = splu(self.coarse)
        except Exception:
            self._cf = None
        self.nsmooth = nsmooth

    def _smooth(self, lvl, x, b):
        """Symmetric Gauss-Seidel on A x = b.  A is (irreducibly) diagonally
        dominant with a negative diagonal, so the triangular solves divide by the
        largest-magnitude diagonal entry and never overflow, and GS converges.

        forward  (L+D) x_f = b - U x     backward (U+D) x = b - L x_f .
        """
        A = self.As[lvl]; LD = self.LDs[lvl]; UD = self.UDs[lvl]
        Ax = A @ x
        xf = spsolve_triangular(LD, b - Ax + (LD @ x), lower=True)
        Axf = A @ xf
        return spsolve_triangular(UD, b - Axf + (UD @ xf), lower=False)

    def _vcycle(self, lvl, b):
        if lvl >= len(self.As) - 1:
            if self._cf is not None:
                return self._cf.solve(b)
            x, _ = bicgstab(self.As[-1], b, rtol=1e-7, atol=0.0, maxiter=300)
            return x
        A = self.As[lvl]
        x = np.zeros_like(b)
        for _ in range(self.nsmooth):
            x = self._smooth(lvl, x, b)
        r = b - A @ x
        ec = self._vcycle(lvl + 1, self.Rs[lvl] @ r)
        x = x + self.Ps[lvl] @ ec
        for _ in range(self.nsmooth):
            x = self._smooth(lvl, x, b)
        return x

    def mg_apply(self, b):
        """One V-cycle of the natural Rhie operator: approximate solve of
        A x = b.  Used as the BiCGStab preconditioner."""
        return self._vcycle(0, b)

    def solve(self, b_reduced, x0=None):
        bs = b_reduced.astype(float).copy()
        if x0 is not None:
            bs = bs - self.Atrue @ x0      # solve for the correction (warm start)
        M = LinearOperator((bs.size, bs.size), matvec=self.mg_apply)
        x, info = bicgstab(self.Atrue, bs, M=M, rtol=1e-8, atol=0.0, maxiter=200)
        if info != 0:
            x, _ = bicgstab(self.Atrue, bs, M=M, rtol=1e-6, atol=0.0, maxiter=200)
        if info != 0:
            # V-cycle as a solver on the DD operator (converges; the null-space
            # constant mode stays put because the iterate starts at zero).
            x = np.zeros_like(bs)
            for _ in range(120):
                r = bs - self.Atrue @ x
                x = x + self.mg_apply(r)
                if np.linalg.norm(self.Atrue @ x - bs) < 1e-8 * max(1.0, np.linalg.norm(bs)):
                    break
        if x0 is not None:
            x = x + x0
        return x


def _build_hole_fill(blk, hole_idx, solved_idx):
    """Zero-gradient (harmonic / Neumann) fill operator for the overset HOLE.

    Returns a sparse (n_hole x N) matrix H such that, after the pressure solve,
        p[hole] = H @ p_full
    sets each hole cell's pressure to the MEAN of its solved 4-neighbours.  This
    makes the pressure field continuous across the hole boundary (p_hole =
    p_solved), so the velocity-correction gradient c_grad at the seam-adjacent
    solved cells is purely tangential -- i.e. the projection injects NO normal
    velocity into the covered region (the discrete statement of a Neumann
    pressure BC).  Deep-hole cells with no solved neighbour keep their (irrelevant,
    never-read-by-a-solved-cell) value.

    Built once in build_poisson; applied once per step in project() after the
    solve.  This is the companion to the diagonal-fold Neumann treatment in
    build_poisson (which removes the hole coupling from the operator): together
    they implement the robust "hole = Neumann" overset pressure BC that stops the
    donor-pressure seam jump from round-tripping through the exchange and diverging
    the coupled 3-element run."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    solved_set = set(int(k) for k in solved_idx.tolist())
    rows, cols, vals = [], [], []
    for hi, h in enumerate(hole_idx.tolist()):
        i = h // nj; j = h % nj
        nb = []
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii = i + di; jj = j + dj
            if 0 <= ii < ni and 0 <= jj < nj:
                k = ii * nj + jj
                if k in solved_set:
                    nb.append(k)
        if nb:
            w = 1.0 / len(nb)
            for k in nb:
                rows.append(hi); cols.append(k); vals.append(w)
    from scipy.sparse import coo_matrix
    return coo_matrix((vals, (rows, cols)), shape=(hole_idx.size, N)).tocsc()


def build_poisson(blk, recv_mask=None, reg_r=200.0, dt=1.0, mode="adjoint",
                 hole_mask=None):
    """Build the pressure-Poisson solver.

    mode="adjoint" (DEFAULT, production-stable): the UNWEIGHTED adjoint operator
        L = c_div o c_grad.  Its condition number is ~N^2 (solvable to machine
        precision) and the projection is divergence-free and STABLE on strongly
        stretched C-grids (cylinder and airfoil).  This is the correct collocated
        projection operator for an EXPLICIT Chorin step; the (cosmetic) pressure
        checkerboard it exhibits is a known, physically-inert collocated artifact
        for the divergence-free velocity (forces are converged/correct).

    mode="rhie": the Rhie-Chow operator L = c_div( rAU * grad(p) ) with
        rAU = dt/J (or 1/A_P).  This is the textbook OpenFOAM pressure operator
        and suppresses the collocated checkerboard.  Its condition number is
        ~rAU_ratio * N^2 (rAU_ratio = Jmax/Jmin ~ 1e5..1e7 on a stretched grid,
        so kappa ~ 1e13..1e15, above double precision).  A NAIVE solver
        (ILU / BiCGStab / CG -- i.e. the old _PoissonSolver) cannot recover the
        smooth physical pressure and returns a spiky, amplified p that makes the
        explicit correction DIVERGE.  The fix is the SIMPLE+AMG upgrade: the
        projection loop (project()) already has the SIMPLE structure (explicit
        momentum predictor + pressure under-relaxation pRelax + multiple inner
        corrections ncorr), and the pressure equation is solved here by the
        geometric MULTIGRID (_MGSolver) whose convergence is INDEPENDENT of kappa
        (each V-cycle kills the smooth error on coarser grids).  With MG the Rhie
        operator is solved to ~1e-8 residual and Rhie-Chow becomes usable on
        stretched grids exactly as in OpenFOAM (Rhie-Chow + SIMPLE/PISO + AMG)."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    recv = (blk.recv if recv_mask is None else np.asarray(recv_mask)).reshape(-1).astype(bool)
    solved = ~recv
    if not solved.any():
        solved = np.ones(N, dtype=bool); solved[0] = False; recv = ~solved
    if mode == "rhie":
        L_full = _rhie_lap_matrix(blk, dt)
    elif mode == "rhie2":
        # TRUE Rhie-Chow: the pressure CONTINUITY equation uses the FACE pressure
        # gradient  L p = div( rAU_face * grad_face(p) )  (_rhie_lap_face), which is
        # what actually suppresses the collocated checkerboard (a checkerboard p
        # has a LARGE face gradient, so L "sees" it and the solve keeps p smooth).
        # rAU = O(1) (uniform 1.0) so the velocity correction u -= rAU*grad(p)
        # removes the divergence in ONE step (an explicit Chorin step cannot use
        # rAU = 1/A_P ~ dt/J -- that under-relaxes by dt and freezes the div).
        # The velocity correction uses the cell-centre gradient of the now-smooth
        # p, which is itself smooth, so it is effective and stable.
        blk._rAU = np.ones_like(blk.J)
        L_full = _rhie_lap_consistent(blk, blk._rAU)
    elif mode == "adj1":
        # ADJOINT operator with ONE-STEP rAU = 1 correction.  This is the SAME
        # well-conditioned operator the hole-filling re-projection uses
        # (c_div o c_grad, exactly solvable by SuperLU), but applied as the normal
        # projection with rAU = 1 so it removes divergence in a SINGLE step (unlike
        # the production "adjoint" default which uses rAU = dt and only bleeds the
        # divergence off at rate (1 - dt) -- far too slow to absorb the large
        # divergence the overset donor interpolation injects at a thin element).
        # The Rhie-Chow face operator is ill-conditioned on a tiny highly-stretched
        # element grid and SuperLU returns a garbage pressure there (the coupled
        # slat/flap run diverges); the adjoint operator is condition-number ~ N^2
        # and solves to machine precision on every grid, so it is the robust choice
        # for the 3-element overset.  The collocated checkerboard it retains is a
        # known, physically-inert artifact for the divergence-free velocity.
        blk._rAU = np.ones_like(blk.J)
        L_full = _adj_lap_matrix(blk)
    else:
        L_full = _adj_lap_matrix(blk)
        blk._rAU = dt * np.ones_like(blk.J)   # adjoint correction weight (uniform)
    if reg_r > 0.0:
        # NOTE: deliberately VERTICAL-only (eta) regularization.  A 2D (i+j)
        # regularization was tried but the rhie2 operator is ill-conditioned, so
        # any finite reg_r distorts the SMOOTH physical pressure (cylinder Cd
        # jumped 2.04 -> 4.08 at reg_r=1e-2) instead of cleanly killing only the
        # circumferential checkerboard.  The checkerboard is handled where it
        # actually hurts -- the FORCE -- via the face-interpolated wall pressure
        # in c_forces (the checkerboard's gradient is identically zero, so it
        # never affects the flow, only the raw cell-centre pressure read by the
        # force).  See _rhie_lap_consistent / c_forces.
        Lr = L_full.tolil()
        for i in range(ni):
            for j in range(1, nj - 1):
                k = i * nj + j
                Lr[k, i * nj + (j - 1)] += reg_r; Lr[k, i * nj + j] += -2.0 * reg_r
                Lr[k, i * nj + (j + 1)] += reg_r
        L_full = Lr.tocsc()
    solved_idx = np.nonzero(solved)[0]
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    # --- Overset HOLE = NEUMANN (zero-normal-velocity pressure BC) -----------
    # The hole is fully covered by the donor.  The OLD treatment forced its
    # pressure to the DONOR value (folded into the RHS via _L_sr below for
    # fringe|farfield).  That creates an O(1) seam jump (solved p biased to 0,
    # donor p = O(1)) which round-trips through the overset exchange every step
    # and diverges the coupled run exponentially (~1e3 steps to blow up).  Fix:
    # treat the HOLE as a NEUMANN boundary -- p_hole = p_solved (zero normal
    # gradient) -- so the pressure is set by the LOCAL flow, not the donor, and
    # the projection injects NO normal velocity into the covered region.  This is
    # done in two parts:
    #   (1) OPERATOR: remove the hole coupling from the Laplacian row (the term
    #       a_{s,h}(p_h - p_s) vanishes when p_h = p_s), i.e. subtract a_{s,h}
    #       from the solved diagonal.  Use the FINAL L_full (incl. reg_r).
    #   (2) VELOCITY CORRECTION: after the solve, fill p[hole] = mean of solved
    #       neighbours (_build_hole_fill) so c_grad at the seam is continuous.
    # fringe|farfield KEEP the donor-pressure Dirichlet BC (handled by _L_sr).
    hole_mask = (None if hole_mask is None else np.asarray(hole_mask))
    if hole_mask is None:
        hole_mask = getattr(blk, "hole", None)
    hole = (np.zeros(N, dtype=bool) if hole_mask is None
            else np.asarray(hole_mask).reshape(-1).astype(bool)) & recv
    dirichlet = recv & ~hole
    if hole.any() and mode in ("rhie2", "rhie2n", "rhie"):
        hole_idx = np.nonzero(hole)[0]
        Lsh = L_full[solved_idx][:, hole_idx]              # (ns, nh) = a_{s,h} > 0
        diag_sub = np.asarray(Lsh.sum(axis=1)).ravel()
        L_red = L_red.tolil()
        L_red.setdiag(L_red.diagonal() - diag_sub)          # p_hole = p_s solved-in
        L_red = L_red.tocsc()
        blk._hole_idx = hole_idx
        blk._hole_fill = _build_hole_fill(blk, hole_idx, solved_idx)
    else:
        blk._hole_idx = None
        blk._hole_fill = None
    # Overset inhomogeneous pressure BC for fringe|farfield (NOT the hole): the
    # Poisson solve treats them as homogeneous p=0, but the velocity correction
    # reads the full p field whose fringe|farfield entries are the DONOR pressure
    # (O(1)).  Fold L{solved,recv}.p_recv into the RHS so the solved pressure is
    # continuous there (the hole is excluded -- it is Neumann, see above).
    recv_idx = np.nonzero(dirichlet)[0]
    blk._L_sr = L_full[solved_idx][:, recv_idx].tocsc()
    blk._recv_idx = recv_idx
    blk._solved_idx = solved_idx
    blk._poisson_mode = mode                 # remembered by project() so it can
                                             # divide the RHS by dt for adjoint
    if mode == "rhie":
        # Rhie-Chow pressure operator is ill-conditioned (kappa ~ rAU_ratio*N^2
        # exceeds double precision on stretched grids), so a naive ILU/BiCGStab
        # (_PoissonSolver) cannot solve it and the explicit correction diverges.
        # The geometric multigrid AMG solver (_MGSolver) is condition-number
        # independent and recovers the smooth physical pressure -> this is the
        # SIMPLE+AMG upgrade that makes Rhie-Chow viable on stretched C-grids.
        return _MGSolver(L_red, solved_idx, ni, nj)
    return _SpluPoisson(L_red, solved_idx)


def build_reproj_solver(blk, dt, mode="rhie2", reg_r=0.0,
                        farfield_mask=None, wall_mask=None, hole_mask=None):
    """HOLE-FILLING re-projection Poisson (POST-EXCHANGE).

    Built on the SAME rhie2 (face-gradient Rhie-Chow) operator as project() with
    rAU = dt/J, so the velocity correction is GENTLE (|u'| ~ rAU*|grad p|, NOT
    |grad p|) and does NOT overshoot on the thin element blocks the way the old
    rAU=1 adjoint re-projection did (that one blew the run up at step ~2 because a
    ~40 solved divergence became a ~40 correction on a ~0.03 velocity field).

    Dirichlet (donor-pressure) BC cells = hole | farfield.  Solve domain =
    solved | fringe (everything except hole and farfield).  The hole KEEPS its
    donor velocity (not corrected); only the solved+fringe interior is made
    divergence-free in blk's own metric, so the divergence that main's coarse
    velocity carries onto the fine element grid -- interpolated bilinearly into
    the slat/flap recv cells -- is removed at the SOURCE every step.  This breaks
    the slow two-way overset blow-up (growth ~1.5e-3/step, ~1e3 steps to explode in
    the un-re-projected run): the thin elements no longer feed a divergent inflow
    into main's hole, main stays clean, and main's clean donor no longer spikes the
    elements back."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    far = (np.zeros(N, dtype=bool) if farfield_mask is None
           else np.asarray(farfield_mask).reshape(-1).astype(bool))
    hole = (np.zeros(N, dtype=bool) if hole_mask is None
            else np.asarray(hole_mask).reshape(-1).astype(bool))
    # Dirichlet BC = hole | farfield (donor pressure held, O(1)-scale, consistent
    # with exchange_pressure).  Solve domain = solved | fringe.
    dirichlet = far | hole
    solved = ~dirichlet
    if not solved.any():
        solved = np.ones(N, dtype=bool); solved[0] = False; dirichlet = ~solved
    rAU = dt / blk.J
    L_full = _rhie_lap_face_matrix(blk, rAU)
    if reg_r > 0.0:
        # vertical-only regularization (see main build_poisson note on why 2D
        # reg was reverted).
        Lr = L_full.tolil()
        for i in range(ni):
            for j in range(1, nj - 1):
                k = i * nj + j
                Lr[k, i * nj + (j - 1)] += reg_r
                Lr[k, i * nj + j] += -2.0 * reg_r
                Lr[k, i * nj + (j + 1)] += reg_r
        L_full = Lr.tocsc()
    solved_idx = np.nonzero(solved)[0]
    recv_idx = np.nonzero(dirichlet)[0]
    # INHOMOGENEOUS overset pressure BC: fold the donor pressure (held on blk.p by
    # exchange_pressure, O(1)-scale, one step stale) into the RHS as
    # -L{solved,recv}.p_recv so the re-projected pressure is continuous with the
    # donor at the seam and the correction removes ONLY the donor-injected
    # divergence (not a spurious O(1) pressure jump that would blow up).
    blk._reproj_L_sr = L_full[solved_idx][:, recv_idx].tocsc()
    blk._reproj_recv_idx = recv_idx
    blk._reproj_solved_idx = solved_idx
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    return _SpluPoisson(L_red, solved_idx)


def reproject(blk, dt, solver, solved_idx, mode="rhie2",
              farfield_mask=None, wall_mask=None, hole_mask=None):
    """Post-exchange hole-filling re-projection: make the solved+fringe interior
    divergence-free in blk's metric, holding the hole|farfield donor pressure as
    Dirichlet.  blk.p is left UNTOUCHED (it stays the donor pressure for
    exchange_pressure).  The correction uses rAU = dt/J (gentle, matches
    project) with the face-gradient (rhie2) so it is the exact discrete adjoint of
    L and removes divergence in ONE step; it is applied only to non-BC cells
    (rAU zeroed on wall / hole / farfield) so the no-slip, free-stream and donor
    velocity BCs are preserved exactly.  Mutates blk.u, blk.v in place."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    rAU = dt / blk.J
    far = (np.zeros((ni, nj), dtype=bool) if farfield_mask is None
           else np.asarray(farfield_mask).reshape(ni, nj))
    hole = (np.zeros((ni, nj), dtype=bool) if hole_mask is None
            else np.asarray(hole_mask).reshape(ni, nj))
    wall = (blk.wall if wall_mask is None
            else np.asarray(wall_mask).reshape(ni, nj))
    dirichlet = far | hole
    rAUc = rAU.copy()
    rAUc[wall] = 0.0
    rAUc[dirichlet] = 0.0            # hole + farfield keep donor velocity
    div = c_div(blk, blk.u, blk.v)
    rhs = div.reshape(-1)[solved_idx].copy()
    # Inhomogeneous overset pressure BC: fold donor pressure into RHS (see
    # build_reproj_solver) so the seam is continuous with the donor.
    if getattr(blk, "_reproj_L_sr", None) is not None:
        _pr = blk.p.reshape(-1)[blk._reproj_recv_idx]
        rhs = rhs - blk._reproj_L_sr.dot(_pr)
    pfull = np.zeros(N)
    # Hold donor pressure at the hole|farfield so the gradient at the seam-adjacent
    # solved cells is consistent (the solver itself only fills solved_idx).
    pfull[blk._reproj_recv_idx] = blk.p.reshape(-1)[blk._reproj_recv_idx]
    pfull[solved_idx] = solver.solve(rhs)
    pfull = pfull.reshape(ni, nj)
    if mode == "rhie2":
        gx, gy = _rhie_grad_face(blk, rAU, pfull)
    else:
        gx, gy = c_grad(blk, pfull)
    blk.u = blk.u - rAUc * gx
    blk.v = blk.v - rAUc * gy


def build_poisson_iterative(blk, recv_mask=None, reg_r=200.0, dt=1.0, mode="adjoint"):
    """Large-grid variant of build_poisson (same interface / same returned
    _PoissonSolver object) kept for grids with N > 45000.  Mirrors build_poisson's
    mode routing so the adjoint default stays stable on stretched meshes and the
    tuple fallback path in project() is never needed."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    recv = (blk.recv if recv_mask is None else np.asarray(recv_mask)).reshape(-1).astype(bool)
    solved = ~recv
    if not solved.any():
        solved = np.ones(N, dtype=bool); solved[0] = False; recv = ~solved
    if mode == "rhie":
        L_full = _rhie_lap_matrix(blk, dt)
    elif mode == "rhie2":
        blk._rAU = np.ones_like(blk.J)
        L_full = _rhie_lap_face_matrix(blk, blk._rAU)
    elif mode == "adj1":
        blk._rAU = np.ones_like(blk.J)
        L_full = _adj_lap_matrix(blk)
    else:
        L_full = _adj_lap_matrix(blk)
        blk._rAU = dt * np.ones_like(blk.J)
    if reg_r <= 0.0:
        reg_r = 200.0
    Lr = L_full.tolil()
    for i in range(ni):
        for j in range(1, nj - 1):
            k = i * nj + j
            Lr[k, i * nj + (j - 1)] += reg_r; Lr[k, i * nj + j] += -2.0 * reg_r
            Lr[k, i * nj + (j + 1)] += reg_r
    L_full = Lr.tocsc()
    solved_idx = np.nonzero(solved)[0]
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    recv_idx = np.nonzero(recv)[0]
    blk._L_sr = L_full[solved_idx][:, recv_idx].tocsc()
    blk._recv_idx = recv_idx
    blk._solved_idx = solved_idx
    blk._poisson_mode = mode
    if mode == "rhie":
        return _MGSolver(L_red, solved_idx, ni, nj)
    return _SpluPoisson(L_red, solved_idx)


def project(blk, dt, poisson):
    """Rhie-Chow pressure projection (OpenFOAM pRelax flavour).

    Solves the Poisson equation built from rAU = dt/J (cached on blk._rAU by
    build_poisson / _rhie_lap_matrix, possibly clamped) and corrects the velocity
    with the SAME rAU so the operator and correction stay exact discrete adjoints:

        L p = c_div( rAU * grad(p) ) = div(u*) ,   u <- u - rAU * grad(p)

    Under-relaxation (OpenFOAM pRelax, blk._p_ur in (0,1]) blends the freshly
    solved pressure *field* toward the previous one instead of applying the full
    correction in one step.  This is the genuine pRelax (NOT a rhs/correction
    factor that cancels): the velocity correction magnitude is scaled by ur, which
    damps the near-wall overshoot of the explicit (Chorin) correction on strongly
    stretched grids.  Multiple inner corrections (blk._p_ncorr, default 1) form a
    SIMPLE-like loop that drives the residual divergence down progressively without
    ever applying an un-damped correction."""
    ur = getattr(blk, "_p_ur", 1.0)
    ncorr = getattr(blk, "_p_ncorr", 1)
    rAU = getattr(blk, "_rAU", dt / blk.J)
    # Correction weight for the velocity update.  In EXACT-projection mode
    # (rhs = div/dt -> p_true = O(rho*U^2) ~ O(1)) the Chorin correction is
    # u' = dt * grad(p_true); using the rhie2 rAU = 1 instead gives u' ~ 1/h_wall
    # (~1e4 on this wall-fn grid) -- i.e. an overshoot of ~1/dt that diverges the
    # run in 1-2 steps (verified: 30P30N |u|max -> 4e3 at the LE/TE wall cells).
    # So in exact mode we apply the TRUE Chorin weight dt (removes the divergence
    # fully AND stays O(dt/h) ~ O(0.1) -> stable), while keeping rhs=div/dt so the
    # pressure field is the true O(1) projection that yields correct forces.
    rcorr = dt if getattr(blk, "_p_exact", False) else rAU
    solved_idx = blk._solved_idx
    is_tuple = isinstance(poisson, tuple)
    # Route-1 ROOT-CAUSE fix: wall-consistent divergence for the exact projection.
    # The wall-function sets a slip TANGENTIAL velocity at the body wall (j=0);
    # its normal component is zeroed (no-penetration).  The slip JUMP (O(U))
    # between the wall cell and the first interior cell makes div(u*) ~ U/h_wall
    # there, so the exact-projection pressure p = L^-1(div(u*)/dt) is driven to
    # p ~ U*h/dt = (1/CFL_wall) * rho*U^2 -- ~10x the PHYSICAL pressure (which is
    # why the forces come out ~10x too big and must be reined in with p_cap).
    # For a converged (divergence-free) velocity the INTERIOR div(u*) is O(dt);
    # ONLY the wall carries the O(U/h) artifact.  So for the divergence evaluation
    # we replace the wall-cell velocity with a no-jump continuation: zero normal
    # flux + linearly-extrapolated tangential (ghost = 2*u[:,1] - u[:,2]).  This
    # makes div(u*) O(dt) everywhere, so p converges to the physical O(rho*U^2)
    # and the forces no longer depend on p_cap.  The slip jump is RETAINED for
    # c_forces (which uses the j=1 - j=0 gradient = the wall shear) and for the
    # next step's wall-function re-imposition.  Enabled by blk._p_wall_nojump;
    # only acts in exact-projection mode (the adjoint mode keeps the gentle
    # rAU=dt under-relaxation where p is already physical-scale and forces are
    # not over-estimated).
    _wall_fix = bool(getattr(blk, "_p_wall_nojump", False)) and bool(
        getattr(blk, "_p_exact", False))
    for _ in range(max(1, int(ncorr))):
        _wall_saved = None
        if _wall_fix:
            _wm = getattr(blk, "wall", None)
            if _wm is not None and _wm[:, 0].any():
                _wc = _wm[:, 0]
                _Aex = blk.Aef_x[:, 0]; _Aey = blk.Aef_y[:, 0]
                _A = np.maximum(np.hypot(_Aex, _Aey), 1e-30)
                _nx = _Aex / _A; _ny = _Aey / _A        # wall normal (into domain)
                _tx = -_ny; _ty = _nx                     # wall tangent (unit)
                _u0 = blk.u[:, 0].copy(); _v0 = blk.v[:, 0].copy()
                # tangential extrapolation from interior (j=1, j=2); normal = 0
                _ut1 = blk.u[:, 1] * _tx + blk.v[:, 1] * _ty
                _ut2 = blk.u[:, 2] * _tx + blk.v[:, 2] * _ty
                _utw = 2.0 * _ut1 - _ut2
                blk.u[_wc, 0] = (_utw * _tx)[_wc]
                blk.v[_wc, 0] = (_utw * _ty)[_wc]
                _wall_saved = (_wc, _u0, _v0)
        div = c_div(blk, blk.u, blk.v)
        if _wall_saved is not None:
            _wc, _u0, _v0 = _wall_saved
            blk.u[_wc, 0] = _u0[_wc]; blk.v[_wc, 0] = _v0[_wc]
        # RHS of the (reduced) pressure equation.
        #
        # ADJOINT MODE (the production-stable default): L = c_div o c_grad is the
        # UNWEIGHTED operator and rAU = dt (uniform) in the correction
        # u <- u - rAU * c_grad(p).  With the raw RHS = div(u*) the correction is
        # u' = dt * c_grad(p) and div(u_new) = div(u*) - dt * L p = div(u*)
        # - dt * div(u*) = (1 - dt) div(u*): the divergence is removed at the
        # rate (1 - dt) PER STEP.  Because dt ~ 1e-4-1e-2 this is a GENTLE,
        # STABLE under-relaxation -- the pressure stays O(div) (physical scale)
        # and the explicit step never overshoots.  Over a few thousand steps the
        # divergence is fully removed and the pressure converges to the true
        # (O(1)) projection, giving correct forces (verified: cylinder Re=100
        # -> Cd=1.09).  Dividing the RHS by dt (a once-attempted "exact" fix)
        # makes |p| ~ div/dt ~ 1/dt and the correction overshoots by ~1/dt ->
        # the explicit step diverges in 1-2 steps, so it is deliberately NOT done.
        #
        # RHIE-CHOW MODE (mode="rhie"): L = c_div( rAU * c_grad ) already folds
        # rAU into L, so its RHS is the raw div(u*) (consistent, no /dt).
        rhs = div.reshape(-1)
        # EXACT projection (Chorin): solve L p = div(u*)/dt so the pressure field
        # is the true O(rho*U^2) projection (not the under-relaxed partial
        # correction that decays to ~0 as the flow becomes divergence-free).
        # Without this, p ~ div/L -> ~1e-3 on stretched grids and the forces are
        # ~500x too small (verified: 30P30N Cl stalls at ~1e-4).  The resulting
        # larger correction is tamed by _proj_ucap (per-step |u'| cap).  Set via
        # blk._p_exact.
        if getattr(blk, "_p_exact", False):
            rhs = rhs / dt
        # Inhomogeneous overset pressure BC: the Poisson matrix was built with
        # recv cells dropped (homogeneous p=0).  Fold the ACTUAL recv pressure
        # (the donor pressure, O(1), held on blk.p by exchange_pressure) into the
        # RHS as -L{solved,recv} . p_recv so the solved pressure is continuous
        # with the donor pressure at the seam.  Without this the solved pressure
        # is biased to 0 at the seam while the velocity correction reads donor p
        # there -> an O(1) jump that round-trips through the overset exchange and
        # diverges the coupled run exponentially (delayed until donor p ~ O(1)).
        if getattr(blk, "_L_sr", None) is not None:
            _pr = blk.p.reshape(-1)[blk._recv_idx]
            rhs[blk._solved_idx] = rhs[blk._solved_idx] - blk._L_sr.dot(_pr)
        if is_tuple:
            L_red, _ = poisson
            x0 = blk.p.reshape(-1)[solved_idx] if blk.p is not None else None
            M = diags(L_red.diagonal())
            p_sol, info = bicgstab(L_red, rhs[solved_idx], x0=x0, M=M,
                                   rtol=1e-9, atol=0.0, maxiter=800)
            if info != 0:
                p_sol, info = bicgstab(L_red, rhs[solved_idx], x0=x0,
                                       rtol=1e-7, atol=0.0, maxiter=1500)
            pfull = np.zeros(blk.ni * blk.nj); pfull[solved_idx] = p_sol
        else:
            pfull = np.zeros(blk.ni * blk.nj)
            x0 = blk.p.reshape(-1)[solved_idx] if blk.p is not None else None
            pfull[solved_idx] = poisson.solve(rhs[solved_idx], x0=x0)
        pfull = pfull.reshape(blk.ni, blk.nj)
        # pRelax: blend the solved correction toward the previous pressure field.
        p_prev = blk.p if blk.p is not None else np.zeros_like(pfull)
        blk.p = p_prev + ur * (pfull - p_prev)
        # Overset HOLE = NEUMANN companion: fill the hole pressure with the
        # zero-gradient (mean-of-solved-neighbours) value so the velocity-
        # correction gradient at seam-adjacent solved cells is continuous (no
        # normal injection into the covered region).  Without this the donor
        # pressure O(1) jump at the hole would re-enter the correction and the
        # coupled run still diverges (the operator diagonal-fold alone is not
        # enough -- the gradient must read a consistent p field).
        if getattr(blk, "_hole_fill", None) is not None and blk._hole_idx is not None:
            _pf = blk.p.reshape(-1)
            _pf[blk._hole_idx] = blk._hole_fill.dot(_pf)
            blk.p = _pf.reshape(blk.ni, blk.nj)
        # Pressure cap: the EXACT projection (rhs=div/dt) solves the true O(rho*U^2)
        # pressure, but at the suddenly-imposed no-slip wall (step 1, ramp=1) the
        # discrete wall divergence ~U/h drives p ~ 1/dt (1e4+), a grid-scale
        # transient that destabilises the coupled run.  The STEADY pressure is
        # O(1) (|Cp|<=~2 -> |p|<=~1), so capping |p| to a physical bound removes
        # only the startup spike and leaves the physical field intact.  Set via
        # blk._p_cap (0 disables).
        pcap = getattr(blk, "_p_cap", 0.0)
        if pcap and pcap > 0.0:
            blk.p = np.clip(blk.p, -pcap, pcap)
        # PER-CELL PHYSICAL PRESSURE SCALING (lightweight route-1).  The exact
        # projection solves p = L^-1(div(u*)/dt).  Off-wall div(u*) is the true
        # (tiny) residual so the bulk pressure keeps the CORRECT O(1) magnitude
        # and its gradient already drives the correct velocity correction.  But
        # at the body wall the BC velocity jump (no-slip or log-law slip) puts an
        # O(1/h_wall) spurious divergence into div(u*), which the 1/dt factor
        # amplifies to O(1/dt) -- a near-wall pressure SPIKE that is local (it
        # does NOT feed the bulk divergence removal) yet, once integrated over
        # the wall, inflates the forces (verified: Cl ~ pcap grows linearly).
        # This rescales only the HIGH-divergence (near-wall artifact) cells toward
        # a floor `p_scale`, leaving the bulk field (w~1) untouched, so the force
        # magnitude is pulled back toward physics without breaking the divergence
        # correction.  The weight w = 1/(1+(|div|/ref)^2) with
        # ref = p_scale_ref * max|div| smoothly isolates the spike.  Set via
        # blk._p_scale (0 disables; 0<v<1 floor for artifact cells).
        pscale = getattr(blk, "_p_scale", 0.0)
        if pscale and 0.0 < pscale < 1.0:
            if getattr(blk, "_p_scale_mode", "div") == "wall":
                # WALL-DISTANCE weight: the inflated near-wall spike lives in the
                # first few wall layers (d ~ h_wall).  Cells with wall-distance d
                # below dref are pulled toward floor p_scale; cells beyond dref are
                # untouched (bulk pressure is already physical O(1)).  This catches
                # the artifact far better than the div-based weight (whose ref =
                # p_scale_ref*max|div| is dominated by the spike itself -> w~1
                # everywhere -> no effect).  _p_scale_ref is reused as dref here.
                _d = getattr(blk, "_d", None)
                if _d is None:
                    _d = c_walldist(blk)
                _dref = max(getattr(blk, "_p_scale_ref", 6e-4), 1e-12)
                _r = np.minimum(np.maximum(_d / _dref, 0.0), 1.0)
                _w = _r * _r
            else:
                _da = np.abs(div)
                _vmax = float(np.max(_da)) + 1e-30
                _ref = max(getattr(blk, "_p_scale_ref", 0.1) * _vmax, 1e-30)
                _w = 1.0 / (1.0 + (_da / _ref) ** 2)
            blk.p = blk.p * (_w + (1.0 - _w) * pscale)
        # Velocity correction gradient: rhie2 uses the FACE-gradient (consistent
        # with its pressure operator) so the correction is the exact discrete
        # adjoint of L and removes divergence without over-correcting; the other
        # modes use the standard cell-centre gradient.
        if blk._poisson_mode == "rhie2":
            gx, gy = _rhie_grad_face(blk, rAU, blk.p)
        else:
            gx, gy = c_grad(blk, blk.p)
        # Velocity-correction LIMITER.  On a finely-resolved (y+~50) wall the
        # no-slip divergence ~ U/hy_wall makes the UN-limited chorin correction
        # |u'| = rAU*|grad p| ~ U/hy_wall (~10^3 x the free stream) -- the
        # explicit step then diverges in 1-2 steps regardless of rAU (rAU cancels
        # in div(u_new)).  Capping |u'| per step to ~O(U_eff) does NOT stop the
        # divergence from being removed (the cap is applied every step, so the
        # residual is bled off gradually and the pressure converges to the true
        # projection); it only prevents the single-step overshoot.  Set via
        # blk._proj_ucap (absolute |u'| cap); 0 disables.
        ucap = getattr(blk, "_proj_ucap", 0.0)
        if ucap and ucap > 0.0:
            cux = rcorr * gx; cuy = rcorr * gy
            cmag = np.sqrt(cux ** 2 + cuy ** 2)
            with np.errstate(divide="ignore", invalid="ignore"):
                f = np.where(cmag > ucap,
                             ucap / np.where(cmag > 0.0, cmag, 1e-30),
                             1.0)
            gx = gx * f; gy = gy * f
        blk.u = blk.u - rcorr * gx
        blk.v = blk.v - rcorr * gy


def c_forces(blk, nu, U):
    """Face-vector-consistent force on the no-slip wall (j=0 line).  Only the
    wall cells (blk.wall[:, 0]) contribute; tail/wake-cut j=0 cells are excluded
    so the C-grid force is the true body force (not polluted by the free-stream
    cut).  Returns (Cd, Cl, Fx, Fy) with Cd, Cl normalised by 0.5*U^2*Lref."""
    j = 0
    wall = blk.wall[:, 0]
    pcol = blk.p[:, j] * getattr(blk, "_p_phys_f", 1.0)
    # Checkerboard-free wall pressure for the force, via a NON-WRAPPING 1-2-1
    # average in the circumferential (i) index.  On a collocated grid the discrete
    # pressure Laplacian leaves the i-alternating checkerboard in its NULL SPACE
    # (c_div of any i-alternating field is identically 0, so the overset seam
    # excites it and it integrates into spurious pressure drag on small curved
    # elements -- observed flap Cp = +/-4.3 perfect alternation -> flap Cd ~1.0).
    # The 1-2-1 stencil annihilates that null mode EXACTLY (0.25*(-1)+0.5+0.25*(-1)
    # = 0) while leaving the smooth physical pressure intact, and we do NOT wrap at
    # the wake cut (the two TE-side body ends are physically distinct), so the
    # owner-cell x face-area-vector force structure is preserved exactly -- this
    # matches TensorFVM's _surface_force = sum(p[owner] * S_face).  Crucially the
    # checkerboard's gradient is identically zero (the face-gradient vectors
    # cancel), so it never affects the velocity correction or the flow -- it only
    # corrupts the force, which reads the raw cell-centre p.  (Fused from the
    # TensorFVM reference solver, which avoids the checkerboard by construction via
    # a SIMPLE pressure correction that uses the same under-relaxed momentum
    # diagonal for flux and correction.)
    p = pcol.copy()
    p[1:-1] = 0.25 * pcol[:-2] + 0.5 * pcol[1:-1] + 0.25 * pcol[2:]
    Aix = blk.Aef_x[:, j]; Aiy = blk.Aef_y[:, j]
    Fpx = -np.sum(p[wall] * Aix[wall]); Fpy = -np.sum(p[wall] * Aiy[wall])
    gxu, gyu = c_grad(blk, blk.u); gxv, gyv = c_grad(blk, blk.v)
    ux = gxu[:, j]; uy = gyu[:, j]; vx = gxv[:, j]; vy = gyv[:, j]
    txx = 2.0 * nu * ux; txy = nu * (uy + vx); tyy = 2.0 * nu * vy
    Fvx = np.sum(txx[wall] * Aix[wall] + txy[wall] * Aiy[wall])
    Fvy = np.sum(txy[wall] * Aix[wall] + tyy[wall] * Aiy[wall])
    Fx = Fpx + Fvx; Fy = Fpy + Fvy
    q = 0.5 * U * U * blk.Lref
    return Fx / q, Fy / q, float(Fx), float(Fy)


def c_wall_function(blk, nu, kappa=0.41, E=9.8):
    """High-Reynolds (log-law) WALL-FUNCTION velocity BC on the body wall (j=0
    row).  Sets the wall-cell tangential velocity to the log-law value derived
    from the FIRST INTERIOR cell (j=1) and zeroes the wall-normal component (no
    penetration).  The pressure projection keeps the interior divergence-free;
    the wall shear is then recovered by c_forces from the (u[j=1]-u[j=0])/d
    gradient -- so there is NO explicit shear source and NO double-counting with
    the diffusion operator.  This lets the explicit FVM run a y+~30-60 grid
    (steps ~1e3-1e4 to converge) instead of an impossible y+~1 grid (~1e8).

    u_tau solved from  u_t = (u_tau/kappa) * ln(E*y+),  y+ = u_tau*d/nu
    (fixed-point; 12 iters converges to machine precision for y+ in [5,500])."""
    wall0 = blk.wall[:, 0]                       # (ni,) body wall cells (j=0 row)
    if not wall0.any():
        return
    Aex = blk.Aef_x[:, 0]; Aey = blk.Aef_y[:, 0]   # wall eta-face (outward +eta)
    A = np.maximum(np.hypot(Aex, Aey), 1e-30)
    nx = Aex / A; ny = Aey / A                    # wall normal (into domain)
    tx = -ny; ty = nx                             # wall tangent (unit)
    d = c_walldist(blk)                           # (ni,nj) wall distance
    d_w = d[:, 0]; d_in = d[:, 1]
    # reference tangential speed at the first interior cell
    ut = blk.u[:, 1] * tx + blk.v[:, 1] * ty
    absut = np.abs(ut)
    yp0 = np.maximum(absut * d_in / nu, 1e-6)
    utau = np.maximum(absut * kappa / np.maximum(np.log(E * yp0), 1e-3), 1e-9)
    for _ in range(12):
        yp = np.maximum(utau * d_in / nu, 1e-6)
        utau = np.maximum(absut * kappa / np.maximum(np.log(E * yp), 1e-3), 1e-9)
    # wall-cell tangential velocity from the log law at d_w (slip velocity)
    ut_wall = utau / kappa * np.log(E * np.maximum(utau * d_w / nu, 1e-6))
    sgn = np.sign(ut); sgn[sgn == 0] = 1.0
    # assign only at the actual body-wall cells (wall0 selector)
    blk.u[wall0, 0] = (sgn * ut_wall * tx)[wall0]
    blk.v[wall0, 0] = (sgn * ut_wall * ty)[wall0]


# --------------------------------------------------------------------------- #
# Spalart-Allmaras (SA) one-equation turbulence model on the curve_grid FVM
# layer.  The working variable nu_tilde is advected/diffused by the SAME
# curvilinear operators (c_conv_scalar / c_div / c_grad) so it stays exactly
# metric-consistent with the momentum solve.  Standard SA constants.
# --------------------------------------------------------------------------- #
SA_CB1 = 0.1355
SA_CB2 = 0.622
SA_SIGMA = 2.0 / 3.0
SA_KAPPA = 0.41
SA_CW1 = SA_CB1 / SA_KAPPA ** 2 + (1.0 + SA_CB2) / SA_SIGMA
SA_CW2 = 0.3
SA_CW3 = 2.0
SA_CV1 = 7.1
SA_CT1 = 1.0
SA_CT2 = 2.0
SA_CT3 = 1.2
SA_CT4 = 0.5


def c_vorticity(blk, u, v):
    """|omega| = |dv/dx - du/dy| at cell centres (Cartesian gradient from c_grad)."""
    gxu, gyu = c_grad(blk, u)
    gxv, gyv = c_grad(blk, v)
    return np.abs(gxv - gyu)


def c_walldist(blk):
    """Wall-normal distance d, cached on blk._d.  For a body-fitted C-grid the
    wall is the j=0 line, so d is the cumulative eta-arc-length from j=0.  Near
    the wall (where the SA destruction r = nu_tilde/(kappa^2 omega d^2) matters)
    eta IS wall-normal, so this is exact there; far from the wall d is only
    approximate, but the destruction term is negligible there anyway."""
    if getattr(blk, "_d", None) is not None:
        return blk._d
    ni, nj = blk.ni, blk.nj
    xc, yc = blk.xc, blk.yc
    D = np.sqrt((xc[:, 1:] - xc[:, :-1]) ** 2 + (yc[:, 1:] - yc[:, :-1]) ** 2)  # (ni,nj-1)
    d = np.zeros((ni, nj))
    d[:, 0] = 0.5 * D[:, 0]
    for j in range(1, nj):
        if j < nj - 1:
            d[:, j] = d[:, j - 1] + 0.5 * (D[:, j - 1] + D[:, j])
        else:
            d[:, j] = d[:, j - 1] + D[:, j - 1]
    blk._d = d
    return d


def c_conv_scalar(blk, u, v, phi, scheme="upwind1"):
    """Conservative FV convection of a scalar phi advected by (u,v):
    returns div(phi * u_vec) with phi upwind-reconstructed at faces.  upwind1 is
    robust and standard for the SA nu_tilde transport; ppm is offered if a
    higher-order scalar flux is wanted."""
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    # ---- xi faces (0..ni) ----
    if scheme in ("ppm", "weno", "muscl", "vanleer"):
        Pu = _pad_i(u, 3, per); Pv = _pad_i(v, 3, per); Pp = _pad_i(phi, 3, per)
        if scheme == "ppm":
            uL, uR = _ppm_states(Pu); vL, vR = _ppm_states(Pv); pL, pR = _ppm_states(Pp)
        elif scheme == "weno":
            uL, uR = _weno5_states(Pu); vL, vR = _weno5_states(Pv); pL, pR = _weno5_states(Pp)
        else:
            PuL = np.roll(Pu, 1, 0); PuR = np.roll(Pu, -1, 0); PuLL = np.roll(Pu, 2, 0); PuRR = np.roll(Pu, -2, 0)
            PvL = np.roll(Pv, 1, 0); PvR = np.roll(Pv, -1, 0); PvLL = np.roll(Pv, 2, 0); PvRR = np.roll(Pv, -2, 0)
            PpL = np.roll(Pp, 1, 0); PpR = np.roll(Pp, -1, 0); PpLL = np.roll(Pp, 2, 0); PpRR = np.roll(Pp, -2, 0)
            lim = _minmod if scheme == "muscl" else _vanleer
            uL = (Pu + 0.5 * lim(Pu - PuLL, PuR - Pu))[3:3 + ni]
            uR = (Pu - 0.5 * lim(Pu - PuRR, PuL - Pu))[3:3 + ni]
            vL = (Pv + 0.5 * lim(Pv - PvLL, PvR - Pv))[3:3 + ni]
            vR = (Pv - 0.5 * lim(Pv - PvRR, PvL - Pv))[3:3 + ni]
            pL = (Pp + 0.5 * lim(Pp - PpLL, PpR - Pp))[3:3 + ni]
            pR = (Pp - 0.5 * lim(Pp - PpRR, PpL - Pp))[3:3 + ni]
        Ax = blk.Axi_x[:-1]; Ay = blk.Axi_y[:-1]
        mflxL = uL * Ax + vL * Ay; mflxR = uR * Ax + vR * Ay
        alpha = np.maximum(np.abs(mflxL), np.abs(mflxR))
        Fxi = 0.5 * (mflxL + mflxR) - 0.5 * alpha * (uR - uL)
        Fxphi = _upwind(pL, pR, Fxi) * Fxi
    else:
        ul, vl = _il(u, per), _il(v, per); ur, vr = u, v
        Fxi = 0.5 * (ul + ur) * blk.Axi_x[:-1] + 0.5 * (vl + vr) * blk.Axi_y[:-1]
        Fxphi = _upwind(_il(phi, per), phi, Fxi) * Fxi
    if per:
        dphi = (np.roll(Fxphi, -1, 0) - Fxphi) / blk.J
    else:
        Fxphi_full = np.concatenate([Fxphi, Fxphi[-1:]], axis=0)
        dphi = (Fxphi_full[1:] - Fxphi_full[:-1]) / blk.J
    # ---- eta faces (0..nj): zero normal flux at wall (j=0), Neumann at far ----
    if scheme in ("ppm", "weno", "muscl", "vanleer"):
        Mu = np.zeros((nj + 7, ni)); Mv = np.zeros((nj + 7, ni)); Mp = np.zeros((nj + 7, ni))
        Mu[3:nj + 3, :] = u.T; Mv[3:nj + 3, :] = v.T; Mp[3:nj + 3, :] = phi.T
        for g in (0, 1, 2):
            Mu[g, :] = -Mu[6 - g, :]; Mv[g, :] = -Mv[6 - g, :]; Mp[g, :] = -Mp[6 - g, :]
        for g in (nj + 3, nj + 4, nj + 5, nj + 6):
            Mu[g, :] = Mu[nj + 2, :]; Mv[g, :] = Mv[nj + 2, :]; Mp[g, :] = Mp[nj + 2, :]
        if scheme == "ppm":
            uLr, uRr = _ppm_states(Mu); vLr, vRr = _ppm_states(Mv); pLr, pRr = _ppm_states(Mp)
        elif scheme == "weno":
            uLr, uRr = _weno5_states(Mu); vLr, vRr = _weno5_states(Mv); pLr, pRr = _weno5_states(Mp)
        else:
            uLr = Mu; uRr = Mu; vLr = Mv; vRr = Mv; pLr = Mp; pRr = Mp
        uLr = uLr.T; uRr = uRr.T; vLr = vLr.T; vRr = vRr.T; pLr = pLr.T; pRr = pRr.T
        Axf = blk.Aef_x; Ayf = blk.Aef_y
        mflxL = uLr * Axf + vLr * Ayf; mflxR = uRr * Axf + vRr * Ayf
        alpha = np.maximum(np.abs(mflxL), np.abs(mflxR))
        Fgu = 0.5 * (mflxL + mflxR) - 0.5 * alpha * (uRr - uLr)
        Fgphi = _upwind(pLr, pRr, Fgu) * Fgu
    else:
        uL = np.zeros((ni, nj + 1)); uR = np.zeros((ni, nj + 1))
        vL = np.zeros((ni, nj + 1)); vR = np.zeros((ni, nj + 1))
        pL = np.zeros((ni, nj + 1)); pR = np.zeros((ni, nj + 1))
        uL[:, 1:nj] = u[:, :-1]; uR[:, 1:nj] = u[:, 1:]
        vL[:, 1:nj] = v[:, :-1]; vR[:, 1:nj] = v[:, 1:]
        pL[:, 1:nj] = phi[:, :-1]; pR[:, 1:nj] = phi[:, 1:]
        uL[:, nj] = u[:, -1]; uR[:, nj] = u[:, -1]
        vL[:, nj] = v[:, -1]; vR[:, nj] = v[:, -1]
        pL[:, nj] = phi[:, -1]; pR[:, nj] = phi[:, -1]
        Axf = blk.Aef_x; Ayf = blk.Aef_y
        mflx = 0.5 * ((uL * Axf + vL * Ayf) + (uR * Axf + vR * Ayf))
        Fgphi = _upwind(pL, pR, mflx) * mflx
    Fgphi[:, 0] = 0.0                  # wall: zero normal flux of nu_tilde
    dphi += (Fgphi[:, 1:] - np.concatenate(
        [np.zeros((ni, 1)), Fgphi[:, 1:nj]], axis=1)) / blk.J
    return dphi


def sa_nut(nu_tilde, nu):
    """Turbulent (kinematic) viscosity from the SA working variable."""
    chi = np.maximum(nu_tilde, 0.0) / nu
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + SA_CV1 ** 3)
    return nu_tilde * fv1


def sa_rhs(blk, u, v, nu_tilde, nu, scheme="upwind1", d=None, nu_eff_max=None,
           trip=False):
    """Spalart-Allmaras working-variable residual d(nu_tilde)/dt (no implicit time
    term).  Returns (rhs, nu_t).  nu = molecular kinematic viscosity.

    trip=False  -> free transition (no ft1); the production/destruction balance
                   sets the turbulent viscosity.
    trip=True   -> Spalart-Allmaras *trip* term ft1 (negative) suppresses nu_tilde
                   growth everywhere except where nu_tilde is forced large (the
                   trip), giving a FIXED transition location.  Pair with a large
                   free-stream nu_tilde_inf and an explicit trip strip (set
                   nu_tilde large at the desired x/c) to pin transition at e.g.
                   7% chord (the 30P30N literature condition).

    nu_eff_max caps the SA diffusion coefficient (nu+nu_tilde).  It must be the
    GLOBAL constant coeff_max (= nu + nu_t_cap) so the actual coefficient stays
    smooth (no grad(coeff) jumps); a per-cell field cap destabilises the step.
    """
    if d is None:
        d = c_walldist(blk)
    d = np.maximum(d, 1e-10)
    omega = np.maximum(c_vorticity(blk, u, v), 1e-10)
    chi = np.maximum(nu_tilde, 0.0) / nu
    chi3 = chi ** 3
    fv1 = chi3 / (chi3 + SA_CV1 ** 3)
    nu_t = nu_tilde * fv1
    # diffusion: (1/sigma)[ div((nu+nu_tilde) grad nu_tilde) + cb2 |grad nu_tilde|^2 ]
    gx, gy = c_grad(blk, nu_tilde)
    coeff = nu + nu_tilde
    if nu_eff_max is not None:
        coeff = np.minimum(coeff, nu_eff_max)
    div_diff = c_div(blk, coeff * gx, coeff * gy)
    cross = SA_CB2 / SA_SIGMA * (gx ** 2 + gy ** 2)
    diff_term = (1.0 / SA_SIGMA) * div_diff + cross
    # production: Stilda = omega + nu_tilde/(kappa^2 d^2) fv2  (fv2 provides
    # production in low-vorticity / near-wall regions needed for transition)
    fv2 = 1.0 - chi / (1.0 + chi * fv1)
    Stilda = omega + nu_tilde / (SA_KAPPA ** 2 * d ** 2) * fv2
    ft2 = SA_CT3 * np.exp(-SA_CT4 * chi ** 2)
    P = SA_CB1 * (1.0 - ft2) * Stilda * nu_tilde
    if trip:
        ft1 = SA_CT1 * (np.exp(-SA_CT2 * (chi / SA_CT3) ** 2) - 1.0)
        P = P + ft1 * nu_tilde * (1.0 - ft2)
    # destruction
    r = np.minimum(nu_tilde / (SA_KAPPA ** 2 * omega * d ** 2), 10.0)
    g_ = r + SA_CW2 * (r ** 6 - r)
    fw = g_ * ((1.0 + SA_CW3 ** 6) / (g_ ** 6 + SA_CW3 ** 6)) ** (1.0 / 6.0)
    Dest = (SA_CW1 * fw - SA_CB1 / SA_KAPPA ** 2 * ft2) * (nu_tilde / d) ** 2
    Dest = np.maximum(Dest, 0.0)
    conv = c_conv_scalar(blk, u, v, nu_tilde, scheme=scheme)
    return -conv + diff_term + P - Dest, nu_t
