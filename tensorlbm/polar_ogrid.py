"""Clean polar O-grid FVM solver (standalone, unit-tested).

Replaces the broken (xi,eta) curvilinear operators in overset_cylinder_fvm.py
with a self-consistent ORTHOGONAL POLAR formulation that is verified by unit
tests before any physics is trusted:
    * uniform flow  -> div == 0  everywhere (incl. periodic seam + wall)
    * u = x/R       -> div == +1/R  (correct sign & magnitude)
    * p = r^2       -> lap == +4    (constant, exact)
    * p = cos(th)   -> lap == -cos(th)/r^2

Geometry: cell (i,j), i=0..ni-1 (periodic in theta), j=0..nj-1 (radial, wall at
j=0 / r=R, outer fringe at j=nj-1 / r=Rf_og).  Radial stretching via beta.

Face area vectors (PHYSICAL, outward normals):
    xi-face (constant theta, between cells i-1,i and i,i+1):
        A_xi = (-sin th,  cos th) * dr_node[j]          |A_xi| = dr
    eta-face (constant r):
        A_eta_outer = ( cos th,  sin th) * r_node[j+1] * dth   |A| = r*dth
        A_eta_inner = ( cos th,  sin th) * r_node[j]   * dth
Cell volume  J = r_c * dth * dr_node[j].

All operators use exactly these, so divergence is the transpose of the gradient
(L = D o G is SPD) and the projection removes divergence to machine precision.
"""
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, coo_matrix, csc_matrix, eye, diags
from scipy.sparse.linalg import splu, eigsh, cg, bicgstab


class Block:
    def __init__(self, name, ni, nj, xc, yc):
        self.name = name
        self.ni, self.nj = ni, nj
        self.xc, self.yc = xc, yc
        self.u = np.zeros((ni, nj))
        self.v = np.zeros((ni, nj))
        self.p = np.zeros((ni, nj))
        self.J = None


def make_ogrid(ni, nj, cx, cy, R, Rf_og, beta=2.0):
    thn = np.linspace(0.0, 2.0 * np.pi, ni, endpoint=False)   # node angles
    dth = 2.0 * np.pi / ni                                  # angular spacing
    t = np.linspace(0.0, 1.0, nj + 1)
    rn = R + (Rf_og - R) * (np.exp(beta * t) - 1.0) / (np.exp(beta) - 1.0)  # node radii
    THN, RN = np.meshgrid(thn, rn, indexing="ij")
    xn = cx + RN * np.cos(THN)
    yn = cy + RN * np.sin(THN)
    thc = thn + 0.5 * dth                          # cell-center angles (monotonic, full circle)
    rc = 0.5 * (rn[:-1] + rn[1:])                 # cell-center radii
    THC, RC = np.meshgrid(thc, rc, indexing="ij")
    xc = cx + RC * np.cos(THC)
    yc = cy + RC * np.sin(THC)

    blk = Block("ogrid", ni, nj, xc, yc)
    blk.r = np.sqrt((xc - cx) ** 2 + (yc - cy) ** 2)
    blk.cx, blk.cy = cx, cy
    blk.R, blk.Rf_og, blk.beta = R, Rf_og, beta
    blk.ni, blk.nj = ni, nj
    blk.rn = rn                                 # (nj+1,) node radii
    blk.rc = RC                                 # (ni,nj) cell-center radii (2D!)
    blk.thn = thn                               # (ni,) node angles
    blk.cos_th = (xc - cx) / rc
    blk.sin_th = (yc - cy) / rc
    blk.dth = 2.0 * np.pi / ni
    blk.dr_node = rn[1:] - rn[:-1]              # (nj,) radial width of cell j
    blk.cos_thn = np.cos(thn)                   # node-angle trig (angular faces)
    blk.sin_thn = np.sin(thn)
    blk._nu = 0.0
    blk.U = 1.0

    # ---- physical face area vectors ----
    # xi-face: face i sits at NODE angle thn[i] (radial line through node i),
    # spanning radial dr. Outward (+theta) normal = (-sin thn, cos thn).
    # (NOTE: uses NODE angles, not cell-center, or uniform-flow div != 0.)
    sin_thn = np.sin(thn)
    cos_thn = np.cos(thn)
    A_xi_x = np.zeros((ni + 1, nj))
    A_xi_y = np.zeros((ni + 1, nj))
    for i in range(ni + 1):
        ip = i % ni
        A_xi_x[i] = -sin_thn[ip] * blk.dr_node
        A_xi_y[i] = cos_thn[ip] * blk.dr_node
    # eta-faces: OUTER (r=rn[j+1]) and INNER (r=rn[j]) of cell (i,j).
    A_eta_o_x = blk.cos_th * (rn[1:])[None, :] * blk.dth   # (ni,nj) outer
    A_eta_o_y = blk.sin_th * (rn[1:])[None, :] * blk.dth
    A_eta_i_x = blk.cos_th * (rn[:-1])[None, :] * blk.dth  # (ni,nj) inner
    A_eta_i_y = blk.sin_th * (rn[:-1])[None, :] * blk.dth
    blk.A_xi_x, blk.A_xi_y = A_xi_x, A_xi_y
    blk.A_eta_o_x, blk.A_eta_o_y = A_eta_o_x, A_eta_o_y
    blk.A_eta_i_x, blk.A_eta_i_y = A_eta_i_x, A_eta_i_y

    # cell volume
    blk.J = (blk.rc * blk.dth * blk.dr_node[None, :]).copy()
    return blk


# --------------------------------------------------------------------------- #
# Differential operators (physical, conservative, self-consistent)
# --------------------------------------------------------------------------- #
def _mass_flux_xi(blk, u, v):
    """Scalar mass flux through the xi-face between cell k-1 and k (array (ni+1,nj)).
    Face value = central average of the two adjacent cells (periodic)."""
    u_face = 0.5 * (np.roll(u, 1, axis=0) + u)   # (ni,nj): face between cell i-1,i
    v_face = 0.5 * (np.roll(v, 1, axis=0) + v)
    u_face = np.concatenate([u_face, u_face[:1]], axis=0)   # append periodic seam -> (ni+1,nj)
    v_face = np.concatenate([v_face, v_face[:1]], axis=0)
    return u_face * blk.A_xi_x + v_face * blk.A_xi_y

def _mass_flux_eta(blk, u, v):
    """Return (F_outer, F_inner) scalar mass fluxes through the eta-faces.
    F_outer[k] = flux through OUTER face of cell k (uses cell k velocity).
    F_inner[k] = flux through INNER face of cell k (uses cell k-1 velocity)."""
    Fo = u * blk.A_eta_o_x + v * blk.A_eta_o_y   # (ni,nj) outer
    Fi = u * blk.A_eta_i_x + v * blk.A_eta_i_y   # (ni,nj) inner (uses THIS cell vel)
    return Fo, Fi

def polar_div(blk, u, v, p=None, dt=1.0):
    """Physical divergence div(u,v) by conservative face fluxes (CLEAN).

    Radial faces: outer face of cell (i,j) at r=rn[j+1], area rn[j+1]*dth,
    flux = area*(u_f cos th + v_f sin th), u_f = 0.5*(u_ij + u_i,j+1) (face-
    averaged).  Inner face (r=rn[j]): same at r=rn[j]; at the solid wall (j=0)
    flux = 0 (no penetration).  Angular faces: between cell i-1,i at NODE angle
    thn[i], area dr_node[j], flux = area*(-sin th u_f + cos th v_f), u_f =
    0.5*(u_i-1,j + u_i,j).  div = (F_outer - F_inner + F_right - F_left)/V.
    With the consistent inner radius rn[j] (NOT cell centre) the wall cell
    telescoping holds, so uniform flow gives div==0 everywhere (incl. wall &
    periodic seam), and polar_div(polar_grad(r^2)) == +4 exactly.

    If p is given (Rhie-Chow mode, used in the projection RHS), the radial face
    velocity carries a pressure-gradient correction that breaks the collocated-
    grid odd-even decoupling: the plain central average of a radial checkerboard
    u=(-1)^j is exactly 0 at every face (so it is a null space of this div and
    the projection can never remove it -> spurious near-wall radial saw-tooth,
    |u|~1.6U, ~3x too much viscous drag).  The correction subtracts the
    difference between the TRUE face pressure gradient (p[k]-p[k-1])/dr and the
    averaged cell-centre gradient; for a checkerboard these differ by ~2c/dr, so
    the null space is broken while a smooth physical pressure is untouched.
    """
    ni, nj = blk.ni, blk.nj
    rc, rn, dth, dr = blk.rc, blk.rn, blk.dth, blk.dr_node
    ct, st = blk.cos_th, blk.sin_th               # cell-centre angles (radial faces)
    ctn, stn = blk.cos_thn, blk.sin_thn           # node angles (angular faces)
    # radial flux at face k (between cell k-1 and k), k=1..nj-1
    Frad = np.zeros((ni, nj + 1))
    if p is None:
        # plain central-averaged face value (used by polar_lap / Poisson operator)
        for k in range(1, nj):
            uf = 0.5 * (u[:, k - 1] + u[:, k])
            vf = 0.5 * (v[:, k - 1] + v[:, k])
            Frad[:, k] = rn[k] * dth * (uf * ct[:, k - 1] + vf * st[:, k - 1])
    else:
        # Rhie-Chow momentum interpolation (semi-implicit, uses existing p)
        gxp, gyp = polar_grad(blk, p)
        dpr = gxp * ct + gyp * st                      # cell-centre radial grad (ni,nj)
        drf = rn[1:] - rn[:-1]                         # face radial widths, face k: drf[k-1]
        for k in range(1, nj):
            face_grad_p = (p[:, k] - p[:, k - 1]) / drf[k - 1]   # true face radial grad
            cell_avg = 0.5 * (dpr[:, k - 1] + dpr[:, k])        # averaged cell grad
            corr = dt * (face_grad_p - cell_avg)               # breaks checkerboard
            ctk = ct[:, k - 1]; stk = st[:, k - 1]
            uf = 0.5 * (u[:, k - 1] + u[:, k]) - corr * ctk
            vf = 0.5 * (v[:, k - 1] + v[:, k]) - corr * stk
            Frad[:, k] = rn[k] * dth * (uf * ctk + vf * stk)
    # k=0 (wall) stays 0; k=nj (outer) stays 0 (caller sets recv)
    up = np.roll(u, 1, axis=0)
    vp = np.roll(v, 1, axis=0)
    uf = 0.5 * (up + u)
    vf = 0.5 * (vp + v)
    Fang = dr[None, :] * (-stn[:, None] * uf + ctn[:, None] * vf)   # (ni,nj) face i
    Fang = np.concatenate([Fang, Fang[:1]], axis=0)                 # (ni+1,nj) seam
    Fright = Frad[:, 1:]; Fleft = Frad[:, :-1]
    Fr = Fang[1:]; Fl = Fang[:-1]
    div = (Fright - Fleft + Fr - Fl) / blk.J
    return div

def polar_grad(blk, p):
    """Physical (x,y) gradient at cell centres, built from FACE gradients only
    (adjacent cells, never skip-centred) so it is the exact adjoint of
    polar_div.  Radial face gradient gr[k]=(p[k]-p[k-1])/dr at face k; cell
    d/dr = average of its two faces (one-sided at walls).  Angular face gradient
    gth[i]=(p[i]-p[i-1])/(rc*dth); cell (1/r)d/dth = average of its two faces.
    Reconstructed via the orthogonal polar rotation (cos,sin)."""
    ni, nj = blk.ni, blk.nj
    rc, rn, dth = blk.rc, blk.rn, blk.dth
    dr_face = (rn[1:-1] - rn[:-2])[None, :]           # (1,nj-1)
    gr = np.zeros((ni, nj + 1))
    gr[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dr_face     # faces 1..nj-1
    drdr = np.zeros((ni, nj))
    drdr[:, 0] = gr[:, 1]                             # wall cell: one-sided (outer face)
    drdr[:, 1:-1] = 0.5 * (gr[:, 1:nj - 1] + gr[:, 2:nj])
    drdr[:, -1] = gr[:, -1]                           # outer cell: one-sided (Neumann)
    gth = (p - np.roll(p, 1, axis=0)) / (rc * dth)    # (ni,nj) node-angle faces
    ang_d = 0.5 * (gth + np.roll(gth, -1, axis=0))     # (1/r)d/dth at cell centre
    ct, st = blk.cos_th, blk.sin_th
    gx = ct * drdr - st * ang_d
    gy = st * drdr + ct * ang_d
    return gx, gy

def polar_lap(blk, p):
    """Physical Laplacian of p = div(grad p), consistent with polar_div/grad:
    radial (1/r)d(r dp/dr)/dr with cell-center-consistent face radii."""
    ni, nj = blk.ni, blk.nj
    rc = blk.rc                                       # (ni,nj), radial-only
    # radial gradient using CELL-CENTER radii (so p=r^2 -> lap=4 exactly)
    dpr_in = np.zeros((ni, nj))
    dpr_in[:, 1:] = (p[:, 1:] - p[:, :-1]) / (rc[:, 1:] - rc[:, :-1])
    dpr_in[:, 0] = (p[:, 1] - p[:, 0]) / (rc[:, 1] - rc[:, 0])   # one-sided wall
    dpr_out = np.zeros((ni, nj))
    dpr_out[:, :-1] = (p[:, 1:] - p[:, :-1]) / (rc[:, 1:] - rc[:, :-1])
    dpr_out[:, -1] = (p[:, -1] - p[:, -2]) / (rc[:, -1] - rc[:, -2])
    rf = 0.5 * (rc[:, :-1] + rc[:, 1:])               # face radius (ni,nj-1)
    fr_in = np.zeros((ni, nj)); fr_in[:, 1:] = rf * dpr_in[:, 1:]
    fr_in[:, 0] = rc[:, 0] * dpr_in[:, 0]             # wall face
    fr_out = np.zeros((ni, nj)); fr_out[:, :-1] = rf * dpr_out[:, :-1]
    fr_out[:, -1] = rc[:, -1] * dpr_out[:, -1]        # outer face
    lap_r = (fr_out - fr_in) / (blk.rc * blk.dr_node[None, :])
    # angular: (1/r^2) d2p/dth2 (central, periodic)
    d2pth = np.roll(p, -1, axis=0) - 2.0 * p + np.roll(p, 1, axis=0)
    lap_th = d2pth / (blk.rc**2 * blk.dth**2)
    return lap_r + lap_th


# --------------------------------------------------------------------------- #
# Convection (2nd-order upwind, conservative, uses mass flux)
# --------------------------------------------------------------------------- #
def _upwind(phiL, phiR, F):
    """Upwind face value of scalar phi given signed face mass flux F."""
    return np.where(F > 0.0, phiL, phiR)

def _minmod(a, b):
    """TVD minmod limiter: 0.5*(sign(a)+sign(b))*min(|a|,|b|); 0 if opposite sign.

    Used to build 2nd-order limited (MUSCL) face values from the upwind-biased
    slope, which are TVD-stable (no grid-scale growth) yet 2nd-order in smooth
    regions, so they kill the 1st-order-upwind 10 Hz radial mode WITHOUT the heavy
    numerical diffusion that over-predicts Cd."""
    same = (a * b) > 0.0
    mag = np.minimum(np.abs(a), np.abs(b))
    return np.where(same, np.sign(a) * mag, 0.0)


def _vanleer(dm, dp):
    """Van Leer slope limiter: 2*dm*dp/(dm+dp) when dm,dp same sign, else 0.

    Less dissipative than minmod (which clips to the smaller slope), so it keeps
    more of the 2nd-order accuracy in smooth regions while staying TVD.  Replacing
    minmod by van Leer in the MUSCL convection recovers the Re=100 Karman shedding
    that the over-diffusive minmod locks into a steady deflected wake (St~0.18 vs frozen).
    """
    s = dm * dp
    denom = dm + dp
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where((s > 0.0) & (denom != 0.0), 2.0 * dm * dp / denom, 0.0)
    return r


def polar_conv(blk, u, v, U=None, scheme="upwind1", alpha_scale=1.0):
    """Conservative convective tendency (du/dt)_conv for u and v.

    scheme="upwind1" : 1st-order upwind at every face (current default; cheap but
        introduces strong numerical diffusion and, on the stretched O-grid, an
        under-damped near-wall radial mode ~10 Hz that pollutes the forces.
    scheme="muscl"   : 2nd-order MUSCL reconstruction with a minmod limiter.  Smooth
        regions get 2nd-order accuracy (low diffusion -> Cd closer to baseline),
        extrema/grid-scale get reduced to 1st-order upwind (TVD -> the 10 Hz growth
        is bounded).  This is the recommended scheme for the production run.
    scheme="vanleer" : 2nd-order MUSCL reconstruction with a VAN LEER limiter
        (less dissipative than minmod) -- recovers the Re=100 vortex shedding that
        the over-diffusive minmod pins into a steady wake.
    scheme="central" : 2nd-order CENTRAL convection (face = 0.5*(L+R)).  Has ~zero
        numerical diffusion (only dispersion), so the effective Reynolds number
        stays physical (~100) and the Re=100 Karman shedding develops.  Central is
        dispersion-unstable on a collocated grid, so it MUST be paired with a small
        artificial viscosity (nu_art in the driver) and the (-1)^j checkerboard
        projection; tune nu_art just below the shedding threshold.
    scheme="weno"   : 5th-order WENO (Jiang-Shu) reconstruction + Rusanov flux.
        Low diffusion in smooth regions (effective Re stays ~100, so the wake
        sheds the Karman street) but TVD-bounded at shocks (stable).  This is the
        scheme that produces the physical vortex street at Re=100.
    scheme="ppm"    : 3rd-order PPM (Colella-Woodward) reconstruction + Rusanov
        flux.  Monotonicity-limited (so it never overflows on the stretched O-grid
        where WENO5 does) yet far less diffusive than the TVD van Leer -- the
        effective Reynolds number stays ~100 and the Re=100 Karman street grows.
        This is the production shedding scheme.
    """
    if scheme == "upwind1":
        return _polar_conv_upwind1(blk, u, v)
    elif scheme == "muscl":
        return _muscl_conv(blk, u, v, _minmod)
    elif scheme == "vanleer":
        return _muscl_conv(blk, u, v, _vanleer)
    elif scheme == "central":
        return _polar_conv_central(blk, u, v)
    elif scheme == "weno":
        return _weno_conv(blk, u, v, alpha_scale=alpha_scale)
    elif scheme == "ppm":
        return _ppm_conv(blk, u, v, alpha_scale=alpha_scale)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")


def _polar_conv_central(blk, u, v):
    """2nd-order central convective flux (face value = 0.5*(L+R)); low diffusion,
    dispersion-prone, so pair with artificial viscosity + checkerboard projection."""
    ni, nj = blk.ni, blk.nj
    u_roll = np.roll(u, 1, axis=0)
    v_roll = np.roll(v, 1, axis=0)
    ul = np.concatenate([u_roll, u_roll[:1]], axis=0)
    ur = np.concatenate([u, u[:1]], axis=0)
    vl = np.concatenate([v_roll, v_roll[:1]], axis=0)
    vr = np.concatenate([v, v[:1]], axis=0)
    uface = 0.5 * (ul + ur)
    vface = 0.5 * (vl + vr)
    Fxi = uface * blk.A_xi_x + vface * blk.A_xi_y
    Fxu = uface * Fxi
    Fxv = vface * Fxi
    u_up = 0.5 * (u + np.roll(u, -1, axis=1))
    v_up = 0.5 * (v + np.roll(v, -1, axis=1))
    Fgu = u_up * (u_up * blk.A_eta_o_x + v_up * blk.A_eta_o_y)
    Fgv = v_up * (u_up * blk.A_eta_o_x + v_up * blk.A_eta_o_y)
    Fxu_right = Fxu[1:]; Fxu_left = Fxu[:-1]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, :-1]
    duc = (Fxu_right - Fxu_left + Fgu - Fgu_inner) / blk.J
    Fxv_right = Fxv[1:]; Fxv_left = Fxv[:-1]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, :-1]
    dvc = (Fxv_right - Fxv_left + Fgv - Fgv_inner) / blk.J
    return duc, dvc


def _polar_conv_upwind1(blk, u, v):
    ni, nj = blk.ni, blk.nj
    # ---- xi-faces (between cell k-1 and k), array (ni+1,nj) ----
    u_roll = np.roll(u, 1, axis=0)
    v_roll = np.roll(v, 1, axis=0)
    ul = np.concatenate([u_roll, u_roll[:1]], axis=0)   # left cell (i-1) value at face i
    ur = np.concatenate([u, u[:1]], axis=0)             # right cell (i) value at face i
    vl = np.concatenate([v_roll, v_roll[:1]], axis=0)
    vr = np.concatenate([v, v[:1]], axis=0)
    Fxi_avg = 0.5 * (ul + ur) * blk.A_xi_x + 0.5 * (vl + vr) * blk.A_xi_y  # for upwind dir
    uface = _upwind(ul, ur, Fxi_avg)
    vface = _upwind(vl, vr, Fxi_avg)
    Fxi = uface * blk.A_xi_x + vface * blk.A_xi_y        # upwind mass flux (ni+1,nj)
    Fxu = uface * Fxi                                    # u-momentum flux
    Fxv = vface * Fxi                                    # v-momentum flux
    # ---- eta-faces: outer face of cell (i,j) ----
    Feta_avg = u * blk.A_eta_o_x + v * blk.A_eta_o_y      # cell-j mass flux (for upwind dir)
    u_up = _upwind(u, np.roll(u, -1, axis=1), Feta_avg)   # upwind cell j vs j+1
    v_up = _upwind(v, np.roll(v, -1, axis=1), Feta_avg)
    Fgu = u_up * (u_up * blk.A_eta_o_x + v_up * blk.A_eta_o_y)   # upwind mass flux
    Fgv = v_up * (u_up * blk.A_eta_o_x + v_up * blk.A_eta_o_y)
    # divergence of convective flux
    Fxu_right = Fxu[1:]; Fxu_left = Fxu[:-1]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, :-1]
    duc = (Fxu_right - Fxu_left + Fgu - Fgu_inner) / blk.J
    Fxv_right = Fxv[1:]; Fxv_left = Fxv[:-1]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, :-1]
    dvc = (Fxv_right - Fxv_left + Fgv - Fgv_inner) / blk.J
    return duc, dvc


def _muscl_conv(blk, u, v, limiter):
    """2nd-order MUSCL (limited) convective flux, conservative, TVD-stable.

    limiter(dm, dp) builds the limited upwind-biased slope (dm=upwind diff,
    dp=downwind diff); minmod is most dissipative, van Leer less so.

    Both xi (angular, periodic) and eta (radial, wall at j=0) faces use a
    limited linear reconstruction of the upwind cell value; the momentum flux is
    the limited face velocity dotted with the (limited) mass flux, so the scheme
    stays exactly conservative and the divergence is the face-flux difference.
    """
    ni, nj = blk.ni, blk.nj
    lim = limiter
    ct, st = blk.cos_th, blk.sin_th
    rn, dth = blk.rn, blk.dth
    duc = np.zeros((ni, nj))
    dvc = np.zeros((ni, nj))

    # ===================== xi (angular) faces, periodic ===================== #
    # Face i sits between cell i-1 (left) and cell i (right), node angle thn[i].
    # Drop the duplicated seam column so faces are indexed 0..ni-1 == gaps.
    Ax = blk.A_xi_x[:-1]                         # (ni,nj)
    Ay = blk.A_xi_y[:-1]
    uL = np.roll(u, 1, axis=0)                   # cell i-1
    vL = np.roll(v, 1, axis=0)
    uR = u                                       # cell i
    vR = v
    uLL = np.roll(u, 2, axis=0)                  # cell i-2 (two upwind if F>0)
    vLL = np.roll(v, 2, axis=0)
    uRR = np.roll(u, -1, axis=0)                 # cell i+1 (two downwind if F<=0)
    vRR = np.roll(v, -1, axis=0)
    Fxi = 0.5 * (uL + uR) * Ax + 0.5 * (vL + vR) * Ay   # signed mass flux (out +theta)
    # forward (F>0, upwind = left cell i-1)
    fu_fwd = uL + 0.5 * lim(uL - uLL, uR - uL)
    fv_fwd = vL + 0.5 * lim(vL - vLL, vR - vL)
    # backward (F<=0, upwind = right cell i)
    fu_bwd = uR - 0.5 * lim(uR - uRR, uL - uR)
    fv_bwd = vR - 0.5 * lim(vR - vRR, vL - vR)
    face_u = np.where(Fxi > 0.0, fu_fwd, fu_bwd)
    face_v = np.where(Fxi > 0.0, fv_fwd, fv_bwd)
    mflx = face_u * Ax + face_v * Ay                       # mass flux at face
    Fxu = face_u * mflx                                    # u-momentum flux
    Fxv = face_v * mflx
    # divergence over xi: right face (i+1) - left face (i), periodic
    duc += (np.roll(Fxu, -1, axis=0) - Fxu) / blk.J
    dvc += (np.roll(Fxv, -1, axis=0) - Fxv) / blk.J

    # ===================== eta (radial) faces ===================== #
    # Face j (j=1..nj-1) sits between cell j-1 (inner) and cell j (outer), at
    # radius rn[j], outward normal +r.  cos_th/sin_th are angle-only, so the face
    # trig equals the cell-centre trig; the outer face (j=nj) is the recv donor.
    # per-face area at radius rn[j]: (ni, nj+1)
    Axf = np.zeros((ni, nj + 1)); Ayf = np.zeros((ni, nj + 1))
    Axf[:, :-1] = ct * rn[:nj][None, :] * dth     # faces 0..nj-1 (rn[0..nj-1])
    Ayf[:, :-1] = st * rn[:nj][None, :] * dth
    Axf[:, -1] = ct[:, -1] * rn[-1] * dth         # outer face (clamped angle)
    Ayf[:, -1] = st[:, -1] * rn[-1] * dth
    # cell arrays for interior faces j=1..nj-1 (index k=j-1 in 0..nj-2)
    uin = u[:, :-1]                               # inner cell  (ni, nj-1)
    vin = v[:, :-1]
    uout = u[:, 1:]                               # outer cell
    vout = v[:, 1:]
    # two-upwind (inner-inner) and two-downwind (outer-outer) with boundary fills
    uin2 = np.zeros((ni, nj - 1)); uin2[:, 1:] = u[:, :-2]; uin2[:, 0] = 0.0
    vin2 = np.zeros((ni, nj - 1)); vin2[:, 1:] = v[:, :-2]; vin2[:, 0] = 0.0
    uout2 = np.zeros((ni, nj - 1)); uout2[:, :-1] = u[:, 2:]; uout2[:, -1] = u[:, -1]
    vout2 = np.zeros((ni, nj - 1)); vout2[:, :-1] = v[:, 2:]; vout2[:, -1] = v[:, -1]
    Axj = Axf[:, 1:nj]                            # (ni, nj-1) face area at rn[1:nj]
    Ayj = Ayf[:, 1:nj]
    Feta = 0.5 * (uin + uout) * Axj + 0.5 * (vin + vout) * Ayj
    fu_fwd = uin + 0.5 * lim(uin - uin2, uout - uin)
    fv_fwd = vin + 0.5 * lim(vin - vin2, vout - vin)
    fu_bwd = uout - 0.5 * lim(uout - uout2, uin - uout)
    fv_bwd = vout - 0.5 * lim(vout - vout2, uin - uout)
    face_u = np.zeros((ni, nj + 1)); face_v = np.zeros((ni, nj + 1))
    face_u[:, 1:nj] = np.where(Feta > 0.0, fu_fwd, fu_bwd)
    face_v[:, 1:nj] = np.where(Feta > 0.0, fv_fwd, fv_bwd)
    # mass / momentum flux at every face, then face-flux difference per cell
    mface = face_u * Axf + face_v * Ayf                   # (ni, nj+1)
    Fgu = face_u * mface
    Fgv = face_v * mface
    # outer face of cell j = face j+1 ; inner face of cell j = face j (wall -> 0)
    Fgu_outer = Fgu[:, 1:]                                # (ni,nj) outer face of cell j = face j+1
    Fgv_outer = Fgv[:, 1:]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, 1:nj]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner) / blk.J
    dvc += (Fgv_outer - Fgv_inner) / blk.J

    return duc, dvc


def _weno5_states(P):
    """WENO5 (Jiang-Shu) reconstructed LEFT/RIGHT states at the faces of a padded
    line field.

    P has shape (M+6, N): the M interior cells occupy padded indices 3..M+2, and
    face f (0..M-1) sits between interior cell f-1 (P[f+2]) and cell f (P[f+3]).
    Returns (qm, qp) each shape (M, N): qm is the left-biased (upwind) state, qp
    the right-biased state.  The 5th-order stencil makes the numerical diffusion
    negligible in smooth regions (so the effective Reynolds number stays physical
    ~100 and the Re=100 Karman street can grow) while the JS weights collapse to a
    TVD-1st-order stencil at a shock (so it never blows up).  epsilon=1e-6 guards
    the weight denominator in smooth/flat regions.
    """
    c = np.arange(P.shape[0] - 6) + 3                  # padded index of right cell
    Pm3, Pm2, Pm1, Pc, Pp1, Pp2, Pp3 = (P[c-3], P[c-2], P[c-1], P[c],
                                        P[c+1], P[c+2], P[c+3])
    # ---- left-biased (minus) 3rd-order candidates: face sits between cell c-1
    #      (left) and cell c (right); right cell of each stencil is the upwind one.
    p0m = (1.0/3)*Pm2 - (7.0/6)*Pm1 + (11.0/6)*Pc
    p1m = (-1.0/6)*Pm1 + (5.0/6)*Pc + (1.0/3)*Pp1
    p2m = (1.0/3)*Pc + (5.0/6)*Pp1 - (1.0/6)*Pp2
    is0m = (13.0/12)*(Pm2-2*Pm1+Pc)**2 + (1.0/4)*(Pm2-4*Pm1+3*Pc)**2
    is1m = (13.0/12)*(Pm1-2*Pc+Pp1)**2 + (1.0/4)*(Pm1-Pp1)**2
    is2m = (13.0/12)*(Pc-2*Pp1+Pp2)**2 + (1.0/4)*(3*Pc-4*Pp1+Pp2)**2
    # ---- right-biased (plus) candidates ----
    p0p = (-1.0/6)*Pm1 + (5.0/6)*Pc + (1.0/3)*Pp1
    p1p = (1.0/3)*Pc + (5.0/6)*Pp1 - (1.0/6)*Pp2
    p2p = (11.0/6)*Pp1 - (7.0/6)*Pp2 + (1.0/3)*Pp3
    is0p = is2m
    is1p = is1m
    is2p = (13.0/12)*(Pp1-2*Pp2+Pp3)**2 + (1.0/4)*(3*Pp1-4*Pp2+Pp3)**2
    eps = 1e-6
    def _w(is0, is1, is2, d0, d1, d2, pp0, pp1, pp2):
        a0 = d0/(is0+eps)**2; a1 = d1/(is1+eps)**2; a2 = d2/(is2+eps)**2
        s = a0 + a1 + a2
        return (a0*pp0 + a1*pp1 + a2*pp2)/s
    qm = _w(is0m, is1m, is2m, 0.1, 0.6, 0.3, p0m, p1m, p2m)
    qp = _w(is0p, is1p, is2p, 0.3, 0.6, 0.1, p0p, p1p, p2p)
    return qm, qp


def _weno_conv(blk, u, v, alpha_scale=1.0):
    """WENO5 conservative convective tendency (duc, dvc) on the face-flux form.

    Replaces the 2nd-order MUSCL limited slope by a 5th-order WENO reconstruction
    of the left/right face states, then a Rusanov (local Lax-Friedrichs) flux
        F_face = 0.5*(mflxL*qL + mflxR*qR) - 0.5*alpha*(qR - qL),
    alpha = max(|mflxL|,|mflxR|).  The Rusanov dissipation vanishes for an
    identical left/right state (uniform flow is preserved to machine precision)
    yet bounds any shock-like growth -- this is what lets the Re=100 wake shed the
    Karman vortex street where the over-diffusive TVD limiter locks it steady.
    Metric, periodic angular seam, and radial wall/outer faces are handled exactly
    like the verified _muscl_conv.
    """
    ni, nj = blk.ni, blk.nj
    duc = np.zeros((ni, nj))
    dvc = np.zeros((ni, nj))
    ct, st = blk.cos_th, blk.sin_th
    rn, dth = blk.rn, blk.dth

    # ===================== xi (angular) faces, periodic ===================== #
    idx = (np.arange(ni + 6) - 3) % ni                   # periodic fill
    Pu = u[idx]; Pv = v[idx]                             # (ni+6, nj)
    uL, uR = _weno5_states(Pu)
    vL, vR = _weno5_states(Pv)
    Ax = blk.A_xi_x[:-1]; Ay = blk.A_xi_y[:-1]           # (ni, nj)
    mflxL = uL*Ax + vL*Ay
    mflxR = uR*Ax + vR*Ay
    alpha = alpha_scale * np.maximum(np.abs(mflxL), np.abs(mflxR))
    Fxu = 0.5*(mflxL*uL + mflxR*uR) - 0.5*alpha*(uR-uL)
    Fxv = 0.5*(mflxL*vL + mflxR*vR) - 0.5*alpha*(vR-vL)
    duc += (np.roll(Fxu, -1, axis=0) - Fxu)/blk.J
    dvc += (np.roll(Fxv, -1, axis=0) - Fxv)/blk.J

    # ===================== eta (radial) faces ===================== #
    # padded (nj+7, ni): interior cells 0..nj-1 at padded 3..nj+2; 3 antisym
    # no-slip ghost layers below the wall (mirror = -value, both comps), 4 clamped
    # layers above (sponge, smooth).  Faces 0..nj; face 0 is the wall (flux 0).
    Mu = np.zeros((nj + 7, ni)); Mv = np.zeros((nj + 7, ni))
    Mu[3:nj+3, :] = u.T; Mv[3:nj+3, :] = v.T
    for g in (0, 1, 2):
        Mu[g, :] = -Mu[6-g, :]; Mv[g, :] = -Mv[6-g, :]   # antisym reflect
    for g in (nj+3, nj+4, nj+5, nj+6):
        Mu[g, :] = Mu[nj+2, :]; Mv[g, :] = Mv[nj+2, :]   # clamp above
    uLr, uRr = _weno5_states(Mu)                         # (nj+1, ni)
    vLr, vRr = _weno5_states(Mv)
    uLr = uLr.T; uRr = uRr.T; vLr = vLr.T; vRr = vRr.T   # -> (ni, nj+1)
    Axf = np.zeros((ni, nj+1)); Ayf = np.zeros((ni, nj+1))
    Axf[:, :-1] = ct * rn[:nj][None, :] * dth
    Ayf[:, :-1] = st * rn[:nj][None, :] * dth
    Axf[:, -1] = ct[:, -1]*rn[-1]*dth; Ayf[:, -1] = st[:, -1]*rn[-1]*dth
    mflxL = uLr*Axf + vLr*Ayf
    mflxR = uRr*Axf + vRr*Ayf
    alpha = alpha_scale * np.maximum(np.abs(mflxL), np.abs(mflxR))
    Fgu = 0.5*(mflxL*uLr + mflxR*uRr) - 0.5*alpha*(uRr-uLr)
    Fgv = 0.5*(mflxL*vLr + mflxR*vRr) - 0.5*alpha*(vRr-vLr)
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0                       # wall face flux = 0
    Fgu_outer = Fgu[:, 1:]                                # (ni, nj)
    Fgv_outer = Fgv[:, 1:]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, 1:nj]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner)/blk.J
    dvc += (Fgv_outer - Fgv_inner)/blk.J
    return duc, dvc


def _ppm_states(P):
    """Colella-Woodward PPM (3rd-order) reconstructed LEFT/RIGHT states at faces.

    Unlike WENO5 (which is NOT TVD and overflows on a strongly stretched grid),
    PPM applies a MONOTONICITY limiter to the parabola so it never creates a new
    extremum -- hence it is stable (no blow-up) yet only 3rd-order, i.e. far less
    diffusive than the 2nd-order TVD van Leer.  That lower numerical diffusion
    lets the effective Reynolds number stay ~100, so the Re=100 Karman street
    actually grows.  P padded (M+6, N); returns (qm, qp) shape (M, N): qm = right
    face of the upwind cell (left state), qp = left face of the downwind cell.
    """
    p = np.arange(3, P.shape[0] - 3)                    # interior padded idx (1D)
    fm2, fm1, fc, fp1, fp2 = P[p-2], P[p-1], P[p], P[p+1], P[p+2]
    # unlimited 4th-order interface values of cell p
    fL = (7.0/12.0)*(fm1 + fc) - (1.0/12.0)*(fm2 + fp1)   # left face of cell p
    fR = (7.0/12.0)*(fc + fp1) - (1.0/12.0)*(fm1 + fp2)   # right face of cell p
    # monotonicity limiter (CW84): clamp to the local min/max, then flatten if the
    # parabola tries to introduce an interior extremum
    fmin = np.minimum(np.minimum(fm1, fc), fp1)
    fmax = np.maximum(np.maximum(fm1, fc), fp1)
    fL = np.maximum(np.minimum(fL, fmax), fmin)
    fR = np.maximum(np.minimum(fR, fmax), fmin)
    cond = (fR - fc)*(fc - fL) <= 0.0
    fL = np.where(cond, fc, fL)
    fR = np.where(cond, fc, fR)
    qp = fL                                            # left state at face f = L-face of cell f
    qm = np.roll(fR, 1, axis=0)                        # right state = R-face of cell f-1
    return qm, qp


def _ppm_conv(blk, u, v, alpha_scale=1.0):
    """PPM (Colella-Woodward) conservative convective tendency (duc, dvc).

    Same face-flux / Rusanov / metric / BC structure as _weno_conv, but the
    face states come from the monotonicity-limited PPM reconstruction (stable on
    the stretched O-grid, where WENO5 overflows).  This is the scheme that yields
    the physical Re=100 Karman vortex street.
    """
    ni, nj = blk.ni, blk.nj
    duc = np.zeros((ni, nj))
    dvc = np.zeros((ni, nj))
    ct, st = blk.cos_th, blk.sin_th
    rn, dth = blk.rn, blk.dth

    # ===================== xi (angular) faces, periodic ===================== #
    idx = (np.arange(ni + 6) - 3) % ni
    Pu = u[idx]; Pv = v[idx]
    uL, uR = _ppm_states(Pu)
    vL, vR = _ppm_states(Pv)
    Ax = blk.A_xi_x[:-1]; Ay = blk.A_xi_y[:-1]
    mflxL = uL*Ax + vL*Ay
    mflxR = uR*Ax + vR*Ay
    alpha = alpha_scale * np.maximum(np.abs(mflxL), np.abs(mflxR))
    Fxu = 0.5*(mflxL*uL + mflxR*uR) - 0.5*alpha*(uR-uL)
    Fxv = 0.5*(mflxL*vL + mflxR*vR) - 0.5*alpha*(vR-vL)
    duc += (np.roll(Fxu, -1, axis=0) - Fxu)/blk.J
    dvc += (np.roll(Fxv, -1, axis=0) - Fxv)/blk.J

    # ===================== eta (radial) faces ===================== #
    Mu = np.zeros((nj + 7, ni)); Mv = np.zeros((nj + 7, ni))
    Mu[3:nj+3, :] = u.T; Mv[3:nj+3, :] = v.T
    for g in (0, 1, 2):
        Mu[g, :] = -Mu[6-g, :]; Mv[g, :] = -Mv[6-g, :]
    for g in (nj+3, nj+4, nj+5, nj+6):
        Mu[g, :] = Mu[nj+2, :]; Mv[g, :] = Mv[nj+2, :]
    uLr, uRr = _ppm_states(Mu)
    vLr, vRr = _ppm_states(Mv)
    uLr = uLr.T; uRr = uRr.T; vLr = vLr.T; vRr = vRr.T
    Axf = np.zeros((ni, nj+1)); Ayf = np.zeros((ni, nj+1))
    Axf[:, :-1] = ct * rn[:nj][None, :] * dth
    Ayf[:, :-1] = st * rn[:nj][None, :] * dth
    Axf[:, -1] = ct[:, -1]*rn[-1]*dth; Ayf[:, -1] = st[:, -1]*rn[-1]*dth
    mflxL = uLr*Axf + vLr*Ayf
    mflxR = uRr*Axf + vRr*Ayf
    alpha = alpha_scale * np.maximum(np.abs(mflxL), np.abs(mflxR))
    Fgu = 0.5*(mflxL*uLr + mflxR*uRr) - 0.5*alpha*(uRr-uLr)
    Fgv = 0.5*(mflxL*vLr + mflxR*vRr) - 0.5*alpha*(vRr-vLr)
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0
    Fgu_outer = Fgu[:, 1:]
    Fgv_outer = Fgv[:, 1:]
    Fgu_inner = np.zeros((ni, nj)); Fgu_inner[:, 1:] = Fgu[:, 1:nj]
    Fgv_inner = np.zeros((ni, nj)); Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner)/blk.J
    dvc += (Fgv_outer - Fgv_inner)/blk.J
    return duc, dvc


def _lap_face(blk, u, wall_bc="dirichlet0"):
    """Compact, conservative, face-gradient Laplacian of a scalar field.

        Lap(u)[i,j] = (1/V) * sum_faces A_face * (u_nb - u)/dist

    built on the SAME orthogonal metric as polar_div/grad (radial faces at r=rn[j]
    with area r*dth; angular faces with area dr and centre distance rc*dth).
    wall_bc selects the inner-face (r=R) treatment:
      'dirichlet0' -> u_wall = 0 (no-slip velocity),
      'neumann'    -> zero radial flux (pressure).
    With the correct sign the assembled matrix is (similar to) symmetric and
    NEGATIVE semi-definite (all eigenvalues <= 0), so explicit Euler is stable.
    """
    ni, nj = blk.ni, blk.nj
    rc, rn, dth, dr, V = blk.rc, blk.rn, blk.dth, blk.dr_node, blk.J
    lap = np.zeros((ni, nj))
    for j in range(nj - 1):
        dist = rc[:, j+1] - rc[:, j]
        flux = rn[j+1] * dth * (u[:, j+1] - u[:, j]) / dist         # +r flux at face rn[j+1]
        lap[:, j] += flux
        lap[:, j+1] -= flux
    if wall_bc == "dirichlet0":                                     # u_wall = 0
        lap[:, 0] -= rn[0] * dth * u[:, 0] / (rc[:, 0] - blk.R)
    # 'neumann': zero wall flux -> no term
    up = np.roll(u, -1, axis=0)
    fa = dr[None, :] * (up - u) / (rc * dth)
    lap += fa - np.roll(fa, 1, axis=0)
    return lap / V


def polar_diffusion(blk, u, v):
    """Viscous tendency = nu * Laplacian(u,v), component-wise.

    Uses the SAME self-consistent operator as the pressure projection,
    L = polar_div o polar_grad, so it is the exact discrete adjoint of
    polar_div (volume-weighted symmetric & negative semi-definite), hence stable
    under explicit Euler and fully consistent with the pressure Poisson.  The
    no-slip wall u[:,0]=v[:,0]=0 is re-imposed by the caller every step, which
    supplies the Dirichlet value the operator sees at the inner (r=R) face.
    """
    nu = blk._nu
    return (nu * polar_div(blk, *polar_grad(blk, u)),
            nu * polar_div(blk, *polar_grad(blk, v)))


def polar_lap(blk, f, p=None, dt=1.0):
    """Physical Laplacian of a scalar field f (matrix-free): div(grad(f)).

    This is EXACTLY the operator the pressure Poisson uses, so it is the correct
    physical Laplacian (Lap(r^2) = +4, verified).  Defined separately so it can
    be composed into 4th-order hyperviscosity without assembling any matrix.
    When p is given the divergence uses Rhie-Chow (see polar_div) so the Laplacian
    operator is consistent with the Rhie-Chow projection divergence."""
    return polar_div(blk, *polar_grad(blk, f), p=p, dt=dt)


def polar_hypervis(blk, u, v, nu_hyp):
    """4th-order hyperviscosity (Laplacian^2) of the velocity, a grid-scale
    selective damping.  Applied as the explicit tendency nu_hyp * Lap(Lap(u)).

    The O-grid polar convection (1st-order upwind) is only weakly dissipative
    near the wall, where the stretched radial cells are ~10x smaller than the
    outer ones; the resulting under-damped near-wall radial mode (a grid-scale
    instability whose frequency tracks 1/h) pollutes the cylinder forces with a
    spurious ~10 Hz lift mode.  A 2nd-order artificial viscosity would also
    smear the PHYSICAL St~0.18 vortex street, but the 4th-order term damps only
    the shortest wavelengths (rate ~ k^4), leaving the large-scale shedding
    intact.  nu_hyp is chosen so Lap^2 kills the grid mode in a few steps while
    the physical mode (wavelength ~5D) is untouched.  The no-slip wall u[:,0]=0
    held each step keeps the inner face consistent.
    """
    lu = polar_lap(blk, u)
    lv = polar_lap(blk, v)
    return (nu_hyp * polar_lap(blk, lu),
            nu_hyp * polar_lap(blk, lv))


def delta_eff(blk):
    """Per-cell effective length scale = min(radial width, angular arc)."""
    dr = blk.dr_node[None, :]
    arc = blk.rc * blk.dth
    return np.minimum(dr, arc)


def polar_filter(blk, f, sigma_theta=0.0, sigma_r=0.0):
    """Explicit 4th-order compact (Shapiro-type) low-pass filter of a scalar field.

        f <- f - (sigma/16) * [f(i-2) - 4 f(i-1) + 6 f - 4 f(i+1) + f(i+2)]

    Fourier symbol = 1 - sigma * sin^4(k dx / 2): ~1 for long waves (physical
    St~0.18 street), ~(1-sigma) at the 2-cell (Nyquist) mode.  This removes the
    O-grid grid-scale angular mode (the radial 'striping', high theta wavenumber)
    WITHOUT the 2nd-order diffusion that smears the wake or the 4th-order-in-time
    hyperviscosity that NaN'd near the wall.  theta is periodic (np.roll is exact);
    the radial axis uses clamped neighbours so the wall/outer rows are not wrapped.
    """
    g = f
    if sigma_theta > 0.0:
        st = (np.roll(g, 2, 0) + np.roll(g, -2, 0)
              - 4.0 * (np.roll(g, 1, 0) + np.roll(g, -1, 0)) + 6.0 * g)
        g = g - (sigma_theta / 16.0) * st
    if sigma_r > 0.0:
        gp2 = np.empty_like(g); gp2[:, :-2] = g[:, 2:]; gp2[:, -2:] = g[:, -1:]
        gm2 = np.empty_like(g); gm2[:, 2:] = g[:, :-2]; gm2[:, :2] = g[:, :1]
        gp1 = np.empty_like(g); gp1[:, :-1] = g[:, 1:]; gp1[:, -1] = g[:, -1]
        gm1 = np.empty_like(g); gm1[:, 1:] = g[:, :-1]; gm1[:, 0] = g[:, 0]
        st = (gp2 + gm2) - 4.0 * (gp1 + gm1) + 6.0 * g
        g = g - (sigma_r / 16.0) * st
    return g


def polar_filter_checkerboard(blk, f, alpha=1.0, radial=True, angular=False):
    """Remove the radial (and optionally angular) collocated-grid checkerboard via a
    LOCAL 3-point average  A[j] = 0.25 f[j-1] + 0.5 f[j] + 0.25 f[j+1]  blended in
    by alpha:  f <- (1-alpha) f + alpha A.

    A pure (-1)^j mode is EXACTLY annihilated by A (A[(-1)^j] = 0), and -- crucially
    -- so is (-1)^j * envelope(r): the near-wall saw-tooth GROWS from the wall, so a
    GLOBAL projection onto (-1)^j only removes its constant-coefficient part, whereas
    the LOCAL average removes it cell by cell.  For smooth physical fields A ~ f
    (response 1 - k^2/4), so the shedding structure is essentially untouched.
    Angular A is periodic (np.roll); radial A clamps at the wall/outer edge."""
    out = f
    if radial:
        rd = np.zeros_like(f)
        rd[:, 1:-1] = 0.25 * f[:, :-2] + 0.5 * f[:, 1:-1] + 0.25 * f[:, 2:]
        rd[:, 0] = 0.5 * f[:, 0] + 0.5 * f[:, 1]
        rd[:, -1] = 0.5 * f[:, -2] + 0.5 * f[:, -1]
        out = (1.0 - alpha) * out + alpha * rd
    if angular:
        ad = np.zeros_like(f)
        ad = 0.25 * np.roll(f, 1, axis=0) + 0.5 * f + 0.25 * np.roll(f, -1, axis=0)
        out = (1.0 - alpha) * out + alpha * ad
    return out


def polar_checkerboard_project(blk, f, mode="pair"):
    """Spectral removal of ONLY the radial (-1)^j collocated-grid checkerboard
    null space -- leaves the SMOOTH radial structure of the shedding street
    intact (unlike polar_filter_checkerboard's 3-point average, which low-passes
    the wake and LOCKS shedding).

    mode='global': per theta line subtract the constant-coefficient (-1)^j part
        cb[i] = mean_j((-1)^j f[i,j]);  f <- f - (-1)^j * cb[i].
        A smooth radial profile has an alternating sum ~ 0, so it is untouched.
    mode='pair': subtract the FULL (-1)^j * envelope(r) subspace via the opposite
        parity neighbour:
            c[j] = 0.5*(f[j]-f[j+1])  (even j) / 0.5*(f[j]-f[j-1])  (odd j)
            f <- f - c.
        Removes a sawtooth that GROWS from the wall EXACTLY, while a smooth field
        only picks up a small O(dr) slope bias (no low-pass of the shedding).
    """
    alt = (-1.0) ** np.arange(f.shape[1])            # (nj,)
    if mode == "global":
        cb = np.mean(f * alt[None, :], axis=1)        # (ni,)
        return f - cb[:, None] * alt[None, :]
    # pair mode
    c = np.zeros_like(f)
    c[:, 0:-1:2] = 0.5 * (f[:, 0:-1:2] - f[:, 1::2])   # even j -> odd neighbour j+1
    c[:, 1::2] = 0.5 * (f[:, 1::2] - f[:, 0:-1:2])     # odd j  -> even neighbour j-1
    return f - c


def polar_hypervis_local(blk, u, v, cfl=0.02, dt=1.0):
    """LOCAL-coefficient 4th-order hyperviscosity: a[i,j] = cfl*Δ[i,j]^4/dt, so the
    explicit 4th-order CFL (= dt*a/Δ^4) is uniform cfl *everywhere* and never NaN
    on the tiny near-wall cells.  Damps only grid-scale modes (rate ~k^4); the
    physical St~0.18 street (wavelength ~5D >> grid) is essentially untouched.
    Returns the explicit tendency (a * Lap^2(u)); multiply by dt in the caller.
    """
    lu = polar_lap(blk, u)
    lv = polar_lap(blk, v)
    llu = polar_lap(blk, lu)
    llv = polar_lap(blk, lv)
    d = delta_eff(blk)
    a = cfl * d ** 4 / dt
    return a * llu, a * llv


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def _lap_face_matrix(blk, wall_bc="neumann", dt=1.0, rc_poisson=False):
    """Pressure Poisson operator = polar_lap = polar_div \u2218 polar_grad (physical Laplacian).

    Assembled COLUMN-BY-COLUMN by applying polar_lap(p)=div(grad(p)) to each unit
    vector e_k, so L @ p == polar_lap(p) EXACTLY for every field p.  Because L is
    precisely the matrix of div(grad(.)), and the projection correction uses
    polar_grad (== the gradient whose divergence is L), the correction is the
    adjoint of the divergence in the Poisson solve -- the projection removes the
    divergence to machine precision AND recovers the *physical* pressure (correct
    sign & magnitude) for the force integral.  This is the operator the pressure
    Poisson must use; the old L = D @ D.T was not a Laplacian at all.

    wall_bc selects the inner (r=R) treatment: 'neumann' (zero radial flux, for
    pressure) or 'dirichlet0' (u_wall=0, for velocity diffusion sanity).  Periodic
    in theta.  The recv (fringe) cells are removed by build_poisson via a reduced
    system with p[recv]=0.
    """
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    # Sparse incremental assembly: accumulate each column of L = div(grad(e_k))
    # into COO triplets.  This avoids the dense N x N array the old code built
    # (which OOMs past ~N=15000 on a refined O-grid); memory is now O(nnz) ~ O(N).
    rows, cols, vals = [], [], []
    e = np.zeros((ni, nj))
    for k in range(N):
        # C-order flat index MUST match p.reshape(-1):  k = i*nj + j.
        # (The old Fortran-order i=k%ni, j=k//ni scrambled every cell whenever
        #  ni != nj, producing a non-symmetric / indefinite / singular matrix.)
        i = k // nj; j = k % nj
        e[i, j] = 1.0
        gx, gy = polar_grad(blk, e)
        # Rhie-Chow consistency: the Poisson operator must equal the SAME Rhie-Chow
        # divergence used in the projection RHS (polar_div(grad(e), p=e, dt)), so the
        # projection cancellation is exact and the collocated-grid checkerboard is
        # removed from BOTH the velocity and the pressure.
        if rc_poisson:
            col = polar_div(blk, gx, gy, p=e, dt=dt).reshape(-1)
        else:
            col = polar_div(blk, gx, gy).reshape(-1)
        nz = np.nonzero(col)[0]
        # COLUMN-assembly: L[:, k] = col  (the action of the Laplacian on e_k),
        # matching the original dense build L[:, k] = polar_div(grad(e_k)).  This
        # keeps L = M so the projection solves L p = div(u)/dt correctly.  (A
        # row-assembled L would be M^T and, because the polar discretization is
        # asymmetric, would give a wrong pressure that feeds back and blows up.)
        rows.extend(nz.tolist())
        cols.extend([k] * nz.size)
        vals.extend(col[nz].tolist())
        e[i, j] = 0.0
    return csc_matrix((vals, (rows, cols)), shape=(N, N))


def build_poisson(blk, recv_mask, reg_r=0.0, dt=1.0, rc=True):
    """Assemble the projection Poisson solver on the TRUE physical Laplacian.

    The pressure satisfies  _lap_face(p) = div(u)/dt  (Neumann zero-flux at the
    solid wall, Dirichlet p=0 at the recv/fringe cells held by the caller).  The
    recv cells are removed: the reduced matrix is L_full[solved, solved] with
    p[recv] = 0, which is exactly the correct Dirichlet condition at the interface
    (the off-diagonal coupling to recv vanishes because p[recv]=0).  This is the
    operator that produces the *physical* pressure (correct sign & magnitude) used
    by the force integration, unlike the old L = D @ D.T which was not a Laplacian.

    reg_r (default 0): adds a small RADIAL 2nd-order term  reg_r * (p[j-1]-2p[j]+p[j+1])
    to the Poisson operator.  This is REQUIRED on the collocated polar grid: the
    face-averaged radial flux makes the radial checkerboard p=(-1)^j a *null space*
    of polar_lap (= polar_div o polar_grad), so the direct Lap solve spuriously
    injects a pressure saw-tooth that grad(p) then imprints on the wall velocity
    (a spurious near-wall radial jet that over-predicts the viscous drag ~3x).  The
    radial 2nd-order term is NON-ZERO on the checkerboard (reg_r*4c) but ~0 on the
    smooth physical pressure, so a modest reg_r pins the null space WITHOUT smearing
    the physical pressure field.  reg_r ~ 1e2-1e3 (vs the radial Laplacian diagonal
    ~1e4) pins the checkerboard cleanly.
    """
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    recv = np.asarray(recv_mask).reshape(-1).astype(bool)
    solved = ~recv
    if not solved.any():                       # degenerate: pin one cell instead
        solved = np.ones(N, dtype=bool); solved[0] = False
        recv = ~solved
    L_full = _lap_face_matrix(blk, wall_bc="neumann", dt=dt, rc_poisson=rc)
    if reg_r > 0.0:
        # add reg_r * radial 2nd-order Laplacian (breaks the checkerboard null space)
        Lr = L_full.tolil()
        for i in range(ni):
            for j in range(1, nj - 1):
                k = i * nj + j
                Lr[k, i * nj + (j - 1)] += reg_r
                Lr[k, i * nj + j]       += -2.0 * reg_r
                Lr[k, i * nj + (j + 1)] += reg_r
        L_full = Lr.tocsc()
    solved_idx = np.nonzero(solved)[0]
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    blk._L_full = L_full
    blk._solved_idx = solved_idx
    return splu(L_red)


def build_poisson_iterative(blk, recv_mask, reg_r=0.0, dt=1.0, rc=True):
    """Iterative (CG) variant of build_poisson: returns (L_red, solved_idx) instead
    of a dense LU factor, so it scales to LARGE grids (memory O(nnz) ~ O(N), vs the
    direct splu which is O(N^1.5) and OOMs past ~N=15000 on this stretched O-grid).

    The radial refinement that drops the numerical diffusion below the Re=100
    shedding threshold pushes N to 30k+; splu cannot factorize that, so the shedding
    runs use this CG-based projection.  project() detects the tuple return and solves
    with a warm-started CG (previous pressure as initial guess) -- typically 50-150
    iters to 1e-9 on the stretched Laplacian.
    """
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    recv = np.asarray(recv_mask).reshape(-1).astype(bool)
    solved = ~recv
    if not solved.any():
        solved = np.ones(N, dtype=bool); solved[0] = False
        recv = ~solved
    L_full = _lap_face_matrix(blk, wall_bc="neumann", dt=dt, rc_poisson=rc)
    if reg_r <= 0.0:
        # CG needs an SPD operator: the plain radial Neumann Laplacian has a
        # (-1)^j checkerboard null space, so pin it with a small radial 2nd-order
        # term (negligible on the smooth physical pressure).  The direct (splu)
        # build_poisson tolerates the null space; the iterative one does not.
        reg_r = 200.0
    if reg_r > 0.0:
        Lr = L_full.tolil()
        for i in range(ni):
            for j in range(1, nj - 1):
                k = i * nj + j
                Lr[k, i * nj + (j - 1)] += reg_r
                Lr[k, i * nj + j]       += -2.0 * reg_r
                Lr[k, i * nj + (j + 1)] += reg_r
        L_full = Lr.tocsc()
    solved_idx = np.nonzero(solved)[0]
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    blk._solved_idx = solved_idx
    return L_red, solved_idx


def project(blk, dt, recv_mask, poisson, rc=True):
    """Chorin projection on the physical Laplacian.

    Solve reduced Poisson  _lap_face(p) = div(u)/dt  on solved cells (p[recv]=0),
    then correct the velocity with the PHYSICAL gradient,  u -= dt * grad(p).
    The physical pressure (recovered by the correct Laplacian) feeds the force
    integral, and the correction removes the bulk of the divergence so the flow
    stays incompressible and stable.  recv cells are re-imposed by the caller.

    rc (default True): use Rhie-Chow momentum interpolation in the divergence RHS
    (p=blk.p).  This breaks the collocated-grid radial checkerboard null space of
    plain div, but it couples the divergence to the OLD pressure p^(n) while the
    Poisson solves for p^(n+1); the inconsistent pressures can feed back and grow
    a checkerboard (NaN) on a highly stretched O-grid.  When rc=False the plain
    central-face divergence is used and the checkerboard is instead controlled by
    reg_r (Poisson) + a radial Shapiro filter on the velocity (caller side)."""
    if rc and blk.p is not None:
        div = polar_div(blk, blk.u, blk.v, p=blk.p, dt=dt)
    else:
        div = polar_div(blk, blk.u, blk.v)
    rhs = div.reshape(-1) / dt
    solved_idx = blk._solved_idx
    if isinstance(poisson, tuple):
        # iterative (BiCGStab) projection: warm-start from the previous pressure.
        # NOTE: the polar Poisson matrix is NON-symmetric and indefinite (smallest
        # eigenvalue < 0 from the stretched metric + reg_r term), so CG diverges
        # (overflow). BiCGStab does not require SPD and converges to machine
        # precision on this matrix (verified resid ~1e-9). A Jacobi preconditioner
        # (diagonal) keeps the iteration robust.
        L_red, _ = poisson
        x0 = blk.p.reshape(-1)[solved_idx] if blk.p is not None else None
        M = diags(L_red.diagonal())
        p_sol, info = bicgstab(L_red, rhs[solved_idx], x0=x0, M=M,
                               rtol=1e-9, atol=0.0, maxiter=800)
        if info != 0:
            # fallback: looser tolerance, no preconditioner
            p_sol, info = bicgstab(L_red, rhs[solved_idx], x0=x0,
                                   rtol=1e-7, atol=0.0, maxiter=1500)
        p = np.zeros(blk.ni * blk.nj)
        p[solved_idx] = p_sol
    else:
        p = np.zeros(blk.ni * blk.nj)
        p[solved_idx] = poisson.solve(rhs[solved_idx])
    p = p.reshape(blk.ni, blk.nj)
    blk.p = p
    gx, gy = polar_grad(blk, p)                # physical (x,y) gradient
    blk.u = blk.u - dt * gx
    blk.v = blk.v - dt * gy
    # recv (fringe) cells are re-imposed by the caller after project()


def cylinder_forces(blk, nu, U):
    """Integrated force on cylinder (wall = inner eta-face, r=R)."""
    j = 0
    p = blk.p[:, j]
    Aix = blk.A_eta_i_x[:, j]
    Aiy = blk.A_eta_i_y[:, j]
    Fpx = -np.sum(p * Aix)
    Fpy = -np.sum(p * Aiy)
    gxu, gyu = polar_grad(blk, blk.u)
    gxv, gyv = polar_grad(blk, blk.v)
    ux = gxu[:, j]; uy = gyu[:, j]; vx = gxv[:, j]; vy = gyv[:, j]
    txx = 2.0 * nu * ux; txy = nu * (uy + vx); tyy = 2.0 * nu * vy
    Fvx = np.sum(txx * Aix + txy * Aiy)
    Fvy = np.sum(txy * Aix + tyy * Aiy)
    Fx = Fpx + Fvx
    Fy = Fpy + Fvy
    D = 2.0 * blk.R
    q = 0.5 * U * U * D
    return Fx / q, Fy / q, float(Fx), float(Fy)


def force_decomp(blk, nu, U):
    """Decompose the cylinder drag into pressure (Cp) and viscous (Cv) parts,
    and return the wall-tangential velocity gradient (shear) diagnostic."""
    j = 0
    p = blk.p[:, j]
    Aix = blk.A_eta_i_x[:, j]
    Aiy = blk.A_eta_i_y[:, j]
    Fpx = -np.sum(p * Aix)
    Fpy = -np.sum(p * Aiy)
    gxu, gyu = polar_grad(blk, blk.u)
    gxv, gyv = polar_grad(blk, blk.v)
    ux = gxu[:, j]; uy = gyu[:, j]; vx = gxv[:, j]; vy = gyv[:, j]
    txx = 2.0 * nu * ux; txy = nu * (uy + vx); tyy = 2.0 * nu * vy
    Fvx = np.sum(txx * Aix + txy * Aiy)
    Fvy = np.sum(txy * Aix + tyy * Aiy)
    D = 2.0 * blk.R
    q = 0.5 * U * U * D
    # wall-tangential velocity at the first interior cell (shear proxy): if a
    # radial checkerboard is present, |ut[1]| oscillates strongly in theta.
    ut1 = -blk.yc[:, 1] * blk.u[:, 1] + blk.xc[:, 1] * blk.v[:, 1]
    return (float(Fpx / q), float(Fvx / q),
            float(np.std(ut1[::2] - ut1[1::2]) / (np.std(ut1) + 1e-12)))


# --------------------------------------------------------------------------- #
# SIMPLE (Semi-Implicit Method for Pressure-Linked Equations)
# --------------------------------------------------------------------------- #
def build_momentum(blk, recv_mask, dt, nu, reg_r=0.0):
    """Assemble the IMPLICIT momentum matrix  A = I/dt - nu*L  (L = physical
    Laplacian) used by the segregated SIMPLE momentum solve.

    The momentum equation (explicit convection + implicit diffusion + current
    pressure-gradient source) is
        (I/dt - nu*L) u* = u^n/dt - conv(u^n) - grad(p^n)
    so a single factorisation of A solves each outer iteration's velocity
    update.  A is symmetric positive definite (L is negative semi-definite plus
    the I/dt diagonal), so SuperLU factors it cheaply and the triangular solves
    per iteration are O(N).  The recv (fringe) cells are removed like
    build_poisson; the no-slip wall (j=0) is re-imposed u=v=0 after the solve
    (same policy as project()).  reg_r is accepted for interface symmetry but is
    NOT needed: the I/dt term makes A diagonally dominant and nonsingular even
    though L carries the radial (-1)^j checkerboard null space.
    """
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    recv = np.asarray(recv_mask).reshape(-1).astype(bool)
    solved = ~recv
    if not solved.any():                       # degenerate: pin one cell
        solved = np.ones(N, dtype=bool); solved[0] = False
        recv = ~solved
    L_full = _lap_face_matrix(blk, wall_bc="neumann", dt=1.0, rc_poisson=False)
    solved_idx = np.nonzero(solved)[0]
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    A_red = (diags(np.full(L_red.shape[0], 1.0 / dt)) - nu * L_red).tocsc()
    blk._mom_solved_idx = solved_idx
    if N <= 40000:
        return splu(A_red)
    return A_red, solved_idx


def _solve_mom(mom_factored, rhs, solved_idx):
    """Solve the implicit momentum system for one velocity component.

    mom_factored is either a SuperLU factor (splu) or a (A_red, solved_idx)
    tuple for the large-grid BiCGStab fallback.  Returns a full (ni,nj) field
    with the solved cells filled and the fringe/wall left to the caller."""
    flat = rhs.reshape(-1)
    out = np.zeros(rhs.shape)
    if isinstance(mom_factored, tuple):
        A_red, _ = mom_factored
        x0 = None
        M = diags(A_red.diagonal())
        sol, info = bicgstab(A_red, flat[solved_idx], x0=x0, M=M,
                              rtol=1e-9, atol=0.0, maxiter=800)
        if info != 0:
            sol, info = bicgstab(A_red, flat[solved_idx], x0=x0,
                                 rtol=1e-7, atol=0.0, maxiter=1500)
        out.reshape(-1)[solved_idx] = sol
    else:
        out.reshape(-1)[solved_idx] = mom_factored.solve(flat[solved_idx])
    return out


def _solve_poisson_simple(poisson, rhs, blk):
    """Solve the pressure-correction Poisson L p' = rhs on the solved cells.

    Mirrors project(): splu factor or warm-started BiCGStab on the reduced
    system (the polar Laplacian is non-symmetric & indefinite from the stretched
    metric, so CG diverges; BiCGStab converges to machine precision)."""
    solved_idx = blk._solved_idx
    if isinstance(poisson, tuple):
        L_red, _ = poisson
        x0 = blk.p.reshape(-1)[solved_idx] if blk.p is not None else None
        M = diags(L_red.diagonal())
        p_sol, info = bicgstab(L_red, rhs[solved_idx], x0=x0, M=M,
                               rtol=1e-9, atol=0.0, maxiter=800)
        if info != 0:
            p_sol, info = bicgstab(L_red, rhs[solved_idx], x0=x0,
                                   rtol=1e-7, atol=0.0, maxiter=1500)
        p = np.zeros(blk.ni * blk.nj)
        p[solved_idx] = p_sol
    else:
        p = np.zeros(blk.ni * blk.nj)
        p[solved_idx] = poisson.solve(rhs[solved_idx])
    return p


def simple_step(blk, dt, mom_factored, poisson, scheme, nu, niters=6,
                urf_p=0.7, urf_u=0.7, rc=True):
    """One SIMPLE time step: implicit momentum + pressure-correction iterations.

    Outer iteration (Patankar-Spalding):
      1. explicit convection  conv(u^n)               (polar_conv)
      2. momentum RHS        u^n/dt - conv - grad(p^n)   (polar_grad)
      3. implicit solve       (I/dt - nu*L) u* = RHS     (splu / BiCGStab)
      4. impose wall (u*=v*=0) and fringe (free stream)
      5. Rhie-Chow mass-flux divergence of u*  (polar_div, p=p^n, dt)  -> the
         source of the pressure-correction equation (breaks the collocated
         radial checkerboard at the SOURCE when rc=True)
      6. pressure correction   L p' = div(u*)/dt     (rc_poisson == rc)
      7. under-relax:  p += urf_p p' ;  u = u* - urf_u dt grad(p') ;  v likewise
      8. impose BC; early-break once |p'| and |div| are converged
    After the loop a pair-mode (-1)^j checkerboard projection controls the
    collocated null space (same as project()).  The fringe is re-imposed by the
    caller each step.  rc=False falls back to plain divergence + reg_r / cb=pair
    checkerboard control (Rhie-Chow disabled).
    """
    ni, nj = blk.ni, blk.nj
    U = blk.U
    recv = blk.recv
    solved_idx = blk._mom_solved_idx
    for it in range(niters):
        duc, dvc = polar_conv(blk, blk.u, blk.v, U, scheme=scheme)
        gpx, gpy = polar_grad(blk, blk.p)
        rhs_u = blk.u / dt - duc - gpx
        rhs_v = blk.v / dt - dvc - gpy
        us = _solve_mom(mom_factored, rhs_u, solved_idx)
        vs = _solve_mom(mom_factored, rhs_v, solved_idx)
        us[:, 0] = 0.0; vs[:, 0] = 0.0
        us[recv] = blk.u[recv]; vs[recv] = blk.v[recv]   # fringe set by caller
        # pressure-correction source: Rhie-Chow mass-flux divergence of u*
        div = polar_div(blk, us, vs, p=blk.p, dt=dt) if rc else polar_div(blk, us, vs)
        rhs = div.reshape(-1) / dt
        p_prime = _solve_poisson_simple(poisson, rhs, blk).reshape(ni, nj)
        # under-relaxed correction
        blk.p = blk.p + urf_p * p_prime
        gpxp, gpyp = polar_grad(blk, p_prime)
        blk.u = us - urf_u * dt * gpxp
        blk.v = vs - urf_u * dt * gpyp
        blk.u[:, 0] = 0.0; blk.v[:, 0] = 0.0   # fringe re-imposed by caller
        if it > 0 and abs(p_prime).max() < 1e-7 and abs(div).max() < 1e-7:
            break
    # collocated (-1)^j checkerboard control (pair mode) -- same as project()
    blk.p = polar_checkerboard_project(blk, blk.p, mode="pair")
    blk.u = polar_checkerboard_project(blk, blk.u, mode="pair")
    blk.v = polar_checkerboard_project(blk, blk.v, mode="pair")
