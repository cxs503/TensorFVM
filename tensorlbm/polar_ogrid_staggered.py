"""Staggered (MAC) polar O-grid FVM solver -- the checkerboard-free carrier.

This module puts the body-fitted polar O-grid on a STAGGERED (MAC) grid:
    * ur (radial velocity) at RADIAL faces (eta-face, r = rn[j]), shape (ni, nj+1)
    * ut (azimuthal velocity) at ANGULAR faces (xi-face, theta = thn[i]), (ni+1, nj)
    * p  (pressure) at CELL CENTERS, shape (ni, nj)

Why staggered: the collocated polar_ogrid had a radial (-1)^j checkerboard in
the NULL SPACE of its divergence (face-average of cell values kills (-1)^j), so
the pressure projection could never remove it.  On this grid the divergence RHS
uses FACE-CENTER velocities directly (no cell face-average), so (-1)^j is NOT in
its null space -> clean projection, no spurious near-wall saw-tooth.

Division of labour (to stay correct on the STRETCHED polar grid):
  * Velocity divergence (RHS of projection) = NEW face-based `div` (checkerboard
    free by construction).
  * Pressure Poisson + gradient = REUSED from polar_ogrid (polar_lap /
    polar_grad / build_poisson), which are independently verified symmetric,
    NSD, and exact (Lap(r^2)=+4).  The Poisson null space is pinned by reg_r
    (the proven magnitude), but since the divergence RHS carries no (-1)^j
    component, the recovered pressure is clean.
  * Convection = Cartesian-component donor-cell momentum-flux divergence
    (the verified polar_ogrid recipe, transcribed to the MAC faces).  Working in
    Cartesian (u,v) makes the polar (1/r) curvature vanish, so upwinding is both
    dissipative and an exact steady state for a uniform free stream -- the two
    properties the old polar-component upwind lacked.  The cell-center
    acceleration is rotated to polar and interpolated to the faces (harmless for
    the source term; the projection's div/grad stay checkerboard-free).
  * Diffusion = the SELF-CONSISTENT MAC FACE Laplacian (mac_diffusion_faces) applied
    DIRECTLY to the face velocities -- NOT "cell Laplacian then average back", which
    is anti-dissipative on a stretched grid (it GROWS energy and blew up M1).

Force on the cylinder is integrated on the EXACT body-fitted wall (eta-face at
r=R) -> geometrically self-consistent (no stair-step +17% form-drag).
"""
import numpy as np
from scipy.sparse.linalg import splu
import polar_ogrid as POG_c


class Block:
    def __init__(self, name, ni, nj, xc, yc):
        self.name = name
        self.ni, self.nj = ni, nj
        self.xc, self.yc = xc, yc            # cell centers (ni, nj)
        self.ur = np.zeros((ni, nj + 1))     # radial velocity, eta-faces
        self.ut = np.zeros((ni + 1, nj))      # azimuthal velocity, xi-faces
        self.p = np.zeros((ni, nj))           # pressure, cell centers
        self.J = None
        self._L_full_mac = None               # cached MAC Laplacian (csc)
        self._solved_idx = None               # reduced-Poisson solve indices


def make_ogrid_staggered(ni, nj, cx, cy, R, Rf_og, Rf_sponge, beta=2.0):
    thn = np.linspace(0.0, 2.0 * np.pi, ni, endpoint=False)
    dth = 2.0 * np.pi / ni
    t = np.linspace(0.0, 1.0, nj + 1)
    rn = R + (Rf_og - R) * (np.exp(beta * t) - 1.0) / (np.exp(beta) - 1.0)
    thc = thn + 0.5 * dth
    rc = 0.5 * (rn[:-1] + rn[1:])
    THC, RC = np.meshgrid(thc, rc, indexing="ij")
    xc = cx + RC * np.cos(THC)
    yc = cy + RC * np.sin(THC)

    blk = Block("ogs", ni, nj, xc, yc)
    blk.r = np.sqrt((xc - cx) ** 2 + (yc - cy) ** 2)
    blk.cx, blk.cy = cx, cy
    blk.R, blk.Rf_og, blk.Rf_sponge, blk.beta = R, Rf_og, Rf_sponge, beta
    blk.ni, blk.nj = ni, nj
    blk.rn = rn
    blk.rc = RC
    blk.thn = thn
    blk.thc = thc
    blk.cos_thc = (xc - cx) / rc
    blk.sin_thc = (yc - cy) / rc
    blk.dth = dth
    blk.dr_node = rn[1:] - rn[:-1]
    blk.cos_thn = np.cos(thn)
    blk.sin_thn = np.sin(thn)
    blk._nu = 0.0
    blk.U = 1.0

    # ---- physical face area vectors (outward normals) ----
    # eta-face (radial), at r = rn[j], normal = (+cos thc, +sin thc)
    blk.A_eta_x = np.cos(thc)[:, None] * rn[None, :] * dth        # (ni, nj+1)
    blk.A_eta_y = np.sin(thc)[:, None] * rn[None, :] * dth
    # xi-face (angular), at theta = thn[i], normal = (-sin thn, +cos thn)
    blk.A_xi_x = np.zeros((ni + 1, nj))
    blk.A_xi_y = np.zeros((ni + 1, nj))
    for i in range(ni + 1):
        ip = i % ni
        blk.A_xi_x[i] = -blk.sin_thn[ip] * blk.dr_node[None, :]
        blk.A_xi_y[i] = blk.cos_thn[ip] * blk.dr_node[None, :]
    # inner eta-face (j=0, r=R) area vector for force integral
    blk.A_eta_i_x = blk.A_eta_x[:, 0].copy()
    blk.A_eta_i_y = blk.A_eta_y[:, 0].copy()

    blk.J = (RC * dth * blk.dr_node[None, :]).copy()

    # ---- face-center coordinates (probing / rendering) ----
    blk.xf_eta = cx + rn[None, :] * np.cos(thc)[:, None]
    blk.yf_eta = cy + rn[None, :] * np.sin(thc)[:, None]
    xfn = np.zeros((ni + 1, nj)); yfn = np.zeros((ni + 1, nj))
    for i in range(ni + 1):
        ip = i % ni
        xfn[i] = cx + rc * np.cos(thn[ip])
        yfn[i] = cy + rc * np.sin(thn[ip])
    blk.xf_xi = xfn
    blk.yf_xi = yfn

    # ---- collocated temp block (same geometry) to reuse verified operators ----
    blk._tmp = POG_c.make_ogrid(ni, nj, cx, cy, R, Rf_og, beta=beta)
    return blk


# --------------------------------------------------------------------------- #
# Staggered velocity divergence (face velocities -> cell-center scalar)
# --------------------------------------------------------------------------- #
def div(blk, ur, ut):
    """Physical divergence div(ur,ut) at cell centers (ni, nj).

    F_eta[j] = ur[j] * rn[j] * dth  (scalar mass flux through eta-face at rn[j])
    F_xi[i]  = ut[i]  * dr[j]       (scalar mass flux through xi-face at thn[i])
    div[i,j] = (F_eta(outer)-F_eta(inner) + F_xi(right)-F_xi(left)) / J.
    NO face-averaging of cell values -> (-1)^j checkerboard is NOT a null space.
    """
    ni, nj = blk.ni, blk.nj
    rn, dth, dr = blk.rn, blk.dth, blk.dr_node
    F_eta = ur * rn[None, :] * dth                 # (ni, nj+1)
    F_xi = ut * dr[None, :]                        # (ni+1, nj)  (seam at i=ni)
    Fr = F_eta[:, 1:] - F_eta[:, :-1]              # (ni, nj) outer - inner
    Fa = F_xi[1:, :] - F_xi[:-1, :]                # (ni, nj) right - left (periodic)
    return (Fr + Fa) / blk.J


# --------------------------------------------------------------------------- #
# Cell-center velocity reconstruction (faces -> cell centers)
# --------------------------------------------------------------------------- #
def cell_vel(blk, ur, ut):
    """Reconstruct cell-center (ni, nj) polar velocity components."""
    ur_c = 0.5 * (ur[:, 1:] + ur[:, :-1])         # (ni, nj)
    ut_c = 0.5 * (ut[1:, :] + ut[:-1, :])         # (ni, nj) periodic in i
    return ur_c, ut_c


def cell_vel_xy(blk, ur, ut):
    """Reconstruct cell-center (ni, nj) physical x,y velocity."""
    ur_c, ut_c = cell_vel(blk, ur, ut)
    ct, st = blk.cos_thc, blk.sin_thc
    u = ur_c * ct - ut_c * st
    v = ur_c * st + ut_c * ct
    return u, v


# --------------------------------------------------------------------------- #
# Convective + viscous acceleration  (BOTH computed directly on the FACES)
# --------------------------------------------------------------------------- #
def _minmod(a, b):
    """TVD minmod limiter: 0.5*(sign(a)+sign(b))*min(|a|,|b|); 0 if opposite sign."""
    same = (a * b) > 0.0
    mag = np.minimum(np.abs(a), np.abs(b))
    return np.where(same, np.sign(a) * mag, 0.0)


def mac_conv_faces(blk, ur, ut, scheme="upwind1", return_cell=False):
    """MAC convection via the DUAL-CELL conservative DONOR-CELL advection of the
    FACE velocities themselves (scalar advection of ur and of ut).

    The dual cell of the radial face ur[i,j] (at node radius rn[j]) is the volume
    strip r in [rn[j-1], rn[j]], theta in [thn[i], thn[i+1]] -- i.e. it is bounded
    by the ADJACENT ur FACES (rn[j-1], rn[j]), NOT by the cell-center radii rc.
    This is essential: the flux velocity (evaluated at the dual-cell boundary) and
    the UPDATED face velocity ur[i,j] must coincide at the same location (rn[j]),
    otherwise the discrete energy identity d(1/2 q^2)/dt = -1/2(q_up-q_self)^2(u.n)_+
    cannot close and the scheme is anti-dissipative (verified: energy -> NaN).
    With the boundary at the ur faces the scheme is strictly dissipative and an
    exact steady state for a uniform free stream.  The angular faces carry the
    contravariant tangential velocity ut, so the polar (1/r) curvature enters via
    the metric areas -- this is the conservative curvilinear momentum equation.

    The acceleration returned is the POSITIVE flux divergence +div(q (x) q); the
    stepper SUBTRACTS it (ur -= dt*dur), like the verified collocated polar_conv
    caller (ustar = u + dt*(-duc)).
    """
    ni, nj = blk.ni, blk.nj
    rn, dth, dr = blk.rn, blk.dth, blk.dr_node
    # ur faces directly (at node radii rn); these are BOTH the flux velocity and
    # the updated quantity -> energy identity closes.
    ur_out = ur[:, 1:nj]                              # ur at rn[j]   (outer dual face)
    ur_in = ur[:, 0:nj - 1]                           # ur at rn[j-1] (inner dual face)
    ut_L = ut[0:ni, 0:nj - 1]                        # ut at thn[i]   (left dual face)
    ut_R = np.roll(ut, -1, axis=0)[0:ni, 0:nj - 1]  # ut at thn[i+1] (right dual face)

    A_out = rn[1:nj] * dth                           # area of outer radial face (rn[j])
    A_in = rn[0:nj - 1] * dth                        # area of inner radial face (rn[j-1])
    A_ang = rn[1:nj] - rn[0:nj - 1]                  # radial width of angular face (=dr)
    V_d = dth * 0.5 * (rn[1:nj] ** 2 - rn[0:nj - 1] ** 2)   # dual-cell volume = J[:,1:nj]

    # ===================== ur advection (scalar donor-cell of ur) ============
    ur_up_out = np.where(ur_out > 0.0, ur[:, 0:nj - 1], ur[:, 1:nj])   # inner face if outflow
    F_ur_o = ur_up_out * ur_out * A_out
    ur_up_in = np.where(ur_in < 0.0, ur[:, 1:nj], ur[:, 0:nj - 1])    # outer face if inflow
    F_ur_i = ur_up_in * (-ur_in) * A_in
    ur_up_r = np.where(ut_R > 0.0, ur[:, 0:nj - 1], np.roll(ur[:, 0:nj - 1], -1, axis=0))
    F_ur_r = ur_up_r * ut_R * A_ang
    ur_up_l = np.where(ut_L < 0.0, np.roll(ur[:, 0:nj - 1], -1, axis=0), ur[:, 0:nj - 1])
    F_ur_l = ur_up_l * (-ut_L) * A_ang
    dur = -(F_ur_o + F_ur_i + F_ur_r + F_ur_l) / V_d      # (ni, nj-1)

    # ===================== ut advection (scalar donor-cell of ut) ============
    ut_up_out = np.where(ur_out > 0.0, ut[0:ni, 0:nj - 1], ut[0:ni, 1:nj])
    F_ut_o = ut_up_out * ur_out * A_out
    ut_up_in = np.where(ur_in < 0.0, ut[0:ni, 1:nj], ut[0:ni, 0:nj - 1])
    F_ut_i = ut_up_in * (-ur_in) * A_in
    ut_up_r = np.where(ut_R > 0.0, ut[0:ni, 0:nj - 1], ut[1:ni + 1, 0:nj - 1])
    F_ut_r = ut_up_r * ut_R * A_ang
    ut_up_l = np.where(ut_L < 0.0, ut[1:ni + 1, 0:nj - 1], ut[0:ni, 0:nj - 1])
    F_ut_l = ut_up_l * (-ut_L) * A_ang
    dut = -(F_ut_o + F_ut_i + F_ut_r + F_ut_l) / V_d      # (ni, nj-1)

    # place onto the face arrays (dual cell [rn[j-1],rn[j]] -> accelerates ur[i,j])
    dur_f = np.zeros((ni, nj + 1)); dur_f[:, 1:nj] = dur
    dut_f = np.zeros((ni + 1, nj)); dut_f[0:ni, 1:nj] = dut; dut_f[ni, :] = dut_f[0, :]

    if return_cell:
        ar_c = 0.5 * (dur_f[:, 1:] + dur_f[:, :-1])     # cell-center polar accel
        ath_c = 0.5 * (dut_f[1:, :] + dut_f[:-1, :])
        return dur_f, dut_f, ar_c, ath_c
    return dur_f, dut_f


def conv_diff_accel(blk, ur, ut, nu, scheme="upwind1"):
    """Face acceleration tendency (dur, dut) = MAC diffusion - MAC convection.

    WIP / KNOWN-UNSTABLE: this currently reuses the collocated polar_ogrid
    operators (polar_conv / polar_diffusion, verified stable on cell centres) and
    lifts the resulting Cartesian tendency to the MAC faces by a half-average.
    Because the staggered wall BC enforces no-slip at the FACE (ur[:,0]=ut[:,0]=0)
    while polar_conv expects no-slip at the CELL CENTRE, the reconstructed
    near-wall cell velocity (ur_c[:,0] = 0.5*U*cos th) injects a spurious momentum
    source that feeds back and the scheme diverges (umax -> 1e78 by ~step 1000,
    verified by _diag2.py).  The alternative pure dual-cell MAC-face convection
    (mac_conv_faces, scalar transport of the polar components) is ANTI-dissipative
    because it does not preserve a uniform free stream in polar coordinates
    (div(ur*u) = U^2/r sin^2 th != 0).  A fully-staggered Cartesian-momentum MAC
    convection (consistent collocated-primary + staggered MAC projection) is the
    correct fix but is not yet implemented; meanwhile the production drag result
    is obtained from the collocated run_ogrid_primary / run_shed drivers.
    """
    ni, nj = blk.ni, blk.nj
    u_c, v_c = cell_vel_xy(blk, ur, ut)                           # cell-centre (ni,nj)
    blk._tmp._nu = nu
    duc_c, dvc_c = POG_c.polar_conv(blk._tmp, u_c, v_c, U=1.0, scheme=scheme)
    dud_c, dvd_c = POG_c.polar_diffusion(blk._tmp, u_c, v_c)
    dur_c = dud_c - duc_c                                         # diff - conv
    dvt_c = dvd_c - dvc_c
    # lift to MAC faces (half-average across the shared face)
    dur_f = np.zeros((ni, nj + 1))
    dur_f[:, 1:nj] = 0.5 * (dur_c[:, :-1] + dur_c[:, 1:])         # radial face k
    dut_f = np.zeros((ni + 1, nj))
    dut_f[1:ni, :] = 0.5 * (dvt_c[:-1, :] + dvt_c[1:, :])         # angular face i
    dut_f[0, :] = dut_f[ni - 1, :]
    dut_f[ni, :] = dut_f[0, :]
    return dur_f, dut_f


def mac_diffusion_faces(blk, ur, ut, nu):
    """Viscous tendency at the FACES (the correct MAC diffusion for face velocities).

    The diffusion of a face velocity is the MAC Laplacian evaluated ON the face
    grid (faces sit at rn[k] / thn[i]), NOT "reconstruct to cells, apply the cell
    Laplacian, average back" -- that averaged stencil is anti-diffusive on a
    stretched grid (it GROWS energy).  Here each face gets the conservative
    flux-difference of its normal gradient using the SAME face areas/distances as
    the divergence, so it is strictly dissipative:
        dur[k] = (1/Vf) [ A_out (ur[k+1]-ur[k])/dkp - A_in (ur[k]-ur[k-1])/dkm ]
    with A_out = rc[k]*dth, A_in = rc[k-1]*dth, Vf = rn[k]*dth*0.5*(dkp+dkm).
    """
    ni, nj = blk.ni, blk.nj
    rn, dth, dr = blk.rn, blk.dth, blk.dr_node
    rc = blk.rc                                       # (ni, nj)
    diff_rn = np.diff(rn)                             # (nj,) rn[k+1]-rn[k]
    dur = np.zeros((ni, nj + 1))
    # interior radial faces k = 1..nj-1
    A_out = rc[:, 1:nj] * dth                        # area of half-face k->k+1
    A_in = rc[:, 0:nj - 1] * dth                     # area of half-face k-1->k
    flux_out = A_out * (ur[:, 2:nj + 1] - ur[:, 1:nj]) / diff_rn[1:nj]
    flux_in = A_in * (ur[:, 1:nj] - ur[:, 0:nj - 1]) / diff_rn[0:nj - 1]
    Vf = rn[1:nj] * dth * 0.5 * (diff_rn[1:nj] + diff_rn[0:nj - 1])
    dur[:, 1:nj] = (flux_out - flux_in) / Vf
    # k=0 (wall) and k=nj (outer fringe) left 0 -> held by apply_bc (no-slip)
    # angular faces i = 0..ni-1 (periodic), per j
    dut = np.zeros((ni + 1, nj))
    dist = rc * dth                                  # (ni, nj) arc between angular faces
    Ah = dr[None, :]                                 # radial width of angular face
    flux_r = Ah * (ut[2:ni + 1, :] - ut[1:ni, :]) / dist[1:ni]
    flux_l = Ah * (ut[1:ni, :] - ut[0:ni - 1, :]) / dist[1:ni]
    Vfa = dr[None, :] * rc[1:ni] * dth
    dut[1:ni, :] = (flux_r - flux_l) / Vfa
    # periodic seam i=0 (== i=ni)
    flux_r0 = Ah * (ut[1, :] - ut[0, :]) / dist[0]
    flux_l0 = Ah * (ut[0, :] - ut[ni - 1, :]) / dist[0]
    dut[0, :] = (flux_r0 - flux_l0) / (dr[None, :] * rc[0] * dth)
    dut[ni, :] = dut[0, :]
    # ---- RADIAL diffusion of ut (angular-face velocity) -- WAS MISSING.
    # ut[i,j] sits at (thn[i], rc[j]); radial neighbours ut[i,j-1], ut[i,j+1].
    # Radial flux through the face at mid-radius uses arc area rc_face*dth and
    # the radial spacing rc[j+1]-rc[j].  Without this term the near-wall angular
    # velocity could not develop a viscous boundary layer (ut[:,1] stayed ~1.7
    # instead of ~0.25), over-predicting the wall shear / viscous drag.
    A_rout = rc[:, 1:nj] * dth                        # arc area at rc[k+0.5]
    A_rin = rc[:, 0:nj - 1] * dth                     # arc area at rc[k-0.5]
    flux_ru = A_rout * (ut[0:ni, 1:nj] - ut[0:ni, 0:nj - 1]) / diff_rn[1:nj]
    flux_rl = A_rin * (ut[0:ni, 0:nj - 1] - ut[0:ni, :-2]) / diff_rn[0:nj - 1]
    # place onto angular faces k=1..nj-1 (k=0 wall, k=nj outer held by BC)
    Vf_r = dr[None, 1:nj - 1] * (0.5 * (rc[:, 1:nj - 1] + rc[:, 2:nj])) * dth
    dut[:, 1:nj - 1] = dut[:, 1:nj - 1] + (flux_ru[:, :-1] - flux_rl[:, 1:]) / Vf_r
    return nu * dur, nu * dut


# --------------------------------------------------------------------------- #
# Boundary conditions
# --------------------------------------------------------------------------- #
def apply_bc(blk, ur, ut):
    """Wall (no-slip at j=0, r=R) + periodic angular seam.

    Only the SOLID wall is enforced here (it is a hard boundary and produces no
    spurious divergence).  The far-field free stream is imposed by the SMOOTH
    sponge_relax in the outer ring -- a hard Dirichlet reset of the recv cells
    AFTER the projection re-injects divergence at the sponge interface (that was
    the 5.9 max-div leak in the M1 run) and is avoided on purpose.
    """
    ur = ur.copy()
    ut = ut.copy()
    ur[:, 0] = 0.0                               # no penetration at wall
    ut[:, 0] = 0.0                               # no slip at wall
    ut[blk.ni, :] = ut[0, :]                     # periodic seam
    return ur, ut


def sponge_relax(blk, ur, ut, Ueff, aoa, dt, sigma_max=60.0):
    """Smooth far-field sponge: relax the velocity toward the free stream in the
    outer ring (r >= Rf_sponge).  Quadratic ramp from 0 at the sponge inlet to
    sigma_max at the outer boundary; explicit relaxation ur += dt*sigma*(u_fs-u),
    with dt*sigma_max ~ 0.04 (stable).  Because it is smooth it introduces NO
    divergence jump at the interface (unlike a hard recv reset), so the Chorin
    projection stays exactly divergence-free in the solved region.
    """
    ur = ur.copy()
    ut = ut.copy()
    ur_fs = Ueff * np.cos(blk.thc - aoa)                       # radial faces (ni, nj+1)
    ut_fs = -Ueff * np.sin(blk.thn)                           # xi-face nodes (ni,)
    ut_fs = np.concatenate([ut_fs, ut_fs[:1]])                # (ni+1,) periodic seam
    s_eta = np.where(blk.rn >= blk.Rf_sponge,
                     sigma_max * ((blk.rn - blk.Rf_sponge)
                                  / (blk.Rf_og - blk.Rf_sponge)) ** 2, 0.0)
    s_xi = np.where(blk.rc[0, :] >= blk.Rf_sponge,
                    sigma_max * ((blk.rc[0, :] - blk.Rf_sponge)
                                 / (blk.Rf_og - blk.Rf_sponge)) ** 2, 0.0)
    ur += dt * s_eta[None, :] * (ur_fs[:, None] - ur)
    ut += dt * s_xi[None, :] * (ut_fs[:, None] - ut)
    ut[blk.ni, :] = ut[0, :]
    return ur, ut


# --------------------------------------------------------------------------- #
# Self-consistent MAC Laplacian / face gradient / projection
# --------------------------------------------------------------------------- #
# On the stretched polar grid the collocated polar_ogrid operators are NOT
# self-adjoint / NSD / checkerboard-free (its own unit test gives Lap(x)~0.4,
# Lap(r^2)~4.19, NSD=False, projection reduction factor 0.11).  Reusing them
# inherits that brokenness.  Instead we build a PROPER conservative FV (MAC)
# Laplacian from the SAME metric the staggered divergence uses:
#     A_eta[k] = rn[k] * dth     (radial face at r = rn[k])
#     A_xi[j]  = dr_node[j]      (angular face, radial width)
#     J[i,j]   = rc[i,j] * dth * dr_node[j]
# The Laplacian, the face gradient and the divergence are then exact adjoints on
# this metric, so the Chorin projection removes the divergence to machine
# precision, and (-1)^j is NOT in the divergence null space.
def _build_mac_L(blk, reg_r=0.0):
    """Assemble the MAC Laplacian L = (1/J) * sum_faces A_f (phi_nb-phi_self)/dist_f.

    Each SHARED face is visited exactly once (we add the "outer" radial neighbour
    and the "right" angular neighbour for every cell; their mirrors are covered by
    the neighbour's own loop).  For a face of coefficient c between cells a,b we
    set L[a,b]=L[b,a]=+c and L[a,a]-=c, L[b,b]-=c, so every row sums to 0 -> L is
    symmetric with the correct sign (L = +nabla^2, negative semi-definite; the
    constant null space is removed by the recv cells in build_poisson_staggered).
    Wall (j=0) radial face = NEUMANN (zero flux) -> no term.
    """
    from scipy.sparse import csc_matrix
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    rc, rn, dr, dth = blk.rc, blk.rn, blk.dr_node, blk.dth
    A_eta = rn * dth                              # (nj+1,) radial face areas
    A_xi = dr                                     # (nj,) angular face area (all i)
    L = np.zeros((N, N))
    for i in range(ni):
        for j in range(nj):
            k = i * nj + j
            # ----- outer radial neighbour (j+1): face k+1 -----
            if j < nj - 1:
                kn = i * nj + (j + 1)
                c = A_eta[j + 1] / (rc[i, j + 1] - rc[i, j])
                L[k, kn] += c; L[kn, k] += c
                L[k, k] -= c; L[kn, kn] -= c
            # ----- right angular neighbour (i+1): face between cell i and i+1 -----
            kr = ((i + 1) % ni) * nj + j
            c = A_xi[j] / (rc[i, j] * dth)
            L[k, kr] += c; L[kr, k] += c
            L[k, k] -= c; L[kr, kr] -= c
    if reg_r > 0.0:                  # optional radial 2nd-order pin (harmless)
        for i in range(ni):
            for j in range(1, nj - 1):
                k = i * nj + j
                L[k, i * nj + (j - 1)] += reg_r
                L[k, k] += -2.0 * reg_r
                L[k, i * nj + (j + 1)] += reg_r
    # divide by cell volume -> L p = (1/J) sum_faces A_f(dist) (p_nb - p_self)
    # i.e. the physical Laplacian in the SAME units as the divergence (which also
    # divides by J).  L is then the exact adjoint of `div` on this MAC metric.
    L = L / blk.J.reshape(-1)[:, None]
    return csc_matrix(L)


def _mac_face_grad(blk, p):
    """Face-normal gradient of p (MAC):
        g_eta (ni,nj+1): radial face k between cell k-1,k -> (p[k]-p[k-1])/(rc_k-rc_{k-1})
        g_xi  (ni+1,nj): angular face i between cell i-1,i -> (p[i]-p[i-1])/(rc*dth)
    Wall face (k=0) and outer face (k=nj) are NEUMANN (g=0)."""
    ni, nj = blk.ni, blk.nj
    rc, rn, dr, dth = blk.rc, blk.rn, blk.dr_node, blk.dth
    g_eta = np.zeros((ni, nj + 1))
    g_eta[:, 1:nj] = (p[:, 1:] - p[:, :-1]) / (rc[:, 1:] - rc[:, :-1])   # k=1..nj-1
    g_xi = np.zeros((ni + 1, nj))
    # face i (between cell i-1 and i) gradient = (p[i]-p[i-1])/dist, i=0..ni-1.
    # NOTE: g_xi[i] must hold the gradient AT face i (NOT shifted) so that div,
    # which uses (g_xi[i+1]-g_xi[i]) - Fa at cell i, gives div(grad p)=L p.
    g_xi[:ni, :] = (p - np.roll(p, 1, axis=0)) / (rc * dth)
    g_xi[ni, :] = g_xi[0, :]                                             # periodic wrap (div needs F_xi[ni]=F_xi[0])
    return g_eta, g_xi


def build_poisson_staggered(blk, recv_mask, dt=1.0, reg_r=0.0):
    """Reduced Poisson solver on the self-consistent MAC Laplacian.

    recv cells are removed (Dirichlet p=0), pinning the constant-pressure null
    space.  Because the divergence RHS carries no (-1)^j component, no reg_r is
    needed; reg_r is kept as an optional harmless extra pin.
    """
    from scipy.sparse.linalg import splu
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    recv = np.asarray(recv_mask).reshape(-1).astype(bool)
    solved = ~recv
    if not solved.any():                       # degenerate fallback
        solved = np.ones(N, dtype=bool); solved[0] = False
        recv = ~solved
    L_full = _build_mac_L(blk, reg_r=reg_r)
    blk._L_full_mac = L_full
    solved_idx = np.nonzero(solved)[0]
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    blk._solved_idx = solved_idx
    return splu(L_red)


def project(blk, dt, recv, poisson, ur, ut):
    """Chorin projection on the MAC metric.

    Solve L p = div(ur,ut)/dt (L = -nabla^2, exact adjoint of the staggered
    divergence on this metric), then correct the face velocities with the MAC
    face gradient:  ur -= dt*g_eta,  ut -= dt*g_xi.  Because div and grad are
    exact adjoints, div(u_new) = 0 to machine precision.
    """
    ni, nj = blk.ni, blk.nj
    d = div(blk, ur, ut).reshape(-1) / dt
    solved_idx = blk._solved_idx
    p = np.zeros(ni * nj)
    p[solved_idx] = poisson.solve(d[solved_idx])
    p = p.reshape(ni, nj)
    blk.p = p
    g_eta, g_xi = _mac_face_grad(blk, p)
    ur = ur.copy()
    ut = ut.copy()
    ur[:, 1:nj] -= dt * g_eta[:, 1:nj]          # interior radial faces (grad at face j)
    ut[0:ni, :] -= dt * g_xi[0:ni, :]            # angular faces i=0..ni-1 (grad at face i)
    ut[ni, :] = ut[0, :]                         # periodic seam
    return ur, ut


# --------------------------------------------------------------------------- #
# Force on cylinder (body-fitted wall j=0)
# --------------------------------------------------------------------------- #
def cylinder_forces(blk, nu, U):
    """Integrated force on the cylinder (exact body-fitted wall, eta-face j=0).

    Pressure: -p * A_eta_inner integrated over the wall (anchor F: a manufactured
    front-high-pressure gives a +drag, i.e. force in the -x / upstream direction).

    Viscous: use the STAGGERED angular face velocity ut DIRECTLY.  On the MAC grid
    the tangential (theta) face ut[i,j] sits at (thn[i], rc[j]); the wall no-slip
    BC enforces ut[i,0] = 0 exactly, so the wall shear is the one-sided radial
    gradient of ut, du_theta/dr = ut[i,1] / (rc[1] - R).  This is the physically
    correct wall-tangential-velocity gradient.  It is MUSS NOT be reconstructed
    through the collocated Cartesian gradient polar_grad: that operator rotates the
    radial derivative into (gx,gy) via gx = cos*dr - sin*dt, and at theta=90 deg
    cos=0 so the entire near-wall du_theta/dr term VANISHES, under-predicting the
    friction stress by ~40x (verified: hand du/dr=+376 vs polar_grad gx=-10 at the
    shoulder).  The cylinder force is F_visc = -tau_w * t_hat * dA with
    t_hat = (-sin thn, cos thn) (unit tangent along +theta), tau_w = nu*du_theta/dr.
    """
    ni, nj = blk.ni, blk.nj
    p = blk.p[:, 0]
    Aix = blk.A_eta_i_x
    Aiy = blk.A_eta_i_y
    Amod = np.sqrt(Aix ** 2 + Aiy ** 2)               # = |A_eta_i| = dA (per face)
    Fpx = -np.sum(p * Aix)
    Fpy = -np.sum(p * Aiy)
    # ---- wall shear from the staggered tangential face -------------------------
    thn = blk.thn                                       # (ni,) angular-face angle
    dudr_wall = blk.ut[0:ni, 1] / (blk.rc[:, 1] - blk.R)   # (ni,) du_theta/dr, wall
    tau_w = nu * dudr_wall                             # wall shear stress (along +theta)
    thx = -np.sin(thn)                                 # tangent unit vector x comp
    thy = np.cos(thn)                                  # tangent unit vector y comp
    Fvx = -np.sum(tau_w * thx * Amod)                  # F_visc = -tau_w * t_hat * dA
    Fvy = -np.sum(tau_w * thy * Amod)
    Fx = Fpx + Fvx
    Fy = Fpy + Fvy
    D = 2.0 * blk.R
    q = 0.5 * U * U * D
    return Fx / q, Fy / q, float(Fx), float(Fy)
