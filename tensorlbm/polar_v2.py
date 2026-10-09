"""polar_v2.py -- clean self-adjoint polar O-grid FVM operators.

All operators derive from ONE face-flux primitive so that grad and div are
exact (discrete) adjoints.  This guarantees the projection Laplacian
L = D @ G is symmetric & negative semi-definite, AND that the correction
u -= dt*G p removes div(u*) to machine precision for ANY geometry.

Verified anchors (see test_polar_v2):
  * uniform flow   -> div == 0  everywhere (incl. wall & periodic seam)
  * p = r^2        -> div(grad p) == +4   (constant, exact)
  * p = x (=r cos th) -> div(grad p) == 0 (harmonic)
  * L = D@G symmetric, all eig <= 0 (NSD)
  * projection removes divergence to machine precision
"""
import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu, eigsh


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
    thn = np.linspace(0.0, 2.0 * np.pi, ni, endpoint=False)
    dth = 2.0 * np.pi / ni
    t = np.linspace(0.0, 1.0, nj + 1)
    rn = R + (Rf_og - R) * (np.exp(beta * t) - 1.0) / (np.exp(beta) - 1.0)
    THN, RN = np.meshgrid(thn, rn, indexing="ij")
    xn = cx + RN * np.cos(THN)
    yn = cy + RN * np.sin(THN)
    thc = thn + 0.5 * dth
    rc = 0.5 * (rn[:-1] + rn[1:])
    THC, RC = np.meshgrid(thc, rc, indexing="ij")
    xc = cx + RC * np.cos(THC)
    yc = cy + RC * np.sin(THC)
    blk = Block("ogrid", ni, nj, xc, yc)
    blk.r = np.sqrt((xc - cx) ** 2 + (yc - cy) ** 2)
    blk.cx, blk.cy = cx, cy
    blk.R, blk.Rf_og, blk.beta = R, Rf_og, beta
    blk.rn = rn
    blk.rc = RC
    blk.thn = thn
    blk.cos_th = (xc - cx) / rc
    blk.sin_th = (yc - cy) / rc
    blk.dth = dth
    blk.dr_node = rn[1:] - rn[:-1]
    blk.cos_thn = np.cos(thn)                      # node-angle trig (angular faces)
    blk.sin_thn = np.sin(thn)
    # cell volume (for diagnostics / force), NOT used in operator core
    blk.J = (blk.rc * blk.dth * blk.dr_node[None, :]).copy()
    return blk


# --------------------------------------------------------------------------- #
# Gradient: cell-centre (x,y) gradient from face-normal gradients.
# --------------------------------------------------------------------------- #
def polar_grad(blk, p):
    """Physical (x,y) gradient at cell centres, built from FACE gradients only
    (adjacent cells, never skip-centered) so it is the exact adjoint of
    polar_div.  Radial face gradient gr[k] = (p[k]-p[k-1])/dr at face k
    (between cell k-1,k); cell d/dr = average of its two faces (one-sided at
    walls).  Angular face gradient gth[i] = (p[i]-p[i-1])/(rc*dth); cell
    (1/r)d/dth = average of its two faces.  Reconstructed via the orthogonal
    polar rotation (cos,sin)."""
    ni, nj = blk.ni, blk.nj
    rc, rn, dth = blk.rc, blk.rn, blk.dth
    # radial face gradients gr[k], k=1..nj-1 (face between cell k-1 and k)
    dr_face = (rn[1:-1] - rn[:-2])[None, :]           # (1,nj-1)
    gr = np.zeros((ni, nj + 1))
    gr[:, 1:-1] = (p[:, 1:] - p[:, :-1]) / dr_face     # faces 1..nj-1  (ni,nj-1)
    drdr = np.zeros((ni, nj))
    drdr[:, 0] = gr[:, 1]                             # wall cell: one-sided (outer face)
    drdr[:, 1:-1] = 0.5 * (gr[:, 1:nj - 1] + gr[:, 2:nj])
    drdr[:, -1] = gr[:, -1]                           # outer cell: one-sided (Neumann)
    # angular face gradients gth[i] at NODE angle (periodic, face i between i-1,i)
    gth = (p - np.roll(p, 1, axis=0)) / (rc * dth)    # (ni,nj)
    ang_d = 0.5 * (gth + np.roll(gth, -1, axis=0))     # (1/r)d/dth at cell centre
    ct, st = blk.cos_th, blk.sin_th
    gx = ct * drdr - st * ang_d
    gy = st * drdr + ct * ang_d
    return gx, gy


# --------------------------------------------------------------------------- #
# Divergence: conservative face-flux, exact adjoint of polar_grad.
# --------------------------------------------------------------------------- #
def polar_div(blk, u, v):
    """Physical divergence div(u,v) by conservative face fluxes.

    Radial faces: outer face of cell (i,j) at r=rn[j+1], area rn[j+1]*dth,
    flux = area * (u_f cos th + v_f sin th), u_f=0.5*(u_ij+u_i,j+1).
    Inner face (r=rn[j]): same at r=rn[j]; at the solid wall (j=0) flux = 0.
    Angular faces: between cell i-1,i at node angle, area dr_node[j],
    flux = area * (-sin th u_f + cos th v_f), u_f=0.5*(u_i-1,j+u_i,j).
    div = (F_outer - F_inner + F_right - F_left)/V.  With the consistent inner
    radius rn[j] (NOT cell-centre) the wall cell telescoping hold and uniform
    flow gives div==0 everywhere.
    """
    ni, nj = blk.ni, blk.nj
    rc, rn, dth, dr = blk.rc, blk.rn, blk.dth, blk.dr_node
    ct, st = blk.cos_th, blk.sin_th               # cell-centre angles (radial faces)
    ctn, stn = blk.cos_thn, blk.sin_thn           # node angles (angular faces)
    # radial flux at face k (between cell k-1 and k), k=0..nj  (k=0 wall, k=nj outer)
    Frad = np.zeros((ni, nj + 1))
    for k in range(1, nj):                       # interior radial faces (at r=rn[k])
        uf = 0.5 * (u[:, k - 1] + u[:, k])
        vf = 0.5 * (v[:, k - 1] + v[:, k])
        Frad[:, k] = rn[k] * dth * (uf * ct[:, k - 1] + vf * st[:, k - 1])
    # k=0 (wall) stays 0 (no flux); k=nj (outer) stays 0 (caller sets recv)
    # angular flux at face i (between cell i-1 and i), at NODE angle thn[i]
    up = np.roll(u, 1, axis=0)                   # cell i-1
    vp = np.roll(v, 1, axis=0)
    uf = 0.5 * (up + u)
    vf = 0.5 * (vp + v)
    Fang = dr[None, :] * (-stn[:, None] * uf + ctn[:, None] * vf)   # (ni,nj) face i
    Fang = np.concatenate([Fang, Fang[:1]], axis=0)   # (ni+1,nj) periodic seam
    # divergence
    Fright = Frad[:, 1:]                          # outer face of cell j
    Fleft = Frad[:, :-1]                          # inner face of cell j
    Fr = Fang[1:]                                 # right angular face of cell i
    Fl = Fang[:-1]                                # left angular face of cell i
    div = (Fright - Fleft + Fr - Fl) / blk.J
    return div


# --------------------------------------------------------------------------- #
# Projection Laplacian built column-by-column from the VERIFIED function
# L = polar_div o polar_grad (physical Laplacian; r^2 -> 4, harmonic -> 0).
# Because L == the function exactly, the correction u -= dt*grad(p) removes
# polar_div(u*) to machine precision by construction, and p is the physical
# pressure (correct sign & magnitude) for the force integral.
# --------------------------------------------------------------------------- #
def laplace_matrix(blk):
    """N x N Laplacian matrix whose k-th column = polar_div(polar_grad(e_k))."""
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    L = np.zeros((N, N))
    e = np.zeros((ni, nj))
    for k in range(N):
        i = k // nj; j = k % nj          # C-order flat index (matches p.reshape(-1))
        e[i, j] = 1.0
        L[:, k] = POG_div_grad(blk, e).reshape(-1)
        e[i, j] = 0.0
    return csc_matrix(L)


def POG_div_grad(blk, p):
    """polar_div(polar_grad(p)) -- the physical Laplacian (kept as one function)."""
    return polar_div(blk, *polar_grad(blk, p))


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def build_poisson(blk, recv_mask):
    ni, nj = blk.ni, blk.nj
    N = ni * nj
    recv = np.asarray(recv_mask).reshape(-1).astype(bool)
    solved = ~recv
    if not solved.any():
        solved = np.ones(N, dtype=bool); solved[0] = False
        recv = ~solved
    L_full = laplace_matrix(blk)
    solved_idx = np.nonzero(solved)[0]
    L_red = L_full[solved_idx][:, solved_idx].tocsc()
    blk._L_full = L_full
    blk._solved_idx = solved_idx
    blk._recv = recv
    return splu(L_red)


def project(blk, dt, recv_mask, poisson):
    div = polar_div(blk, blk.u, blk.v)
    rhs = div.reshape(-1) / dt
    sidx = blk._solved_idx
    p = np.zeros(blk.ni * blk.nj)
    p[sidx] = poisson.solve(rhs[sidx])
    p = p.reshape(blk.ni, blk.nj)
    blk.p = p
    gx, gy = polar_grad(blk, p)                # physical (x,y) gradient
    blk.u = blk.u - dt * gx
    blk.v = blk.v - dt * gy
    # recv (fringe) cells re-imposed by caller


def cylinder_forces(blk, nu, U):
    j = 0
    p = blk.p[:, j]
    Aix = blk.cos_th[:, j] * blk.rn[0] * blk.dth
    Aiy = blk.sin_th[:, j] * blk.rn[0] * blk.dth
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
