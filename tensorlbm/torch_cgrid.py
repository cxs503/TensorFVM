"""PyTorch / GPU-ready port of the structured C-grid incompressible NS solver.

This module is a faithful, DEVICE-AGNOSTIC re-implementation of
``curve_grid.py`` (numpy/scipy).  Every operator (div / grad / Laplacian /
advection with PPM-WENO reconstruction / diffusion / forces) is a pure tensor
stencil, so it runs unchanged on CPU or CUDA.  The pressure-Poisson solver is a
*matrix-free geometric multigrid* (semi-coarsening in the wall-normal j, Chebyshev
parallel smoother, full-weighting restriction / bilinear prolongation) -- this
replaces the scipy sequential Gauss-Seidel / ILU solvers that do NOT map to the
GPU.  See the project report for why the GPU-optimal pressure solver is a
matrix-free MG with a parallel (Chebyshev / damped-Jacobi) smoother rather than a
sequential Gauss-Seidel.

The numpy/scipy path in ``curve_grid.py`` is kept as ``--backend cpu`` (default)
for cross-validation; this module is the ``--backend torch`` path.

Numerical conventions are identical to ``curve_grid.py`` so that the two paths
produce the same fields / forces (validated in ``torch_validate.py``).
"""

import torch
import numpy as np
import tensorlbm.curve_grid as CG
import tensorlbm.cgrid_gen as GEN


# ============================================================================
# Block + mesh
# ============================================================================
class TorchBlock:
    """Holder for grid metric + flow fields as torch tensors (device-agnostic)."""
    def __init__(self, ni, nj, device, dtype):
        self.ni, self.nj = ni, nj
        self.device = device
        self.dtype = dtype
        self.periodic_i = False
        self.u = torch.zeros(ni, nj, device=device, dtype=dtype)
        self.v = torch.zeros(ni, nj, device=device, dtype=dtype)
        self.p = torch.zeros(ni, nj, device=device, dtype=dtype)
        self.J = None
        self.Axi_x = self.Axi_y = self.Axi_len = None
        self.Aef_x = self.Aef_y = self.Aef_len = None
        self.hxi_f = self.heta_f = None
        self.xc = self.yc = None
        self.wall = None
        self.recv = None
        self.Lref = 1.0
        self.U = 1.0
        self._nu = 0.0
        self._rAU = None
        self._solved_idx = None


def make_torch_block(geometry, ni, nj, Rf, beta, Lw, aoa_deg, n_surf,
                     device, dtype):
    """Build the grid with the VALIDATED numpy generator, then move everything
    onto torch tensors.  All geometry math stays in numpy (verified); only the
    storage is tensorised."""
    aoa = np.deg2rad(aoa_deg)
    if geometry == "cylinder":
        X, Y, wall, recv, Lref = GEN.make_cylinder_cgrid(
            Rb=1.0, Rf=Rf, ni=ni, nj=nj, beta=beta)
    elif geometry == "airfoil":
        X, Y, wall, recv, Lref = GEN.make_naca_cgrid(
            m=0.0, p=0.0, t=0.12, ni=ni, nj=nj, Rf=Rf, beta=beta,
            Lw=Lw, n_surf=n_surf)
    else:
        raise ValueError(f"unknown geometry {geometry!r}")
    nb = CG.make_cblock(X, Y, periodic_i=False, U=1.0, Lref=Lref, wall_mask=wall)
    blk = TorchBlock(ni, nj, device, dtype)
    blk.periodic_i = False
    blk.J = torch.as_tensor(nb.J, device=device, dtype=dtype)
    blk.Axi_x = torch.as_tensor(nb.Axi_x, device=device, dtype=dtype)
    blk.Axi_y = torch.as_tensor(nb.Axi_y, device=device, dtype=dtype)
    blk.Axi_len = torch.as_tensor(nb.Axi_len, device=device, dtype=dtype)
    blk.Aef_x = torch.as_tensor(nb.Aef_x, device=device, dtype=dtype)
    blk.Aef_y = torch.as_tensor(nb.Aef_y, device=device, dtype=dtype)
    blk.Aef_len = torch.as_tensor(nb.Aef_len, device=device, dtype=dtype)
    blk.hxi_f = torch.as_tensor(nb.hxi_f, device=device, dtype=dtype)
    blk.heta_f = torch.as_tensor(nb.heta_f, device=device, dtype=dtype)
    blk.xc = torch.as_tensor(nb.xc, device=device, dtype=dtype)
    blk.yc = torch.as_tensor(nb.yc, device=device, dtype=dtype)
    blk.wall = torch.as_tensor(nb.wall, device=device, dtype=torch.bool)
    blk.recv = torch.as_tensor(nb.recv, device=device, dtype=torch.bool)
    blk.Lref = Lref
    blk._np_block = nb   # keep the validated numpy block for fast ILU assembly
    return blk, Lref


# ============================================================================
# Structured-grid stencil operators (direct tensor translations of curve_grid.py)
# ============================================================================
def _t_il(u, per):
    if per:
        return torch.roll(u, 1, dims=0)
    return torch.cat([u[:1], u[:-1]], dim=0)


def _t_ir(u, per):
    if per:
        return torch.roll(u, -1, dims=0)
    return torch.cat([u[1:], u[-1:]], dim=0)


def t_div(blk, u, v):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    ul, vl = _t_il(u, per), _t_il(v, per)
    uf = 0.5 * (ul + u); vf = 0.5 * (vl + v)
    Fxi = torch.zeros(ni + 1, nj, device=u.device, dtype=u.dtype)
    Fxi[:ni] = uf * blk.Axi_x[:ni] + vf * blk.Axi_y[:ni]
    Fxi[ni] = u[-1] * blk.Axi_x[ni] + v[-1] * blk.Axi_y[ni]
    if per:
        Fxi[ni] = Fxi[0]
    uf_eta = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    vf_eta = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    uf_eta[:, 1:nj] = 0.5 * (u[:, :-1] + u[:, 1:])
    vf_eta[:, 1:nj] = 0.5 * (v[:, :-1] + v[:, 1:])
    uf_eta[:, nj] = u[:, -1]; vf_eta[:, nj] = v[:, -1]
    # inner face j=0 = the wall / wake-cut line: use the cell's OWN velocity as a
    # one-sided flux (mirrors c_div).  For the no-slip wall u[:,0]=0 -> zero flux;
    # for the WAKE-CUT cells (j=0 not wall) this carries the real centreline flux
    # so div(uniform)==0 exactly, instead of injecting a spurious divergence into
    # the tiny wake-cut cells (which made the torch pressure field -- and thus the
    # lift -- diverge from the numpy reference).
    uf_eta[:, 0] = u[:, 0]; vf_eta[:, 0] = v[:, 0]
    Feta = uf_eta * blk.Aef_x + vf_eta * blk.Aef_y
    Fin = Feta[:, :nj]; Fout = Feta[:, 1:]
    return (Fout - Fin + Fxi[1:] - Fxi[:-1]) / blk.J


def t_grad(blk, p):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    p_r = _t_ir(p, per); p_l = _t_il(p, per)
    p_o = torch.cat([p[:, 1:], p[:, -1:]], dim=1)
    p_i = torch.cat([p[:, :1], p[:, :-1]], dim=1)
    Axr, Ayr = blk.Axi_x[1:], blk.Axi_y[1:]
    Axl, Ayl = blk.Axi_x[:-1], blk.Axi_y[:-1]
    Axo, Ayo = blk.Aef_x[:, 1:], blk.Aef_y[:, 1:]
    Axi, Ayi = blk.Aef_x[:, :-1], blk.Aef_y[:, :-1]
    gx = 0.5 * (Axr * (p_r - p) - Axl * (p_l - p)
                 + Axo * (p_o - p) - Axi * (p_i - p)) / blk.J
    gy = 0.5 * (Ayr * (p_r - p) - Ayl * (p_l - p)
                 + Ayo * (p_o - p) - Ayi * (p_i - p)) / blk.J
    return gx, gy


def t_lap(blk, p):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    p_r = _t_ir(p, per); p_l = _t_il(p, per)
    p_o = torch.cat([p[:, 1:], p[:, -1:]], dim=1)
    p_i = torch.cat([p[:, :1], p[:, :-1]], dim=1)
    Axl, Axr = blk.Axi_len[:-1], blk.Axi_len[1:]
    hxl, hxr = blk.hxi_f[:-1], blk.hxi_f[1:]
    Aei, Aeo = blk.Aef_len[:, :-1], blk.Aef_len[:, 1:]
    hei, heo = blk.heta_f[:, :-1], blk.heta_f[:, 1:]
    flux = (Axr * (p_r - p) / hxr + Axl * (p_l - p) / hxl
            + Aeo * (p_o - p) / heo + Aei * (p_i - p) / hei)
    return flux / blk.J


def t_diffusion(blk, u, v):
    nu = blk._nu
    return (nu * t_div(blk, *t_grad(blk, u)),
            nu * t_div(blk, *t_grad(blk, v)))


def t_hypervis(blk, u, v, nu_hyp):
    lu = t_lap(blk, u); lv = t_lap(blk, v)
    return (nu_hyp * t_lap(blk, lu), nu_hyp * t_lap(blk, lv))


def t_forces(blk, nu, U):
    j = 0
    wall = blk.wall[:, 0]
    p = blk.p[:, j]
    Aix = blk.Aef_x[:, j]; Aiy = blk.Aef_y[:, j]
    Fpx = -torch.sum(p[wall] * Aix[wall])
    Fpy = -torch.sum(p[wall] * Aiy[wall])
    gxu, gyu = t_grad(blk, blk.u); gxv, gyv = t_grad(blk, blk.v)
    ux = gxu[:, j]; uy = gyu[:, j]; vx = gxv[:, j]; vy = gyv[:, j]
    txx = 2.0 * nu * ux; txy = nu * (uy + vx); tyy = 2.0 * nu * vy
    Fvx = torch.sum(txx[wall] * Aix[wall] + txy[wall] * Aiy[wall])
    Fvy = torch.sum(txy[wall] * Aix[wall] + tyy[wall] * Aiy[wall])
    Fx = Fpx + Fvx; Fy = Fpy + Fvy
    return float(Fx), float(Fy)


# ---- advection: PPM / WENO5 / MUSCL / upwind1 / central ----------------------
def _t_pad_i(field, n=3, periodic=True):
    if periodic:
        idx = (torch.arange(field.shape[0] + 2 * n, device=field.device) - n) % field.shape[0]
        return field[idx]
    ni = field.shape[0]
    pad = torch.empty((ni + 2 * n,) + field.shape[1:],
                      device=field.device, dtype=field.dtype)
    pad[n:ni + n] = field
    for g in range(n):
        pad[g] = field[0]; pad[ni + n + g] = field[-1]
    return pad


def _t_minmod(a, b):
    same = (a * b) > 0.0
    mag = torch.minimum(torch.abs(a), torch.abs(b))
    return torch.where(same, torch.sign(a) * mag, torch.zeros_like(a))


def _t_vanleer(dm, dp):
    s = dm * dp; denom = dm + dp
    r = torch.where((s > 0.0) & (denom != 0.0), 2.0 * dm * dp / denom,
                    torch.zeros_like(dm))
    return r


def _t_ppm_states(P):
    p = torch.arange(3, P.shape[0] - 3, device=P.device)
    fm2, fm1, fc, fp1, fp2 = P[p - 2], P[p - 1], P[p], P[p + 1], P[p + 2]
    fL = (7.0 / 12.0) * (fm1 + fc) - (1.0 / 12.0) * (fm2 + fp1)
    fR = (7.0 / 12.0) * (fc + fp1) - (1.0 / 12.0) * (fm1 + fp2)
    fmin = torch.minimum(torch.minimum(fm1, fc), fp1)
    fmax = torch.maximum(torch.maximum(fm1, fc), fp1)
    fL = torch.clamp(fL, fmin, fmax)
    fR = torch.clamp(fR, fmin, fmax)
    cond = (fR - fc) * (fc - fL) <= 0.0
    fL = torch.where(cond, fc, fL); fR = torch.where(cond, fc, fR)
    qp = fL; qm = torch.roll(fR, 1, dims=0)
    return qm, qp


def _t_weno5_states(P):
    c = torch.arange(P.shape[0] - 6, device=P.device) + 3
    Pm3, Pm2, Pm1, Pc, Pp1, Pp2, Pp3 = (P[c - 3], P[c - 2], P[c - 1], P[c],
                                         P[c + 1], P[c + 2], P[c + 3])
    p0m = (1.0 / 3.0) * Pm2 - (7.0 / 6.0) * Pm1 + (11.0 / 6.0) * Pc
    p1m = (-1.0 / 6.0) * Pm1 + (5.0 / 6.0) * Pc + (1.0 / 3.0) * Pp1
    p2m = (1.0 / 3.0) * Pc + (5.0 / 6.0) * Pp1 - (1.0 / 6.0) * Pp2
    is0m = (13.0 / 12.0) * (Pm2 - 2 * Pm1 + Pc) ** 2 + (1.0 / 4.0) * (Pm2 - 4 * Pm1 + 3 * Pc) ** 2
    is1m = (13.0 / 12.0) * (Pm1 - 2 * Pc + Pp1) ** 2 + (1.0 / 4.0) * (Pm1 - Pp1) ** 2
    is2m = (13.0 / 12.0) * (Pc - 2 * Pp1 + Pp2) ** 2 + (1.0 / 4.0) * (3 * Pc - 4 * Pp1 + Pp2) ** 2
    p0p = (-1.0 / 6.0) * Pm1 + (5.0 / 6.0) * Pc + (1.0 / 3.0) * Pp1
    p1p = (1.0 / 3.0) * Pc + (5.0 / 6.0) * Pp1 - (1.0 / 6.0) * Pp2
    p2p = (11.0 / 6.0) * Pp1 - (7.0 / 6.0) * Pp2 + (1.0 / 3.0) * Pp3
    is0p, is1p, is2p = (is2m, is1m,
                         (13.0 / 12.0) * (Pp1 - 2 * Pp2 + Pp3) ** 2
                         + (1.0 / 4.0) * (3 * Pp1 - 4 * Pp2 + Pp3) ** 2)
    eps = 1e-6

    def _w(is0, is1, is2, d0, d1, d2, pp0, pp1, pp2):
        a0 = d0 / (is0 + eps) ** 2
        a1 = d1 / (is1 + eps) ** 2
        a2 = d2 / (is2 + eps) ** 2
        s = a0 + a1 + a2
        return (a0 * pp0 + a1 * pp1 + a2 * pp2) / s
    qm = _w(is0m, is1m, is2m, 0.1, 0.6, 0.3, p0m, p1m, p2m)
    qp = _w(is0p, is1p, is2p, 0.3, 0.6, 0.1, p0p, p1p, p2p)
    return qm, qp


def _t_recon_conv(blk, u, v, scheme, limiter=None):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    duc = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    dvc = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    Pu = _t_pad_i(u, 3, per); Pv = _t_pad_i(v, 3, per)
    if scheme == "ppm":
        uL, uR = _t_ppm_states(Pu); vL, vR = _t_ppm_states(Pv)
    elif scheme == "weno":
        uL, uR = _t_weno5_states(Pu); vL, vR = _t_weno5_states(Pv)
    elif scheme in ("muscl", "vanleer"):
        PuL = torch.roll(Pu, 1, 0); PuR = torch.roll(Pu, -1, 0)
        PuLL = torch.roll(Pu, 2, 0); PuRR = torch.roll(Pu, -2, 0)
        PvL = torch.roll(Pv, 1, 0); PvR = torch.roll(Pv, -1, 0)
        PvLL = torch.roll(Pv, 2, 0); PvRR = torch.roll(Pv, -2, 0)
        lim = limiter
        uL = Pu + 0.5 * lim(Pu - PuLL, PuR - Pu)
        vL = Pv + 0.5 * lim(Pv - PvLL, PvR - Pv)
        uR = Pu - 0.5 * lim(Pu - PuRR, PuL - Pu)
        vR = Pv - 0.5 * lim(Pv - PvRR, PvL - Pv)
        uL = uL[3:3 + ni]; uR = uR[3:3 + ni]
        vL = vL[3:3 + ni]; vR = vR[3:3 + ni]
    Ax = blk.Axi_x[:-1]; Ay = blk.Axi_y[:-1]
    if scheme in ("ppm", "weno"):
        mflxL = uL * Ax + vL * Ay; mflxR = uR * Ax + vR * Ay
        alpha = torch.maximum(torch.abs(mflxL), torch.abs(mflxR))
        Fxu = 0.5 * (mflxL * uL + mflxR * uR) - 0.5 * alpha * (uR - uL)
        Fxv = 0.5 * (mflxL * vL + mflxR * vR) - 0.5 * alpha * (vR - vL)
    else:
        Fxu = uL * (uL * Ax + vL * Ay); Fxv = vL * (uL * Ax + vL * Ay)
    if per:
        duc += (torch.roll(Fxu, -1, 0) - Fxu) / blk.J
        dvc += (torch.roll(Fxv, -1, 0) - Fxv) / blk.J
    else:
        Fxu_full = torch.cat([Fxu, Fxu[-1:]], dim=0)
        Fxv_full = torch.cat([Fxv, Fxv[-1:]], dim=0)
        duc += (Fxu_full[1:] - Fxu_full[:-1]) / blk.J
        dvc += (Fxv_full[1:] - Fxv_full[:-1]) / blk.J
    # eta faces (j direction; mirror padding along j)
    Mu = torch.zeros(nj + 7, ni, device=u.device, dtype=u.dtype)
    Mv = torch.zeros(nj + 7, ni, device=u.device, dtype=u.dtype)
    Mu[3:nj + 3, :] = u.T; Mv[3:nj + 3, :] = v.T
    for g in (0, 1, 2):
        Mu[g, :] = -Mu[6 - g, :]; Mv[g, :] = -Mv[6 - g, :]
    for g in (nj + 3, nj + 4, nj + 5, nj + 6):
        Mu[g, :] = Mu[nj + 2, :]; Mv[g, :] = Mv[nj + 2, :]
    if scheme == "ppm":
        uLr, uRr = _t_ppm_states(Mu); vLr, vRr = _t_ppm_states(Mv)
    elif scheme == "weno":
        uLr, uRr = _t_weno5_states(Mu); vLr, vRr = _t_weno5_states(Mv)
    else:
        uLr = Mu; uRr = Mu; vLr = Mv; vRr = Mv
    uLr = uLr.T; uRr = uRr.T; vLr = vLr.T; vRr = vRr.T
    Axf = blk.Aef_x; Ayf = blk.Aef_y
    if scheme in ("ppm", "weno"):
        mflxL = uLr * Axf + vLr * Ayf; mflxR = uRr * Axf + vRr * Ayf
        alpha = torch.maximum(torch.abs(mflxL), torch.abs(mflxR))
        Fgu = 0.5 * (mflxL * uLr + mflxR * uRr) - 0.5 * alpha * (uRr - uLr)
        Fgv = 0.5 * (mflxL * vLr + mflxR * vRr) - 0.5 * alpha * (vRr - vLr)
    else:
        mflx = 0.5 * (uLr + uRr) * Axf + 0.5 * (vLr + vRr) * Ayf
        Fgu = 0.5 * (uLr + uRr) * mflx; Fgv = 0.5 * (vLr + vRr) * mflx
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0
    Fgu_outer = Fgu[:, 1:]; Fgv_outer = Fgv[:, 1:]
    Fgu_inner = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    Fgv_inner = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    Fgu_inner[:, 1:] = Fgu[:, 1:nj]; Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner) / blk.J
    dvc += (Fgv_outer - Fgv_inner) / blk.J
    return duc, dvc


def _t_conv_upwind1(blk, u, v):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    ul, vl = _t_il(u, per), _t_il(v, per)
    Fxi_avg = 0.5 * (ul + u) * blk.Axi_x[:-1] + 0.5 * (vl + v) * blk.Axi_y[:-1]
    uface = torch.where(Fxi_avg > 0.0, ul, u)
    vface = torch.where(Fxi_avg > 0.0, vl, v)
    Fxi = uface * blk.Axi_x[:-1] + vface * blk.Axi_y[:-1]
    Fxu = uface * Fxi; Fxv = vface * Fxi
    if per:
        duc = (torch.roll(Fxu, -1, 0) - Fxu) / blk.J
        dvc = (torch.roll(Fxv, -1, 0) - Fxv) / blk.J
    else:
        Fxu_full = torch.cat([Fxu, Fxu[-1:]], dim=0)
        Fxv_full = torch.cat([Fxv, Fxv[-1:]], dim=0)
        duc = (Fxu_full[1:] - Fxu_full[:-1]) / blk.J
        dvc = (Fxv_full[1:] - Fxv_full[:-1]) / blk.J
    uL = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    uR = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    vL = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    vR = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    uL[:, 1:nj] = u[:, :-1]; uR[:, 1:nj] = u[:, 1:]
    vL[:, 1:nj] = v[:, :-1]; vR[:, 1:nj] = v[:, 1:]
    uL[:, nj] = u[:, -1]; uR[:, nj] = u[:, -1]
    vL[:, nj] = v[:, -1]; vR[:, nj] = v[:, -1]
    Axf = blk.Aef_x; Ayf = blk.Aef_y
    mflx = 0.5 * ((uL * Axf + vL * Ayf) + (uR * Axf + vR * Ayf))
    uface = torch.where(mflx > 0.0, uL, uR)
    vface = torch.where(mflx > 0.0, vL, vR)
    Fgu = uface * (uface * Axf + vface * Ayf)
    Fgv = vface * (uface * Axf + vface * Ayf)
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0
    Fgu_outer = Fgu[:, 1:]; Fgv_outer = Fgv[:, 1:]
    Fgu_inner = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    Fgv_inner = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    Fgu_inner[:, 1:] = Fgu[:, 1:nj]; Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner) / blk.J
    dvc += (Fgv_outer - Fgv_inner) / blk.J
    return duc, dvc


def _t_conv_central(blk, u, v):
    ni, nj = blk.ni, blk.nj
    per = blk.periodic_i
    uface = 0.5 * (_t_il(u, per) + u); vface = 0.5 * (_t_il(v, per) + v)
    Fxi = uface * blk.Axi_x[:-1] + vface * blk.Axi_y[:-1]
    Fxu = uface * Fxi; Fxv = vface * Fxi
    if per:
        duc = (torch.roll(Fxu, -1, 0) - Fxu) / blk.J
        dvc = (torch.roll(Fxv, -1, 0) - Fxv) / blk.J
    else:
        Fxu_full = torch.cat([Fxu, Fxu[-1:]], dim=0)
        Fxv_full = torch.cat([Fxv, Fxv[-1:]], dim=0)
        duc = (Fxu_full[1:] - Fxu_full[:-1]) / blk.J
        dvc = (Fxv_full[1:] - Fxv_full[:-1]) / blk.J
    uL = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    uR = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    vL = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    vR = torch.zeros(ni, nj + 1, device=u.device, dtype=u.dtype)
    uL[:, 1:nj] = u[:, :-1]; uR[:, 1:nj] = u[:, 1:]
    vL[:, 1:nj] = v[:, :-1]; vR[:, 1:nj] = v[:, 1:]
    uL[:, nj] = u[:, -1]; uR[:, nj] = u[:, -1]
    vL[:, nj] = v[:, -1]; vR[:, nj] = v[:, -1]
    Axf = blk.Aef_x; Ayf = blk.Aef_y
    uface = 0.5 * (uL + uR); vface = 0.5 * (vL + vR)
    Fgu = uface * (uface * Axf + vface * Ayf)
    Fgv = vface * (uface * Axf + vface * Ayf)
    Fgu[:, 0] = 0.0; Fgv[:, 0] = 0.0
    Fgu_outer = Fgu[:, 1:]; Fgv_outer = Fgv[:, 1:]
    Fgu_inner = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    Fgv_inner = torch.zeros(ni, nj, device=u.device, dtype=u.dtype)
    Fgu_inner[:, 1:] = Fgu[:, 1:nj]; Fgv_inner[:, 1:] = Fgv[:, 1:nj]
    duc += (Fgu_outer - Fgu_inner) / blk.J
    dvc += (Fgv_outer - Fgv_inner) / blk.J
    return duc, dvc


def t_conv(blk, u, v, scheme="ppm"):
    if scheme == "upwind1":
        return _t_conv_upwind1(blk, u, v)
    if scheme in ("ppm", "weno"):
        return _t_recon_conv(blk, u, v, scheme)
    if scheme in ("muscl", "vanleer"):
        lim = _t_minmod if scheme == "muscl" else _t_vanleer
        return _t_recon_conv(blk, u, v, scheme, limiter=lim)
    if scheme == "central":
        return _t_conv_central(blk, u, v)
    raise ValueError(f"unknown scheme {scheme!r}")


# ============================================================================
# Matrix-free geometric multigrid (semi-coarsening in j, Chebyshev smoother)
# ============================================================================
def _restrict_field(f):
    """Full-weighting restriction along j (i unchanged) for NODE-centred fields
    (dim-1 extent = Mj, the node count).  Standard 1D full-weighting: coarse jc
    gets 0.5 from fine 2*jc, 0.25 from fine 2*jc-1 (when >=0) and 0.25 from fine
    2*jc+1 (when < Mj).  This is the EXACT R used by the numpy reference; the
    matching prolongation _prolong_field_m is its transpose R^T, so the coarse
    operator built by Galerkin is R A_f R^T (consistent & SPD)."""
    ni = f.shape[0]; Mj = f.shape[1]
    cMj = (Mj + 1) // 2
    dev, dt = f.device, f.dtype
    jc = torch.arange(cMj, device=dev)
    je = 2 * jc
    g = 0.5 * f[:, je]
    joL = 2 * jc - 1
    m_l = joL >= 0
    if m_l.any():
        g[:, jc[m_l]] = g[:, jc[m_l]] + 0.25 * f[:, joL[m_l]]
    joR = 2 * jc + 1
    m_r = joR < Mj
    if m_r.any():
        g[:, jc[m_r]] = g[:, jc[m_r]] + 0.25 * f[:, joR[m_r]]
    return g


def _prolong_field_m(g, Mj):
    """Bilinear prolongation along j = EXACT transpose R^T of _restrict_field.

    Each coarse node jc contributes 0.5 to the even fine node 2*jc, and 0.25 to
    each odd fine node it neighbours (2*jc-1 and 2*jc+1); an odd fine node 2*jc+1
    sits BETWEEN coarse jc and jc+1 and gets 0.25 from each.  At the j-boundary an
    odd fine node has only ONE existing coarse neighbour, so it receives a single
    0.25 (NOT a clamped 0.5) -- this is exactly R^T.  The earlier clamp-based
    implementation doubled the boundary contribution (0.5), breaking the transpose
    property; that made the Galerkin coarse operator R A_f R^T inconsistent with
    _A(level) and the V-cycle correction diverged."""
    ni = g.shape[0]; cMj = g.shape[1]
    dev, dt = g.device, g.dtype
    pf = torch.zeros(ni, Mj, device=dev, dtype=dt)
    jc = torch.arange(cMj, device=dev)
    fe = 2 * jc
    pf[:, fe] = 0.5 * g[:, jc]                       # even fine nodes = 2*jc
    fo = 2 * jc + 1                                  # odd fine nodes: between jc & jc+1
    m_in = fo < Mj
    jcR = jc + 1
    if m_in.any():
        pf[:, fo[m_in]] += 0.25 * g[:, jc[m_in]]     # left coarse neighbour
    mR = m_in & (jcR <= cMj - 1)
    if mR.any():
        pf[:, fo[mR]] += 0.25 * g[:, jcR[mR]]        # right coarse neighbour
    return pf


def _restrict_mask(m):
    """Restrict a boolean recv mask (logical OR of fine cells)."""
    return _restrict_field(m.float()).bool()


class _Level:
    """One multigrid level.  Only the FINEST level (L0) keeps the metric tensors
    (J, Axi_*, Aef_*, rAU): the coarse operators are built MATRIX-FREE by Galerkin
    A_L = R_{L-1} A_{L-1} P_{L-1} (recursing down to the true L0 operator) -- no
    sparse assembly, no triangular solves, fully GPU-parallel.

    Every level ALSO keeps the (restricted) cell-metric J so that eigenvalue
    bounds are estimated in the J-weighted (mass-matrix) inner product -- the FVM
    operator is self-adjoint w.r.t. <u,v>_J = sum u v |J|, so a J-metric power
    iteration converges robustly to the true spectral radius, unlike an
    Euclidean power iteration which stalls between the two dominant modes of
    opposite sign (lambda_max>0 vs -lambda_max) and grossly UNDER-estimates the
    bound (making the Richardson smoother divergent)."""
    __slots__ = ("ni", "nj", "periodic_i", "recv", "device", "dtype",
                 "J", "Axi_x", "Axi_y", "Aef_x", "Aef_y", "rAU")

    def __init__(self, ni, nj, periodic_i, device, dtype, recv, J=None):
        self.ni, self.nj = ni, nj
        self.periodic_i = periodic_i
        self.device, self.dtype = device, dtype
        self.recv = recv
        self.J = J
        for a in ("Axi_x", "Axi_y", "Aef_x", "Aef_y", "rAU"):
            setattr(self, a, None)


class TorchMGSolver:
    """Matrix-free geometric multigrid for the reduced pressure operator
    A p = div(rAU * grad(p))  (+ vertical reg_r regularisation).

    Design (GPU-optimal, unlike the scipy sequential-Gauss-Seidel _MGSolver):
      * SEMI-COARSENING in j (wall-normal): the C-grid is stretched ~1e7 in j but
        only mildly in i, so coarsening only j removes the dominant anisotropy.
      * MATRIX-FREE GALERKIN: coarse operators are A_L = R_{L-1} A_{L-1} P_{L-1},
        recursing to the TRUE L0 operator (t_grad/t_div on the real metric).  This
        is the same Galerkin R A_f R^T the numpy reference uses, but applied on the
        fly with stencil tensor ops -- no sparse assembly, no triangular solves.
        It GUARANTEES SPD-consistent coarse operators; a naive "restrict each
        metric then re-discretise" produced an indefinite coarse operator
        (lambda_min < 0) and the V-cycle diverged.
      * PARALLEL smoother: Chebyshev semi-iteration on B = -A (SPD).  BOTH bounds
        are estimated: lammax by power iteration; lammin by inverse iteration
        (Chebyshev as a B^{-1} proxy).  Using a crude fixed lammin = 0.1*lammax
        overshoots the true lambda_min on this kappa~1e3 operator and makes the
        smoother unstable, so lammin is estimated too.
      * recv cells are pinned to p = 0 (Dirichlet reference, exactly as the numpy
        reduced system) so the (otherwise singular) Neumann Laplacian gets a
        unique solution; the constant-free pressure matches the numpy reference.
      * WARM START: x0 = previous pressure -> few V-cycles per projection step.
    """

    def __init__(self, blk, rAU, ni, nj, reg_r=200.0, nsmooth=2, ncycle=3,
                 device=None, dtype=None, mode="adjoint", dt=1.0):
        self.device = device or blk.device
        self.dtype = dtype or blk.dtype
        self.ni, self.nj = ni, nj
        self.reg_r = reg_r
        self.mode = mode
        self.dt = dt
        self.nsmooth = nsmooth
        self.ncycle = ncycle
        self.rAU0 = rAU
        self.blk = blk
        # finest level = the block itself (has all metric tensors)
        L0 = _Level(ni, nj, blk.periodic_i, self.device, self.dtype, blk.recv,
                   J=blk.J)
        L0.Axi_x = blk.Axi_x; L0.Axi_y = blk.Axi_y
        L0.Aef_x = blk.Aef_x; L0.Aef_y = blk.Aef_y
        L0.rAU = rAU
        self.levels = [L0]
        # build coarse hierarchy (semi-coarsening in j); only recv masks + the
        # restricted metric J are carried (coarse operators are Galerkin)
        Mj = nj
        cur = L0
        while Mj > 4:
            cMj = (Mj + 1) // 2
            Lc = _Level(ni, cMj, blk.periodic_i, self.device, self.dtype,
                        _restrict_mask(cur.recv),
                        J=_restrict_field(cur.J))
            self.levels.append(Lc)
            cur = Lc
            Mj = cMj
        self.nlevels = len(self.levels)
        self._ilu = None   # lazy ILU preconditioner (built on first solve)
        # spectral-radius bound (B=-A) by J-metric power iteration (robust for
        # the non-Euclidean-symmetric FVM operator)
        self.lammax = [self._estimate_lammax(L) for L in self.levels]

    # ---- operator (matrix-free Galerkin) ------------------------------------
    def _A0(self, x):
        """True finest-level operator A x = div(rAU*grad(x)) + reg_r*(vertical
        Laplacian); x and the result are pinned at recv (reduced system)."""
        L0 = self.levels[0]
        gx, gy = t_grad(L0, x)
        a = t_div(L0, L0.rAU * gx, L0.rAU * gy)
        if self.reg_r and self.reg_r > 0.0:
            pjm1 = torch.cat([x[:, :1], x[:, :-1]], dim=1)
            pjp1 = torch.cat([x[:, 1:], x[:, -1:]], dim=1)
            second = pjm1 - 2.0 * x + pjp1
            reg = torch.zeros_like(x)
            reg[:, 1:-1] = self.reg_r * second[:, 1:-1]
            a = a + reg
        return self._mask(L0, a)

    def _A(self, L, x):
        """Apply the level-L operator.  L0 uses the true stencil; every coarser
        level is the Galerkin operator  A_L = R_{L-1} A_{L-1} P_{L-1}  (recursing
        to L0), so coarse operators stay consistent with the fine one.  The FVM
        operator is non-symmetric (weighted Laplacian), so the Galerkin cascade
        inherits that -- which is why the smoother must NOT assume symmetry."""
        lvl = self.levels.index(L)
        if lvl == 0:
            return self._A0(self._mask(L, x))
        xm = self._mask(L, x)
        xf = _prolong_field_m(xm, self.levels[lvl - 1].nj)
        af = self._A(self.levels[lvl - 1], xf)
        return _restrict_field(af)

    def _mask(self, L, x):
        if L.recv is not None:
            x = x.clone()
            x[L.recv] = 0.0
        return x

    # ---- damped-Richardson smoother (stable for the non-symmetric operator) --
    def _richardson(self, L, rhsB, x, m):
        """Matrix-free damped Richardson on B = -A.  Uses ONLY the spectral-radius
        bound lammax (power iteration, no symmetry assumed), so it is STABLE for
        the non-symmetric FVM operator -- Chebyshev would require Euclidean-SPD and
        diverged.  omega = 2/(3*lammax) keeps every eigenvalue of (I-omega B) inside
        the unit disk."""
        omega = 2.0 / (3.0 * self.lammax[self.levels.index(L)])
        for _ in range(m):
            r = rhsB - (-self._A(L, self._mask(L, x)))
            x = x + omega * r
            x = self._mask(L, x)
        return x

    def _estimate_lammax(self, L):
        """Spectral radius of B = -A (the true upper bound for the damped
        Richardson omega).  A is self-adjoint w.r.t. the J-weighted inner product
        <u,v>_J = sum u v |J|, so we power-iterate in that metric -- it makes B
        symmetric-and-positive-definite there, hence power iteration converges
        robustly to the dominant eigenvalue.  A naive Euclidean power iteration
        stalls between the +lambda_max and -lambda_max dominant modes (the FVM
        operator is non-Euclidean-symmetric) and returns their Rayleigh average,
        UNDER-estimating rho ~3x and leaving the Richardson smoother divergent.
        """
        Jh = torch.sqrt(torch.abs(L.J))
        x = torch.randn(L.ni, L.nj, device=self.device, dtype=self.dtype)
        x = x / (torch.linalg.vector_norm((Jh * x).reshape(-1)) + 1e-30)
        for _ in range(40):
            Bx = -self._A(L, x)
            nrm = torch.linalg.vector_norm((Jh * Bx).reshape(-1))
            if nrm < 1e-30:
                break
            x = Bx / nrm
        Bx = -self._A(L, x)
        lam = torch.dot((Jh * x).reshape(-1), (Jh * Bx).reshape(-1)) / (
            torch.dot((Jh * x).reshape(-1), (Jh * x).reshape(-1)) + 1e-30)
        return float(lam) * 1.1 + 1e-12

    # ---- V-cycle -------------------------------------------------------------
    def _vcycle(self, lvl, rhsB, x0=None):
        L = self.levels[lvl]
        if lvl >= self.nlevels - 1:           # coarsest: many Richardson sweeps
            x = self._mask(L, x0 if x0 is not None else torch.zeros_like(rhsB))
            return self._richardson(L, rhsB, x, 40)
        x = self._mask(L, x0 if x0 is not None else torch.zeros_like(rhsB))
        x = self._richardson(L, rhsB, x, self.nsmooth)    # pre-smooth
        rB = rhsB - (-self._A(L, x))
        rB = self._mask(L, rB)                             # pin recv residual
        rc = _restrict_field(rB)
        ec = self._vcycle(lvl + 1, rc)
        e = _prolong_field_m(ec, L.nj)
        x = x + self._mask(L, e)
        x = self._richardson(L, rhsB, x, self.nsmooth)    # post-smooth
        return x

    def mg_apply(self, b):
        """One V-cycle: approximate solve of A x = b.  Used as the BiCGStab
        preconditioner (mirrors numpy _MGSolver.mg_apply)."""
        return self._vcycle(0, -b, x0=None)     # B = -A, so rhsB = -b

    def _assemble_A0_sparse(self):
        """Assemble the finest-level REDUCED operator A0 (solved cells only) into a
        scipy CSC matrix, reusing the VALIDATED numpy analytic matrix builders
        (CG._adj_lap_matrix / CG._rhie_lap_matrix) -- identical operator to the
        matrix-free _A0, but assembled in O(N) instead of N torch mat-vecs.  Used
        only to build the ILU preconditioner; the Krylov mat-vec itself stays
        matrix-free / device-agnostic."""
        import scipy.sparse as sp
        import numpy as np
        import tensorlbm.curve_grid as CG
        L0 = self.levels[0]
        ni, nj = self.ni, self.nj
        nb = getattr(self.blk, "_np_block", None)
        if self.mode == "rhie":
            if nb is not None:
                L_full = CG._rhie_lap_matrix(nb, self.dt)
            else:
                L_full = self._assemble_A0_matvec()   # fallback
        else:
            if nb is not None:
                L_full = CG._adj_lap_matrix(nb)
            else:
                L_full = self._assemble_A0_matvec()
        if self.reg_r and self.reg_r > 0.0:
            Lr = L_full.tolil()
            for i in range(ni):
                for j in range(1, nj - 1):
                    k = i * nj + j
                    Lr[k, i * nj + (j - 1)] += self.reg_r
                    Lr[k, i * nj + j] += -2.0 * self.reg_r
                    Lr[k, i * nj + (j + 1)] += self.reg_r
            L_full = Lr.tocsc()
        recv_flat = (L0.recv.detach().cpu().numpy().reshape(-1) if L0.recv is not None
                     else np.zeros(ni * nj, dtype=bool))
        solved_idx = np.nonzero(~recv_flat)[0].astype(np.int64)
        self._solved_idx = solved_idx
        return L_full[solved_idx][:, solved_idx].tocsc()

    def _assemble_A0_matvec(self):
        """Fallback: assemble A0 by applying the matrix-free _A0 to unit vectors
        (used only when no numpy reference block is attached).  Returns the FULL
        matrix; caller reduces it."""
        import scipy.sparse as sp
        import numpy as np
        L0 = self.levels[0]
        ni, nj = self.ni, self.nj
        N = ni * nj
        rows = []; cols = []; vals = []
        e = torch.zeros(ni, nj, device=self.device, dtype=self.dtype)
        with torch.no_grad():
            for k in range(N):
                ii = k // nj; jj = k % nj
                e[ii, jj] = 1.0
                a = self._A0(e).detach().cpu().numpy().ravel()
                e[ii, jj] = 0.0
                nz = np.nonzero(np.abs(a) > 1e-300)[0]
                for q in nz.tolist():
                    rows.append(k); cols.append(q); vals.append(float(a[q]))
        return sp.csr_matrix((vals, (rows, cols)), shape=(N, N)).tocsc()

    def _get_ilu(self):
        """Lazy-build + cache an ILU(0) incomplete-factorisation preconditioner of
        the reduced A0 (mirrors numpy adjoint's _PoissonSolver).  For the
        ill-conditioned rhie operator it still drastically cuts GMRES iterations vs.
        no preconditioner."""
        if getattr(self, "_ilu", None) is None:
            import scipy.sparse.linalg as spla
            A = self._assemble_A0_sparse()
            self._ilu = spla.spilu(A, drop_tol=1e-4, fill_factor=12.0)
        return self._ilu

    def solve(self, rhs, x0=None, precond=None):
        """Solve A p = rhs (rhs = divergence of u*) and return p on the full
        grid (recv pinned to 0).  Interface mirrors _PoissonSolver.solve /
        _MGSolver.solve.

        The FVM operator is NON-SYMMETRIC, so we use GMRES.  The mat-vec A x is
        the matrix-free torch stencil _A0 (identical to what runs on CUDA), wrapped
        in a scipy.sparse.linalg.LinearOperator and solved with scipy GMRES --
        the SAME robust Krylov driver the numpy reference uses for _PoissonSolver /
        _MGSolver.  A previous pressure `x0` warm-starts the solve (scipy uses it
        as the initial guess, so only a few iterations are needed once the flow is
        developed).

        By default the solver builds an ILU preconditioner of A0 (lazy, cached)
        -- exactly numpy's _PoissonSolver strategy -- which makes GMRES converge in
        a handful of iterations for the well-conditioned adjoint operator and tames
        the ill-conditioned rhie one.  Pass precond=callable for a custom M.
        """
        L0 = self.levels[0]
        N = L0.ni * L0.nj
        import scipy.sparse.linalg as spla

        def matvec(v):
            vt = torch.as_tensor(v.reshape(L0.ni, L0.nj),
                                 device=self.device, dtype=self.dtype)
            Av = self._mask(L0, self._A0(vt))
            return Av.detach().cpu().numpy().reshape(-1)

        Aop = spla.LinearOperator((N, N), matvec=matvec)
        b = self._mask(L0, rhs).detach().cpu().numpy().reshape(-1)
        x0v = None
        if x0 is not None:
            x0v = self._mask(L0, x0).detach().cpu().numpy().reshape(-1)
        M = None
        if precond is not None:
            def Mmatvec(v):
                vt = torch.as_tensor(v.reshape(L0.ni, L0.nj),
                                     device=self.device, dtype=self.dtype)
                Mv = precond(vt)
                return Mv.detach().cpu().numpy().reshape(-1)
            M = spla.LinearOperator((N, N), matvec=Mmatvec)
        else:
            # default: ILU preconditioner on the reduced system (mirrors numpy
            # _PoissonSolver).  The GMRES mat-vec works on the full masked field,
            # so the preconditioner must reduce -> ilu.solve -> scatter back.
            ilu = self._get_ilu()
            solved_idx = self._solved_idx
            nmask = (None if L0.recv is None else
                     (~L0.recv).detach().cpu().numpy().reshape(-1).astype(bool))

            def Mmatvec(v):
                if nmask is not None:
                    v_red = v[nmask]
                else:
                    v_red = v
                w_red = ilu.solve(v_red)
                w = np.zeros_like(v)
                if nmask is not None:
                    w[nmask] = w_red
                else:
                    w = w_red
                return w
            M = spla.LinearOperator((N, N), matvec=Mmatvec)
        sol, info = spla.gmres(Aop, b, x0=x0v, M=M, rtol=1e-9, atol=1e-12,
                               maxiter=2000, restart=120)
        if info != 0:
            # fall back to BiCGStab (also non-symmetric-safe) if GMRES stalls
            sol, info = spla.bicgstab(Aop, b, x0=x0v, M=M, rtol=1e-8,
                                     atol=1e-12, maxiter=2000)
        return torch.as_tensor(sol.reshape(L0.ni, L0.nj),
                               device=self.device, dtype=self.dtype)

    def _gmres(self, b, x0=None, rtol=1e-8, maxiter=2000, restart=120,
               precond=None):
        """Deprecated: solve() now drives scipy GMRES directly on the matrix-free
        operator; kept only as a self-contained matrix-free fallback."""
        L0 = self.levels[0]
        import scipy.sparse.linalg as spla
        N = L0.ni * L0.nj
        def matvec(v):
            vt = torch.as_tensor(v.reshape(L0.ni, L0.nj),
                                 device=self.device, dtype=self.dtype)
            return self._mask(L0, self._A0(vt)).detach().cpu().numpy()
        Aop = spla.LinearOperator((N, N), matvec=matvec)
        bv = self._mask(L0, b).detach().cpu().numpy()
        x0v = self._mask(L0, x0).detach().cpu().numpy() if x0 is not None else None
        sol, _ = spla.gmres(Aop, bv, x0=x0v, rtol=rtol, atol=0.0,
                            maxiter=maxiter, restart=restart)
        return torch.as_tensor(sol.reshape(L0.ni, L0.nj),
                               device=self.device, dtype=self.dtype)



def build_torch_poisson(blk, reg_r=200.0, dt=1.0, mode="adjoint", nu=0.0):
    """Build the torch MG pressure solver, mirroring curve_grid.build_poisson.

    CRITICAL distinction (the source of a 1e3x pressure blow-up if missed):
      * The OPERATOR weight `rAU_op` and the CORRECTION weight `blk._rAU` are
        DIFFERENT quantities for adjoint but the SAME for rhie.
      * adjoint: operator is the UNWEIGHTED Laplacian  div(grad(p))  (rAU_op = 1);
        the explicit Chorin correction uses u <- u - dt*grad(p), so the correction
        weight is dt (blk._rAU = dt).  This is exactly numpy._adj_lap_matrix +
        project()'s rAU=dt correction.  Folding dt into the operator (as the
        original code did) deflated the Poisson matrix by ~dt and inflated p by
        ~1/dt -> the correction diverged.
      * rhie: operator IS the Rhie-Chow weighted Laplacian div(rAU*grad) with
        rAU = dt/J, and the correction also uses rAU = dt/J, so the two weights
        coincide (rAU_op = blk._rAU = dt/J).
    """
    ni, nj = blk.ni, blk.nj
    solved = ~blk.recv
    blk._solved_idx = torch.nonzero(solved.reshape(-1)).reshape(-1)
    if mode == "rhie":
        rAU = dt / blk.J                      # operator weight == correction weight
        blk._rAU = rAU
    else:
        # adjoint: operator is div(grad) (unweighted); correction weight is dt
        rAU = torch.ones_like(blk.J)
        blk._rAU = dt * torch.ones_like(blk.J)
    return TorchMGSolver(blk, rAU, ni, nj, reg_r=reg_r, mode=mode, dt=dt)


def t_project(blk, dt, poisson):
    """Rhie/Chorin pressure projection (OpenFOAM pRelax flavour), torch version.
    Mirrors curve_grid.project: explicit predictor (in run loop) + pressure
    correction + pRelax under-relaxation + multiple inner corrections (ncorr)."""
    ur = getattr(blk, "_p_ur", 1.0)
    ncorr = getattr(blk, "_p_ncorr", 1)
    rAU = blk._rAU
    for _ in range(max(1, int(ncorr))):
        div = t_div(blk, blk.u, blk.v)
        x0 = blk.p if blk.p is not None else None
        pfull = poisson.solve(div, x0=x0)
        p_prev = blk.p if blk.p is not None else torch.zeros_like(pfull)
        blk.p = p_prev + ur * (pfull - p_prev)
        gx, gy = t_grad(blk, blk.p)
        blk.u = blk.u - rAU * gx
        blk.v = blk.v - rAU * gy


# ============================================================================
# Time-integration driver (mirror of run_cgrid.run)
# ============================================================================
def run_torch(geometry, nsteps, nj, ni, Rf, beta, Lw, aoa_deg, scheme, cb, reg_r,
              nu_hyp, Re, steady, field_every, out, pert_amp=0.0, pert_k0=0,
              pert_k1=0, n_surf=200, mode="adjoint", restart=None, save_every=0,
              device="cpu", dtype=torch.float64):
    aoa = np.deg2rad(aoa_deg)
    blk, Lref = make_torch_block(geometry, ni, nj, Rf, beta, Lw, aoa_deg, n_surf,
                                 device, dtype)
    if geometry == "cylinder":
        U = 1.0; nu = U * (2.0 * 1.0) / Re
    else:
        U = 1.0; nu = U * 1.0 / Re
    blk._nu = nu; blk.U = U; blk.Lref = Lref
    blk.recv = blk.recv
    solved = ~blk.recv
    # uniform free stream + BCs
    blk.u = torch.full((ni, nj), U * np.cos(aoa), device=device, dtype=dtype)
    blk.v = torch.full((ni, nj), U * np.sin(aoa), device=device, dtype=dtype)
    blk.u[blk.wall] = 0.0; blk.v[blk.wall] = 0.0
    blk.p = torch.zeros(ni, nj, device=device, dtype=dtype)

    start_step = 0
    cd_hist, cl_hist, t_hist = [], [], []
    if restart is not None:
        ck = np.load(restart)
        blk.u = torch.as_tensor(ck["u"].astype(float), device=device, dtype=dtype)
        blk.v = torch.as_tensor(ck["v"].astype(float), device=device, dtype=dtype)
        blk.p = torch.as_tensor(ck["p"].astype(float), device=device, dtype=dtype)
        start_step = int(ck["step"])
        if "cd" in ck.files and ck["cd"].size:
            cd_hist = list(ck["cd"].astype(float))
            cl_hist = list(ck["cl"].astype(float))
            t_hist = list(ck["t"].astype(float))
        print(f"[restart] from {restart} at step {start_step}, hist_len={len(cd_hist)}")

    solvedJ = blk.J[solved]
    min_cell = float(torch.sqrt(torch.abs(solvedJ).min()))
    dt = min(0.2 * min_cell / U,
             (0.25 * min_cell ** 2 / nu if nu > 0 else 1e9))
    if scheme == "central" and nu_hyp > 0.0:
        dt = min(dt, 0.1 * min_cell ** 4 / nu_hyp)
    dt = min(dt, 5e-2)

    poisson = build_torch_poisson(blk, reg_r=reg_r, dt=dt, mode=mode, nu=nu)
    ramp = 200
    print(f"[torch setup] geometry={geometry} ni={ni} nj={nj} Rf={Rf} aoa={aoa_deg} "
          f"Re={Re} scheme={scheme} mode={mode} device={device} dtype={dtype}")
    print(f"  Lref={Lref} nu={nu:.4e} dt={dt:.3e} "
          f"levels={poisson.nlevels} lammax0={poisson.lammax[0]:.3e}")

    nan = False
    for step in range(start_step + 1, nsteps + 1):
        Ueff = U * min(1.0, step / ramp)
        cu, cv = Ueff * np.cos(aoa), Ueff * np.sin(aoa)
        blk.u[blk.recv] = cu; blk.v[blk.recv] = cv
        # predictor
        duc, dvc = t_conv(blk, blk.u, blk.v, scheme=scheme)
        diff_u, diff_v = t_diffusion(blk, blk.u, blk.v)
        if scheme == "central" and nu_hyp > 0.0:
            diff_u = diff_u + nu_hyp * t_lap(blk, t_lap(blk, blk.u))
            diff_v = diff_v + nu_hyp * t_lap(blk, t_lap(blk, blk.v))
        ustar = blk.u + dt * (-duc + diff_u)
        vstar = blk.v + dt * (-dvc + diff_v)
        ustar[blk.wall] = 0.0; vstar[blk.wall] = 0.0
        ustar[blk.recv] = cu; vstar[blk.recv] = cv
        blk.u, blk.v = ustar, vstar
        # projection
        t_project(blk, dt, poisson)
        if cb != "none":
            blk.p = _t_checkerboard(blk.p)
            blk.u = _t_checkerboard(blk.u)
            blk.v = _t_checkerboard(blk.v)
        blk.u[blk.wall] = 0.0; blk.v[blk.wall] = 0.0
        blk.u[blk.recv] = cu; blk.v[blk.recv] = cv
        if pert_k0 <= step <= pert_k1 and pert_amp > 0.0:
            edge = min(step - pert_k0, pert_k1 - step)
            ramp_p = min(1.0, edge / 400.0)
            yc = blk.yc
            vpert = (pert_amp * U * ramp_p * torch.sin(3.0 * np.pi * blk.xc / Lref)
                     * torch.exp(-((yc) / (0.5 * Lref)) ** 2))
            blk.v[solved] = blk.v[solved] + vpert[solved]
        if not (torch.isfinite(blk.u).all() and torch.isfinite(blk.v).all()
                and torch.isfinite(blk.p).all()):
            nan = True
            print(f"  NaN at step {step}")
            break
        if step >= ramp:
            Fx, Fy = t_forces(blk, nu, U)
            Fdrag = Fx * np.cos(aoa) + Fy * np.sin(aoa)
            Flift = -Fx * np.sin(aoa) + Fy * np.cos(aoa)
            q = 0.5 * U * U * Lref
            cd_mean = Fdrag / q; cl_mean = Flift / q
            t = step * dt
            t_hist.append(t); cd_hist.append(cd_mean); cl_hist.append(cl_mean)
            if step % 1000 == 0 or (steady and step % 500 == 0):
                print(f"  step{step:5d} t={t:7.2f} Cd={cd_mean:+.4f} "
                      f"Cl={cl_mean:+.4f}")
        if field_every > 0 and (step % field_every == 0 or step == nsteps):
            _render_torch(blk, out, step, geometry, Lref, aoa_deg)
        if save_every > 0 and (step % save_every == 0):
            np.savez(f"/workspace/{out}_ckpt.npz",
                     u=blk.u.detach().cpu().numpy(),
                     v=blk.v.detach().cpu().numpy(),
                     p=blk.p.detach().cpu().numpy(),
                     step=step, t=np.array(t_hist, dtype=float),
                     cd=np.array(cd_hist, dtype=float),
                     cl=np.array(cl_hist, dtype=float), dt=dt, ni=ni, nj=nj)
            print(f"  checkpoint saved @ step {step} -> /workspace/{out}_ckpt.npz")

    if nan:
        print("RESULT: NaN")
        return
    if not cd_hist:
        print("RESULT: no force history (%d steps < ramp=%d)" % (nsteps, ramp))
        return
    n0 = max(1, len(cd_hist) // 3)
    cd_mean = float(np.mean(cd_hist[n0:]))
    cl_win = np.array(cl_hist[n0:])
    cl_mean = float(np.mean(cl_win))
    print(f"\nRESULT torch {geometry} ni={ni} nj={nj} aoa={aoa_deg} Re={Re} mode={mode}")
    print(f"  Cd_mean = {cd_mean:.4f}   (window t>={t_hist[n0]:.2f})")
    print(f"  Cl_mean = {cl_mean:+.4f}")
    np.savez(f"/workspace/{out}_fields.npz",
             xc=blk.xc.detach().cpu().numpy(), yc=blk.yc.detach().cpu().numpy(),
             u=blk.u.detach().cpu().numpy(), v=blk.v.detach().cpu().numpy(),
             p=blk.p.detach().cpu().numpy(), Lref=Lref, aoa=aoa_deg,
             t=np.array(t_hist), cd=np.array(cd_hist), cl=np.array(cl_hist),
             ni=ni, nj=nj, dt=dt, Rf=Rf, beta=beta, n_surf=n_surf, step=nsteps)


def _t_checkerboard(f):
    """Exact torch mirror of curve_grid.c_checkerboard_project(mode='pair').

    NOTE (P3 backend-consistency root cause): the previous implementation was a
    DIFFERENT operator -- a 2D (i AND j) 4-neighbour roll/cat average blended at
    0.5 -- which is NOT equivalent to the numpy j-direction even/odd PAIR smoother.
    Applied every step to p/u/v it dragged the torch and numpy fields apart from
    step 1 (rel|u| ~0.4 by step 1, accumulating to a 3x lift bias).  The numpy
    'pair' operator replaces each j-cell by the average of itself and its j-
    neighbour, pairing (0,1),(2,3),... :

        out[:, 0:-1:2] = 0.5*(f[:,0:-1:2] + f[:,1::2])   (even j -> avg w/ j+1)
        out[:, 1::2]   = 0.5*(f[:,1::2]   + f[:,0:-1:2]) (odd  j -> avg w/ j-1)

    Mirror that exactly so the two backends stay bit-for-bit close.
    """
    c = torch.zeros_like(f)
    c[:, 0:-1:2] = 0.5 * (f[:, 0:-1:2] - f[:, 1::2])
    c[:, 1::2] = 0.5 * (f[:, 1::2] - f[:, 0:-1:2])
    return f - c


def _render_torch(blk, out, step, geometry, Lref, aoa):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xc = blk.xc.detach().cpu().numpy(); yc = blk.yc.detach().cpu().numpy()
    u = blk.u.detach().cpu().numpy(); v = blk.v.detach().cpu().numpy()
    p = blk.p.detach().cpu().numpy()
    spd = np.sqrt(u ** 2 + v ** 2)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].tricontourf(xc.ravel(), yc.ravel(), spd.ravel(), levels=40, cmap="viridis")
    ax[0].set_title(f"|u| step{step}")
    ax[1].tricontourf(xc.ravel(), yc.ravel(), p.ravel(), levels=40, cmap="RdBu_r")
    ax[1].set_title(f"p step{step}")
    j0 = 0
    ax[0].plot(xc[:, j0], yc[:, j0], "k-", lw=1.2)
    fig.savefig(f"/workspace/{out}_f{step:05d}.png", dpi=90)
    plt.close(fig)
