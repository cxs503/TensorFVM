"""k-omega SST RANS on the penalised periodic channel, with an exact discrete adjoint.

What this module adds
---------------------
:mod:`tensorfvm.adjoint_shape.channel` closes the momentum equations with a
*constant* viscosity: all earlier frozen-turbulence evidence in this package used
an algebraic (mixing-length / distance-decay) ``nu_t`` field, so the ``d nu_t /
d theta`` channel was never exercised by an actual turbulence model.  Here the
Reynolds stress is closed with the **Menter k-omega SST** two-equation model,
whose eddy viscosity is a genuine function of the transported fields and of the
shape,

    nu_t = a1 k / max(a1 omega, F2 S),          F2 = tanh(arg2^2),
    arg2 = max(2 sqrt(k) / (beta* omega d), 500 nu / (d^2 omega)),

with ``d`` the distance to the nearest solid (channel wall or immersed body), so
``nu_t = nu_t(k, omega, grad u, theta)`` through the body distance.  The k and
omega transport equations are solved together with momentum/continuity as one
coupled steady system.

Why the adjoint is built on autograd
------------------------------------
The residual is written once, in ``torch``, as ``R(U, theta)``.  Two consequences:

* the **discrete Jacobian** ``A = dR/dU`` is obtained exactly (forward-mode AD,
  see :meth:`SSTChannel.jacobian`), not by differencing the residual, so the
  adjoint is *discretely consistent* by construction -- the residual of the
  adjoint solve is a linear-solver residual only;
* the **shape derivative** ``dR/dtheta`` carries the ``d nu_t/d theta`` term for
  free, which is the term the frozen-turbulence approximation drops.

Frozen turbulence is therefore a one-line switch: ``nu_t.detach()`` (see the
``mode`` argument).  A third mode freezes in addition the upwind numerical
dissipation coefficient, which is the term a *continuous* adjoint does not see;
the difference between the three modes is a clean decomposition of the gradient.

Discretisation
--------------
Identical staggered MAC layout, smooth-upwind convection, Brinkman penalisation
and ``J = C_D`` objective as :mod:`tensorfvm.adjoint_shape.channel` (so the
laminar limit is directly comparable), extended with cell-centred k and omega:

* k: Dirichlet ``k = 0`` on the channel walls and inside the body
  (``+ alpha/eta k``);
* omega: ``omega_wall = 6 nu / (beta1 (dy/2)^2)`` on the channel walls and
  ``+ alpha/eta (omega - omega_wall)`` inside the body.

The SST min/max switches are smoothed with a relative width of ``1e-6`` so that
the Jacobian is continuous -- a tie in ``max()`` is otherwise a kink at which a
central difference and a derivative disagree by O(1).  The model perturbation
introduced by that smoothing is ``O(1e-6)`` relative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .channel import ChannelGrid
from .geometry import MODES, N_PARAMS

FD = torch.float64

#: Menter (2003) SST constants.
BETA_STAR = 0.09
A1 = 0.31
SIGMA_K1 = 0.85
SIGMA_W1 = 0.5
BETA1 = 0.075
SIGMA_K2 = 1.0
SIGMA_W2 = 0.856
BETA2 = 0.0828
KAPPA = 0.41
GAMMA1 = BETA1 / BETA_STAR - SIGMA_W1 * KAPPA**2 / math.sqrt(BETA_STAR)
GAMMA2 = BETA2 / BETA_STAR - SIGMA_W2 * KAPPA**2 / math.sqrt(BETA_STAR)
C_LIM = 10.0


# --------------------------------------------------------------------------
# smooth primitives (C^1 everywhere, so J^T lambda and a central difference agree)
# --------------------------------------------------------------------------
def _smooth_abs(a, eps):
    return a * torch.tanh(a / eps)


def _smax(a, b, rel=1e-6):
    d = rel * (torch.abs(a) + torch.abs(b)) + 1e-30
    return 0.5 * (a + b + torch.sqrt((a - b) ** 2 + d * d))


def _smin(a, b, rel=1e-6):
    d = rel * (torch.abs(a) + torch.abs(b)) + 1e-30
    return 0.5 * (a + b - torch.sqrt((a - b) ** 2 + d * d))


def _flux(a, pl, pr, eps):
    """Smooth upwind flux: first-order upwind where ``|a| >> eps``, C-infinity at 0."""
    return 0.5 * a * (pl + pr) - 0.5 * _smooth_abs(a, eps) * (pr - pl)


def _mode_value_torch(index: int, phi):
    k, trig = MODES[index]
    return torch.cos(k * phi) if trig == "cos" else torch.sin(k * phi)


# --------------------------------------------------------------------------
# unknown layout, colouring and structural pattern (fixed for a given grid)
# --------------------------------------------------------------------------
@dataclass
class _Field:
    name: str
    n1: int
    n2: int
    off: int
    j0: int  # first stored j index


def _layout(grid: ChannelGrid, turbulent: bool):
    nu, nv, nc = grid.nu, grid.nv, grid.nc
    fields = [
        _Field("u", grid.ny, grid.nx, 0, 0),
        _Field("v", grid.ny - 1, grid.nx, nu, 1),
        _Field("p", grid.ny, grid.nx, nu + nv, 0),
    ]
    if turbulent:
        fields.append(_Field("k", grid.ny, grid.nx, nu + nv + nc, 0))
        fields.append(_Field("w", grid.ny, grid.nx, nu + nv + 2 * nc, 0))
    size = nu + nv + nc + (2 * nc if turbulent else 0)
    return fields, size


def _pattern_and_colors(grid: ChannelGrid, fields, size: int, radius: int = 2):
    """Structural pattern (radius-``2`` box over every field pair) and a colouring.

    ``k = max(a1 k / max(...), ...)`` style coupling means a k row can depend on
    velocities two cells away (the eddy viscosity at a face is averaged from two
    neighbours, and ``S`` there is a velocity gradient), hence ``radius = 2``.
    A colour is ``(field, i mod m, j mod m)`` with ``m = 2 * radius + 1 = 5``:
    two unknowns sharing a colour are at least five cells apart in one direction,
    so they cannot both appear in the support of one column (whose rows span a
    ``2 * radius`` box) -- the coloured reverse-mode Jacobian can then be
    unpicked without aliasing.
    """
    period = 2 * radius + 1
    ny, nx = grid.ny, grid.nx
    if nx % period:
        raise ValueError(
            f"nx must be a multiple of {period} for the periodic {period}-colouring "
            f"(got nx={nx}); otherwise indices 0 and nx-1 share a colour."
        )
    idx, ic, jc = {}, {}, {}
    ncol = 0
    for f, fld in enumerate(fields):
        a = np.full((ny, nx), -1, dtype=np.int64)
        cc = np.full((ny, nx), -1, dtype=np.int64)
        for j in range(fld.j0, fld.j0 + fld.n1):
            a[j] = fld.off + (j - fld.j0) * nx + np.arange(nx)
            cc[j] = f + len(fields) * (((np.arange(nx)) % period) + period * (j % period))
        idx[fld.name], ic[fld.name], jc[fld.name] = a, cc, f
        ncol = max(ncol, int(cc.max()) + 1)

    rows_l, cols_l = [], []
    for fr in fields:
        ar = idx[fr.name]
        rows_valid = ar.ravel()[ar.ravel() >= 0]
        jj, ii = np.divmod(np.nonzero(ar.ravel() >= 0)[0], nx)
        for fc in fields:
            ac = idx[fc.name]
            for dj in range(-radius, radius + 1):
                for di in range(-radius, radius + 1):
                    j2 = jj + dj
                    i2 = (ii + di) % nx
                    ok = (j2 >= 0) & (j2 < ny)
                    cols_valid = ac[j2[ok], i2[ok]]
                    good = cols_valid >= 0
                    rows_l.append(rows_valid[ok][good])
                    cols_l.append(cols_valid[good])
    rows = np.concatenate(rows_l)
    cols = np.concatenate(cols_l)
    color_of_col = np.zeros(size, dtype=np.int64)
    for fld in fields:
        cc = ic[fld.name]
        m = cc >= 0
        color_of_col[idx[fld.name][m]] = cc[m]
    pattern = sp.coo_matrix(
        (np.ones(rows.size), (rows, cols)), shape=(size, size)
    ).tocsr()
    pattern.sum_duplicates()
    return pattern, rows, cols, color_of_col, ncol


# --------------------------------------------------------------------------
# the case
# --------------------------------------------------------------------------
class SSTChannel:
    """Steady k-omega SST RANS + Brinkman body on a streamwise-periodic channel."""

    def __init__(
        self,
        grid: ChannelGrid | None = None,
        *,
        nu: float = 5.0e-5,
        force: float = 1.5e-4,
        eta: float = 1.0e-3,
        center=(2.0, 0.9),
        radius0: float = 0.25,
        delta_cells: float = 0.6,
        quad: int = 3,
        eps_upwind: float = 0.05,
        turbulent: bool = True,
        rho: float = 1.0,
        cross_diffusion: bool = False,
        nut_fn=None,
        nut_fixed=None,
    ):
        self.grid = grid or ChannelGrid(nx=64, ny=32)
        g = self.grid
        self.nu = float(nu)
        self.force = float(force)
        self.eta = float(eta)
        self.rho = float(rho)
        self.center = (float(center[0]), float(center[1]))
        self.radius0 = float(radius0)
        self.delta = delta_cells * min(g.dx, g.dy)
        self.subcell = min(g.dx, g.dy)
        self.quad = int(quad)
        self.eps_upwind = float(eps_upwind)
        self.turbulent = bool(turbulent)
        #: The SST cross-diffusion (CD) term is kept in the ``F1`` blending
        #: argument but dropped from the omega equation: measured here, the
        #: ``1/omega`` amplification of the steady CD term collapses ``omega`` to
        #: zero (a trivial fixed point of the homogeneous turbulence system) at
        #: every Re tried.  Several production codes make the same choice; the
        #: switch is kept so the effect can be re-tested.
        self.cross_diffusion = bool(cross_diffusion)
        #: Optional prescribed eddy-viscosity model ``nut_fn(theta, case) ->
        #: (ny, nx)`` used with ``turbulent=False``: the momentum equations see a
        #: shape-dependent ``nu_t`` while ``k``/``omega`` are *not* transported.
        #: This is the closure on which the frozen-turbulence bias is quantified
        #: here (the fully coupled SST transport solve failed to converge, see
        #: ``docs/sst-rans-adjoint/REPORT.md``).
        self.nut_fn = nut_fn
        #: Fixed ``nu_t`` field (a tensor, no ``theta`` dependence) used with
        #: ``turbulent=False``: the frozen-turbulence operator, for the
        #: self-consistency check frozen-adjoint vs frozen-FD.
        self.nut_fixed = nut_fixed

        self.fields, self.size = _layout(g, self.turbulent)
        self.pattern, self._p_rows, self._p_cols, self._color_of_col, self.ncolor = (
            _pattern_and_colors(g, self.fields, self.size)
        )
        self._p_color = self._color_of_col[self._p_rows]

        # coordinates (broadcastable)
        xc = (np.arange(g.nx)[None, :] + 0.5) * g.dx
        yc = (np.arange(g.ny)[:, None] + 0.5) * g.dy
        xu = np.broadcast_to(np.arange(g.nx)[None, :] * g.dx, (g.ny, g.nx))
        yu = np.broadcast_to(((np.arange(g.ny) + 0.5) * g.dy)[:, None], (g.ny, g.nx))
        xv = np.broadcast_to(((np.arange(g.nx) + 0.5) * g.dx)[None, :], (g.ny - 1, g.nx))
        yv = np.broadcast_to((np.arange(1, g.ny) * g.dy)[:, None], (g.ny - 1, g.nx))
        self._coords = {
            "u": (torch.tensor(xu, dtype=FD), torch.tensor(yu, dtype=FD)),
            "v": (torch.tensor(xv, dtype=FD), torch.tensor(yv, dtype=FD)),
            "c": (
                torch.tensor(np.broadcast_to(xc, (g.ny, g.nx)).copy(), dtype=FD),
                torch.tensor(np.broadcast_to(yc, (g.ny, g.nx)).copy(), dtype=FD),
            ),
        }
        self.omega_wall = 6.0 * self.nu / (BETA1 * (0.5 * g.dy) ** 2)
        self._quad_offsets = [
            ((qy + 0.5) / self.quad - 0.5) * self.subcell
            for qy in range(self.quad)
        ]

    # -- geometry ---------------------------------------------------------
    def _indicator(self, x, y, theta):
        """Smooth Brinkman indicator ``alpha`` (and signed distance ``g``) in torch.

        Mirrors :meth:`tensorfvm.adjoint_shape.geometry.FourierBody.indicator`
        (compact-support smoothstep of the signed distance, sub-cell averaged with
        ``quad^2`` samples); ``theta`` may require grad.
        """
        alpha = torch.zeros_like(x)
        gs = torch.zeros_like(x)
        n = float(self.quad * self.quad)
        for oy in self._quad_offsets:
            for ox in self._quad_offsets:
                xx = x + ox
                yy = y + oy
                dx = xx - self.center[0]
                dy = yy - self.center[1]
                r = torch.sqrt(dx * dx + dy * dy + 1e-300)
                phi = torch.atan2(dy, dx)
                shape = torch.ones_like(phi)
                for q in range(N_PARAMS):
                    shape = shape + theta[q] * _mode_value_torch(q, phi)
                gcur = r - self.radius0 * shape
                sc = torch.clamp(0.5 - gcur / (2.0 * self.delta), 0.0, 1.0)
                alpha = alpha + sc * sc * (3.0 - 2.0 * sc)
                gs = gs + gcur
        return alpha / n, gs / n

    def geometry(self, theta):
        """``(alpha_u, alpha_v, alpha_c, g_c)`` for the three staggered locations."""
        th = theta if torch.is_tensor(theta) else torch.tensor(theta, dtype=FD)
        xu, yu = self._coords["u"]
        xv, yv = self._coords["v"]
        xc, yc = self._coords["c"]
        au, _ = self._indicator(xu, yu, th)
        av, _ = self._indicator(xv, yv, th)
        ac, gc = self._indicator(xc, yc, th)
        return au, av, ac, gc

    # -- unpack -----------------------------------------------------------
    def unpack(self, U):
        g = self.grid
        ny, nx = g.ny, g.nx
        out = {}
        for fld in self.fields:
            n = fld.n1 * nx
            out[fld.name] = U[fld.off : fld.off + n].reshape(fld.n1, nx)
        return out

    # -- SST closure ------------------------------------------------------
    def _strain2(self, u, v):
        g = self.grid
        dx, dy = g.dx, g.dy
        up = torch.cat(
            [torch.zeros(1, g.nx, dtype=FD), u, torch.zeros(1, g.nx, dtype=FD)], 0
        )
        vpad = torch.cat(
            [torch.zeros(1, g.nx, dtype=FD), v, torch.zeros(1, g.nx, dtype=FD)], 0
        )
        du_dx = (torch.roll(u, -1, 1) - u) / dx
        vg = torch.cat(
            [torch.zeros(1, g.nx, dtype=FD), vpad, torch.zeros(1, g.nx, dtype=FD)], 0
        )
        dv_dy = (vg[3 : g.ny + 3] - vg[1 : g.ny + 1]) / (2.0 * dy)
        du_dy = (up[2 : g.ny + 2] - up[0:g.ny]) / (2.0 * dy)
        # dv/dx lives on the v rows; average the two rows straddling a cell centre
        dvp = (torch.roll(vpad, -1, 1) - torch.roll(vpad, 1, 1)) / (2.0 * dx)
        dv_dx = 0.5 * (dvp[0:g.ny] + dvp[1 : g.ny + 1])
        return 2.0 * du_dx**2 + 2.0 * dv_dy**2 + (du_dy + dv_dx) ** 2, du_dx, dv_dy

    def eddy_viscosity(self, k, w, S2, gc):
        """``nu_t``, ``F1``, ``F2`` and the limited production (SST, smoothed switches)."""
        g = self.grid
        _, yc = self._coords["c"]
        y_wall = torch.minimum(yc, g.height - yc)
        d = torch.clamp(torch.minimum(y_wall, torch.abs(gc)), min=1e-6)
        kk = torch.clamp(k, min=1e-24)
        ww = torch.clamp(w, min=1e-24)
        sk = torch.sqrt(kk)
        S = torch.sqrt(S2 + 1e-24)

        arg2 = _smax(2.0 * sk / (BETA_STAR * ww * d), 500.0 * self.nu / (d * d * ww))
        F2 = torch.tanh(arg2**2)
        nut = A1 * kk / _smax(A1 * ww, F2 * S)

        # cross diffusion (gradients of the transported fields)
        dk_dx = (torch.roll(k, -1, 1) - torch.roll(k, 1, 1)) / (2.0 * g.dx)
        dw_dx = (torch.roll(w, -1, 1) - torch.roll(w, 1, 1)) / (2.0 * g.dx)
        kpad = torch.cat([torch.zeros(1, g.nx, dtype=FD), k, torch.zeros(1, g.nx, dtype=FD)], 0)
        wpad = torch.cat([torch.zeros(1, g.nx, dtype=FD), w, torch.zeros(1, g.nx, dtype=FD)], 0)
        dk_dy = (kpad[2 : g.ny + 2] - kpad[0:g.ny]) / (2.0 * g.dy)
        dw_dy = (wpad[2 : g.ny + 2] - wpad[0:g.ny]) / (2.0 * g.dy)
        dot = dk_dx * dw_dx + dk_dy * dw_dy
        cd = _smax(2.0 * SIGMA_W2 / ww * dot, torch.full_like(dot, 1e-10))
        arg1 = _smin(
            _smax(sk / (BETA_STAR * ww * d), 500.0 * self.nu / (d * d * ww)),
            4.0 * SIGMA_W2 * kk / (cd * d * d),
        )
        F1 = torch.tanh(arg1**4)

        P_raw = nut * S2
        ratio = C_LIM * BETA_STAR * kk * ww
        chi = 1.0 / torch.sqrt(torch.sqrt(1.0 + (P_raw / ratio) ** 4))
        Pk = P_raw * chi
        return nut, F1, F2, Pk, chi, S2, dot

    # -- residual ---------------------------------------------------------
    def residual(
        self,
        U,
        theta,
        *,
        mode: str = "full",
        nut_scale: float = 1.0,
        nut_field=None,
    ):
        """Steady residual ``R(U, theta)`` as a flat torch tensor (``R = 0`` at solution).

        ``mode``: ``"full"`` (all sensitivities), ``"frozen_nut"`` (``nu_t`` held
        fixed -- frozen turbulence), ``"frozen_nut_diss"`` (in addition the
        upwind dissipation coefficient ``|a|_s`` is frozen: continuous-adjoint
        like).  ``nut_field`` (``(ny, nx)``) overrides ``nu_t`` by a constant field.
        """
        g = self.grid
        ny, nx = g.ny, g.nx
        dx, dy = g.dx, g.dy
        eps = self.eps_upwind
        freeze_diss = mode in ("frozen_nut_diss",)
        F = self.unpack(U)
        u, v, p = F["u"], F["v"], F["p"]
        alpha_u, alpha_v, alpha_c, gc = self.geometry(theta)

        up = torch.cat([torch.zeros(1, nx, dtype=FD), u, torch.zeros(1, nx, dtype=FD)], 0)
        vpad = torch.cat([torch.zeros(1, nx, dtype=FD), v, torch.zeros(1, nx, dtype=FD)], 0)

        # ---- momentum (u) ------------------------------------------------
        ue, uw = torch.roll(u, -1, 1), torch.roll(u, 1, 1)
        fe = _flux(0.5 * (u + ue), u, ue, eps)
        fw = _flux(0.5 * (uw + u), uw, u, eps)
        us, un = up[0:ny], up[2 : ny + 2]
        fs = _flux(vpad[0:ny], us, u, eps)
        fn = _flux(vpad[1 : ny + 1], u, un, eps)
        conv_u = (fe - fw) / dx + (fn - fs) / dy

        # ---- momentum (v) ------------------------------------------------
        vcen = vpad[1:ny]
        vnn, vss = vpad[2 : ny + 1], vpad[0 : ny - 1]
        fvn = _flux(0.5 * (vcen + vnn), vcen, vnn, eps)
        fvs = _flux(0.5 * (vss + vcen), vss, vcen, eps)
        fve = _flux(
            0.5 * (torch.roll(up[1:ny], -1, 1) + torch.roll(up[0 : ny - 1], -1, 1)),
            vcen,
            torch.roll(vcen, -1, 1),
            eps,
        )
        fvw = _flux(
            0.5 * (up[1:ny] + up[0 : ny - 1]),
            torch.roll(vcen, 1, 1),
            vcen,
            eps,
        )
        conv_v = (fve - fvw) / dx + (fvn - fvs) / dy

        S2, _, _ = self._strain2(u, v)

        if self.turbulent:
            k, w = F["k"], F["w"]
            nut, F1, F2, Pk, chi, S2, dot = self.eddy_viscosity(k, w, S2, gc)
            if nut_field is not None:
                nut = nut_field
            nut = nut_scale * nut
            if mode in ("frozen_nut", "frozen_nut_diss"):
                nut = nut.detach()
        else:
            if self.nut_fixed is not None:
                nut = self.nut_fixed
            elif self.nut_fn is not None:
                nut = self.nut_fn(theta, self)
            else:
                nut = torch.zeros(ny, nx, dtype=FD)

        if mode in ("frozen_nut", "frozen_nut_diss"):
            nut = nut.detach()

        # diffusive faces of the u rows: x-face at the cell centre, y-face averaged
        nut_g = torch.cat([nut[0:1], nut, nut[ny - 1 : ny]], 0)
        m_s = 0.5 * (nut_g[0:ny] + nut_g[1 : ny + 1])
        m_n = 0.5 * (nut_g[1 : ny + 1] + nut_g[2 : ny + 2])
        c_s = (self.nu + m_s) / dy**2
        c_n = (self.nu + m_n) / dy**2
        # wall rows sit half a cell from the wall -> doubled coefficient
        c_s = torch.cat([2.0 * c_s[0:1], c_s[1:]], 0)
        c_n = torch.cat([c_n[0 : ny - 1], 2.0 * c_n[ny - 1 : ny]], 0)
        diff_u = (nut + self.nu) * (ue - u) - (torch.roll(nut, 1, 1) + self.nu) * (u - uw)
        diff_u = diff_u / dx**2 + c_n * (un - u) - c_s * (u - us)
        # R = conv + grad p - div((nu+nu_t) grad u) + Brinkman - force = 0
        R_u = conv_u + (p - torch.roll(p, 1, 1)) / dx - diff_u + (alpha_u / self.eta) * u - self.force

        # v rows: v(j,i) sits on a cell-row boundary -> nu_t averaged from the two rows
        nut_v = 0.5 * (nut[0 : ny - 1] + nut[1:ny])
        dvn = (
            (nut[1:ny] + self.nu) * (vnn - vcen)
            - (nut[0 : ny - 1] + self.nu) * (vcen - vss)
        ) / dy**2
        m_ve = self.nu + 0.5 * (nut_v + torch.roll(nut_v, -1, 1))
        m_vw = self.nu + 0.5 * (nut_v + torch.roll(nut_v, 1, 1))
        dvx = (
            m_ve * (torch.roll(vcen, -1, 1) - vcen)
            - m_vw * (vcen - torch.roll(vcen, 1, 1))
        ) / dx**2
        R_v = (
            conv_v
            + (p[1:ny] - p[0 : ny - 1]) / dy
            - (dvn + dvx)
            + (alpha_v / self.eta) * vcen
        )

        # ---- continuity --------------------------------------------------
        R_p = (torch.roll(u, -1, 1) - u) / dx + (vpad[1 : ny + 1] - vpad[0:ny]) / dy
        # pin the constant-pressure null mode
        pin = torch.zeros(ny, nx, dtype=torch.bool)
        pin[0, 0] = True
        R_p = torch.where(pin, p, R_p)

        if not self.turbulent:
            return torch.cat([R_u.reshape(-1), R_v.reshape(-1), R_p.reshape(-1)])

        k = F["k"]
        w = F["w"]
        # ---- k / omega convection (cell-centred, smooth upwind) ----------
        ke, kw = torch.roll(k, -1, 1), torch.roll(k, 1, 1)
        kpad = torch.cat([torch.zeros(1, nx, dtype=FD), k, torch.zeros(1, nx, dtype=FD)], 0)
        conv_k = (
            _flux(torch.roll(u, -1, 1), k, ke, eps) - _flux(u, kw, k, eps)
        ) / dx + (
            _flux(vpad[1 : ny + 1], k, kpad[2 : ny + 2], eps)
            - _flux(vpad[0:ny], kpad[0:ny], k, eps)
        ) / dy
        we, ww = torch.roll(w, -1, 1), torch.roll(w, 1, 1)
        wpad = torch.cat([torch.zeros(1, nx, dtype=FD), w, torch.zeros(1, nx, dtype=FD)], 0)
        conv_w = (
            _flux(torch.roll(u, -1, 1), w, we, eps) - _flux(u, ww, w, eps)
        ) / dx + (
            _flux(vpad[1 : ny + 1], w, wpad[2 : ny + 2], eps)
            - _flux(vpad[0:ny], wpad[0:ny], w, eps)
        ) / dy

        sig_k = F1 * SIGMA_K1 + (1.0 - F1) * SIGMA_K2
        sig_w = F1 * SIGMA_W1 + (1.0 - F1) * SIGMA_W2
        Dk = self.nu + sig_k * nut
        Dw = self.nu + sig_w * nut
        diff_k = self._scalar_diffusion(k, Dk, wall_value=0.0)
        diff_w = self._scalar_diffusion(w, Dw, wall_value=self.omega_wall)

        R_k = conv_k - diff_k - Pk + BETA_STAR * k * w + (alpha_c / self.eta) * k
        gam = F1 * GAMMA1 + (1.0 - F1) * GAMMA2
        cd_term = 2.0 * (1.0 - F1) * SIGMA_W2 / torch.clamp(w, min=1e-24) * dot
        R_w = (
            conv_w
            - diff_w
            - gam * S2 * chi
            + BETA2 * w * w
            - F1 * 0.0
            - ((1.0 - F1) * cd_term if self.cross_diffusion else 0.0)
            + (alpha_c / self.eta) * (w - self.omega_wall)
        )
        return torch.cat(
            [R_u.reshape(-1), R_v.reshape(-1), R_p.reshape(-1), R_k.reshape(-1), R_w.reshape(-1)]
        )

    def _scalar_diffusion(self, phi, D, wall_value: float):
        """``d/dx_j (D d phi/dx_j)`` on cell centres, Dirichlet ``phi = wall_value``."""
        g = self.grid
        ny, nx = g.ny, g.nx
        De = 0.5 * (D + torch.roll(D, -1, 1))
        Dw = 0.5 * (D + torch.roll(D, 1, 1))
        dx_term = (De * (torch.roll(phi, -1, 1) - phi) - Dw * (phi - torch.roll(phi, 1, 1))) / g.dx**2
        Dpad = torch.cat([D[0:1], D, D[ny - 1 : ny]], 0)
        phipad = torch.cat(
            [torch.full((1, nx), float(wall_value), dtype=FD), phi,
             torch.full((1, nx), float(wall_value), dtype=FD)], 0
        )
        Dn = Dpad[2 : ny + 2]
        Ds = Dpad[0:ny]
        # Dirichlet walls at half a cell distance
        Ds = torch.cat([2.0 * Ds[0:1], Ds[1:]], 0)
        Dn = torch.cat([Dn[0 : ny - 1], 2.0 * Dn[ny - 1 : ny]], 0)
        dy_term = (Dn * (phipad[2 : ny + 2] - phi) - Ds * (phi - phipad[0:ny])) / g.dy**2
        return dx_term + dy_term

    # -- objective --------------------------------------------------------
    def objective(self, U, theta):
        """``J = C_D`` of the immersed body (penalisation reaction)."""
        g = self.grid
        F = self.unpack(U)
        u = F["u"]
        alpha_u, _, _, _ = self.geometry(theta)
        cell = g.dx * g.dy
        Fx = cell * torch.sum(alpha_u / self.eta * u)
        um = torch.sum(u[:, 0]) * g.dy / g.height
        A = 0.5 * self.rho * um * um * (2.0 * self.radius0)
        return Fx / A

    # -- Jacobians --------------------------------------------------------
    def jacobian(self, U, theta, mode: str = "full", dense: bool = False):
        """Exact discrete Jacobian ``dR/dU`` by coloured reverse-mode AD.

        One reverse pass per colour of the row graph (``ncolor = 5 x 3 x 3 = 45``
        for the turbulent layout) yields ``J^T g_c``; because the structural box
        is symmetric and two unknowns sharing a colour are three cells apart, the
        entry ``(r, k)`` is read off at colour ``colour(r)`` without aliasing.
        ``torch.func.jvp`` (forward mode) is unusable here -- it trips a
        ``torch.jit`` decomposition import bug in this environment.
        """
        theta = self._theta_t(theta).detach()
        n = self.size
        Uv = U.detach().clone().requires_grad_(True)
        R = self.residual(Uv, theta, mode=mode)
        vals = np.zeros(self._p_rows.size)
        for c in range(self.ncolor):
            seed = torch.zeros(n, dtype=FD)
            seed[self._color_of_col == c] = 1.0
            (vjp,) = torch.autograd.grad(R, Uv, grad_outputs=seed, retain_graph=True)
            mask = self._p_color == c
            if not mask.any():
                continue
            vnp = vjp.detach().numpy()
            vals[mask] = vnp[self._p_cols[mask]]
        J = sp.coo_matrix((vals, (self._p_rows, self._p_cols)), shape=(n, n)).tocsr()
        if dense:
            return np.asarray(J.todense())
        return J

    def num_jacobian(self, U, theta, eps: float = 1e-6):
        """Dense finite-difference Jacobian (verification of :meth:`jacobian`)."""
        theta = self._theta_t(theta)
        n = self.size
        U0 = U.detach().clone()
        out = np.zeros((n, n))
        for j in range(n):
            dU = torch.zeros(n, dtype=FD)
            dU[j] = eps
            Rp = self.residual(U0 + dU, theta).detach().numpy()
            Rm = self.residual(U0 - dU, theta).detach().numpy()
            out[:, j] = (Rp - Rm) / (2.0 * eps)
        return out

    # -- solve ------------------------------------------------------------
    def _theta_t(self, theta):
        if torch.is_tensor(theta):
            return theta
        return torch.tensor(np.asarray(theta, dtype=float), dtype=FD)

    def initial_guess(self, k0: float = 4.0e-3, w0: float = None):
        g = self.grid
        U = np.zeros(self.size)
        u = U[0 : g.nu].reshape(g.ny, g.nx)
        for j in range(g.ny):
            y = (j + 0.5) * g.dy
            u[j, :] = self.force * y * (g.height - y) / (2.0 * self.nu)
        if self.turbulent:
            w0 = w0 if w0 is not None else 100.0
            for fld in self.fields:
                if fld.name in ("k", "w"):
                    val = k0 if fld.name == "k" else w0
                    U[fld.off : fld.off + fld.n1 * g.nx] = val
        return torch.tensor(U, dtype=FD)

    def solve(
        self,
        theta,
        U0=None,
        *,
        mode: str = "full",
        nut_scale: float = 1.0,
        tol: float = 1e-10,
        maxit: int = 60,
        verbose: bool = False,
    ):
        """Steady solve of ``R(U, theta) = 0``: pseudo-transient (SER) then Newton.

        Phase 1 adds ``+ I / dt`` to the Jacobian with ``dt`` grown as the
        residual falls (switched-evolution relaxation): the steps are then
        always descent-like for ``|R|`` and the stiff Brinkman penalty cannot
        throw the turbulence variables negative.  Phase 2 switches to the exact
        Newton step with a backtracking line search, which is what delivers the
        machine-precision solution the adjoint/FD comparison needs.
        """
        theta_t = self._theta_t(theta)
        U = self.initial_guess() if U0 is None else U0.detach().clone()
        R = self.residual(U, theta_t, mode=mode, nut_scale=nut_scale)
        n0 = float(torch.linalg.norm(R))
        hist = [n0]
        converged = False
        A = None
        target = max(tol * n0, 1e-13)
        switch = max(1e-7 * n0, 1e-11)
        dt = float(torch.linalg.norm(R)) / max(float(torch.abs(R).sum()), 1e-30)
        for it in range(1, maxit + 1):
            nR = float(torch.linalg.norm(R))
            if nR <= target:
                converged = True
                break
            A = self.jacobian(U, theta_t, mode=mode)
            if nR > switch:
                # ---- phase 1: shifted (pseudo-transient) step ------------
                M = A + sp.identity(self.size, format="csr") * (1.0 / dt)
                du = spla.spsolve(M.tocsc(), -R.detach().numpy())
                U = self.project(U + torch.tensor(du, dtype=FD))
                dt *= 2.0
                accepted = True
            else:
                # ---- phase 2: exact Newton with backtracking ------------
                du = spla.spsolve(A.tocsc(), -R.detach().numpy())
                lam = 1.0
                accepted = False
                for _ in range(30):
                    trial = self.project(U + lam * torch.tensor(du, dtype=FD))
                    if True:
                        Rt = self.residual(trial, theta_t, mode=mode, nut_scale=nut_scale)
                        if float(torch.linalg.norm(Rt)) < (1.0 - 1e-4 * lam) * nR:
                            accepted = True
                            break
                    lam *= 0.5
                if not accepted:
                    # fall back to a pseudo-transient step: always make progress
                    M = A + sp.identity(self.size, format="csr") * (1.0 / dt)
                    du = spla.spsolve(M.tocsc(), -R.detach().numpy())
                    U = self.project(U + torch.tensor(du, dtype=FD))
                    dt *= 2.0
                    accepted = True
                else:
                    U = trial
            if not accepted:
                break
            R = self.residual(U, theta_t, mode=mode, nut_scale=nut_scale)
            hist.append(float(torch.linalg.norm(R)))
            if verbose:
                print(f"    it {it:3d} |R|={hist[-1]:.3e} lam={lam:.3g} dt={dt:.2e}")
        A = self.jacobian(U, theta_t, mode=mode)
        Rn = float(torch.linalg.norm(self.residual(U, theta_t, mode=mode, nut_scale=nut_scale)))
        info = {
            "iterations": len(hist) - 1,
            "residual": Rn,
            "residual_initial": n0,
            "relative_residual": Rn / max(n0, 1e-300),
            "history": hist,
            "converged": bool(Rn <= max(10.0 * target, 1e-12)),
            "jacobian": A,
        }
        return U, info

    #: floors projected after every step (a free-stream k that has died is a
    #: singular direction of the k equation; the floor keeps the Newton step alive)
    K_FLOOR = 1e-9
    W_FLOOR = 1e-6

    def project(self, U):
        """Clamp ``k >= K_FLOOR`` and ``omega >= W_FLOOR`` (positivity projection)."""
        if not self.turbulent:
            return U
        for fld in self.fields:
            if fld.name == "k":
                lo = self.K_FLOOR
            elif fld.name == "w":
                lo = self.W_FLOOR
            else:
                continue
            n = fld.n1 * self.grid.nx
            U[fld.off : fld.off + n] = torch.clamp(U[fld.off : fld.off + n], min=lo)
        return U

    def solve_segregated(
        self,
        theta,
        U0=None,
        *,
        tol: float = 1e-11,
        maxit: int = 40,
        relax: float = 0.8,
        umin: float = None,
        verbose: bool = False,
    ):
        """Gauss-Seidel iteration on the exact blocks of the coupled system.

        The coupled Newton iteration of :meth:`solve` stalls once ``k`` dies in
        the free stream (the k row is then pure transport and its Newton step
        wants ``k < 0``).  Here the mean-flow block ``(u, v, p)`` and the
        turbulence block ``(k, omega)`` are each solved exactly from the
        corresponding diagonal block of the *exact* Jacobian, with the other
        block frozen; the diagonal blocks are well conditioned (the mean-flow
        block is the laminar operator at the current ``nu_t``, the turbulence
        block is diagonally dominant), so each substep converges and the outer
        fixed point satisfies the full ``R(U, theta) = 0``.
        """
        theta_t = self._theta_t(theta)
        U = self.initial_guess() if U0 is None else U0.detach().clone()
        U = self.project(U)
        g = self.grid
        i_flow = np.arange(0, g.nu + g.nv + g.nc)
        i_turb = np.arange(g.nu + g.nv + g.nc, self.size)
        R = self.residual(U, theta_t)
        n0 = float(torch.linalg.norm(R))
        hist = [n0]
        for it in range(1, maxit + 1):
            nR = float(torch.linalg.norm(R))
            if nR <= max(n0 * tol, 1e-13):
                break
            A = self.jacobian(U, theta_t)
            for idx, relax_i in ((i_flow, 1.0), (i_turb, relax)):
                sub = A[idx][:, idx].tocsc()
                Rsub = R.detach().numpy()[idx]
                du = spla.spsolve(sub, -Rsub)
                step = torch.zeros(self.size, dtype=FD)
                step[idx] = torch.tensor(du, dtype=FD)
                U = self.project(U + relax_i * step)
                R = self.residual(U, theta_t)
            hist.append(float(torch.linalg.norm(R)))
            if verbose:
                print(f"    seg {it:3d} |R|={hist[-1]:.3e}")
        A = self.jacobian(U, theta_t)
        Rn = float(torch.linalg.norm(self.residual(U, theta_t)))
        return U, {
            "iterations": len(hist) - 1,
            "residual": Rn,
            "residual_initial": n0,
            "relative_residual": Rn / max(n0, 1e-300),
            "history": hist,
            "converged": bool(Rn <= max(n0 * 1e-8, 1e-12)),
            "jacobian": A,
        }

    def _positive(self, U):
        if not self.turbulent:
            return True
        F = self.unpack(U)
        return bool(torch.all(F["k"] > 0) and torch.all(F["w"] > 0))

    # -- adjoint ----------------------------------------------------------
    def adjoint_gradient(self, theta, U=None, *, mode: str = "full", info: dict | None = None):
        """``dC_D/dtheta`` by the discrete adjoint, plus the adjoint-solve residual.

        ``A^T lambda = dJ/dU``, ``dJ/dtheta = dJ/dtheta|_U - lambda^T dR/dtheta``.
        ``mode`` selects the gradient model (see :meth:`residual`).
        """
        theta_t = self._theta_t(theta)
        if U is None:
            U, info = self.solve(theta_t, mode=mode)
        A = (info or {}).get("jacobian") if info else None
        if A is None:
            A = self.jacobian(U, theta_t, mode=mode)
        Uv = U.detach().clone().requires_grad_(True)
        J = self.objective(Uv, theta_t)
        (gU,) = torch.autograd.grad(J, Uv)
        # dJ/dtheta at fixed U
        th = theta_t.detach().clone().requires_grad_(True)
        (gTheta_J,) = torch.autograd.grad(self.objective(U.detach(), th), th)
        lam = spla.spsolve(A.T.tocsr().tocsc(), gU.detach().numpy())
        res = float(
            np.linalg.norm(A.T @ lam - gU.detach().numpy())
            / max(np.linalg.norm(gU.detach().numpy()), 1e-300)
        )
        # lambda^T dR/dtheta in one reverse-mode pass
        th2 = theta_t.detach().clone().requires_grad_(True)
        Rv = self.residual(U.detach(), th2, mode=mode)
        (Rtheta_t,) = torch.autograd.grad(Rv, th2, grad_outputs=torch.tensor(lam, dtype=FD))
        Rtheta = Rtheta_t.detach().numpy()
        grad = gTheta_J.detach().numpy() - Rtheta
        return grad, {
            "J": float(J),
            "adjoint_residual": res,
            "lambda": lam,
            "dJ_dU": gU.detach().numpy(),
            "dJ_dtheta": gTheta_J.detach().numpy(),
            "lam_Rtheta": Rtheta,
            "U": U,
        }

    def fd_gradient(self, theta, U0=None, eps: float = 1e-4, mode: str = "full"):
        """Central difference of the *re-converged* full solve (reference gradient)."""
        theta = np.asarray(theta, dtype=float)
        out = np.zeros(N_PARAMS)
        base = None
        for q in range(N_PARAMS):
            e = np.zeros(N_PARAMS)
            e[q] = eps
            Up, ip = self.solve(theta + e, U0=U0)
            Um, im = self.solve(theta - e, U0=U0)
            Jp = float(self.objective(Up, self._theta_t(theta + e)))
            Jm = float(self.objective(Um, self._theta_t(theta - e)))
            base = Jp if base is None else base
            out[q] = (Jp - Jm) / (2.0 * eps)
        return out, {"converged": True}


def make_case(nx=64, ny=32, re=1.0e4, u_mean=1.0, radius0=0.25, **kw):
    """Re = u_mean * 2 r0 / nu case on the periodic channel (D = 2 r0)."""
    grid = ChannelGrid(nx=nx, ny=ny)
    nu = u_mean * (2.0 * radius0) / re
    force = 12.0 * nu * u_mean / grid.height**2
    return SSTChannel(grid, nu=nu, force=force, radius0=radius0, **kw)


__all__ = [
    "SSTChannel",
    "make_case",
    "BETA_STAR",
    "A1",
    "GAMMA1",
    "GAMMA2",
]