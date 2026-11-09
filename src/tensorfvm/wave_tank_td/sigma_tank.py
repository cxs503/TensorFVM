"""Time-domain numerical wave tank on a **surface-fitted (sigma) grid**.

Why a different discretisation from :mod:`tensorfvm.wave_tank`
--------------------------------------------------------------
The legacy VOF tank recorded one blocking defect: the first-harmonic wave
amplitude decayed by ~80 % within one wavelength, i.e. the *numerical* scheme,
not the physics, set the wave field.  Every candidate cause was an interface
scheme (THINC re-steepening, cell-centred surface Dirichlet, upwind blend) --
dissipation that exists only because the free surface cuts through a Cartesian
mesh.

This module removes the root cause instead of tuning it: the vertical
coordinate is normalised by the *instantaneous* free surface,

    sigma in [0, 1],   y(x, sigma, t) = -h + sigma * H(x, t),   H = h + eta(x, t)

so the free surface **is** the top coordinate line.  No interface to
reconstruct, no cell cut by the surface, no air phase, and the surface pressure
Dirichlet ``p = 0`` sits exactly where the surface is.  Wave damping can then
only come from the momentum discretisation, which is what one wants to measure.

Continuity in the sigma frame is the plain cell divergence

    A_x[i+1] u[i+1,j] - A_x[i] u[i,j] + A_s (w[i,j+1] - w[i,j]) = 0,
    A_x = dsig*Hf,   A_s = dx,

because the grid-velocity terms of the moving sigma lines cancel identically:
the geometric conservation law holds *by construction* (the contravariant
vertical velocity through the surface line is zero).  Momentum is

    du/dt = -u du/dx - w du/dy + nu lap(u) - dp/dx          (rho = 1)
    dw/dt = -u dw/dx - w dw/dy + nu lap(w) - dp/dy - g

with a central/upwind blend on the advective derivatives.

One time step (:meth:`SigmaWaveTank.step`)
-----------------------------------------
1. rebuild the geometry ``H, Ax, Wx, Ws, y`` from ``eta`` (the sigma grid moves);
2. body mask -> open face areas ``ox`` (x-faces) and ``os`` (sigma-faces);
3. momentum predictor ``u*, w*`` (advection + gravity);
4. relaxation zone ``u* <- (1-b) u* + b u_target`` for ``x < L_r`` (waves2Foam
   recipe, smoothstep weight) and sponge ``u* <- u* (1 - s dt)`` for
   ``x > L - L_d``;
5. pressure projection: solve ``(B W^-1 B^T) p = B u*/dt`` with ``p = 0`` on the
   free surface, ``dp/dn = 0`` on the bottom, the walls and the body.  ``B`` is
   assembled from the *open* face areas, so the projection is an exact algebraic
   projection: after the correction ``max|div u|`` is at round-off even with a
   body inside the tank (:attr:`div_residual` records it every step);
6. ``eta += dt * w_surface`` -- mass conservation is a flux statement, so the
   water volume is conserved to round-off with no volume ledger;
7. body: pressure integral over the wetted, partially covered faces -> heave
   force and pitch moment -> added-mass-informed integration (surge fixed,
   heave and pitch free).

The last point is why a sigma grid is worth the complexity: the measured
elevation is a state variable, not something recovered from a smeared ``alpha``
field, so it cannot be quantised to ``dy``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import cg

from ..wave_tank.theory import GRAVITY
from .wave_target import WaveTarget

__all__ = ["TankConfig", "SigmaWaveTank", "points_in_polygon"]


def points_in_polygon(px, py, poly) -> np.ndarray:
    """Vectorised even-odd (ray crossing) point-in-polygon test."""
    px = np.asarray(px, dtype=float).ravel()
    py = np.asarray(py, dtype=float).ravel()
    inside = np.zeros(px.shape, dtype=bool)
    poly = np.asarray(poly, dtype=float)
    n = poly.shape[0]
    for k in range(n):
        x1, y1 = poly[k]
        x2, y2 = poly[(k + 1) % n]
        if y1 == y2:
            continue
        straddle = (y1 > py) != (y2 > py)
        if not straddle.any():
            continue
        xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= straddle & (px < xint)
    return inside


def polygon_properties(poly, density: float = 1000.0):
    """Area, centroid and second moment of a simple polygon (per unit width)."""
    p = np.asarray(poly, dtype=float)
    x, y = p[:, 0], p[:, 1]
    xn, yn = np.roll(x, -1), np.roll(y, -1)
    cross = x * yn - xn * y
    area2 = float(np.sum(cross))
    area = 0.5 * area2
    if abs(area) < 1e-14:
        raise ValueError("degenerate section polygon")
    cx = float(np.sum((x + xn) * cross)) / (3.0 * area2)
    cy = float(np.sum((y + yn) * cross)) / (3.0 * area2)
    # second moment about the centroid (per unit width, rho = 1)
    i_o = 0.0
    for k in range(p.shape[0]):
        x1, y1 = p[k]
        x2, y2 = p[(k + 1) % p.shape[0]]
        cr = x1 * y2 - x2 * y1
        i_o += cr * (x1 * x1 + x1 * x2 + x2 * x2 + y1 * y1 + y1 * y2 + y2 * y2)
    i_o /= 12.0
    i_c = abs(i_o) - abs(area) * (cx * cx + cy * cy)
    return float(abs(area)), cx, cy, abs(i_c) * density


@dataclass
class TankConfig:
    length: float = 12.0          # tank length [m]
    depth: float = 1.0            # still-water depth h [m]
    nx: int = 240                 # columns
    ns: int = 32                  # sigma cells (uniform in sigma)
    gravity: float = GRAVITY
    density: float = 1000.0
    advection_blend: float = 0.15   # 0 = central (non-dissipative), 1 = upwind
    nu: float = 0.0                 # kinematic viscosity [m^2/s]
    relax_waves: float = 1.0        # relaxation-zone length in wavelengths
    damp_waves: float = 1.5         # sponge length in wavelengths (>= 1.5 required)
    damp_strength: float = 6.0      # sponge rate = damp_strength * omega [1/s]
    cg_rtol: float = 1e-11
    cg_maxiter: int = 4000
    mask_sub: int = 8               # face sub-samples for the body fractions


def _solve_spd(A, b, x0, rtol, maxiter):
    try:
        return cg(A, b, x0=x0, rtol=rtol, maxiter=maxiter)
    except TypeError:                                     # scipy < 1.12
        return cg(A, b, x0=x0, tol=rtol, maxiter=maxiter)


class SigmaWaveTank:
    """2-D vertical-plane numerical wave tank with heave/pitch body support."""

    def __init__(self, cfg: TankConfig):
        self.cfg = cfg
        self.nx = int(cfg.nx)
        self.ns = int(cfg.ns)
        self.dx = cfg.length / self.nx
        self.dsig = 1.0 / self.ns
        self.x = (np.arange(self.nx) + 0.5) * self.dx       # column centres
        self.xf = np.arange(self.nx + 1) * self.dx          # x-faces
        self.sigf = np.arange(self.ns + 1) * self.dsig
        self.sigc = (np.arange(self.ns) + 0.5) * self.dsig
        self.eta = np.zeros(self.nx)
        self.u = np.zeros((self.nx + 1, self.ns))
        self.w = np.zeros((self.nx, self.ns + 1))
        self.t = 0.0
        self.wave: WaveTarget | None = None
        self.body: dict | None = None
        self._build_grid()
        self.p = np.zeros((self.nx, self.ns))               # non-hydrostatic pressure
        self.ox = np.ones((self.nx + 1, self.ns))
        self.os = np.ones((self.nx, self.ns + 1))
        self.solid = np.zeros((self.nx, self.ns), dtype=bool)
        self.div_residual = 0.0
        self.volume0 = self.volume()
        self.volume_drift = 0.0

    # ------------------------------------------------------------------ grid
    def _build_grid(self):
        cfg = self.cfg
        self.H = cfg.depth + self.eta
        self.Hf = np.empty(self.nx + 1)
        self.Hf[1:-1] = 0.5 * (self.H[:-1] + self.H[1:])
        self.Hf[0] = self.H[0]
        self.Hf[-1] = self.H[-1]
        self.Ax = self.dsig * self.Hf                        # x-face area
        self.Wx = self.dx * self.dsig * self.Hf              # x-face dual volume
        self.V = self.dx * self.dsig * self.H                # cell volume
        self.Ws = self.V.copy()                              # sigma-face dual volume
        self.yf = -cfg.depth + np.outer(self.H, self.sigf)   # (nx, ns+1)
        self.yc = -cfg.depth + np.outer(self.H, self.sigc)   # (nx, ns)
        self.dy = np.broadcast_to(self.dsig * self.H[:, None], (self.nx, self.ns))

    def volume(self) -> float:
        return float(np.sum(self.dx * self.H))

    # ----------------------------------------------------------------- setup
    def set_wave(self, amp: float, period: float, phase0: float = 0.0,
                 order2: bool = True) -> WaveTarget:
        self.wave = WaveTarget(amp, period, self.cfg.depth, self.cfg.gravity,
                               phase0=phase0, order2=order2)
        return self.wave

    def set_body(self, polygon, mass=None, inertia=None, added_mass=(0.0, 0.0),
                 ref=None):
        """Install a floating 2-D section.

        ``ref`` is the reference point for the pitch rotation and the moment; it
        defaults to the section's own vertical axis ``(cx, 0)``.  Rotating about
        the *tank origin* instead would sink the body whenever it pitches
        (``x * theta`` of vertical displacement), which is a bookkeeping error
        that looks exactly like a spurious pitch-heave coupling.
        """
        poly = np.asarray(polygon, dtype=float)
        area, cx, cy, inertia_c = polygon_properties(poly, self.cfg.density)
        if ref is None:
            ref = (float(cx), 0.0)
        ref = np.asarray(ref, dtype=float)
        if mass is None:
            mass = self.cfg.density * area
        if inertia is None:
            inertia = inertia_c + mass * ((cx - ref[0]) ** 2 + (cy - ref[1]) ** 2)
        self.body = {
            "poly": poly, "mass": float(mass), "inertia": float(inertia),
            "area": float(area), "cg": np.array([cx, cy]),
            "added": np.asarray(added_mass, dtype=float),
            "ref": np.asarray(ref, dtype=float),
            "xi": np.zeros(2), "vel": np.zeros(2), "acc": np.zeros(2),
            "force": np.zeros(2), "moment": 0.0,
            "time_series": {"t": [], "heave": [], "pitch": [], "fx": [], "fy": [],
                            "m5": []},
        }
        return self.body

    def initialize_from_wave(self, t: float = 0.0):
        """Set eta, u, w to the analytic target wave (propagation test start)."""
        if self.wave is None:
            raise RuntimeError("set_wave() first")
        self.eta = np.asarray(self.wave.eta(self.x, t), dtype=float).copy()
        self._build_grid()
        ut, wt = self._target_fields(t)
        self.u = ut.copy()
        self.w = wt.copy()
        self.p = np.zeros((self.nx, self.ns))
        self.t = float(t)
        self.volume0 = self.volume()

    # -------------------------------------------------------------- body mask
    def body_polygon(self):
        b = self.body
        if b is None:
            return None
        th = float(b["xi"][1])
        c, s = math.cos(th), math.sin(th)
        p = b["poly"] @ np.array([[c, s], [-s, c]])       # rotate
        p = p + np.array([0.0, float(b["xi"][0])])
        return p

    def _update_mask(self):
        nx, ns, cfg = self.nx, self.ns, self.cfg
        poly = self.body_polygon()
        ox_sub = np.ones((nx + 1, ns))
        os_sub = np.ones((nx, ns + 1))
        self.solid = np.zeros((nx, ns), dtype=bool)
        if poly is not None:
            sub = int(cfg.mask_sub)
            ssub = (np.arange(sub) + 0.5) / sub
            lo, hi = poly.min(axis=0), poly.max(axis=0)
            yf_f = -cfg.depth + np.outer(self.Hf, self.sigf)
            ylo, yhi = yf_f[:, :-1], yf_f[:, 1:]
            y = ylo[:, :, None] + ssub[None, None, :] * (yhi - ylo)[:, :, None]
            ok = (self.xf >= lo[0] - self.dx) & (self.xf <= hi[0] + self.dx)
            if ok.any():
                ii = np.nonzero(ok)[0]
                xx = np.broadcast_to(self.xf[ii, None, None], (ii.size, ns, sub))
                cnt = points_in_polygon(xx, y[ii], poly).reshape(ii.size, ns, sub).sum(-1)
                ox_sub[ii] = 1.0 - cnt / sub
            xs = self.xf[:-1][:, None] + ssub[None, :] * self.dx
            okj = (self.x >= lo[0] - self.dx) & (self.x <= hi[0] + self.dx)
            if okj.any():
                jj = np.nonzero(okj)[0]
                xx = np.broadcast_to(xs[jj][:, None, :], (jj.size, ns + 1, sub))
                yy = np.broadcast_to(self.yf[jj][:, :, None], (jj.size, ns + 1, sub))
                cnt = points_in_polygon(xx, yy, poly).reshape(jj.size, ns + 1, sub).sum(-1)
                os_sub[jj] = 1.0 - cnt / sub
            self.solid = points_in_polygon(
                np.broadcast_to(self.x[:, None], (nx, ns)), self.yc, poly).reshape(nx, ns)
        # topological closure: a face with exactly one fully solid neighbour is
        # blocked.  Grid-aligned sections put every sub-sample exactly on the
        # polygon edge, where a point-in-polygon test is ambiguous; this term
        # makes the mask exact for them and leaves cut cells to the sub-sampling.
        closed_x = np.zeros((nx + 1, ns), dtype=bool)
        closed_x[1:-1] = self.solid[1:, :] ^ self.solid[:-1, :]
        closed_s = np.zeros((nx, ns + 1), dtype=bool)
        closed_s[:, 1:-1] = self.solid[:, 1:] ^ self.solid[:, :-1]
        self.ox = np.where(closed_x, 0.0, ox_sub)
        self.os = np.where(closed_s, 0.0, os_sub)
        self.ox[0, :] = 1.0 if self.wave is not None else 0.0
        self.ox[-1, :] = 0.0
        self.os[:, 0] = 0.0
        closed = ((self.ox[:-1, :] <= 1e-12) & (self.ox[1:, :] <= 1e-12)
                  & (self.os[:, :-1] <= 1e-12) & (self.os[:, 1:] <= 1e-12))
        self.solid = self.solid | closed

    # ------------------------------------------------------------- predictor
    def _ddx(self, f, uvel):
        cen = np.empty_like(f)
        cen[1:-1] = (f[2:] - f[:-2]) / (2.0 * self.dx)
        cen[0] = (f[1] - f[0]) / self.dx
        cen[-1] = (f[-1] - f[-2]) / self.dx
        up = np.empty_like(f)
        up[1:-1] = np.where(uvel[1:-1] > 0.0,
                            (f[1:-1] - f[:-2]) / self.dx,
                            (f[2:] - f[1:-1]) / self.dx)
        up[0] = (f[1] - f[0]) / self.dx
        up[-1] = (f[-1] - f[-2]) / self.dx
        b = self.cfg.advection_blend
        return (1.0 - b) * cen + b * up

    def _ddy(self, f, wvel):
        dy1 = self.dsig * self.H                             # (nx,)
        dy = dy1[:, None]
        cen = np.empty_like(f)
        cen[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dy)
        cen[:, 0] = (f[:, 1] - f[:, 0]) / dy1
        cen[:, -1] = (f[:, -1] - f[:, -2]) / dy1
        up = np.empty_like(f)
        up[:, 1:-1] = np.where(wvel[:, 1:-1] > 0.0,
                               (f[:, 1:-1] - f[:, :-2]) / dy,
                               (f[:, 2:] - f[:, 1:-1]) / dy)
        up[:, 0] = (f[:, 1] - f[:, 0]) / dy1
        up[:, -1] = (f[:, -1] - f[:, -2]) / dy1
        b = self.cfg.advection_blend
        return (1.0 - b) * cen + b * up

    def _predict(self, dt: float):
        g = self.cfg.gravity
        u, w = self.u, self.w
        uc = 0.5 * (u[:-1, :] + u[1:, :])
        wc = 0.5 * (w[:, :-1] + w[:, 1:])
        adv_u = uc * self._ddx(uc, uc) + wc * self._ddy(uc, wc)
        adv_w = uc * self._ddx(wc, uc) + wc * self._ddy(wc, wc)
        if self.cfg.nu > 0.0:
            dy = (self.dsig * self.H)[:, None]
            lap_u = np.zeros_like(uc)
            lap_u[1:-1] = (uc[2:] - 2 * uc[1:-1] + uc[:-2]) / self.dx ** 2
            lap_u[:, 1:-1] = lap_u[:, 1:-1] + (uc[:, 2:] - 2 * uc[:, 1:-1] + uc[:, :-2]) / dy ** 2
            lap_w = np.zeros_like(wc)
            lap_w[1:-1] = (wc[2:] - 2 * wc[1:-1] + wc[:-2]) / self.dx ** 2
            lap_w[:, 1:-1] = lap_w[:, 1:-1] + (wc[:, 2:] - 2 * wc[:, 1:-1] + wc[:, :-2]) / dy ** 2
            adv_u = adv_u - self.cfg.nu * lap_u
            adv_w = adv_w - self.cfg.nu * lap_w
        a_u = np.zeros_like(u)
        a_u[1:-1, :] = 0.5 * (adv_u[:-1, :] + adv_u[1:, :])
        a_w = np.zeros_like(w)
        a_w[:, 1:-1] = 0.5 * (adv_w[:, :-1] + adv_w[:, 1:])
        a_w[:, -1] = adv_w[:, -1]
        # The pressure is split as p = g(eta - y) + q with q the NON-hydrostatic
        # part (q = 0 on the free surface).  The barotropic part g*d(eta)/dx is
        # then an exact column quantity -- no sigma metric error -- and gravity is
        # cancelled analytically by the hydrostatic part in the vertical
        # momentum.  Only q is projected, which is the whole point: the sigma
        # pressure-gradient error can only pollute the small non-hydrostatic
        # remainder.
        detadx = np.zeros(self.nx + 1)
        detadx[1:-1] = (self.eta[1:] - self.eta[:-1]) / self.dx
        us = u - dt * (a_u + g * detadx[:, None])
        ws = w - dt * a_w
        us[0, :] = 0.0
        us[-1, :] = 0.0
        ws[:, 0] = 0.0
        return us, ws

    # ---------------------------------------------------------- forcing zones
    def relax_weight(self) -> np.ndarray:
        """Smoothstep weight 1 at the generation wall -> 0 at the zone end."""
        cfg, wave = self.cfg, self.wave
        if wave is None or cfg.relax_waves <= 0.0:
            return np.zeros(self.nx + 1)
        xr = cfg.relax_waves * wave.wavelength
        t = np.clip(self.xf / xr, 0.0, 1.0)
        return 1.0 - (3.0 * t ** 2 - 2.0 * t ** 3)

    def sponge_rate(self, x) -> np.ndarray:
        """Quadratic ramp of the absorbing-beach rate [1/s] along the tank."""
        cfg, wave = self.cfg, self.wave
        x = np.asarray(x, dtype=float)
        if wave is None or cfg.damp_waves <= 0.0 or cfg.damp_strength <= 0.0:
            return np.zeros_like(x)
        x0 = cfg.length - cfg.damp_waves * wave.wavelength
        s = np.clip((x - x0) / max(cfg.length - x0, 1e-12), 0.0, 1.0) ** 2
        return cfg.damp_strength * wave.omega * s

    def _target_fields(self, t):
        wave = self.wave
        left = np.vstack([self.yc[:1], self.yc[:-1]])
        yf_u = 0.5 * (left + self.yc)                        # (nx, ns)
        yt_u = np.vstack([yf_u, self.yc[-1:]])               # (nx+1, ns)
        xt_u = np.broadcast_to(self.xf[:, None], (self.nx + 1, self.ns))
        ut, _ = wave.velocities(xt_u, yt_u, t)
        xt_w = np.broadcast_to(self.x[:, None], (self.nx, self.ns + 1))
        _, wt = wave.velocities(xt_w, self.yf, t)
        return ut, wt

    def _apply_zones(self, us, ws, t, dt):
        wrx = self.relax_weight()
        if wrx.any():
            ut, wt = self._target_fields(t)
            us[:] = wrx[:, None] * ut + (1.0 - wrx[:, None]) * us
            wrw = (0.5 * (wrx[:-1] + wrx[1:]))[:, None]
            ws[:] = wrw * wt + (1.0 - wrw) * ws
        rate_x = self.sponge_rate(self.xf)
        if rate_x.any():
            us *= np.clip(1.0 - dt * rate_x, 0.0, 1.0)[:, None]
            rate_c = self.sponge_rate(self.x)
            ws *= np.clip(1.0 - dt * rate_c, 0.0, 1.0)[:, None]
        return us, ws

    # ------------------------------------------------------------ projection
    def _poisson_matrix(self):
        nx, ns = self.nx, self.ns
        Aopen_x = self.ox * self.Ax[:, None]
        Aopen_s = self.os * self.dx
        cax = Aopen_x ** 2 / self.Wx[:, None]
        cas = Aopen_s ** 2 / self.Ws[:, None]
        idx = np.arange(nx * ns).reshape(nx, ns)
        ii, jj = np.meshgrid(np.arange(1, nx), np.arange(ns), indexing="ij")
        c = cax[1:nx, :]
        r1, r2 = idx[ii, jj], idx[ii - 1, jj]
        kk = np.arange(1, ns)
        ii2, kk2 = np.meshgrid(np.arange(nx), kk, indexing="ij")
        c2 = cas[ii2, kk2]
        ra, rb = idx[ii2, kk2], idx[ii2, kk2 - 1]
        ra3 = idx[np.arange(nx), ns - 1]
        # The free surface is a Dirichlet face half a cell above the top pressure
        # node: its dual volume is half a cell, so A/W = 2/dy (not 1/dy).  Both
        # the matrix entry and the velocity correction carry that factor 2, which
        # is what makes the discrete free-surface stiffness equal g.
        c3 = 2.0 * cas[np.arange(nx), ns]
        # Symmetric operator M = B W^-1 B^T (positive definite).  The projection is
        # u = u* - dt W^-1 B^T p with flux(u) = 0, hence  M p = -flux(u*)/dt.
        rows = np.concatenate([r1.ravel(), r1.ravel(), r2.ravel(), r2.ravel(),
                               ra.ravel(), ra.ravel(), rb.ravel(), rb.ravel(), ra3])
        cols = np.concatenate([r1.ravel(), r2.ravel(), r2.ravel(), r1.ravel(),
                               ra.ravel(), rb.ravel(), rb.ravel(), ra.ravel(), ra3])
        vals = np.concatenate([c.ravel(), -c.ravel(), c.ravel(), -c.ravel(),
                               c2.ravel(), -c2.ravel(), c2.ravel(), -c2.ravel(), c3])
        M = sparse.coo_matrix((vals, (rows, cols)), shape=(nx * ns, nx * ns)).tocsr()
        free = ~self.solid.ravel()
        M = sparse.diags(free.astype(float)) @ M + sparse.diags((~free).astype(float))
        return M, free

    def _project(self, us, ws, dt):
        nx, ns = self.nx, self.ns
        Aopen_x = self.ox * self.Ax[:, None]
        Aopen_s = self.os * self.dx
        M, free = self._poisson_matrix()
        flux = (Aopen_x[1:, :] * us[1:, :] - Aopen_x[:-1, :] * us[:-1, :]
                + Aopen_s[:, 1:] * ws[:, 1:] - Aopen_s[:, :-1] * ws[:, :-1])
        rhs = -flux.ravel() / dt
        rhs[~free] = 0.0
        p, info = _solve_spd(M, rhs, self.p.ravel(), self.cfg.cg_rtol,
                             self.cfg.cg_maxiter)
        p = p.reshape(nx, ns)
        coef_u = dt * Aopen_x / self.Wx[:, None]
        coef_s = dt * Aopen_s / self.Ws[:, None]
        u_new = us.copy()
        u_new[1:-1, :] -= coef_u[1:-1, :] * (p[1:, :] - p[:-1, :])
        w_new = ws.copy()
        if ns > 1:
            w_new[:, 1:-1] -= coef_s[:, 1:-1] * (p[:, 1:] - p[:, :-1])
        w_new[:, -1] = ws[:, -1] + 2.0 * coef_s[:, -1] * p[:, -1]   # p_out = 0, half cell
        w_new[:, 0] = 0.0
        flux_new = (Aopen_x[1:, :] * u_new[1:, :] - Aopen_x[:-1, :] * u_new[:-1, :]
                    + Aopen_s[:, 1:] * w_new[:, 1:] - Aopen_s[:, :-1] * w_new[:, :-1])
        scale = max(float(np.abs(flux).max()), 1e-30)
        self.div_residual = (float(np.abs(flux_new.ravel()[free]).max() / scale)
                             if free.any() else 0.0)
        self.p = p
        return u_new, w_new

    # ------------------------------------------------------------------- step
    def step(self, dt: float, t=None):
        if t is None:
            t = self.t
        self._update_mask()
        us, ws = self._predict(dt)
        us, ws = self._apply_zones(us, ws, t, dt)
        u_new, w_new = self._project(us, ws, dt)
        self.u, self.w = u_new, w_new
        top_open = ~self.solid[:, -1]
        self.eta = self.eta + np.where(top_open, dt * w_new[:, -1], 0.0)
        # Absorbing beach: in an incompressible projection solver a velocity-only
        # sponge is undone by the pressure correction (mass is re-routed, the
        # beach only produces a set-up).  The surface itself must be relaxed, so
        # the beach acts as the open boundary it stands for.
        rate_c = self.sponge_rate(self.x)
        if rate_c.any():
            self.eta = self.eta * np.clip(1.0 - dt * rate_c, 0.0, 1.0)
        self.t = t + dt
        self._build_grid()
        self.volume_drift = self.volume() - self.volume0
        if self.body is not None:
            self._body_dynamics(dt)
            self._record_body(self.t)
        return self.t

    def total_pressure(self) -> np.ndarray:
        """Physical (kinematic) pressure ``g(eta - y) + q`` on cell centres."""
        g = self.cfg.gravity
        return g * (self.eta[:, None] - self.yc) + self.p

    # ----------------------------------------------------------- body forces
    def _cell_coverage(self):
        return 0.25 * ((1 - self.ox[:-1, :]) + (1 - self.ox[1:, :])
                       + (1 - self.os[:, :-1]) + (1 - self.os[:, 1:]))

    def body_force(self):
        """Force ``(Fx, Fy)`` [N/m] and pitch moment about ``ref`` [Nm/m].

        The pressure is split as ``p = g(eta - y) + q``.  Archimedes gives the
        hydrostatic part exactly from the *polygon* (``rho g nabla`` acting at the
        centre of buoyancy), so only the non-hydrostatic ``q`` is integrated over
        the stair-step wetted faces.  That keeps the buoyancy free of the O(dy)
        mask bias while the dynamic load keeps the cut-cell resolution.
        """
        b = self.body
        g, rho = self.cfg.gravity, self.cfg.density
        nx, ns = self.nx, self.ns
        zc = np.zeros((1, ns))
        zs = np.zeros((nx, 1))
        th_c = self._cell_coverage()                         # (nx, ns)
        p = self.p                                        # non-hydrostatic part
        ref = b["ref"]
        fx = fy = m5 = 0.0
        # --- x-faces: face i separates cell i-1 (left) and cell i (right)
        thx = 1.0 - self.ox                                  # (nx+1, ns)
        if thx.any():
            thl = np.vstack([zc, th_c])                      # cell i-1
            thr = np.vstack([th_c, zc])                      # cell i
            right = thr > thl                                # solid on the right
            pcell = np.where(right, np.vstack([zc, p]), np.vstack([p, zc]))
            ymid = -self.cfg.depth + np.outer(self.Hf, self.sigc)
            pface = pcell
            # only a face with exactly one fully solid neighbour is wetted
            use = (thr > 0.5) ^ (thl > 0.5)
            contrib = np.where(use, pface * thx * self.Ax[:, None], 0.0)
            sgn = np.where(right, 1.0, -1.0)
            fx += rho * float(np.sum(sgn * contrib))
            m5 += rho * float(np.sum(-(ymid - ref[1]) * sgn * contrib))
        # --- sigma-faces: face k separates cell k-1 (below) and cell k (above)
        thy = 1.0 - self.os                                  # (nx, ns+1)
        if thy.any():
            th_dn = np.hstack([zs, th_c])                    # cell k-1 (below)
            th_up = np.hstack([th_c, zs])                    # cell k   (above)
            up = th_up > th_dn                               # solid above -> fluid below
            pcell = np.where(up, np.hstack([zs, p]), np.hstack([p, zs]))
            pface = pcell
            use = (th_up > 0.5) ^ (th_dn > 0.5)
            contrib = np.where(use, pface * thy * self.dx, 0.0)
            sgn = np.where(up, 1.0, -1.0)
            fy += rho * float(np.sum(sgn * contrib))
            xmid = np.broadcast_to(self.x[:, None], (nx, ns + 1))
            m5 += rho * float(np.sum((xmid - ref[0]) * sgn * contrib))
        # --- exact hydrostatic part (Archimedes) from the section polygon
        wet = self.wetted_section()
        if wet is not None and len(wet) >= 3:
            area, cx, _cz, _ = polygon_properties(wet, rho)
            fy += rho * g * area
            m5 += rho * g * area * (cx - ref[0])
        return fx, fy, m5

    def wetted_section(self):
        """Polygon of the section below the still-water line ``y = 0``."""
        if self.body is None:
            return None
        poly = self.body_polygon()
        out = []
        n = len(poly)
        for i in range(n):
            a, bb = poly[i], poly[(i + 1) % n]
            ina, inb = a[1] <= 0.0, bb[1] <= 0.0
            if ina:
                out.append(a)
            if ina != inb:
                t = (0.0 - a[1]) / (bb[1] - a[1])
                out.append(np.array([a[0] + t * (bb[0] - a[0]), 0.0]))
        return np.array(out) if out else None

    def displaced_volume(self, sub: int = 6) -> float:
        """Sub-cell solid volume of the section per unit width [m^2].

        The step-time mask uses face fractions (cheap); for buoyancy bookkeeping
        the cell volume fraction is integrated properly, so a grid-aligned
        section reproduces ``length * draft`` to the sampling error.
        """
        if self.body is None:
            return 0.0
        poly = self.body_polygon()
        s = (np.arange(sub) + 0.5) / sub
        xs = self.xf[:-1, None] + s[None, :] * self.dx            # (nx, sub)
        frac = np.zeros(self.nx)
        for j in range(self.ns):
            ylo = self.yf[:, j][:, None]
            yhi = self.yf[:, j + 1][:, None]
            y = ylo + s[None, :] * (yhi - ylo)                    # (nx, sub)
            xx = np.broadcast_to(xs[:, :, None], (self.nx, sub, sub))
            yy = np.broadcast_to(y[:, None, :], (self.nx, sub, sub))
            ok = (self.x >= poly[:, 0].min() - self.dx) & (self.x <= poly[:, 0].max() + self.dx)
            if ok.any():
                idx = np.nonzero(ok)[0]
                cnt = points_in_polygon(xx[idx], yy[idx], poly).reshape(idx.size, sub, sub)
                frac[idx] += cnt.mean(axis=(1, 2))
        return float(np.sum(frac * self.V))

    def _body_dynamics(self, dt: float):
        b = self.body
        g = self.cfg.gravity
        fx, fy, m5 = self.body_force()
        b["force"] = np.array([fx, fy])
        b["moment"] = m5
        m_eff = b["mass"] + b["added"][0]
        i_eff = b["inertia"] + b["added"][1]
        # moment of the weight about the reference point: the CG rides with the
        # body, so this is what cancels the buoyancy moment of an off-centre
        # section and what makes the pitch restoring (via the rotating wetted
        # polygon) a physical rather than a bookkeeping quantity.
        th = float(b["xi"][1])
        dx = b["cg"][0] - b["ref"][0]
        dz = b["cg"][1] - b["ref"][1]
        c, sn = math.cos(th), math.sin(th)
        xg = b["ref"][0] + dx * c - dz * sn
        m_weight = -(xg - b["ref"][0]) * b["mass"] * g
        b["acc"] = np.array([(fy - b["mass"] * g) / m_eff, (m5 + m_weight) / i_eff])
        b["vel"] = b["vel"] + dt * b["acc"]
        b["xi"] = b["xi"] + dt * b["vel"]
        b["xi"][1] = float(np.clip(b["xi"][1], -0.35, 0.35))

    def _record_body(self, t):
        b = self.body
        ts = b["time_series"]
        ts["t"].append(float(t))
        ts["heave"].append(float(b["xi"][0]))
        ts["pitch"].append(float(b["xi"][1]))
        ts["fx"].append(float(b["force"][0]))
        ts["fy"].append(float(b["force"][1]))
        ts["m5"].append(float(b["moment"]))

    # --------------------------------------------------------------- probing
    def eta_at(self, xq) -> np.ndarray:
        return np.interp(np.asarray(xq, dtype=float), self.x, self.eta)

    def u_at(self, xq, yq):
        """Bilinear (column, sigma) interpolation of the horizontal velocity."""
        xq = np.atleast_1d(np.asarray(xq, dtype=float))
        yq = np.atleast_1d(np.asarray(yq, dtype=float))
        sig = np.clip((yq + self.cfg.depth) / (self.cfg.depth + self.eta_at(xq)), 0.0, 1.0)
        fi = np.clip(np.searchsorted(self.xf, xq) - 1, 0, self.nx - 1)
        fj = np.clip((sig * self.ns - 0.5), 0, self.ns - 1.001)
        j0 = np.floor(fj).astype(int)
        tj = fj - j0
        return (1 - tj) * self.u[fi + 1, j0] + tj * self.u[fi + 1, j0 + 1]

    def momentum_flux(self, i_face: int) -> float:
        """``int (rho u^2 + p) dy`` at x-face ``i_face`` -- control-surface check."""
        i = int(np.clip(i_face, 1, self.nx - 1))
        ptot = self.total_pressure()
        pface = 0.5 * (ptot[i - 1, :] + ptot[i, :])
        return float(self.cfg.density * self.Ax[i] * np.sum(self.u[i, :] ** 2 + pface))