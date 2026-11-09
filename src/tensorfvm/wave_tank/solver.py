"""Time-domain numerical wave tank (NWT): 2-D free-surface MAC solver.

Model
-----
Two-dimensional vertical-plane tank on a staggered MAC grid with a
**single-phase water / empty-air** free-surface formulation: cells with
``alpha > 0.5`` are water, the rest is atmospheric air whose gauge pressure is
exactly zero (Harlow & Welch 1965; Hirt & Nichols 1981 "SOLA-VOF").  Air dynamics
are not integrated: at a density ratio of 1000 the neglected air inertia is
O(1e-3) of the hydrostatic/orbital terms, and dropping it removes the 1000:1
Poisson coefficient contrast that would otherwise dominate the projection.

One time step (:meth:`WaveTank.step`):

1. **Relaxation-zone generation** (``x < L_r``): ``alpha`` and the face
   velocities are blended toward the analytic target wave,
   ``phi <- (1-w) phi + w phi_target`` with a smoothstep ``w`` from 1 at the wall
   to 0 at ``x = L_r``.  The target elevation carries the Stokes-II bound
   harmonic (:func:`tensorfvm.wave_tank.theory.stokes2_elevation`).
2. **Momentum predictor**: central/upwind-blended advection, water viscosity and
   gravity; obstacle faces carry the rigid-body velocity.
3. **Pressure projection**: masked 5-point Poisson (Jacobi-preconditioned CG,
   warm-started) with ``p = 0`` at the free surface -- the north-face Dirichlet
   distance is the *reconstructed sub-cell* surface distance, so the wave
   amplitude is not quantised to ``dy``.
4. **VOF transport**: flux-form THINC reconstruction (Xiao et al. 2005) sharing
   the corrected face fluxes; the water volume changes only through the domain
   boundary.
5. **Damping zone** (``x > L - L_d``): exponential velocity sponge plus an
   interface relaxation toward still water -- the standard absorbing beach.
6. **Body coupling**: pressure integral over the wetted body faces gives the
   heave force and pitch moment; the 6-DOF equation (surge fixed, heave and
   pitch free) is integrated with an added-mass-informed left-hand side, which
   is what makes loose coupling stable when ``a33 ~ 2 m``.

Probes read ``eta_i = y_bottom + dy * sum_j alpha_ij``: the surface is recovered
**from mass** in each column, not from the smeared interface shape, so the
measured amplitude is insensitive to interface diffusion.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .geometry import polygon_mask
from .theory import dispersion_k, stokes2_elevation

__all__ = ["TankConfig", "RigidBody2D", "WaveTank"]

_TWO_PI = 2.0 * math.pi


@dataclass
class TankConfig:
    """Discretisation and physical parameters of the tank."""

    length: float = 14.0
    height: float = 1.5
    depth: float = 1.0
    nx: int = 560
    ny: int = 90
    rho: float = 1000.0
    nu: float = 1.0e-6
    g: float = 9.80665
    cfl: float = 0.35
    advection_blend: float = 0.25
    thinc_beta: float = 2.6
    relax_wavelengths: float = 1.0
    damp_wavelengths: float = 1.5
    damp_strength: float = 8.0
    cg_rtol: float = 3.0e-6
    cg_max_iter: int = 400
    warm_start: bool = True
    wave_order2: bool = True
    dtype: torch.dtype = torch.float64

    def __post_init__(self) -> None:
        if self.depth <= 0.0 or self.depth >= self.height:
            raise ValueError("need 0 < depth < height (positive freeboard)")
        if self.nx < 16 or self.ny < 12:
            raise ValueError("grid too coarse for a wave tank")
        if not 0.0 <= self.advection_blend <= 1.0:
            raise ValueError("advection_blend must be in [0, 1]")


@dataclass
class RigidBody2D:
    """Heave + pitch rigid body (surge fixed), per unit length."""

    length: float = 2.4
    draft: float = 0.30
    mass: float = 0.0
    inertia: float = 0.0
    radius_gyration_ratio: float = 0.29
    added_mass_heave: float = 0.0
    added_inertia_pitch: float = 0.0
    x_centre: float = 0.0
    heave: float = 0.0
    heave_dot: float = 0.0
    heave_ddot: float = 0.0
    pitch: float = 0.0
    pitch_dot: float = 0.0
    pitch_ddot: float = 0.0
    force_y: float = 0.0
    force_x: float = 0.0
    moment: float = 0.0

    @property
    def pivot_y(self) -> float:
        """Vertical position of the pitch axis (section centre of gravity)."""
        return self.heave - 0.5 * self.draft

    def polygon(self, heave: float | None = None, pitch: float | None = None):
        h = self.heave if heave is None else heave
        p = self.pitch if pitch is None else pitch
        L, T = self.length, self.draft
        local = np.array([[-L / 2, -T / 2], [L / 2, -T / 2],
                          [L / 2, T / 2], [-L / 2, T / 2]])
        c, s = math.cos(p), math.sin(p)
        pts = local @ np.array([[c, s], [-s, c]])
        pts[:, 0] += self.x_centre
        pts[:, 1] += h - 0.5 * T
        return pts


class WaveTank:
    """2-D free-surface MAC wave tank (see the module docstring)."""

    def __init__(self, config: TankConfig | None = None, body: RigidBody2D | None = None):
        c = self.cfg = config or TankConfig()
        self.dx = c.length / c.nx
        self.dy = c.height / c.ny
        self.xc = (np.arange(c.nx) + 0.5) * self.dx
        self.yc = (np.arange(c.ny) + 0.5) * self.dy - c.depth
        self.xu = np.arange(c.nx + 1) * self.dx
        self.yv = np.arange(c.ny + 1) * self.dy - c.depth
        self._yc_t = torch.from_numpy(self.yc).unsqueeze(1)

        ny, nx = c.ny, c.nx
        self.alpha = torch.zeros((ny, nx), dtype=c.dtype)
        yc_col = torch.from_numpy(self.yc).unsqueeze(1)
        self.alpha = torch.clamp((torch.zeros((ny, nx), dtype=c.dtype)
                                  - (yc_col - 0.5 * self.dy)) / self.dy, 0.0, 1.0)
        self.u = torch.zeros((ny, nx + 1), dtype=c.dtype)
        self.v = torch.zeros((ny + 1, nx), dtype=c.dtype)
        self.p = torch.zeros((ny, nx), dtype=c.dtype)
        self.solid = torch.zeros((ny, nx), dtype=torch.bool)
        self.u_body = torch.zeros((ny, nx + 1), dtype=c.dtype)
        self.v_body = torch.zeros((ny + 1, nx), dtype=c.dtype)
        self._dp_warm = None
        self.body = body
        try:  # tiny grids: the per-op thread overhead dominates

            torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
        except Exception:
            pass
        self.t = 0.0
        self.steps = 0
        self.wave = None
        self.x_relax = 0.0
        self.x_damp_start = c.length
        self.diag = {"cg_iterations": 0, "div_residual": 0.0, "volume": 0.0,
                     "volume_drift": 0.0, "body_force_y": 0.0, "body_moment": 0.0}
        if body is not None:
            self._init_body(body)
        self._update_solid_masks()
        self.initialize_hydrostatic()
        self._vol0 = self.liquid_volume()

    # ---------------------------------------------------------------- setup
    def set_wave(self, amp: float, period: float, phase0: float = 0.0) -> dict:
        omega = _TWO_PI / period
        k = dispersion_k(omega, self.cfg.depth, self.cfg.g)
        lam = _TWO_PI / k
        self.wave = {"amp": float(amp), "period": float(period), "omega": omega,
                     "k": k, "lambda": lam, "phase0": float(phase0),
                     "depth": self.cfg.depth, "g": self.cfg.g,
                     "omega_over_sqrt_g": omega / math.sqrt(self.cfg.g)}
        self.x_relax = self.cfg.relax_wavelengths * lam
        self.x_damp_start = self.cfg.length - self.cfg.damp_wavelengths * lam
        return self.wave

    def _init_body(self, body: RigidBody2D) -> None:
        c = self.cfg
        if body.mass <= 0.0:
            body.mass = c.rho * body.length * body.draft
        if body.inertia <= 0.0:
            k55 = body.radius_gyration_ratio * body.length
            body.inertia = body.mass * k55 * k55
        if body.added_mass_heave <= 0.0:
            body.added_mass_heave = 0.5 * c.rho * math.pi * (body.length / 2.0) ** 2
        if body.added_inertia_pitch <= 0.0:
            body.added_inertia_pitch = body.added_mass_heave * (body.length / 4.0) ** 2
        if body.x_centre <= 0.0:
            body.x_centre = 0.5 * c.length

    def _update_solid_masks(self) -> None:
        if self.body is None:
            self.solid.zero_()
            self.u_body.zero_()
            self.v_body.zero_()
            return
        b = self.body
        self.solid = torch.from_numpy(polygon_mask(self.xc, self.yc, b.polygon()))
        # water cannot live inside the hull; the displaced volume is booked by the
        # volume diagnostic so the (small) stair-step mass exchange stays visible
        self.alpha = torch.where(self.solid, torch.zeros_like(self.alpha), self.alpha)
        XU, YU = np.meshgrid(self.xu, self.yc, indexing="xy")
        XV, _YV = np.meshgrid(self.xc, self.yv, indexing="xy")
        self.u_body = torch.from_numpy(
            np.ascontiguousarray(-b.pitch_dot * (YU - b.pivot_y)))
        self.v_body = torch.from_numpy(
            np.ascontiguousarray(b.heave_dot + b.pitch_dot * (XV - b.x_centre)))

    # ------------------------------------------------------------ diagnostics
    def surface_elevation(self) -> torch.Tensor:
        """Free surface per column from the liquid volume (sub-cell accurate).

        With a body present the hull's submerged cells are counted as water:
        they *displace* water rather than removing it, so a naive column sum
        would read a surface one draft too low over the hull -- which then feeds
        back into the free-surface Dirichlet distance and corrupts the pressure
        field over the whole hull (measured: buoyancy came out 12x too small
        before this was fixed).
        """
        a = self.alpha
        if self.body is not None:
            a = a + self.solid.to(a.dtype) * (self._yc_t < 0.0).to(a.dtype)
        return a.sum(dim=0) * self.dy - self.cfg.depth

    def liquid_volume(self) -> float:
        return float(self.alpha.sum()) * self.dx * self.dy

    def probe(self, x: float) -> float:
        """Free-surface elevation measured from the column mass at ``x`` (m)."""
        i = int(np.clip(round(x / self.dx - 0.5), 0, self.cfg.nx - 1))
        return float(self.surface_elevation()[i])

    def _zone_weights(self):
        c = self.cfg
        x = self.xc
        if self.x_relax > 0.0:
            s = np.clip(x / self.x_relax, 0.0, 1.0)
            w_relax = 1.0 - s * s * (3.0 - 2.0 * s)
        else:
            w_relax = np.zeros_like(x)
        span = max(c.length - self.x_damp_start, 1e-9)
        s = np.clip((x - self.x_damp_start) / span, 0.0, 1.0)
        return (torch.from_numpy(w_relax).unsqueeze(0),
                torch.from_numpy(s * s).unsqueeze(0))

    def target_fields(self, t: float):
        w = self.wave
        k, om, a = w["k"], w["omega"], w["amp"]
        th = k * self.xc - om * t + w["phase0"]
        eta = stokes2_elevation(self.xc, t, a, om, k, self.cfg.depth,
                                order2=self.cfg.wave_order2)
        sh = math.sinh(k * self.cfg.depth)
        u = om * a * np.cosh(k * (self.yc[:, None] + self.cfg.depth)) / sh * np.cos(th)[None, :]
        v = om * a * np.sinh(k * (self.yc[:, None] + self.cfg.depth)) / sh * np.sin(th)[None, :]
        return eta, u, v

    def _apply_relaxation(self, t: float) -> None:
        w_relax, _ = self._zone_weights()
        if float(w_relax.max()) <= 0.0:
            return
        eta, u_t, v_t = self.target_fields(t)
        alpha_t = torch.clamp((torch.from_numpy(np.asarray(eta)).unsqueeze(0)
                               - (self._yc_t - 0.5 * self.dy)) / self.dy, 0.0, 1.0)
        free = ~self.solid
        w = (w_relax * free).clamp(0.0, 1.0)
        self.alpha = torch.where(free, self.alpha * (1.0 - w) + alpha_t * w, self.alpha)

        u_tf = torch.from_numpy(np.asarray(u_t))
        u_face = torch.zeros_like(self.u)
        u_face[:, 1:-1] = 0.5 * (u_tf[:, :-1] + u_tf[:, 1:])
        u_face[:, 0] = u_tf[:, 0]
        u_face[:, -1] = u_tf[:, -1]
        wu = torch.zeros_like(self.u)
        wu[:, 1:-1] = 0.5 * (w_relax[:, :-1] + w_relax[:, 1:])
        wu[:, 0] = w_relax[:, 0]
        wu[:, -1] = w_relax[:, -1]
        wu = wu.clamp(0.0, 1.0)
        self.u = torch.where(self._open_u_mask(), self.u * (1.0 - wu) + u_face * wu, self.u)

        v_tf = torch.from_numpy(np.asarray(v_t))
        v_face = torch.zeros_like(self.v)
        v_face[1:-1, :] = 0.5 * (v_tf[:-1, :] + v_tf[1:, :])
        v_face[0, :] = v_tf[0, :]
        v_face[-1, :] = v_tf[-1, :]
        wv = w_relax.expand(self.v.shape).clone().clamp(0.0, 1.0)
        self.v = torch.where(self._open_v_mask(), self.v * (1.0 - wv) + v_face * wv, self.v)

    def _apply_damping(self, dt: float) -> None:
        _, w_damp = self._zone_weights()
        if float(w_damp.max()) <= 0.0:
            return
        sigma = self.cfg.damp_strength * w_damp
        fac = torch.clamp(1.0 - sigma * dt, 0.0, 1.0)
        fac_u = torch.cat([fac[:, :1], 0.5 * (fac[:, :-1] + fac[:, 1:]), fac[:, -1:]], 1)
        self.u = torch.where(self._open_u_mask(), self.u * fac_u, self.u)
        self.v = torch.where(self._open_v_mask(), self.v * fac.expand(self.v.shape), self.v)
        alpha_swl = torch.clamp((torch.zeros_like(self.alpha) - (self._yc_t - 0.5 * self.dy))
                                / self.dy, 0.0, 1.0)
        w = torch.clamp(sigma * dt, 0.0, 1.0)
        liq = (self.alpha > 0.5) & (~self.solid)
        self.alpha = torch.where(liq, self.alpha * (1.0 - w) + alpha_swl * w, self.alpha)

    # ------------------------------------------------------------------ faces
    def _open_u_mask(self) -> torch.Tensor:
        """Active x-faces: not inside the hull *and* touching water.

        Air-only faces are held at zero: the air region is not integrated (its
        free fall would otherwise set the global time step through
        :meth:`stable_dt`).
        """
        s = self.solid
        wet = (self.alpha > 0.5) & (~s)
        m = torch.zeros_like(self.u, dtype=torch.bool)
        m[:, 1:-1] = ~(s[:, :-1] & s[:, 1:]) & (wet[:, :-1] | wet[:, 1:])
        m[:, 0] = ~s[:, 0] & wet[:, 0]
        m[:, -1] = ~s[:, -1] & wet[:, -1]
        return m

    def _open_v_mask(self) -> torch.Tensor:
        """Active y-faces: not inside the hull, not a wall, touching water."""
        s = self.solid
        wet = (self.alpha > 0.5) & (~s)
        m = torch.zeros_like(self.v, dtype=torch.bool)
        m[1:-1, :] = ~(s[:-1, :] & s[1:, :]) & (wet[:-1, :] | wet[1:, :])
        m[0, :] = False
        m[-1, :] = False
        return m

    def _blocked_u(self) -> torch.Tensor:
        s = self.solid
        b = torch.zeros_like(self.u, dtype=torch.bool)
        b[:, 1:-1] = s[:, :-1] ^ s[:, 1:]
        b[:, 0] = s[:, 0]
        b[:, -1] = s[:, -1]
        return b

    def _blocked_v(self) -> torch.Tensor:
        s = self.solid
        b = torch.zeros_like(self.v, dtype=torch.bool)
        b[1:-1, :] = s[:-1, :] ^ s[1:, :]
        b[0, :] = True
        b[-1, :] = True
        return b

    @staticmethod
    def _shift_x(a, d):
        """``out[:, i] = a[:, i + d]`` (zero outside)."""
        out = torch.zeros_like(a)
        n = a.shape[1]
        if d > 0:
            out[:, :n - d] = a[:, d:]
        elif d < 0:
            out[:, -d:] = a[:, :n + d]
        else:
            out = a.clone()
        return out

    @staticmethod
    def _shift_y(a, d):
        """``out[j, :] = a[j + d, :]`` (zero outside)."""
        out = torch.zeros_like(a)
        n = a.shape[0]
        if d > 0:
            out[:n - d, :] = a[d:, :]
        elif d < 0:
            out[-d:, :] = a[:n + d, :]
        else:
            out = a.clone()
        return out

    # --------------------------------------------------------------- momentum
    def _advective_accel(self):
        b = self.cfg.advection_blend
        dx, dy = self.dx, self.dy
        u, v = self.u, self.v
        vpad = torch.cat([v[:, :1], v, v[:, -1:]], dim=1)
        v_c_u = 0.25 * (vpad[:-1, :-1] + vpad[1:, :-1] + vpad[:-1, 1:] + vpad[1:, 1:])
        dudx_c = (self._shift_x(u, +1) - self._shift_x(u, -1)) / (2.0 * dx)
        dudx_up = torch.where(u[:, 1:-1] > 0.0, (u[:, 1:-1] - u[:, :-2]) / dx,
                              (u[:, 2:] - u[:, 1:-1]) / dx)
        dudx = (1.0 - b) * dudx_c[:, 1:-1] + b * dudx_up
        dudy = (self._shift_y(u, +1) - self._shift_y(u, -1)) / (2.0 * dy)
        adv_u = torch.zeros_like(u)
        adv_u[:, 1:-1] = u[:, 1:-1] * dudx + v_c_u[:, 1:-1] * dudy[:, 1:-1]

        upad = torch.cat([u[:1, :], u, u[-1:, :]], dim=0)
        u_c_v = 0.25 * (upad[:-1, :-1] + upad[:-1, 1:] + upad[1:, :-1] + upad[1:, 1:])
        dvdy_c = (self._shift_y(v, +1) - self._shift_y(v, -1)) / (2.0 * dy)
        dvdy_up = torch.where(v[1:-1, :] > 0.0, (v[1:-1, :] - v[:-2, :]) / dy,
                              (v[2:, :] - v[1:-1, :]) / dy)
        dvdy = (1.0 - b) * dvdy_c[1:-1, :] + b * dvdy_up
        dvdx = (self._shift_x(v, +1) - self._shift_x(v, -1)) / (2.0 * dx)
        adv_v = torch.zeros_like(v)
        adv_v[1:-1, :] = v[1:-1, :] * dvdy + u_c_v[1:-1, :] * dvdx[1:-1, :]
        return adv_u, adv_v

    def _lap_u(self, f):
        """5-point Laplacian on x-faces (interior faces only, neighbours clamped)."""
        ax = (torch.cat([f[:, 1:], f[:, -1:]], 1) + torch.cat([f[:, :1], f[:, :-1]], 1) - 2.0 * f)
        ay = (torch.cat([f[1:, :], f[-1:, :]], 0) + torch.cat([f[:1, :], f[:-1, :]], 0) - 2.0 * f)
        out = torch.zeros_like(f)
        out[:, 1:-1] = ax[:, 1:-1] / self.dx ** 2 + ay[:, 1:-1] / self.dy ** 2
        return out

    def _lap_v(self, f):
        """5-point Laplacian on y-faces (interior faces only, neighbours clamped)."""
        ax = (torch.cat([f[:, 1:], f[:, -1:]], 1) + torch.cat([f[:, :1], f[:, :-1]], 1) - 2.0 * f)
        ay = (torch.cat([f[1:, :], f[-1:, :]], 0) + torch.cat([f[:1, :], f[:-1, :]], 0) - 2.0 * f)
        out = torch.zeros_like(f)
        out[1:-1, :] = ax[1:-1, :] / self.dx ** 2 + ay[1:-1, :] / self.dy ** 2
        return out

    def _face_distances(self, p):
        """Free-surface / wall face distances used by both gradient and correction."""
        dy = self.dy
        ny, nx = self.cfg.ny, self.cfg.nx
        liq = (self.alpha > 0.5) & (~self.solid)
        eta = self.surface_elevation().unsqueeze(0)
        d_surf = torch.clamp(eta - self._yc_t, min=0.15 * dy, max=1.5 * dy)
        north_air_cell = liq[:-1, :] & (~liq[1:, :]) & (~self.solid[1:, :])
        south_air_cell = liq[1:, :] & (~liq[:-1, :]) & (~self.solid[:-1, :])
        north_air = torch.cat([torch.zeros((1, nx), dtype=torch.bool), north_air_cell,
                               torch.zeros((1, nx), dtype=torch.bool)], dim=0)
        south_air = torch.cat([torch.zeros((1, nx), dtype=torch.bool), south_air_cell,
                               torch.zeros((1, nx), dtype=torch.bool)], dim=0)
        d_prev = torch.cat([torch.zeros((1, nx), dtype=p.dtype), d_surf], dim=0)
        d_here = torch.cat([d_surf, torch.zeros((1, nx), dtype=p.dtype)], dim=0)
        dy_full = torch.full((ny + 1, nx), dy, dtype=p.dtype)
        d_nf = torch.where(north_air, d_prev, dy_full)
        d_sf = torch.where(south_air, d_here, dy_full)
        return d_nf, d_sf, north_air, south_air

    def _grad_p(self, p):
        """Face-wise pressure gradient with the free-surface face distances."""
        c = self.cfg
        dx = self.dx
        ny, nx = c.ny, c.nx
        liq = (self.alpha > 0.5) & (~self.solid)
        p = torch.where(liq, p, torch.zeros_like(p))
        zc = torch.zeros((ny, 1), dtype=p.dtype)
        zr = torch.zeros((1, nx), dtype=p.dtype)
        p_east = torch.cat([p, zc], dim=1)
        p_west = torch.cat([zc, p], dim=1)
        p_north = torch.cat([p, zr], dim=0)
        p_south = torch.cat([zr, p], dim=0)
        d_nf, d_sf, _, south_air = self._face_distances(p)
        gu = (p_east - p_west) / dx
        gu[:, 0] = 0.0
        gu[:, -1] = 0.0
        gv = torch.where(south_air, (p_north - p_south) / d_sf, (p_north - p_south) / d_nf)
        gv[0, :] = 0.0
        return gu, gv

    def initialize_hydrostatic(self, level: float = 0.0) -> None:
        """Set ``p`` to the hydrostatic head of the current free surface.

        The incremental projection (:meth:`_predict` carries ``-grad(p)/rho``)
        needs a hydrostatic pressure to start from; otherwise a pool at rest
        free-falls for one step and the projection cannot recover (a uniform
        free-fall velocity field is divergence free, so the projection never
        "sees" gravity -- this is the classic Chorin-projection pitfall).
        """
        c = self.cfg
        liq = (self.alpha > 0.5) & (~self.solid)
        eta = self.surface_elevation().unsqueeze(0)
        head = torch.clamp(eta - self._yc_t, min=0.0)
        self.p = torch.where(liq, c.rho * c.g * head, torch.zeros_like(head))
        self._dp_warm = None

    def _predict(self, dt: float):
        c = self.cfg
        adv_u, adv_v = self._advective_accel()
        gu, gv = self._grad_p(self.p)
        u_star = self.u + dt * (-adv_u - gu / c.rho + c.nu * self._lap_u(self.u))
        v_star = self.v + dt * (-adv_v - gv / c.rho + c.nu * self._lap_v(self.v) - c.g)
        u_star = torch.where(self._blocked_u(), self.u_body, u_star)
        v_star = torch.where(self._blocked_v(), self.v_body, v_star)
        u_star = torch.where(self._open_u_mask(), u_star, self.u)
        v_star = torch.where(self._open_v_mask(), v_star, self.v)
        return u_star, v_star

    # ------------------------------------------------------------- projection
    def _poisson_coeffs(self):
        dx, dy = self.dx, self.dy
        liq = (self.alpha > 0.5) & (~self.solid)
        s = self.solid
        inv_dx2, inv_dy2 = 1.0 / dx ** 2, 1.0 / dy ** 2
        nb_l = {k: torch.zeros_like(liq) for k in ("e", "w", "n", "s")}
        nb_s = {k: torch.zeros_like(liq) for k in ("e", "w", "n", "s")}
        nb_l["e"][:, :-1] = liq[:, 1:]
        nb_l["w"][:, 1:] = liq[:, :-1]
        nb_l["n"][:-1, :] = liq[1:, :]
        nb_l["s"][1:, :] = liq[:-1, :]
        nb_s["e"][:, :-1] = s[:, 1:]
        nb_s["w"][:, 1:] = s[:, :-1]
        nb_s["n"][:-1, :] = s[1:, :]
        nb_s["s"][1:, :] = s[:-1, :]
        # domain walls are no-flux: mark them as "neighbour is solid"
        nb_s["w"][:, 0] = True
        nb_s["e"][:, -1] = True
        nb_s["s"][0, :] = True

        coef = {k: torch.zeros_like(self.p) for k in ("e", "w", "n", "s")}
        off = {k: torch.zeros_like(self.p) for k in ("e", "w", "n", "s")}
        for k, c0 in (("e", inv_dx2), ("w", inv_dx2), ("s", inv_dy2)):
            coef[k][nb_l[k]] = c0
            off[k][nb_l[k]] = c0
            air = liq & (~nb_l[k]) & (~nb_s[k])
            coef[k][air] = c0
        coef["n"][nb_l["n"]] = inv_dy2
        off["n"][nb_l["n"]] = inv_dy2
        eta = self.surface_elevation().unsqueeze(0)
        d_surf = torch.clamp(eta - self._yc_t, min=0.15 * dy, max=1.5 * dy)
        air_n = liq & (~nb_l["n"]) & (~nb_s["n"])
        coef["n"] = torch.where(air_n, (1.0 / (dy * d_surf)).clamp(max=100.0 / dy ** 2),
                                coef["n"])
        # rows of non-liquid cells are identity (q = 0 there): without this the
        # air rows would act as extra constraints coupling into the water and the
        # CG would not converge
        for k in ("e", "w", "n", "s"):
            coef[k] = torch.where(liq, coef[k], torch.zeros_like(coef[k]))
            off[k] = torch.where(liq, off[k], torch.zeros_like(off[k]))
        diag = coef["e"] + coef["w"] + coef["n"] + coef["s"]
        return liq, diag, off, d_surf

    def _poisson(self, u_star, v_star, dt: float):
        c = self.cfg
        liq, diag, off, _ = self._poisson_coeffs()
        div = ((u_star[:, 1:] - u_star[:, :-1]) / self.dx
               + (v_star[1:, :] - v_star[:-1, :]) / self.dy)
        rhs = torch.where(liq, (-c.rho / dt) * div, torch.zeros_like(div))
        diag_safe = torch.where(diag > 0.0, diag, torch.ones_like(diag))

        def matvec(q):
            return (diag_safe * q
                    - off["e"] * self._shift_x(q, +1) - off["w"] * self._shift_x(q, -1)
                    - off["n"] * self._shift_y(q, +1) - off["s"] * self._shift_y(q, -1))

        rhs_norm = float(torch.linalg.vector_norm(rhs))
        if rhs_norm == 0.0:
            self.diag["cg_iterations"] = 0
            self.diag["div_residual"] = 0.0
            return torch.zeros_like(rhs)
        if c.warm_start and self._dp_warm is not None:
            q = torch.where(liq, self._dp_warm, torch.zeros_like(rhs))
            r = rhs - matvec(q)
        else:
            q = torch.zeros_like(rhs)
            r = rhs.clone()
        z = r / diag_safe
        dirv = z.clone()
        rz = float((r * z).sum())
        target = max(rhs_norm * c.cg_rtol, 1e-18)
        it = 0
        for it in range(1, c.cg_max_iter + 1):
            ad = matvec(dirv)
            den = float((dirv * ad).sum())
            if den <= 0.0 or not math.isfinite(den):
                break
            step = rz / den
            q = q + step * dirv
            r = r - step * ad
            if float(torch.linalg.vector_norm(r)) <= target:
                break
            z = r / diag_safe
            rz_new = float((r * z).sum())
            dirv = z + (rz_new / rz) * dirv
            rz = rz_new
        q = torch.where(liq, q, torch.zeros_like(q))
        self._dp_warm = q
        self.diag["cg_iterations"] = it
        self.diag["div_residual"] = float(torch.linalg.vector_norm(rhs - matvec(q)) / rhs_norm)
        return q

    def _correct(self, u_star, v_star, phi, dt: float):
        """Incremental projection: ``u = u* - dt/rho grad(phi)``, ``p += phi``."""
        c = self.cfg
        gu, gv = self._grad_p(phi)
        u = u_star - dt / c.rho * gu
        v = v_star - dt / c.rho * gv
        u = torch.where(self._blocked_u(), self.u_body, u)
        v = torch.where(self._blocked_v(), self.v_body, v)
        u = torch.where(self._open_u_mask(), u, self.u)
        v = torch.where(self._open_v_mask(), v, self.v)
        liq = (self.alpha > 0.5) & (~self.solid)
        self.p = torch.where(liq, self.p + phi, torch.zeros_like(phi))
        return u, v

    # ------------------------------------------------------------------- VOF
    def _thinc_flux(self, alpha, courant, axis: int = 1):
        """Volume fraction leaving the donor cell through each face (THINC).

        ``courant = u dt / dx`` (signed, face-centred); ``axis=1`` for x-faces
        (``alpha`` is ``(ny, nx)``, ``courant`` is ``(ny, nx+1)``), ``axis=0`` for
        y-faces.  For ``courant > 0`` the donor is the low-index cell.

        THINC reconstruction (Xiao et al. 2005)::

            phi(xi) = 0.5 (1 + tanh(beta (xi - delta))),  xi in [-0.5, 0.5]

        with ``delta`` solved from the donor cell average by bisection.  The
        volume crossing during one step (Courant ``C``) is the closed form::

            F = 0.5 C + (1/(2 beta)) [ ln cosh(beta (0.5 - delta))
                                     - ln cosh(beta (0.5 - C - delta)) ]

        which is bounded by the donor's cell content, so the scheme is
        compressive (sharp interface) *and* conservative in flux form.
        """
        beta = self.cfg.thinc_beta
        ny, nx = alpha.shape
        pos = courant > 0.0
        if axis == 1:
            low = torch.cat([torch.zeros((ny, 1), dtype=alpha.dtype), alpha], dim=1)
            high = torch.cat([alpha, torch.zeros((ny, 1), dtype=alpha.dtype)], dim=1)
        else:
            low = torch.cat([torch.zeros((1, nx), dtype=alpha.dtype), alpha], dim=0)
            high = torch.cat([alpha, torch.zeros((1, nx), dtype=alpha.dtype)], dim=0)
        donor = torch.where(pos, low, high)
        sgn = torch.where(pos, torch.ones_like(courant), -torch.ones_like(courant))
        C = courant.abs().clamp(0.0, 1.0)
        lo = torch.full_like(donor, -0.6)
        hi = torch.full_like(donor, 0.6)
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            val = 0.5 + (1.0 / (2.0 * beta)) * (
                torch.log(torch.cosh(beta * (0.5 - mid)))
                - torch.log(torch.cosh(beta * (0.5 + mid))))
            hi = torch.where(val < donor, mid, hi)
            lo = torch.where(val >= donor, mid, lo)
        delta = 0.5 * (lo + hi)
        flux = 0.5 * C + (1.0 / (2.0 * beta)) * (
            torch.log(torch.cosh(beta * (0.5 - delta)))
            - torch.log(torch.cosh(beta * (0.5 - C - delta))))
        flux = sgn * torch.minimum(torch.clamp(flux, min=0.0), C)
        flux[..., 0] = 0.0
        flux[..., -1] = 0.0
        if axis == 0:
            flux[0, :] = 0.0
            flux[-1, :] = 0.0
        else:
            flux[:, 0] = 0.0
            flux[:, -1] = 0.0
        return flux

    def _advect_alpha(self, dt: float, u, v):
        cu = u * dt / self.dx
        cv = v * dt / self.dy
        fe = self._thinc_flux(self.alpha, cu, axis=1)          # (ny, nx+1)
        fn = self._thinc_flux(self.alpha, cv, axis=0)          # (ny+1, nx)
        a = self.alpha - (fe[:, 1:] - fe[:, :-1]) - (fn[1:, :] - fn[:-1, :])
        return a.clamp(0.0, 1.0)

    # ------------------------------------------------------------------ body
    def _body_loads(self, p):
        """Pressure force and pitch moment on the wetted hull faces (per metre).

        ``dF = p A n_ls`` with ``n_ls`` the unit normal pointing from the liquid
        cell into the solid cell -- the direction the water pushes the hull.
        """
        if self.body is None:
            return 0.0, 0.0, 0.0
        b = self.body
        s = self.solid
        dx, dy = self.dx, self.dy
        p_liq = torch.where(self.alpha > 0.5, p, torch.zeros_like(p))
        # extrapolate the cell pressure to the *face* (hydrostatic part exact):
        # using the cell-centre value directly biases buoyancy by rho g dy/2,
        # i.e. by dy/(2T) of the displacement (measured: 8.3 % for dy/T = 1/6)
        rho_g = self.cfg.rho * self.cfg.g
        liq_w = (~s[:, :-1]) & s[:, 1:]
        sol_w = s[:, :-1] & (~s[:, 1:])
        fx = float((p_liq[:, :-1] * liq_w - p_liq[:, 1:] * sol_w).sum()) * dy
        xc = torch.from_numpy(self.xc).unsqueeze(0)
        mo_x = float(((p_liq[:, :-1] * liq_w - p_liq[:, 1:] * sol_w)
                      * (self._yc_t - b.pivot_y)).sum()) * dy
        liq_s = (~s[:-1, :]) & s[1:, :]
        sol_s = s[:-1, :] & (~s[1:, :])
        yf_col = torch.from_numpy(self.yv[1:-1]).unsqueeze(1)
        p_s = p_liq[:-1, :] - rho_g * (yf_col - self._yc_t[:-1, :])
        p_n = p_liq[1:, :] - rho_g * (yf_col - self._yc_t[1:, :])
        f_face = p_s * liq_s - p_n * sol_s
        fy = float(f_face.sum()) * dx
        mo_y = -float((f_face * (xc - b.x_centre)).sum()) * dx
        return fx, fy, mo_x + mo_y

    def _integrate_body(self, dt: float, fx: float, fy: float, mo: float):
        b = self.body
        m_eff = b.mass + b.added_mass_heave
        i_eff = b.inertia + b.added_inertia_pitch
        b.heave_ddot = (fy - b.mass * self.cfg.g) / m_eff
        b.pitch_ddot = mo / i_eff
        b.heave_dot += dt * b.heave_ddot
        b.pitch_dot += dt * b.pitch_ddot
        b.heave += dt * b.heave_dot
        b.pitch = float(np.clip(b.pitch + dt * b.pitch_dot, -0.30, 0.30))
        b.force_x, b.force_y, b.moment = fx, fy, mo
        self.diag["body_force_y"] = fy
        self.diag["body_moment"] = mo

    # ------------------------------------------------------------------ step
    def stable_dt(self) -> float:
        c = self.cfg
        u_max = max(float(self.u.abs().max()), 1e-6)
        v_max = max(float(self.v.abs().max()), 1e-6)
        return float(min(c.cfl * self.dx / u_max, c.cfl * self.dy / v_max,
                         0.25 * math.sqrt(self.dy / c.g)))

    def step(self, dt: float | None = None) -> dict:
        dt = self.stable_dt() if dt is None else float(dt)
        if self.body is not None:
            self._update_solid_masks()
        u_star, v_star = self._predict(dt)
        phi = self._poisson(u_star, v_star, dt)
        self.u, self.v = self._correct(u_star, v_star, phi, dt)
        vol_before = self.liquid_volume()
        self.alpha = self._advect_alpha(dt, self.u, self.v)
        vol_after = self.liquid_volume()
        self._apply_relaxation(self.t + dt)
        self._apply_damping(dt)
        if self.body is not None:
            fx, fy, mo = self._body_loads(self.p)
            self._integrate_body(dt, fx, fy, mo)
            self._update_solid_masks()
        self.t += dt
        self.steps += 1
        self.diag["volume"] = vol_after
        self.diag["volume_drift"] = (vol_after - self._vol0) / max(abs(self._vol0), 1e-12)
        return {"t": self.t, "dt": dt, "vof_increment": vol_after - vol_before, **self.diag}
