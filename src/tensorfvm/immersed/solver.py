"""Staggered MAC fractional-step solver for immersed solids, with a mass ledger.

Two projection arms share one grid, one momentum predictor and one time step:

``masked``
    Solid cells are removed from the pressure-correction system and every face of
    a solid cell carries a hard zero normal velocity.  Rigidity is a *constraint*,
    so the penalty parameter ``kappa`` no longer enters the answer at all.

``legacy``
    The historical Brinkman/penalty treatment: an unmasked (whole-domain) pressure
    Poisson solve, the momentum penalty sink, and a post-projection relaxation
    ``u <- u / (1 + kappa*dt)`` inside the solid, which is exactly the operation
    that re-introduces a mass source after continuity has been enforced.

The mass ledger records, every step:

``div_max_fluid``     ``max_cells |int_cell div u| / (U H)`` over fluid cells
``div_max_solid``     the same over solid cells (spurious mass source inside the body)
``solid_flux``        ``int_{dS_solid} u . n dS`` (net spurious volume flux into the body)
``solid_flux_gross``  the same in absolute value (gross, not telescoping)
``drift``             accumulated ``solid_flux`` in units of ``U H**2`` (volume drift)
``mass_imbalance``    ``(Q_out - Q_in) / Q_in``

which separates *where* mass is created from *whether the outlet compensates*.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

import torch

from .geometry import CartesianGrid, parabolic_inlet
from .projection import MaskedPoisson

ProjectionMode = Literal["masked", "legacy"]


# --------------------------------------------------------------------- helpers
def _east(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[:, 1:], a[:, -1:]], dim=1)


def _west(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[:, :1], a[:, :-1]], dim=1)


def _north(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[1:, :], a[-1:, :]], dim=0)


def _south(a: torch.Tensor) -> torch.Tensor:
    return torch.cat([a[:1, :], a[:-1, :]], dim=0)


def _avgx_to_u(a: torch.Tensor) -> torch.Tensor:
    """Cell-centred ``(ny, nx)`` -> ``u`` faces ``(ny, nx+1)`` (edge-extrapolated)."""
    padded = torch.cat([a[:, :1], a, a[:, -1:]], dim=1)
    return 0.5 * (padded[:, :-1] + padded[:, 1:])


def _avgy_to_v(a: torch.Tensor) -> torch.Tensor:
    """Cell-centred ``(ny, nx)`` -> ``v`` faces ``(ny+1, nx)`` (edge-extrapolated)."""
    padded = torch.cat([a[:1, :], a, a[-1:, :]], dim=0)
    return 0.5 * (padded[:-1, :] + padded[1:, :])


def face_closures(grid: CartesianGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """Boolean *closed* masks for the ``u`` (ny,nx+1) and ``v`` (ny+1,nx) faces.

    Domain planes are handled by their boundary conditions (inlet prescribed,
    outlet free, ``y`` planes walls), so only interior faces are marked here.
    """
    solid = grid.solid
    ny, nx = solid.shape
    z = torch.zeros((ny, 1), dtype=torch.bool)
    u_closed = torch.cat([z, solid[:, :-1] | solid[:, 1:], z], dim=1)
    zr = torch.zeros((1, nx), dtype=torch.bool)
    v_closed = torch.cat([zr, solid[:-1, :] | solid[1:, :], zr], dim=0)
    return u_closed, v_closed


def solid_face_indicator(grid: CartesianGrid) -> tuple[torch.Tensor, torch.Tensor]:
    """1.0 on faces touching a solid cell (zero on boundary planes)."""
    solid = grid.solid
    ny, nx = grid.solid.shape
    s_u = torch.zeros((ny, nx + 1), dtype=torch.float64)
    s_u[:, 1:-1] = (solid[:, :-1] | solid[:, 1:]).to(torch.float64)
    s_v = torch.zeros((ny + 1, nx), dtype=torch.float64)
    s_v[1:-1, :] = (solid[:-1, :] | solid[1:, :]).to(torch.float64)
    return s_u, s_v


def solid_boundary_flux(grid: CartesianGrid, u: torch.Tensor,
                        v: torch.Tensor) -> tuple[float, float]:
    """Net and gross volume flux *into* the immersed solid (2-D, per unit depth).

    A face is counted when one of its two cells is solid and the other is fluid.
    The sign convention is the solid's outward normal, so a positive value is
    fluid being created inside the body (a mass source) over the step.
    """
    solid = grid.solid
    hx = grid.dx
    hy = grid.dy
    net = 0.0
    gross = 0.0
    # u faces: (j, i) between cell i-1 and cell i
    west_solid = torch.zeros_like(u, dtype=torch.bool)
    east_solid = torch.zeros_like(u, dtype=torch.bool)
    west_solid[:, 1:-1] = solid[:, :-1]
    east_solid[:, 1:-1] = solid[:, 1:]
    interface = (west_solid ^ east_solid)
    if bool(interface.any()):
        vals = u[interface] * hy
        # outward normal of the solid points +x when the solid is west of the face
        sign = torch.where(west_solid[interface], -1.0, 1.0)
        contrib = sign * vals
        net += float(contrib.sum())
        gross += float(contrib.abs().sum())
    south_solid = torch.zeros_like(v, dtype=torch.bool)
    north_solid = torch.zeros_like(v, dtype=torch.bool)
    south_solid[1:-1, :] = solid[:-1, :]
    north_solid[1:-1, :] = solid[1:, :]
    interface_v = (south_solid ^ north_solid)
    if bool(interface_v.any()):
        vals = v[interface_v] * hx
        sign = torch.where(south_solid[interface_v], -1.0, 1.0)
        contrib = sign * vals
        net += float(contrib.sum())
        gross += float(contrib.abs().sum())
    return net, gross


# ----------------------------------------------------------------------- config
@dataclass
class ImmersedConfig:
    grid: CartesianGrid
    viscosity: float
    density: float = 1.0
    projection: ProjectionMode = "masked"
    kappa: float = 1.0e4
    inlet_mean_velocity: float = 1.0
    inlet_bounds: tuple[float, float] | None = None
    reference_length: float = 1.0
    dt: float | None = None
    cfl: float = 0.4
    diffusion_limit: float = 0.2
    t_end: float = 20.0
    max_steps: int = 200_000
    poisson_tol: float = 1e-10
    poisson_maxiter: int = 4000
    mg_cycles: int = 0
    probe_every: int = 200
    legacy_relaxation: bool = True
    name: str = ""

    def validate(self) -> None:
        if self.viscosity <= 0:
            raise ValueError("viscosity must be positive")
        if self.projection not in ("masked", "legacy"):
            raise ValueError("projection must be 'masked' or 'legacy'")
        if self.dt is not None and self.dt <= 0:
            raise ValueError("dt must be positive")
        if self.cfl <= 0 or self.diffusion_limit <= 0:
            raise ValueError("cfl and diffusion_limit must be positive")


# ----------------------------------------------------------------------- solver
@dataclass
class Ledger:
    steps: int = 0
    time: float = 0.0
    div_max_fluid: float = 0.0
    div_max_solid: float = 0.0
    solid_flux_abs_max: float = 0.0
    solid_flux_last: float = 0.0
    drift: float = 0.0
    drift_relative: float = 0.0
    throughput: float = 0.0
    mass_imbalance_max: float = 0.0
    solid_speed_max: float = 0.0
    history: list[dict[str, float]] = field(default_factory=list)


@dataclass
class ImmersedFlow:
    config: ImmersedConfig
    u: torch.Tensor = field(init=False)
    v: torch.Tensor = field(init=False)
    p: torch.Tensor = field(init=False)
    ledger: Ledger = field(init=False, default_factory=Ledger)
    wall_shear_history: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        cfg = self.config
        cfg.validate()
        grid = cfg.grid
        self.u = torch.zeros((grid.ny, grid.nx + 1), dtype=torch.float64)
        self.v = torch.zeros((grid.ny + 1, grid.nx), dtype=torch.float64)
        self.p = torch.zeros((grid.ny, grid.nx), dtype=torch.float64)
        self.u_closed, self.v_closed = face_closures(grid)
        self.s_u, self.s_v = solid_face_indicator(grid)
        bounds = cfg.inlet_bounds
        if bounds is None:
            raise ValueError("inlet_bounds must be given (open part of the inlet plane)")
        self.inlet_profile = parabolic_inlet(grid, cfg.inlet_mean_velocity,
                                             lower=bounds[0], upper=bounds[1])
        self._apply_constraints()
        # Solve mask: the masked arm excludes solid cells, the legacy arm does not.
        if cfg.projection == "masked":
            self._mask = grid.fluid
        else:
            self._mask = torch.ones_like(grid.solid)
        self._poisson = MaskedPoisson(self._mask, outlet_dirichlet=True)
        self._phi_prev = torch.zeros((grid.ny, grid.nx), dtype=torch.float64)
        m = self._mask
        self._u_open = torch.zeros_like(self.u, dtype=torch.bool)
        self._u_open[:, 1:-1] = m[:, :-1] & m[:, 1:]
        self._u_open[:, -1] = m[:, -1]
        self._v_open = torch.zeros_like(self.v, dtype=torch.bool)
        self._v_open[1:-1, :] = m[:-1, :] & m[1:, :]
        self._nu = cfg.viscosity
        self._h = 0.5 * (grid.dx + grid.dy)

    # ------------------------------------------------------------- constraints
    def _apply_constraints(self) -> None:
        """Hard kinematics: inlet profile, closed faces, wall planes."""
        cfg = self.config
        self.u[:, 0] = self.inlet_profile
        if cfg.projection == "masked":
            self.u[:, 1:-1] = torch.where(self.u_closed[:, 1:-1],
                                          torch.zeros_like(self.u[:, 1:-1]),
                                          self.u[:, 1:-1])
            self.v[1:-1, :] = torch.where(self.v_closed[1:-1, :],
                                          torch.zeros_like(self.v[1:-1, :]),
                                          self.v[1:-1, :])
        self.v[0, :] = 0.0
        self.v[-1, :] = 0.0

    def _enforce_outlet(self) -> None:
        self.u[:, -1] = self.u[:, -2]

    # ------------------------------------------------------------------- timing
    def timestep(self) -> float:
        cfg = self.config
        if cfg.dt is not None:
            return cfg.dt
        h = self._h
        speed = max(float(self.u.abs().max()), float(self.v.abs().max()),
                    cfg.inlet_mean_velocity, 1e-12)
        dt_conv = cfg.cfl * h / speed
        dt_diff = cfg.diffusion_limit * h * h / cfg.viscosity
        return float(min(dt_conv, dt_diff))

    # ----------------------------------------------------------------- momentum
    def _momentum_rhs(self, dt: float) -> tuple[torch.Tensor, torch.Tensor]:
        u, v, p = self.u, self.v, self.p
        hx, hy = self.config.grid.dx, self.config.grid.dy
        nu = self._nu

        # ---- u momentum: face (j, i), i = 1 .. nx-1
        u_e = 0.5 * (u + _east(u))
        u_w = 0.5 * (_west(u) + u)
        u_e_up = torch.where(u_e > 0.0, u, _east(u))
        u_w_up = torch.where(u_w > 0.0, _west(u), u)
        v_n = _avgx_to_u(v[1:, :])
        v_s = _avgx_to_u(v[:-1, :])
        u_n_up = torch.where(v_n > 0.0, u, _north(u))
        u_s_up = torch.where(v_s > 0.0, _south(u), u)
        conv_u = ((u_e * u_e_up - u_w * u_w_up) / hx
                  + (v_n * u_n_up - v_s * u_s_up) / hy)
        # diffusion: wall planes are half a cell from the first face -> u_ghost = -u
        u_n_ghost = _north(u).clone()
        u_n_ghost[-1, :] = -u[-1, :]
        u_s_ghost = _south(u).clone()
        u_s_ghost[0, :] = -u[0, :]
        lap_u = ((_east(u) - 2.0 * u + _west(u)) / (hx * hx)
                 + (u_n_ghost - 2.0 * u + u_s_ghost) / (hy * hy))
        p_pad_x = torch.cat([p[:, :1], p, p[:, -1:]], dim=1)
        grad_p_u = (p_pad_x[:, 1:] - p_pad_x[:, :-1]) / hx
        rhs_u = -conv_u + nu * lap_u - grad_p_u / self.config.density

        # ---- v momentum: face (j, i), j = 1 .. ny-1
        v_n_avg = 0.5 * (v + _north(v))
        v_s_avg = 0.5 * (_south(v) + v)
        v_n_up = torch.where(v_n_avg > 0.0, v, _north(v))
        v_s_up = torch.where(v_s_avg > 0.0, _south(v), v)
        u_e_v = _avgy_to_v(u[:, 1:])
        u_w_v = _avgy_to_v(u[:, :-1])
        v_e_up = torch.where(u_e_v > 0.0, v, _east(v))
        v_w_up = torch.where(u_w_v > 0.0, _west(v), v)
        conv_v = ((u_e_v * v_e_up - u_w_v * v_w_up) / hx
                  + (v_n_avg * v_n_up - v_s_avg * v_s_up) / hy)
        v_e_ghost = _east(v).clone()
        v_e_ghost[:, -1] = -v[:, -1]
        v_w_ghost = _west(v).clone()
        v_w_ghost[:, 0] = -v[:, 0]
        lap_v = ((v_e_ghost - 2.0 * v + v_w_ghost) / (hx * hx)
                 + (_north(v) - 2.0 * v + _south(v)) / (hy * hy))
        p_pad = torch.cat([p[:1, :], p, p[-1:, :]], dim=0)
        grad_p_v = (p_pad[1:, :] - p_pad[:-1, :]) / hy
        rhs_v = -conv_v + nu * lap_v - grad_p_v / self.config.density
        return rhs_u, rhs_v

    # --------------------------------------------------------------- projection
    def _project(self, dt: float) -> None:
        cfg = self.config
        grid = cfg.grid
        hx, hy = grid.dx, grid.dy
        u, v = self.u, self.v
        du = u[:, 1:] - u[:, :-1]
        dv = v[1:, :] - v[:-1, :]
        b = -(hx * du + hy * dv) / dt
        phi = self._poisson.solve(b, tol=cfg.poisson_tol, maxiter=cfg.poisson_maxiter,
                                  cycles=cfg.mg_cycles, x0=self._phi_prev)
        self._phi_prev = phi
        zero_col = torch.zeros((phi.shape[0], 1), dtype=phi.dtype)
        phi_east = torch.cat([phi, zero_col], dim=1)      # phi = 0 outside the outlet
        phi_west = torch.cat([zero_col, phi], dim=1)      # switch to Neumann form below
        corr_u = phi_east - phi_west
        corr_u = torch.where(self._u_open, corr_u, torch.zeros_like(corr_u))
        corr_u[:, 0] = 0.0                   # inlet: prescribed, Neumann for phi
        self.u = u - dt * corr_u / hx
        corr_v = torch.zeros_like(v)
        corr_v[1:-1, :] = phi[1:, :] - phi[:-1, :]
        corr_v = torch.where(self._v_open, corr_v, torch.zeros_like(corr_v))
        self.v = v - dt * corr_v / hy
        self.p = self.p + self.config.density * phi

    def _penalise(self, dt: float) -> None:
        """Implicit Brinkman sink (both arms) -- no-op on closed faces."""
        kdt = self.config.kappa * dt
        if kdt <= 0.0:
            return
        self.u = self.u / (1.0 + kdt * self.s_u)
        self.v = self.v / (1.0 + kdt * self.s_v)

    def _legacy_relax(self, dt: float) -> None:
        kdt = self.config.kappa * dt
        if kdt <= 0.0:
            return
        self.u = self.u / (1.0 + kdt * self.s_u)
        self.v = self.v / (1.0 + kdt * self.s_v)

    # --------------------------------------------------------------------- step
    def step(self) -> float:
        cfg = self.config
        dt = self.timestep()
        self._enforce_outlet()
        rhs_u, rhs_v = self._momentum_rhs(dt)
        self.u = self.u + dt * rhs_u
        self.v = self.v + dt * rhs_v
        self._penalise(dt)
        self._apply_constraints()
        self._project(dt)
        if cfg.projection == "legacy" and cfg.legacy_relaxation:
            self._legacy_relax(dt)
        self._apply_constraints()
        self._update_ledger(dt)
        return dt

    # ------------------------------------------------------------------- ledger
    def divergence(self) -> torch.Tensor:
        grid = self.config.grid
        return ((self.u[:, 1:] - self.u[:, :-1]) / grid.dx
                + (self.v[1:, :] - self.v[:-1, :]) / grid.dy)

    def _update_ledger(self, dt: float) -> None:
        cfg = self.config
        grid = cfg.grid
        led = self.ledger
        div = self.divergence()
        scale = cfg.inlet_mean_velocity * cfg.reference_length
        cell_vol = grid.dx * grid.dy
        fluid = grid.fluid
        solid = grid.solid
        div_fluid = float((div.abs() * fluid).max()) if bool(fluid.any()) else 0.0
        div_solid = float((div.abs() * solid).max()) if bool(solid.any()) else 0.0
        led.div_max_fluid = max(led.div_max_fluid, cell_vol * div_fluid / scale)
        led.div_max_solid = max(led.div_max_solid, cell_vol * div_solid / scale)
        net, gross = solid_boundary_flux(grid, self.u, self.v)
        led.solid_flux_last = net
        led.solid_flux_abs_max = max(led.solid_flux_abs_max, abs(net))
        v_ref = cfg.reference_length * cfg.reference_length
        led.drift += dt * net / (cfg.inlet_mean_velocity * v_ref)
        q_in = float(self.u[:, 0].sum() * grid.dy)
        q_out = float(self.u[:, -1].sum() * grid.dy)
        led.throughput += dt * abs(q_in) / v_ref
        led.drift_relative = (abs(led.drift) / led.throughput) if led.throughput > 0 else 0.0
        if abs(q_in) > 0:
            led.mass_imbalance_max = max(led.mass_imbalance_max,
                                         abs(q_out - q_in) / abs(q_in))
        if bool(solid.any()):
            speed = 0.5 * (self.u[:, :-1].abs() + self.u[:, 1:].abs())
            sv = 0.5 * (self.v[:-1, :].abs() + self.v[1:, :].abs())
            speed = torch.maximum(speed, sv)
            led.solid_speed_max = max(led.solid_speed_max,
                                      float((speed * solid).max())
                                      / cfg.inlet_mean_velocity)
        led.steps += 1
        led.time += dt

    # ------------------------------------------------------------------- probes
    def wall_shear(self, wall: Literal["bottom", "top"] = "bottom") -> torch.Tensor:
        """Wall shear stress on ``u`` faces adjacent to the wall (per unit rho)."""
        h = self.config.grid.dy
        row = 0 if wall == "bottom" else self.config.grid.ny - 1
        return 2.0 * self._nu * self.u[row, :] / h

    def snapshot(self) -> dict[str, object]:
        fluid = self.config.grid.fluid
        uc = 0.5 * (self.u[:, :-1] + self.u[:, 1:])
        vc = 0.5 * (self.v[:-1, :] + self.v[1:, :])
        speed = torch.sqrt(uc * uc + vc * vc)
        return {
            "u_max": float((uc * fluid).max()),
            "speed_max": float((speed * fluid).max()),
            "p_min": float((self.p * fluid).min()),
            "p_max": float((self.p * fluid).max()),
        }

    # --------------------------------------------------------------------- run
    def run(self, t_end: float | None = None, max_steps: int | None = None,
            probe: bool = True) -> Ledger:
        cfg = self.config
        t_end = cfg.t_end if t_end is None else t_end
        max_steps = cfg.max_steps if max_steps is None else max_steps
        t0 = time.perf_counter()
        next_probe = 0.0
        worst_step = 0.0
        while self.ledger.time < t_end - 1e-12 and self.ledger.steps < max_steps:
            t_step = time.perf_counter()
            self.step()
            worst_step = max(worst_step, time.perf_counter() - t_step)
            if probe and self.ledger.time >= next_probe:
                self.ledger.history.append(self._history_point())
                next_probe = self.ledger.time + max(t_end / 40.0, 1e-9)
        self.ledger.history.append(self._history_point())
        self.wall_time = time.perf_counter() - t0
        self.worst_step_time = worst_step
        return self.ledger

    def _history_point(self) -> dict[str, float]:
        led = self.ledger
        return {
            "time": led.time,
            "steps": float(led.steps),
            "drift": led.drift,
            "drift_relative": led.drift_relative,
            "div_max_fluid": led.div_max_fluid,
            "div_max_solid": led.div_max_solid,
            "solid_flux": led.solid_flux_last,
            "mass_imbalance": led.mass_imbalance_max,
            "solid_speed_max": led.solid_speed_max,
        }

    def step_time(self) -> float:
        w = getattr(self, "wall_time", 0.0)
        return w / max(self.ledger.steps, 1)


def field_l2(flow_a: ImmersedFlow, flow_b: ImmersedFlow) -> float:
    """Relative L2 difference of the cell-centred velocity fields."""
    def cells(flow: ImmersedFlow) -> torch.Tensor:
        uc = 0.5 * (flow.u[:, :-1] + flow.u[:, 1:])
        vc = 0.5 * (flow.v[:-1, :] + flow.v[1:, :])
        return torch.stack([uc, vc])
    diff = cells(flow_a) - cells(flow_b)
    ref = cells(flow_a)
    num = float((diff * diff).sum())
    den = float((ref * ref).sum())
    if den <= 0.0:
        return math.inf
    return math.sqrt(num / den)