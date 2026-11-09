"""Case definitions for the internal-flow momentum-ledger study.

Three families, all on the staggered MAC grid of :mod:`tensorfvm.inc_bench`:

``expansion``  planar sudden expansion, area ratio 2:1 and 3:1, ``Re_D = 100``
``step``       planar backward-facing step, expansion ratio 2:1, ``Re = 100/400``
``cavity``     differentially heated square cavity, ``Ra = 1e4/1e5/1e6``

Reynolds conventions are stated once and used everywhere:

* expansion: ``Re_D = U_mean * 2 h_in / nu`` with ``U_mean`` the inlet mean
  velocity and ``h_in`` the inlet height (the classic sudden-expansion basis);
* step: ``Re = U_mean * 2 h_in / nu`` (Armaly et al. 1983 use the same
  ``D_h = 2 h_in`` basis; ``h_in`` is the inlet channel height, the step height
  is ``S = h_in`` so the expansion ratio is exactly 2:1);
* cavity: free-fall scaling ``nu = sqrt(Pr/Ra)``, ``alpha = nu/Pr``,
  ``g beta dT = 1``, ``Pr = 0.71``.

Outlet treatments (three tiers, used for the outlet-sensitivity criterion):

``dirichlet``  zero-gradient velocity + ``p = 0`` gauge on the exit face (the
               baseline of :class:`~tensorfvm.inc_bench.solver.SteadyFlow`);
``extended``   identical boundary, but the computational domain continues six
               outlet heights downstream so that the *physical* exit station is
               an interior plane;
``traction``   "do nothing" / traction-free outflow ``p = 2 mu du/dx`` on the
               exit face instead of a pressure Dirichlet gauge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from ..inc_bench.solver import SolverOptions, SteadyFlow

OUTLET_MODES = ("dirichlet", "extended", "traction")

EXPANSION_LEVELS = {"coarse": 8, "medium": 16, "fine": 32}     # cells per inlet height
STEP_LEVELS = {"coarse": 12, "medium": 24, "fine": 48}         # cells per step height
CAVITY_LEVELS = {"coarse": 32, "medium": 64, "fine": 128}      # cells per side

EXTENSION = 6.0  # outlet heights of extra domain for the ``extended`` mode


class TractionOutletFlow(SteadyFlow):
    """SteadyFlow with a traction-free ("do nothing") outflow condition.

    ``p = 2 mu du/dx`` on the exit cell replaces the pressure Dirichlet gauge; the
    normal stress of the outflow then vanishes instead of the pressure itself.
    Only the exit column of ``p`` is touched, so the interior discretisation is
    bit-identical to the base class.
    """

    def step(self, ramp: float = 0.0) -> None:  # type: ignore[override]
        super().step(ramp)
        if bool(self.u_outlet.any()):
            self.p = self.p.clone()
            self.p[:, -1] = 2.0 * self.nu * (self.u[:, -1] - self.u[:, -2]) / self.dx


@dataclass
class InternalCase:
    """A fully described case: geometry mask, fluid properties, stations."""

    family: str
    name: str
    level: str
    fluid: torch.Tensor
    dx: float
    dy: float
    viscosity: float
    outlet_mode: str = "dirichlet"
    inlet_profile: torch.Tensor | None = None
    outlet: bool = True
    conductivity: float | None = None
    buoyancy: float = 0.0
    t_ref: float = 0.0
    t_hot: float | None = None
    t_cold: float | None = None
    params: dict = field(default_factory=dict)
    geometry: dict = field(default_factory=dict)
    stations: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.fluid.shape[0]), int(self.fluid.shape[1])

    @property
    def cells(self) -> int:
        return int(self.fluid.sum())

    def solver(self, options: SolverOptions | None = None) -> SteadyFlow:
        cls = TractionOutletFlow if self.outlet_mode == "traction" else SteadyFlow
        return cls(self.fluid, dx=self.dx, dy=self.dy, viscosity=self.viscosity,
                   inlet_profile=self.inlet_profile, outlet=self.outlet,
                   conductivity=self.conductivity, buoyancy=self.buoyancy,
                   t_ref=self.t_ref, t_hot=self.t_hot, t_cold=self.t_cold,
                   options=options)

    def summary(self) -> dict[str, object]:
        ny, nx = self.shape
        return {"family": self.family, "case": self.name, "level": self.level,
                "outlet_mode": self.outlet_mode, "ny": ny, "nx": nx,
                "fluid_cells": self.cells, "dx": self.dx, "dy": self.dy,
                "viscosity": self.viscosity,
                "params": dict(self.params), "geometry": dict(self.geometry),
                "stations": dict(self.stations)}


def solver_options(family: str, *, tolerance: float = 1e-6,
                   max_iterations: int = 60000) -> SolverOptions:
    """Relaxation/iteration settings tuned per family (documented in the report)."""
    if family == "cavity":
        return SolverOptions(alpha_u=0.5, alpha_p=0.3, alpha_t=0.7,
                             momentum_sweeps=3, energy_sweeps=4,
                             pressure_sweeps=20, tolerance=tolerance,
                             max_iterations=max_iterations, check_every=50,
                             warmup_iterations=400, ramp_iterations=300,
                             alpha_u_warmup=0.3, alpha_p_warmup=0.1,
                             min_iterations=200)
    return SolverOptions(alpha_u=0.7, alpha_p=0.3, pressure_sweeps=16,
                         tolerance=tolerance, max_iterations=max_iterations,
                         check_every=50, warmup_iterations=300,
                         ramp_iterations=200, alpha_u_warmup=0.4,
                         alpha_p_warmup=0.15, min_iterations=150)


# ------------------------------------------------------------------ expansion
def _inlet_mask(ny: int, dy: float, h_out: float, h_in: float) -> torch.Tensor:
    y = (torch.arange(ny) + 0.5) * dy
    y0 = 0.5 * (h_out - h_in)
    return (y >= y0 - 1e-12) & (y <= y0 + h_in + 1e-12)


def sudden_expansion(area_ratio: float = 2.0, level: str = "coarse",
                     outlet_mode: str = "dirichlet", *, reynolds_d: float = 100.0,
                     upstream_inlet_heights: float = 3.0,
                     downstream_outlet_heights: float = 10.0) -> InternalCase:
    """Planar sudden expansion with a parabolic (fully developed) inlet profile."""
    if area_ratio not in (2.0, 3.0):
        raise ValueError("area_ratio must be 2 or 3 for this study")
    if level not in EXPANSION_LEVELS:
        raise ValueError(f"level must be one of {sorted(EXPANSION_LEVELS)}")
    if outlet_mode not in OUTLET_MODES:
        raise ValueError(f"outlet_mode must be one of {OUTLET_MODES}")
    n_h = EXPANSION_LEVELS[level]
    h_out = 1.0
    h_in = h_out / area_ratio
    dy = h_in / n_h
    dx = dy
    ny = int(round(h_out / dy))
    x_up = upstream_inlet_heights * h_in
    x_exit = downstream_outlet_heights * h_out
    x_end = x_exit + (EXTENSION if outlet_mode == "extended" else 0.0)
    nx = int(round((x_up + x_end) / dx))
    x = (torch.arange(nx) + 0.5) * dx - x_up
    inlet = _inlet_mask(ny, dy, h_out, h_in)
    fluid = (x[None, :] >= 0.0) | inlet[:, None]
    rows = inlet
    y = (torch.arange(ny) + 0.5) * dy
    y0 = 0.5 * (h_out - h_in)
    eta = (y - y0) / h_in
    profile = torch.where(rows, 6.0 * eta * (1.0 - eta), torch.zeros(ny))
    # mass flux of the parabolic profile is exactly 1 * h_in
    nu = 2.0 * h_in / reynolds_d
    i_in = 0
    i_step = int(round(x_up / dx))
    i_exit = int(round((x_up + x_exit) / dx))
    i_mid = int(round((x_up + 0.5 * x_exit) / dx))
    recovery_length = 4.0 * h_out          # past reattachment (x_r ~ 2.3 h_out)
    i_recovery = int(round((x_up + recovery_length) / dx))
    geometry = {
        "area_ratio": area_ratio, "h_out": h_out, "h_in": h_in,
        "inlet_cells": int(rows.sum()), "expansion_ratio_mask": h_out / (inlet.sum() * dy),
        "x_step": 0.0, "x_exit": x_exit, "x_end": x_end,
        "x_recovery": recovery_length,
        "upstream_length": x_up, "outlet_extension": x_end - x_exit,
        "reynolds_basis": "U_mean * 2 h_in / nu",
        "inlet_flux": float(profile.sum() * dy),
    }
    return InternalCase(
        family="expansion", name=f"expansion_ar{area_ratio:g}_re{reynolds_d:g}",
        level=level, fluid=fluid, dx=dx, dy=dy, viscosity=nu,
        outlet_mode=outlet_mode, inlet_profile=profile, outlet=True,
        params={"area_ratio": area_ratio, "reynolds_d": reynolds_d,
                "outlet_mode": outlet_mode, "inlet_mean_velocity": 1.0},
        geometry=geometry,
        stations={"inlet_face": i_in, "step_face": i_step, "mid_face": i_mid,
                  "recovery_face": i_recovery, "exit_face": i_exit, "end_face": nx},
    )


# ----------------------------------------------------------------------- step
def backward_facing_step(reynolds: float = 100.0, level: str = "coarse",
                         outlet_mode: str = "dirichlet", *,
                         step_height: float = 1.0,
                         upstream_inlet_heights: float = 3.0,
                         downstream_step_heights: float = 14.0) -> InternalCase:
    """Planar backward-facing step, expansion ratio 2:1, parabolic inlet."""
    if level not in STEP_LEVELS:
        raise ValueError(f"level must be one of {sorted(STEP_LEVELS)}")
    if outlet_mode not in OUTLET_MODES:
        raise ValueError(f"outlet_mode must be one of {OUTLET_MODES}")
    n_s = STEP_LEVELS[level]
    dy = step_height / n_s
    dx = dy
    h_in = step_height                      # ER = 2
    h_out = h_in + step_height
    ny = int(round(h_out / dy))
    x_up = upstream_inlet_heights * h_in
    x_exit = downstream_step_heights * step_height
    x_end = x_exit + (EXTENSION * h_out if outlet_mode == "extended" else 0.0)
    nx = int(round((x_up + x_end) / dx))
    x = (torch.arange(nx) + 0.5) * dx - x_up
    y = (torch.arange(ny) + 0.5) * dy
    fluid = (x[None, :] >= 0.0) | (y[:, None] >= h_in)
    rows = y >= h_in
    eta = (y - h_in) / h_in
    profile = torch.where(rows, 6.0 * eta * (1.0 - eta), torch.zeros(ny))
    nu = 2.0 * h_in / reynolds
    i_step = int(round(x_up / dx))
    i_exit = int(round((x_up + x_exit) / dx))
    geometry = {
        "expansion_ratio": h_out / h_in, "step_height": step_height,
        "h_in": h_in, "h_out": h_out, "inlet_cells": int(rows.sum()),
        "x_step": 0.0, "x_exit": x_exit, "x_end": x_end,
        "upstream_length": x_up, "outlet_extension": x_end - x_exit,
        "reynolds_basis": "U_mean * 2 h_in / nu",
        "inlet_flux": float(profile.sum() * dy),
    }
    return InternalCase(
        family="step", name=f"bfs_re{reynolds:g}", level=level, fluid=fluid,
        dx=dx, dy=dy, viscosity=nu, outlet_mode=outlet_mode,
        inlet_profile=profile, outlet=True,
        params={"reynolds": reynolds, "outlet_mode": outlet_mode,
                "inlet_mean_velocity": 1.0},
        geometry=geometry,
        stations={"inlet_face": 0, "step_face": i_step, "exit_face": i_exit,
                  "end_face": nx},
    )


# --------------------------------------------------------------------- cavity
def natural_cavity(rayleigh: float = 1.0e4, level: str = "coarse", *,
                   prandtl: float = 0.71) -> InternalCase:
    """Differentially heated square cavity in free-fall scaling."""
    if level not in CAVITY_LEVELS:
        raise ValueError(f"level must be one of {sorted(CAVITY_LEVELS)}")
    n = CAVITY_LEVELS[level]
    dx = dy = 1.0 / n
    fluid = torch.ones((n, n), dtype=torch.bool)
    nu = math.sqrt(prandtl / rayleigh)
    geometry = {"cells_per_side": n, "aspect_ratio": 1.0, "rayleigh": rayleigh,
                "prandtl": prandtl, "viscosity": nu, "diffusivity": nu / prandtl,
                "buoyancy_scale": 1.0, "hot_wall": "west", "cold_wall": "east"}
    return InternalCase(
        family="cavity", name=f"cavity_ra{rayleigh:g}", level=level, fluid=fluid,
        dx=dx, dy=dy, viscosity=nu, outlet=False, outlet_mode="closed",
        conductivity=nu / prandtl, buoyancy=1.0, t_ref=0.5, t_hot=1.0, t_cold=0.0,
        params={"rayleigh": rayleigh, "prandtl": prandtl},
        geometry=geometry,
        stations={"mid_column": n // 2, "mid_row": n // 2,
                  "left_half": (0, n, 0, n // 2), "right_half": (0, n, n // 2, n)})


def build(family: str, level: str, **kwargs) -> InternalCase:
    if family == "expansion":
        return sudden_expansion(kwargs.pop("area_ratio", 2.0), level, **kwargs)
    if family == "step":
        return backward_facing_step(kwargs.pop("reynolds", 100.0), level, **kwargs)
    if family == "cavity":
        kwargs.pop("outlet_mode", None)
        return natural_cavity(kwargs.pop("rayleigh", 1.0e4), level, **kwargs)
    raise ValueError(f"unknown family {family!r}")


def warm_start(flow: SteadyFlow, *, u=None, v=None, p=None, T=None) -> None:
    """Seed a solver with a coarser/simpler solution (grid or Ra continuation)."""
    if u is not None:
        flow.u = u.to(flow.u.dtype).clone()
    if v is not None:
        flow.v = v.to(flow.v.dtype).clone()
    if p is not None:
        flow.p = p.to(flow.p.dtype).clone()
    if T is not None:
        flow.T = T.to(flow.T.dtype).clone()
    flow._refresh_known()