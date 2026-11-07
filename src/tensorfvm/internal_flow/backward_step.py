"""Backward-facing step at ``Re_H = 800`` on a uniform MAC grid.

Geometry (the classical 2-D test case of Gartling 1990 and of Armaly *et al.*
1983, reduced to its standard laminar form)
--------------------------------------------------------------
* step height ``h_s = H``, inlet channel height ``H`` (so the expansion ratio is
  ``H_out / H = 2`` with ``H_out = 2 H`` the downstream channel height),
* the inlet plane ``x = 0`` carries a fully developed parabolic profile over
  ``y in [H, 2H]`` (mean velocity ``U``); the rest of the plane, ``y in [0, H]``,
  is the solid step face,
* no-slip walls at ``y = 0`` (the long wall on which the bubble sits) and
  ``y = 2H`` (the upper wall),
* zero-gradient velocity outlet at ``x = L``, preceded by a long enough buffer
  that the reattachment point is unaffected.

Lengths below are quoted in units of the step height ``H`` (so the reattachment
length is also ``L_r / H``); the same number expressed in downstream channel
heights (``H_out = 2H``) is what the literature table uses, and both are
reported.

The Reynolds number is ``Re_H = U H / nu`` with ``U`` the mean inlet velocity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .mac import MACConfig, StaggeredMAC
from .reference import BACKWARD_STEP_REFERENCE, channel_wall_friction


@dataclass(frozen=True)
class BackwardStepConfig:
    """Mesh and solver controls of the backward-facing step case."""

    cells_per_step_height: int = 40
    reynolds: float = 800.0
    length_in_step_heights: float = 22.0
    cfl: float = 0.4
    time_step: float | None = None
    max_steps: int = 400_000
    tolerance: float = 1e-9
    check_every: int = 200
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.cells_per_step_height < 4:
            raise ValueError("cells_per_step_height must be >= 4")
        for name in ("reynolds", "length_in_step_heights", "cfl"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")

    @property
    def step_height(self) -> float:
        return 1.0

    @property
    def viscosity(self) -> float:
        return 1.0 / self.reynolds

    @property
    def mesh(self) -> tuple[int, int]:
        n = self.cells_per_step_height
        return int(round(self.length_in_step_heights * n)), 2 * n


def inlet_profile(cells_per_step_height: int) -> tuple[float, ...]:
    """Fully developed parabolic inlet over ``y in [H, 2H]``, zero elsewhere."""
    n = cells_per_step_height
    ny = 2 * n
    h = 1.0 / n
    values = []
    for j in range(ny):
        y = (j + 0.5) * h
        if y <= 1.0:
            values.append(0.0)
        else:
            eta = (y - 1.0) / 1.0              # 0 at the step lip, 1 at the upper wall
            values.append(1.5 * (1.0 - (2.0 * eta - 1.0) ** 2))
    return tuple(values)


class BackwardStepCase:
    """Backward-facing step solver built on :class:`StaggeredMAC`."""

    def __init__(self, config: BackwardStepConfig | None = None) -> None:
        self.config = config or BackwardStepConfig()
        nx, ny = self.config.mesh
        self.solver = StaggeredMAC(MACConfig(
            nx=nx, ny=ny,
            length=self.config.length_in_step_heights * self.config.step_height,
            height=2.0 * self.config.step_height,
            viscosity=self.config.viscosity,
            reference_velocity=1.0,
            cfl=self.config.cfl,
            time_step=self.config.time_step,
            max_steps=self.config.max_steps,
            tolerance=self.config.tolerance,
            check_every=self.config.check_every,
            left="inlet",
            right="outlet",
            inlet_profile=inlet_profile(self.config.cells_per_step_height),
            device=self.config.device,
        ))

    # ------------------------------------------------------------------ driving
    def run(self, **kwargs) -> dict[str, object]:
        return self.solver.run(**kwargs)

    # -------------------------------------------------------------- diagnostics
    def cf_profile(self) -> tuple[torch.Tensor, torch.Tensor]:
        """``(x, C_f)`` along the bottom wall, ``x`` in step heights."""
        x = torch.arange(self.solver.nx + 1, dtype=self.solver.dtype) * self.solver.hx
        return x, self.solver.bottom_wall_friction()

    def reattachment(self) -> float | None:
        """Reattachment distance from the step, in step heights."""
        return self.solver.reattachment_length()

    def diagnostics(self) -> dict[str, object]:
        """Scalar acceptance quantities plus the arrays used by the report plots."""
        cfg = self.config
        n = cfg.cells_per_step_height
        solver = self.solver
        x, cf = self.cf_profile()
        xr = self.reattachment()
        # downstream channel: height 2H, mean velocity U/2, reference velocity U
        developed = channel_wall_friction(2.0 * cfg.step_height,
                                         0.5 * self.solver.config.reference_velocity,
                                         cfg.viscosity,
                                         self.solver.config.reference_velocity)
        stations = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 14.0, 18.0]
        station_cf = {}
        for station in stations:
            i = int(round(station * n))
            if 0 <= i <= solver.nx:
                station_cf[f"x/H={station:g}"] = float(cf[i])
        bubble = cf[1:solver.nx]
        min_index = int(torch.argmin(bubble)) + 1
        h_out = 2.0 * cfg.step_height
        return {
            "mesh": {"nx": solver.nx, "ny": solver.ny,
                     "cells_per_step_height": n,
                     "h_over_H": float(solver.hx)},
            "reynolds": cfg.reynolds,
            "viscosity": cfg.viscosity,
            "steps": solver.steps,
            "dt": solver.dt,
            "residual": solver.residual,
            "max_divergence": float(solver.divergence().abs().max()),
            "reattachment_in_step_heights": xr,
            "reattachment_in_downstream_heights": None if xr is None else xr / h_out,
            "reference_reattachment": BACKWARD_STEP_REFERENCE,
            "developed_cf_analytic": developed,
            "cf_at_outlet": float(cf[-1]),
            "cf_stations": station_cf,
            "min_cf": float(cf[min_index]),
            "min_cf_x_over_H": float(x[min_index]),
            "x": x.tolist(),
            "cf": cf.tolist(),
        }