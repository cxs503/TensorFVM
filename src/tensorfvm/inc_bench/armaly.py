"""Reproduction of the Armaly *et al.* (1983) backward-facing-step experiment.

The reproduction target is deliberately narrow and machine checkable:

1. lower-wall reattachment length ``L_r / S`` at ``Re_Dh = 100`` and
   ``Re_Dh = 389`` against the digitised experimental values, tolerance 15 %;
2. the outlet velocity profile reaches the fully developed plane-Poiseuille
   parabola, maximum deviation < 5 % of the peak velocity.

Everything below reuses the existing steady SIMPLE kernel
(:mod:`tensorfvm.inc_bench.solver`) and the existing Armaly case builder
(:func:`tensorfvm.inc_bench.cases.backward_step`); this module only adds the
analysis surfaces the reproduction needs (experimental anchors, downstream
development metric, one-call run record).  No solver physics is re-implemented.

Conventions (identical to the rest of ``inc_bench``):

* ``Re = U_mean D_h / nu`` with ``D_h = 2 H_in`` -- Armaly's definition;
* lengths are quoted in step heights ``S`` (the paper's axis) *and* in
  downstream-channel heights ``H_out`` (the Gartling/Barton axis);
* ``u`` is normalised by the inlet mean velocity ``U_in = 1``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .cases import CaseSetup, backward_step, backward_step_metrics
from .solver import SolverOptions, SteadyFlow, cell_velocity


# --------------------------------------------------------------------- anchors
@dataclass(frozen=True)
class ExperimentPoint:
    """One digitised experimental anchor of Armaly *et al.* (1983)."""

    reynolds_dh: float
    reattachment_over_step: float
    band_over_step: float
    note: str = ""


#: Armaly *et al.* (1983), J. Fluid Mech. **127**, 473-496, Fig. 4 experimental
#: lower-wall reattachment length for the 2-D regime.  ``Re_Dh = 389`` is the
#: last tabulated Reynolds number below the onset of spanwise (three-dimensional)
#: structure at ``Re ~ 400``, which is why a 2-D solver may be compared with it.
#:
#: Provenance grade REAL, with an explicit +/-0.2 S digitisation band: the values
#: are the ones the 2-D benchmark literature quotes for this experiment (Armaly
#: *et al.* 1983; Williams & Baker 1997; Barton 1997; Erturk 2008), transcribed
#: here; they were **not** re-digitised from the published figure in this
#: environment (no offline copy of the paper), and the band records that gap.
ARMALY_1983_EXPERIMENT: dict[float, ExperimentPoint] = {
    100.0: ExperimentPoint(100.0, 3.0, 0.2,
                           "2-D, experimental error bar is inside the band"),
    389.0: ExperimentPoint(389.0, 7.9, 0.2,
                           "last 2-D case below the Re ~ 400 3-D onset"),
}

ARMALY_1983_CITATION = (
    "Armaly, B. F., Durst, F., Pereira, J. C. F. & Schoenung, B. (1983), "
    "Experimental and theoretical investigation of backward-facing step flow, "
    "J. Fluid Mech. 127, 473-496")

TOLERANCE_REATTACHMENT = 0.15
TOLERANCE_PROFILE = 0.05


def experiment_point(reynolds_dh: float) -> ExperimentPoint:
    """Experimental anchor at ``Re_Dh``; never interpolated."""
    for key, point in ARMALY_1983_EXPERIMENT.items():
        if abs(reynolds_dh / key - 1.0) < 1.0e-9:
            return point
    raise ValueError(
        f"no Armaly (1983) experimental anchor at Re_Dh = {reynolds_dh:g}; "
        f"available: {sorted(ARMALY_1983_EXPERIMENT)}")


# ------------------------------------------------------ downstream development
def column_at(setup: CaseSetup, x_target: float) -> int:
    """Cell column whose centre is closest to the streamwise station ``x_target``.

    ``x_target`` is measured from the step plane (``x = 0``) in step heights.
    """
    nx = int(setup.fluid.shape[1])
    x_up = setup.geometry["inlet_heights_upstream"] * setup.params["h_in"]
    centres = (torch.arange(nx, dtype=torch.float64) + 0.5) * setup.dx - x_up
    return int(torch.argmin(torch.abs(centres - x_target)))


def _parabolic(eta: torch.Tensor, bulk: float) -> torch.Tensor:
    return 6.0 * bulk * eta * (1.0 - eta)


def profile_development(solver: SteadyFlow, setup: CaseSetup, column: int,
                        *, rows: int = 0) -> dict[str, float]:
    """Deviation of the streamwise velocity profile from fully developed flow.

    The comparison profile is plane Poiseuille flow with the *same mass flux*
    through the same column, ``u_par(eta) = 6 U_b eta (1 - eta)`` with
    ``eta = y / H_out`` -- so the metric measures shape error only and cannot be
    hidden by a bulk-velocity mismatch.

    ``max_rel`` is the infinity norm divided by the peak of the reference
    parabola (the 5 % gate); ``l2_rel`` is the flux-weighted L2 norm.
    """
    u_cell, _ = cell_velocity(solver.u, solver.v)
    fluid = setup.fluid[:, column]
    if rows:
        fluid = fluid & (torch.arange(solver.ny) < rows)
    heights = float(fluid.sum()) * setup.dy            # wetted height of the column
    bulk = float(u_cell[fluid, column].mean()) if bool(fluid.any()) else 0.0
    y = (torch.arange(solver.ny, dtype=torch.float64) + 0.5) * setup.dy
    eta = y / heights
    reference = _parabolic(eta, bulk)
    u = u_cell[:, column].to(torch.float64)
    diff = (u - reference)[fluid]
    peak = float(reference[fluid].abs().max())
    if peak <= 0.0 or not bool(fluid.any()):
        return {"max_rel": float("nan"), "l2_rel": float("nan"),
                "bulk_velocity": bulk, "peak_velocity": 0.0,
                "height": heights, "x_over_step": float("nan")}
    max_rel = float(diff.abs().max()) / peak
    l2_rel = float(torch.sqrt((diff ** 2).sum() / (reference[fluid] ** 2).sum()))
    x_up = setup.geometry["inlet_heights_upstream"] * setup.params["h_in"]
    x = (column + 0.5) * setup.dx - x_up
    return {"max_rel": max_rel, "l2_rel": l2_rel, "bulk_velocity": bulk,
            "peak_velocity": peak, "height": heights, "x_over_step": x}


# ------------------------------------------------------------------ run record
@dataclass
class ReproductionRun:
    """One (Reynolds, grid, domain) point of the reproduction matrix."""

    reynolds_dh: float
    level: str
    downstream_step_heights: float
    check_stations: tuple[float, ...] = (0.5, 0.75, 1.0)
    options: SolverOptions = field(default_factory=SolverOptions)
    max_iterations: int | None = None

    def run(self) -> dict[str, object]:
        setup = backward_step(self.level, reynolds_dh=self.reynolds_dh,
                              downstream_step_heights=self.downstream_step_heights)
        solver = setup.solver(self.options)
        started = time.time()
        record = solver.solve(max_iterations=self.max_iterations)
        elapsed = time.time() - started
        metrics = backward_step_metrics(solver, setup)
        # back out the physical half-cell column positions of the wall row to
        # normalise the reattachment length: L_r is quoted in step heights.
        l_r_over_step = float(metrics["reattachment_over_step"])
        h_out = setup.params["h_out"]
        point = experiment_point(self.reynolds_dh) \
            if self.reynolds_dh in ARMALY_1983_EXPERIMENT else None
        stations = [profile_development(solver, setup,
                                        column_at(setup, s * self.downstream_step_heights))
                    for s in self.check_stations]
        outlet = profile_development(solver, setup, int(setup.fluid.shape[1]) - 1)
        geometry = dict(setup.geometry)
        return {
            "reynolds_dh": float(self.reynolds_dh),
            "reynolds_h_out": float(self.reynolds_dh / 2.0),
            "level": self.level,
            "nx": int(setup.fluid.shape[1]),
            "ny": int(setup.fluid.shape[0]),
            "cells": int(setup.cells),
            "fluid_cells": int(setup.fluid.sum()),
            "downstream_step_heights": float(self.downstream_step_heights),
            "expansion_ratio_mask": float(geometry["expansion_ratio_mask"]),
            "expansion_ratio_rel_error": float(geometry["expansion_ratio_rel_error"]),
            "converged": bool(record["converged"]),
            "iterations": int(record["iterations"]),
            "change": float(record["change"]),
            "divergence": float(metrics["divergence"]),
            "seconds": round(elapsed, 1),
            "reattachment_over_step": l_r_over_step,
            "reattachment_over_h_out": l_r_over_step / h_out,
            "friction_min": float(metrics["friction_min"]),
            "experiment_over_step": None if point is None else point.reattachment_over_step,
            "experiment_band_over_step": None if point is None else point.band_over_step,
            "reattachment_rel_error": None if point is None else
            abs(l_r_over_step - point.reattachment_over_step) / point.reattachment_over_step,
            "reattachment_pass_15pct": None if point is None else
            abs(l_r_over_step - point.reattachment_over_step)
            / point.reattachment_over_step < TOLERANCE_REATTACHMENT,
            "profile_stations": stations,
            "profile_outlet_column": outlet,
            "profile_outlet_pass_5pct": bool(
                math.isfinite(outlet["max_rel"])
                and outlet["max_rel"] < TOLERANCE_PROFILE),
            "profile_max_rel_stations": [s["max_rel"] for s in stations],
            "profile": _profile_table(solver, setup),
            "field": _field_snapshot(solver, setup),
            "friction_profile": {
                "x_over_step": [v / 1.0 for v in metrics["friction_profile"]["x_over_step"]],
                "cf": metrics["friction_profile"]["cf"],
            },
            "solver_options": {
                "alpha_u": self.options.alpha_u, "alpha_p": self.options.alpha_p,
                "pressure_iterations": self.options.pressure_iterations,
                "pressure_tolerance": self.options.pressure_tolerance,
                "tolerance": self.options.tolerance,
                "high_order": self.options.high_order,
                "momentum_sweeps": self.options.momentum_sweeps,
            },
        }


def _field_snapshot(solver: SteadyFlow, setup: CaseSetup) -> dict[str, object]:
    """Cell-centre velocity/streamfunction field of the converged state.

    Saved so the report figures (contour + streamline plot) are drawn from the
    very field whose numbers are quoted, not from a re-run.
    """
    u_cell, v_cell = cell_velocity(solver.u, solver.v)
    u_cell = u_cell.to(torch.float64)
    v_cell = v_cell.to(torch.float64)
    psi = torch.zeros_like(u_cell)
    psi[:, 1:] = torch.cumsum(u_cell[:, :-1], dim=1) * setup.dy
    x = ((torch.arange(int(setup.fluid.shape[1]), dtype=torch.float64) + 0.5)
         * setup.dx - setup.geometry["inlet_heights_upstream"] * setup.params["h_in"])
    y = (torch.arange(solver.ny, dtype=torch.float64) + 0.5) * setup.dy
    fluid = setup.fluid
    rounded = lambda t: [round(float(v), 6) for v in t[fluid]]
    return {
        "x": [round(float(v), 6) for v in x],
        "y": [round(float(v), 6) for v in y],
        "fluid_index": [int(i) for i in torch.nonzero(fluid.flatten()).flatten()],
        "u": rounded(u_cell),
        "v": rounded(v_cell),
        "psi": rounded(psi),
        "shape": [int(fluid.shape[1]), int(fluid.shape[0])],
    }


def _profile_table(solver: SteadyFlow, setup: CaseSetup) -> dict[str, list[float]]:
    """Outlet-plane velocity profile, for plotting and for the report table."""
    u_cell, _ = cell_velocity(solver.u, solver.v)
    column = int(setup.fluid.shape[1]) - 1
    fluid = setup.fluid[:, column]
    y = ((torch.arange(solver.ny, dtype=torch.float64) + 0.5) * setup.dy).tolist()
    u = u_cell[:, column].to(torch.float64).tolist()
    return {"y": y, "u": u, "fluid": [bool(v) for v in fluid]}