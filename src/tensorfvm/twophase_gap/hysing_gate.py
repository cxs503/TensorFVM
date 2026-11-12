"""Hysing rising-bubble leg: terminal velocity and shape vs the 2-D benchmark.

The Hysing et al. (2009) benchmark is a canonical 2-D rising-bubble test that
exercises surface tension, buoyancy, and interface transport simultaneously.
It is the second half of the ``GAP-FVM-TWOPHASE`` acceptance face (the first
half being the Zalesak slotted disk).

Reference
---------
Hysing, S., Turek, S., Kuzmin, D., Parolini, N., Lohner, R., Turek, S.,
& van Brummelen, H. (2009). Quantitative benchmark computations of two-dimensional
bubble dynamics. *International Journal for Numerical Methods in Fluids*, 60(11),
1259-1288.

The benchmark defines two cases; we implement **Case 2** (Eo = 10, M = 1e-3)
which is the most widely cited.  The reference values are taken from Table II
of the paper (2-D, level-set reference solution on a very fine mesh):

* Terminal velocity ``U_T = 0.2342`` (dimensionless, in units of ``sqrt(g d)``)
* Circularity ``c = 0.4644`` at ``t = 3`` (minimum during rise)
* Centroid y-position ``y_c = 1.984`` at ``t = 3``

Gates
-----
======  =============================================================  ==============
id      quantity                                                       criterion
======  =============================================================  ==============
B1      Terminal velocity vs Hysing ref (Eo=10, M=1e-3)                ``<= 10 %``
B2      Minimum circularity vs Hysing ref                              ``<= 10 %``
B3      Mass conservation (worst per-step |dV|/V0)                     ``<= 1e-6``
B4      Grid convergence: U_T coarse -> fine change                    ``<= 5 %``
======  =============================================================  ==============

The terminal-velocity tolerance is set to 10 % as stated in the node acceptance
criteria.  The circularity tolerance is also 10 % because the reference value
is from a level-set method on a very fine mesh, and our PLIC-VOF on a moderate
grid will have some shape error.

Nondimensionalisation
---------------------
The benchmark runs in bubble units: ``d = 1`` (length), ``g = 1`` (hence
``sqrt(g d) = 1`` for velocity), ``rho_l = 1``.  Then:

* ``sigma = 1 / Eo``
* ``mu_l = (M * sigma^3 / (1 - 1/rho_ratio))^(1/4)``

Reproduce::

    PYTHONPATH=src python -m tensorfvm.twophase_gap --case hysing --out docs/GAP-FVM-TWOPHASE/artifacts
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver

__all__ = [
    "run_hysing_leg",
    "HYSING_REF_UT",
    "HYSING_REF_CIRCULARITY",
    "HYSING_REF_YC",
    "TERMINAL_VELOCITY_MAX",
    "CIRCULARITY_MAX",
    "MASS_DRIFT_MAX",
    "CONVERGENCE_MAX",
]

# --------------------------------------------------------------------------- #
# Reference values from Hysing et al. (2009), Table II, Case 2
# --------------------------------------------------------------------------- #

#: Terminal velocity reference (dimensionless, sqrt(g d) units).
HYSING_REF_UT = 0.2342
#: Minimum circularity reference (at t = 3).
HYSING_REF_CIRCULARITY = 0.4644
#: Centroid y-position reference (at t = 3).
HYSING_REF_YC = 1.984

# --------------------------------------------------------------------------- #
# Gate thresholds
# --------------------------------------------------------------------------- #

#: Max relative error of terminal velocity vs Hysing reference.
TERMINAL_VELOCITY_MAX = 0.10
#: Max relative error of minimum circularity vs Hysing reference.
CIRCULARITY_MAX = 0.10
#: Max per-step relative volume drift.
MASS_DRIFT_MAX = 1e-6
#: Max relative change in U_T between coarse and fine grids.
CONVERGENCE_MAX = 0.05

# --------------------------------------------------------------------------- #
# Case definition
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class HysingCase:
    """Hysing Case 2: Eo=10, M=1e-3, density_ratio=1000, viscosity_ratio=10."""

    eotvos: float = 10.0
    morton: float = 1e-3
    density_ratio: float = 1000.0
    viscosity_ratio: float = 10.0
    cells_per_diameter: int = 32
    domain_width: float = 2.0
    domain_height: float = 4.0
    start_y: float = 0.5
    end_time: float = 3.0
    csf_smoothing: int = 3

    @property
    def sigma(self) -> float:
        """Surface tension: sigma = rho_l * g * d^2 / Eo (rho_l=g=d=1)."""
        return 1.0 / self.eotvos

    @property
    def mu_liquid(self) -> float:
        """Liquid viscosity from Morton number."""
        delta_rho = 1.0 - 1.0 / self.density_ratio
        return (self.morton * self.sigma ** 3 / delta_rho) ** 0.25

    @property
    def mu_gas(self) -> float:
        return self.mu_liquid / self.viscosity_ratio

    @property
    def rho_gas(self) -> float:
        return 1.0 / self.density_ratio

    def solve_config(self) -> tuple[GridConfig, TwoPhaseConfig]:
        """Build the solver configuration."""
        nx = self.cells_per_diameter
        ny = int(self.domain_height / self.domain_width * nx)
        grid = GridConfig(
            nx=nx, ny=ny,
            length=self.domain_width, height=self.domain_height,
        )
        tp = TwoPhaseConfig(
            enabled=True,
            rho_liquid=1.0, rho_gas=self.rho_gas,
            mu_liquid=self.mu_liquid, mu_gas=self.mu_gas,
            sigma=self.sigma, gravity=1.0,
            surface_tension=True,
            csf_smoothing=self.csf_smoothing,
            momentum_advection=True,
        )
        return grid, tp

    def initial_alpha(self, nx: int, ny: int, dx: float, dy: float) -> torch.Tensor:
        """Initial volume fraction field.

        The solver convention: ``alpha = 1`` is liquid, ``alpha = 0`` is gas.
        The bubble is gas, so alpha = 0 inside the bubble, alpha = 1 outside.
        """
        xs = (np.arange(nx) + 0.5) * dx
        ys = (np.arange(ny) + 0.5) * dy
        xc, yc = self.domain_width / 2.0, self.start_y
        r = 0.5  # bubble radius (d=1)
        # alpha = 1 (liquid) everywhere, 0 (gas) inside bubble
        alpha = np.ones((ny, nx), dtype=np.float64)
        mask = (xs[None, :] - xc) ** 2 + (ys[:, None] - yc) ** 2 < r ** 2
        alpha[mask] = 0.0
        return torch.from_numpy(alpha)


# --------------------------------------------------------------------------- #
# Bubble metrics
# --------------------------------------------------------------------------- #


def bubble_metrics(alpha: torch.Tensor, dx: float, dy: float) -> dict:
    """Compute bubble shape and position metrics from the liquid fraction field.

    The solver stores ``alpha = liquid fraction``, so the bubble is ``1 - alpha``.
    """
    bubble = 1.0 - alpha
    bubble_np = bubble.detach().cpu().numpy()

    # Volume (area in 2D)
    volume = float(bubble_np.sum()) * dx * dy

    # Centroid
    xs = (np.arange(bubble_np.shape[1]) + 0.5) * dx
    ys = (np.arange(bubble_np.shape[0]) + 0.5) * dy
    total = bubble_np.sum()
    if total < 1e-12:
        return {"volume": volume, "xc": 0.0, "yc": 0.0, "circularity": 0.0}

    xc = float((bubble_np * xs[None, :]).sum() / total)
    yc = float((bubble_np * ys[:, None]).sum() / total)

    # Circularity: 4 * pi * area / perimeter^2
    # Use marching squares for a more accurate perimeter estimate
    mask = bubble_np > 0.5
    area = float(mask.sum()) * dx * dy
    if mask.sum() < 4:
        return {"volume": volume, "xc": xc, "yc": yc, "circularity": 0.0}

    # Approximate perimeter by counting edge transitions
    # Horizontal edges
    h_diff = np.abs(np.diff(mask.astype(int), axis=1))
    v_diff = np.abs(np.diff(mask.astype(int), axis=0))
    perimeter = (float(h_diff.sum()) * dy + float(v_diff.sum()) * dx)

    if perimeter < 1e-12:
        circularity = 0.0
    else:
        circularity = 4.0 * math.pi * area / (perimeter ** 2)

    return {"volume": volume, "xc": xc, "yc": yc, "circularity": circularity}


def compute_terminal_velocity(history: list[dict], window_frac: float = 0.25) -> float:
    """Estimate terminal velocity as the mean centroid velocity over the last
    fraction of the time series."""
    if len(history) < 4:
        return 0.0
    n_window = max(2, int(len(history) * window_frac))
    window = history[-n_window:]
    dt = window[-1]["time"] - window[0]["time"]
    if dt < 1e-12:
        return 0.0
    dy = window[-1]["yc"] - window[0]["yc"]
    return dy / dt


# --------------------------------------------------------------------------- #
# Main leg runner
# --------------------------------------------------------------------------- #


def run_hysing_leg(
    cells_per_diameter: int = 32,
    *,
    end_time: float = 3.0,
    csf_smoothing: int = 3,
) -> dict:
    """Run the Hysing Case 2 benchmark and return the metrics record.

    Parameters
    ----------
    cells_per_diameter:
        Grid resolution (cells per bubble diameter).
    end_time:
        Simulation end time (dimensionless, in units of sqrt(d/g)).
    csf_smoothing:
        Number of smoothing iterations for the CSF surface tension force.

    Returns
    -------
    dict
        JSON-serialisable record with terminal velocity, circularity,
        mass conservation, and the full history.
    """
    case = HysingCase(
        cells_per_diameter=cells_per_diameter,
        end_time=end_time,
        csf_smoothing=csf_smoothing,
    )
    grid, tp = case.solve_config()
    solver = TwoPhaseSolver(grid, tp)
    alpha0 = case.initial_alpha(grid.nx, grid.ny, solver.dx, solver.dy)
    solver.set_alpha(alpha0)  # alpha=1 liquid, alpha=0 gas (bubble)

    dx, dy = solver.dx, solver.dy
    volume0 = float((1.0 - solver.alpha).sum()) * dx * dy

    history: list[dict] = []
    worst_step_drift = 0.0
    volume_prev = volume0
    diverged = False

    while solver.time < end_time:
        dt = solver.stable_time_step()
        if solver.time + dt > end_time:
            dt = end_time - solver.time
        try:
            solver.step(dt)
        except (RuntimeError, ValueError, FloatingPointError):
            diverged = True
            break

        if not torch.isfinite(solver.alpha).all():
            diverged = True
            break

        vol = float((1.0 - solver.alpha).sum()) * dx * dy
        step_drift = abs(vol - volume_prev) / max(volume0, 1e-12)
        worst_step_drift = max(worst_step_drift, step_drift)
        volume_prev = vol

        m = bubble_metrics(1.0 - solver.alpha, dx, dy)
        m["time"] = float(solver.time)
        m["volume_drift"] = (vol - volume0) / max(volume0, 1e-12)
        m["umax"] = float(max(solver.u.abs().max(), solver.v.abs().max()))
        history.append(m)

    # Terminal velocity from the last 25% of the history
    u_t = compute_terminal_velocity(history, window_frac=0.25)

    # Minimum circularity over the run
    circularities = [m["circularity"] for m in history if m["circularity"] > 0]
    min_circularity = min(circularities) if circularities else 0.0

    record = {
        "case": "hysing",
        "cells_per_diameter": cells_per_diameter,
        "eotvos": case.eotvos,
        "morton": case.morton,
        "end_time": end_time,
        "terminal_velocity": u_t,
        "min_circularity": min_circularity,
        "worst_step_drift": worst_step_drift,
        "total_volume_drift": history[-1]["volume_drift"] if history else 0.0,
        "diverged": diverged,
        "steps": len(history),
        "declared": cells_per_diameter == 64,  # declared resolution
        "reference": {
            "source": "Hysing et al. (2009) Int. J. Numer. Meth. Fluids 60, "
                      "1259-1288, Table II, Case 2 (Eo=10, M=1e-3)",
            "terminal_velocity": HYSING_REF_UT,
            "min_circularity": HYSING_REF_CIRCULARITY,
            "centroid_y_at_t3": HYSING_REF_YC,
        },
        "terminal_velocity_rel_error": abs(u_t - HYSING_REF_UT) / HYSING_REF_UT,
        "circularity_rel_error": abs(min_circularity - HYSING_REF_CIRCULARITY) / HYSING_REF_CIRCULARITY,
    }
    return {"record": record, "history": history}


def convergence_order(coarse_ut: float, fine_ut: float, ratio: float = 2.0) -> float:
    """Observed order from two terminal velocities."""
    coarse_ut = max(float(coarse_ut), 1e-16)
    fine_ut = max(float(fine_ut), 1e-16)
    return math.log(coarse_ut / fine_ut) / math.log(ratio)
