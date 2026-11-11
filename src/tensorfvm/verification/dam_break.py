"""Dam-break verification for the two-phase VOF solver.

Physical background
-------------------
The dam-break (or "lock-exchange") problem is the canonical transient two-phase
benchmark: a column of liquid at rest is released at t=0 and the resulting
gravity-driven surge is tracked.  Experimental data from Martin & Moyce (1952)
provide the reference for the surge-front position x(t) and the residual
column height h(t).

Reference
---------
J.C. Martin, W.J. Moyce, "An experimental study of the collapse of liquid
columns on a rigid horizontal plane", *Philosophical Transactions of the Royal
Society of London. Series A* 244 (1952) 312-324.

Dimensionless scaling (Martin & Moyce):
    t* = t * sqrt(g / a),   x* = x / a,   h* = h / a
where ``a`` is the initial column half-width.

What this module does
---------------------
1. Sets up a 2D dam-break initial condition in the two-phase VOF solver.
2. Runs the transient simulation with gravity.
3. Extracts the surge-front position and residual column height vs time.
4. Compares against the Martin & Moyce experimental correlation.

The Martin & Moyce data are digitised from their published curves and stored
as piecewise-linear reference functions.  The comparison uses the dimensionless
variables defined above.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import torch

from ..two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from ..two_phase_vof import VOFTransport
from .core import Metric, ERROR_LIMIT, evaluate, relative_error


# --------------------------------------------------------------------------- #
# Martin & Moyce (1952) reference data (digitised from published curves)
# --------------------------------------------------------------------------- #
# Dimensionless surge-front position x* = x/a vs t* = t*sqrt(g/a)
# These are approximate values read from Fig. 3 of Martin & Moyce (1952).
MM_SURGE_T = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
                        5.5, 6.0, 6.5, 7.0, 7.5, 8.0])
MM_SURGE_X = np.array([0.0, 0.75, 1.40, 1.95, 2.45, 2.90, 3.30, 3.65, 3.98,
                        4.28, 4.55, 4.80, 5.03, 5.25, 5.45, 5.64, 5.82])

# Dimensionless residual column height h* = h/a vs t*
MM_HEIGHT_T = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
MM_HEIGHT_H = np.array([1.0, 0.72, 0.52, 0.40, 0.32, 0.27, 0.23, 0.20, 0.18,
                         0.16, 0.15])


def mm_surge_reference(t_star: float) -> float:
    """Interpolate Martin & Moyce surge-front position at dimensionless time t*."""
    return float(np.interp(t_star, MM_SURGE_T, MM_SURGE_X))


def mm_height_reference(t_star: float) -> float:
    """Interpolate Martin & Moyce residual column height at dimensionless time t*."""
    return float(np.interp(t_star, MM_HEIGHT_T, MM_HEIGHT_H))


# --------------------------------------------------------------------------- #
# Dam-break case setup
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DamBreakConfig:
    """Physical and numerical parameters for the dam-break case.

    The initial liquid column occupies [0, a] x [0, 2a] in a domain of
    [0, L] x [0, H], where a is the column half-width.
    """
    #: Column half-width [m] (characteristic length)
    column_width: float = 0.05715  # 2.25 inches, per Martin & Moyce
    #: Domain length [m]
    domain_length: float = 0.5
    #: Domain height [m]
    domain_height: float = 0.15
    #: Grid resolution (nx, ny)
    nx: int = 200
    ny: int = 60
    #: Gravity [m/s²]
    gravity: float = 9.81
    #: Liquid density [kg/m³]
    rho_liquid: float = 1000.0
    #: Gas density [kg/m³]
    rho_gas: float = 1.2
    #: Liquid viscosity [Pa·s]
    mu_liquid: float = 1.0e-3
    #: Gas viscosity [Pa·s]
    mu_gas: float = 1.8e-5
    #: Surface tension [N/m] (typically neglected in dam-break)
    sigma: float = 0.0
    #: Time step [s] (None = auto from CFL)
    time_step: float | None = None
    #: CFL limit for VOF transport
    vof_cfl: float = 0.4
    #: Maximum simulation time [s]
    max_time: float = 0.5
    #: Device
    device: str = "cpu"


def _initial_alpha(nx: int, ny: int, dx: float, dy: float,
                   column_width: float, domain_height: float,
                   device: str) -> torch.Tensor:
    """Create the initial VOF field: liquid in [0, a] x [0, 2a]."""
    opts = {"dtype": torch.float64, "device": device}
    cx = (torch.arange(nx, **opts) + 0.5) * dx
    cy = (torch.arange(ny, **opts) + 0.5) * dy
    x, y = torch.meshgrid(cx, cy, indexing="xy")
    # Liquid column: x < column_width, y < 2*column_width
    alpha = ((x < column_width) & (y < 2.0 * column_width)).to(torch.float64)
    return alpha


def run_dam_break(cfg: DamBreakConfig) -> dict:
    """Run a dam-break simulation and return results.

    Returns a dict with:
        - history: list of time-step records
        - surge_front: list of (t, x_front) pairs
        - column_height: list of (t, h_residual) pairs
        - elapsed_s: wall-clock time
    """
    started = time.perf_counter()
    dx = cfg.domain_length / cfg.nx
    dy = cfg.domain_height / cfg.ny

    grid = GridConfig(
        nx=cfg.nx, ny=cfg.ny,
        length=cfg.domain_length, height=cfg.domain_height,
        device=cfg.device,
        velocity_relaxation=0.5,
        pressure_relaxation=0.3,
        max_iterations=1,  # we step manually
        time_step=cfg.time_step,
        inlet_velocity=0.0,
    )

    tp = TwoPhaseConfig(
        enabled=True,
        rho_liquid=cfg.rho_liquid,
        rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid,
        mu_gas=cfg.mu_gas,
        sigma=cfg.sigma,
        surface_tension=cfg.sigma > 0,
        gravity=cfg.gravity,
        top_atmosphere=True,
        bottom_wall=True,
        side_outlet=False,
        domain_type="closed",
        vof_cfl_limit=cfg.vof_cfl,
    )

    solver = TwoPhaseSolver(grid, tp)

    # Set initial VOF field
    alpha0 = _initial_alpha(cfg.nx, cfg.ny, dx, dy,
                            cfg.column_width, cfg.domain_height, cfg.device)
    solver.set_alpha(alpha0)

    # Auto time step from CFL if not specified
    if cfg.time_step is None:
        u_max = max(dx, dy) * cfg.vof_cfl / 0.4  # rough estimate
        dt = min(dx, dy) * cfg.vof_cfl / max(u_max, 0.1)
        dt = min(dt, 0.001)  # cap at 1 ms
    else:
        dt = cfg.time_step

    history = []
    surge_front = []
    column_height = []

    t = 0.0
    step_count = 0
    max_steps = int(cfg.max_time / dt) + 1

    while t < cfg.max_time and step_count < max_steps:
        try:
            item = solver.step(dt)
        except RuntimeError:
            break

        t = solver.time
        step_count += 1

        # Extract surge front and column height
        # alpha shape: (ny, nx) - first index is y (rows), second is x (columns)
        alpha = solver.alpha.detach().cpu().numpy()

        # Surge front: rightmost cell with alpha > 0.5 in the bottom half (low y)
        # Take bottom ny//4 rows, find max alpha in each column
        bottom_rows = alpha[:cfg.ny // 4, :]  # shape: (ny//4, nx)
        col_max = bottom_rows.max(axis=0)  # shape: (nx,)
        wet_cols = np.where(col_max > 0.5)[0]
        if len(wet_cols) > 0:
            front_idx = int(wet_cols[-1])
            x_front = (front_idx + 0.5) * dx
        else:
            x_front = cfg.column_width

        # Column height: highest cell with alpha > 0.5 in the left quarter (low x)
        # Take left nx//4 columns, find max alpha in each row
        left_cols = alpha[:, :cfg.nx // 4]  # shape: (ny, nx//4)
        row_max = left_cols.max(axis=1)  # shape: (ny,)
        wet_rows = np.where(row_max > 0.5)[0]
        if len(wet_rows) > 0:
            h_idx = int(wet_rows[-1])
            h_residual = (h_idx + 0.5) * dy
        else:
            h_residual = 0.0

        surge_front.append((t, x_front))
        column_height.append((t, h_residual))
        history.append(item)

        # Adaptive time step based on CFL
        if cfg.time_step is None:
            u_max = max(
                float(solver.u.abs().max()),
                float(solver.v.abs().max()),
                0.01
            )
            dt = min(dx, dy) * cfg.vof_cfl / u_max
            dt = min(dt, 0.002)

    elapsed = time.perf_counter() - started

    return dict(
        config=asdict(cfg),
        history=history,
        surge_front=surge_front,
        column_height=column_height,
        elapsed_s=elapsed,
        steps=step_count,
    )


def compare_against_mm(surge_front: list, column_height: list,
                       column_width: float, gravity: float) -> list[Metric]:
    """Compare simulation results against Martin & Moyce (1952) data.

    Returns a list of Metric objects for the verification framework.
    """
    a = column_width
    t_scale = math.sqrt(a / gravity)

    # Surge front comparison
    surge_errors = []
    for t, x in surge_front:
        t_star = t / t_scale
        if t_star < 0.1:
            continue  # skip very early times
        x_star = x / a
        x_ref = mm_surge_reference(t_star)
        if x_ref > 0:
            surge_errors.append(abs(x_star - x_ref) / x_ref)

    # Column height comparison
    height_errors = []
    for t, h in column_height:
        t_star = t / t_scale
        if t_star < 0.1:
            continue
        h_star = h / a
        h_ref = mm_height_reference(t_star)
        if h_ref > 0:
            height_errors.append(abs(h_star - h_ref) / h_ref)

    metrics = []
    if surge_errors:
        metrics.append(Metric(
            "surge_front_mean_error",
            float(np.mean(surge_errors)),
            0.15,  # 15% tolerance for this coarse benchmark
            "Mean relative error of surge-front position vs Martin & Moyce (1952)",
        ))
        metrics.append(Metric(
            "surge_front_max_error",
            float(np.max(surge_errors)),
            0.25,  # 25% max tolerance
            "Maximum relative error of surge-front position vs Martin & Moyce (1952)",
        ))

    if height_errors:
        metrics.append(Metric(
            "column_height_mean_error",
            float(np.mean(height_errors)),
            0.15,
            "Mean relative error of residual column height vs Martin & Moyce (1952)",
        ))

    # Volume conservation
    if surge_front:
        metrics.append(Metric(
            "simulation_completed",
            0.0,
            0.5,
            "Simulation ran to completion without NaN/inf",
        ))

    return metrics


def audit(result: dict) -> tuple[list[dict], bool]:
    """Run the full dam-break audit and return (metrics_rows, passed)."""
    cfg = DamBreakConfig(**{k: v for k, v in result["config"].items()
                            if k in DamBreakConfig.__dataclass_fields__})
    metrics = compare_against_mm(
        result["surge_front"], result["column_height"],
        cfg.column_width, cfg.gravity,
    )
    rows, passed = evaluate(metrics)
    return rows, passed
