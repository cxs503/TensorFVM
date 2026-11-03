"""Temporal order of accuracy of the periodic MAC kernel (verification debt).

Why this module exists
----------------------
``scripts/run_periodic_mac.py`` certifies the *spatial* order of
:class:`tensorfvm.periodic_mac.PeriodicMACSolver` and reports a three-level
``temporal_order`` gate, but that gate compares the discrete solution with a
reference field that is itself contaminated by a **dt-independent** error floor
(the O(h^2) mismatch between the discrete convective term and a discrete
gradient, plus the linear semi-discrete decay eigenmode).  A floor caps the
measured error, so a log2 ratio of errors-versus-reference is *not* a temporal
order once the floor dominates.  The pressure field was never measured at all,
and ``physical_accuracy_qualified`` is published as ``False``.

This module closes that debt with a measurement that is immune to the floor:

``self-convergence``
    successive solutions on the *same* grid are differenced.  Every
    dt-independent error cancels exactly in the difference, so if
    ``E(dt) = e0 + C dt^p`` then ``||E(dt) - E(dt/2)|| = C dt^p (1 - 2^-p)``
    and the ratio of successive differences still recovers ``p``.  The
    ``estimator`` section below pins this algebra on synthetic sequences, so
    the tool is certified before it is applied.

The exact solution used is the Taylor-Green vortex, which is an exact
Navier-Stokes solution: on the periodic box ``[0, 2*pi)^3``

    u = ( A sin x cos y, -A cos x sin y, 0 ) exp(-2 nu t)

with the convective acceleration balanced by ``p = A^2/4 (cos 2x + cos 2y)
exp(-4 nu t)``.  The field is z-invariant, so a four-plane grid exercises the
three-dimensional operator at a two-dimensional cost.

Scope, stated honestly: this is a smooth laminar solution on a uniform
periodic box.  It says nothing about wall-bounded, turbulent or sharp-gradient
regimes, and nothing about the cylinder/FSI backends, which this prototype
explicitly does not provide.
"""
import math
from dataclasses import dataclass

import torch

from tensorfvm.periodic_mac import PeriodicMACConfig, PeriodicMACSolver

__all__ = [
    "TemporalStudyConfig",
    "face_coordinates",
    "taylor_green_state",
    "taylor_green_exact_velocity",
    "taylor_green_exact_pressure",
    "taylor_green_exact_kinetic_energy",
    "run_midpoint",
    "run_first_order_control",
    "relative_difference",
    "observed_orders",
    "richardson_order",
    "estimator_analytic_check",
    "study",
]


@dataclass(frozen=True)
class TemporalStudyConfig:
    """Grid, time window and exact-solution parameters for the order study."""

    n: int = 16
    planes: int = 4
    viscosity_pa_s: float = 0.1
    amplitude: float = 1.0
    final_time_s: float = 1.0
    time_steps_s: tuple[float, ...] = (0.2, 0.1, 0.05, 0.025)

    def __post_init__(self):
        if self.n < 4 or self.planes < 2:
            raise ValueError("n >= 4 and planes >= 2 are required")
        if not self.viscosity_pa_s > 0 or not self.final_time_s > 0:
            raise ValueError("viscosity and final time must be positive")
        if not self.amplitude > 0:
            raise ValueError("amplitude must be positive")
        for dt in self.time_steps_s:
            if not dt > 0 or not math.isfinite(dt):
                raise ValueError("time steps must be positive and finite")
        for dt in self.time_steps_s:
            steps = self.final_time_s / dt
            if abs(steps - round(steps)) > 1e-12:
                raise ValueError("each time step must divide the final time")


def _grid(config: TemporalStudyConfig, time_step_s: float) -> PeriodicMACSolver:
    return PeriodicMACSolver(PeriodicMACConfig(
        nx=config.n, ny=config.n, nz=config.planes,
        time_step_s=time_step_s, viscosity_pa_s=config.viscosity_pa_s,
        nonlinear_iterations=400))


def face_coordinates(solver: PeriodicMACSolver, axis: int) -> tuple:
    """``(x, y, z)`` arrays with axis ``axis`` on faces and the others on centres.

    ``face_coordinates`` shifts exactly one axis by ``h/2``, so each velocity
    component must be sampled with its *own* coordinate triple; reusing the
    x-face abscissa for the y component silently destroys the discrete
    divergence-free property (``step`` checks it and raises ``ValueError``).
    """
    return tuple(solver.face_coordinates(axis))


def taylor_green_state(solver: PeriodicMACSolver, amplitude: float = 1.0) -> None:
    """Load the Taylor-Green face velocity (divergence-free to round-off)."""
    first = face_coordinates(solver, 0)   # x on faces, y on centres
    second = face_coordinates(solver, 1)  # x on centres, y on faces
    solver.velocity[0] = amplitude * torch.sin(first[0]) * torch.cos(first[1])
    solver.velocity[1] = -amplitude * torch.cos(second[0]) * torch.sin(second[1])
    solver.velocity[2].zero_()


def taylor_green_exact_velocity(solver: PeriodicMACSolver, viscosity_pa_s: float,
                                time_s: float, amplitude: float = 1.0) -> torch.Tensor:
    """Exact Navier-Stokes Taylor-Green face velocity at ``time_s``."""
    decay = math.exp(-2.0 * viscosity_pa_s * time_s)
    reference = torch.zeros_like(solver.velocity)
    first = face_coordinates(solver, 0)
    second = face_coordinates(solver, 1)
    reference[0] = amplitude * decay * torch.sin(first[0]) * torch.cos(first[1])
    reference[1] = -amplitude * decay * torch.cos(second[0]) * torch.sin(second[1])
    return reference


def taylor_green_exact_pressure(solver: PeriodicMACSolver, viscosity_pa_s: float,
                               time_s: float, amplitude: float = 1.0) -> torch.Tensor:
    """Exact Taylor-Green pressure on cell centres, zero mean (``p`` gauge)."""
    x, y, _ = solver.centers()
    decay = math.exp(-4.0 * viscosity_pa_s * time_s)
    pressure = 0.25 * amplitude ** 2 * decay * (torch.cos(2 * x) + torch.cos(2 * y))
    return pressure - pressure.mean()


def taylor_green_exact_kinetic_energy(solver: PeriodicMACSolver, viscosity_pa_s: float,
                                      time_s: float, amplitude: float = 1.0) -> float:
    """Exact kinetic energy ``K(t) = K(0) exp(-4 nu t)`` of the Taylor-Green vortex."""
    volume = solver.volume * solver.config.density_kg_m3
    reference = taylor_green_exact_velocity(solver, viscosity_pa_s, time_s, amplitude)
    return 0.5 * volume * float((reference.square()).sum())


def _advance(solver: PeriodicMACSolver, steps: int) -> None:
    for _ in range(steps):
        solver.step()


def run_midpoint(config: TemporalStudyConfig, time_step_s: float) -> PeriodicMACSolver:
    """Advance the shipped implicit-midpoint kernel to the final time."""
    solver = _grid(config, time_step_s)
    taylor_green_state(solver, config.amplitude)
    _advance(solver, round(config.final_time_s / time_step_s))
    return solver


def run_first_order_control(config: TemporalStudyConfig, time_step_s: float) -> PeriodicMACSolver:
    """Negative control: forward-Euler momentum update with the same projection.

    The state update is replaced by an explicit Euler predictor evaluated at
    the old (divergence-free) state, then projected with the shipped FFT
    solver.  Reusing the production operators isolates the *time* integration:
    only the trapezoidal averaging is removed, so the measured order must fall
    to one.  If it did not, the estimator would be blind to a genuine
    first-order defect and would not be worth trusting.
    """
    solver = _grid(config, time_step_s)
    taylor_green_state(solver, config.amplitude)
    c = solver.config
    for _ in range(round(config.final_time_s / time_step_s)):
        old = solver.velocity.clone()
        flux = solver.convective_fluxes(old)
        stress = solver.full_stress(old)
        acceleration = (-solver.convective_divergence(flux)
                        + solver.stress_divergence(stress)) / c.density_kg_m3
        updated, pressure, _ = solver.project(old + c.time_step_s * acceleration)
        solver.velocity = updated
        solver.pressure = pressure
        solver.time += c.time_step_s
    return solver


def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    """L2 difference of two discrete fields normalised by the finer one."""
    return float(torch.linalg.vector_norm(a - b)) / float(torch.linalg.vector_norm(b))


def observed_orders(differences: list[float]) -> list[float]:
    """log2 slope between successive differences of a halving refinement series."""
    orders = []
    for coarse, fine in zip(differences, differences[1:]):
        if coarse <= 0.0 or fine <= 0.0:
            raise ValueError("refinement differences must be positive")
        orders.append(math.log(coarse / fine, 2))
    return orders


def richardson_order(differences: list[float]) -> float:
    """Least-squares order from a series of differences taken at halving ``dt``.

    ``differences[k] = C (dt0 2^-k)^p``, so the slope of ``log(difference)``
    against ``k`` is ``-p log 2``; dividing by ``log 2`` returns ``p``.
    """
    if len(differences) < 2:
        raise ValueError("at least two differences are required")
    logs = [math.log(d) for d in differences]
    steps = list(range(len(differences)))
    mean_x = sum(steps) / len(steps)
    mean_y = sum(logs) / len(logs)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(steps, logs))
    denominator = sum((x - mean_x) ** 2 for x in steps)
    return -numerator / denominator / math.log(2.0)


def estimator_analytic_check(order: float = 2.0, floor: float = 1e-3,
                            levels: int = 4) -> dict[str, object]:
    """Certify the estimator on sequences with a known order and a known floor.

    The synthetic series is ``E(dt_k) = dt_k**p + floor`` with ``dt_k = 2**-k``.
    Two lessons are pinned analytically rather than numerically:

    * the floor makes the *error-versus-reference* ratio blind to the temporal
      order: its apparent order drifts towards zero as ``dt`` shrinks;
    * differencing successive solutions on the same grid cancels the floor
      exactly (``E_k - E_{k+1} = dt_k**p (1 - 2**-p)``) and still returns ``p``.
    """
    if order <= 0 or floor <= 0 or levels < 3:
        raise ValueError("order, floor and levels must be positive; levels >= 3")
    errors = [2.0 ** (-order * k) + floor for k in range(levels)]
    differences = [coarse - fine for coarse, fine in zip(errors, errors[1:])]
    return {
        "synthetic_order": order,
        "floor": floor,
        "errors": errors,
        "error_orders": [math.log(a / b, 2) for a, b in zip(errors, errors[1:])],
        "difference_orders": observed_orders(differences),
        "difference_richardson_order": richardson_order(differences),
    }


def scaled_time_steps(config: TemporalStudyConfig, grid: int) -> list[float]:
    """``dt`` series for ``grid``, halved per grid doubling to hold the CFL number.

    The kernel raises (rather than returning NaNs) when the Picard iteration
    attached to the implicit midpoint step cannot close; at ``n = 32`` and
    ``dt = 0.2`` the convective CFL number is ~1 and it does exactly that.  The
    order study therefore keeps ``dt / h`` fixed across grids, which is also the
    only way the two grids see the same physical time-step regime.
    """
    if grid % config.n or grid < config.n:
        raise ValueError("grid must be an integer multiple of config.n")
    factor = grid // config.n
    steps = [dt / factor for dt in config.time_steps_s]
    for dt in steps:
        count = config.final_time_s / dt
        if abs(count - round(count)) > 1e-12:
            raise ValueError("scaled time step must divide the final time")
    return sorted(steps, reverse=True)


def study(config: TemporalStudyConfig = TemporalStudyConfig(),
          grids: tuple[int, ...] = (16, 32)) -> dict[str, object]:
    """Run the temporal order study on each grid and return unrounded numbers.

    Two grids are reported so that a mesh-dependent artefact can be told apart
    from a genuine temporal order: the observed orders must not depend on
    ``n``, while the error-versus-exact floor must fall with ``n`` if it is
    spatial.
    """
    per_grid = []
    for grid in grids:
        dts = scaled_time_steps(config, grid)
        local = TemporalStudyConfig(**{**config.__dict__, "n": grid,
                                       "time_steps_s": tuple(dts)})
        midpoint = [run_midpoint(local, dt) for dt in dts]
        control = [run_first_order_control(local, dt) for dt in dts]

        def series(runs, attribute):
            fields = [getattr(run, attribute) for run in runs]
            if attribute == "pressure":
                fields = [field - field.mean() for field in fields]
            return [relative_difference(a, b) for a, b in zip(fields, fields[1:])]

        def enriched(differences):
            return {
                "differences": differences,
                "two_level_orders": observed_orders(differences) if len(differences) > 1 else [],
                "richardson_order": richardson_order(differences) if len(differences) > 1 else None,
            }

        floor_rows = []
        for dt, run in zip(dts, midpoint):
            exact_velocity = taylor_green_exact_velocity(
                run, local.viscosity_pa_s, local.final_time_s, local.amplitude)
            exact_pressure = taylor_green_exact_pressure(
                run, local.viscosity_pa_s, local.final_time_s, local.amplitude)
            exact_energy = taylor_green_exact_kinetic_energy(
                run, local.viscosity_pa_s, local.final_time_s, local.amplitude)
            floor_rows.append({
                "dt_s": dt,
                "velocity_error_vs_exact": relative_difference(run.velocity, exact_velocity),
                "pressure_error_vs_exact": relative_difference(
                    run.pressure - run.pressure.mean(), exact_pressure),
                "kinetic_energy_error_vs_exact": abs(
                    float(run.kinetic_energy(run.velocity)) - exact_energy) / exact_energy,
                "nonlinear_iterations": max(h["nonlinear_iterations"] for h in run.history),
                "all_steps_converged": all(h["nonlinear_converged"] for h in run.history),
            })
        per_grid.append({
            "n": grid,
            "time_steps_s": list(dts),
            "velocity_at_final_time_l2": float(torch.linalg.vector_norm(midpoint[-1].velocity)),
            "cells": grid * grid * local.planes,
            "velocity": enriched(series(midpoint, "velocity")),
            "pressure": enriched(series(midpoint, "pressure")),
            "first_order_control_velocity": enriched(series(control, "velocity")),
            "error_vs_exact": floor_rows,
        })

    return {
        "schema": "tensorfvm.periodic-mac-temporal-order/1",
        "config": config.__dict__ | {"time_steps_s": list(config.time_steps_s)},
        "grids": per_grid,
        "estimator": estimator_analytic_check(),
        "notes": [
            "orders come from self-convergence on a fixed grid, so every "
            "dt-independent (spatial) error cancels in the difference",
            "pressure is compared after removing its spatial mean: the projection "
            "potential is defined up to an additive constant",
            "the first-order control keeps the production projection and replaces "
            "only the trapezoidal momentum averaging, so it isolates the time scheme",
            "error_vs_exact is reported next to the orders on purpose: it saturates "
            "at the spatial floor and must not be read as a temporal order",
            "each grid refines dt only: the time-step series is halved per grid "
            "doubling so that dt/h, and hence the convective CFL number, is fixed",
        ],
    }


if __name__ == "__main__":  # pragma: no cover - manual study entry point
    import json
    print(json.dumps(study(), indent=2))