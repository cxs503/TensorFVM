"""Frozen-time Taylor--Green vortex verification for the native hybrid NS backend.

The periodic MAC kernel is time marched against the decaying Taylor--Green
vortex (:mod:`tensorfvm.verification.taylor_green_mac`).  The native hybrid 3-D
backend is a *steady* solver and has no transient integrator, so here the exact
solution is frozen at one instant and the time term is supplied as a
manufactured body force ``f = -d u/d t = 2 nu u``:

    (u.grad) u = -grad(p) + nu lap(u) + f.

Two layers are tested:

1. *The reference itself.*  The analytic velocity/pressure are pushed through
   the incompressibility constraint, the unsteady momentum balance and the
   frozen-force identity by finite differences before they judge anything else.
2. *The solver against the reference.*  Discrete divergence-free acceptance,
   the frozen velocity field, and a two-point grid-refinement ladder.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.native_taylor_green import (
    BOX,
    TaylorGreenCase,
    observed_order,
    run_study,
    tgv_fields,
    tgv_solve,
)

STUDY = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "verification-native-taylor-green"
    / "study.json"
)

# The closed frozen-time problem stiffens as nu -> 0; nu = 1.0 is the case the
# steady Newton iteration solves reliably (nu = 0.05 does not converge, see the
# module docstring).
CASE = TaylorGreenCase(amplitude=1.0, viscosity=1.0)
FROZEN_TIME = 0.5


# --------------------------------------------------------------------------- #
# 1. the reference verifies itself
# --------------------------------------------------------------------------- #
def _derivative(case, points, axis, time_s, step=1e-4):
    """Central difference of the velocity along one Cartesian axis."""
    shift = np.zeros(3)
    shift[axis] = step
    plus = tgv_fields(points + shift, case, time_s)["velocity"]
    minus = tgv_fields(points - shift, case, time_s)["velocity"]
    return (plus - minus) / (2 * step)


def test_analytic_fields_satisfy_continuity():
    """The analytic velocity field is divergence-free in all three directions."""
    case = TaylorGreenCase(amplitude=1.3, viscosity=0.07)
    rng = np.random.default_rng(42)
    points = rng.uniform(0.0, BOX, size=(64, 3))
    divergence = sum(_derivative(case, points, axis, 0.0)[:, axis] for axis in range(3))
    assert np.max(np.abs(divergence)) < 1e-10


def test_analytic_fields_satisfy_unsteady_momentum():
    """The analytic fields solve the unsteady and the frozen steady momentum balance."""
    case = TaylorGreenCase(amplitude=1.0, viscosity=0.3)
    rng = np.random.default_rng(17)
    points = rng.uniform(0.0, BOX, size=(32, 3))
    time_s = 0.37
    nu = case.viscosity

    fields = tgv_fields(points, case, time_s)
    velocity = fields["velocity"]

    # Time derivative, spatial derivatives and Laplacian by finite differences.
    dt = 1e-4
    dudt = (
        tgv_fields(points, case, time_s + dt)["velocity"]
        - tgv_fields(points, case, time_s - dt)["velocity"]
    ) / (2 * dt)
    gradient = [_derivative(case, points, axis, time_s) for axis in range(3)]
    laplacian = np.zeros_like(velocity)
    for axis in range(3):
        shift = np.zeros(3)
        shift[axis] = 1e-4
        plus = tgv_fields(points + shift, case, time_s)["velocity"]
        minus = tgv_fields(points - shift, case, time_s)["velocity"]
        laplacian += (plus - 2 * velocity + minus) / (1e-4) ** 2
    pressure_gradient = np.column_stack(
        [
            (
                tgv_fields(points + np.eye(3)[axis] * 1e-4, case, time_s)["pressure"]
                - tgv_fields(points - np.eye(3)[axis] * 1e-4, case, time_s)["pressure"]
            )
            / (2 * 1e-4)
            for axis in range(3)
        ]
    )

    # gradient[axis][:, i] = d u_i / d x_axis, so stack on axis 1 gives
    # stack[n, axis, i]; contract the derivative axis with the exact velocity.
    convection = np.einsum("na,nai->ni", velocity, np.stack(gradient, axis=1))

    unsteady = dudt + convection + pressure_gradient - nu * laplacian
    frozen = convection + pressure_gradient - nu * laplacian - fields["force"]
    assert np.max(np.abs(unsteady)) < 1e-7
    assert np.max(np.abs(frozen)) < 1e-7


def test_frozen_force_is_minus_the_physical_time_derivative():
    """``f = 2 nu u`` equals ``-d u/d t`` of the decaying exact solution."""
    case = TaylorGreenCase(amplitude=0.7, viscosity=0.4)
    points = np.random.default_rng(3).uniform(0.0, BOX, size=(16, 3))
    time_s = 0.8
    dt = 1e-5
    dudt = (
        tgv_fields(points, case, time_s + dt)["velocity"]
        - tgv_fields(points, case, time_s - dt)["velocity"]
    ) / (2 * dt)
    force = tgv_fields(points, case, time_s)["force"]
    np.testing.assert_allclose(force, -dudt, rtol=1e-8, atol=1e-10)
    np.testing.assert_allclose(
        force,
        case.decay_exponent * tgv_fields(points, case, time_s)["velocity"],
        atol=1e-12,
    )


def test_invalid_case_parameters_are_rejected():
    """Non-physical parameters cannot silently change the reference."""
    for kwargs in (
        dict(amplitude=0.0),
        dict(viscosity=-1.0),
        dict(viscosity=float("nan")),
        dict(density=2.0),
    ):
        with pytest.raises(ValueError):
            TaylorGreenCase(**kwargs)
    with pytest.raises(ValueError):
        tgv_fields(np.zeros((2, 2)), CASE)
    with pytest.raises(ValueError):
        tgv_fields(np.zeros((2, 3)), CASE, time_s=-1.0)


# --------------------------------------------------------------------------- #
# 2. solver against reference
# --------------------------------------------------------------------------- #
_LADDER: dict = {}


def _ladder() -> dict:
    """Solve n = 4 and n = 6 once; both solver tests reuse the accepted states."""
    if not _LADDER:
        for n in (4, 6):
            mesh, result, exact = tgv_solve(n, CASE, time_s=FROZEN_TIME)
            from tensorfvm.verification.native_taylor_green import tgv_metrics

            metrics = tgv_metrics(mesh, result, exact)
            metrics["n"] = n
            metrics["spacing_m"] = BOX / n
            _LADDER[n] = metrics
    return _LADDER


def test_solver_recovers_frozen_velocity_and_continuity():
    """The accepted steady state reproduces the frozen exact velocity."""
    metrics = _ladder()[6]
    assert metrics["velocity_relative_l2"] < 0.25, metrics
    assert metrics["max_divergence_s_inv"] < 1e-12, metrics
    # The exact field has w = 0; the z direction must stay inactive.
    assert metrics["velocity_z_max_abs_m_s"] < 1e-3, metrics


def test_solver_velocity_shows_second_order_refinement():
    """Halving h reduces the velocity error at close to second order."""
    ladder = _ladder()
    order = observed_order(ladder[4], ladder[6])
    assert ladder[6]["velocity_relative_l2"] < ladder[4]["velocity_relative_l2"]
    assert 1.5 < order < 2.5, f"observed order: {order}"


def test_run_study_reports_the_same_accepted_states():
    """``run_study`` is the study entry point and must agree with the direct solve."""
    coarse = run_study([4], CASE, time_s=FROZEN_TIME)[0]
    reference = _ladder()[4]
    assert coarse["velocity_relative_l2"] == pytest.approx(
        reference["velocity_relative_l2"]
    )
    assert coarse["max_divergence_s_inv"] == pytest.approx(
        reference["max_divergence_s_inv"]
    )
    assert coarse["resolution"] == "4x4x4"
    assert coarse["spacing_m"] == pytest.approx(BOX / 4)


# --------------------------------------------------------------------------- #
# 3. published study
# --------------------------------------------------------------------------- #
def test_published_study_exists():
    """The published study file must exist and carry the gate record."""
    if not STUDY.exists():
        pytest.skip(f"published study absent from this checkout: {STUDY}")
    study = json.loads(STUDY.read_text())
    assert study["schema"] == "tensorfvm.native-taylor-green-frozen-study/1"
    summary = study["gate_summary"]
    assert summary["spatial_operator_qualified"] is True
    gates = study["gates"]
    for name in ("velocity", "continuity", "spatial_order", "inactive_z"):
        assert gates[name]["passed"] is True, name
    assert gates["pressure"]["passed"] is False
    assert "odd" in gates["pressure"]["reason"].lower()
    ladder = study["spatial_ladder"]
    assert [row["n"] for row in ladder] == [4, 6, 8]
    assert ladder[-1]["velocity_relative_l2"] < study["case"]["velocity_gate"]
    assert math.isfinite(ladder[-1]["pressure_relative_linf"])
