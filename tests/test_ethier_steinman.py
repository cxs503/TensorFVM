"""Ethier--Steinman 3-D manufactured verification for the native hybrid NS backend.

Two layers, in the order of the module's contract:

1. *The reference itself* -- the closed form is pushed through the
   incompressibility constraint, the atom-algebra derivatives and the
   unforced-momentum curl test before it is allowed to judge a solver.
2. *The solver against the reference* -- discrete divergence-free acceptance,
   the gated velocity ``L_inf`` error and a two-point refinement ladder.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.verification.ethier_steinman import (
    EthierSteinmanCase,
    es_fields,
    es_metrics,
    es_mesh,
    es_solve,
    es_solver,
    observed_order,
    reference_audit,
)

TIME = 0.25


@pytest.fixture(scope="module")
def reference_free_case() -> EthierSteinmanCase:
    """a = d = 1, nu = 1: the Re = 1 unforced (curl(f) = 0) member of the family."""
    return EthierSteinmanCase(a=1.0, d=1.0, viscosity=1.0)


@pytest.fixture(scope="module")
def high_reynolds_case() -> EthierSteinmanCase:
    """Re = 100 as the reference poses it: nu = d^2/a^2 = 0.01 with a = 0.2."""
    return EthierSteinmanCase(a=0.2, d=0.02, viscosity=0.01)


def test_reference_is_divergence_free_and_exact(reference_free_case):
    audit = reference_audit(reference_free_case, samples=8)
    assert audit["analytic_divergence_max"] < 1e-12
    assert audit["finite_difference_divergence_max"] < 1e-8
    # atom algebra (closed-form derivatives) against central differences
    assert audit["gradient_max_error"] < 1e-8
    assert audit["laplacian_max_error"] < 1e-4
    # the velocity is a Laplacian eigenfunction, lap(u) = -a^2 u
    points = np.random.default_rng(3).uniform(-0.5, 0.5, (5, 3))
    state = es_fields(points, reference_free_case, TIME)
    assert np.abs(
        state["velocity_laplacian"] + reference_free_case.a**2 * state["velocity"]
    ).max() < 1e-12 * max(1.0, np.abs(state["velocity"]).max())
    # unforced Navier-Stokes: the momentum residual is a pure pressure gradient
    assert audit["unforced_momentum_curl_max"] < 1e-6
    assert audit["force_curl_max"] < 1e-6


def test_viscosity_outside_the_unforced_relation_needs_a_body_force():
    """Away from nu = d^2/a^2 the same field is a *manufactured* case: curl(f) != 0."""
    manufactured = EthierSteinmanCase(a=1.0, d=1.0, viscosity=0.01)
    assert not manufactured.is_unforced
    audit = reference_audit(manufactured, samples=8)
    assert audit["force_curl_max"] > 1e-2
    assert audit["analytic_divergence_max"] < 1e-12


def test_high_reynolds_member_is_also_unforced(high_reynolds_case):
    """Re = 100 is a genuine (unforced) member: the relation is nu = d^2/a^2, not nu = 1."""
    assert high_reynolds_case.is_unforced
    assert high_reynolds_case.reynolds_number == pytest.approx(100.0)
    audit = reference_audit(high_reynolds_case, samples=8)
    assert audit["analytic_divergence_max"] < 1e-15
    assert audit["unforced_momentum_curl_max"] < 1e-8
    assert audit["force_curl_max"] < 1e-8


def test_high_reynolds_solve_converges_but_is_not_accuracy_qualified(high_reynolds_case):
    """Re = 100 solves (discrete divergence free) but is *not* accuracy qualified at n = 4.

    Measured here: the frozen steady problem at nu = 0.01 is a weak balance of
    small terms, so the velocity is not recovered to 10% on a 4^3 grid (the
    study logs the number instead of gating it).  This test therefore asserts
    only what is true -- convergence and a divergence-free accepted flux.
    """
    mesh, result, exact = es_solve(4, high_reynolds_case, time_s=TIME)
    metrics = es_metrics(mesh, result, exact)
    assert metrics["max_divergence_s_inv"] < 1e-12
    assert np.isfinite(metrics["velocity_relative_linf"])
    assert metrics["velocity_relative_linf"] < 5.0


def test_time_decay_and_frozen_force_consistency(reference_free_case):
    case = reference_free_case
    points = np.random.default_rng(5).uniform(-0.5, 0.5, (4, 3))
    early = es_fields(points, case, 0.0)
    late = es_fields(points, case, 1.0)
    ratio = late["velocity"] / early["velocity"]
    np.testing.assert_allclose(ratio, math.exp(-case.decay_rate), rtol=1e-12)
    # f = du/dt + (u.grad)u - nu lap(u) with du/dt = -d^2 u
    np.testing.assert_allclose(
        early["force"],
        -case.decay_rate * early["velocity"]
        + (early["acceleration"] + case.decay_rate * early["velocity"])
        - case.viscosity * early["velocity_laplacian"],
        rtol=1e-10,
        atol=1e-12,
    )
    # the material acceleration is the time term plus the convective term
    np.testing.assert_allclose(
        early["acceleration"] + case.decay_rate * early["velocity"],
        np.einsum("nj,nji->ni", early["velocity"], early["velocity_gradient"]),
        rtol=1e-10,
        atol=1e-12,
    )


def test_mesh_geometry():
    mesh = es_mesh(3, half_width=1.0)
    assert len(mesh.cells) == 27
    np.testing.assert_allclose(np.sum(mesh.cell_volumes), 8.0, rtol=1e-12)
    assert mesh.cell_centers[:, 0].min() == pytest.approx(-2.0 / 3.0)


def test_solver_reproduces_the_exact_field(reference_free_case):
    mesh, result, exact = es_solve(4, reference_free_case, time_s=TIME)
    metrics = es_metrics(mesh, result, exact)
    assert metrics["max_divergence_s_inv"] < 1e-12
    assert metrics["velocity_relative_linf"] < 0.10
    assert metrics["velocity_relative_l2"] < 0.08


def test_second_order_consistent_ladder(reference_free_case):
    rows = []
    for n in (4, 5):
        mesh, result, exact = es_solve(n, reference_free_case, time_s=TIME)
        metrics = es_metrics(mesh, result, exact)
        metrics["spacing_m"] = 2.0 / n
        rows.append(metrics)
    assert observed_order(rows[0], rows[1]) > 0.8


def test_unconverged_case_raises_instead_of_returning_a_field():
    """A stiff, under-resolved member must raise, not fake a metric."""
    with pytest.raises(RuntimeError):
        es_solve(4, EthierSteinmanCase(a=1.0, d=0.1, viscosity=0.01), time_s=TIME,
                 max_iterations=3)


def test_continuation_state_is_accepted(reference_free_case):
    """A converged state may initialise the next solve (used for viscosity continuation)."""
    mesh, solver, exact = es_solver(4, reference_free_case, time_s=TIME)
    first = solver.solve_newton(max_iterations=25, tolerance=1e-10)
    assert first["numerically_converged"]
    state, pressure = solver.state.copy(), solver.p.copy()
    mesh, solver, exact = es_solver(4, reference_free_case, time_s=TIME)
    solver.state = state.copy()
    solver.p = pressure.copy()
    again = solver.solve_newton(max_iterations=25, tolerance=1e-10)
    assert again["numerically_converged"]