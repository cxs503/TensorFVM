"""Verification suite for the discrete-adjoint shape-sensitivity module.

Every test is a *machine-checkable* statement about the implementation; the
acceptance numbers quoted in ``docs/adjoint-shape-sensitivity/`` come from
``python -m tensorfvm.adjoint_shape.study``, which reuses the same functions.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.adjoint_shape.adjoint import (
    adjoint_gradient,
    central_difference_gradient,
    objective_only,
    taylor_remainder,
)
from tensorfvm.adjoint_shape.channel import ChannelGrid, ChannelSolver, poiseuille
from tensorfvm.adjoint_shape.geometry import (
    MODES,
    N_PARAMS,
    PARAM_NAMES,
    FourierBody,
)
from tensorfvm.adjoint_shape.optimize import drag_reduction

R0 = 0.25
CENTER = (2.0, 0.85)
NU = 0.025
FORCE = 12.0 * NU * 1.0 / 4.0


def make(nx=32, ny=16, with_body=True, eta=1.0e-3):
    grid = ChannelGrid(nx=nx, ny=ny)
    body = None
    if with_body:
        body = FourierBody(
            center=CENTER, radius0=R0, delta=1.0 * min(grid.dx, grid.dy)
        )
    return ChannelSolver(grid, body, nu=NU, force=FORCE, eta=eta)


# ---------------------------------------------------------------- geometry --
def test_eight_shape_parameters_and_mode_order():
    assert N_PARAMS == 8
    assert len(PARAM_NAMES) == 8
    assert len(set(PARAM_NAMES)) == 8
    assert [k for k, _ in MODES] == [2, 2, 3, 3, 4, 4, 5, 5]


def test_radius_closed_form_and_derivative():
    body = FourierBody(center=CENTER, radius0=R0, delta=0.1)
    theta = np.array([0.01, -0.02, 0.03, -0.04, 0.05, -0.06, 0.07, -0.08])
    phi = np.linspace(-np.pi, np.pi, 17)
    x = CENTER[0] + 0.3 * np.cos(phi)
    y = CENTER[1] + 0.3 * np.sin(phi)
    expected = R0 * (
        1.0
        + sum(
            theta[q] * (np.cos(k * phi) if trig == "cos" else np.sin(k * phi))
            for q, (k, trig) in enumerate(MODES)
        )
    )
    assert np.allclose(body.radius(x, y, theta), expected, atol=1e-14)
    # theta -> 0 gives the circular body
    assert np.allclose(body.radius(x, y, np.zeros(N_PARAMS)), R0, atol=1e-15)


def test_indicator_is_compact_support_and_differentiable():
    delta = 0.05
    body = FourierBody(center=CENTER, radius0=R0, delta=delta)
    theta = np.array([0.03, 0.0, -0.02, 0.0, 0.01, 0.0, 0.0, 0.0])
    r = np.array([R0 - 3 * delta, R0, R0 + 3 * delta])
    x = np.full_like(r, CENTER[0]) + r
    y = np.full_like(r, CENTER[1])
    alpha, dalpha = body.indicator(x, y, theta)
    assert alpha[0] == pytest.approx(1.0, abs=1e-12)
    assert alpha[2] == pytest.approx(0.0, abs=1e-12)
    assert 0.0 < alpha[1] < 1.0
    assert dalpha.shape == (N_PARAMS, 3)
    # the analytic theta-derivative agrees with a central difference in theta
    eps = 1e-6
    x2 = np.array([CENTER[0] + R0, CENTER[0], CENTER[1] + R0])
    y2 = np.array([CENTER[1], CENTER[1] + R0, CENTER[1] - R0 + 2 * delta])
    x2 = np.array([CENTER[0] + R0, CENTER[0] + R0, CENTER[0]])
    y2 = np.array([CENTER[1], CENTER[1] + delta, CENTER[1] + R0])
    _, dalpha2 = body.indicator(x2, y2, theta)
    for q in range(N_PARAMS):
        e = np.zeros(N_PARAMS)
        e[q] = eps
        ap, _ = body.indicator(x2, y2, theta + e)
        am, _ = body.indicator(x2, y2, theta - e)
        fd = (ap - am) / (2 * eps)
        assert np.allclose(dalpha2[q], fd, rtol=1e-6, atol=1e-9)


# ------------------------------------------------------------------ solver --
def test_poiseuille_profile_is_the_discrete_solution():
    solver = make(nx=32, ny=16, with_body=False)
    theta = np.zeros(N_PARAMS)
    U, info = solver.solve(theta)
    g = solver.grid
    u = U[: g.nu].reshape(g.ny, g.nx)
    exact = np.array(
        [poiseuille(g.y_u(j), g.height, solver.force, solver.nu) for j in range(g.ny)]
    )
    # the half-cell wall stencil is second order in h: at h = 0.125 the profile
    # error is ~6e-3 and it quarters when h halves (see the study's order check)
    assert np.abs(u - exact[:, None]).max() < 1.0e-2
    assert info["iterations"] <= 3
    assert abs(solver.bulk_velocity(U) - 1.0) < 2.0e-2


def test_empty_channel_solution_is_uniform_and_symmetric():
    """Body-free flow must be x-uniform (periodic) and symmetric in y."""
    solver = make(nx=24, ny=12, with_body=False)
    g = solver.grid
    U, info = solver.solve(np.zeros(N_PARAMS))
    R, _ = solver.assemble(np.zeros(N_PARAMS), U, jac=False)
    assert np.linalg.norm(R) / max(np.linalg.norm(U), 1e-30) < 1e-9
    assert info["converged"]
    u = U[: g.nu].reshape(g.ny, g.nx)
    assert np.abs(u - u[:, [0]]).max() < 1e-12          # periodic in x
    assert np.abs(u - u[::-1, :]).max() < 1e-12          # symmetric in y
    Uv = U[g.nu : g.nu + g.nv]
    assert np.abs(Uv).max() < 1e-12


def test_analytic_jacobian_matches_finite_differences():
    solver = make(nx=8, ny=5)
    g = solver.grid
    theta = np.array([0.02, -0.01, 0.0, 0.03, -0.02, 0.0, 0.01, 0.0])
    rng = np.random.default_rng(3)
    U = solver.initial_guess() + 0.2 * rng.standard_normal(g.size)
    _, A = solver.assemble(theta, U)
    A = A.toarray()
    n = g.size
    h = 1e-7
    fd = np.zeros((n, n))
    for c in range(n):
        e = np.zeros(n)
        e[c] = h
        rp = solver.assemble(theta, U + e, jac=False)[0]
        rm = solver.assemble(theta, U - e, jac=False)[0]
        fd[:, c] = (rp - rm) / (2 * h)
    rel = np.linalg.norm(A - fd) / np.linalg.norm(fd)
    assert rel < 1e-6, f"Jacobian / FD mismatch: {rel}"


def test_newton_repairs_the_penalised_flow():
    solver = make(nx=32, ny=16)
    U, info = solver.solve(np.zeros(N_PARAMS))
    assert info["converged"]
    assert info["relative_residual"] < 1.0e-9
    assert info["iterations"] <= 10
    g = solver.grid
    alpha, _, _, _ = solver.geometry(np.zeros(N_PARAMS))
    u = U[: g.nu].reshape(g.ny, g.nx)
    assert np.abs(u[alpha > 0.99]).max() < 0.05  # solid is nearly at rest


def test_drag_closes_the_momentum_budget():
    solver = make(nx=32, ny=16)
    theta = np.zeros(N_PARAMS)
    U, _ = solver.solve(theta)
    bal = solver.momentum_balance(theta, U)
    assert abs(bal["residual"]) / bal["F_drive"] < 1e-8
    assert bal["F_body_penalisation"] > 0.0
    assert bal["ratio"] == pytest.approx(1.0, abs=1e-8)


# ----------------------------------------------------------------- adjoint --
def test_adjoint_transposed_solve_residual():
    solver = make(nx=32, ny=16)
    theta = np.array([0.02, -0.01, 0.0, 0.03, -0.02, 0.0, 0.01, 0.0])
    _, info = adjoint_gradient(solver, theta)
    assert info["adjoint_residual"] < 1e-10
    assert info["lambda_norm"] > 0.0


def test_adjoint_gradient_matches_central_difference():
    solver = make(nx=32, ny=16)
    theta = np.array([0.02, -0.01, 0.0, 0.03, -0.02, 0.0, 0.01, 0.0])
    cd = central_difference_gradient(solver, theta, epsilons=(1e-4,))
    rel = cd["by_epsilon"][1e-4]["rel_error"]
    assert np.max(rel) < 5e-2, f"adjoint/FD disagreement: {rel}"
    assert np.all(np.isfinite(cd["adjoint"]))


def test_taylor_remainder_is_second_order():
    solver = make(nx=32, ny=16)
    theta = 0.02 * np.array([1.0, -0.5, 0.3, -0.7, 0.2, 0.4, -0.1, 0.6])
    out = taylor_remainder(
        solver, theta, epsilons=np.array([3e-4, 1e-3, 3e-3, 1e-2])
    )
    assert 1.7 <= out["slope"] <= 2.2, f"Taylor slope {out['slope']}"


def test_objective_partial_derivatives_are_analytic():
    """``dJ/dtheta`` at fixed ``U`` must equal a central difference of ``C_D``."""
    solver = make(nx=24, ny=12)
    theta = np.array([0.02, -0.01, 0.0, 0.03, -0.02, 0.0, 0.01, 0.0])
    U, _ = solver.solve(theta)
    _, _, dJ_dtheta = solver.objective(theta, U)
    eps = 1e-7
    for q in (0, 3, 7):
        e = np.zeros(N_PARAMS)
        e[q] = eps
        jp = solver.objective(theta + e, U)[0]
        jm = solver.objective(theta - e, U)[0]
        fd = (jp - jm) / (2 * eps)
        assert dJ_dtheta[q] == pytest.approx(fd, rel=1e-6, abs=1e-12)


def test_bulk_velocity_gradient_is_exact():
    solver = make(nx=24, ny=12)
    U = solver.initial_guess()
    g = solver.bulk_velocity_gradient()
    base = solver.bulk_velocity(U)
    eps = 1e-4
    for idx in (0, g.size // 2, g.size - 1):
        Up = U.copy()
        Up[idx] += eps
        assert (solver.bulk_velocity(Up) - base) / eps == pytest.approx(
            g[idx], rel=1e-6, abs=1e-9
        )


def test_objective_converges_under_refinement():
    """``C_D`` must converge monotonically enough that the fine pair agrees to 20 %."""
    theta = np.zeros(N_PARAMS)
    values = []
    for nx, ny in ((32, 16), (48, 24), (64, 32)):
        solver = make(nx=nx, ny=ny)
        values.append(solver.drag_coefficient(theta, solver.solve(theta)[0]))
    coarse_change = abs(values[1] - values[0]) / values[1]
    fine_change = abs(values[2] - values[1]) / values[2]
    assert fine_change < coarse_change, (values, coarse_change, fine_change)
    assert fine_change < 0.20, (values, fine_change)


# -------------------------------------------------------------- optimiser --
def test_drag_reduction_loop_decreases_objective():
    solver = make(nx=32, ny=16)
    theta0 = np.zeros(N_PARAMS)
    theta0[0] = -0.09
    res = drag_reduction(solver, theta0=theta0, maxiter=4, bound=0.12)
    assert res["J_final"] < res["J0"]
    assert res["reduction"] > 0.0
    assert len(res["history"]["J"]) >= 2
    assert np.all(np.abs(res["theta_opt"]) <= 0.12 + 1e-12)


def test_evaluate_shape_is_reproducible():
    solver = make(nx=32, ny=16)
    theta = np.array([0.01, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0, 0.0])
    j1, U1 = objective_only(solver, theta)
    j2, _ = objective_only(solver, theta, U0=U1)
    assert j1 == pytest.approx(j2, rel=1e-9)