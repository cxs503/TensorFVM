"""Regression + verification tests for the moving-wall channel and Stokes' 2nd problem.

The full criterion sweep lives in ``verification.stokes_second_problem`` (run it
as a script); these tests keep the fast, decisive checks in the suite.
"""
import math

import numpy as np
import pytest
import torch

from tensorfvm.channel_periodic import PeriodicChannelConfig, PeriodicChannelSolver
from tensorfvm.verification import stokes_second_problem as S


# ------------------------------------------------------------------ closed form
def test_closed_form_satisfies_the_momentum_equation():
    for w in (2.0, 5.0, 10.0):
        r = S.closed_form_residual(omega=S.omega_for(w))
        assert r["relative"] < 1e-6


def test_slow_limit_is_couette_and_fast_limit_is_semi_infinite():
    lim = S.limits()
    assert lim["couette_relative_error"] < 1e-3
    assert lim["semi_infinite_relative_error"] < 1e-9


def test_wall_shear_agrees_with_a_numeric_gradient_of_the_profile():
    omega = S.omega_for(5.0)
    for phase in np.arange(4) * (math.pi / 2.0):
        t = phase / omega
        exact = S.exact_wall_gradient(t, omega=omega)
        numeric = S.wall_gradient_numeric(0.0, t, omega=omega)
        assert abs(numeric - exact) / abs(exact) < 1e-6


def test_textbook_semi_infinite_wall_shear_amplitude_and_phase():
    far = S.exact_amplitudes(womersley=200.0)
    assert far["shear_amplitude"] == pytest.approx(far["semi_infinite_shear_amplitude"], rel=1e-9)
    assert far["plate_shear_phase_deg"] == pytest.approx(45.0, abs=1e-6)


def test_bulk_amplitude_matches_numeric_integration_of_the_profile():
    omega = S.omega_for(5.0)
    y = np.linspace(0.0, S.H, 200001)
    numeric = S.exact_profile(y, 0.0, omega=omega).__abs__()  # placeholder for shape
    exact = S.exact_bulk_amplitude(omega=omega)
    # integrate the complex amplitude numerically (trapezoid, 2e5 points)
    complex_y = S.complex_amplitude(y, omega=omega)
    numeric_bulk = np.sum(0.5 * (complex_y[1:] + complex_y[:-1]) * np.diff(y)) / S.H
    assert abs(numeric_bulk - exact) / abs(exact) < 1e-8
    assert numeric.shape == y.shape


# ------------------------------------------------------------- linear algebra fix
@pytest.mark.parametrize("n", [9, 33, 257])       # both branches of the solver
def test_vectorised_tridiagonal_solve_matches_the_banded_reference(n):
    torch.manual_seed(n)
    m = 4
    lower = -torch.rand(n, m, dtype=torch.float64)
    upper = -torch.rand(n, m, dtype=torch.float64)
    diag = 3.0 + torch.rand(n, m, dtype=torch.float64)      # diagonally dominant
    lower[0] = 0.0
    upper[-1] = 0.0
    rhs = torch.randn(n, m, dtype=torch.complex128)
    ref = PeriodicChannelSolver._thomas(lower, diag, upper, rhs)
    fast = PeriodicChannelSolver._tridiagonal_solve(lower, diag, upper, rhs)
    assert float((ref - fast).abs().max()) < 1e-12 * float(ref.abs().max())


# ----------------------------------------------------------------- moving walls
def test_static_wall_solution_is_unchanged():
    cfg = PeriodicChannelConfig(nx=4, ny=32, length_m=1.0, height_m=1.0,
                                density_kg_m3=1.0, viscosity_pa_s=0.01,
                                body_force_m_s2=0.12, time_step_s=0.5)
    solver = PeriodicChannelSolver(cfg)
    info = solver.solve_steady(tolerance=1e-12)
    assert info["converged"]
    assert solver.wall_velocity_m_s == (0.0, 0.0)
    assert solver.momentum_balance()["relative_residual"] < 1e-8
    # half-cell wall flux of a static wall: mu * 2 u_0 / dy, positive into the wall
    lower, upper = solver.wall_shear("flux")
    assert lower > 0 and upper > 0


@pytest.mark.parametrize("wall_flux", ["half-cell", "quadratic"])
def test_steady_moving_wall_reproduces_couette(wall_flux):
    control = S.steady_wall_control(ny=32, wall_flux=wall_flux)
    assert control["converged"]
    assert control["max_profile_error"] < 1e-10
    assert control["max_wall_normal_velocity"] == 0.0
    assert control["bulk_velocity"] == pytest.approx(0.5, rel=1e-8)


def test_uniform_profile_between_moving_walls_has_zero_shear():
    cfg = PeriodicChannelConfig(nx=4, ny=16, length_m=1.0, height_m=1.0,
                                density_kg_m3=1.0, viscosity_pa_s=0.01,
                                body_force_m_s2=1e-12, time_step_s=0.1)
    solver = PeriodicChannelSolver(cfg)
    solver.set_wall_velocity(0.3, 0.3)
    solver.set_velocity_profile(np.full(16, 0.3))
    for estimator in ("flux", "quadratic"):
        lower, upper = solver.wall_shear(estimator)
        assert abs(lower) < 1e-12 and abs(upper) < 1e-12


def test_wall_velocity_is_validated_and_persists_between_steps():
    cfg = PeriodicChannelConfig(nx=4, ny=8, length_m=1.0, height_m=1.0,
                                density_kg_m3=1.0, viscosity_pa_s=0.01,
                                body_force_m_s2=1e-12, time_step_s=0.1)
    solver = PeriodicChannelSolver(cfg)
    solver.step(wall_velocity_m_s=0.25)
    assert solver.wall_velocity_m_s == (0.25, 0.0)
    solver.step()                                   # keeps the prescribed wall speed
    assert solver.wall_velocity_m_s == (0.25, 0.0)
    solver.step(wall_velocity_m_s=(0.1, -0.1))
    assert solver.wall_velocity_m_s == (0.1, -0.1)
    with pytest.raises(ValueError):
        solver.step(wall_velocity_m_s=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        solver.step(wall_velocity_m_s=float("nan"))


# ------------------------------------------------------------ production solver
def test_oscillating_wall_matches_the_closed_form_and_closes_the_momentum_budget():
    run = S.solve_oscillating_wall(womersley=2.0, ny=32, dt_fraction=1.0 / 200.0,
                                   settle_tol=1e-4)
    assert run["settle_periods"] >= 1                # the wall drove the flow from rest
    assert run["profile_relative_l2"] < 3e-2
    assert S.momentum_budget(run)["relative"] < 1e-10
    assert run["max_wall_normal_velocity"] == 0.0
    assert run["max_divergence"] < 1e-10
    exact = S.exact_amplitudes(womersley=2.0)
    assert abs(run["bulk_amplitude"]["amplitude"] / exact["bulk_amplitude"] - 1.0) < 3e-2


def test_a_solver_with_static_walls_cannot_pass_the_oscillating_benchmark():
    y = np.linspace(0.0, S.H, 21)
    omega = S.omega_for(5.0)
    oscillating = S.exact_profile(y, 0.25 * 2 * math.pi / omega, omega=omega)
    couette = S.couette_profile(y)
    relative = float(np.sqrt(np.mean((oscillating - couette) ** 2))
                     / np.sqrt(np.mean(oscillating ** 2)))
    assert relative > 0.1
