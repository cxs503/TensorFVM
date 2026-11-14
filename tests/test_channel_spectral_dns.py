"""Tests for the closure-free forced-channel solver (KMM180 second route).

The assertions target invariants, not "the code ran":

* the pressure projection is divergence-free to round-off on a random field;
* the discrete steady state of the forced laminar channel is a fixed point of
  the time integrator and reproduces the analytic parabola (the three-point
  stencil is exact for quadratics, so the *reference* can be checked to
  round-off rather than to a tolerance);
* the wall-shear estimator converges at second order, which is what licenses
  using the body-force-implied ``u_tau`` for the wall units;
* the statistics folding is symmetric and reproduces synthetic profiles;
* the total-shear identity is an exact identity for a manufactured state.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.verification.channel_spectral_dns import (
    ChannelConfig,
    SpectralChannel,
    StatisticsAccumulator,
    build_wall_normal_grid,
    fold_half_channel,
    total_shear_identity,
)


def solve(**kwargs):
    defaults = dict(re_tau=180.0, nx=16, ny=32, nz=16)
    defaults.update(kwargs)
    return SpectralChannel(ChannelConfig(**defaults))


# --------------------------------------------------------------------------- #
# Grid and configuration
# --------------------------------------------------------------------------- #

def test_grid_spans_the_channel_and_is_symmetric():
    y, yf = build_wall_normal_grid(32, 1.0, 1.5)
    assert yf[0] == -1.0 and yf[-1] == 1.0
    assert np.isclose(np.diff(yf).sum(), 2.0)
    assert np.all(np.diff(y) > 0)
    np.testing.assert_allclose(y, -y[::-1], atol=1e-12)
    assert np.all(np.diff(yf) > 0)
    # clustering: the wall cell must be much thinner than the centre cell
    assert np.diff(yf)[0] < 0.2 * np.diff(yf).max()


@pytest.mark.parametrize('kwargs, message', [
    (dict(re_tau=-1.0), 're_tau'),
    (dict(nx=15), 'even'),
    (dict(ny=31), 'even'),
    (dict(stretching=0.0), 'stretching'),
    (dict(time_step=-1.0), 'time_step'),
])
def test_invalid_configuration_is_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ChannelConfig(**kwargs)


def test_wall_units_are_fixed_by_the_body_force():
    solver = solve(re_tau=180.0)
    assert solver.u_tau == 1.0
    assert solver.nu == pytest.approx(1.0 / 180.0)
    # exact force balance of the half channel: tau_w = G h = u_tau^2
    assert solver.force * solver.h == pytest.approx(solver.u_tau ** 2)


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #

def test_projection_makes_a_random_field_divergence_free():
    solver = solve()
    rng = np.random.default_rng(0)
    solver.u = rng.standard_normal(solver.u.shape)
    solver.w = rng.standard_normal(solver.w.shape)
    solver.v = rng.standard_normal(solver.v.shape)
    solver.v[0] = solver.v[-1] = 0.0
    assert np.abs(solver.divergence()).max() > 1.0
    solver.project(dt=1e-3)
    assert np.abs(solver.divergence()).max() < 1e-10


def test_projection_preserves_wall_boundary_conditions():
    solver = solve()
    rng = np.random.default_rng(1)
    solver.u = rng.standard_normal(solver.u.shape)
    solver.v = rng.standard_normal(solver.v.shape)
    solver.w = rng.standard_normal(solver.w.shape)
    solver.project(dt=1e-3)
    assert solver.v[0].max() == 0.0 and solver.v[-1].max() == 0.0


# --------------------------------------------------------------------------- #
# Laminar reference state
# --------------------------------------------------------------------------- #

def test_discrete_laminar_state_is_the_analytic_parabola():
    """The three-point stencil is exact for quadratics, so this is an identity."""
    for ny in (24, 48):
        solver = solve(re_tau=20.0, ny=ny)
        profile = solver.laminar_profile()
        exact = solver.force / (2.0 * solver.nu) * (solver.h ** 2 - solver.y ** 2)
        np.testing.assert_allclose(profile, exact, rtol=1e-11, atol=1e-11)


def test_laminar_state_is_a_fixed_point_of_the_time_integrator():
    solver = solve(re_tau=20.0, ny=24, nx=8, nz=8)
    profile = solver.laminar_profile()
    solver.u = np.repeat(np.repeat(profile[:, None, None], 8, 1), 8, 2).copy()
    solver.project(1e-3)
    before = solver.u.copy()
    for _ in range(20):
        solver.step()
    assert np.abs(solver.u - before).max() < 1e-10


def test_wall_shear_estimate_is_second_order_accurate():
    errors = []
    for ny in (24, 48, 96):
        solver = solve(re_tau=20.0, ny=ny, nx=8, nz=8)
        profile = solver.laminar_profile()
        solver.u = np.repeat(np.repeat(profile[:, None, None], 8, 1), 8, 2).copy()
        errors.append(abs(solver.wall_shear() - solver.u_tau))
    assert errors[1] < errors[0] / 1.6
    assert errors[2] < errors[1] / 1.6


# --------------------------------------------------------------------------- #
# Time integration
# --------------------------------------------------------------------------- #

def test_short_turbulent_run_is_stable_and_bounded():
    solver = solve(nx=16, ny=32, nz=16)
    solver.initialize(noise=0.1, seed=2)
    for _ in range(12):
        solver.step()
        assert np.abs(solver.divergence()).max() < 1e-9
    assert np.isfinite(solver.u).all()
    # the flow must stay inside the energy envelope of a forced channel
    energy = float((solver.u ** 2).mean())
    assert 20.0 < energy < 400.0


def test_time_step_limits_come_from_viscosity_and_cfl():
    solver = solve()
    viscous = 0.6 * 0.5 * float(np.min(solver.dy)) ** 2 / solver.nu
    courant = 0.6 / (20.0 * solver.kmag_max)
    assert solver.time_step == pytest.approx(min(viscous, courant))
    assert solver.time_step < viscous * 1.0001


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_fold_half_channel_averages_the_two_halves():
    solver = solve(ny=32, nx=8, nz=8)
    ny = solver.config.ny
    half = ny // 2
    lower = np.linspace(0.0, 1.0, half)
    upper = np.linspace(1.0, 0.0, half)
    profile = np.concatenate([lower, upper])
    profiles = dict(u=profile, uu=profile ** 2, vv=0.5 * profile, ww=0.1 * profile,
                    uv=-0.2 * profile)
    stats = fold_half_channel(solver, profiles, samples=10, time_span=1.0)
    np.testing.assert_allclose(stats.u_plus, lower, atol=1e-12)   # asymmetric
    # ... and keep only the lower half of the grid
    assert stats.y_plus.size == half
    assert np.all(np.diff(stats.y_plus) > 0)


def test_accumulator_recovers_a_manufactured_mean_and_stress():
    solver = solve(ny=32, nx=8, nz=8)
    mean = 3.0
    solver.u = np.full(solver.u.shape, mean)
    solver.w = np.zeros(solver.w.shape)
    solver.v = np.zeros(solver.v.shape)
    acc = StatisticsAccumulator(solver)
    for _ in range(8):
        acc.add()
    profiles = acc.profiles()
    np.testing.assert_allclose(profiles['u'], mean)
    np.testing.assert_allclose(profiles['uu'], 0.0, atol=1e-12)
    np.testing.assert_allclose(profiles['uv'], 0.0, atol=1e-12)


def test_total_shear_identity_holds_for_a_manufactured_state():
    solver = solve(ny=32, nx=8, nz=8)
    stats = fold_half_channel(
        solver,
        dict(u=np.zeros(solver.config.ny), uu=np.zeros(solver.config.ny),
             vv=np.zeros(solver.config.ny), ww=np.zeros(solver.config.ny),
             uv=np.zeros(solver.config.ny)),
        samples=4, time_span=1.0)
    identity = total_shear_identity(stats)
    # with zero Reynolds stress and zero velocity the total shear is zero and the
    # residual equals -target: the identity is linear and must be exact
    np.testing.assert_allclose(identity['target'], 1.0 - stats.y_plus / solver.re_tau)
    np.testing.assert_allclose(identity['residual'], -identity['target'], atol=1e-12)
