"""Closure-level checks for the three turbulence models under test.

The important test here is not that the constants are what the papers say (they
are copied once, in :mod:`tensorfvm.rans_validation.models`); it is that the
*transported equations* built from them admit the logarithmic layer whose slope
the closed form ``implied_kappa`` claims.  That is checked independently by
bisecting the wall-normal budget of the second transport equation, so a
transcription slip in beta, gamma or sigma would be caught.
"""
import numpy as np
import pytest

from tensorfvm.rans_validation.models import (DEFAULT_CONSTANTS, MODELS, constants_for,
                                              equilibrium_closure, implied_kappa, wall_values)


def budget_residual(model, kappa, *, nu=1e-6, ymin=.05, ymax=.25, samples=401, constants=None):
    """Relative residual of the second transport equation for a log-layer ansatz."""
    c = constants_for(model) if constants is None else dict(constants)
    root = np.sqrt(c['beta_star']) if 'beta_star' in c else np.sqrt(c['c_mu'])
    y = np.linspace(ymin, ymax, samples)
    k = np.full_like(y, 1 / root)
    variable = 1 / (root * kappa * y) if model != 'kepsilon' else 1 / (kappa * y)
    dk = np.zeros_like(y)
    dvariable = -variable / y
    state = equilibrium_closure(model, y, k, variable, dk, dvariable, nu, c, shear=1 / (kappa * y))
    flux = state['variable_diffusion'] * np.gradient(variable, y, edge_order=2)
    residual = np.gradient(flux, y, edge_order=2) + state['variable_production'] - state['variable_sink'] * variable
    inner = slice(5, -5)
    scale = np.abs(state['variable_sink'] * variable)[inner]
    return float(np.mean(residual[inner] / scale))


def bisected_kappa(model, *, constants=None, bounds=(.2, .7)):
    """Root of the signed budget residual -- the slope the closure really admits."""
    from scipy.optimize import brentq
    return brentq(lambda kappa: budget_residual(model, kappa, constants=constants), *bounds, xtol=1e-12)


@pytest.mark.parametrize('model,expected', [('sst', .4082), ('komega', .4336), ('kepsilon', .4327)])
def test_implied_kappa_matches_closed_form(model, expected):
    assert implied_kappa(model) == pytest.approx(expected, rel=2e-3)


@pytest.mark.parametrize('model', MODELS)
def test_transport_budget_admits_the_implied_log_layer(model):
    """The slope implied in closed form zeroes the budget built from the closure."""
    kappa = implied_kappa(model)
    assert abs(budget_residual(model, kappa)) < .02
    assert abs(budget_residual(model, kappa * 1.2)) > 5 * abs(budget_residual(model, kappa))
    assert bisected_kappa(model) == pytest.approx(kappa, rel=.02)


def test_implied_kappa_tracks_a_perturbed_beta_star():
    """A ten per cent change of beta* must move the implied slope, not hide it."""
    base = implied_kappa('sst')
    perturbed = implied_kappa('sst', {**DEFAULT_CONSTANTS['sst'], 'beta_star': .099})
    assert perturbed < base
    assert abs(perturbed / base - 1) > .05


@pytest.mark.parametrize('model', MODELS)
def test_constants_are_copied_and_overrides_validated(model):
    base = constants_for(model)
    copied = constants_for(model)
    assert base == copied and base is not copied
    name = 'beta_star' if model != 'kepsilon' else 'c_mu'
    assert constants_for(model, {name: .1})[name] == .1
    with pytest.raises(ValueError):
        constants_for(model, {'not_a_constant': 1.})
    with pytest.raises(ValueError):
        constants_for(model, {name: -1.})
    with pytest.raises(ValueError):
        constants_for(model, {name: float('nan')})
    with pytest.raises(ValueError):
        constants_for('k-omega-sst')


def test_wall_values_follow_the_published_treatments():
    nu, d = 1 / 590., .02
    k, omega = wall_values('sst', d, nu)
    assert k == 0. and omega == pytest.approx(60 * nu / (.075 * d * d))
    k, omega = wall_values('komega', d, nu)
    assert k == 0. and omega == pytest.approx(6 * nu / (.075 * d * d))
    k, epsilon = wall_values('kepsilon', d, nu)
    assert k == pytest.approx(1 / np.sqrt(.09))
    assert epsilon == pytest.approx(1 / (.41 * d))
    with pytest.raises(ValueError):
        wall_values('sst', 0., nu)


@pytest.mark.parametrize('model', MODELS)
def test_closure_rejects_inadmissible_state(model):
    y = np.array([.01, .1])
    k = np.array([.5, .5])
    variable = np.array([10., 5.])
    zero = np.zeros(2)
    with pytest.raises(ValueError):
        equilibrium_closure(model, -y, k, variable, zero, zero, .01)
    with pytest.raises(ValueError):
        equilibrium_closure(model, y, -k, variable, zero, zero, .01)
    with pytest.raises(ValueError):
        equilibrium_closure(model, y, k, 0 * variable, zero, zero, .01)
    with pytest.raises(ValueError):
        equilibrium_closure(model, y, k, variable, zero, zero, 0.)


@pytest.mark.parametrize('model', MODELS)
def test_eddy_viscosity_and_shear_satisfy_the_total_stress_identity(model):
    """(nu + nu_t) dU/dy = u_tau^2 (1 - y) must hold exactly, cell by cell."""
    y = np.linspace(.002, .98, 40)
    k = .5 + y
    variable = np.where(model == 'kepsilon', .4 + y, 40. / (1 + 60 * y))
    dk = np.gradient(k, y)
    dvariable = np.gradient(variable, y)
    state = equilibrium_closure(model, y, k, variable, dk, dvariable, .01)
    np.testing.assert_allclose((.01 + state['nut']) * state['shear'], 1 - y, rtol=1e-12)
    assert np.all(state['nut'] > 0)
    assert np.all(state['k_production'] >= 0)


def test_kepsilon_sink_is_epsilon_over_k_and_production_is_viscous():
    y = np.linspace(.05, .9, 12)
    k = 1 + 0 * y
    epsilon = .3 + 0 * y
    zero = np.zeros_like(y)
    state = equilibrium_closure('kepsilon', y, k, epsilon, zero, zero, .01)
    np.testing.assert_allclose(state['k_sink'] * k, epsilon, rtol=1e-13)
    np.testing.assert_allclose(state['nut'], .09 * k * k / epsilon, rtol=1e-13)
    np.testing.assert_allclose(state['variable_sink'] * epsilon, 1.92 * epsilon ** 2 / k, rtol=1e-13)
    np.testing.assert_allclose(state['k_diffusion'], .01 + state['nut'], rtol=1e-13)


def test_komega_eddy_viscosity_is_k_over_omega():
    y = np.linspace(.02, .9, 12)
    k = .7 + 0 * y
    omega = 20. + 0 * y
    zero = np.zeros_like(y)
    state = equilibrium_closure('komega', y, k, omega, zero, zero, .01)
    np.testing.assert_allclose(state['nut'], k / omega, rtol=1e-13)
    np.testing.assert_allclose(state['variable_sink'] * omega, .075 * omega ** 2, rtol=1e-13)


def test_sst_production_limiter_engages_at_high_strain():
    """Above 10 beta* omega k the production must be capped, not exceeded."""
    y = np.array([.5])
    k = np.array([1e-6])
    omega = np.array([1e-3])
    zero = np.array([0.])
    state = equilibrium_closure('sst', y, k, omega, zero, zero, 1e-8)
    assert state['k_production'][0] == pytest.approx(10 * .09 * omega[0] * k[0])


def test_sst_blending_brackets_the_inner_and_outer_constants():
    """F1 must fall away from the wall and drag beta towards its outer value."""
    nu = 1 / 590.
    near = equilibrium_closure('sst', np.array([1e-3]), np.array([.2]), np.array([5e3]),
                               np.array([0.]), np.array([-1e7]), nu)
    far = equilibrium_closure('sst', np.array([.9]), np.array([3.]), np.array([9.]),
                              np.array([-1.]), np.array([-30.]), nu)
    assert near['F1'][0] > .9 and far['F1'][0] < .5
    assert near['beta'][0] == pytest.approx(.075, rel=1e-2)
    assert .075 < far['beta'][0] < .0828 and far['beta'][0] > near['beta'][0]
    assert near['gamma'][0] == pytest.approx(5 / 9, rel=1e-2)
    assert .44 < far['gamma'][0] < 5 / 9