"""Verification tests for the RANS ship-resistance surface.

The tests are ordered as the acceptance criteria: analytic anchors first
(Blasius), then the closures, then the marching solver on the flat plate, then
the KVLCC2 geometry audit and the post-processing used by the campaign.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.rans_ship_resistance.anchors import (
    BLASIUS_FPP, SIMMAN2008_KVLCC2, blasius_solution, blasius_eta_at_fraction,
    one_seventh_mean_cf, one_seventh_profile,
    schlichting_mean_cf,
)
from tensorfvm.rans_ship_resistance.boundary_layer import march_boundary_layer
from tensorfvm.rans_ship_resistance.closures import (
    SA_CONSTANTS, SST_CONSTANTS, sa_eddy_viscosity, sa_source_terms,
    sst_blending, sst_coefficients, sst_eddy_viscosity, sst_production,
)
from tensorfvm.rans_ship_resistance.kvlcc2 import (
    KVLCC2_S_FULL_SCALE, build_kvlcc2_hull, equivalent_body_radius,
    equivalent_body_velocity,
)
from tensorfvm.rans_ship_resistance.speed_sweep import (
    criterion_c3, fit_ct_froude, grid_convergence_index,
)


# ---------------------------------------------------------------- anchors
def test_blasius_wall_derivative_matches_tabulated_value():
    sol = blasius_solution()
    assert sol["fpp0"] == pytest.approx(BLASIUS_FPP, abs=1e-8)
    assert 0.95 < sol["fp"][-1] <= 1.0 + 1e-9


def test_blasius_thickness_constants():
    # eta_99 = 4.910 for f' = 0.99 and eta_999 = 6.012 for f' = 0.999
    assert blasius_eta_at_fraction(0.99) == pytest.approx(4.910, abs=2e-2)
    assert blasius_eta_at_fraction(0.999) == pytest.approx(6.012, abs=5e-2)


def test_friction_laws_are_ordered_and_positive():
    re = np.array([1e5, 1e6, 1e7, 1e8])
    assert np.all(np.diff(schlichting_mean_cf(re)) < 0)
    assert np.all(np.diff(one_seventh_mean_cf(re)) < 0)
    # the two laws cross near Re ~ 3e6: the 1/7 power law is slightly above the
    # logarithmic line at low Re and increasingly below it at high Re
    ratio = one_seventh_mean_cf(re) / schlichting_mean_cf(re)
    assert ratio[0] > 1.0 and ratio[-1] < 0.9        # the crossing is inside the band
    assert ratio[-1] < ratio[0] * 0.95


def test_one_seventh_profile_endpoints():
    assert one_seventh_profile(0.0) == pytest.approx(0.0)
    assert one_seventh_profile(1.0) == pytest.approx(1.0)
    assert one_seventh_profile(2.0) == pytest.approx(1.0)


def test_simman_reference_scales_and_propagates_provenance():
    ref = SIMMAN2008_KVLCC2
    assert ref.scale_ratio == pytest.approx(320.0 / 7.0)
    assert ref.s_model == pytest.approx(27578.0 / (320.0 / 7.0) ** 2)
    assert ref.provenance == "withheld"
    fn = ref.froude_for_ship_speed(15.5)
    assert fn == pytest.approx(0.142, abs=1e-3)
    assert ref.model_speed_for_ship_speed(15.5) == pytest.approx(
        ref.speed_for_froude(fn))


# --------------------------------------------------------------- closures
def test_sst_blending_limits():
    k = np.array([1e-4, 1e-4])
    omega = np.array([1e3, 1.0])
    d = np.array([1e-5, 100.0])
    blend = sst_blending(k, omega, np.zeros(2), np.zeros(2), d, 1e-5)
    assert 0.0 <= blend["f1"][0] <= 1.0 and 0.0 <= blend["f1"][1] <= 1.0
    # near the wall F1 -> 1, far from the wall F1 -> 0
    assert blend["f1"][0] > blend["f1"][1]


def test_sst_eddy_viscosity_limiter():
    k, omega, strain, f2 = 1e-3, 10.0, 1e4, 1.0
    nu_t = sst_eddy_viscosity(k, omega, strain, f2)
    assert nu_t == pytest.approx(SST_CONSTANTS["a1"] * k / (strain * f2))
    nu_t_small_strain = sst_eddy_viscosity(k, omega, 1e-6, f2)
    assert nu_t_small_strain == pytest.approx(k / omega)


def test_sst_production_limiter_engages():
    nu_t, strain, k, omega = 1e-2, 1e4, 1e-6, 1e2
    limited = sst_production(nu_t, strain, k, omega)
    assert limited == pytest.approx(10.0 * SST_CONSTANTS["beta_star"] * k * omega)


def test_sst_coefficients_blend_between_the_two_sets():
    inner = sst_coefficients(np.array([1.0]))
    outer = sst_coefficients(np.array([0.0]))
    assert inner["beta"][0] == pytest.approx(SST_CONSTANTS["beta1"])
    assert outer["beta"][0] == pytest.approx(SST_CONSTANTS["beta2"])


def test_sa_constants_and_asymptotics():
    assert SA_CONSTANTS["cw1"] == pytest.approx(
        SA_CONSTANTS["cb1"] / SA_CONSTANTS["kappa"] ** 2
        + (1.0 + SA_CONSTANTS["cb2"]) / SA_CONSTANTS["sigma"])
    nu = 1e-5
    # fv1 -> 0 for small nu_tilde and -> 1 for large nu_tilde
    assert sa_eddy_viscosity(np.array([1e-9]), nu)[0] < 1e-9
    assert sa_eddy_viscosity(np.array([1.0]), nu)[0] == pytest.approx(1.0, rel=1e-6)


def test_sa_destruction_is_quadratic_and_positive():
    nu = 1e-5
    nu_tilde = np.array([1e-4, 2e-4])
    strain = np.array([10.0, 10.0])
    d = np.array([1e-3, 1e-3])
    terms = sa_source_terms(nu_tilde, strain, d, nu)
    destruction = terms["destruction_coefficient"] * nu_tilde ** 2
    assert np.all(destruction > 0)
    assert destruction[1] / destruction[0] == pytest.approx(4.0, rel=1e-2)


# ---------------------------------------------------------------- solver
@pytest.fixture(scope="module")
def laminar_plate():
    nu, u, length = 1e-7, 1.0, 1.0
    x = np.geomspace(0.002, length, 120)
    result = march_boundary_layer(x, np.full_like(x, u), nu, model="laminar", ny=128)
    return x, u, nu, length, result


def test_laminar_limit_reproduces_blasius_friction(laminar_plate):
    x, u, nu, length, result = laminar_plate
    blasius = 0.664 / np.sqrt(u * x / nu)
    developed = x > 0.2 * length
    assert np.max(np.abs(result.cf[developed] / blasius[developed] - 1.0)) < 0.03


def test_laminar_limit_reproduces_blasius_profile(laminar_plate):
    x, u, nu, length, result = laminar_plate
    sol = blasius_solution()
    eta = result.y * np.sqrt(u / (nu * length))
    expected = np.interp(eta, sol["eta"], sol["fp"])
    mask = eta < 4.0
    assert np.max(np.abs(result.u[mask] / u - expected[mask])) < 0.03


def test_laminar_integral_thicknesses(laminar_plate):
    x, u, nu, length, result = laminar_plate
    delta_star = 1.7208 * math.sqrt(nu * length / u)
    theta = 0.664 * math.sqrt(nu * length / u)
    assert result.delta_star[-1] / delta_star == pytest.approx(1.0, abs=0.07)
    assert result.theta[-1] / theta == pytest.approx(1.0, abs=0.07)
    assert result.H[-1] == pytest.approx(1.7208 / 0.664, rel=0.03)


def test_mass_conservation_is_satisfied_to_round_off():
    nu, u = 1e-7, 1.0
    x = np.geomspace(0.002, 1.0, 100)
    result = march_boundary_layer(x, np.full_like(x, u), nu, model="laminar", ny=96)
    assert result.continuity_residual < 1e-6


@pytest.mark.parametrize("model", ["sst", "sa"])
def test_turbulent_flat_plate_matches_schlichting(model):
    nu, u, length = 1e-7, 1.0, 1.0
    x = np.geomspace(0.002, length, 120)
    result = march_boundary_layer(x, np.full_like(x, u), nu, model=model, ny=128,
                                  trip_x=0, picard=15, under_relaxation=0.5)
    mean_cf = float(np.trapezoid(result.cf, x) / (x[-1] - x[0]))
    reference = float(schlichting_mean_cf(u * length / nu))
    assert abs(mean_cf / reference - 1.0) < 0.05
    # the von Karman momentum identity is the independent momentum check
    developed = x > 0.3 * length
    residual = np.nanmax(np.abs(result.karman_residual[developed])) / np.nanmax(
        0.5 * result.cf[developed])
    assert residual < 0.03


@pytest.mark.parametrize("model", ["sst", "sa", "laminar"])
def test_flow_reversal_free_and_positive_eddy_viscosity(model):
    nu, u = 1e-7, 1.0
    x = np.geomspace(0.002, 1.0, 80)
    result = march_boundary_layer(x, np.full_like(x, u), nu, model=model, ny=64,
                                  trip_x=(None if model == "laminar" else 0))
    assert np.all(result.u >= -1e-12)
    assert np.all(result.nu_t >= 0.0)
    assert np.all(result.cf > 0.0)


# ---------------------------------------------------------------- geometry
def test_kvlcc2_geometry_is_verified_before_resistance():
    hull = build_kvlcc2_hull()
    coefficients = hull.coefficients_check
    assert coefficients["Cb"] == pytest.approx(0.8098, abs=5e-3)
    assert coefficients["Cm"] == pytest.approx(0.9982, abs=5e-3)
    assert coefficients["Cwp"] == pytest.approx(0.870, abs=5e-3)
    assert coefficients["LCB_pct"] == pytest.approx(-1.688, abs=0.05)


def test_wetted_surface_is_within_five_percent_of_published():
    hull = build_kvlcc2_hull()
    published = KVLCC2_S_FULL_SCALE / SIMMAN2008_KVLCC2.scale_ratio ** 2
    assert abs(hull.wetted_surface() / published - 1.0) < 0.06


def test_equivalent_body_keeps_the_sectional_area():
    hull = build_kvlcc2_hull()
    x = np.linspace(0.1 * hull.L, 0.9 * hull.L, 25)
    radius = equivalent_body_radius(hull, x)
    area = hull.area_shape(hull._xi(x))
    assert np.allclose(np.pi * radius ** 2, area)


def test_equivalent_body_surface_velocity_is_physical():
    hull = build_kvlcc2_hull()
    flow = equivalent_body_velocity(hull, 1.0, n=121)
    ue = flow["ue"]
    assert np.all(ue > 0.0)
    assert np.max(ue) < 1.6          # slender body: modest overspeed
    assert np.max(ue) > 1.0          # but a real acceleration somewhere


# --------------------------------------------------------------- campaign
def test_grid_convergence_index_recovers_a_known_order():
    exact = 0.003
    # error proportional to h^2 with h halving
    fine = exact + 1e-5
    medium = exact + 4e-5
    coarse = exact + 16e-5
    gci = grid_convergence_index(fine, medium, coarse, ratio=2.0)
    assert gci["observed_order"] == pytest.approx(2.0, abs=1e-6)
    assert gci["extrapolated"] == pytest.approx(exact, rel=1e-6)
    assert gci["gci_fine"] < 0.05


def test_froude_fit_recovers_synthetic_coefficients():
    fn = np.linspace(0.05, 0.15, 7)
    a, b, c = 1e-3, 0.05, 0.4
    ct = a + b * fn ** 2 + c * fn ** 4
    fit = fit_ct_froude(fn, ct)
    assert fit["r_squared"] == pytest.approx(1.0, abs=1e-12)
    assert fit["a"] == pytest.approx(a, rel=1e-8)
    assert fit["b"] == pytest.approx(b, rel=1e-8)


def test_criterion_c3_never_passes_with_a_withheld_anchor():
    points = {"sst": [{"Fn": 0.142, "C_V": 4.0e-3}],
              "sa": [{"Fn": 0.142, "C_V": 4.05e-3}]}
    verdict = criterion_c3(points)
    assert verdict["anchor_provenance"] == "withheld"
    assert verdict["verdict"] == "withheld"
    assert verdict["rows"]["sst"]["C_V"] == pytest.approx(4.0e-3)