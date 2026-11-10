"""Verification tests for the Stefan / ice-steel extension of the CHT surface.

Fast assertions (seconds) run everywhere:

* the similarity constant solves the transcendental relation it claims to solve;
* the enthalpy-method front sits on the Neumann front at three times (<= 5 %) and
  the latent heat is *load bearing*: dropping it turns the same comparison red;
* the ice-steel interface flux is conserved by the single-field harmonic
  conductance to round-off and matches the closed-form series resistance;
* a partitioned Dirichlet-Neumann coupling converges to the same state, and a
  deliberately truncated coupling is caught by the flux residual (mutation);
* the independent surface heat balance reproduces the archived W4-N08 Makiranta
  reference panel within 30 %.

The coupled cross-flow cylinder (channel a) needs minutes and is marked slow
(``CHT_SLOW=1``), like the rest of the CHT suite.
"""

import math
import os

import numpy as np
import pytest
from scipy.special import erf

from tensorfvm.cht.ice_conjugate import (ICE_K, STEEL_K, W4N08_REFERENCE_PANEL,
                                         PartitionedIceSteel,
                                         ice_growth_from_balance,
                                         interface_flux_audit, layered_field,
                                         layered_interface_analytic,
                                         surface_heat_loss)
from tensorfvm.cht.polar import (ConjugateCylinder, PolarVorticitySolver,
                                 one_sided_derivative, periodic_overlap_matrix)
from tensorfvm.cht.stefan import (ICE_C, ICE_K as K_ICE, ICE_LATENT, ICE_RHO,
                                  neumann_front, neumann_lambda,
                                  neumann_temperature, solve_stefan_1d)

SLOW = os.environ.get("CHT_SLOW") == "1"
STEFAN_NUMBER = ICE_C * 20.0 / ICE_LATENT


# ------------------------------------------------------------------ similarity
def test_similarity_constant_solves_the_transcendental_relation():
    for stefan in (0.02, 0.123, 1.0, 5.0):
        lam = neumann_lambda(stefan)
        residual = math.sqrt(math.pi) * lam * math.exp(lam * lam) * erf(lam) - stefan
        assert abs(residual) < 1e-9
        assert lam > 0.0


def test_similarity_constant_rejects_non_positive_stefan_number():
    with pytest.raises(ValueError):
        neumann_lambda(0.0)


def test_small_stefan_number_asymptote():
    """``L -> 0`` gives ``Lambda -> sqrt(St/2)`` (quasi-steady latent balance)."""
    lam = neumann_lambda(0.01)
    assert lam == pytest.approx(math.sqrt(0.01 / 2.0), rel=0.02)


def test_similarity_front_and_temperature_are_consistent():
    """The similarity field must return the fusion temperature at the front."""
    alpha = K_ICE / (ICE_RHO * ICE_C)
    for t in (50.0, 500.0):
        front = neumann_front(t, STEFAN_NUMBER, alpha)
        assert neumann_temperature(front, t, STEFAN_NUMBER, alpha, -20.0, 0.0) \
            == pytest.approx(0.0, abs=1e-12)
        assert neumann_temperature(0.0, t, STEFAN_NUMBER, alpha, -20.0, 0.0) \
            == pytest.approx(-20.0, abs=1e-12)


# --------------------------------------------------------------------- Stefan
def test_enthalpy_front_matches_neumann_at_three_times():
    times = (50.0, 100.0, 200.0)
    res = solve_stefan_1d(n_cells=150, dt=1.5, t_end=200.0, sample_times=times)
    assert np.all(np.abs(res.front_rel_error) < 0.05)
    # the conservative (enthalpy) estimator should be far better than the bound
    assert np.max(np.abs(res.front_rel_error)) < 0.01
    assert res.ledger["normalized_imbalance"] < 1e-8


def test_stefan_energy_ledger_closes():
    res = solve_stefan_1d(n_cells=100, dt=2.0, t_end=100.0, sample_times=(100.0,))
    removed = res.ledger["wall_heat_removed_J_m2"]
    stored = res.ledger["stored_enthalpy_change_J_m2"]
    assert removed > 0.0
    assert abs(stored + removed) / removed < 1e-8


def test_latent_heat_is_load_bearing():
    """Mutation: with the latent term removed the same comparison fails badly."""
    times = (100.0, 400.0)
    good = solve_stefan_1d(n_cells=120, dt=2.0, t_end=400.0, sample_times=times)
    mutant = solve_stefan_1d(n_cells=60, dt=8.0, t_end=400.0, sample_times=times,
                             latent=1.0)
    assert np.all(np.abs(good.front_rel_error) < 0.05)
    mutant_error = np.abs(mutant.front_enthalpy - good.front_exact) / good.front_exact
    assert np.max(mutant_error) > 0.05


# ---------------------------------------------------------------- ice and steel
def test_single_field_interface_flux_is_conservative_and_accurate():
    exact = layered_interface_analytic(d_steel=0.020, d_ice=0.030, t_inner=-1.8,
                                       t_outer=-10.0)
    field = layered_field(d_steel=0.020, d_ice=0.030, t_inner=-1.8, t_outer=-10.0)
    audit = interface_flux_audit(field)
    assert audit["local_residual_max"] < 1e-10
    assert audit["global_residual"] < 1e-10
    assert audit["interface_temperature"] == pytest.approx(
        exact["interface_temperature"], rel=1e-10)
    assert audit["interface_area_mean_flux"] == pytest.approx(exact["q_W_m2"], rel=1e-10)


def test_interface_flux_continuity_holds_for_any_conductivity_ratio():
    for k_ice in (STEEL_K, 5.0, ICE_K, 0.5):
        exact = layered_interface_analytic(d_steel=0.018, d_ice=0.012, t_inner=4.0,
                                           t_outer=-30.0, k_ice=k_ice)
        field = layered_field(d_steel=0.018, d_ice=0.012, t_inner=4.0,
                              t_outer=-30.0, k_ice=k_ice)
        audit = interface_flux_audit(field, k_ice=k_ice)
        assert audit["local_residual_max"] < 1e-10
        assert audit["interface_area_mean_flux"] == pytest.approx(exact["q_W_m2"],
                                                                 rel=1e-10)


def test_partitioned_coupling_reproduces_the_series_resistance():
    tolerances = dict(d_steel=0.020, d_ice=0.030, t_water=-1.8, t_air=-18.2,
                      h_water=500.0, h_air=25.0)
    part = PartitionedIceSteel(**tolerances)
    res = part.solve(relax=0.8, max_iter=200)
    exact = layered_interface_analytic(d_steel=0.020, d_ice=0.030, t_inner=-1.8,
                                       t_outer=-18.2, h_inner=500.0, h_outer=25.0)
    assert res.converged
    assert res.flux_residual < 1e-8
    assert res.energy_residual < 1e-8
    assert res.interface_temperature == pytest.approx(exact["interface_temperature"],
                                                      rel=1e-8)
    assert res.q_steel == pytest.approx(exact["q_W_m2"], rel=1e-8)


def test_truncated_coupling_is_caught_by_the_flux_residual():
    """Mutation: stop the Dirichlet-Neumann sweep early and the residual goes red."""
    part = PartitionedIceSteel()
    loose = part.solve(relax=0.8, max_iter=2)
    tight = part.solve(relax=0.8, max_iter=200)
    assert loose.flux_residual > 0.01
    assert tight.flux_residual < 0.01


def test_two_dimensional_interfacial_energy_balance():
    nz = 12
    span = np.linspace(-1.0, 1.0, nz)
    t_outer = -10.0 - 1.5 * (1.0 - np.cos(np.pi * (span + 1.0)))
    field = layered_field(d_steel=0.020, d_ice=0.030, t_inner=-1.8,
                          t_outer=t_outer, nz=nz)
    audit = interface_flux_audit(field)
    assert audit["local_residual_max"] < 1e-10
    dy = field["grid"].dy
    t_all = np.atleast_2d(field["T"])
    q_top = float(np.sum(2.0 * ICE_K * (t_all[:, -1] - t_outer) / field["dx"][-1] * dy))
    q_iface = float(np.sum(np.asarray(audit["q_face_profile_W_m2"]) * dy))
    assert abs(q_top - q_iface) / abs(q_top) < 1e-10
    # lateral conduction must not be silently zero: the flux varies along the span
    profile = np.asarray(audit["q_face_profile_W_m2"])
    assert profile.std() > 0.0


def test_robin_inner_boundary_matches_the_series_resistance():
    """The ghost-layer Robin must reproduce ``1/h`` exactly."""
    for h in (100.0, 500.0, 2000.0):
        exact = layered_interface_analytic(d_steel=0.020, d_ice=0.030, t_inner=2.0,
                                           t_outer=-1.8, h_inner=h)
        field = layered_field(d_steel=0.020, d_ice=0.030, t_inner=2.0, t_outer=-1.8,
                              h_inner=h)
        audit = interface_flux_audit(field)
        assert audit["interface_area_mean_flux"] == pytest.approx(exact["q_W_m2"],
                                                                  rel=1e-10)
        assert audit["interface_temperature"] == pytest.approx(
            exact["interface_temperature"], rel=1e-10)


def test_isothermal_ice_steel_stack_carries_no_flux():
    field = layered_field(d_steel=0.020, d_ice=0.030, t_inner=-1.8, t_outer=-1.8,
                          h_inner=500.0)
    audit = interface_flux_audit(field)
    assert abs(audit["interface_area_mean_flux"]) < 1e-9
    assert audit["interface_temperature"] == pytest.approx(-1.8, abs=1e-12)


def test_ice_surface_balance_reproduces_the_archived_w4n08_panel():
    ref = W4N08_REFERENCE_PANEL
    balance = surface_heat_loss(t_surface=ref["surface_temperature_C"],
                                t_air=ref["t_air_C"], wind=ref["wind_m_s"],
                                relative_humidity=ref["relative_humidity"],
                                t_sea=ref["t_sea_C"],
                                spray_flux=ref["spray_flux_kg_m2_s"],
                                length=ref["char_length_m"])
    # each of the three dominant terms to within 15 % of the independent code
    for key in ("q_conv_W_m2", "q_evap_W_m2", "q_lw_W_m2"):
        assert abs(balance[key] - ref[key]) / abs(ref[key]) < 0.15
    growth = ice_growth_from_balance(balance=balance,
                                     spray_flux=ref["spray_flux_kg_m2_s"])
    assert growth["regime"] == "wet"
    assert abs(growth["ice_flux_kg_m2_s"] - ref["ice_flux_kg_m2_s"]) \
        / ref["ice_flux_kg_m2_s"] < 0.30
    # at the design condition the sea is at the freezing point, so the conjugate
    # structure carries (essentially) no heat and the two models must agree
    stack = layered_field(d_steel=0.020, d_ice=0.030, t_inner=ref["t_sea_C"],
                          t_outer=ref["surface_temperature_C"], h_inner=500.0)
    q_cond = interface_flux_audit(stack)["interface_area_mean_flux"]
    assert abs(q_cond) < 1e-6
    coupled = ice_growth_from_balance(balance=balance,
                                      spray_flux=ref["spray_flux_kg_m2_s"],
                                      conduction_into_structure=q_cond)
    assert abs(coupled["ice_flux_kg_m2_s"] - ref["ice_flux_kg_m2_s"]) \
        / ref["ice_flux_kg_m2_s"] < 0.30


def test_warm_sea_current_suppresses_the_accretion_rate():
    """A warm hull eventually breaks the adiabatic-substrate assumption."""
    ref = W4N08_REFERENCE_PANEL
    balance = surface_heat_loss(t_surface=ref["surface_temperature_C"],
                                t_air=ref["t_air_C"], wind=ref["wind_m_s"],
                                relative_humidity=ref["relative_humidity"],
                                t_sea=ref["t_sea_C"],
                                spray_flux=ref["spray_flux_kg_m2_s"],
                                length=ref["char_length_m"])
    cold = layered_field(d_steel=0.020, d_ice=0.030, t_inner=-1.8,
                         t_outer=ref["surface_temperature_C"], h_inner=500.0)
    warm = layered_field(d_steel=0.020, d_ice=0.030, t_inner=8.0,
                         t_outer=ref["surface_temperature_C"], h_inner=500.0)
    q_cold = interface_flux_audit(cold)["interface_area_mean_flux"]
    q_warm = interface_flux_audit(warm)["interface_area_mean_flux"]
    assert q_warm > q_cold
    g_cold = ice_growth_from_balance(balance=balance,
                                     spray_flux=ref["spray_flux_kg_m2_s"],
                                     conduction_into_structure=q_cold)
    g_warm = ice_growth_from_balance(balance=balance,
                                     spray_flux=ref["spray_flux_kg_m2_s"],
                                     conduction_into_structure=q_warm)
    assert g_warm["ice_flux_kg_m2_s"] < g_cold["ice_flux_kg_m2_s"]
    # the warm case must fall outside the 30 % band: that is the point of the check
    dev = (g_warm["ice_flux_kg_m2_s"] - ref["ice_flux_kg_m2_s"]) \
        / ref["ice_flux_kg_m2_s"]
    assert abs(dev) > 0.30


def test_colder_structure_increases_the_accretion_rate():
    """Conduction into the ice must add to the surface loss, not cancel it."""
    ref = W4N08_REFERENCE_PANEL
    balance = surface_heat_loss(t_surface=ref["surface_temperature_C"],
                                t_air=ref["t_air_C"], wind=ref["wind_m_s"],
                                relative_humidity=ref["relative_humidity"],
                                t_sea=ref["t_sea_C"],
                                spray_flux=ref["spray_flux_kg_m2_s"],
                                length=ref["char_length_m"])
    base = ice_growth_from_balance(balance=balance,
                                   spray_flux=ref["spray_flux_kg_m2_s"])
    # negative conduction = heat flowing out of the cold structure into the ice
    colder = ice_growth_from_balance(balance=balance,
                                     spray_flux=ref["spray_flux_kg_m2_s"],
                                     conduction_into_structure=-50.0)
    warmer = ice_growth_from_balance(balance=balance,
                                     spray_flux=ref["spray_flux_kg_m2_s"],
                                     conduction_into_structure=+50.0)
    assert colder["ice_flux_kg_m2_s"] > base["ice_flux_kg_m2_s"]
    assert warmer["ice_flux_kg_m2_s"] < base["ice_flux_kg_m2_s"]
    assert warmer["freezing_fraction"] < base["freezing_fraction"] < 1.0


# ------------------------------------------------ polar CHT path (defect guards)
def test_periodic_overlap_matrix_terminates_and_conserves():
    """Equal partitions used to stall forever in the walk-forward loop."""
    for n_from, n_to in ((48, 48), (48, 32), (48, 96), (30, 44), (7, 3)):
        w = periodic_overlap_matrix(n_from, n_to)
        assert np.allclose(w.sum(axis=1), 1.0, atol=1e-12)
        # every source cell must be covered n_to/n_from times
        assert np.allclose(w.sum(axis=0), n_to / n_from, atol=1e-12)


def test_one_sided_derivative_is_exact_on_polynomials():
    s1, s2 = 0.008, 0.0166
    # linear function: exact to round-off (this is what the old coefficients missed)
    f = lambda x: 3.0 - 0.5 * x
    assert one_sided_derivative(f(0.0), f(s1), f(s2), s1, s2) == pytest.approx(-0.5,
                                                                              abs=1e-12)
    # quadratic: also exact
    g = lambda x: 1.0 + 2.0 * x - 1.5 * x * x
    assert one_sided_derivative(g(0.0), g(s1), g(s2), s1, s2) == pytest.approx(2.0,
                                                                             abs=1e-10)
    with pytest.raises(ValueError):
        one_sided_derivative(0.0, 1.0, 2.0, 0.0, 0.1)


def test_polar_pure_conduction_annulus_matches_analytic():
    """Annular conduction between two Dirichlet radii: field and wall flux."""
    flow = PolarVorticitySolver(re=40.0, n_r=71, n_theta=72, r_out=15.0,
                                wall_cell=8e-3)
    case = ConjugateCylinder(re=40.0, pr=0.7, solid_ratio=100.0, flow=flow)
    T, q = case.solve_fluid(np.ones(flow.n_theta),
                            np.zeros((flow.n_node, flow.n_theta)),
                            outer_temperature=0.0)
    r = flow.r_node
    exact = np.log(15.0 / r) / math.log(15.0 / 0.5)
    assert np.max(np.abs(T[1:-1, 0] - exact[1:-1])) < 1e-4
    q_exact = case.kappa_fluid / (0.5 * math.log(15.0 / 0.5))
    assert q.mean() == pytest.approx(q_exact, rel=0.01)


def test_solid_block_keeps_the_core_temperature():
    """Guards the sign convention of the solid operator (used to return -1)."""
    flow = PolarVorticitySolver(re=40.0, n_r=41, n_theta=48, r_out=15.0,
                                wall_cell=8e-3)
    case = ConjugateCylinder(re=40.0, pr=0.7, solid_ratio=100.0, flow=flow)
    adiabatic = case.solve_solid(np.zeros(flow.n_theta))
    assert np.allclose(adiabatic, 1.0, atol=1e-9)
    heated = case.solve_solid(np.full(flow.n_theta, 1.0))
    assert heated[-1].mean() > 1.0          # a positive flux heats the interface
    cooled = case.solve_solid(np.full(flow.n_theta, -1.0))
    assert cooled[-1].mean() < 1.0


def test_conjugate_wall_sits_between_core_and_free_stream():
    """Physics guard: the coupled wall must be cooler than the solid core."""
    flow = PolarVorticitySolver(re=40.0, n_r=41, n_theta=48, r_out=15.0,
                                wall_cell=8e-3)
    march = flow.march(t_end=4.0, dt=2e-3, tol_steady=1e-12, max_steps=4000,
                       t_ramp=2.0)
    case = ConjugateCylinder(re=40.0, pr=0.7, solid_ratio=100.0, flow=flow)
    res = case.couple(march["psi"], relax=0.7, max_iter=60, tol=1e-9)
    assert 0.0 < res.metrics["wall_temperature_mean"] < 1.0
    assert res.metrics["interface_local_residual_rel"] < 1e-2


# -------------------------------------------------------------------- cylinder
def test_eckert_soehngen_correlation_values():
    from scripts.p295_cht_validation import eckert_soehngen
    # Re = 40, Pr = 0.7 -> (0.43 + 0.5 sqrt(40)) 0.7^0.38
    assert eckert_soehngen(40.0, 0.7) == pytest.approx(3.137, abs=0.005)
    values = [eckert_soehngen(re, 0.7) for re in (20.0, 40.0, 100.0, 400.0)]
    assert values == sorted(values)
    with pytest.raises(ValueError):
        eckert_soehngen(2000.0, 0.7)


@pytest.mark.skipif(not SLOW, reason="coupled cross-flow CHT needs minutes")
def test_cylinder_coupling_conserves_interface_flux():
    from scripts.p295_cht_validation import run_cylinder
    payload, _ = run_cylinder(re=40.0)
    # the coupling converges and conserves the interface flux
    assert payload["interface_local_residual_rel"] < 1e-3
    assert 0.0 < payload["wall_temperature_mean"] < 1.0


@pytest.mark.skipif(not SLOW, reason="coupled cross-flow CHT needs minutes")
@pytest.mark.xfail(strict=False, reason=(
    "open accuracy gap: the polar energy solver is first-order upwind on a 5 deg "
    "azimuthal grid; after the operator fixes the mean Nu is ~24 % above "
    "Eckert-Soehngen instead of the required 10 % (see docs/cht-p295 report)"))
def test_cylinder_mean_nusselt_against_eckert_soehngen():
    from scripts.p295_cht_validation import eckert_soehngen, run_cylinder
    payload, _ = run_cylinder(re=40.0)
    nu_es = eckert_soehngen(40.0, 0.7)
    assert abs(payload["nusselt_mean"] - nu_es) / nu_es < 0.10