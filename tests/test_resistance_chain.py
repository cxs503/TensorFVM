"""Tests for the unified resistance chain (friction lines, allowance regimes,
form-factor adjudication, validation-uncertainty budget).

Reference values: the ``holtrop_*`` expectations below were produced by the
ship platform's authoritative implementation
(``backend/physics/holtrop/mennen.py``, the single Holtrop source of truth after
the resistance-family merge) for the KVLCC2 model particulars.  They are
hardcoded here on purpose: this suite audits the TensorFVM port against a
different implementation, so it must not import it.
"""
import math

import pytest

from tensorfvm.resistance_chain import (
    DEFAULT_SHIP_ROUGHNESS_KS, LAMINAR_RE_LIMIT, MODEL_ROUGHNESS_KS,
    HullPrincipal, compare_form_factor, correlation_allowance_table,
    extrapolate_to_ship, form_factor_from_double_body, friction_band,
    holtrop_correlation_allowance, holtrop_form_factor, holtrop_lr,
    holtrop_wetted_surface, ittc57_cf, ittc78_delta_cf, reynolds_table, schoenherr_cf, validation_uncertainty_budget, viscous_chain,
)

KVLCC2_MODEL = HullPrincipal(name="KVLCC2_model", L=7.0, B=0.936, T=0.342,
                             Cb=0.708, Cm=0.985, Cwp=0.808, lcb_pct=-1.688)
KVLCC2_SHIP = HullPrincipal(name="KVLCC2_ship", L=320.0, B=58.0, T=20.8,
                            Cb=0.8098, Cm=0.9982, Cwp=0.87, lcb_pct=-3.0)
#: platform reference: Holtrop (1+k) for the KVLCC2 model particulars
HOLTROP_ONE_PLUS_K = 1.1867361694951244
#: platform reference: Holtrop (1984) wetted surface [m^2]
HOLTROP_S = 8.705806048081895
#: platform reference: Holtrop (1984) correlation allowance [-]
HOLTROP_CA = 0.0007908600922694607
#: KVLCC2 model design speed from Fn = 0.142, L = 7 m
V_DESIGN = 0.142 * math.sqrt(9.80665 * 7.0)


def test_ittc57_closed_form_value_is_exact_at_one_million():
    # 0.075 / (6 - 2)^2 == 4.6875e-3 exactly
    assert ittc57_cf(1e6) == pytest.approx(4.6875e-3, rel=0, abs=1e-18)
    assert ittc57_cf(1e7) == pytest.approx(0.075 / 25.0, rel=1e-15)


def test_three_friction_laws_agree_within_the_published_band():
    for re in (1e6, 1e7, 1e8, 1e9):
        band = friction_band(re)
        assert band["spread"] < 0.08, (re, band)
    # they are different historical fits, not one identity: the disagreement is
    # ~6 % at 1e6 and collapses as the log law is approached
    assert friction_band(1e6)["values"]["schoenherr"] == pytest.approx(
        friction_band(1e6)["values"]["ittc57"], rel=0.08)
    assert friction_band(1e8)["values"]["schoenherr"] == pytest.approx(
        friction_band(1e8)["values"]["ittc57"], rel=0.02)
    assert friction_band(1e9)["spread"] < friction_band(1e6)["spread"]


def test_schoenherr_residual_is_actually_zero():
    for re in (5e5, 1e6, 3.3e6, 7.2e6, 1e8):
        cf = schoenherr_cf(re)
        assert 0.242 / math.sqrt(cf) - math.log10(re * cf) == pytest.approx(0, abs=1e-12)


def test_turbulent_laws_reject_laminar_reynolds():
    with pytest.raises(ValueError):
        ittc57_cf(LAMINAR_RE_LIMIT * 0.5)
    with pytest.raises(ValueError):
        schoenherr_cf(-1.0)
    with pytest.raises(TypeError):
        ittc57_cf("1e6")


def test_ittc78_allowance_matches_the_published_formula_at_full_scale():
    # [105 (ks/L)^(1/3) - 0.64] * 1e-3 with ks = 150 um and L = 320 m
    expected = (105.0 * (150e-6 / 320.0) ** (1.0 / 3.0) - 0.64) * 1e-3
    assert ittc78_delta_cf(150e-6, 320.0) == pytest.approx(expected, rel=1e-15)
    assert ittc78_delta_cf(150e-6, 320.0) == pytest.approx(1.7565e-4, rel=1e-3)


def test_ittc78_allowance_floors_at_zero_for_smooth_surfaces():
    # a 1 um roughness on a 320 m ship drives the raw formula negative
    assert (105.0 * (1e-6 / 320.0) ** (1.0 / 3.0) - 0.64) < 0.0
    assert ittc78_delta_cf(1e-6, 320.0) == 0.0
    assert ittc78_delta_cf(0.0, 100.0) == 0.0


def test_ittc78_allowance_has_the_cube_root_roughness_scaling():
    low = ittc78_delta_cf(150e-6, 320.0) + 0.64e-3
    high = ittc78_delta_cf(150e-6 * 8.0, 320.0) + 0.64e-3
    assert high / low == pytest.approx(2.0, rel=1e-12)


def test_model_scale_allowance_is_much_larger_than_ship_scale():
    # (ks/L) is two orders of magnitude larger on a 7 m model: the ITTC-78
    # allowance is a *ship* convention and must not be applied at model scale.
    model = ittc78_delta_cf(DEFAULT_SHIP_ROUGHNESS_KS, KVLCC2_MODEL.L)
    ship = ittc78_delta_cf(DEFAULT_SHIP_ROUGHNESS_KS, KVLCC2_SHIP.L)
    assert model > 10.0 * ship


def test_holtrop_form_factor_reproduces_the_platform_value():
    assert holtrop_lr(7.0, 0.708 / 0.985, -1.688) == pytest.approx(1.6967663161829167, rel=1e-12)
    assert holtrop_form_factor(KVLCC2_MODEL) == pytest.approx(HOLTROP_ONE_PLUS_K, rel=1e-12)


def test_holtrop_form_factor_grows_with_fullness():
    slender = HullPrincipal(name="slender", L=7.0, B=0.936, T=0.342, Cb=0.56,
                            Cm=0.85, Cwp=0.70, lcb_pct=-1.0)
    assert holtrop_form_factor(KVLCC2_MODEL) > holtrop_form_factor(slender)
    assert holtrop_form_factor(KVLCC2_SHIP) > holtrop_form_factor(KVLCC2_MODEL)


def test_holtrop_wetted_surface_and_allowance_match_platform_reference():
    assert holtrop_wetted_surface(KVLCC2_MODEL) == pytest.approx(HOLTROP_S, rel=1e-12)
    assert holtrop_correlation_allowance(KVLCC2_MODEL) == pytest.approx(HOLTROP_CA, rel=1e-12)


def test_hull_principal_rejects_inconsistent_coefficients():
    with pytest.raises(ValueError):
        HullPrincipal(name="bad", L=7.0, B=0.9, T=0.3, Cb=0.9, Cm=0.8, Cwp=0.8)
    with pytest.raises(ValueError):
        HullPrincipal(name="bad", L=0.0, B=0.9, T=0.3, Cb=0.7, Cm=0.9, Cwp=0.8)


def test_double_body_decomposition_recovers_the_form_factor_exactly():
    q, s = 500.0, 8.7
    cf, cp = 3.2e-3, 0.6e-3
    split = form_factor_from_double_body(pressure_force_n=cp * q * s,
                                        shear_force_n=cf * q * s,
                                        dynamic_pressure_pa=q, wetted_area_m2=s)
    assert split["C_F"] == pytest.approx(cf, rel=1e-12)
    assert split["C_Pv"] == pytest.approx(cp, rel=1e-12)
    assert split["one_plus_k"] == pytest.approx((cf + cp) / cf, rel=1e-12)
    assert split["C_V"] == pytest.approx(cf + cp, rel=1e-12)
    assert split["pressure_fraction"] == pytest.approx(cp / (cf + cp), rel=1e-12)


def test_double_body_decomposition_rejects_a_sign_error():
    # friction must point downstream; a flipped wall normal shows up here
    with pytest.raises(ValueError):
        form_factor_from_double_body(1.0, -1.0, 500.0, 8.7)
    with pytest.raises(ValueError):
        form_factor_from_double_body(1.0, 1.0, 0.0, 8.7)
    with pytest.raises(ValueError):
        form_factor_from_double_body(1.0, 1.0, 500.0, -8.7)
    # a *negative* viscous pressure drag is physical (streamlined body) and must
    # not be rejected; it just lowers (1+k) below 1
    split = form_factor_from_double_body(-0.5e-3 * 500.0 * 8.7, 3.2e-3 * 500.0 * 8.7,
                                        500.0, 8.7)
    assert split["one_plus_k"] < 1.0


def test_form_factor_adjudication_gate_is_five_percent():
    holtrop = holtrop_form_factor(KVLCC2_MODEL)
    inside = compare_form_factor(holtrop * 1.04, KVLCC2_MODEL)
    outside = compare_form_factor(holtrop * 1.06, KVLCC2_MODEL)
    assert inside["passes"] and inside["absolute_deviation"] < 0.05
    assert not outside["passes"] and outside["absolute_deviation"] > 0.05
    assert inside["one_plus_k_holtrop"] == pytest.approx(holtrop, rel=1e-15)


def test_chain_counts_the_allowance_exactly_once():
    ittc = viscous_chain(KVLCC2_SHIP, 8.0, regime="ittc78_roughness",
                         ks=DEFAULT_SHIP_ROUGHNESS_KS)
    holtrop = viscous_chain(KVLCC2_SHIP, 8.0, regime="holtrop_ca")
    bare = viscous_chain(KVLCC2_SHIP, 8.0, regime="none")
    cf_ship = ittc57_cf(ittc.reynolds)
    for result, allowance in ((ittc, ittc78_delta_cf(DEFAULT_SHIP_ROUGHNESS_KS, KVLCC2_SHIP.L)),
                              (holtrop, holtrop_correlation_allowance(KVLCC2_SHIP)),
                              (bare, 0.0)):
        assert result.allowance == pytest.approx(allowance, rel=1e-15)
        assert result.C_F_wetted == pytest.approx(result.one_plus_k * cf_ship, rel=1e-12)
        assert result.C_T == pytest.approx(result.C_F_wetted + allowance, rel=1e-15)
        # the two conventions are never added together
        assert result.C_T < result.C_F_wetted + ittc78_delta_cf(
            DEFAULT_SHIP_ROUGHNESS_KS, KVLCC2_SHIP.L) + holtrop_correlation_allowance(
            KVLCC2_SHIP)


def test_chain_refuses_the_double_counted_convention():
    with pytest.raises(ValueError):
        # ks says "roughness regime", regime says "shape regression": ambiguous
        viscous_chain(KVLCC2_SHIP, 8.0, regime="holtrop_ca", ks=150e-6)
    with pytest.raises(ValueError):
        # the roughness regime needs a roughness
        viscous_chain(KVLCC2_SHIP, 8.0, regime="ittc78_roughness")
    with pytest.raises(ValueError):
        viscous_chain(KVLCC2_SHIP, 8.0, regime="both")


def test_chain_records_which_form_factor_source_was_used():
    regression = viscous_chain(KVLCC2_MODEL, V_DESIGN, regime="none")
    cfd = viscous_chain(KVLCC2_MODEL, V_DESIGN, regime="none", one_plus_k=1.20,
                        cf_law="schoenherr")
    assert regression.one_plus_k_source == "holtrop_regression"
    assert cfd.one_plus_k_source == "user_supplied"
    assert cfd.one_plus_k == 1.20 and cfd.cf_law == "schoenherr"
    assert cfd.reynolds == pytest.approx(V_DESIGN * 7.0 / 1.19e-6, rel=1e-12)


def test_reynolds_table_is_log_spaced_and_allowance_is_reynolds_independent():
    rows = reynolds_table(re_min=1e6, re_max=1e9, n_points=8, L=7.0)
    assert len(rows) == 8
    assert rows[0]["reynolds"] == pytest.approx(1e6, rel=1e-12)
    assert rows[-1]["reynolds"] == pytest.approx(1e9, rel=1e-9)
    ratios = [rows[i + 1]["reynolds"] / rows[i]["reynolds"] for i in range(7)]
    assert max(ratios) - min(ratios) < 1e-9
    allowances = {row["ittc78_delta_cf"] for row in rows}
    assert len(allowances) == 1          # dCf depends on ks/L only, not on Re
    assert allowances.pop() == pytest.approx(ittc78_delta_cf(150e-6, 7.0), rel=1e-15)
    for row in rows:
        assert row["ittc57_cf"] == pytest.approx(ittc57_cf(row["reynolds"]), rel=1e-15)
        assert row["schoenherr_cf"] == pytest.approx(schoenherr_cf(row["reynolds"]), rel=1e-14)


def test_allowance_table_flags_the_smooth_surface_floor():
    rows = correlation_allowance_table([7.0, 320.0], [1e-6, 150e-6])
    assert len(rows) == 4
    by_key = {(r["L_m"], r["ks_m"]): r for r in rows}
    assert by_key[(320.0, 1e-6)]["floored_to_zero"] is True
    assert by_key[(7.0, 150e-6)]["floored_to_zero"] is False
    assert by_key[(320.0, 150e-6)]["delta_cf"] == pytest.approx(
        ittc78_delta_cf(150e-6, 320.0), rel=1e-15)


def test_unified_chain_brings_the_validation_uncertainty_inside_the_gate():
    # the budget is evaluated where the double-body CFD lives: model scale, with
    # a CFD form factor 1 % above the regression value
    budget = validation_uncertainty_budget(
        KVLCC2_MODEL, V_DESIGN, regime="ittc78_roughness",
        ks=50e-6, one_plus_k_cfd=holtrop_form_factor(KVLCC2_MODEL) * 1.01, u_cfd=0.02)
    assert budget["U_V"] <= 0.05 and budget["passes"] is True
    # before unification the doubled allowance dominates the budget
    assert budget["U_V_before_unification"] > 0.05
    assert budget["U_V_before_unification"] > 2.0 * budget["U_V"]
    assert budget["unmerged_double_counted_allowance"] == pytest.approx(
        ittc78_delta_cf(50e-6, KVLCC2_MODEL.L)
        + holtrop_correlation_allowance(KVLCC2_MODEL), rel=1e-15)
    assert set(budget["components"]) == {"friction", "form", "allowance", "cfd"}
    # a ship-scale chain in the Holtrop convention also closes the gate
    ship = validation_uncertainty_budget(KVLCC2_SHIP, 8.0, regime="holtrop_ca",
                                         one_plus_k_cfd=holtrop_form_factor(KVLCC2_SHIP))
    assert ship["U_V"] <= 0.05


def test_uncertainty_budget_degrades_when_the_form_factor_disagrees():
    tight = validation_uncertainty_budget(KVLCC2_SHIP, 8.0, regime="holtrop_ca",
                                          one_plus_k_cfd=holtrop_form_factor(KVLCC2_SHIP) * 1.01)
    loose = validation_uncertainty_budget(KVLCC2_SHIP, 8.0, regime="holtrop_ca",
                                          one_plus_k_cfd=holtrop_form_factor(KVLCC2_SHIP) * 1.15)
    assert loose["U_V"] > tight["U_V"]
    assert tight["form_factor_source"] == "cfd_vs_holtrop"
    withheld = validation_uncertainty_budget(KVLCC2_SHIP, 8.0, regime="holtrop_ca")
    assert withheld["cfd_discretisation_withheld"] is True
    assert withheld["form_factor_source"] == "prior_band"


def test_model_to_ship_extrapolation_is_self_consistent():
    v_kn = 8.0
    speed = v_kn * 0.5144444
    one_plus_k = holtrop_form_factor(KVLCC2_SHIP)
    re_model = speed * KVLCC2_MODEL.L / 1.139e-6
    re_ship = speed * KVLCC2_SHIP.L / 1.19e-6
    c_r_true = 3.0e-4
    c_t_model = one_plus_k * ittc57_cf(re_model) + c_r_true
    out = extrapolate_to_ship(c_t_model, KVLCC2_MODEL, KVLCC2_SHIP, v_kn,
                              one_plus_k=one_plus_k)
    assert out["C_R"] == pytest.approx(c_r_true, rel=1e-12)
    assert out["C_TS"] == pytest.approx(
        one_plus_k * ittc57_cf(re_ship) + c_r_true
        + ittc78_delta_cf(DEFAULT_SHIP_ROUGHNESS_KS, KVLCC2_SHIP.L), rel=1e-12)
    assert out["reynolds_model"] == pytest.approx(re_model, rel=1e-12)
    assert out["C_TS"] < out["C_TM"]          # friction drops faster than the allowance grows
    assert out["C_AA"] == 0.0
    assert extrapolate_to_ship(c_t_model, KVLCC2_MODEL, KVLCC2_SHIP, v_kn,
                               one_plus_k=one_plus_k, include_air=True)["C_AA"] > 0.0


def test_ittc78_allowance_is_a_ship_scale_convention_not_a_model_one():
    # the formula only floors for ks/L below (0.64/105)^3 ~ 2.3e-7, which no
    # model-scale test can reach: applying it to a 7 m model would add ~1.4e-3
    # to Cf, i.e. a 36 % resistance increase
    threshold = (0.64 / 105.0) ** 3
    assert ittc78_delta_cf(threshold * KVLCC2_SHIP.L * 0.99, KVLCC2_SHIP.L) == 0.0
    model_value = ittc78_delta_cf(MODEL_ROUGHNESS_KS, KVLCC2_MODEL.L)
    ship_value = ittc78_delta_cf(DEFAULT_SHIP_ROUGHNESS_KS, KVLCC2_SHIP.L)
    assert model_value > 5.0 * ship_value
    cf_model = ittc57_cf(V_DESIGN * KVLCC2_MODEL.L / 1.139e-6)
    assert model_value / cf_model > 0.3
    assert MODEL_ROUGHNESS_KS < DEFAULT_SHIP_ROUGHNESS_KS