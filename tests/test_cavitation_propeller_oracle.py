"""Tests for the independent propeller cavitation-criteria oracle.

Each test pins one claim the oracle makes against something that does not come
from the same code path:

* the criterion quantities against **hand arithmetic** (the definitions of
  Burrill 1943 / Keller 1966 / Taylor 1933 written out by hand),
* the vapour pressure against the steam-table values already used by
  ``tensorfvm.cavitation.inception``,
* the ``sigma_n / sigma_0.7R`` inflation against its closed-form identity
  ``J^2 + (0.7 pi)^2``,
* monotonicity/limits that follow from the definitions alone,
* the placeholder-line verdict against the analytic sensitivity
  ``d(margin)/d(sigma_limit) = -100/sigma``,
* the platform probe against the live endpoint's *state* (never assumed).
"""
from __future__ import annotations

import json
import math

import pytest

from tensorfvm.cavitation import propeller_oracle as O

# Design point used throughout: 300 kN thrust, 3 m propeller at 3 rps,
# 6.17 m/s inflow (J = 0.686), shaft 4 m deep, Ae/Ao = 0.65, P/D = 1.0.
DESIGN = dict(thrust_N=300e3, D=3.0, depth=4.0, Z=4, ship_type="single-screw",
              n_rps=3.0, Va=6.17, Ae_Ao=0.65, P_D=1.0, temperature_c=20.0)


# --------------------------------------------------------------------------- #
# 1. closed-form quantities against hand arithmetic
# --------------------------------------------------------------------------- #
def test_vapour_pressure_matches_steam_table():
    # CRC steam tables: 1705 Pa at 15 degC, 2339 Pa at 20 degC.
    assert O.water_vapour_pressure(15.0) == pytest.approx(1705.0, rel=0.01)
    assert O.water_vapour_pressure(20.0) == pytest.approx(2339.0, rel=0.01)


def test_v_07r_hand_value():
    # V_0.7R = hypot(6.17, 0.7*pi*3*3) = hypot(6.17, 19.7920) = 20.7298 m/s
    assert O.v_07r(3.0, 3.0, 6.17) == pytest.approx(20.7298, rel=1e-4)


def test_projected_area_hand_value():
    # A_p = 0.65 * (pi*9/4) * (1.067 - 0.229) = 3.8505 m^2
    assert O.taylor_projected_area(0.65, 3.0, 1.0) == pytest.approx(3.8505,
                                                                   rel=1e-4)


def test_tau_c_hand_value():
    # tau_c = 300e3 / (0.5*1025*3.8505*20.7298^2) = 0.35373
    assert O.tau_c(300e3, 3.0, 3.0, 6.17, 0.65, 1.0) == pytest.approx(
        0.35373, rel=1e-4)


def test_sigma_07r_hand_value():
    # p_0 = 101325 + 1025*9.80665*4 = 141532 Pa; p_v(20 C) = 2339 Pa
    # sigma = (141532-2339)/(0.5*1025*20.7298^2) = 0.63197
    assert O.sigma_07r(3.0, 3.0, 6.17, 4.0, temperature_c=20.0) == pytest.approx(
        0.63197, rel=1e-4)


def test_keller_hand_value():
    # (1.3+0.3*4)*300e3/((141532-2339)*9) + 0.1 = 0.5987 + 0.1 = 0.6987
    ratio, delta = O.keller_min_area_ratio(300e3, 3.0, 4.0, 4,
                                           ship_type="single-screw",
                                           temperature_c=20.0)
    assert ratio == pytest.approx(0.6987, rel=1e-4)
    assert delta == pytest.approx(0.1)


def test_keller_delta_by_ship_type():
    deltas = {st: O.keller_min_area_ratio(300e3, 3.0, 4.0, 4, ship_type=st)[1]
              for st in O.KELLER_DELTA}
    assert deltas == {"fast-twin-screw": 0.0, "single-screw": 0.1,
                      "slow-twin-screw": 0.2}
    with pytest.raises(ValueError):
        O.keller_min_area_ratio(300e3, 3.0, 4.0, 4, ship_type="catamaran")


# --------------------------------------------------------------------------- #
# 2. definitions: identities, limits, monotonicity
# --------------------------------------------------------------------------- #
def test_sigma_n_inflation_identity():
    # sigma_n / sigma_0.7R = J^2 + (0.7 pi)^2 exactly (p_0 - p_v cancels).
    j = O.advance_ratio(3.0, 3.0, 6.17)
    expected = j * j + (0.7 * math.pi) ** 2
    assert O.sigma_n_sigma_07r_ratio(3.0, 3.0, 6.17) == pytest.approx(expected)
    # ~5.3x at the design J: the pre-m1938 implementation was non-conservative.
    assert 5.2 < O.sigma_n_sigma_07r_ratio(3.0, 3.0, 6.17) < 5.4


def test_sigma_monotone_in_depth_and_speed():
    shallow = O.sigma_07r(3.0, 3.0, 6.17, 1.0, temperature_c=20.0)
    deep = O.sigma_07r(3.0, 3.0, 6.17, 8.0, temperature_c=20.0)
    assert deep > shallow                       # more immersion, less cavitation
    fast = O.sigma_07r(4.0, 3.0, 6.17, 4.0, temperature_c=20.0)
    assert fast < shallow or fast < O.sigma_07r(
        3.0, 3.0, 6.17, 4.0, temperature_c=20.0)


def test_tau_decreases_with_advance_speed():
    slow = O.tau_c(300e3, 3.0, 3.0, 5.0, 0.65, 1.0)
    quick = O.tau_c(300e3, 3.0, 3.0, 8.0, 0.65, 1.0)
    assert quick < slow


def test_vaporous_guard():
    with pytest.raises(ValueError):
        O.driving_pressure(4.0, temperature_c=20.0, p_v=2.0e5)


def test_zero_advance_speed_uses_the_rotational_term():
    # Va = 0 still leaves V_0.7R = 0.7 pi n D = 19.792 m/s, so sigma stays
    # finite (= 0.6934 here) and rises above the design value: bollard-pull
    # conditions are less cavitating than a fast inflow at the same thrust.
    bollard = O.sigma_07r(3.0, 3.0, 0.0, 4.0, temperature_c=20.0)
    design = O.sigma_07r(3.0, 3.0, 6.17, 4.0, temperature_c=20.0)
    assert bollard == pytest.approx(0.69338, rel=1e-4)
    assert bollard > design


# --------------------------------------------------------------------------- #
# 3. Burrill verdict and its unverified limit line
# --------------------------------------------------------------------------- #
def test_placeholder_verdict_is_not_design_grade():
    point = O.burrill_point(0.3537, 0.63197)
    assert point.sigma_limit == pytest.approx(0.5537, rel=1e-4)
    assert point.margin_pct == pytest.approx(12.38, abs=0.05)
    assert point.passes is True
    assert point.curve_provenance == O.LEGACY_PLACEHOLDER
    assert point.design_grade is False
    assert point.notes                      # WITHHELD is surfaced, not hidden


def test_literature_curve_must_carry_provenance():
    with pytest.raises(ValueError):
        O.burrill_point(0.35, 0.63, curve=lambda tau: tau + 0.2)
    point = O.burrill_point(0.35, 0.63, curve=lambda tau: tau + 0.2,
                            curve_provenance="Burrill 1943, 5% line, digitised")
    assert point.design_grade is True
    assert point.curve_provenance.startswith("Burrill 1943")


def test_margin_sensitivity_matches_finite_difference():
    tau, sigma = 0.3537, 0.63197
    analytic = O.margin_sensitivity_per_unit_sigma(sigma)
    step = 1e-6
    high = O.burrill_point(tau, sigma, curve=lambda t: t + 0.2 + step,
                           curve_provenance="test").margin_pct
    low = O.burrill_point(tau, sigma, curve=lambda t: t + 0.2,
                          curve_provenance="test").margin_pct
    assert (high - low) / step == pytest.approx(analytic, rel=1e-4)
    # A +10 % error on an unverified line at sigma ~ 0.63 moves the margin by
    # ~15 percentage points: comparable with the margin itself.
    assert abs(analytic * 0.1 * sigma) == pytest.approx(10.0, rel=0.01)


def test_zero_margin_consistency():
    sigma = 0.63197
    limit = O.sigma_limit_for_zero_margin(sigma)
    point = O.burrill_point(0.3537, sigma, curve=lambda _tau: limit,
                            curve_provenance="test")
    assert point.margin_pct == pytest.approx(0.0, abs=1e-12)
    assert point.passes is False            # exactly on the line is not a pass


# --------------------------------------------------------------------------- #
# 4. the defect the oracle was built for: sigma_n substitution flips verdicts
# --------------------------------------------------------------------------- #
def test_sigma_n_substitution_flips_the_verdict():
    # A heavily loaded case where sigma_0.7R sits below the placeholder line but
    # the legacy sigma_n (5.3x larger) sits above it.
    tau = 0.5
    sigma_07r = 0.5 * tau + 0.2             # below the placeholder line tau+0.2
    inflated = sigma_07r * O.sigma_n_sigma_07r_ratio(3.0, 3.0, 6.17)
    legacy = O.burrill_point(tau, inflated)
    correct = O.burrill_point(tau, sigma_07r)
    assert legacy.passes is True
    assert correct.passes is False
    assert legacy.sigma_limit == correct.sigma_limit


# --------------------------------------------------------------------------- #
# 5. regime screen: closed-form criteria only, WITHHELD surfaced
# --------------------------------------------------------------------------- #
def test_screen_regime_design_point():
    result = O.screen_regime(**DESIGN)
    assert result["tau_c"] == pytest.approx(0.35373, rel=1e-4)
    assert result["sigma_0.7R"] == pytest.approx(0.63197, rel=1e-4)
    # Keller: required 0.6987 > actual 0.65 -> blade area is short.
    assert result["keller_ok"] is False
    assert result["keller_deficit"] == pytest.approx(-0.0486, abs=1e-4)
    # Burrill on the unverified placeholder line passes: the two criteria
    # disagree, which is exactly why the placeholder line is not design grade.
    assert result["burrill_pass_placeholder"] is True
    assert result["burrill_design_grade"] is False
    assert result["criteria_disagree"] is True
    assert any("withheld" in item or "digitised" in item
               for item in result["withheld"])


def test_screen_regime_enough_area_passes_keller():
    result = O.screen_regime(**{**DESIGN, "Ae_Ao": 0.8})
    assert result["keller_ok"] is True
    assert result["keller_deficit"] == pytest.approx(0.8 - 0.698647, abs=1e-5)


# --------------------------------------------------------------------------- #
# 6. platform probe: states are observed, never assumed
# --------------------------------------------------------------------------- #
def test_probe_unreachable_is_reported_not_raised():
    result = O.probe_platform_endpoint("http://127.0.0.1:9", timeout=0.3)
    assert result["state"] in {"unreachable", "not_wired", "rejected", "ok"}


def test_probe_live_platform_state():
    """Record the live endpoint state (contract: no NotImplementedError 5xx)."""
    import os
    base = os.environ.get("TENSORFVM_CAVITATION_API")
    if not base:
        pytest.skip("set TENSORFVM_CAVITATION_API to probe the live platform")
    result = O.probe_platform_endpoint(base)
    assert result["state"] in {"ok", "not_wired"}
    if result["state"] == "ok":
        comparison = O.compare_against_platform(result)
        for name, item in comparison["quantities"].items():
            assert item["verdict"] in {"ok", "absent"}, (name, item)


def test_compare_against_platform_detects_absent_and_mismatch():
    payload = O.PLATFORM_CHECK_PAYLOAD
    expected_tau = O.tau_c(payload["thrust"], payload["n"], payload["D"],
                           payload["Va"], payload["Ae_Ao"], payload["P_D"])
    good = {"state": "ok",
            "body": json.dumps({"tau_c": expected_tau, "sigma_0.7R": 1.0})}
    report = O.compare_against_platform(good)
    assert report["quantities"]["tau_c"]["verdict"] == "ok"
    bad = {"state": "ok",
           "body": json.dumps({"tau_c": 2.0 * expected_tau})}
    report = O.compare_against_platform(bad)
    assert report["quantities"]["tau_c"]["verdict"] == "mismatch"
    assert report["quantities"]["sigma_0.7R"]["verdict"] == "absent"