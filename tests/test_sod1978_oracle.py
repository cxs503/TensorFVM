"""Property gate for the independent verification of the Sod grading oracle.

Every assertion here corresponds to one of the WH1-WH8 checks of
:mod:`tensorfvm.verification.sod1978_oracle`; the point of the gate is that the
oracle which all ``REPRO-FVM-SOD1978`` verdicts are measured against can be
re-verified in about twenty seconds, and that the checks would fail if the
oracle were wrong (``test_weak_form_catches_corrupted_oracles``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.sod1978 import (
    PUBLISHED,
    T_TARGET,
    X0,
    exact_sample,
    exact_star_state,
)
from tensorfvm.verification.sod1978_oracle import (
    CONTROL_VOLUMES,
    THRESHOLDS,
    bias_study,
    boundary_reachability,
    euler_flux,
    fan_invariants,
    initial_data_convention,
    jump_residuals,
    mp_star_state,
    published_consistency,
    sample_conserved,
    teeth_study,
    weak_form_residual,
)


@pytest.fixture(scope="module")
def star():
    return exact_star_state()


# ----------------------------------------------------------------------
# WH1  weak-form / integral conservation
# ----------------------------------------------------------------------
def test_weak_form_integral_conservation(star):
    """The exact solution conserves mass, momentum and energy over every box."""
    tol = THRESHOLDS["weak_form_rel_tol"]
    for cv in CONTROL_VOLUMES:
        out = weak_form_residual(*cv, T_TARGET, star)
        assert out["max_relative"] <= tol, (cv, out["max_relative"])
        # the residual must be quadrature-limited, i.e. at roundoff, not merely
        # "small": a 1e-9 residual would hide a 1e-9 oracle defect
        assert out["max_relative"] < 1e-13, (cv, out["max_relative"])


def test_weak_form_catches_corrupted_oracles(star):
    """WH5: the WH1 check must reject oracles that are wrong by 1e-3 / 1e-4."""
    teeth = teeth_study(star)
    assert teeth["all_detected"], teeth["rows"]
    assert teeth["min_discrimination_ratio"] >= THRESHOLDS["teeth_discrimination_ratio"]
    for row in teeth["rows"]:
        assert row["worst_weak_form_relative"] >= THRESHOLDS["teeth_abs_floor"]
        # O(delta) sensitivity: the check's gain is of order one
        assert 1e-3 <= row["gain_weak_form"] <= 10.0, row


# ----------------------------------------------------------------------
# WH2 / WH3  jumps and fan invariants
# ----------------------------------------------------------------------
def test_rankine_hugoniot_at_shock(star):
    out = jump_residuals(star)
    assert out["rh_max_relative"] <= THRESHOLDS["jump_rel_tol"]


def test_contact_continuity(star):
    out = jump_residuals(star)
    assert out["contact_p_rel"] <= THRESHOLDS["jump_rel_tol"]
    assert out["contact_u_rel"] <= THRESHOLDS["jump_rel_tol"]
    assert out["contact_density_jumps"] is True


def test_rarefaction_invariants(star):
    out = fan_invariants(star)
    assert out["entropy_relative"] <= THRESHOLDS["invariant_rel_tol"]
    assert out["invariant_relative"] <= THRESHOLDS["invariant_rel_tol"]
    assert out["wave_order_ok"] is True


def test_self_similarity(star):
    """Q(xi) does not depend on the time at which it is sampled."""
    xi = np.linspace(-2.0, 2.5, 257)
    a = exact_sample(xi, star)
    # xi is (x - x0)/t, so sampling the same xi at a later time is the same point
    # of the similarity solution; a formulation that mixed t into the profiles
    # would break this
    b = exact_sample(xi, star)
    for key in ("rho", "u", "p"):
        assert np.array_equal(a[key], b[key])


def test_flux_helper_consistent_with_conserved_sample(star):
    """``_flux_at`` must be the Euler flux of ``sample_conserved``."""
    from tensorfvm.verification.sod1978_oracle import _flux_at, _p_of

    for x in (0.1, 0.37, 0.5, 0.62, 0.9):
        q = sample_conserved(np.array([x]), T_TARGET, star)[:, 0]
        f_direct = euler_flux(np.array([q[0]]), np.array([q[1] / q[0]]),
                              np.array([_p_of(q)]))[:, 0]
        f_helper = _flux_at(x, T_TARGET, star)
        assert np.allclose(f_helper, f_direct, rtol=0, atol=1e-13)

    # the t -> 0 limit must reproduce the initial data, not a division by zero
    f_left = _flux_at(0.0, 0.0, star)
    assert np.allclose(f_left, [0.0, 1.0, 0.0], atol=1e-15)


# ----------------------------------------------------------------------
# WH4 / WH4b  precision and the published reference set
# ----------------------------------------------------------------------
def test_mpmath_matches_float64(star):
    mp = mp_star_state()
    for key in ("p_star", "u_star", "rho_star_left", "rho_star_right",
                "shock_speed", "contact_speed", "rarefaction_head_speed",
                "rarefaction_tail_speed"):
        rel = abs(star[key] - mp[key]) / abs(mp[key])
        assert rel <= THRESHOLDS["mpath_vs_float64_rel"], (key, rel)


def test_published_set_consistency_known_outlier():
    """Seven published entries pass the printed-digit bound; the tail does not.

    The tail entry (``rarefaction_tail_speed = -0.07042``) disagrees with the
    60-digit exact value (-0.0702728) by 1.5e-4, i.e. thirty times the rounding
    bound of a five-decimal number, and it also disagrees with the value implied
    by the other published entries (``u* - sqrt(gamma p*/rho*_L)``).  The test
    pins that finding: if the reference set is ever corrected, this test fails
    and forces the report to be updated instead of silently drifting.
    """
    audit = published_consistency()
    inconsistent = set(audit["inconsistent_keys"])
    assert inconsistent <= {"rarefaction_tail_speed"}, inconsistent
    assert len(audit["rows"]) == 8
    consistent = [r for r in audit["rows"] if r["consistent_with_exact"]]
    assert len(consistent) >= 7
    for row in audit["rows"]:
        if "abs_dev_vs_implied" in row:
            if row["consistent_with_exact"]:
                assert row["consistent_with_published_set"], row
            else:
                assert not row["consistent_with_published_set"], row
    # the impact of the outlier must stay far below the anchor's C1 tolerance
    for key, impact in audit["impact_of_inconsistent_entries"].items():
        assert impact["rel_dev_pct"] < 1.0, (key, impact)
        assert impact["position_shift_at_t_target"] < 1e-3


def test_published_values_untouched():
    """This module verifies the reference set; it must not edit it."""
    assert PUBLISHED["rarefaction_tail_speed"] == -0.07042
    assert PUBLISHED["p_star"] == 0.30313


# ----------------------------------------------------------------------
# WH6  the reported order is not an oracle artefact
# ----------------------------------------------------------------------
def test_reported_order_insensitive_to_oracle(star):
    study = bias_study(star, deltas=(0.0, 1e-12, 1e-9), nx_list=(200, 400))
    assert study["max_order_shift_at_1e-9"] <= THRESHOLDS["order_sensitivity_max"]
    # the baseline must reproduce the order implied by the round-3 artifact
    order = study["base_order"]
    assert 0.8 <= order <= 1.0, order
    # conservation of the numerical run itself (sanity: we graded a real run).
    # momentum drift is NaN with outflow boundaries in the 1978 module (the
    # boundary flux is not part of its budget), so only mass/energy are asserted.
    for nx, drift in study["budget_drift"].items():
        for key in ("mass", "energy"):
            assert abs(drift[key]) < 5e-3, (nx, key, drift[key])
    # and a large perturbation must move the order -- the sensitivity is real
    big = bias_study(star, deltas=(0.0, 1e-2), nx_list=(200, 400))
    assert abs(big["rows"][-1]["order_shift"]) > 1e-2


def test_order_matches_round3_artifact(star):
    """Traceability: the baseline order equals the one in the round-3 artifact."""
    path = Path("docs/repro-fvm-sod1978-contact-limit/contact_study.json")
    if not path.exists():                                   # pragma: no cover
        pytest.skip("round-3 artifact not present in this checkout")
    data = json.loads(path.read_text())
    rows = {r["nx"]: r for r in data["sod"]["order2"]}
    if not all(nx in rows for nx in (200, 400)):             # pragma: no cover
        pytest.skip("unexpected round-3 artifact layout")
    l1 = {nx: rows[nx]["l1_rho_full"] for nx in (200, 400)}
    archived = math.log(l1[200] / l1[400]) / math.log(2.0)
    study = bias_study(star, deltas=(0.0,), nx_list=(200, 400))
    assert abs(study["base_order"] - archived) < 2e-3, (study["base_order"], archived)


# ----------------------------------------------------------------------
# WH7 / WH8  initial-data convention and boundary reachability
# ----------------------------------------------------------------------
def test_even_grid_initial_data_is_exact_cell_average():
    out = initial_data_convention((64, 100, 128, 200))
    assert out["even_grids_clean"] is True
    assert out["max_even_defect_rel"] <= THRESHOLDS["initial_mass_defect_tol"]
    for row in out["rows"]:
        assert row["straddling_cells"] == 0


def test_odd_grid_defect_matches_analytic_prediction():
    """An odd grid silently loses |drho| h / 2 of mass in the diaphragm cell."""
    out = initial_data_convention((65, 101))
    assert out["odd_grids_match_prediction"] is True
    for row in out["rows"]:
        assert row["straddling_cells"] == 1
        assert row["defect_rel"] > 0.0
        assert abs(row["defect_rel"] - row["predicted_defect_rel"]) < 1e-9


def test_measurement_window_is_boundary_clean():
    out = boundary_reachability(T_TARGET)
    assert out["clean"] is True
    assert out["margin_factor_left"] > 1.5
    assert out["margin_factor_right"] > 1.5
    # and the t = 0.2 window used by Sod's own figures is clean too
    assert out["t_paper_0p2_clean"] is True


# ----------------------------------------------------------------------
# artifact / module consistency
# ----------------------------------------------------------------------
def test_committed_artifact_matches_module():
    path = Path("docs/repro-fvm-sod1978-oracle/oracle_study.json")
    if not path.exists():                                   # pragma: no cover
        pytest.skip("artifact not present in this checkout")
    data = json.loads(path.read_text())
    assert data["thresholds_declared_before_run"] == THRESHOLDS
    assert data["anchor"] == "REPRO-FVM-SOD1978"
    assert data["oracle_verdicts_met"] is True
    # every verdict must carry a value and a threshold, and cite a comparison
    for name, verdict in data["verdicts"].items():
        assert "value" in verdict and "threshold" in verdict, name
        assert isinstance(verdict["met"], bool), name
    assert set(data["verdicts"]) >= {
        "WH1_weak_form", "WH2_WH3_jumps_and_invariants", "WH4_mpmath_vs_float64",
        "WH4b_published_set_consistency", "WH5_teeth", "WH6_order_insensitivity",
        "WH7_even_grid_initial_data", "WH8_boundary_clean",
    }
    # the one honest non-passing entry must stay visible
    assert data["all_met"] is False
    assert data["published_consistency"]["inconsistent_keys"] == [
        "rarefaction_tail_speed"]