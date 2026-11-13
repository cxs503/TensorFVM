"""Tests for the Sod (1978) asymptotic-grid convergence study.

Two layers:

* ``test_smoke_*`` build a fresh measurement through the public API, so the
  module cannot rot into an unused script;
* ``test_artifact_*`` re-derive every quoted verdict from the raw per-grid
  errors stored in the committed artifact.  The point of the second layer is
  that the *verdict* cannot be hand-edited: if somebody flips
  ``c3_literal_asymptotic`` or ``sublinear_persists`` without re-running the
  solver, these tests fail.

Run: ``PYTHONPATH=src pytest tests/test_sod1978_asymptotic.py -q``
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfvm.verification.sod1978_asymptotic import (
    BAND,
    DECLARED_GRIDS,
    _trend_towards_one,
    fitted_exponent,
    measure,
    successive_orders,
)

ARTIFACT = (Path(__file__).resolve().parents[1] / "docs" /
            "repro-fvm-sod1978-asymptotic" / "artifacts" / "asymptotic_study.json")


# ----------------------------------------------------------------------
# smoke: the measurement path works and stays physical
# ----------------------------------------------------------------------
def test_smoke_measure_keys_and_positivity():
    m = measure(50, order=2)
    for k in ("nx", "dx", "l1_rho_full", "l1_rho_cellavg", "l1_rho_restricted",
              "wave_error", "positivity_fixes", "budget_drift_mass"):
        assert k in m, k
    assert m["nx"] == 50
    assert m["positivity_fixes"] == 0
    assert abs(m["budget_drift_mass"]) < 1e-10
    assert m["l1_rho_full"] > 0


def test_trend_detects_motion_towards_one():
    # distances to one: 0.4, 0.2, 0.05 -> monotone approach
    up = _trend_towards_one([{"rate": 0.6}, {"rate": 0.8}, {"rate": 0.95}])
    assert up["monotone"] is True and up["steps_towards_one"] == 2
    down = _trend_towards_one([{"rate": 0.95}, {"rate": 0.8}, {"rate": 0.6}])
    assert down["monotone"] is False


# ----------------------------------------------------------------------
# artifact: verdicts must follow from the stored errors
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def study():
    if not ARTIFACT.exists():
        pytest.skip(f"artifact not built: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


def test_artifact_declares_the_full_ladder_before_the_run(study):
    thr = study["thresholds_declared_before_run"]
    assert set(thr["declared_grids"]) <= set(DECLARED_GRIDS)
    # audit m15344 B3 asks for a sweep that reaches the asymptotic range
    assert max(thr["declared_grids"]) >= 1600
    assert len(thr["declared_grids"]) >= 5
    assert thr["c3_literal_rate"] == 1.0
    assert thr["sublinear_band"] == BAND
    assert thr["roundoff_floor"] > 0
    assert "before the run" in thr["note"]


def test_artifact_rows_cover_every_declared_grid(study):
    for order in ("1", "2"):
        got = [r["nx"] for r in study["rows"][order]]
        assert got == list(study["thresholds_declared_before_run"]["declared_grids"]), (order, got)
        # refinement must reduce the whole-field error monotonically
        errs = [r["l1_rho_full"] for r in study["rows"][order]]
        assert all(b < a for a, b in zip(errs[:-1], errs[1:])), errs


@pytest.mark.parametrize("order", ["1", "2"])
def test_artifact_verdicts_recomputed_from_raw_errors(study, order):
    rows = study["rows"][order]
    v = study["verdicts"][order]

    succ = successive_orders(rows, "l1_rho_full")
    assert len(succ) == len(rows) - 1
    assert math.isclose(succ[-1]["rate"], v["finest_pair_rate"], rel_tol=1e-9)
    for a, b in zip(succ, study["rate_tables"][order]["l1_rho_full"]["successive"]):
        assert math.isclose(a["rate"], b["rate"], rel_tol=1e-9)

    exp3 = fitted_exponent(rows, "l1_rho_full", finest=3)
    exp4 = fitted_exponent(rows, "l1_rho_full", finest=4)
    assert math.isclose(exp3, v["fitted_exponent_finest3"], rel_tol=1e-9)
    assert math.isclose(exp4, v["fitted_exponent_finest4"], rel_tol=1e-9)

    # the two graded booleans are functions of those numbers, nothing else
    assert v["c3_literal_asymptotic"] == bool(
        v["finest_pair_rate"] >= 1.0 and v["fitted_exponent_finest3"] >= 1.0)
    assert v["sublinear_persists"] == bool(v["fitted_exponent_finest3"] < 1.0 - BAND)

    # and the trend block is the one derivable from the successive rates
    trend = _trend_towards_one(study["rate_tables"][order]["l1_rho_full"]["successive"])
    assert trend["steps_towards_one"] == v["order_trend"]["steps_towards_one"]
    assert math.isclose(trend["distance_to_one_last"],
                        v["order_trend"]["distance_to_one_last"], rel_tol=1e-9)


@pytest.mark.parametrize("order", ["1", "2"])
def test_artifact_runs_stayed_physical(study, order):
    rows = study["rows"][order]
    assert sum(r["positivity_fixes"] for r in rows) == 0
    assert max(abs(r["budget_drift_mass"]) for r in rows) < 1e-10
    for r in rows:
        assert r["steps"] > 0 and r["seconds"] >= 0.0


def test_artifact_restricted_floor_flags_match_threshold(study):
    floor = study["thresholds_declared_before_run"]["roundoff_floor"]
    for order in ("1", "2"):
        per = study["restricted_floor"][order]["per_grid"]
        for nx, entry in per.items():
            assert entry["above_floor"] == bool(entry["l1"] > floor)
        # on the fine grids the smooth-window measure is expected to collapse
        # into round-off; the study must say so instead of quoting a rate
        assert study["verdicts"][order]["restricted_usable_grids"] == [
            int(nx) for nx, e in per.items() if e["above_floor"]]


@pytest.mark.parametrize("order", ["1", "2"])
def test_artifact_smooth_design_order_recomputed(study, order):
    sw = study["smooth_wave"][order]
    rows = sw["rows"]
    succ = successive_orders(rows, "l1_rho")
    assert math.isclose(succ[-1]["rate"], sw["finest_pair_rate"], rel_tol=1e-9)
    design = float(order)
    assert math.isclose(sw["finest_pair_rel_err_pct"],
                        abs(sw["finest_pair_rate"] - design) / design * 100.0, rel_tol=1e-9)
    assert sw["finest_pair_within_3pct"] == bool(sw["finest_pair_rel_err_pct"] <= 3.0)


def test_artifact_reports_the_finest_profile(study):
    profs = study["profiles"]
    assert profs, "no profile snapshot stored"
    p = profs[-1]
    assert p["nx"] >= 1600
    for k in ("x", "rho", "rho_exact", "u", "u_exact", "p", "p_exact"):
        assert len(p[k]) == p["nx"], k