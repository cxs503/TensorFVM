"""Tests for convergence diagnosis round-2 (superlinear-aware forecast).

These tests verify that the round-2 improvements:
1. Achieve forecast error ≤ 15 % on all 6 decaying cases (v1 had 1/6).
2. Maintain rate error ≤ 15 % on all 6 decaying cases (v1 had 6/6).
3. Maintain verdict agreement 8/8 (v1 had 8/8).
4. Correctly detect superlinear convergence on GMRES/CG cases.
5. Do not falsely detect superlinear convergence on stationary-rate cases.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
#: the driver (``tensorfvm.verification.convergence_xval_round2.run_xval_round2``)
#: and the committed artifact both live under ``docs/xval-round2``; this fixture
#: used to point at ``artifacts/xval-round2``, which does not exist, so every
#: assertion in this file silently *skipped* and the claim was gated by nothing.
ARTIFACT_DIR = ROOT / "docs" / "xval-round2"


@pytest.fixture(scope="module")
def round2_results():
    """Load the round-2 XVAL results (fail loudly if absent, never skip)."""
    artifact = ARTIFACT_DIR / "xval_round2.json"
    if not artifact.exists():
        pytest.fail(f"round-2 artifact missing: {artifact}; run "
                    "`python -m tensorfvm.verification.convergence_xval_round2`")
    return json.loads(artifact.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v1_results():
    """Load the v1 XVAL results for comparison."""
    v1_artifact = ROOT / "docs" / "convergence-diagnosis-v2" / "xval.json"
    if not v1_artifact.exists():
        return None
    return json.loads(v1_artifact.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Acceptance criteria
# ---------------------------------------------------------------------------

def test_rate_pass_all_decaying(round2_results):
    """C1: Rate error ≤ 15 % on all 6 decaying cases."""
    results = round2_results["results"]
    decaying = [r for r in results if r["rate_err"] is not None]
    assert len(decaying) == 6, f"Expected 6 decaying cases, got {len(decaying)}"
    for r in decaying:
        assert r["rate_pass"], f"{r['case']}: rate error {r['rate_err']*100:.2f}% > 15%"


def test_forecast_pass_all_decaying(round2_results):
    """C2: Forecast error ≤ 15 % on all 6 decaying cases (v1 had 1/6)."""
    results = round2_results["results"]
    decaying = [r for r in results if r["fc_err"] is not None]
    assert len(decaying) == 6, f"Expected 6 decaying cases with fc_err, got {len(decaying)}"
    for r in decaying:
        assert r["fc_pass"], f"{r['case']}: forecast error {r['fc_err']*100:.2f}% > 15%"


def test_verdict_agree_all(round2_results):
    """C3: Verdict agreement 8/8."""
    results = round2_results["results"]
    n_agree = sum(1 for r in results if r["verdict_agree"])
    assert n_agree == 8, f"Expected 8/8 verdict agreement, got {n_agree}/8"


def test_diverging_cases_correct(round2_results):
    """C4: Non-converging cases have non-converging verdict."""
    results = round2_results["results"]
    non_converging_verdicts = ("diverging", "oscillating", "stagnating", "insufficient")
    for r in results:
        if r["case"] in ("cd-pe10-cg-misuse", "poisson-n32-richardson"):
            assert not r["ref_converged"], f"{r['case']} should not converge"
            assert r["v2_verdict"] in non_converging_verdicts, \
                f"{r['case']}: expected non-converging verdict, got {r['v2_verdict']}"


def test_no_false_superlinear_on_stationary(round2_results):
    """C5: Superlinear detection must not trigger on stationary-rate cases."""
    results = round2_results["results"]
    richardson = [r for r in results if "richardson" in r["case"]]
    for r in richardson:
        if r.get("superlinear_fit") is not None:
            assert not r["superlinear_fit"].get("is_superlinear", False), \
                f"{r['case']}: false superlinear detection on stationary rate"


# ---------------------------------------------------------------------------
# Improvement over v1
# ---------------------------------------------------------------------------

def test_forecast_improvement_over_v1(round2_results, v1_results):
    """Round-2 forecast error must be strictly better than v1."""
    if v1_results is None:
        pytest.skip("v1 results not available for comparison")

    v1_by_case = {r["case"]: r for r in v1_results.get("results", [])}
    for r in round2_results["results"]:
        case = r["case"]
        if case in v1_by_case and r["fc_err"] is not None:
            v1_fc_err = v1_by_case[case].get("fc_err")
            if v1_fc_err is not None:
                assert r["fc_err"] <= v1_fc_err, \
                    f"{case}: round2 fc_err {r['fc_err']*100:.2f}% > v1 {v1_fc_err*100:.2f}%"


def test_benchmark_pass(round2_results):
    """Overall benchmark must pass."""
    assert round2_results["summary"]["benchmark_pass"] is True


# ---------------------------------------------------------------------------
# Superlinear detection on GMRES
# ---------------------------------------------------------------------------

def test_superlinear_fit_structure_on_gmres(round2_results):
    """GMRES cases that trigger superlinear detection must have valid fit structure.

    Strengthened from the original ``n_superlinear >= 0`` (tautology) to actually
    verify the structure of any superlinear fit that is produced.
    """
    results = round2_results["results"]
    gmres_cases = [r for r in results if "gmres" in r["case"] and r["rate_err"] is not None]
    for r in gmres_cases:
        fit = r.get("superlinear_fit")
        if fit is not None:
            # Verify the fit has all required fields
            assert "slope" in fit, f"{r['case']}: superlinear_fit missing 'slope'"
            assert "intercept" in fit, f"{r['case']}: superlinear_fit missing 'intercept'"
            assert "r_squared" in fit, f"{r['case']}: superlinear_fit missing 'r_squared'"
            assert "n_points" in fit, f"{r['case']}: superlinear_fit missing 'n_points'"
            assert "extrapolated_rate" in fit, f"{r['case']}: superlinear_fit missing 'extrapolated_rate'"
            assert "is_superlinear" in fit, f"{r['case']}: superlinear_fit missing 'is_superlinear'"
            # Verify physical constraints
            assert 0.05 <= fit["extrapolated_rate"] <= 0.99, \
                f"{r['case']}: extrapolated_rate {fit['extrapolated_rate']} out of [0.05, 0.99]"
            assert fit["r_squared"] >= 0.0, \
                f"{r['case']}: r_squared {fit['r_squared']} is negative"
            assert fit["n_points"] >= 3, \
                f"{r['case']}: n_points {fit['n_points']} < 3 (minimum for fit)"


# ---------------------------------------------------------------------------
# Artifact self-consistency
# ---------------------------------------------------------------------------

def test_artifact_numbers_recompute(round2_results):
    """Verify that the summary numbers match the individual results."""
    results = round2_results["results"]
    summary = round2_results["summary"]

    n_decaying = sum(1 for r in results if r["rate_err"] is not None)
    n_rate_pass = sum(1 for r in results if r["rate_pass"])
    n_fc_pass = sum(1 for r in results if r["fc_pass"])
    n_verdict_agree = sum(1 for r in results if r["verdict_agree"])

    assert summary["cases"] == len(results)
    assert summary["rate_pass"] == f"{n_rate_pass}/{n_decaying}"
    assert summary["forecast_pass"] == f"{n_fc_pass}/{n_decaying}"
    assert summary["verdict_agree"] == f"{n_verdict_agree}/{len(results)}"


# ---------------------------------------------------------------------------
# Adversarial note on what the numbers above actually measure
# ---------------------------------------------------------------------------

def test_forecast_metric_is_ex_post_identity(round2_results):
    """The 0.00 % forecast error is an identity, not a forecast.

    ``run_case_v2`` solves to the reference stopping iteration and only then
    asks for a forecast; the diagnosis answers ``converged`` and returns the
    iteration count it has already recorded.  So ``forecast_v2 == fc_ref`` by
    construction on every converging case.  This test documents that fact
    instead of reading it as skill -- the causal metric lives in
    ``tests/test_convergence_causal_audit.py``.
    """
    converging = [r for r in round2_results["results"] if r["ref_converged"]]
    assert converging, "fixture should contain converging cases"
    for r in converging:
        assert r["forecast_v2"] == r["fc_ref"], r["case"]
        assert r["fc_err"] == 0.0, r["case"]
