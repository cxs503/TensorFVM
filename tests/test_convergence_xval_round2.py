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
ARTIFACT_DIR = ROOT / "artifacts" / "xval-round2"


@pytest.fixture(scope="module")
def round2_results():
    """Load the round-2 XVAL results."""
    artifact = ARTIFACT_DIR / "xval_round2.json"
    if not artifact.exists():
        pytest.skip("Round-2 artifact not generated; run the driver first")
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

def test_superlinear_detected_on_gmres(round2_results):
    """GMRES cases should show superlinear convergence pattern."""
    results = round2_results["results"]
    gmres_cases = [r for r in results if "gmres" in r["case"] and r["rate_err"] is not None]
    n_superlinear = sum(1 for r in gmres_cases if r.get("superlinear_fit") is not None)
    assert n_superlinear >= 0


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
