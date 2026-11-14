"""Tests for convergence diagnosis round-3 (phase-aware forecast).

Round-3 introduces phase detection and quadratic rate extrapolation.
The causal audit (``docs/xval-round3/causal_audit_v3.json``) shows that
round-3 does **not** improve over round-2 in causal forecast accuracy:
both achieve 1/6 at first gate opening and 4/6 at 25 % remaining lead.

This is an honest result: the phase-aware model correctly identifies when
NOT to extrapolate (quadratic triggers = 0 on all 6 decaying cases), but
this conservative behavior doesn't create new predictive capability.

The tests here verify the committed audit artifact is self-consistent and
matches the expected round-3 behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "docs" / "xval-round3"


@pytest.fixture(scope="module")
def audit_v3():
    """Load the round-3 causal audit artifact (fail loudly if absent)."""
    artifact = ARTIFACT_DIR / "causal_audit_v3.json"
    if not artifact.exists():
        pytest.fail(f"round-3 artifact missing: {artifact}; run the audit script first")
    return json.loads(artifact.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Artifact integrity
# ---------------------------------------------------------------------------

def test_audit_covers_all_cases(audit_v3):
    assert audit_v3["meta"]["n_cases"] == 8
    assert audit_v3["meta"]["n_decaying"] == 6


def test_artifact_self_consistent(audit_v3):
    """Summary numbers must match individual case results."""
    meta = audit_v3["meta"]
    cases = audit_v3["cases"]
    decaying = [r for r in cases if r["decaying"] and r["stopping_iteration"]]

    n_first_pass = sum(1 for r in decaying
                       if r["arm_v3"].get("first") is not None
                       and abs(r["arm_v3"]["first"]["forecast"] - r["stopping_iteration"]) / r["stopping_iteration"] <= 0.15)
    assert meta["first_causal_forecast_within_15pct"]["v3"] == n_first_pass

    n_lead25_pass = sum(1 for r in decaying
                        if r["arm_v3"].get("at_lead_0.25") is not None
                        and r["arm_v3"]["at_lead_0.25"]["error"] <= 0.15)
    assert meta["within_15pct_at_lead_25pct"]["v3"] == n_lead25_pass


# ---------------------------------------------------------------------------
# Causal forecast accuracy (honest results)
# ---------------------------------------------------------------------------

def test_first_causal_forecast_v3(audit_v3):
    """Round-3 first causal forecast: 1/6 pass at 15% (same as round-2).

    This is an honest result: the phase-aware model doesn't improve early
    forecast accuracy because the fundamental limitation (can't predict
    GMRES superlinear convergence from early data) remains.
    """
    counts = audit_v3["meta"]["first_causal_forecast_within_15pct"]
    assert counts["v3"] == 1


def test_lead_25pct_v3(audit_v3):
    """Round-3 at 25% remaining lead: 4/6 pass at 15% (same as round-2)."""
    counts = audit_v3["meta"]["within_15pct_at_lead_25pct"]
    assert counts["v3"] == 4


def test_lead_10pct_v3(audit_v3):
    """Round-3 at 10% remaining lead: 5/6 pass at 15% (same as round-2)."""
    counts = audit_v3["meta"]["within_15pct_at_lead_10pct"]
    assert counts["v3"] == 5


# ---------------------------------------------------------------------------
# Phase detection behavior
# ---------------------------------------------------------------------------

def test_quadratic_never_triggers_on_decaying(audit_v3):
    """The quadratic model should not trigger on any decaying case.

    This is the key finding: the phase detection correctly identifies that
    extrapolation is not reliable early on, so the quadratic model is never
    used. The conservative behavior prevents making forecasts worse, but
    doesn't create new predictive capability.
    """
    for row in audit_v3["cases"]:
        if row["decaying"] and row["stopping_iteration"]:
            arm = row["arm_v3"]
            assert arm["quadratic_triggers"] == 0, \
                f"{row['case']}: quadratic should not trigger (got {arm['quadratic_triggers']})"


def test_verdict_accuracy_v3(audit_v3):
    """All cases should have correct verdicts."""
    for row in audit_v3["cases"]:
        arm = row["arm_v3"]
        verdict = arm["verdict_at_end"]
        if row["decaying"]:
            assert verdict == "converged", \
                f"{row['case']}: expected converged, got {verdict}"
        else:
            assert verdict in ("diverging", "oscillating", "stagnating", "insufficient"), \
                f"{row['case']}: expected non-converging, got {verdict}"


# ---------------------------------------------------------------------------
# Comparison with round-2 (honest assessment)
# ---------------------------------------------------------------------------

def test_v3_no_better_than_v2_at_first_gate(audit_v3):
    """Round-3 does not improve over round-2 at first gate opening.

    Round-2 (from CAUSAL-AUDIT.md): 1/6 pass at first gate.
    Round-3: 1/6 pass at first gate.

    The phase-aware model correctly avoids harmful extrapolation but doesn't
    create new predictive capability. The fundamental limitation is that
    GMRES convergence cannot be predicted from early residual data because
    the superlinear acceleration hasn't manifested yet.
    """
    counts = audit_v3["meta"]["first_causal_forecast_within_15pct"]
    assert counts["v3"] == 1


def test_v3_no_better_than_v2_at_lead_25(audit_v3):
    """Round-3 does not improve over round-2 at 25% remaining lead.

    Round-2: 4/6 pass at 25% lead.
    Round-3: 4/6 pass at 25% lead.
    """
    counts = audit_v3["meta"]["within_15pct_at_lead_25pct"]
    assert counts["v3"] == 4


# ---------------------------------------------------------------------------
# Per-case detail verification
# ---------------------------------------------------------------------------

def test_cd_pe10_gmres_first_forecast(audit_v3):
    """cd-pe10-gmres: first forecast at k=108, error 307% (building phase)."""
    row = next(r for r in audit_v3["cases"] if r["case"] == "cd-pe10-gmres")
    first = row["arm_v3"]["first"]
    assert first is not None
    assert first["k"] == 108
    err = abs(first["forecast"] - row["stopping_iteration"]) / row["stopping_iteration"]
    assert err == pytest.approx(3.0702, abs=0.01)


def test_poisson_n32_tol1e8_first_forecast(audit_v3):
    """poisson-n32-tol1e-8: first forecast at k=14, error 5.3% (passes)."""
    row = next(r for r in audit_v3["cases"] if r["case"] == "poisson-n32-tol1e-8")
    first = row["arm_v3"]["first"]
    assert first is not None
    err = abs(first["forecast"] - row["stopping_iteration"]) / row["stopping_iteration"]
    assert err <= 0.15  # This is the 1/6 that passes
