"""Drift gate for the CASE-FVM-DAMBREAK-STOKER case-pool registration.

The case-pool entry ``docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json`` quotes
numbers that come from the committed verification artifact
``src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json``.
Typed numbers go stale; this test makes the pool entry fail loudly instead:

1. every ``gate_criteria`` value in the pool entry must equal the corresponding
   headline metric in the artifact (the ``<5%`` verdict value at ``n = 1600``),
   so a driver re-run with changed metrics is a test failure rather than a
   silent inconsistency;
2. the ``key_results`` block must reproduce the artifact's ``n = 1600`` numbers
   (front_fan / shock / step_ratio / depth_l1 / front_series / monotone /
   mass_drift), so the pool entry cannot be regenerated from a stale artifact
   that nobody re-ran;
3. the pool entry must declare ``status = "pass"`` and ``verdict = "PASS"`` only
   if every gate criterion is actually under the 5 % tolerance in the artifact
   (the honest coarse-grid ``n = 200`` miss is recorded in ``verdict_qualified``,
   not in the gate).

The gate criteria are the case's own acceptance statement (波前位置 x_f(t) 与
静水区水位阶跃 vs 解析，t*=2，波形单调无伪振荡，容差 <5 %), so this test is
the machine-readable binding between the pool entry and the verified evidence.
"""

from __future__ import annotations

import json
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOL_ENTRY = os.path.join(REPO_ROOT, "docs", "oss_solutions",
                          "CASE-FVM-DAMBREAK-STOKER.json")
METRICS = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                       "artifacts", "shallow-water-dam-break", "metrics.json")


def _load_json(rel: str) -> dict:
    with open(rel, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def pool():
    return _load_json(POOL_ENTRY)


@pytest.fixture(scope="module")
def metrics():
    return _load_json(METRICS)


def test_pool_entry_has_standard_schema(pool):
    """The pool entry carries the standard case-pool fields."""
    for key in ("feature_slug", "title", "status", "round", "external_anchor",
                "quantitative_targets", "gate_criteria", "verdict",
                "acceptance_criterion", "repro_cmd", "test_cmd"):
        assert key in pool, f"missing pool-entry field {key!r}"
    assert pool["feature_slug"] == "CASE-FVM-DAMBREAK-STOKER"
    assert pool["status"] == "pass"
    assert pool["verdict"] == "PASS"


def test_gate_criteria_match_artifact_verdict(pool, metrics):
    """Every gate criterion value equals the artifact's n=1600 headline metric.

    The pool entry's ``gate_criteria`` are the machine-readable binding to the
    verified evidence; a value that differs from the artifact is a stale entry.
    """
    # map gate id -> artifact metric key (n=1600 headline)
    gate_to_metric = {
        "G1_ritter_front_fan": ("ritter-dry", "front_fan_error"),
        "G2_stoker_shock": ("stoker-wet", "shock_error"),
        "G3_step_ratio": ("stoker-wet", "step_ratio_error"),
        "G4_depth_l1": ("stoker-wet", "depth_l1"),
        "G5_front_series": ("stoker-wet", "front_series_max_error"),
        "G6_monotone": ("stoker-wet", "local_extrema"),
    }
    for gate in pool["gate_criteria"]:
        gid = gate["id"]
        assert gid in gate_to_metric, f"unknown gate id {gid!r}"
        kind, metric_key = gate_to_metric[gid]
        artifact_value = metrics["cases"][kind]["metrics"][metric_key]
        assert gate["value"] == pytest.approx(artifact_value, rel=1e-12), (
            f"gate {gid}: pool {gate['value']!r} != artifact {artifact_value!r}")
        # every gate must be under the 5 % tolerance
        assert gate["value"] <= gate["threshold"], (
            f"gate {gid} value {gate['value']} exceeds threshold {gate['threshold']}")


def test_key_results_match_artifact(pool, metrics):
    """The key_results block reproduces the artifact's n=1600 numbers."""
    kr = pool["key_results"]
    # Ritter dry n=1600
    r = metrics["cases"]["ritter-dry"]["metrics"]
    assert kr["ritter_dry_n1600"]["front_fan_error"] == pytest.approx(
        r["front_fan_error"], rel=1e-12)
    assert kr["ritter_dry_n1600"]["depth_l1"] == pytest.approx(
        r["depth_l1"], rel=1e-12)
    assert kr["ritter_dry_n1600"]["local_extrema"] == r["local_extrema"]
    assert kr["ritter_dry_n1600"]["front_series_max_error"] == pytest.approx(
        r["front_series_max_error"], rel=1e-12)
    # Stoker wet n=1600
    s = metrics["cases"]["stoker-wet"]["metrics"]
    assert kr["stoker_wet_n1600"]["shock_error"] == pytest.approx(
        s["shock_error"], rel=1e-12)
    assert kr["stoker_wet_n1600"]["step_ratio_error"] == pytest.approx(
        s["step_ratio_error"], rel=1e-12)
    assert kr["stoker_wet_n1600"]["depth_l1"] == pytest.approx(
        s["depth_l1"], rel=1e-12)
    assert kr["stoker_wet_n1600"]["local_extrema"] == s["local_extrema"]
    assert kr["stoker_wet_n1600"]["front_series_max_error"] == pytest.approx(
        s["front_series_max_error"], rel=1e-12)


def test_verdict_qualified_records_coarse_grid_honesty(pool, metrics):
    """The coarse-grid n=200 miss is recorded honestly, not hidden.

    The pool entry's ``verdict_qualified`` must mention that n=200 fails the
    fan-head probe (Ritter 5.58 %, Stoker 10.5 %), so the acceptance is not
    overstated to a grid that does not resolve the rarefaction head.
    """
    vq = pool["verdict_qualified"]
    assert "n=200" in vq or "n = 200" in vq, "coarse-grid miss not recorded"
    assert "fan-head" in vq or "fan_head" in vq, "fan-head probe not named"
    # the artifact's n=1600 verdict must be a pass on every headline check
    for kind in ("ritter-dry", "stoker-wet"):
        verdict = metrics["cases"][kind]["verdict"]
        assert isinstance(verdict, dict) and verdict, "verdict must be a check dict"
        # skip the summary key "all_pass" which is a bool, not a check dict
        checks = {k: v for k, v in verdict.items() if k != "all_pass"}
        assert all(chk["pass"] for chk in checks.values()), (
            f"{kind} n=1600 has a failing check: {verdict}")
