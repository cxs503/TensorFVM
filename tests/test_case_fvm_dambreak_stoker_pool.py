"""Drift gate for the CASE-FVM-DAMBREAK-STOKER case-pool registration.

The case-pool entry ``docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json`` quotes
numbers that come from two committed artifact bundles:

* ``src/tensorfvm/verification/artifacts/shallow-water-dam-break/{metrics,convergence}.json``
  -- the headline metrics at ``n = 1600`` and the graded-set margins;
* ``docs/case-fvm-dambreak-stoker/artifacts/reproduction.json`` -- the
  probe-based grid ladder that ``verdict_qualified`` summarises.

Typed numbers go stale; this test makes the pool entry fail loudly instead:

1. every ``gate_criteria`` value and every ``key_results`` number in the pool
   entry must equal the artifact's ``n = 1600`` metric, so a driver re-run with
   changed metrics is a test failure rather than a silent inconsistency;
2. the two structured blocks (``probe_ladder``, ``gate_margins``) must be the
   *derived* payload of :mod:`tensorfvm.verification.dambreak_pool_claims` --
   i.e. injectable by script, never typed by hand;
3. every number that is still written in prose (the coarse-grid percentages and
   the margin sentences) must re-derive from one of the two bundles, and the
   ``only``/``weakest``/grid readings it claims must match the margin table.
   This is the check that catches *mis-scoped* numbers: the wording shipped
   before 2026-11-16 was arithmetically derivable but told the wrong story (the
   retracted sentences are kept below as negative-control fixtures);
4. the entry must declare ``status = "pass"`` / ``verdict = "PASS"`` only if
   every graded check actually passes in the artifact (the honest coarse-grid
   ``n = 200`` miss lives in ``verdict_qualified``, not in the gate).

The gate criteria are the case's own acceptance statement (波前位置 x_f(t) 与
静水区水位阶跃 vs 解析，t*=2，波形单调无伪振荡，容差 <5 %), so this test is
the machine-readable binding between the pool entry and the verified evidence.
"""

from __future__ import annotations

import copy
import json
import os

import pytest

from tensorfvm.verification import dambreak_pool_claims as claims

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POOL_ENTRY = os.path.join(REPO_ROOT, "docs", "oss_solutions",
                          "CASE-FVM-DAMBREAK-STOKER.json")
METRICS = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                       "artifacts", "shallow-water-dam-break", "metrics.json")
CONVERGENCE = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                           "artifacts", "shallow-water-dam-break",
                           "convergence.json")
PROBE_BUNDLE = os.path.join(REPO_ROOT, "docs", "case-fvm-dambreak-stoker",
                            "artifacts", "reproduction.json")

#: the wording retracted on 2026-11-16 -- kept verbatim as a negative control so
#: the checker cannot silently lose the power to catch it again.  Its numbers
#: were each individually derivable (5.58 %, 10.5 %, 1.73x are all real), but the
#: margin *readings* were wrong: 100x overstates the headline six (weakest 5.92x)
#: and 11.8x is the ritter-dry front_series margin, not the weakest margin of the
#: acceptance elements (4.98x).
RETRACTED_VERDICT_QUALIFIED = (
    "PASS at n>=800 (both Ritter dry and Stoker wet). At n=200 the fan-head probe "
    "fails (Ritter 5.58%, Stoker 10.5%) — the coarse grid does not resolve the "
    "rarefaction head; this is recorded honestly, not hidden. Headline metrics "
    "(front_fan, shock_position, step_ratio, depth_l1, front_series, monotone) all "
    "pass at n=1600 with 100x+ margin under the 5% gate.")
RETRACTED_GRADED_SET_NOTE = (
    "Among these, front_wetted<5% is graded but the same artifact bundle publishes "
    "the quantity as 'front_wetted_diagnostic' with a NON-CONVERGENT ladder. It is "
    "the only graded key with a margin under 12x (1.73x). The acceptance sentence's "
    "three elements do not depend on it (their weakest margin is 11.8x).")


def _load_json(rel: str) -> dict:
    with open(rel, encoding="utf-8") as handle:
        return json.load(handle)


@pytest.fixture(scope="module")
def pool():
    return _load_json(POOL_ENTRY)


@pytest.fixture(scope="module")
def metrics():
    return _load_json(METRICS)


@pytest.fixture(scope="module")
def convergence():
    return _load_json(CONVERGENCE)


@pytest.fixture(scope="module")
def probe_bundle():
    return _load_json(PROBE_BUNDLE)


def test_pool_entry_has_standard_schema(pool):
    """The pool entry carries the standard case-pool fields."""
    for key in ("feature_slug", "title", "status", "round", "external_anchor",
                "quantitative_targets", "gate_criteria", "verdict",
                "acceptance_criterion", "repro_cmd", "test_cmd",
                "probe_ladder", "gate_margins"):
        assert key in pool, f"missing pool-entry field {key!r}"
    assert pool["feature_slug"] == "CASE-FVM-DAMBREAK-STOKER"
    assert pool["status"] == "pass"
    assert pool["verdict"] == "PASS"


def test_both_quoted_bundles_are_listed_as_artifacts(pool):
    """A reader must be able to reach every bundle the entry quotes."""
    artifacts = " ".join(pool["artifacts"])
    assert "shallow-water-dam-break/metrics.json" in artifacts
    assert "case-fvm-dambreak-stoker/artifacts/reproduction.json" in artifacts


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


def test_key_results_bind_the_remaining_quoted_numbers(pool, metrics):
    """The key_results numbers the headline test does not touch, bound too.

    ``front_wetted_error``, ``tv_ratio``, ``mass_drift`` and the Stoker plateau
    depths were quoted without any assertion behind them; they are the numbers a
    re-run would move first (total variation and mass conservation are exactly
    the invariants a scheme change breaks).
    """
    kr = pool["key_results"]
    r = metrics["cases"]["ritter-dry"]["metrics"]
    assert kr["ritter_dry_n1600"]["front_wetted_error"] == pytest.approx(
        r["front_wetted_error"], rel=1e-12)
    assert kr["ritter_dry_n1600"]["tv_ratio"] == pytest.approx(
        r["tv_ratio"], rel=1e-12)
    assert kr["ritter_dry_n1600"]["mass_drift"] == pytest.approx(
        r["mass_drift"], rel=1e-12)
    s = metrics["cases"]["stoker-wet"]["metrics"]
    assert kr["stoker_wet_n1600"]["tv_ratio"] == pytest.approx(
        s["tv_ratio"], rel=1e-12)
    assert kr["stoker_wet_n1600"]["mass_drift"] == pytest.approx(
        s["mass_drift"], rel=1e-12)
    assert kr["stoker_wet_n1600"]["h2_measured"] == pytest.approx(
        s["h2_measured"], rel=1e-12)
    assert kr["stoker_wet_n1600"]["h1"] == pytest.approx(
        s["h1_measured"], rel=1e-12)


def test_verdict_qualified_records_coarse_grid_honesty(pool, metrics):
    """The coarse-grid n=200 miss is recorded honestly, not hidden."""
    vq = pool["verdict_qualified"]
    assert "n=200" in vq or "n = 200" in vq, "coarse-grid miss not recorded"
    assert "fan_head" in vq or "fan-head" in vq, "fan-head probe not named"
    # the artifact's n=1600 verdict must be a pass on every graded check
    for kind in ("ritter-dry", "stoker-wet"):
        verdict = metrics["cases"][kind]["verdict"]
        assert isinstance(verdict, dict) and verdict, "verdict must be a check dict"
        # skip the summary key "all_pass" which is a bool, not a check dict
        checks = {k: v for k, v in verdict.items() if k != "all_pass"}
        assert all(chk["pass"] for chk in checks.values()), (
            f"{kind} n=1600 has a failing check: {verdict}")


# --------------------------------------------------------------------------- #
# derived blocks: injectable, not typed
# --------------------------------------------------------------------------- #
def test_probe_ladder_block_is_the_derived_payload(pool, probe_bundle):
    """``probe_ladder`` is exactly what the reproduction bundle derives to."""
    assert pool["probe_ladder"] == claims.probe_ladder(probe_bundle)


def test_gate_margins_block_is_the_derived_payload(pool, metrics):
    """``gate_margins`` is exactly what the metrics artifact derives to."""
    assert pool["gate_margins"] == claims.gate_margins(metrics)


def test_probe_ladder_states_the_accepted_grid(pool, probe_bundle):
    """The ladder block carries the grid claim ``verdict_qualified`` makes."""
    ladder = claims.probe_ladder(probe_bundle)
    assert ladder["accepted_n"] == probe_bundle["accepted_n"]
    assert ladder["failing_rungs"]["ritter-dry"] == [200]
    assert ladder["failing_rungs"]["stoker-wet"] == [200, 400]
    assert ladder["first_passing_rung_n"] == {"ritter-dry": 400, "stoker-wet": 800}


# --------------------------------------------------------------------------- #
# prose numbers: every one of them re-derives
# --------------------------------------------------------------------------- #
def test_every_quoted_number_re_derives(pool, metrics, probe_bundle, convergence):
    """No prose number in the entry is unbound (the audit passes clean)."""
    problems = claims.claim_problems(pool, metrics, probe_bundle, convergence)
    assert problems == [], "\n".join(problems)


def test_margin_table_pins_the_smallest_margin(pool, metrics):
    """``front_wetted<5%`` is the smallest finite margin, at 1.73x."""
    margins = claims.graded_margins(metrics)
    case, gate, margin = claims.smallest_finite(margins)
    assert (case, gate) == ("ritter-dry", "front_wetted<5%")
    assert margin == pytest.approx(
        margins[case][gate]["limit"] / margins[case][gate]["value"], rel=1e-12)
    assert 1.72 < margin < 1.74


def test_acceptance_weakest_margin_is_the_tv_no_growth_gate(metrics):
    """The acceptance elements' weakest margin is tv_no_growth, not front_series.

    This is the 2026-11-16 correction in machine form: the entry used to quote
    11.8x (the ritter-dry ``front_series`` margin) for the acceptance sentence.
    """
    overall = claims.acceptance_weakest(metrics)["overall"]
    assert overall["case"] == "stoker-wet"
    assert overall["gate"] == "tv_no_growth<0.5%"
    value = metrics["cases"]["stoker-wet"]["verdict"]["tv_no_growth<0.5%"]
    assert overall["margin"] == pytest.approx(
        value["limit"] / value["value"], rel=1e-12)
    assert overall["margin"] < 5.0, "the retracted 11.8x claim must stay impossible"


def test_headline_six_margin_is_not_three_digit(metrics):
    """The headline six are not separated from their gate by two orders of magnitude."""
    block = claims.gate_margins(metrics)["headline_weakest"]
    assert block["ritter-dry"]["weakest_gate"] == "front_series<5%"
    assert block["stoker-wet"]["weakest_gate"] == "front_series<5%"
    assert min(row["weakest_margin"] for row in block.values()) < 10.0


# --------------------------------------------------------------------------- #
# power: the audit fails on the wording it was written to catch
# --------------------------------------------------------------------------- #
def test_audit_catches_the_retracted_margin_rhetoric(pool, metrics, probe_bundle,
                                                     convergence):
    """The pre-2026-11-16 sentences are rejected, not silently tolerated."""
    stale = dict(pool)
    stale["verdict_qualified"] = RETRACTED_VERDICT_QUALIFIED
    problems = claims.claim_problems(stale, metrics, probe_bundle, convergence)
    assert any("100x" in problem for problem in problems), problems

    stale = dict(pool)
    stale["graded_set_note"] = RETRACTED_GRADED_SET_NOTE
    problems = claims.claim_problems(stale, metrics, probe_bundle, convergence)
    assert any("below 12x" in problem for problem in problems), problems
    assert any("11.8x" in problem for problem in problems), problems


def test_audit_catches_a_drifted_probe_bundle(pool, metrics, probe_bundle,
                                              convergence):
    """Moving one probe number invalidates the block *and* the quoted percent."""
    tampered = copy.deepcopy(probe_bundle)
    tampered["runs"]["ritter-dry"]["200"]["verdict"]["checks"]["fan_head"] = 0.04
    problems = claims.claim_problems(pool, metrics, tampered, convergence)
    assert any(problem.startswith("probe_ladder") for problem in problems)
    assert any("5.584" in problem for problem in problems), problems
    # and the derived ladder itself sees the rung stop failing
    assert claims.probe_ladder(tampered)["failing_rungs"]["ritter-dry"] == []


def test_audit_catches_a_drifted_metrics_artifact(pool, metrics, probe_bundle,
                                                  convergence):
    """Moving a graded value invalidates ``gate_margins`` and the quoted margin."""
    tampered = copy.deepcopy(metrics)
    tampered["cases"]["stoker-wet"]["verdict"]["front_series<5%"]["value"] *= 2.0
    problems = claims.claim_problems(pool, metrics=tampered, bundle=probe_bundle,
                                     convergence=convergence)
    assert any(problem.startswith("gate_margins") for problem in problems)
    assert any("5.92x" in problem for problem in problems), problems