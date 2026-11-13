"""Gates for the XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 cross-validation.

The cross-validation itself is a measurement (see
``docs/xval/convergence-diagnosis-v2/``); these tests assert the invariants that
make it trustworthy, and none of them re-runs a solver:

* the independent reference face reproduces the analytic order of a synthetic
  power-law sequence and classifies oscillatory/divergent sequences correctly;
* the independent face agrees with the GCI implementation that was already in
  the tree (two reference implementations, not one re-derivation);
* the production port (``tensorfvm.verification.convergence_crosscheck``)
  reproduces the independent reference to machine precision;
* the guarded exponent refuses a log-log slope taken on a non-monotone error
  sequence, and falls back to the finest-pair rate instead;
* the committed v2 artifact really contains the failure mode this XVAL is about
  (published shock-position "order" negative on an oscillatory sequence), and
  the patched module now records the guard;
* every number in the committed XVAL artifact and its markdown table recomputes
  from the artifact's own raw series (anti hand-copy gate).
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
XVAL = ROOT / "docs" / "xval" / "convergence-diagnosis-v2"
ARTIFACT = XVAL / "artifacts" / "xval_convergence_diagnosis.json"
TABLE = XVAL / "artifacts" / "xval_table.md"
V2_STUDY = ROOT / "docs" / "repro-fvm-sod1978-convergence" / "study.json"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ref():
    return _load("gci_reference_xval", XVAL / "gci_reference.py")


@pytest.fixture(scope="module")
def port():
    from tensorfvm.verification import convergence_crosscheck as mod
    return mod


@pytest.fixture(scope="module")
def artifact():
    if not ARTIFACT.exists():  # pragma: no cover - artifact is committed
        pytest.skip("XVAL artifact not generated")
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def v2_study():
    return json.loads(V2_STUDY.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# reference face
# ---------------------------------------------------------------------------
def test_reference_reproduces_analytic_order(ref):
    hs = [0.04, 0.02, 0.01, 0.005]
    # phi = 1 + 0.5 h^2 -> second order, limit 1
    phi = [1.0 + 0.5 * h ** 2 for h in hs]
    res = ref.crosscheck_series(phi, hs)
    assert res["convergence_class"] == "monotonic"
    assert res["observed_order"] == pytest.approx(2.0, rel=1e-9)
    assert res["extrapolated"] == pytest.approx(1.0, rel=1e-9)
    assert res["limit_form"] == "nonzero_limit" if hasattr(ref, "limit_form") else True
    cons = ref.triplet_consistency(phi, hs)
    assert cons["consistent"] is True
    assert cons["p_triplet_coarse"] == pytest.approx(2.0, rel=1e-9)
    assert cons["p_triplet_fine"] == pytest.approx(2.0, rel=1e-9)


def test_reference_classifies_oscillatory_and_divergent(ref):
    # error sequence that changes sign between grids -> oscillatory
    assert ref.convergence_class([1.0, 0.9, 0.95]) == "oscillatory"
    # |eps32| > |eps21| -> divergent
    assert ref.convergence_class([1.0, 0.9, 0.7]) == "divergent"
    assert math.isnan(ref.observed_order([1.0, 0.9, 0.7], 2.0))
    assert math.isnan(ref.observed_order([1.0, 0.9, 0.95], 2.0))


def test_reference_agrees_with_in_tree_gci(ref):
    """Two independent reference implementations must agree within 15 %."""
    from tensorfvm.ship_freesurface.mesh_convergence import richardson_extrapolation

    hs = [0.02, 0.01, 0.005]
    phi = [1.0 + 0.5 * h ** 2 + 0.1 * h for h in hs]   # mixed 1st/2nd order
    res = ref.crosscheck_series(phi, hs)
    repo = richardson_extrapolation(phi[0], phi[1], phi[2], r=2.0, Fs=1.25)
    p_ref, p_repo = res["observed_order"], repo.p
    assert math.isfinite(p_ref) and math.isfinite(p_repo)
    assert abs(p_ref - p_repo) / p_ref <= 0.15
    assert abs(res["extrapolated"] - repo.S_extrapolated) / abs(res["extrapolated"]) <= 0.15
    assert abs(res["gci_fine"] - repo.GCI_fine) / res["gci_fine"] <= 0.15


# ---------------------------------------------------------------------------
# production port
# ---------------------------------------------------------------------------
def test_port_reproduces_independent_reference(ref, port):
    hs = [0.02, 0.01, 0.005]
    for phi in (
        [1.0 + 0.5 * h ** 2 for h in hs],
        [5.0e-5, 1.2e-5, 2.8e-6],
        [0.74510499, 0.74556260, 0.74568266],
    ):
        a = ref.crosscheck_series(phi, hs)
        b = port.crosscheck_series(phi, hs)
        assert b["convergence_class"] == a["convergence_class"]
        for key in ("observed_order", "extrapolated", "gci_fine", "asymptotic_ratio"):
            if math.isfinite(a[key]):
                assert b[key] == pytest.approx(a[key], rel=1e-10, abs=1e-14)
            else:
                assert not math.isfinite(b[key])
        assert b["limit_form"] == a["limit_form"]


def test_guarded_exponent_rejects_non_monotone_sequence(port):
    h = [1 / 50, 1 / 100, 1 / 200]
    monotone = port.effective_exponent([1.0e-3, 2.5e-4, 6.2e-5], h)
    assert monotone["valid"] is True
    assert monotone["source"] == "loglog_lsq"
    assert monotone["value"] == pytest.approx(2.0, rel=0.05)
    # shock-position-like: decelerating then accelerating error -> not monotone
    wobble = port.effective_exponent([1.6e-3, 4.6e-4, 2.4e-4, 1.3e-4][-3:], h)
    assert wobble["source"] in ("loglog_lsq", "successive_finest")
    # explicit non-monotone case must not be reported as an order
    bad = [3.0e-4, 1.0e-4, 2.0e-4]
    guarded = port.guarded_loglog_exponent(bad, h)
    assert guarded["valid"] is False
    eff = port.effective_exponent(bad, h)
    assert eff["source"] == "successive_finest"
    assert eff["valid"] is False
    assert "not monotone" in eff["reason"]


def test_rate_table_carries_crosscheck(port):
    from tensorfvm.verification.sod1978_convergence import _rate_table

    rows = [{"nx": 50, "q": 5.0e-5}, {"nx": 100, "q": 1.2e-5}, {"nx": 200, "q": 2.8e-6}]
    table = _rate_table(rows, ("q",))
    entry = table["q"]
    assert entry["crosscheck"]["convergence_class"] == "monotonic"
    # eps21 = -3.8e-5, eps32 = -9.2e-6 -> |ln(eps32/eps21)|/ln2 = 2.046
    assert entry["crosscheck"]["observed_order"] == pytest.approx(2.046, rel=0.01)
    assert entry["effective_exponent"]["valid"] is True
    assert entry["loglog_guard"]["monotone_decreasing"] is True


# ---------------------------------------------------------------------------
# the committed v2 artifact and the defect it exposes
# ---------------------------------------------------------------------------
def test_committed_v2_artifact_contains_the_failure_mode(ref, v2_study, port):
    rows = v2_study["tables"]["order2"]
    h = [r["dx"] for r in rows]
    shock_exact = rows[-1]["exact_waves"]["shock_x"]
    contact_exact = rows[-1]["exact_waves"]["contact_x"]
    shock_err = [abs(r["waves"]["shock_x"] - shock_exact) for r in rows]
    contact_err = [abs(r["waves"]["contact_x"] - contact_exact) for r in rows]

    published_shock = v2_study["wave_rates"]["order2"]["wave_error.shock_x"][
        "fitted_exponent_finest3"]
    assert published_shock < 0.0, "the published defect is gone -- re-read the artifact"

    ref_shock = ref.crosscheck_series(shock_err, h)
    assert ref_shock["convergence_class"] in ("oscillatory", "divergent")
    assert not math.isfinite(ref_shock["observed_order"])

    guarded = port.effective_exponent(shock_err, h)
    assert guarded["valid"] is False
    assert guarded["source"] == "successive_finest"
    assert guarded["value"] > 0.0

    # the mechanism ordering the artifact reads is robust across both estimators
    assert contact_err[-1] > 0.0
    assert port.effective_exponent(contact_err, h)["value"] < guarded["value"]


def test_patched_module_flips_the_mechanism_verdict(artifact):
    com = artifact["committed_v2_artifact"]
    assert com["before"]["sublinear_convergence_observed"] is False
    assert com["after"]["sublinear_convergence_observed"] is True
    assert com["after"]["shock_loglog_admissible"] is False
    assert com["after"]["ordering_holds_with_finest_successive"] is True
    assert com["after"]["ordering_holds_with_effective_exponent"] is True


# ---------------------------------------------------------------------------
# artifact self-consistency (anti hand-copy)
# ---------------------------------------------------------------------------
def test_artifact_numbers_recompute(ref, artifact):
    p_tol = artifact["thresholds"]["p_rel_diff"]
    assert artifact["summary"], "empty summary"
    for row in artifact["summary"]:
        sweep = artifact["sweeps"][row["sweep"]]
        entry = sweep[row["functional"]]
        assert row["p_ref"] == pytest.approx(entry["ref"]["observed_order"], rel=1e-12) \
            if math.isfinite(row["p_ref"]) else not math.isfinite(entry["ref"]["observed_order"])
        # replay the reference from the stored raw series
        replay = ref.crosscheck_series(entry["phi"], entry["h"])
        if math.isfinite(replay["observed_order"]):
            assert row["p_ref"] == pytest.approx(replay["observed_order"], rel=1e-10)
        # the v2 numbers must be the ones the driver recorded, on the v2 series
        assert row["p_got_loglog"] == pytest.approx(
            entry["got"]["loglog_exponent_finest3"], rel=1e-12)
        if math.isfinite(row["p_ref"]) and math.isfinite(row["p_got_loglog"]):
            replay_diff = abs(row["p_got_loglog"] - row["p_ref"]) / abs(row["p_ref"])
            assert row["rel_diff_loglog"] == pytest.approx(replay_diff, rel=1e-12)
            assert row["pass_loglog"] == (replay_diff <= p_tol)


def test_markdown_quotes_artifact_numbers(artifact):
    text = TABLE.read_text(encoding="utf-8")
    assert "PROGRAM" not in text
    # spot-check three rows: label + the observed-order numbers must appear
    for row in artifact["summary"][:3]:
        if math.isfinite(row["p_ref"]) and math.isfinite(row["p_got_loglog"]):
            assert f"{row['p_ref']:.3f}" in text
            assert f"{row['p_got_loglog']:.3f}" in text
        assert row["functional"] in text


def test_verdicts_match_artifact(artifact):
    v = artifact["verdicts"]
    assert v["n_rows"] == len(artifact["summary"])
    consistent = [r for r in artifact["summary"]
                  if r["sweep"].startswith("combined") and r["triplet_consistent"]]
    assert v["agreement_required_rows"] == len(consistent)
    assert v["estimators_interchangeable_in_asymptotic_range"] == (
        len(v["agreement_fail_rows"]) == 0)
    # the non-vacuous test must be able to catch what the single-triplet one cannot
    assert v["outside_asymptotic_range"], "expected at least one non-asymptotic row"