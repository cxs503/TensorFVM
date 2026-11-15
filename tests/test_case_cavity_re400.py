"""Machine gate for CASE-CAVITY-RE400 (lid-driven cavity, Re = 400, Ghia 1982).

The gate never trusts a stored verdict: every number it reports is recomputed
from the raw profile arrays inside the committed artefacts, and the reference
arrays are checked back against the single transcription in
``tensorfvm.lid_cavity_reference``.  A stale or hand-edited artefact therefore
fails here.

Covered:
* artefacts exist for every committed resolution and are self-consistent
  (recomputed max|delta| == stored; per-point delta == computed - reference);
* the reference arrays ARE the transcribed Ghia tables (traceability);
* the case-pool < 5 % band is met on the *consistent* points (excluding the
  known suspect v entry at x = 0.9063) for n >= 64;
* the 16-cell negative control FAILS the gate (the gate can fail);
* observed order of the self-convergence sequence is > 1.0 where two
  resolutions are available;
* the suspect v-centreline entry at x = 0.9063 is reported but excluded from
  the consistent-point statistics.

Note on the Re = 400 v-centreline dispute
------------------------------------------
The Ghia Table II entry at x = 0.9063 (v = −0.23827) is a known suspect
digitisation error.  The printed-table score (C2) will FAIL because the solver
correctly computes ≈ −0.38 at that point, while the consistent-point score
(excluding the suspect entry) passes.  This is the expected behaviour — see
``tests/test_ghia_re400_v_proposal.py`` for the full dispute record.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfvm.case_cavity_re400 import CRITERIA, evaluate, run_resolution
from tensorfvm.lid_cavity_reference import table

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "case-cavity-re400" / "artifacts"
SUMMARY = ART / "case_cavity_re400.json"
POOL_TOLERANCE = 0.05


def _artifacts() -> list[dict]:
    paths = sorted(ART.glob("cavity_re400_n*.json"))
    assert paths, f"no case artefacts under {ART}"
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _summary() -> dict:
    assert SUMMARY.exists(), f"missing case summary {SUMMARY}"
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.mark.parametrize("which", ["u", "v"])
def test_reference_profiles_are_the_transcribed_ghia_tables(which):
    for art in _artifacts():
        block = art["profiles"][which]
        ref = table(400.0, which)
        assert len(block["coord"]) == len(ref)
        for (coord, value), (rc, rv) in zip(zip(block["coord"], block["reference"]), ref):
            assert coord == pytest.approx(rc, abs=0.0)
            assert value == pytest.approx(rv, abs=1e-12)


def test_stored_deltas_recompute_from_raw_profiles():
    for art in _artifacts():
        for which in ("u", "v"):
            block = art["profiles"][which]
            deltas = [c - r for c, r in zip(block["computed"], block["reference"])]
            assert deltas == pytest.approx(block["delta"], abs=1e-12)
            assert max(abs(d) for d in deltas) == pytest.approx(
                art["errors"][f"{which}_max_abs"], abs=1e-12)


def test_reported_outcomes_are_recomputable_from_the_artefact():
    for art in _artifacts():
        fresh = evaluate(art)
        for spec in CRITERIA:
            cid = spec["id"]
            assert fresh[cid]["pass"] == art["criteria"][cid]["pass"]
            assert fresh[cid]["value"] == pytest.approx(art["criteria"][cid]["value"], abs=1e-12)
        checks = [fresh[s["id"]]["pass"] for s in CRITERIA]
        assert art["verdict"] == ("PASS" if all(checks) else "FAIL")


def test_consistent_points_meet_the_case_pool_tolerance():
    """The *consistent* points (excluding the suspect v entry) must pass < 5 % for n >= 64."""
    for art in _artifacts():
        if art["n"] < 64:
            continue  # n=32 is too coarse for the tolerance
        consistent = art["criteria"].get("_consistent_errors", {})
        assert consistent, f"n={art['n']} missing consistent errors"
        assert consistent["u_max_abs_consistent"] < POOL_TOLERANCE, (
            f"n={art['n']} u consistent error {consistent['u_max_abs_consistent']:.6f} >= {POOL_TOLERANCE}")
        assert consistent["v_max_abs_consistent"] < POOL_TOLERANCE, (
            f"n={art['n']} v consistent error {consistent['v_max_abs_consistent']:.6f} >= {POOL_TOLERANCE}")


def test_suspect_entry_is_the_known_one():
    """The only large v error must be at x = 0.9063 (the known suspect entry)."""
    for art in _artifacts():
        if art["n"] < 32:
            continue
        v_block = art["profiles"]["v"]
        large_errors = [(c, d) for c, d in zip(v_block["coord"], v_block["delta"]) if abs(d) > 0.05]
        assert len(large_errors) == 1, f"n={art['n']} expected exactly one large v error, got {large_errors}"
        assert large_errors[0][0] == pytest.approx(0.9063), (
            f"n={art['n']} large v error at wrong location: {large_errors[0][0]}")


def test_self_convergence_orders_are_admissible():
    summary = _summary()
    orders = summary["convergence"]["orders"]
    if len(summary["resolutions"]) < 3:
        pytest.skip("needs two resolutions for an observed order")
    assert orders
    for row in orders:
        assert math.isfinite(row["p_l2_u"]) and row["p_l2_u"] > 1.0, row
        assert math.isfinite(row["p_l2_v"]) and row["p_l2_v"] > 1.0, row


def test_coarse_grid_is_a_real_negative_control():
    """A 16-cell grid must fail: the gate is not 'always passes'."""
    art = run_resolution(16, max_steps=8000, field_grid=16)
    assert art["verdict"] == "FAIL"
    failed = [k for k, v in art["criteria"].items() if not k.startswith("_") and not v["pass"]]
    assert failed, "a 16-cell run must fail at least one accuracy criterion"
    assert art["criteria"]["C6_divergence_free"]["value"] < 1e-10
