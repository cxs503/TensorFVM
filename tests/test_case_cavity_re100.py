"""Machine gate for CASE-CAVITY-RE100 (lid-driven cavity, Re = 100, Ghia 1982).

The gate never trusts a stored verdict: every number it reports is recomputed
from the raw profile arrays inside the committed artefacts, and the reference
arrays are checked back against the single transcription in
``tensorfvm.lid_cavity_reference``.  A stale or hand-edited artefact therefore
fails here.

Covered:
* artefacts exist for every committed resolution and are self-consistent
  (recomputed max|delta| == stored; per-point delta == computed - reference);
* the reference arrays ARE the transcribed Ghia tables (traceability);
* every one of the seven case criteria passes on the finest grid, and the
  case-pool < 5 % band is met with margin;
* the 16-cell negative control FAILS the gate (the gate can fail);
* observed order of the self-convergence sequence is > 1.5 where three
  resolutions are available.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorfvm.case_cavity_re100 import CRITERIA, evaluate, run_resolution
from tensorfvm.lid_cavity_reference import table

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "case-cavity-re100" / "artifacts"
SUMMARY = ART / "case_cavity_re100.json"
POOL_TOLERANCE = 0.05


def _artifacts() -> list[dict]:
    paths = sorted(ART.glob("cavity_re100_n*.json"))
    assert paths, f"no case artefacts under {ART}"
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def _summary() -> dict:
    assert SUMMARY.exists(), f"missing case summary {SUMMARY}"
    return json.loads(SUMMARY.read_text(encoding="utf-8"))


@pytest.mark.parametrize("which", ["u", "v"])
def test_reference_profiles_are_the_transcribed_ghia_tables(which):
    for art in _artifacts():
        block = art["profiles"][which]
        ref = table(100.0, which)
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


def test_finest_grid_meets_the_case_pool_tolerance():
    summary = _summary()
    finest = max(_artifacts(), key=lambda a: a["n"])
    assert summary["finest"] == finest["n"]
    assert finest["n"] >= 64, "case must be demonstrated past the coarsest grid"
    assert finest["verdict"] == "PASS", finest["criteria"]
    for cid in ("C1_u_profile_max_abs", "C2_v_profile_max_abs",
                "C3_u_min_relative", "C4_v_max_relative"):
        assert finest["criteria"][cid]["value"] < POOL_TOLERANCE
        assert finest["criteria"][cid]["margin"] > 0.0
    assert summary["verdict"] == "PASS"


def test_every_committed_resolution_passes():
    for art in _artifacts():
        if art["n"] < 32:
            continue
        assert art["verdict"] == "PASS", (art["n"], art["criteria"])


def test_self_convergence_orders_are_second_order_or_better():
    summary = _summary()
    orders = summary["convergence"]["orders"]
    if len(summary["resolutions"]) < 3:
        pytest.skip("needs three resolutions for an observed order")
    assert orders
    for row in orders:
        assert math.isfinite(row["p_l2_u"]) and row["p_l2_u"] > 1.5, row
        assert math.isfinite(row["p_l2_v"]) and row["p_l2_v"] > 1.5, row


def test_coarse_grid_is_a_real_negative_control():
    """A 16-cell grid must fail C5: the gate is not 'always passes'."""
    art = run_resolution(16, max_steps=8000, field_grid=16)
    assert art["verdict"] == "FAIL"
    failed = [k for k, v in art["criteria"].items() if not k.startswith("_") and not v["pass"]]
    assert failed, "a 16-cell run must fail at least one accuracy criterion"
    assert art["criteria"]["C6_divergence_free"]["value"] < 1e-10
