"""Regression gate for the ``CASE-WOODWARD-COLELLA`` case-pool contract.

The gate is deliberately numeric: every expected value is either read from a
committed metric artifact or recomputed from a committed profile, so the test
fails if the pool record and the raw data drift apart.  Three families of
assertions:

1. **contract integrity** -- each criterion carries quantity/comparator/
   threshold/unit, the tolerance is the ledger tolerance (10 %) and is shared
   with :mod:`tensorfvm.compressible.wc_reference`, not re-declared.
2. **reproducibility** -- recomputing every stored grid from the raw ``.npz``
   profiles reproduces the committed ``metrics_calibrated_*.json`` to <1e-6.
3. **the gate can fail** -- the verdict is not "PASS" by construction: the
   coarser grids must FAIL, and the criterion that fails there must be the peak
   density (the accuracy-limiting quantity).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorfvm.compressible import wc_case_pool as pool
from tensorfvm.compressible.wc_reference import TOLERANCE_PERCENT

DOCS = (Path(__file__).resolve().parents[1] / "docs"
        / "woodward-colella-double-blast")


def _stored(nx: int) -> dict:
    return json.loads((DOCS / "reference" / f"metrics_calibrated_nx{nx}_hllc.json")
                      .read_text())


def test_contract_integrity():
    assert pool.POOL_TOLERANCE == TOLERANCE_PERCENT == 10.0
    names = [c["name"] for c in pool.CRITERIA]
    assert len(names) == len(set(names)) == 6
    for c in pool.CRITERIA:
        assert c["comparator"] == "lt"
        assert isinstance(c["threshold"], float) and c["threshold"] > 0
        assert c["unit"] in {"%", "x"}
        assert c["quantity"]
        assert isinstance(c["primary"], bool)
    excluded = [c for c in pool.CRITERIA if not c["primary"]]
    assert [c["name"] for c in excluded] == ["low_plateau"]
    assert "benchmark bias" in excluded[0]["excluded_reason"]


def test_reference_tables_are_two_digitisation_passes():
    for kind in ("calibrated", "committed"):
        ref = pool.load_ref(kind)
        assert ref["x"].size == ref["rho"].size > 100
        assert ref["x"].min() <= 0.4 and ref["x"].max() >= 0.99
        # the two panels of one digitisation agree to the digitisation budget
        assert ref["panel_spread"] < 1e-3
    cal, com = pool.load_ref("calibrated"), pool.load_ref("committed")
    # the calibrated table is the committed one shifted by the tick/label bias
    offset = cal["rho"] - com["rho"]
    assert offset.mean() == pytest.approx(0.022017, abs=1e-4)
    assert offset.std() < 1e-6


def test_committed_metrics_are_reproduced_from_raw_profiles():
    audit = pool.audit_committed_metrics()
    assert audit["all_reproduced"], [r for r in audit["rows"] if not r["reproduced"]]
    assert audit["worst"] < 1e-6
    assert {r["nx"] for r in audit["rows"]} == set(pool.GRID_STEPS)


def test_pool_entry_passes_on_the_pool_grid_with_the_stored_numbers():
    entry = pool.case_pool_entry(pool.POOL_GRID)
    assert entry["verdict"] == "PASS" and entry["all_primary_pass"]
    assert entry["failed_primary"] == []
    stored = _stored(pool.POOL_GRID)
    peak = next(c for c in entry["criteria"] if c["criterion"] == "peak_density")
    # binding: the pool record must quote exactly the stored metric, not a copy
    assert peak["value"] == pytest.approx(
        stored["calibrated_reference"]["peak_density"]["rel_error_percent"], abs=1e-9)
    assert peak["value"] < peak["threshold"]
    # the peak is under-predicted: the solved maximum is *below* the reference
    assert entry["metrics"]["peak_density"]["got"] < entry["metrics"]["peak_density"]["ref"]


def test_gate_can_fail_on_coarse_grids():
    coarse = pool.evaluate_grid(200)
    assert coarse["verdict"] == "FAIL"
    assert {"peak_density", "peak_position", "l1_profile"} <= set(coarse["failed_primary"])
    marginal = pool.evaluate_grid(400)
    assert marginal["verdict"] == "FAIL"
    # nx = 400 misses the 10 % peak tolerance on *both* references -> the claim
    # "<10 % holds for every nx >= 400" is false on the peak criterion.
    for kind in ("calibrated", "committed"):
        peak = next(c for c in pool.evaluate_grid(400, kind)["criteria"]
                    if c["criterion"] == "peak_density")
        assert peak["value"] > pool.POOL_TOLERANCE
    # ... and it is only the peak: the well-posed integrals already pass
    for name in ("window_mass", "l1_profile", "faces"):
        row = next(c for c in marginal["criteria"] if c["criterion"] == name)
        assert row["pass"], name


def test_minimum_passing_grid_is_derived_not_declared():
    assert pool.minimum_passing_grid() == pool.POOL_GRID == 800
    ladder = {r["nx"]: r["verdict"] for r in pool.ladder()}
    assert ladder[200] == ladder[400] == "FAIL"
    assert all(ladder[nx] == "PASS" for nx in (800, 1600, 3200))


def test_calibration_explains_the_old_mass_and_trough_gaps():
    rows = pool.calibration_effect()
    assert [r["nx"] for r in rows] == list(pool.GRID_STEPS)
    for r in rows:
        # the mass integral was ~1.4 % off on every grid with the biased table
        # and is essentially exact with the calibrated one: a constant benchmark
        # offset, not a grid-dependent discretisation error.
        assert r["window_mass_error_percent"]["committed"] > 1.0
        assert r["window_mass_error_percent"]["calibrated"] < 0.5
        # likewise the near-vacuum trough: 13-14 % -> below 1 % everywhere
        assert r["low_plateau_error_percent"]["committed"] > 10.0
        assert r["low_plateau_error_percent"]["calibrated"] < 1.0


def test_cli_exit_code_tracks_the_verdict(tmp_path):
    out = tmp_path / "entry.json"
    assert pool.main(["--nx", str(pool.POOL_GRID), "--json", str(out)]) == 0
    assert json.loads(out.read_text())["verdict"] == "PASS"
    assert pool.main(["--audit"]) == 0
    assert pool.main(["--ladder"]) == 0