"""Strictness gate for ``CASE-FVM-CAVITY-RE1000`` (board audit m20230, F1/F2/F3/F5).

The committed bundle used to be graded by a single flat 5 % band while the judge
row registered in ``tensor_benchmark_cases`` grades ``u_min`` / ``v_max`` /
centreline ``max|du|`` at **3 %** and the primary-vortex centre at ``<= 0.01 L``
for ``n >= 128``.  A solution degrading into the 3 %..5 % window therefore passed
the repository gate, and both spellings of the anchor (``CASE-FVM-CAVITY-RE1000``,
``CASE-CAVITY-RE1000-FVM``) had to be understood by hand.

These tests assert the *invariant* that was missing, not merely that code runs:

* the gate's thresholds are the registered ones, and the stored n = 128 entry is
  graded in that band (not silently in the 5 % diagnostic band);
* a synthetic solution inside the 3 %..5 % window **fails** at the judge grid and
  still passes at the coarse diagnostic grid (the two bands really differ);
* a vortex centre between 0.01 L and 0.05 L fails at the judge grid;
* the bundle's own numbers are reproducible from the raw converged *fields*;
* the verdict does not depend on the interpolation scheme used to sample them.

    PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000_gate_registered.py -q
"""

from __future__ import annotations

import copy
import json
import pathlib

import pytest

from tensorfvm.case_cavity_re1000 import (
    ANCHOR_ALIASES,
    CASE_POOL_TOLERANCE,
    CASE_SLUG,
    CRITERIA,
    JUDGE_GRID,
    REGISTERED_TOLERANCE,
    VORTEX_TOLERANCE_L,
    evaluate,
    resolve_anchor,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "docs" / "CASE-FVM-CAVITY-RE1000"
RUNS = CASE_DIR / "runs"


def _json(name: str) -> dict:
    return json.loads((RUNS / name).read_text())


@pytest.fixture(scope="module")
def summary() -> dict:
    return _json("summary.json")


@pytest.fixture(scope="module")
def graded(summary) -> dict:
    rows = {int(c["n"]): c for c in summary["cases"]}
    assert JUDGE_GRID in rows, sorted(rows)
    return rows[JUDGE_GRID]


def _thresholds() -> dict[str, float]:
    return {str(s["id"]): float(s["threshold"]) for s in CRITERIA}


def test_gate_matches_registered_row() -> None:
    """The gate numbers are the registered judge row's, not the pool's 5 % band."""
    thr = _thresholds()
    assert thr["C1_u_profile_max_abs"] == pytest.approx(REGISTERED_TOLERANCE)
    assert thr["C2_v_profile_max_abs"] == pytest.approx(REGISTERED_TOLERANCE)
    assert thr["C3_u_min_relative"] == pytest.approx(REGISTERED_TOLERANCE)
    assert thr["C4_v_max_relative"] == pytest.approx(REGISTERED_TOLERANCE)
    assert thr["C5_vortex_centre"] == pytest.approx(VORTEX_TOLERANCE_L)
    assert REGISTERED_TOLERANCE < CASE_POOL_TOLERANCE, "gate must be tighter than the pool band"
    assert JUDGE_GRID == 128
    for spec in CRITERIA:
        if str(spec["id"]).startswith(("C1", "C2", "C3", "C4", "C5")):
            assert int(spec["graded_from_n"]) == JUDGE_GRID


def test_registration_sql_is_consistent_with_the_code() -> None:
    """The SQL artefact that registers the row must quote the same numbers."""
    sql = (CASE_DIR / "register_benchmark_case.sql").read_text()
    assert CASE_SLUG in sql
    for alias in ANCHOR_ALIASES:
        assert alias in sql, alias
    assert "<3%" in sql.replace(" ", "")
    assert "0.01" in sql
    assert "n>=128" in sql.replace(" ", "")


def test_stored_finest_entry_is_graded_in_the_registered_band(graded) -> None:
    for cid in ("C1_u_profile_max_abs", "C2_v_profile_max_abs",
                "C3_u_min_relative", "C4_v_max_relative"):
        entry = graded["criteria"][cid]
        assert entry["band"] == "registered"
        assert entry["threshold"] == pytest.approx(REGISTERED_TOLERANCE)
        assert entry["pass"] is True
    vortex = graded["criteria"]["C5_vortex_centre"]
    assert vortex["band"] == "registered"
    assert vortex["threshold"] == pytest.approx(VORTEX_TOLERANCE_L)
    assert graded["verdict"] == "PASS"


def test_coarse_entries_stay_in_the_diagnostic_band(summary) -> None:
    rows = {int(c["n"]): c for c in summary["cases"]}
    for n in sorted(rows):
        if n >= JUDGE_GRID:
            continue
        entry = rows[n]["criteria"]["C1_u_profile_max_abs"]
        assert entry["band"] == "diagnostic"
        assert entry["threshold"] == pytest.approx(CASE_POOL_TOLERANCE)
        assert rows[n]["gate"]["graded"] is False


def _degraded(case: dict, *, u_offset: float = 0.0, v_offset: float = 0.0,
              vortex: float | None = None) -> dict:
    """Copy an artefact and push one number into a chosen error window."""
    out = copy.deepcopy(case)
    for which, offset in (("u", u_offset), ("v", v_offset)):
        if not offset:
            continue
        prof = out[which]["profiles"]
        ref, got = prof["reference"], prof["computed"]
        j = 1  # an interior sample: keeps the endpoint (lid/wall) values exact
        got[j] = ref[j] + offset
    if vortex is not None:
        out["vortex_centre"]["displacement"] = float(vortex)
    out["criteria"] = evaluate(out)
    checks = [v for k, v in out["criteria"].items() if not k.startswith("_")]
    out["verdict"] = "PASS" if all(v["pass"] for v in checks) else "FAIL"
    return out


def test_solution_inside_the_3_to_5_percent_window_is_rejected(graded) -> None:
    """The defect the audit found: 4 % error used to pass the repository gate."""
    bad = _degraded(graded, u_offset=0.04, v_offset=0.04)
    c1 = bad["criteria"]["C1_u_profile_max_abs"]
    assert REGISTERED_TOLERANCE < c1["value"] <= CASE_POOL_TOLERANCE, c1
    assert c1["pass"] is False
    assert bad["verdict"] == "FAIL"

    # same numbers on a coarse (diagnostic) grid are still reported as a pass:
    # that is the band difference, and it is why the judge grid matters
    coarse = copy.deepcopy(bad)
    coarse["n"] = 64
    coarse["criteria"] = evaluate(coarse)
    assert coarse["criteria"]["C1_u_profile_max_abs"]["band"] == "diagnostic"
    assert coarse["criteria"]["C1_u_profile_max_abs"]["pass"] is True


def test_vortex_drift_between_the_two_bands_is_rejected(graded) -> None:
    bad = _degraded(graded, vortex=0.02)
    entry = bad["criteria"]["C5_vortex_centre"]
    assert VORTEX_TOLERANCE_L < entry["value"] <= CASE_POOL_TOLERANCE, entry
    assert entry["pass"] is False
    assert bad["verdict"] == "FAIL"


def test_committed_bundle_passes_the_registered_gate(graded) -> None:
    """The real solution is not merely inside 5 %: it is inside the 3 % band."""
    basis = graded["criteria"]
    assert basis["C3_u_min_relative"]["value"] < REGISTERED_TOLERANCE
    assert basis["C4_v_max_relative"]["value"] < REGISTERED_TOLERANCE
    assert basis["C1_u_profile_max_abs"]["value"] < REGISTERED_TOLERANCE
    assert basis["C2_v_profile_max_abs"]["value"] < REGISTERED_TOLERANCE
    assert basis["C5_vortex_centre"]["value"] < VORTEX_TOLERANCE_L
    for cid, entry in basis.items():
        if cid.startswith("_"):
            continue
        assert entry["margin"] >= 0.0, (cid, entry)


def test_rescore_from_raw_fields_reproduces_the_bundle() -> None:
    """Re-derive every metric from the stored MAC fields (no solver re-run)."""
    import sys

    sys.path.insert(0, str(CASE_DIR))
    from rescore_registered_gate import RECONCILE_ATOL, rescore  # noqa: E402

    report = rescore(RUNS, write=False, log=lambda *a, **k: None)
    assert report["judge"]["all_graded_pass"] is True
    for row in report["grids"]:
        assert row["reconcile"]["u"] <= RECONCILE_ATOL, row["n"]
        assert row["reconcile"]["v"] <= RECONCILE_ATOL, row["n"]
        assert row["vortex_centre"]["estimator_gap_cells"] <= 2.0, row["n"]
        for key, pair in row["reconcile_metrics"].items():
            if key == "C5":
                continue
            assert pair["value_recomputed"] == pytest.approx(pair["value_stored"], abs=1e-9)
        if row["n"] < JUDGE_GRID:
            continue
        assert row["verdict"] == "PASS"
        assert row["schemes"]["hermite"]["max_abs_u"] < REGISTERED_TOLERANCE
        assert row["schemes"]["hermite"]["max_abs_v"] < REGISTERED_TOLERANCE


def test_verdict_is_interpolation_scheme_independent() -> None:
    """Audit F5: the reported error is scheme-sensitive on coarse grids only."""
    report = _json("gate_registered.json")
    rows = {int(r["n"]): r for r in report["grids"]}
    fine = rows[JUDGE_GRID]
    # on the judge grid the centreline metric difference between Hermite and PCHIP
    # is ~1e-7 U (coarse grids differ by ~0.12 U, which is why they are diagnostic)
    assert fine["scheme_sensitivity"]["max_abs_u"] < 1e-5
    assert fine["scheme_sensitivity"]["max_abs_v"] < 1e-3
    assert fine["scheme_sensitivity"]["u_min"] < 1e-3
    assert fine["scheme_sensitivity"]["v_max"] < 1e-3
    assert rows[32]["scheme_sensitivity"]["max_abs_u"] > 0.05
    assert fine["verdict"] == "PASS"


def test_anchor_spellings_resolve_to_one_case(summary) -> None:
    assert resolve_anchor(CASE_SLUG) == CASE_SLUG
    assert resolve_anchor("CASE-CAVITY-RE1000-FVM") == CASE_SLUG
    assert resolve_anchor("FVM-CAVITY-RE1000") == CASE_SLUG
    assert summary["case"] == CASE_SLUG
    assert summary["anchor"] == CASE_SLUG
    assert set(summary["anchor_aliases"]) == set(ANCHOR_ALIASES)
    with pytest.raises(KeyError):
        resolve_anchor("CASE-NOT-A-CAVITY")


def test_registered_gate_artefact_is_complete() -> None:
    report = _json("gate_registered.json")
    assert report["case"] == CASE_SLUG
    assert report["judge_grid"] == JUDGE_GRID
    assert report["registered_tolerance"] == pytest.approx(REGISTERED_TOLERANCE)
    assert report["vortex_tolerance_L"] == pytest.approx(VORTEX_TOLERANCE_L)
    assert {int(g["n"]) for g in report["grids"]} == {32, 64, 128}
    assert report["judge"]["verdict"] == {str(JUDGE_GRID): "PASS"}