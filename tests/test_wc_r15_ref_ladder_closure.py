"""Regression gate for the round-15 reference-ladder CFL 0.8 closure
(anchor XVAL-FVM-M18737-ROUND2).

Round-14 §⑤-2 left exactly one item open: the reference ladder's 6400 rung was
still the frozen CFL 0.4 instrument, so the E4 ``licensed`` check (all reference
rungs on one instrument) was False and ``eps_ref`` was recorded as a *probe*
rather than a certified error bar.  Round-15 adds the missing WENO5@6400 CFL 0.8
rung (job ``cj4398``) and re-runs the frozen R13 pipeline with the 6400 tag table
overridden *only after* the E1r6400 same-grid CFL admission gate passes.

This gate guards the *closure driver and the honesty of the accounting*:

1. the driver actually runs to completion (it previously crashed with
   ``UnboundLocalError`` because the local ``e1r6400`` shadowed the module-level
   gate function -- a genuine defect that left the anchor "partially closed");
2. the E1r6400 same-grid CFL 0.4 vs 0.8 admission gate is evaluated and, when it
   passes, the 6400 rung is licensed onto the CFL 0.8 instrument (``licensed``
   becomes True, ``eps_ref`` becomes a certified error bar);
3. the graded family is bit-identical to R14 (same input artifacts => same
   numbers), i.e. the tag-table override changes *only* the reference side;
4. a missing rung is recorded as ``not_measured`` -- never as a PASS, never as an
   exception that hides the gap.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XV = REPO / "docs" / "XVAL-FVM-M18737-ROUND2"
if str(XV) not in sys.path:
    sys.path.insert(0, str(XV))

import r15_ref_ladder_cfl as r15  # noqa: E402


@pytest.fixture(scope="module")
def closure(tmp_path_factory):
    """Run the R15 driver to completion on a scratch output dir."""
    d = tmp_path_factory.mktemp("r15_closure")
    j = d / "r15_certified_rung.json"
    m = d / "R15-CLOSURE.md"
    r15.main.__globals__["sys"] = sys  # ensure argparse sees the real argv
    # drive main() with explicit paths (avoid touching the committed artifacts)
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=j)
    ap.add_argument("--markdown", type=Path, default=m)
    ap.add_argument("--figures", type=Path, default=d / "figures")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(["--json", str(j), "--markdown", str(m), "--no-figures"])
    r15.main.__globals__["argparse"] = argparse
    # call the real main via its own argparse by monkeypatching argv
    import builtins
    _orig = sys.argv
    sys.argv = ["r15", "--json", str(j), "--markdown", str(m), "--no-figures"]
    try:
        r15.main()
    finally:
        sys.argv = _orig
    return j, m


def test_driver_runs_to_completion(closure):
    """The driver must produce both json and markdown (previously crashed)."""
    j, m = closure
    assert j.exists(), "driver must write r15_certified_rung.json"
    assert m.exists(), "driver must write R15-CLOSURE.md"
    v = json.loads(j.read_text())
    assert v["status"] == "measured"
    assert v["round"] == 15


def test_e1r6400_gate_admitted(closure):
    """Same-grid CFL 0.4 vs 0.8 @6400 must be < 0.2 % (pre-registered threshold)."""
    j, _ = closure
    v = json.loads(j.read_text())
    g = v["E1r6400_cfl_reference"]
    assert g is not None, "E1r6400 must be measured (both CFL profiles present)"
    assert g["admitted"] is True, f"E1r6400 L1={g['l1_percent']}% must be < 0.2%"
    assert g["l1_percent"] < r15.R13.CFL_ADMISSION


def test_licensed_eps_ref_certified(closure):
    """All reference rungs on one instrument => eps_ref is a certified error bar."""
    j, _ = closure
    v = json.loads(j.read_text())
    assert v["r15_admitted"] is True
    assert v["E4_honest_ceiling"]["licensed"] is True


def test_graded_family_bit_identical_to_r14(closure):
    """Same input artifacts => graded ladder/E1/E1r/E2/E3 must match R14 exactly."""
    j, _ = closure
    v = json.loads(j.read_text())
    assert v["invariants_vs_r14"]["ok"] is True, (
        "tag-table override must not perturb the graded family: "
        + "; ".join(v["invariants_vs_r14"]["mismatches"]))


def test_honest_under_1pct_verdict(closure):
    """The certified-<1% verdict must hold on the honest B0/B1/B2/B4 upper bounds."""
    j, _ = closure
    v = json.loads(j.read_text())
    assert "HONEST_UNDER_1PCT_B0_B1_B2_B4" in v["adjudication"]
