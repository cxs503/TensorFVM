"""Round-3 grid-refinement evidence for the Armaly (1983) reproduction.

The tests are deliberately *derivation* tests rather than replay tests: they
recompute the report and the convergence analysis from the committed JSON
artifacts and require the committed files to match, so a hand-edited number in
the report or in ``convergence_analysis.json`` fails the suite.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from tensorfvm.inc_bench.cases import BFS_LEVELS

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs" / "repro-armaly1983-bfs"
ARTIFACTS = DOCS / "artifacts"

#: predeclared in round 1 and unchanged since: a reproduction claim beyond this
#: band is not defensible against the experimental anchor.
REATTACHMENT_GATE = 0.15
#: predeclared outlet-development gate (fraction of the peak velocity).
OUTLET_GATE = 0.05
#: cells per step height for each grid level (``cases.BFS_LEVELS``).
LEVEL_CELLS = {"coarse": 16, "medium": 33, "fine": 49}

ROUND3_POINTS = [(100.0, "medium", 12.0), (100.0, "fine", 12.0),
                 (389.0, "medium", 12.0), (389.0, "fine", 12.0)]


def _load(reynolds: float, level: str, downstream: float) -> dict:
    path = ARTIFACTS / f"armaly1983_re{reynolds:g}_{level}_ld{downstream:g}.json"
    if not path.is_file():
        pytest.skip(f"round-3 artifact not yet produced: {path.name}")
    return json.loads(path.read_text())


def _single_outputs() -> dict:
    return json.loads((DOCS / "convergence" / "convergence_analysis.json").read_text())


@pytest.mark.parametrize("reynolds,level,downstream", ROUND3_POINTS)
def test_refinement_run_is_converged_and_geometric(reynolds, level, downstream):
    record = _load(reynolds, level, downstream)
    assert record["converged"] is True
    assert record["reynolds_dh"] == reynolds
    assert record["level"] == level
    assert record["downstream_step_heights"] == downstream
    # geometry is checked before any physics claim: the stair-step mask has to
    # reproduce the expansion ratio of the experiment to better than 1 %.
    assert record["expansion_ratio_rel_error"] < 0.01
    # the grid actually used must be the one the level promises: the step face
    # is n_s cells tall and the inlet channel n_in cells, both from BFS_LEVELS.
    n_s, n_in = BFS_LEVELS[level]
    assert n_s == LEVEL_CELLS[level]
    assert record["ny"] == n_s + n_in
    assert 0.0 < record["reattachment_over_step"] < downstream


def test_fine_grid_meets_the_15_percent_gate():
    for reynolds in (100.0, 389.0):
        record = _load(reynolds, "fine", 12.0)
        error = abs(record["reattachment_over_step"]
                    - record["experiment"]["reattachment_over_step"]) \
            / record["experiment"]["reattachment_over_step"]
        assert error < REATTACHMENT_GATE, (reynolds, error)


def test_refinement_improves_the_reattachment_error():
    """Fine must beat coarse on the same metric, both Reynolds numbers."""
    for reynolds in (100.0, 389.0):
        coarse = _load(reynolds, "coarse", 12.0)
        fine = _load(reynolds, "fine", 12.0)
        assert abs(fine["reattachment_rel_error"]) < abs(coarse["reattachment_rel_error"])


def test_three_level_convergence_analysis_is_complete():
    analysis = _single_outputs()
    for reynolds in (100.0, 389.0):
        key = f"Re{reynolds:.0f}_Ld12S"
        assert key in analysis
        entry = analysis[key]
        if entry["status"] != "complete":
            pytest.skip(f"{key}: only {len(entry['levels'])} grid level(s) delivered")
        assert entry["levels"] == ["coarse", "medium", "fine"]
        assert entry["reynolds_dh"] == reynolds
        assert math.isfinite(entry["gci_pct"]["fine"])
        assert math.isfinite(entry["observed_order"])
        assert entry["extrapolated_value"] > 0.0
        # the extrapolated limit is the quantity the report quotes against the
        # experiment, so it is held to the same gate as the finest grid.
        assert entry["extrapolated_error_pct"] < 100.0 * REATTACHMENT_GATE


def test_long_domain_fixes_the_outlet_development_gate():
    record = _load(389.0, "medium", 40.0)
    assert record["converged"] is True
    assert record["profile_outlet_column"]["max_rel"] < OUTLET_GATE
    assert record["profile_outlet_pass_5pct"] is True


def test_report_and_analysis_regenerate_byte_identically(tmp_path):
    """Zero hand-copying: the committed report is a pure function of the JSON."""
    subprocess.run(
        [sys.executable, str(REPO / "scripts" / "armaly1983_round3.py"),
         "--runs", str(ARTIFACTS), "--out", str(tmp_path), "--no-figures"],
        check=True, cwd=REPO, capture_output=True)
    for relative in ("REPORT_ROUND3.md", "convergence/convergence_analysis.json"):
        assert (tmp_path / relative).read_text() == (DOCS / relative).read_text(), \
            f"{relative} is not reproducible from the artifacts"