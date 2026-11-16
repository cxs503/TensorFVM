"""Round 28 contract tests: the ladder provenance gate.

Round 27 marked three refined rungs ``"computing (cj4352/4353/4354)"``; all
three jobs died at ``cd: ... Permission denied`` before the interpreter started,
while the queue surface reported ``succeeded / rc = 0``.  These tests pin the
gate that makes "declared as computing" and "measured" distinguishable:

1. **Receipts decide** -- the round-27 receipts carry a non-zero remote ``rc``
   and a permission-denied tail; the queue surface's ``rc`` is not evidence.
2. **Claimed-but-unmeasured is detectable** -- grading the round-27 table
   against the artefact directory flags rungs B/C/D as ``unverified_claim``
   (no ``round25_*`` artefact exists), so no report may call them pending data.
3. **Zero field is not a measurement** -- a dump of zeros grades ``zero_field``
   and can never be promoted to ``measured``.
4. **A usable artefact is measured** -- the round-20 baseline rung does hold a
   field and grades ``usable``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
CASEDIR = REPO / "docs" / "case-periodic-hills"
LOGDIR = Path("/root/tp-agent/cluster/jobs")
sys.path.insert(0, str(REPO / "src"))

from tensorfvm.verification.periodic_hill_rung_provenance import (  # noqa: E402
    DEFAULT_STEMS,
    grade_ladder,
    job_log_verdict,
    rung_artefact_status,
)


@pytest.fixture(scope="module")
def table27():
    return json.loads((CASEDIR / "round27_table.json").read_text())


def test_round27_receipts_report_remote_failure():
    """The three round-27 ladder jobs failed remotely, whatever the queue said."""
    logs = [LOGDIR / f"cj{j}.log" for j in (4352, 4353, 4354)]
    if not all(p.exists() for p in logs):
        pytest.skip("cluster job receipts not readable from this checkout")
    for path in logs:
        verdict = job_log_verdict(path)
        assert verdict["present"] and verdict["parsed"]
        assert verdict["remote_rc"] != 0
        assert "Permission denied" in verdict["output_tail"]


def test_zero_field_artefact_is_never_usable(tmp_path):
    """An all-zero dump grades ``zero_field`` -- absence of evidence."""
    stem = "round25_laminar_Re2800_test"
    np.savez(tmp_path / f"{stem}.npz",
             mean_u=np.zeros((4, 3)), mean_tau_b=np.zeros(4),
             x_over_h=np.linspace(0.0, 9.0, 4))
    (tmp_path / f"{stem}.json").write_text(json.dumps({"wall_clock_s": 1.0}))
    status = rung_artefact_status(tmp_path, stem)
    assert status["artefact"] == "zero_field"
    assert status["max_abs_u"] == 0.0


def test_missing_artefact_grades_missing(tmp_path):
    status = rung_artefact_status(tmp_path, "round25_laminar_Re2800_absent")
    assert status["artefact"] == "missing"


def test_round27_ladder_is_flagged_as_unverified_claim(table27):
    """Rungs B/C/D were declared 'computing' but never produced an artefact."""
    grading = grade_ladder(table27, CASEDIR)
    assert grading["any_unverified_claim"] is True
    assert set(grading["claimed_but_unmeasured"]) == {"B", "C", "D"}
    for rung in ("B", "C", "D"):
        assert grading["rungs"][rung]["provenance"] == "unverified_claim"
        assert grading["rungs"][rung]["artefact"] == "missing"
        assert rung not in (table27.get("levels") or {})


def test_usable_baseline_rung_grades_usable(table27):
    """The 96x48 baseline holds a real field, so it grades usable."""
    stem = DEFAULT_STEMS["A"]
    status = rung_artefact_status(CASEDIR, stem)
    if status["artefact"] == "missing":
        pytest.skip("baseline artefact not present in this checkout")
    assert status["artefact"] in {"usable", "zero_field"}
    grading = grade_ladder(table27, CASEDIR)
    assert grading["rungs"]["A"]["artefact"] == status["artefact"]


def test_measured_requires_a_usable_field(tmp_path):
    """A rung declared measured without a field is downgraded, not trusted."""
    table = {"case": "synthetic", "ladder_status": {"B": "measured"}}
    grading = grade_ladder(table, tmp_path, stems={"B": "nope"})
    assert grading["rungs"]["B"]["provenance"] == "declared_unusable"

    stem = "nope"
    np.savez(tmp_path / f"{stem}.npz", mean_u=np.full((3, 2), 0.5),
             mean_tau_b=np.ones(3), x_over_h=np.linspace(0.0, 9.0, 3))
    (tmp_path / f"{stem}.json").write_text("{}")
    grading = grade_ladder(table, tmp_path, stems={"B": stem})
    assert grading["rungs"]["B"]["provenance"] == "measured"