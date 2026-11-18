"""Round-3 reopen gates for the certified-reference closure (m24511).

PR #943 (m21153 deliverables) was ruled closed on 2026-11-05 and the mission
reopened; this gate re-pins the closure on the current main:

1. the machine-readable ``note`` emitted by ``wc_certified_ref.run()`` must
   agree with the ``certified`` flag in the same verdict (a stale hardcoded
   note previously claimed the honest bound "stays <1 %" while the flag said
   ``certified=False`` at honest bound 1.6007 %);
2. the numbers quoted in ``CERTIFIED-REF-CLOSURE.md`` must still match the
   stored ladder artifacts (doc-vs-data drift gate).

Pure numpy on stored profiles -- no solver call.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "woodward-colella-double-blast" / "xval"
if str(XVAL) not in sys.path:
    sys.path.insert(0, str(XVAL))

import wc_certified_ref as wcr  # noqa: E402

CLOSURE = (REPO / "docs" / "XVAL-FVM-M18737-ROUND2"
           / "CERTIFIED-REF-CLOSURE.md")


@pytest.fixture(scope="module")
def verdict() -> dict:
    return wcr.run()


def test_note_agrees_with_certified_flag(verdict):
    """The note must state the actual verdict, not a stale hardcoded claim."""
    note = verdict["note"]
    assert ("certified=True" in note) == verdict["certified"]
    # the stale pre-errata phrasing must not come back
    assert "stays <1%" not in note
    # the honest upper bound must appear numerically in the note ...
    bound = verdict["honest_upper_bound_percent"]
    assert f"{bound:.4f}%" in note
    # ... with the correct comparison against the 1 % certified bar
    ok = bound < verdict["certified_bar_percent"]
    assert ("bound < 1%" if ok else "bound >= 1%") in note


def test_honest_bound_pins_certified_false(verdict):
    """Round-3 state: point estimate True, honest bound 1.6007 % > 1 %."""
    assert verdict["certified_point_estimate"] is True
    assert verdict["certified"] is False
    assert verdict["honest_upper_bound_percent"] == pytest.approx(1.6007, abs=0.01)


def test_closure_doc_selfconvergence_matches_data():
    """Doc table: WENO5@6400 vs 12800 L1 0.544 % / peak 0.023 %."""
    a = wcr._load("nx6400_weno5")
    b = wcr._load("nx12800_weno5")
    l1 = wcr.l1_percent(a["x"], a["rho"], b["x"], b["rho"])
    pk = wcr.peak_percent(a["x"], a["rho"], b["x"], b["rho"])
    assert l1 == pytest.approx(0.544, abs=0.005)
    assert pk == pytest.approx(0.023, abs=0.005)
    text = CLOSURE.read_text()
    assert "0.544" in text and "0.023" in text


def test_closure_doc_graded_rung_matches_data():
    """Doc table: MUSCL/HLLC nx3200 vs WENO5@12800 L1 1.776 % / peak 0.758 %."""
    a = wcr._load("nx3200_hllc")
    b = wcr._load("nx12800_weno5")
    l1 = wcr.l1_percent(a["x"], a["rho"], b["x"], b["rho"])
    pk = wcr.peak_percent(a["x"], a["rho"], b["x"], b["rho"])
    ms = wcr.mass_percent(a["x"], a["rho"], b["x"], b["rho"])
    assert l1 == pytest.approx(1.776, abs=0.005)
    assert pk == pytest.approx(0.758, abs=0.005)
    assert ms == pytest.approx(0.008, abs=0.005)


def test_closure_doc_ladder_rows_match_metrics_json():
    """Doc table '窗 L1 (vs 已发表参考)' rows: 5.02 % @6400, 5.48 % @12800."""
    text = CLOSURE.read_text()
    for nx, quoted in ((6400, "5.02"), (12800, "5.48")):
        d = json.loads((wcr.DATA / f"ladder_metrics_nx{nx}_weno5.json").read_text())
        got = d["l1_relative"]["percent"]
        assert got == pytest.approx(float(quoted), abs=0.005)
        assert quoted in text


def test_fresh_profiles_present_and_finite():
    """The two rungs PR #943 delivered must stay loadable and finite."""
    for nx in (6400, 12800):
        z = np.load(wcr.DATA / f"ladder_profile_nx{nx}_weno5.npz")
        assert set(("x", "rho", "p", "u")) <= set(z.files)
        for k in ("x", "rho", "p", "u"):
            assert np.all(np.isfinite(z[k])), (nx, k)
