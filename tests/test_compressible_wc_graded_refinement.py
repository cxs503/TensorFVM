"""Regression gate for the graded-family refinement closure (WC double blast).

Guards the claims of ``docs/woodward-colella-double-blast/xval/
wc_graded_refinement.py`` (anchor ``XVAL-FVM-M18737-ROUND2``, L1 gate of the
certified-<1 % claim).  Everything is derived from stored ladder profiles, so the
gate is pure numpy (a few seconds, no solver call).

Guarded invariants:

1. The independent reference is still self-certified <1 % (WENO5@6400 vs 12800),
   otherwise no grading number below it means anything.
2. The published round-2 anchors are reproduced exactly (800 -> 6.046 %,
   1600 -> 3.299 %, 3200 -> 1.776 % L1 vs the WENO5@12800 reference); these are
   the numbers ``CERTIFIED-REF-CLOSURE.md`` reports, so a drift here means the
   metric convention changed silently.
3. The graded family converges monotonically towards the reference on every
   adjacent pair present in the ladder (no refinement may *increase* L1).
4. The floor classification is consistent: ``NO_FLOOR_DETECTED`` <=> the clamped
   floor is 0 and the raw estimate is <= 0; ``GRID_LIMITED``/``SCHEME_LIMITED``
   <=> a positive floor below/above the 1 % bar.
5. The verdict is consistent: ``smallest_grid_under_1pct_l1`` equals the first
   rung whose L1 is <1 %, and ``CERTIFIED_UNDER_1PCT_AT_NX<N>`` is only emitted
   when such a rung exists (peak must be <1 % there too).
6. Conservation: window mass ``∫ρ dx`` on [0.4,1] stays <0.1 % on the finest
   graded rung (format-independent check).

The 6400/12800 rungs are *pending* cluster work (``runs/m21429/wc_ladder``);
if they are not in ``ladder/data/`` yet the tests still pass on 800/1600/3200 and
start covering the finer rungs automatically once they land -- that is the point
of deriving everything from stored profiles.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "woodward-colella-double-blast" / "xval"
if str(XVAL) not in sys.path:
    sys.path.insert(0, str(XVAL))

import wc_graded_refinement as g  # noqa: E402


@pytest.fixture(scope="module")
def analysis() -> dict:
    return g.run()


def test_reference_is_self_certified(analysis):
    r = analysis["reference"]
    assert r["self_certified"] is True
    assert r["self_certified_l1_percent"] < 1.0
    assert r["self_certified_peak_percent"] < 1.0


def test_published_anchors_are_reproduced(analysis):
    """The three rungs reported by CERTIFIED-REF-CLOSURE.md must not drift."""
    want = {"800": 6.046, "1600": 3.299, "3200": 1.776}
    for nx, l1 in want.items():
        assert nx in analysis["ladder"], f"rung {nx} vanished from the ladder"
        got = analysis["ladder"][nx]["l1_percent"]
        assert abs(got - l1) < 5e-3, (nx, got, l1)


def test_l1_converges_monotonically_towards_reference(analysis):
    for pair, r in analysis["order_vs_reference"].items():
        assert r["monotone"] is True, (pair, r)
        assert r["fine_l1"] < r["coarse_l1"]


def test_self_convergence_is_monotone(analysis):
    """Reference-independent: the graded family must also converge to itself."""
    sc = analysis["self_convergence_l1_percent"]
    assert sc, "no adjacent graded pair available"
    for pair, val in sc.items():
        coarse = int(pair.split("->")[0])
        prev = f"{coarse // 2}->{coarse}"
        if prev in sc:
            assert val < sc[prev], (pair, val, sc[prev])


def test_floor_classification_is_consistent(analysis):
    a = analysis["asymptote"]
    f = a["fixed_order"]
    assert f["ok"] is True, f
    raw, clamped, cls = (f["e_inf_percent_raw"], f["e_inf_percent"],
                         a["floor_class"])
    if cls == "NO_FLOOR_DETECTED":
        assert raw <= 0.0 <= clamped
        assert a["floor_detected"] is False
    elif cls == "GRID_LIMITED":
        assert 0.0 < raw < 1.0 and clamped == raw
        assert a["floor_detected"] is True
    else:
        assert cls == "SCHEME_LIMITED"
        assert raw >= 1.0 and clamped == raw


def test_prediction_and_verdict_are_consistent(analysis):
    a = analysis["asymptote"]
    preds = a["predicted_l1_percent"]
    assert preds, "power-law predictions missing"
    vals = list(preds.values())
    assert all(v2 < v1 for v1, v2 in zip(vals, vals[1:])), preds

    under = [int(n) for n, r in analysis["ladder"].items()
             if r["l1_percent"] < 1.0]
    assert analysis["smallest_grid_under_1pct_l1"] == (min(under) if under else None)
    st = analysis["status"]
    if under:
        assert st == f"CERTIFIED_UNDER_1PCT_AT_NX{min(under)}", st
        # a <1 % L1 claim is only made where the peak is <1 % too
        assert analysis["ladder"][str(min(under))]["peak_percent"] < 1.0
    else:
        assert not st.startswith("CERTIFIED_UNDER_1PCT"), st


def test_finest_rung_is_conservative(analysis):
    finest = max(analysis["ladder"], key=int)
    assert analysis["ladder"][finest]["mass_percent"] < 0.1


def test_pending_rungs_are_declared_not_silently_missing(analysis):
    """A rung that is absent must be *listed* as missing (no silent drop)."""
    assert sorted(analysis["missing_rungs"]) == sorted(
        n for n in g.GRADED if str(n) not in analysis["ladder"])