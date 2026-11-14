"""Regression gate for the round-2 cross-validation of CASE-WOODWARD-COLELLA.

The claims in ``docs/CASE-WOODWARD-COLELLA-round2-xval.md`` are invariants of the
stored profiles plus the two reference tables, so they are cheap to re-check
(~6 s, pure numpy, no solver call).  Guarded claims:

1. Our own discretisation error is monotone in the grid (reference-free
   grid-to-grid L1), and the finest rung is under 2 %.
2. The peak amplitude extrapolates to a continuum value ~3 % **above** the
   published (calibrated) reference peak -- i.e. the reference carries its own
   discretisation error at the peak; this test fails loudly if that statement is
   ever silently softened into "<3 % everywhere".
3. The earlier over-claim ("all key metrics < 3 % on every grid >= 800") is
   false at nx = 3200 and must stay recorded as such.
4. The error budget decomposition is arithmetically consistent, and the
   reference's steepest gradient is exceeded by our solution at fine grids
   (the physical reason the pointwise comparison cannot converge to zero).
5. Smoothing our profile cannot remove most of the residual, so the residual is
   a genuine profile difference, not a sharpness artefact.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "woodward-colella-double-blast" / "xval"
if str(XVAL) not in sys.path:
    sys.path.insert(0, str(XVAL))

import wc_xval  # noqa: E402


@pytest.fixture(scope="module")
def analysis() -> dict:
    return wc_xval.run()


def test_all_ladder_rungs_are_scored(analysis):
    tags = set(analysis["profiles"])
    for tag in ("nx400_hllc", "nx800_hllc", "nx1600_hllc", "nx3200_hllc"):
        assert tag in tags, f"ladder rung {tag} missing from the cross-validation"


def test_self_convergence_is_monotone(analysis):
    diffs = analysis["self_convergence"]["fvm"]["differences"]
    vals = [d["l1_window_percent"] for d in diffs]
    assert vals == sorted(vals, reverse=True), f"grid-to-grid L1 not monotone: {vals}"
    assert vals[-1] < 2.0, vals[-1]
    orders = analysis["self_convergence"]["fvm"]["observed_orders"]
    assert all(o > 0 for o in orders), orders


def test_peak_extrapolation_is_above_reference(analysis):
    pe = analysis["peak_extrapolation_fvm"]
    assert pe["asymptotic"] is True
    assert 2.0 <= pe["order"] <= 3.5, pe["order"]
    ref_peak = analysis["reference"]["calibrated"]["peak"]
    gap = 100 * (pe["peak_infinity"] - ref_peak) / ref_peak
    # the continuum peak exceeds the published peak by ~3 % (2.5 .. 3.8 %)
    assert 2.5 <= gap <= 3.8, gap


def test_earlier_under_3_percent_claim_is_not_restored(analysis):
    """nx=3200 calibrates to >3 % on peak and L1: the old claim must not return."""
    r = analysis["profiles"]["nx3200_hllc"]
    # 2.998 % -- printed as 3.00, so the guard is >= 2.9 (it must stay at the
    # 3 % line, i.e. the old "every grid < 3 %" statement cannot be reused)
    assert r["peak_percent_calibrated"] > 2.9, r["peak_percent_calibrated"]
    assert r["l1_percent_calibrated"] > 3.0, r["l1_percent_calibrated"]
    # ... while the ledger tolerance (<10 %) is met on every robust metric
    assert r["peak_percent_calibrated"] < analysis["ledger_tolerance_percent"]
    assert r["l1_percent_calibrated"] < analysis["ledger_tolerance_percent"]
    assert r["mass_percent_calibrated"] < 0.1


def test_error_budget_identity(analysis):
    b = analysis["error_budget_finest"]["l1"]
    recon = np.hypot(b["own_percent"], b["reference_side_percent"])
    assert abs(recon - b["total_percent"]) < 1e-9 * max(1.0, b["total_percent"])
    assert 0 < b["own_fraction_percent"] < 100


def test_reference_resolving_power_is_exceeded(analysis):
    ref_grad = analysis["reference"]["calibrated"]["max_gradient"]
    g3200 = analysis["profiles"]["nx3200_hllc"]["max_gradient"]
    g1600 = analysis["profiles"]["nx1600_hllc"]["max_gradient"]
    assert g3200 > ref_grad > g1600, (g1600, ref_grad, g3200)


def test_smoothing_does_not_explain_the_residual(analysis):
    sf = analysis["smoothing_floor_finest"]
    assert sf["best_percent"] <= sf["base_percent"]
    assert sf["reduction_percent"] < 0.6, sf["reduction_percent"]


def test_shift_scan_finds_no_global_phase_error(analysis):
    ss = analysis["shift_scan_finest"]
    assert abs(ss["best_shift"]) < 2e-3, ss["best_shift"]
    assert ss["reduction_percent"] < 0.2, ss["reduction_percent"]


def test_mass_moment_is_grid_independent(analysis):
    masses = [r["mass_percent_calibrated"] for r in analysis["profiles"].values()
              if r["family"] == "fvm"]
    assert max(masses) < 0.25, masses