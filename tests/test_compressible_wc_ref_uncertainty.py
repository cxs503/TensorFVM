"""Regression gate for the reference-side uncertainty budget (W&C double blast).

The round-2 closure (``wc_certified_ref.py``, gate 1) certifies its reference
with a **two-level** check, which cannot produce an observed order or an error
bar.  ``wc_ref_uncertainty.py`` supplies the missing third level and the
Richarson extrapolation; this gate pins the invariants that make its numbers
trustworthy rather than merely reproducible:

1. the WENO5 ladder really is three levels deep (without that the whole budget
   falls back to the two-level bound and says so);
2. the reference family self-converges (successive-grid L1 distances decrease);
3. the error bar is the Richardson formula applied to those distances
   (``eps = d_fine / (2**p - 1)``), not an independent fudge factor;
4. the extrapolated reference field is the Richardson combination of the two
   finest WENO5 profiles and conserves the window mass to <0.1 %;
5. the two *independent* error routes for the graded solution -- cross-family
   vs the extrapolated reference, and the reference-free intra-family
   Richardson estimate -- agree to within the reference's own uncertainty
   (this is the reconciliation the report claims);
6. the verdict's "<1 % provable" flag is exactly ``upper_bound < 1 %``, so the
   honest downgrade cannot be edited away without failing a test.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "woodward-colella-double-blast" / "xval"
if str(XVAL) not in sys.path:
    sys.path.insert(0, str(XVAL))

import wc_ref_uncertainty as unc  # noqa: E402


@pytest.fixture(scope="module")
def budget() -> dict:
    return unc.run()


@pytest.fixture(scope="module")
def ref(budget) -> dict:
    return budget["reference"]


@pytest.fixture(scope="module")
def grading(budget) -> dict:
    return budget["grading"]


# ---------------------------------------------------------------- ladder ----
def test_weno5_ladder_has_three_levels(ref):
    assert ref["status"] == "OK", ref.get("note")
    assert len(ref["ladder_available"]) >= 3, ref["ladder_available"]
    assert ref["missing"] == [], ref["missing"]


def test_reference_family_self_converges(ref):
    """Successive-grid distances must shrink as the mesh is refined."""
    ds = [s["l1_percent"] for s in ref["successive"]]
    assert len(ds) >= 2
    assert all(d > 0.0 for d in ds)
    assert ds[-1] < ds[0], ds


def test_error_bar_is_the_richardson_formula(ref):
    d_fine = ref["successive"][-1]["l1_percent"]
    p = ref["observed_order"]
    assert ref["order_reason"] == "none"
    assert unc.ORDER_BAND[0] <= p <= unc.ORDER_BAND[1]
    assert ref["eps_ref_percent"] == pytest.approx(d_fine / (2.0 ** p - 1.0), rel=1e-12)
    assert ref["eps_ref_gci_percent"] == pytest.approx(
        unc.SAFETY_FACTOR * ref["eps_ref_percent"], rel=1e-12)
    # the order-based bar must be < 2x the naive two-level distance it replaces
    assert ref["eps_ref_percent"] < 2.0 * d_fine


def test_error_bar_is_metric_dependent(ref):
    """The peak converges at a much higher order than the L1 field distance, so
    its bar must be tighter; one bar for both metrics would be wrong."""
    pk = ref["peak"]
    assert pk["observed_order"] > ref["observed_order"], (pk, ref["observed_order"])
    assert 0.0 < pk["eps_ref_percent"] < ref["eps_ref_percent"]
    assert pk["eps_ref_percent"] == pytest.approx(
        pk["l1_percent_two_level"] / (2.0 ** pk["observed_order"] - 1.0), rel=1e-12)


def test_published_reference_is_not_used_as_an_anchor(ref):
    """The digitised published curve must stay a diagnostic: its residual grows
    with our refinement, so it cannot certify anything."""
    diag = ref["published_reference_diagnostic"]
    assert "finest_weno5_vs_published_percent" in diag
    assert diag["extrapolated_vs_published_percent"] > diag["finest_weno5_vs_published_percent"]


def test_extrapolated_reference_conserves_mass(ref):
    assert ref["mass_extrapolated_vs_fine_percent"] < 0.1


def test_extrapolated_field_is_the_richardson_combination():
    """rho_inf = rho_f + (rho_f - rho_c)/(2**p - 1), checked pointwise."""
    fine = unc._load(12800, "weno5")
    coarse = unc._load(6400, "weno5")
    p = (np.log(unc.l1_percent(unc._load(6400, "weno5"), unc._load(3200, "weno5"))
                / unc.l1_percent(fine, coarse)) / np.log(2.0))
    field = unc.extrapolate_field(fine, coarse, float(p))
    rho_c = np.interp(field["x"], coarse["x"], coarse["rho"])
    expect = fine["rho"] + (fine["rho"] - rho_c) / (2.0 ** p - 1.0)
    assert np.allclose(field["rho"], expect, rtol=1e-13, atol=1e-13)
    # the extrapolation must move the profile *away* from the coarse one
    assert np.all(np.sign(field["rho"] - fine["rho"])
                  == np.sign(fine["rho"] - rho_c + 1e-30))


# --------------------------------------------------------------- grading ----
def test_graded_band_contains_the_corrected_estimate(grading):
    """The two-route reconciliation: the cross-family estimate vs the
    extrapolated reference must fall inside the triangle band built from the
    raw distance and the reference error bar."""
    assert grading["status"] == "OK"
    assert grading["graded"], "no graded rung found"
    for nx, g in grading["graded"].items():
        assert g["band_contains_extrapolated"], (nx, g)


def test_routes_agree_within_reference_uncertainty(grading):
    v = grading["verdict"]
    assert v["route_2_intra_family_percent"] is not None, "intra-family route missing"
    assert v["routes_agree_percent"] < v["reference_uncertainty_percent"], v


def test_finest_graded_rung_is_the_finest_rung(grading):
    graded = grading["graded"]
    assert grading["verdict"]["graded_nx"] == max(int(k) for k in graded)


def test_honest_bound_and_provability_flag(grading):
    v = grading["verdict"]
    assert v["honest_upper_bound_percent"] == pytest.approx(
        v["route_1_cross_family_percent"] + v["reference_uncertainty_percent"], rel=1e-12)
    assert v["certified_under_1pct_provable"] == (v["honest_upper_bound_percent"] < 1.0)
    assert v["certified_under_1pct_point_estimate"] == (
        v["route_1_cross_family_percent"] < 1.0)
    lo, hi = v["cross_family_interval_percent"]
    assert lo <= v["route_1_cross_family_percent"] <= hi


def test_intra_family_order_band_brackets_the_estimate(grading):
    """Order sensitivity: quoting the fitted order alone would understate the
    uncertainty, because the L1-on-shocks order drifts towards 1."""
    v = grading["verdict"]
    band = v["route_2_order_band_percent"]
    assert band is not None
    lo, hi = band
    assert lo <= hi
    assert lo <= v["route_2_intra_family_percent"] <= hi


# ------------------------------------------------------- guards / edges -----
def test_missing_rung_returns_none():
    assert unc._load(999999, "hllc") is None


def test_insufficient_ladder_is_reported_not_guessed(monkeypatch):
    """With fewer than three levels the budget must refuse to invent an order."""
    monkeypatch.setattr(unc, "WENO5_LADDER", (6400, 12800))
    out = unc.reference_uncertainty()
    assert out["status"] == "INSUFFICIENT_LADDER"
    assert "observed order" in out["note"]


def test_bad_distances_fall_back_with_reason():
    p, reason = unc.observed_order(1.0, 0.0)
    assert p == 1.0 and reason
    p, reason = unc.observed_order(-1.0, 0.5)
    assert p == 1.0 and reason
    p, reason = unc.observed_order(10.0, 1e-6)   # absurd order -> fallback
    assert p == 1.0 and "outside" in reason


def test_grade_skips_when_reference_budget_unavailable():
    out = unc.grade({"status": "INSUFFICIENT_LADDER"})
    assert out["status"] == "SKIPPED"