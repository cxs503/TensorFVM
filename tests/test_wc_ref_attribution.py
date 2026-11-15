"""Regression gates for the certified-<1 % *attribution* round (anchor XVAL-FVM-M18737-ROUND2).

``CERTIFIED-REF-CLOSURE.md`` already established the honest FAIL (L1 = 1.776 %
> 1 %).  This gate guards the *attribution* evidence added afterwards:

1. the reference-floor pair (WENO5@12800 vs @6400) still moves by <1 %, so the
   reference family is the tight one used for grading;
2. the headline L1 carries a metric band whose two conventions agree to <0.1
   percentage points -- i.e. the "not <1 %" verdict is not an artefact of the
   interpolation direction;
3. the graded residual is bounded (1 % <= L1 < 2 %): the round-2 honest-FAIL
   statement is a *bounded* claim, not a pinned snapshot (an improved solver
   must be free to pass this gate while a regression past 2 % fails it);
4. the region split is a partition (shares sum to 100 %, widths sum to 1);
5. the WENO7 probe's derived coefficients are the textbook ones and the
   audited WENO5 scheme really is ~5th order on a smooth entropy wave (the
   claim the reference rests on).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
XVAL = REPO / "docs" / "woodward-colella-double-blast" / "xval"
XCODE = REPO / "docs" / "woodward-colella-double-blast" / "xcode"
for p in (XVAL, XCODE):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import wc_ref_attribution as wca  # noqa: E402
import wc_weno7 as w7            # noqa: E402


@pytest.fixture(scope="module")
def analysis() -> dict:
    return wca.run()


# --- 1/2/3: the graded number and its honesty ---------------------------------
def test_reference_floor_is_under_one_percent(analysis):
    floor = analysis["reference_floor"]
    assert floor["pair_12800_vs_6400_l1_percent"] < 1.0, floor


def test_metric_band_is_narrow(analysis):
    lo, hi = analysis["l1_band_percent"]
    assert hi - lo < 0.1, analysis["l1_band_percent"]
    assert hi < 2.0


def test_graded_residual_is_bounded_not_pinned(analysis):
    # honest FAIL: the <1 % bar is not met ...
    assert analysis["l1_percent"] >= 1.0, analysis["l1_percent"]
    assert analysis["certified_under_1_percent"] is False
    # ... but it stays under the honest 2 % ceiling, and under the ledger bar
    assert analysis["l1_percent"] < 2.0, analysis["l1_percent"]
    assert analysis["ledger_under_10_percent"] is True
    assert analysis["peak_percent"] < 1.0, analysis["peak_percent"]


def test_window_mass_is_conserved(analysis):
    assert analysis["mass_percent"] < 0.1, analysis["mass_percent"]


# --- 4: the region split is a partition ---------------------------------------
def test_regions_partition_the_window(analysis):
    reg = analysis["regions"]
    share = sum(reg[k]["share_of_residual"] for k in ("shock", "plateau", "smooth"))
    width = sum(reg[k]["window_fraction"] for k in ("shock", "plateau", "smooth"))
    assert share == pytest.approx(100.0, abs=0.5), share
    assert width == pytest.approx(1.0, abs=1e-6), width
    assert reg["shock"]["l1_percent_of_ref"] >= 0.0


def test_graded_side_dominates_and_is_consistent_with_our_own_order(analysis):
    att = analysis["attribution"]
    assert att["graded_side_dominates"] is True, att
    # our own ladder (reference-free) must agree with the graded-side bracket
    assert 0.5 <= att["own_order"] <= 1.5, att["own_order"]
    lo, hi = att["graded_side_bracket_percent"]
    assert lo <= att["own_discretisation_error_percent"] <= 2.0 * hi, att


# --- 5: the WENO7 probe's coefficients are the textbook ones -------------------
def test_fv_face_coefficients_reproduce_the_textbook_three_cell_stencil():
    c = w7._fv_face_coeffs((-2, -1, 0))
    assert np.allclose(c, [1.0 / 3.0, -7.0 / 6.0, 11.0 / 6.0], atol=1e-12), c


def test_optimal_weights_are_derived_and_exact():
    assert np.allclose(w7.LIN_WEIGHTS, [1 / 35, 12 / 35, 18 / 35, 4 / 35], atol=1e-12)
    exact7 = w7._fv_face_coeffs(np.arange(-3, 4))
    assert np.abs(w7.optimal_face_coeffs() - exact7).max() < 1e-12


def test_beta_matrices_are_symmetric_and_vanish_on_constants():
    for Q in w7.BETA_MATS:
        assert np.abs(Q - Q.T).max() < 1e-12
        assert abs(float(np.ones(4) @ Q @ np.ones(4))) < 1e-12
        ramp = np.arange(4, dtype=float)
        assert float(ramp @ Q @ ramp) >= 0.0


def test_audited_weno5_scheme_is_high_order_on_a_smooth_entropy_wave():
    conv = w7.convergence_order(sizes=(80, 160, 320), cfl_exp=1.0)
    assert conv["weno5_audited"]["order_finest"] > 4.0, conv["weno5_audited"]
    # the WENO7 probe is honestly *not* 7th order with JS weights -- record it
    assert conv["weno7"]["order_finest"] < 7.0