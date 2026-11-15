"""Regression gate for the reference-side budget machinery (``wc_ref_uncertainty``).

``src/tensorfvm/verification/wc_ref_uncertainty.py`` turns a measured grade into an honest
one by attaching the *reference's own* uncertainty, ``U_ref = e/(2**p - 1)``.  Every number
the ``XVAL-FVM-M18737-ROUND2`` closures quote depends on it, so the invariants are pinned
here with **synthetic ladders whose exact answer is known**: if the machinery is fed profiles
whose discretisation error is ``C h**p`` by construction, it must return that ``p`` and that
``C h_fine**p``.  A shared mistake in the machinery and in the real ladder would otherwise be
invisible.

Covered:

1. :func:`selfcert_ladder` recovers a synthetic, exactly known order and ``U_ref``;
2. a three-rung ladder (two pairs) yields a *measured* finest-scale order -- the case that
   round-4 mislabelled ``extrapolated`` -- and a single pair yields no claim at all;
3. :func:`u_ref_order_band` propagates an order band monotonically and brackets the point
   order's ``U_ref``;
4. :func:`adjudicate_prediction` says "unadjudicable" (not "hit") when nothing was frozen;
5. on the stored W&C ladder (when present) the extended reference is self-consistent:
   its ``U_ref`` equals ``e/(2**p - 1)`` for the order the ladder actually reports.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import wc_ref_uncertainty as U
from tensorfvm.verification import wc_refine as wcr

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "docs" / "woodward-colella-double-blast" / "ladder" / "data"


def write_synthetic_ladder(tmp_path: Path, rungs, order: float, amplitude: float,
                           family: str = "weno5") -> Path:
    """Profiles whose cell-average error is exactly ``amplitude * h**order``.

    ``q(x) = 1 + x`` is linear, so its cell average equals its centre value on every grid;
    the discretisation error is added as a constant offset, so the conservative restriction
    of the fine profile differs from the coarse profile by ``A (h_c**p - h_f**p)`` -- exactly
    the quantity the Richardson estimator consumes.
    """
    for nx in rungs:
        x = (np.arange(nx) + 0.5) / nx
        h = 1.0 / nx
        rho = 1.0 + x + amplitude * h ** order
        np.savez_compressed(tmp_path / f"ladder_profile_nx{nx}_{family}.npz",
                            x=x, rho=rho, p=np.ones(nx), u=np.zeros(nx))
    return tmp_path


def _mean_q() -> float:
    """Mean of ``q(x)=1+x`` over the comparison window x in [0.4, 1.0]."""
    return 1.0 + 0.5 * (wcr.WINDOW[0] + wcr.WINDOW[1])


def test_selfcert_ladder_recovers_known_order_and_uncertainty(tmp_path):
    order, amp = 0.75, 3.0
    write_synthetic_ladder(tmp_path, (1600, 3200, 6400), order, amp)
    ladder = U.selfcert_ladder(tmp_path)
    finest = ladder[-1]
    assert finest["fine_nx"] == 6400
    # order is a property of the *sequence*, so the three-rung ladder measures it
    assert finest["order"] == pytest.approx(order, abs=5e-3)
    expected = 100.0 * amp * (1.0 / 6400) ** order / _mean_q()
    assert finest["u_ref_percent"] == pytest.approx(expected, rel=2e-3)
    # and the self-certification difference is the ratio the estimator inverts
    assert finest["l1_percent"] == pytest.approx(expected * (2 ** order - 1), rel=1e-2)


def test_three_rung_ladder_measures_its_finest_order(tmp_path):
    """Two pairs = three grids: the finest order is *measured*, not extrapolated."""
    write_synthetic_ladder(tmp_path, (1600, 3200, 6400), 1.0, 1.0)
    ladder = U.selfcert_ladder(tmp_path)
    assert len(ladder) == 2
    finest = ladder[-1]
    assert finest["order_source"] == "measured_triple"
    assert finest["order_scale"]["fine_nx"] == 3200
    assert finest["u_ref_percent"] is not None
    # the coarser pair borrows the same three-grid order (its own successor is the finest
    # pair), so both rows quote one measured p -- not two different ones
    assert ladder[0]["order_source"] == "measured_pair"
    assert ladder[0]["order"] == pytest.approx(finest["order"])


def test_single_pair_makes_no_claim(tmp_path):
    write_synthetic_ladder(tmp_path, (1600, 3200), 1.0, 1.0)
    ladder = U.selfcert_ladder(tmp_path)
    assert len(ladder) == 1
    assert ladder[0]["order"] is None and ladder[0]["u_ref_percent"] is None
    ur = U.u_ref_of_reference(tmp_path, 3200, ladder=ladder)
    assert ur["u_ref_percent"] is None
    assert ur["measurable"] is True          # the rung exists, but nothing is known about it
    # "self_certified" only says the pair's *difference* is below the bar; it is not a
    # statement about U_ref, which stays unknown here
    assert ur["self_certified"] == (ladder[0]["l1_percent"] < U.BAR_PERCENT)


def test_u_ref_order_band_brackets_the_point_value():
    e = 0.3094
    band = U.u_ref_order_band(e, (0.5, 1.1))
    point = U.u_ref_from_pair(e, 0.8133)
    assert band["min_percent"] <= point <= band["max_percent"]
    # U_ref decreases with the order: the higher order gives the lower uncertainty
    assert band["min_percent"] == pytest.approx(U.u_ref_from_pair(e, 1.1))
    assert band["max_percent"] == pytest.approx(U.u_ref_from_pair(e, 0.5))
    assert band["spread_percent"] == pytest.approx(band["max_percent"] - band["min_percent"])
    with pytest.raises(ValueError):
        U.u_ref_order_band(e, (0.0, 1.0))


def test_adjudicate_prediction_distinguishes_miss_and_unadjudicable():
    hit = U.adjudicate_prediction(0.31, point=0.3094, band=(0.232, 0.387))
    assert hit["verdict"] == "hit" and hit["inside_band"] is True
    miss = U.adjudicate_prediction(0.90, point=0.3094, band=(0.232, 0.387))
    assert miss["verdict"] == "miss" and miss["deviation_percent"] > 0
    tol = U.adjudicate_prediction(1.10, point=1.00, tolerance_percent=15.0)
    assert tol["verdict"] == "hit"
    unknown = U.adjudicate_prediction(0.5)
    assert unknown["verdict"] == "unadjudicable" and unknown["deviation_percent"] is None


@pytest.mark.skipif(wcr.try_load_profile(DATA, 25600, "weno5") is None,
                    reason="extended WENO5@25600 reference not present in this worktree")
def test_extended_reference_budget_is_self_consistent():
    """The U_ref claimed for the extended reference is the formula, not a fudge factor."""
    ladder = U.selfcert_ladder(DATA)
    row = next(r for r in ladder if r["fine_nx"] == 25600)
    assert row["order"] is not None and row["order"] > 0.0
    assert row["u_ref_percent"] == pytest.approx(
        row["l1_percent"] / (2.0 ** row["order"] - 1.0), rel=1e-12)
    # a finer reference must be more self-converged than the one it extends
    coarser = next(r for r in ladder if r["fine_nx"] == 12800)
    assert row["l1_percent"] < coarser["l1_percent"]
    prof = wcr.load_profile(DATA, 25600, "weno5")
    assert prof["nx"] == 25600
    assert np.all(np.isfinite(prof["rho"])) and prof["rho"].min() > 0.0