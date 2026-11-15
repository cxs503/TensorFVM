"""Reference-side uncertainty budget (``wc_ref_uncertainty``): formula, gates, real data.

The point of these tests is that the budget machinery must be *falsifiable*: the Richardson
estimate has to recover an error that was injected with a known magnitude, the certificate
gate has to refuse a grade whose uncertainty is of its own size, and the real round-4 data
(when the extended reference exists) has to reproduce the predeclared prediction.  Merely
executing the code is not a test.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import wc_ref_uncertainty as U
from tensorfvm.verification import wc_refine as wcr

DATA = wcr.default_data_dir()


# --------------------------------------------------------------------------- #
# the Richardson estimate must recover an injected error
# --------------------------------------------------------------------------- #
def _manufactured_rungs(order: float, amplitude: float, rungs=(400, 800, 1600)):
    """Profiles whose error against the exact solution is ``amplitude * h**order * s(x)``.

    ``s(x) = sin(3 pi x)`` gives the error a spatial structure so that the L1 norms are not
    trivially equal to the pointwise magnitude; the error magnitude is grid independent
    (``amplitude * s``), which is exactly the asymptotic regime the Richardson estimate
    assumes.
    """
    out = {}
    for nx in rungs:
        x = (np.arange(nx) + 0.5) / nx
        h = 1.0 / nx
        # the exact solution is deliberately *not* included in q: the injected error is the
        # whole signal, so its L1 norm is known in closed form
        out[nx] = {"x": x, "exact": np.zeros_like(x),
                   "q": amplitude * h ** order * np.sin(3 * np.pi * x)}
    return out


def test_u_ref_recovers_injected_reference_error():
    rungs = (400, 800, 1600)
    prof = _manufactured_rungs(order=2.0, amplitude=0.5, rungs=rungs)
    # L1 difference of consecutive rungs, exactly as the real ladder computes it
    e = {}
    for coarse, fine in zip(rungs, rungs[1:]):
        diff = np.abs(prof[coarse]["q"] - wcr.restrict(prof[fine]["q"], fine // coarse))
        e[fine] = float(np.mean(diff)) + 1e-300
    p = math.log(e[800] / e[1600], 2)                       # observed order, not assumed
    assert p == pytest.approx(2.0, abs=1e-3)
    u = U.u_ref_from_pair(e[1600], p)
    true_fine = float(np.mean(np.abs(prof[1600]["q"] - prof[1600]["exact"])))
    # restriction averages the sub-cell error, so this recovers the injected error up to the
    # O(h^2) variation of sin(3 pi x) across a cell -- a wrong formula would be off by >= 3x
    assert u == pytest.approx(true_fine, rel=5e-3)


def test_u_ref_tracks_order_not_just_magnitude():
    # a first-order reference is much less trustworthy than a second-order one at same diff
    assert U.u_ref_from_pair(0.5, 1.0) == pytest.approx(0.5, rel=1e-12)
    assert U.u_ref_from_pair(0.5, 2.0) == pytest.approx(0.5 / 3.0, rel=1e-12)
    assert U.u_ref_from_pair(0.5, 3.0) == pytest.approx(0.5 / 7.0, rel=1e-12)


def test_u_ref_rejects_degenerate_orders():
    with pytest.raises(ValueError):
        U.u_ref_from_pair(0.5, 0.0)
    with pytest.raises(ValueError):
        U.u_ref_from_pair(0.5, -1.0)
    with pytest.raises(ValueError):
        U.u_ref_from_pair(-0.5, 2.0)


# --------------------------------------------------------------------------- #
# certificate gate
# --------------------------------------------------------------------------- #
def test_gate_refuses_certificate_when_uncertainty_is_of_its_own_size():
    # the round-3 situation: 0.8931 % measured against a reference good to 0.7181 %
    got = U.grade_with_budget(0.8931, 0.7181)
    assert got["verdict"] == "upper_bound_only"
    assert got["worst_case_percent"] == pytest.approx(0.8931 + 0.7181)
    assert got["margin_percent"] < 0


def test_gate_certifies_only_with_room_to_spare():
    assert U.grade_with_budget(0.20, 0.10)["verdict"] == "certified"
    assert U.grade_with_budget(0.20, 0.90)["verdict"] == "upper_bound_only"
    assert U.grade_with_budget(1.20, 0.01)["verdict"] == "fail"
    assert U.grade_with_budget(1.20, None)["verdict"] == "unresolved"
    # boundary: exactly at the bar is not "certified" (strict inequality)
    assert U.grade_with_budget(0.5, 0.5)["verdict"] == "upper_bound_only"


def test_reference_resolution_demand_is_self_consistent():
    need = U.nx_ref_for_bar(0.8931, 0.7181, 12800, 0.8133)
    assert need["satisfiable"] is True
    nx = need["nx_ref_needed"]
    allowance = 1.0 - 0.8931
    # extrapolating nx**-order from the current rung must land exactly on the allowance
    assert 0.7181 * (nx / 12800) ** (-0.8133) == pytest.approx(allowance, rel=1e-12)
    assert nx > 12800                                    # a finer reference is needed
    # a graded error already above the bar cannot be rescued by any reference
    assert U.nx_ref_for_bar(1.2, 0.1, 12800, 1.0)["satisfiable"] is False
    # an already sufficient reference needs no refinement
    assert U.nx_ref_for_bar(0.2, 0.05, 12800, 1.0)["nx_ref_needed"] == 12800


# --------------------------------------------------------------------------- #
# real ladder data
# --------------------------------------------------------------------------- #
def _need(nx: int, family: str) -> Path:
    if wcr.try_load_profile(DATA, nx, family) is None:
        pytest.skip(f"profile nx={nx} {family} not computed")
    return DATA


def test_real_selfcert_ladder_is_ordered():
    _need(1600, "weno5")
    ladder = U.selfcert_ladder(DATA)
    assert len(ladder) >= 3
    for row in ladder:
        assert row["l1_percent"] > 0
    # the reference family's self-certification improvement must not be claimed without an
    # order: every paired row carries one
    for row in ladder[:-1]:
        assert row["order"] > 0
        assert row["u_ref_percent"] > row["l1_percent"] / 2.0   # U > e/2 for p < 1.6


def test_unpaired_reference_is_reported_as_unmeasurable():
    _need(12800, "weno5")
    present = [n for n in wcr.WENO5_RUNGS if wcr.try_load_profile(DATA, n, "weno5") is not None]
    got = U.u_ref_of_reference(DATA, 4 * max(present))
    assert got["measurable"] is False and got["u_ref_percent"] is None
    assert got["note"]  # an unpaired reference must say why it has no uncertainty
    # the finest *paired* rung has no successor to measure its order against: the order is
    # extrapolated and must be labelled as such, never presented as measured
    rows = U.selfcert_ladder(DATA)
    finest_paired = rows[-1]
    assert finest_paired["order_source"] in {"measured_pair", "measured_triple", "extrapolated"}
    if finest_paired["order_source"] == "extrapolated":
        assert finest_paired["order"] == pytest.approx(rows[-2]["order"])


def test_real_regrade_of_a_graded_rung_carries_its_budget():
    _need(1600, "hllc")
    rows = U.grade_across_references(DATA, 1600, ref_rungs=(6400, 12800))
    measured = [r for r in rows if r["measured"]]
    assert len(measured) >= 1
    for row in measured:
        assert row["verdict"] in {"certified", "upper_bound_only", "fail", "unresolved"}
        assert row["worst_case_percent"] == pytest.approx(row["measured_percent"]
                                                         + row["u_ref_percent"])
    if len(measured) == 2:
        # The measured error may move in *either* direction when the reference is refined
        # (a finer reference is closer to the truth, not closer to the graded solution), so
        # the invariant is the budget: the move must fit inside the uncertainty claimed for
        # the coarser reference.  Outside it, the coarser grade was reference-limited.
        assert measured[0]["within_claimed_u_ref"] is True
        assert abs(measured[0]["delta_l1_to_next_percent"]) <= measured[0]["reference_u_ref_percent"]


# --------------------------------------------------------------------------- #
# round-4 contract: predeclaration precedes evidence, evidence honours it
# --------------------------------------------------------------------------- #
ROUND4 = Path(__file__).resolve().parents[1] / "docs" / "XVAL-FVM-M18737-ROUND2"


def test_predeclaration_is_frozen_from_rungs_below_the_target():
    path = ROUND4 / "predeclaration_ref_uncertainty.json"
    if not path.exists():
        pytest.skip("round-4 predeclaration not present")
    frozen = json.loads(path.read_text())
    assert set(frozen) >= {"predicted_l1_12800_vs_25600_percent", "u_ref_percent_at_12800",
                           "nx_ref_needed_for_bar", "order_finest_pair"}
    band = frozen["predicted_band_percent"]
    assert band[0] < frozen["predicted_l1_12800_vs_25600_percent"] < band[1]


def test_extended_reference_meets_the_frozen_prediction():
    """P1/P2: the nx=25600 reference must land in the frozen band with a sane order."""
    if wcr.try_load_profile(DATA, 25600, "weno5") is None:
        pytest.skip("extended WENO5 reference (nx=25600) not computed")
    frozen = json.loads((ROUND4 / "predeclaration_ref_uncertainty.json").read_text())
    ladder = U.selfcert_ladder(DATA)
    pair = [r for r in ladder if r["fine_nx"] == 25600]
    assert pair, "the 12800->25600 pair must be measurable once the rung exists"
    row = pair[0]
    lo, hi = frozen["predicted_band_percent"]
    assert lo <= row["l1_percent"] <= hi, (row["l1_percent"], lo, hi)
    band = frozen["order_band"]
    assert band[0] <= row["order"] <= band[1], row["order"]
    # a finer reference must be no worse than the one it replaces
    prev = [r for r in ladder if r["fine_nx"] == 12800][0]
    assert row["l1_percent"] < prev["l1_percent"]