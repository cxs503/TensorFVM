"""Regression gate for the round-2 certified-<1% reference closure.

The claims in ``docs/CASE-WOODWARD-COLELLA-round2-xval.md`` (anchor
``XVAL-FVM-M18737-ROUND2``) are invariants of the stored profiles plus the
reference tables, so they are cheap to re-check (pure numpy, no solver call).
This gate guards the *new* round-2 closure: the independent WENO5 reference is
pushed to finer grids (6400, 12800) and the TensorFVM MUSCL/HLLC solution is
re-graded against it to a **certified <1 %** statement.

Guarded claims:

1. The finer WENO5 rungs (6400, 12800) exist in ``ladder/data/``.
2. The WENO5 reference is self-converged: WENO5@6400 vs WENO5@12800 agree to
   <1 % on L1 and peak (otherwise no "<1 %" claim may be made).
3. The TensorFVM MUSCL/HLLC nx=6400 solution agrees with the certified WENO5
   reference to <1 % on L1 and peak -- i.e. the round-2 "~3 % vs the published
   (digitised) reference" residual is *not* a solver defect but the reference's
   own discretisation error, which the finer independent reference removes.
   (nx=3200 was the binding gate at L1=1.776% > 1%; nx=6400 achieves L1=0.893%.)
4. The window-mass integral stays <0.1 % for the certified rung (nx=6400).
5. **The two-level agreement above is a point estimate, not a certificate.** The
   reference has its own Richardson error bar (three-level WENO5 ladder:
   ``eps_ref`` = 0.708 %), so the honest bound is ``L1 + eps_ref`` = 1.601 %,
   which is *not* <1 %. The gate therefore pins ``certified == False`` while
   ``certified_point_estimate == True`` -- the earlier round-2 claim of an
   achieved "certified <1 %" is downgraded here (see
   ``R11-ERROR-BAR-ROBUSTNESS.md``, and ``docs/XVAL-FVM-M18737-ROUND2/
   CERTIFIED-NX6400-ERRATA.md``). A future solver change that really reaches
   <1 % must flip *both* flags; dropping ``eps_ref`` would silently re-inflate
   the claim, so the bar is asserted independently of the module.
6. The reference-side bar computed here must agree with the independent
   implementation in ``tensorfvm.verification.wc_ref_uncertainty`` (different
   code path, same stored profiles) to <0.1 %.
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

import wc_certified_ref as wcr  # noqa: E402

FINE = wcr.FINE_WENO5


@pytest.fixture(scope="module")
def analysis() -> dict:
    return wcr.run()


def test_finer_weno5_rungs_exist():
    for nx in FINE:
        p = wcr.DATA / f"ladder_profile_nx{nx}_weno5.npz"
        assert p.exists(), f"finer WENO5 rung nx={nx} missing (run wc_ladder_weno5.py --sizes {','.join(map(str,FINE))})"


def test_reference_is_self_certified(analysis):
    """WENO5@6400 vs WENO5@12800 must agree to <1 % before any <1 % claim."""
    r = analysis["reference"]
    assert r["self_l1_percent"] < 1.0, r["self_l1_percent"]
    assert r["self_peak_percent"] < 1.0, r["self_peak_percent"]


def test_muscl_hllc_nx6400_agrees_under_1_percent_as_point_estimate(analysis):
    """The graded MUSCL/HLLC nx=6400 solution vs the certified WENO5 reference.

    The nx=3200 rung was the binding gate at L1=1.776% > 1% (honest-FAIL in
    the original round-2 closure).  With nx=6400 both L1 (0.893%) and peak
    (0.044%) are <1% *against the raw WENO5@12800 reference*.

    This is the **point estimate**; it is not a certificate (see the next test).

    Guard: if L1 ever drops below 1% at nx=3200 (e.g. due to a solver change),
    this test should be updated to check nx=3200 instead.
    """
    g = analysis["vs_certified_reference"]["nx6400_hllc"]
    assert g["peak_percent"] < 1.0, g["peak_percent"]
    assert g["l1_percent"] < 1.0, (
        f"nx6400 L1={g['l1_percent']:.4f}% >= 1% -- point estimate not met")
    assert analysis["graded_under_1"] is True
    assert analysis["certified_point_estimate"] is True


def test_certified_requires_the_reference_error_bar(analysis):
    """<1 % must survive adding the reference's own error bar -- it does not.

    ``L1(MUSCL@6400, WENO5@12800) = 0.893 %`` compares two numerical solutions
    that each carry a discretisation error of the same order, so the difference
    is not the graded solution's error.  Adding the reference bar
    (``eps_ref`` = 0.708 %) gives the honest bound 1.601 % > 1 %: the claim has
    to be downgraded to "point estimate only".
    """
    bar = analysis["reference_error_bar"]
    assert 0.5 < bar["eps_ref_percent"] < 1.0, bar["eps_ref_percent"]
    assert bar["order_fallback"] is False
    g = analysis["vs_certified_reference"]["nx6400_hllc"]
    bound = g["l1_percent"] + bar["eps_ref_percent"]
    assert bound == pytest.approx(analysis["honest_upper_bound_percent"], rel=1e-12)
    assert bound > 1.0, f"honest bound {bound:.4f}% -- if this drops below 1% " \
                        "the certified flag may be reclaimed (update this test)"
    assert analysis["certified"] is False
    assert analysis["certified_provable"] is False


def test_reference_bar_matches_the_independent_implementation(analysis):
    """Same stored profiles, two code paths -> the reference bar must agree.

    ``wc_certified_ref.reference_error_bar`` (linear interpolation on the fine
    grid) and ``tensorfvm.verification.wc_ref_uncertainty.u_ref_of_reference``
    (conservative restriction: coarse cell mean = mean of the fine cells)
    measure the same ladder distance two ways.  They are **not** bit-identical:
    the restriction variant lands 1.7 % lower (0.5437 % vs 0.5527 %), shifting
    ``eps_ref`` by 1.5 % (0.7181 % vs 0.7076 %).  That spread is a systematic of
    the metric implementation, so it is pinned as a band; the honest bound is
    quoted with its upper end in the errata document, so the verdict does not
    depend on which implementation is used.
    """
    from tensorfvm.verification import wc_ref_uncertainty as U

    data = REPO / "docs" / "woodward-colella-double-blast" / "ladder" / "data"
    mine = analysis["reference_error_bar"]["eps_ref_percent"]
    theirs = U.u_ref_of_reference(str(data), FINE[-1])
    assert theirs["measurable"] is True, theirs
    assert mine == pytest.approx(theirs["u_ref_percent"], rel=0.05), (mine, theirs)
    assert analysis["reference_error_bar"]["l1_percent_6400_vs_12800"] == pytest.approx(
        theirs["l1_percent"], rel=0.05)
    l1 = analysis["vs_certified_reference"]["nx6400_hllc"]["l1_percent"]
    for eps in (mine, theirs["u_ref_percent"]):
        assert l1 + eps > 1.0, (l1, eps)


def test_nx3200_was_the_binding_gate(analysis):
    """Historical guard: nx=3200 L1 was >1% (the original honest-FAIL).

    This documents why the certified claim required nx=6400: at nx=3200 the
    L1 was 1.776% > 1%, making it the binding gate.  If this ever changes
    (e.g. solver improvement), the certified claim would be achievable at
    coarser resolution.
    """
    g = analysis["vs_certified_reference"]["nx3200_hllc"]
    assert g["l1_percent"] > 1.0, (
        f"nx3200 L1={g['l1_percent']:.4f}% unexpectedly <1% -- "
        "the historical binding gate has changed; update tests accordingly")
    assert g["peak_percent"] < 1.0  # peak was always <1%


def test_finer_reference_removes_the_published_floor(analysis):
    """The certified reference must beat the published (digitised) reference."""
    assert analysis["floor_removed"] is True
    assert (analysis["vs_certified_reference"]["nx3200_hllc"]["l1_percent"]
            < analysis["vs_published_reference"]["l1_percent"])


def test_window_mass_is_conserved(analysis):
    """∫ρ dx over [0.4,1] stays <0.1 % for the certified rung (nx>=1600).

    nx=800 has mass error 0.136% (slightly above 0.1%) which is expected at
    that resolution; the certified rung nx=6400 has 0.0002%.
    """
    for tag in ("nx1600_hllc", "nx3200_hllc", "nx6400_hllc"):
        g = analysis["vs_certified_reference"][tag]
        assert g["mass_percent"] < 0.1, f"{tag}: {g['mass_percent']}"


def test_convergence_monotone_under_1(analysis):
    """L1 error decreases monotonically toward the <1% bar."""
    vs = analysis["vs_certified_reference"]
    l1s = [vs[t]["l1_percent"] for t in ("nx800_hllc", "nx1600_hllc",
                                          "nx3200_hllc", "nx6400_hllc")]
    for i in range(len(l1s) - 1):
        assert l1s[i] > l1s[i + 1], (
            f"L1 not monotone: {l1s[i]:.4f}% -> {l1s[i+1]:.4f}%")
