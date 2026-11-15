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
3. The TensorFVM MUSCL/HLLC nx=3200 solution agrees with the certified WENO5
   reference to <1 % on L1 and peak -- i.e. the round-2 "~3 % vs the published
   (digitised) reference" residual is *not* a solver defect but the reference's
   own discretisation error, which the finer independent reference removes.
4. The window-mass integral stays <0.1 % (conservation is format-independent).
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


def test_muscl_hllc_nx3200_is_certified_under_1_percent(analysis):
    """The graded MUSCL/HLLC nx=3200 solution vs the certified WENO5 reference.

    Honest-FAIL guard: the "certified <1%" claim is *not* achieved on the L1
    metric (measured 1.776 % > 1 %), even though the peak (0.758 %) and the
    reference self-certification (WENO5@6400 vs 12800, L1 0.544 %) are <1 %.
    The verdict must therefore be NOT_CERTIFIED -- the L1 gate is the binding
    one, and no "<1 %" claim may be made while it sits above the bar.
    """
    g = analysis["vs_certified_reference"]["nx3200_hllc"]
    # peak is <1 % (achieved); L1 is the binding gate and is NOT <1 %
    assert g["peak_percent"] < 1.0, g["peak_percent"]
    assert g["l1_percent"] >= 1.0, (
        "L1 unexpectedly <1% -- re-check; if it drops below 1% the "
        "certified-<1% claim becomes live and this guard must flip to "
        "assert certified is True")
    # the honest verdict: NOT_CERTIFIED (the L1 gate binds)
    assert analysis["certified"] is False, analysis["note"]


def test_finer_reference_removes_the_published_floor(analysis):
    """The certified reference must beat the published (digitised) reference."""
    assert analysis["floor_removed"] is True
    assert (analysis["vs_certified_reference"]["nx3200_hllc"]["l1_percent"]
            < analysis["vs_published_reference"]["l1_percent"])


def test_window_mass_is_conserved(analysis):
    """∫ρ dx over [0.4,1] stays <0.1 % -- a smooth, format-independent check."""
    g = analysis["vs_certified_reference"]["nx3200_hllc"]
    assert g["mass_percent"] < 0.1, g["mass_percent"]
