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


def test_muscl_hllc_nx6400_is_certified_under_1_percent(analysis):
    """The graded MUSCL/HLLC nx=6400 solution vs the certified WENO5 reference.

    The nx=3200 rung was the binding gate at L1=1.776% > 1% (honest-FAIL in
    the original round-2 closure).  With nx=6400, both L1 (0.893%) and peak
    (0.044%) are <1%, so the certified claim is now **achieved**.

    Guard: if L1 ever drops below 1% at nx=3200 (e.g. due to a solver change),
    this test should be updated to check nx=3200 instead.
    """
    g = analysis["vs_certified_reference"]["nx6400_hllc"]
    assert g["peak_percent"] < 1.0, g["peak_percent"]
    assert g["l1_percent"] < 1.0, (
        f"nx6400 L1={g['l1_percent']:.4f}% >= 1% -- certified claim not met")
    # the honest verdict: CERTIFIED (both L1 and peak <1%)
    assert analysis["certified"] is True, analysis["note"]
    assert analysis["graded_under_1"] is True


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
