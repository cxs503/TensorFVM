"""Cross-code gate for CASE-WOODWARD-COLELLA (Woodward & Colella 1984).

The committed verification of the two blast waves (``double_blast``) anchors on
quantities the HLLC/MUSCL solver computes about itself: conservation, exact
Riemann fronts before the waves reach a wall, and convergence towards its own
fine-grid run.  None of those can expose a *systematic* scheme error that
survives refinement.  This gate closes that hole with an independent
discretisation -- finite-difference WENO5 + SSP-RK3 + Rusanov flux, no Riemann
solver, different limiter, different time integrator -- implemented in
``docs/woodward-colella-double-blast/xcode/wc_crosscode.py`` and validated
separately against the *exact* Sod solution.

The grids are deliberately small (CI budget); the production ladder is run by
``python docs/woodward-colella-double-blast/xcode/wc_crosscode.py``.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]
                       / "docs" / "woodward-colella-double-blast" / "xcode"))

import wc_crosscode as xc  # noqa: E402


@pytest.fixture(scope="module")
def codes():
    fvm = xc.tensorfvm_solve(200, xc.WC_T_END)
    wen = xc.weno5_solve(200, xc.WC_T_END, datum="blast")
    return fvm, wen


def test_independent_code_reproduces_exact_sod():
    """The cross-check code is only useful if it is correct: Sod, 5 % gate."""
    leg = xc.sod_validation((200,))
    # plateau accuracy is the correctness gate; the refinement verdict needs two
    # grids and is exercised by the production script (200 -> 400 here)
    assert leg["verdicts"]["sod_plateaus_within_tolerance"], leg["rows"]
    assert leg["rows"][0]["worst_rel"] < xc.THRESHOLDS["sod_plateau_rel"]


def test_two_schemes_agree_on_the_blast_at_the_ledger_tolerance(codes):
    """Ledger tolerance is 10 %; peak density, L1 profile and peak position."""
    fvm, wen = codes
    row = xc.compare_profiles(fvm, wen)
    assert row["peak_density_rel_diff"] < xc.THRESHOLDS["peak_density_rel"], row
    assert row["l1_rho_rel"] < xc.THRESHOLDS["l1_profile_rel"], row
    assert row["peak_position_abs_diff"] <= xc.THRESHOLDS["peak_position_abs"], row


def test_closed_tube_conservation_in_both_codes(codes):
    """Both codes are conservative to round-off on the closed tube."""
    fvm, wen = codes
    assert abs(fvm["drift"]["mass"]) < 1e-12
    assert abs(fvm["drift"]["energy"]) < 1e-12
    assert wen["mass_drift"] < xc.THRESHOLDS["mass_conserved"]
    assert fvm["positivity_fixes"] == 0


def test_both_codes_see_the_same_wave_structure(codes):
    """Neither code may invent structure: same pressure peak location, same
    number of cells above the ambient density by a wide margin."""
    fvm, wen = codes
    import numpy as np

    xf = np.asarray(fvm["x"])
    pf = np.asarray(fvm["p"])
    pw = np.asarray(wen["p"])
    assert abs(xf[int(pf.argmax())] - xf[int(pw.argmax())]) <= 0.02
    n_hot_f = int((pf > 10.0).sum())
    n_hot_w = int((pw > 10.0).sum())
    assert abs(n_hot_f - n_hot_w) <= 0.10 * max(n_hot_f, n_hot_w)
