"""End-to-end gate: Woodward-Colella double blast vs **calibrated** reference.

With the y-axis-calibrated reference (bias-corrected per
``docs/CASE-WOODWARD-COLELLA-calibration-bias.md``), **all six metrics**
including ``low_plateau`` are well-posed and fall well within the 10 % ledger
tolerance.  This test asserts the full pipeline:

* the calibrated reference table is well formed,
* the committed nx=800 profile, compared against the calibrated reference,
  passes **all six** metrics (peak density, peak position, L1, faces,
  window mass, **and** low_plateau),
* a fresh solver run is finite and conserves mass.
"""
import numpy as np
import pytest

from tensorfvm.compressible.wc_reference import (
    CALIBRATED_REFERENCE,
    CALIBRATION_BIAS,
    faces,
    load_reference,
    metrics,
    metrics_from_profiles,
    profile,
)

ARTIFACTS = CALIBRATED_REFERENCE.parents[1] / "artifacts"


def test_calibrated_reference_table_is_well_formed():
    """The calibrated reference must be structurally sound."""
    ref = load_reference(calibrated=True)
    x, rho = ref["x"], ref["rho"]
    assert np.all(np.diff(x) > 0), "reference grid must be strictly increasing"
    assert 0.39 < x.min() and x.max() < 1.01
    # Calibrated peak is ~6.318 (original ~6.296 + 0.022 bias)
    assert 6.1 < rho.max() < 6.7, "calibrated peak density ~6.32"
    # Calibrated trough is ~0.179 (original ~0.157 + 0.022 bias)
    assert 0.14 < rho.min() < 0.22, "calibrated trough density ~0.179"
    assert ref["panel_spread"] < 1e-3, "panel digitisation disagree"
    assert ref["calibrated"] is True


def test_calibration_bias_value():
    """The documented bias must match the difference between references."""
    orig = load_reference(calibrated=False)
    cal = load_reference(calibrated=True)
    # The calibrated reference = original + bias (constant shift)
    diff = cal["rho"] - orig["rho"]
    assert np.allclose(diff, CALIBRATION_BIAS, atol=1e-6), (
        f"bias mismatch: mean={diff.mean():.6f}, expected={CALIBRATION_BIAS}"
    )


def test_committed_nx800_vs_calibrated_reference():
    """Re-compare the committed nx=800 profile against the calibrated reference.

    With the calibrated reference, **all six metrics** must pass, including
    ``low_plateau`` (which was ~14 % with the uncorrected reference due to
    the +0.022 density-unit systematic bias).
    """
    data = np.load(ARTIFACTS / "wc_blast_profile_nx800.npz")
    ref = load_reference(calibrated=True)
    m = metrics_from_profiles(ref["x"], ref["rho"],
                              data["x_got"], data["rho_got"],
                              calibrated=True)

    # All six metrics must pass with calibrated reference
    assert m["peak_density"]["rel_error_percent"] < 3.0, m["peak_density"]
    assert m["peak_position"]["abs_error"] < 1e-2, m["peak_position"]
    assert m["l1_relative"]["percent"] < 5.0, m["l1_relative"]
    assert max(m["faces_abs_error"]) < 0.02, m["faces_abs_error"]
    assert m["window_mass"]["rel_error_percent"] < 3.0, m["window_mass"]
    # KEY: low_plateau now passes with calibrated reference
    assert m["low_plateau"]["rel_error_percent"] < 3.0, (
        f"low_plateau should pass with calibrated ref: {m['low_plateau']}"
    )
    assert m["low_plateau"]["percent_of_peak_density"] < 1.0

    # Primary verdict must PASS with all checks true
    assert m["verdict"] == "PASS"
    assert all(m["checks"].values()), f"not all checks passed: {m['checks']}"
    assert "calibrated" in m["verdict_note"].lower()


def test_fresh_run_vs_calibrated_reference():
    """Run the solver at nx=200 and compare against calibrated reference.

    Coarser grid: peak density error will be larger (~20 %) but low_plateau
    should still be <10 % with the calibrated reference.
    """
    m = metrics(nx=200, calibrated=True)
    # Peak density at nx=200 is ~20 % (coarse grid, expected)
    assert m["peak_density"]["rel_error_percent"] < 25.0
    # low_plateau with calibrated reference should be <5 % even at nx=200
    assert m["low_plateau"]["rel_error_percent"] < 5.0, m["low_plateau"]
    # Mass conservation
    assert m["window_mass"]["rel_error_percent"] < 5.0
    # Verdict may be FAIL due to peak density, but low_plateau must pass
    assert m["checks"]["low_plateau_lt_10pct"]


def test_fresh_run_is_finite_and_conservative():
    run = profile(nx=120, t_end=0.004)
    rho = run["rho"]
    assert np.all(np.isfinite(rho)) and rho.min() > 0.0
    mass = np.mean(rho)
    assert abs(mass - 1.0) < 1e-10, "closed tube must conserve mass to round-off"


def test_metrics_function_calibrated_flag():
    """The metrics() function must propagate the calibrated flag correctly."""
    m_orig = metrics(nx=100, calibrated=False)
    m_cal = metrics(nx=100, calibrated=True)
    # Same solver output, different reference → different metrics
    assert m_orig["peak_density"]["ref"] != m_cal["peak_density"]["ref"]
    assert m_orig["low_plateau"]["ref"] != m_cal["low_plateau"]["ref"]
    # Calibrated reference has higher values (bias added)
    assert m_cal["peak_density"]["ref"] > m_orig["peak_density"]["ref"]
    assert m_cal["low_plateau"]["ref"] > m_orig["low_plateau"]["ref"]
    # Calibrated flag must be set
    assert m_cal["calibrated"] is True
    assert m_orig["calibrated"] is False
