"""2-D confirmation of the Ritter/Stoker dam-break anchors.

The anchor ``CASE-FVM-DAMBREAK-STOKER`` is graded on the 1-D solver; these
tests grade the **2-D** solver on the identical problem (channel uniform in
``y``, ``y``-averaged profile) with the same estimators, and pin the two
transverse checks declared in :mod:`tensorfvm.shallow_water.dam_break2d`:

* unperturbed initial data must keep the transverse direction at round-off
  (structural check, graded at ``TRANSVERSE_ROUNDOFF``),
* a transverse depth modulation must not be amplified and must not move the
  graded wave position outside half the acceptance tolerance.

Runtime is kept small (nx = 200) on purpose: the graded evidence lives in the
artifact run (``scripts/run_shallow_water_dam_break_2d.py``), these tests only
guard the thresholds.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    RITTER_DRY,
    STOKER_WET,
    TOLERANCE,
    data_driven_plateau,
)
from tensorfvm.shallow_water.dam_break2d import (
    FAN_RESOLVED_CELLS,
    MODULATION_AMPLITUDE,
    ONE_D_AGREEMENT_MAX,
    TRANSVERSE_GROWTH_MAX,
    TRANSVERSE_ROUNDOFF,
    channel_profile,
    compare_with_1d,
    data_driven_series_metrics,
    metrics_2d,
    run_2d_case,
    sample_2d,
    transverse_defect,
    transverse_modulation_study,
    verdict_2d,
)
from tensorfvm.shallow_water.swe2d import ShallowWater2D


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET], ids=lambda c: c.key)
def test_two_d_solver_reproduces_the_one_d_exact_solution(case):
    """Headline anchor metric from the 2-D solver, same 5 % criterion.

    Graded at nx = 400: the monotonicity check (no interior extremum above
    ``1e-4 h0``) needs that resolution on the wet bed, exactly as in the 1-D
    benchmark, where n = 200 still carries one sub-threshold extremum.
    """
    run = run_2d_case(case, nx=400, ny=8)
    m = run["metrics"]
    assert m["headline_error"] <= TOLERANCE
    assert m["depth_l1"] <= TOLERANCE
    assert run["verdict"]["all_pass"], run["verdict"]


def test_wet_bed_level_step_is_measured_from_the_profile():
    """The still-water step (h2/h1) is a data-driven reading, not an oracle."""
    run = run_2d_case(STOKER_WET, nx=200, ny=8)
    m = run["metrics"]
    assert m["step_ratio_error"] <= TOLERANCE
    assert m["h2_error"] <= TOLERANCE
    assert m["bore_method"] == "mid_level_crossing"


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET], ids=lambda c: c.key)
def test_transverse_direction_stays_at_round_off(case):
    """Uniform-in-y data must stay uniform in y to machine precision."""
    run = run_2d_case(case, nx=200, ny=8)
    m = run["metrics"]
    assert m["transverse_defect"] <= TRANSVERSE_ROUNDOFF
    assert m["transverse_defect_mid_vs_avg"] <= TRANSVERSE_ROUNDOFF
    assert m["v_max"] <= TRANSVERSE_ROUNDOFF
    assert m["mass_drift"] <= 1e-12
    assert m["h_min"] >= 0.0


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET], ids=lambda c: c.key)
def test_transverse_modulation_is_not_amplified(case):
    """A 1 % transverse depth modulation must stay bounded and local.

    Both limits are predeclared module constants: the peak transverse defect
    may not exceed ``TRANSVERSE_GROWTH_MAX`` times the initial amplitude, and the
    graded wave-position error may not leave half the acceptance tolerance.
    """
    study = transverse_modulation_study(case, nx=200, ny=8,
                                        amplitude=MODULATION_AMPLITUDE)
    assert study["growth_factor"] <= TRANSVERSE_GROWTH_MAX
    assert study["modulated_headline_error"] <= 0.5 * TOLERANCE
    assert all(item["pass"] for item in study["criteria"].values())


def test_two_d_operator_agrees_with_the_one_d_operator():
    """Same problem, same nx: the two implementations must agree to 0.5 pp."""
    comp = compare_with_1d(STOKER_WET, nx=200, ny=8)
    assert comp["max_difference"] <= ONE_D_AGREEMENT_MAX, comp["rows"]
    assert comp["pass"]


def test_data_driven_bore_gate_excludes_the_unresolved_fan_regime():
    """The oracle-free bore detector is only graded once the fan is resolved.

    Documented failure mode: while the exact rarefaction fan is narrower than a
    few cells its jump is the steepest feature of the profile, the detector
    latches onto it and reads ~100 ... 140 % error -- identically on every grid.
    The predeclared gate is ``FAN_RESOLVED_CELLS``.
    """
    run = run_2d_case(STOKER_WET, nx=200, ny=8)
    series = run["metrics"]["front_series_data_driven"]
    assert series["samples_graded"] > 0
    assert series["pass"], series
    # the excluded samples are exactly the unresolved-fan ones and they are
    # badly wrong -> they must have been excluded, not silently graded
    assert series["unresolved_samples"] > 0
    assert series["unresolved_max_error"] > TOLERANCE
    # and the gate is what excludes them
    for sample in run["metrics"]["time_to_front_series"]:
        if "data_driven_resolved" not in sample:
            continue
        if sample["data_driven_resolved"]:
            assert sample["fan_cells"] >= FAN_RESOLVED_CELLS
        else:
            assert (sample["fan_cells"] < FAN_RESOLVED_CELLS
                    or not sample["resolved"])


def test_graded_final_state_bore_is_well_resolved():
    """At t* = 2 the fan spans many cells, so the data-driven bore is graded."""
    run = run_2d_case(STOKER_WET, nx=200, ny=8)
    last = run["metrics"]["time_to_front_series"][-1]
    assert last["data_driven_resolved"]
    assert last["fan_cells"] > 10.0


def test_data_driven_series_metrics_reports_unresolved_samples():
    """Synthetic series: graded-only error is reported, the rest stays visible."""
    good = {"data_driven_resolved": True, "data_driven_error": 0.01, "t_star": 1.0}
    bad = {"data_driven_resolved": False, "data_driven_error": 1.2, "t_star": 0.01}
    out = data_driven_series_metrics([bad, good])
    assert out["samples"] == 2
    assert out["samples_graded"] == 1
    assert out["max_error"] == pytest.approx(0.01)
    assert out["unresolved_samples"] == 1
    assert out["unresolved_max_error"] == pytest.approx(1.2)
    assert out["pass"]


def test_profile_reduction_and_defect_helpers():
    """The y-average reduction and the transverse defect are consistent."""
    solver = ShallowWater2D(8, 4, -4.0, 4.0, 0.0, 1.0)
    h = np.zeros((8, 4))
    h[:, :] = 1.0
    h[4:, :] = 0.5   # cell centres are -3.5 ... 3.5
    solver.set_state(h)
    xc, avg, mid = channel_profile(solver)
    assert xc.shape == (8,)
    assert np.allclose(avg, np.where(xc <= 0.0, 1.0, 0.5), atol=1e-15)
    assert np.allclose(mid, avg)
    assert transverse_defect(h, 1.0) == pytest.approx(0.0)
    h2 = h.copy()
    h2[:, 0] += 0.01
    # one column of four carries +0.01 -> the y-mean moves by 0.01/ny
    assert transverse_defect(h2, 1.0) == pytest.approx(0.01 * (1.0 - 1.0 / 4))


def test_verdict_records_the_transverse_checks():
    """The verdict dict carries the transverse criteria, not just the headline."""
    run = run_2d_case(RITTER_DRY, nx=200, ny=8)
    verdict = run["verdict"]
    assert "transverse_symmetry<=1e-12" in verdict
    assert "v_transverse<=1e-12" in verdict
    assert verdict["all_pass"]


def test_wet_plateau_reading_matches_the_closed_form_at_the_final_state():
    """Cross-check of the 2-D final state against the closed-form bore state."""
    run = run_2d_case(STOKER_WET, nx=200, ny=8)
    # direct re-evaluation on the stored profile (no solver state needed)
    prof_run = run["profile"]
    dd = data_driven_plateau(np.asarray(prof_run["x"]),
                             np.asarray(prof_run["h_numeric_y_average"]))
    assert abs(dd["h2"] / 0.726920 - 1.0) < 0.01
    assert dd["n_plateau"] > 10