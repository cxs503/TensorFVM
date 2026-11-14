"""Tests for the dry-bed front metric repair (fan-shape estimator).

The defect being locked down: ``front_wetted`` (outermost cell with ``h > 0``)
is a depth-floor detector, so it does not converge under refinement and it ranks
a first-order scheme *better* than a second-order one.  The replacement is
:func:`tensorfvm.shallow_water.dam_break.fan_front_estimate`, which uses the
exact rarefaction identity ``x_f = x_i + 3 t sqrt(g h_i)``.

See ``docs/verification-dam-break-front-metric/REPORT.md``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    FAN_WINDOW,
    RITTER_DRY,
    STOKER_WET,
    fan_front_estimate,
    run_case,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D


def _profile_run(n: int, **kwargs) -> dict:
    return run_case(RITTER_DRY, n=n, **kwargs)


def test_estimator_is_exact_identity_on_the_closed_form_profile():
    """Applying the estimator to the exact profile must return x_f to round-off."""
    t = 0.7
    c0 = math.sqrt(RITTER_DRY.gravity * RITTER_DRY.h0)
    x_f = 2.0 * c0 * t
    x = np.linspace(-1.0, 3.9, 4001)
    c = (2.0 * c0 - x / t) / 3.0
    h = np.clip(c, 0.0, None) ** 2 / RITTER_DRY.gravity
    est = fan_front_estimate(x, h, t, RITTER_DRY.gravity, RITTER_DRY.h0)
    assert est["cells"] > 100
    assert abs(est["front"] - x_f) / x_f < 1e-12
    assert est["slope_error"] < 1e-12


def test_estimator_guards_degenerate_input():
    x = np.linspace(0.0, 1.0, 11)
    h = np.zeros_like(x)
    zero_time = fan_front_estimate(x, h, 0.0, 9.81, 1.0)
    assert math.isnan(zero_time["front"]) and zero_time["cells"] == 0
    dry = fan_front_estimate(x, h, 1.0, 9.81, 1.0)
    assert math.isnan(dry["front"]) and dry["cells"] == 0


def test_estimator_sees_a_gravity_error():
    """Negative control: a 5 % gravity error must show up as ~2.47 % front error."""
    n, factor = 200, 1.05
    g2 = factor * RITTER_DRY.gravity
    t_end = RITTER_DRY.t_star_end * math.sqrt(RITTER_DRY.h0 / RITTER_DRY.gravity)
    solver = ShallowWater1D(n, RITTER_DRY.x_min, RITTER_DRY.x_max, gravity=g2,
                            cfl=0.4, order=2)
    solver.set_state(np.where(solver.x < RITTER_DRY.x_dam, RITTER_DRY.h0,
                              RITTER_DRY.h1))
    solver.advance(t_end)
    est = fan_front_estimate(solver.x, solver.h, solver.time, g2, RITTER_DRY.h0)
    nominal = 2.0 * math.sqrt(RITTER_DRY.gravity * RITTER_DRY.h0) * solver.time
    measured = (est["front"] - nominal) / nominal
    predicted = math.sqrt(factor) - 1.0
    # the coarse grid adds its own 0.33 % fan error, i.e. 13 % of the control
    # signal; the n=800 control in the verification driver reads 1.5 %.
    assert abs(measured - predicted) / predicted < 0.2


def test_fan_metric_beats_the_wetted_edge_and_converges():
    runs = {n: _profile_run(n) for n in (200, 400)}
    fan = [runs[n]["metrics"]["front_fan_error"] for n in (200, 400)]
    wetted = [runs[n]["metrics"]["front_wetted_error"] for n in (200, 400)]
    assert fan[1] < fan[0]                      # converges
    assert fan[1] < wetted[1]                   # and is far more accurate
    assert runs[400]["metrics"]["front_fan_cells"] > 50


def test_wetted_edge_defect_signature_is_locked():
    """The legacy metric must stay visibly non-convergent (the defect we report)."""
    errs = [run_case(RITTER_DRY, n=n)["metrics"]["front_wetted_error"]
            for n in (200, 400, 800)]
    assert errs[1] < errs[0]
    assert errs[2] > errs[1], f"expected non-monotone wetted errors, got {errs}"


def test_fan_metric_ranks_scheme_order_correctly():
    calls = {order: run_case(RITTER_DRY, n=800, order=order) for order in (1, 2)}
    fan1 = calls[1]["metrics"]["front_fan_error"]
    fan2 = calls[2]["metrics"]["front_fan_error"]
    wet1 = calls[1]["metrics"]["front_wetted_error"]
    wet2 = calls[2]["metrics"]["front_wetted_error"]
    assert fan1 > 3.0 * fan2, (fan1, fan2)
    # the legacy metric gets this backwards -- kept as evidence, not as a claim
    assert wet1 < wet2, (wet1, wet2)


def test_verdict_and_legacy_numbers_still_reported():
    run = run_case(RITTER_DRY, n=1600)
    metrics = run["metrics"]
    verdict = run["verdict"]
    assert "front_fan<5%" in verdict and "front_wetted<5%" in verdict
    assert verdict["front_fan<5%"]["pass"] and verdict["all_pass"]
    # published number from docs/shallow-water-dam-break/SUMMARY.md
    assert abs(metrics["front_wetted_error"] - 0.02891) < 2e-3
    assert metrics["front_fan_error"] < 1e-3
    assert metrics["front_fan_window"] == [FAN_WINDOW[0], FAN_WINDOW[1]]


def test_wet_branch_verdict_keys_unchanged():
    # n=1600 is the accepted grid: at n=200 the bore carries one spurious
    # extremum above 1e-4 h0 (recorded in the benchmark's own grid table), so
    # ``all_pass`` is grid-dependent there and is not asserted at n=200.
    run = run_case(STOKER_WET, n=1600)
    # ``front_series<5%`` is the anchor acceptance itself (the whole x_f(t)
    # series, not only the t* = 2 snapshot); it was promoted from ``metrics``
    # into the verdict when it was found that the dry branch reported
    # ``all_pass`` at n = 400 while its graded series error was 6.61 %.
    keys = set(run["verdict"]) - {"all_pass"}
    assert keys == {"shock_position<5%", "step_ratio<5%", "plateau_h2<5%",
                    "depth_l1<5%", "monotone(no_extrema)", "tv_no_growth<0.5%",
                    "front_series<5%"}
    assert run["verdict"]["all_pass"]


def test_front_definition_study_reports_the_fan_definition():
    from tensorfvm.shallow_water.dam_break import front_definition_study
    study = front_definition_study(resolutions=(200, 400))
    assert study["max_fan_shape_error_at_accepted_grid"] is not None
    row = study["rows"][-1]
    assert row["fan_shape_error"] < row["wetted_edge_error"]
    assert row["fan_shape_spread"] > 0.0


def test_convergence_study_headline_is_the_fan_metric():
    from tensorfvm.shallow_water.dam_break import convergence_study
    study = convergence_study(RITTER_DRY, resolutions=(200, 400, 800))
    assert "front_fan" in study["rows"][-1]["headline"]
    assert "front_wetted_diagnostic" in study["rows"][-1]["headline"]
    assert math.isfinite(study["observed_order_headline"])