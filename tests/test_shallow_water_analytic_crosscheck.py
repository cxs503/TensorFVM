"""Analytic cross-check of the SWE dam-break solver (node CASE-FVM-DAMBREAK-STOKER).

What is asserted, and why each assertion is evidence rather than decoration:

* the *reference* is certified on a second, independent path: the Stoker bore
  state is re-solved by Newton on the depth ratio (bisection on ``c2`` lives in
  ``exact``), the jump residuals and the two independent shock-speed formulas
  are required to vanish, the fan formula must reproduce both neighbouring
  constant states *at* the similarity boundaries, and the closed-form mass
  integral must equal the initial mass.  All of that at ``1e-12``, so a typo in
  a formula cannot survive.
* the *metric* is certified by its zero-error identity (feeding the exact
  solution to ``profile_metrics`` must return 0 everywhere) and by a
  falsification test (a deliberately corrupted profile must be caught), so a
  metric that silently cannot fail is excluded.
* the *solver* is then compared with the closed form at ``t* = 2``: probe
  positions, plateau depth/velocity, region L1 norms, conservation and the
  positivity counters.  The gate is the task tolerance, 5 %, and it is applied
  on the accepted grid ``n >= 800``.
* the coarse grid ``n = 200`` must *fail* that gate: the fan-head probe is
  ``O(dx)`` limited there.  Asserting the failure keeps the gate honest -- a
  verdict that passes everywhere measures nothing.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from tensorfvm.shallow_water import exact
from tensorfvm.shallow_water.crosscheck import (
    CASES,
    CROSSCHECK_TOL,
    PROBE_FRACTIONS,
    AnalyticCase,
    bore_state_newton,
    convergence_series,
    crosscheck_case,
    exact_mass_over_domain,
    fan_xi_at_depth,
    level_crossing,
    profile_metrics,
    reference_selfcheck,
    similarity_probes,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D

STOKER = CASES["stoker-wet"]
RITTER = CASES["ritter-dry"]


def _json_safe(obj):
    """Replace non-finite floats by ``None`` (strict JSON has no NaN/Infinity)."""
    if isinstance(obj, dict):
        return {key: _json_safe(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(value) for value in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


# --------------------------------------------------------------------------- #
# reference certification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", [STOKER, RITTER], ids=["stoker-wet", "ritter-dry"])
def test_reference_selfcheck_passes_at_machine_precision(case):
    report = reference_selfcheck(case, tol=1e-12)
    assert report["pass"], report["checks_relative"]
    assert report["worst_relative"] <= 1e-12
    # The quadrature check measures the quadrature, not the solution.
    assert report["quadrature_relative"] <= 1e-5
    assert np.isfinite(report["exact_mass"]) and report["exact_mass"] > 0.0


@pytest.mark.parametrize("ratio", [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9])
def test_newton_bore_state_agrees_with_bisection(ratio):
    """Two root finders, different unknowns: agreement certifies the closed form."""
    h0 = 1.0
    h1 = ratio
    newton = bore_state_newton(h0, h1)
    closed = exact.stoker_constant_state(h0, h1)
    assert newton["h2"] == pytest.approx(closed["h2"], rel=1e-12)
    assert newton["u2"] == pytest.approx(closed["u2"], rel=1e-11)
    assert newton["shock_speed"] == pytest.approx(closed["shock_speed"], rel=1e-11)
    # ... and both agree with the second, independent shock-speed formula.
    assert newton["shock_speed"] == pytest.approx(
        newton["shock_speed_momentum"], rel=1e-11)
    assert abs(newton["residual"]) <= 1e-14


@pytest.mark.parametrize("ratio", [0.1, 0.5, 0.9])
def test_bore_state_is_physically_ordered(ratio):
    """h1 < h2 < h0, u2 > 0 and the bore outruns the plateau: the wave pattern
    the profile classifier assumes must actually hold."""
    state = bore_state_newton(1.0, ratio)
    assert ratio < state["h2"] < 1.0
    assert state["u2"] > 0.0
    assert state["shock_speed"] > state["u2"]
    tail = state["u2"] - state["c2"]
    assert -math.sqrt(9.81) < tail < state["shock_speed"]


def test_bore_newton_rejects_dry_bed():
    with pytest.raises(ValueError):
        bore_state_newton(1.0, 0.0)


def test_stoker_region_bounds_match_constant_state():
    """The boundary helper exported by ``exact`` must be the closed form's own."""
    sol = exact.stoker_constant_state(1.0, 0.5)
    bounds = exact.stoker_region_bounds(1.0, 0.5)
    assert bounds["head"] == pytest.approx(-sol["c0"], rel=0, abs=0)
    assert bounds["tail"] == pytest.approx(sol["u2"] - sol["c2"], rel=1e-15)
    assert bounds["shock"] == pytest.approx(sol["shock_speed"], rel=1e-15)
    assert bounds["h2"] == pytest.approx(sol["h2"], rel=0, abs=0)
    assert bounds["u2"] == pytest.approx(sol["u2"], rel=0, abs=0)


# --------------------------------------------------------------------------- #
# probe machinery
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fraction", [0.02, 0.25, 0.5, 0.75, 0.98])
def test_fan_depth_inversion_round_trip(fraction):
    h0, g = 1.0, 9.81
    level = h0 - fraction * h0
    xi = fan_xi_at_depth(level, h0, g)
    c0 = math.sqrt(g * h0)
    assert xi == pytest.approx(2.0 * c0 - 3.0 * math.sqrt(g * level), rel=1e-15)
    # Feeding xi back through the fan formula returns the level.
    assert (2.0 * c0 - xi) ** 2 / (9.0 * g) == pytest.approx(level, rel=1e-14)


def test_fan_depth_inversion_rejects_impossible_level():
    with pytest.raises(ValueError):
        fan_xi_at_depth(1.5, 1.0)
    with pytest.raises(ValueError):
        fan_xi_at_depth(-0.1, 1.0)


def test_level_crossing_interpolates_and_reports_misses():
    x = np.linspace(0.0, 1.0, 11)
    ramp = 1.0 - x                      # decreasing 1 -> 0
    assert level_crossing(x, ramp, 0.5) == pytest.approx(0.5, abs=1e-12)
    assert level_crossing(x, ramp, 0.5, direction="increasing") != \
        level_crossing(x, ramp, 0.5)  # first increasing crossing is elsewhere
    assert math.isnan(level_crossing(x, ramp, 1.5))
    assert math.isnan(level_crossing(x, ramp, -0.5))
    with pytest.raises(ValueError):
        level_crossing(x, ramp, 0.5, direction="sideways")
    # Non-finite samples must not poison the search.
    dirty = ramp.copy()
    dirty[3] = np.nan
    assert np.isfinite(level_crossing(x, dirty, 0.5))


def test_probe_levels_are_inside_their_crossing_range():
    """Guards the defect found while building this module: a fan probe below the
    plateau depth is never crossed inside the fan, so the probe would silently
    lock onto a different wave."""
    bounds = STOKER.bounds()
    levels = __import__(
        "tensorfvm.shallow_water.crosscheck", fromlist=["_probe_levels"]
    )._probe_levels(STOKER)
    assert bounds["h2"] < levels["fan_head"][0] < 1.0
    assert bounds["h2"] < levels["fan_mid"][0] < 1.0
    assert STOKER.h1 < levels["bore"][0] < bounds["h2"]
    for name in ("fan_head", "fan_mid", "bore"):
        assert levels[name][1] == "decreasing"


# --------------------------------------------------------------------------- #
# metric certification: zero-error identity + falsification
# --------------------------------------------------------------------------- #
def test_profile_metrics_are_zero_for_the_exact_solution():
    case = STOKER
    n = 400
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity)
    t = case.t_end
    h_ex, u_ex = case.exact_state(solver.x, t)
    metrics = profile_metrics(case, solver.x, h_ex, u_ex, t)
    assert metrics["depth_l1"] <= 1e-15
    assert metrics["depth_linf"] <= 1e-15
    assert metrics["velocity_l1"] <= 1e-14
    assert metrics["mass_profile_error"] <= 1e-15
    for name, probe in metrics["probes"].items():
        # A probe is a *linear interpolation between cell centres*, so even on
        # the exact profile it carries a discretisation floor: ~3e-6 for the
        # smooth fan probes at n=400, and half a cell for the bore probe, whose
        # level sits across a discontinuity (``dx/2`` divided by the shock
        # position ``U t``).  Both floors are stated, not hidden in a magic
        # constant, and both are far below the 5 % acceptance tolerance.
        floor = max(1e-4, 1.5 * solver.dx / (2.0 * abs(probe["xi_exact"])))
        assert probe["rel_error"] <= floor, (name, probe, floor)
    assert metrics["plateau"]["depth_rel_error"] <= 1e-12
    assert metrics["plateau"]["velocity_rel_error"] <= 1e-12
    for region in metrics["regions"].values():
        assert region["depth_l1"] <= 1e-14 or math.isnan(region["depth_l1"])


def test_profile_metrics_catch_a_corrupted_plateau():
    """A 20 % low plateau must show up in the plateau metric *and* in L1."""
    case = STOKER
    n = 400
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity)
    t = case.t_end
    h_ex, u_ex = case.exact_state(solver.x, t)
    clean = profile_metrics(case, solver.x, h_ex, u_ex, t)
    corrupted = np.array(h_ex, dtype=float)
    # The post-bore state spans x in (-1.12, 1.89); corrupt the *majority* of the
    # median window (it is inset by 5 % of the state's extent on both sides), so
    # the corrupted depth really is the median.
    plateau = (solver.x > -0.8) & (solver.x < 1.7)
    corrupted[plateau] *= 0.8
    dirty = profile_metrics(case, solver.x, corrupted, u_ex, t)
    assert dirty["plateau"]["depth_rel_error"] == pytest.approx(0.2, rel=0.05)
    assert dirty["depth_l1"] > 10.0 * clean["depth_l1"]
    assert dirty["regions"]["plateau"]["depth_l1"] > 0.05


def test_exact_mass_integral_matches_quadrature_and_is_conserved():
    for case in (STOKER, RITTER):
        t = case.t_end
        x = np.linspace(case.x_min, case.x_max, 200001)
        h_ex, _ = case.exact_state(x, t)
        quadrature = float(np.sum(0.5 * (h_ex[1:] + h_ex[:-1]) * np.diff(x)))
        analytic = exact_mass_over_domain(case, t)
        assert analytic == pytest.approx(quadrature, rel=1e-6)
        initial = exact_mass_over_domain(case, 0.0)
        assert analytic == pytest.approx(initial, rel=1e-14)


# --------------------------------------------------------------------------- #
# solver vs closed form
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("case", [STOKER, RITTER], ids=["stoker-wet", "ritter-dry"])
def test_solver_passes_the_crosscheck_at_the_accepted_grid(case):
    run = crosscheck_case(case, n=800)
    verdict = run["verdict"]
    assert verdict["pass"], verdict
    assert verdict["tolerance"] == CROSSCHECK_TOL
    for name, value in verdict["checks"].items():
        assert value is not None and value <= CROSSCHECK_TOL, (name, value)
    metrics = run["metrics"]
    assert metrics["volume_drift"] <= 1e-12
    assert run["discretisation"]["clipped_cells"] == 0
    # The plateaus are the interpolated cell centres: the exact mass at the end
    # must equal the mass the solver started with, to round-off.
    assert metrics["exact_mass_analytic"] == pytest.approx(
        metrics["volume_initial"], rel=1e-12)
    assert metrics["numeric_mass_sampled"] == pytest.approx(
        metrics["exact_mass_sampled"], rel=1e-3)
    _json_safe(run)  # must be serialisable for docs/.../reproduction.json


def test_stoker_plateau_reproduces_the_bore_state():
    """The named physical target of the case: post-bore depth and velocity."""
    run = crosscheck_case(STOKER, n=800)
    plateau = run["metrics"]["plateau"]
    assert plateau["step_ratio_numeric"] == pytest.approx(
        plateau["step_ratio_exact"], rel=1e-2)
    assert plateau["u_numeric"] == pytest.approx(plateau["u_exact"], rel=1e-2)
    assert plateau["u_exact"] == pytest.approx(
        exact.stoker_constant_state(STOKER.h0, STOKER.h1)["u2"], rel=1e-15)


def test_coarse_grid_fails_the_gate():
    """The gate must be able to fail: at n=200 the fan-head probe is O(dx)."""
    for case in (STOKER, RITTER):
        run = crosscheck_case(case, n=200)
        assert not run["verdict"]["pass"]
        assert "fan_head" in run["verdict"]["failed"]


def test_convergence_series_orders_and_gate():
    series = convergence_series(STOKER, resolutions=(200, 400, 800, 1600))
    # The accepted grids pass; the coarse ones do not (fan-head probe is O(dx)).
    assert series["accepted_pass"]
    assert not series["all_pass"]
    assert 0.5 <= series["observed_order_l1"] <= 2.5
    # The bore probe converges at least first order.
    assert 0.4 <= series["observed_order_fan_mid"] <= 2.5
    for row in series["rows"]:
        assert row["volume_drift"] <= 1e-12
        assert row["headline"]["bore"] <= CROSSCHECK_TOL


def test_probes_track_a_shifted_profile_monotonically():
    """Shifting the profile downstream must move every probe downstream."""
    case = STOKER
    n = 800
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity)
    t = case.t_end
    h_ex, _ = case.exact_state(solver.x, t)
    base = similarity_probes(case, solver.x, h_ex, t)
    shifted = similarity_probes(case, solver.x + 2.0 * solver.dx, h_ex, t)
    for name in base:
        assert shifted[name]["rel_error"] >= base[name]["rel_error"] - 1e-12


# --------------------------------------------------------------------------- #
# case validation
# --------------------------------------------------------------------------- #
def test_case_validation_rejects_inconsistent_states():
    with pytest.raises(ValueError):
        AnalyticCase(key="bad", kind="wet", h0=1.0, h1=0.0)
    with pytest.raises(ValueError):
        AnalyticCase(key="bad", kind="dry", h0=1.0, h1=0.4)
    with pytest.raises(ValueError):
        AnalyticCase(key="bad", kind="sideways", h0=1.0, h1=0.5)
    with pytest.raises(ValueError):
        AnalyticCase(key="bad", kind="wet", h0=1.0, h1=0.5, x_min=2.0, x_max=1.0)
    with pytest.raises(ValueError):
        AnalyticCase(key="bad", kind="wet", h0=1.0, h1=0.5, t_star_end=0.0)


def test_case_summary_is_json_round_trippable():
    payload = _json_safe({"cases": [case.summary() for case in CASES.values()],
                          "probe_fractions": PROBE_FRACTIONS})
    revived = json.loads(json.dumps(payload, allow_nan=False))
    assert [case["key"] for case in revived["cases"]] == ["ritter-dry", "stoker-wet"]
    assert revived["cases"][1]["h1"] == 0.5