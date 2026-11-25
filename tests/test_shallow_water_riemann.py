"""Tests for the general shallow-water Riemann solver and its verification.

Four independent lines of defence, each of which can fail on its own:

1. **the reference** -- jump/invariant residuals, weak-form conservation across
   the discontinuities, smooth-region ODE residuals, and the two degenerate
   limits (Stoker dam break, Ritter dry bed) which must be reproduced to machine
   precision from :mod:`tensorfvm.shallow_water.exact`;
2. **the metric** -- a deliberately corrupted profile must be caught, and the
   oscillation count must agree cell-by-cell with ``dam_break``'s predicate, so
   a metric that cannot fail is excluded;
3. **the solver** -- every one of the four wave patterns is graded at the
   accepted grid with the anchor tolerance (5 %), and the coarse grid must
   *fail* at least one case, which keeps the gate from being vacuous;
4. **the limits** -- a dry-middle Riemann problem raises ``ValueError`` instead
   of returning a plausible-looking number.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water import dam_break, exact
from tensorfvm.shallow_water.riemann import (
    DEFAULT_CASES,
    EXTREMUM_FRACTION,
    TOLERANCE,
    RiemannCase,
    certification,
    convergence_series,
    degenerate_limits,
    profile_metrics,
    riemann_residuals,
    run_case,
    sample_exact,
    smooth_region_ode_residuals,
    solve_riemann,
    verdict,
    weak_form_residuals,
)

PATTERNS = {case.name: case.solution().pattern for case in DEFAULT_CASES}


def test_the_four_cases_cover_the_four_wave_patterns():
    assert set(PATTERNS.values()) == {
        "1-rarefaction+2-shock", "1-rarefaction+2-rarefaction",
        "1-shock+2-shock", "1-shock+2-rarefaction"}


def test_jump_and_invariant_residuals_are_machine_precision():
    worst = 0.0
    for case in DEFAULT_CASES:
        res = riemann_residuals(case.solution())
        worst = max(worst, res["matching_relative"],
                    res["left_riemann_invariant_relative"]
                    if res["left_kind"] == "rarefaction"
                    else res["left_mass_jump_relative"],
                    res["right_riemann_invariant_relative"]
                    if res["right_kind"] == "rarefaction"
                    else res["right_momentum_jump_relative"])
        if res["left_kind"] == "shock":
            worst = max(worst, res["left_momentum_jump_relative"])
        if res["right_kind"] == "shock":
            worst = max(worst, res["right_mass_jump_relative"])
    assert worst < 1e-12, worst


def test_weak_form_conservation_holds_across_discontinuities():
    for case in DEFAULT_CASES:
        res = weak_form_residuals(case.solution(), n=200_001)
        assert res["mass_relative"] < 1e-4, (case.name, res)
        assert res["momentum_relative"] < 1e-4, (case.name, res)


def test_weak_form_residual_refines_with_the_sample_count():
    """A single small number is not evidence; the identity must converge."""
    sol = DEFAULT_CASES[2].solution()          # two shocks: worst-case jumps
    coarse = weak_form_residuals(sol, n=50_001)
    fine = weak_form_residuals(sol, n=400_001)
    assert fine["mass_relative"] < coarse["mass_relative"]
    assert fine["momentum_relative"] < coarse["momentum_relative"]


def test_smooth_region_ode_residuals_are_small():
    for case in DEFAULT_CASES:
        res = smooth_region_ode_residuals(case.solution())
        assert res["masked_samples"] > 0.5 * res["samples"]
    # the two-shock case has no smooth region but the two constant states, so
    # its residual is exactly zero; the rarefaction cases carry discretisation
    # noise from the finite differences and are only bounded.
    for name in ("rarefaction-shock", "rarefaction-rarefaction",
                 "shock-rarefaction"):
        res = smooth_region_ode_residuals(
            next(c for c in DEFAULT_CASES if c.name == name).solution())
        assert res["mass_ode_relative"] < 1e-6, (name, res)
        assert res["momentum_ode_relative"] < 1e-5, (name, res)


def test_stoker_dam_break_is_reproduced_to_machine_precision():
    limits = degenerate_limits()
    for key, entry in limits.items():
        if not key.startswith("stoker"):
            continue
        assert entry["pattern"] == "1-rarefaction+2-shock"
        assert entry["h_star_relative_error"] < 1e-12
        assert entry["u_star_absolute_error"] < 1e-12
        assert entry["shock_speed_absolute_error"] < 1e-11
        assert entry["left_fan_head_absolute_error"] == 0.0


def test_ritter_dry_limit_converges_to_the_closed_form():
    """``h1 -> 0`` must approach Ritter; the shock branch is *singular* there."""
    limits = degenerate_limits()["ritter_limit"]
    assert limits["dam_depth_relative_error"] < 1e-12
    assert limits["ritter_state_dam_depth"] == pytest.approx(4.0 / 9.0 * 1.0)
    error = [abs(solve_riemann(1.0, 0.0, h1, 0.0).right_shock_speed
                 - 2.0 * math.sqrt(9.81)) / (2.0 * math.sqrt(9.81))
             for h1 in (1e-2, 1e-4, 1e-6, 1e-8)]
    assert error == sorted(error, reverse=True), error     # monotone decrease
    assert error[-1] < 2e-2


def test_dry_middle_raises_instead_of_returning_a_number():
    with pytest.raises(ValueError):
        solve_riemann(1.0, -8.0, 1.0, 8.0)   # diverging faster than 2(cL+cR)
    with pytest.raises(ValueError):
        solve_riemann(1.0, 0.0, 0.0, 0.0)


def test_sampling_reproduces_the_initial_data_at_zero_time():
    case = DEFAULT_CASES[2]
    sol = case.solution()
    x = np.linspace(case.x_min, case.x_max, 101)
    h, u = sample_exact(sol, x, 0.0)
    h0, u0 = case.initial_state(x)
    assert np.array_equal(h, h0) and np.array_equal(u, u0)


def test_oscillation_predicate_matches_the_dam_break_benchmark():
    rng = np.random.default_rng(20261101)
    for _ in range(20):
        h = np.cumsum(rng.normal(0.0, 1e-3, 200)) + 1.0
        floor = EXTREMUM_FRACTION
        counted, _ = __import__(
            "tensorfvm.shallow_water.riemann", fromlist=["_local_extrema"]
        )._local_extrema(h, floor)
        reference = dam_break._local_extrema(h, floor)
        assert counted == reference


def test_corrupted_profile_is_caught():
    """The metric must fail on a profile that is *not* a Riemann solution."""
    case = DEFAULT_CASES[0]
    from tensorfvm.shallow_water.swe1d import ShallowWater1D

    n = 1600
    solver = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity)
    h0, u0 = case.initial_state(solver.x)
    solver.set_state(h0, u0)
    solver.advance(case.t_end)
    h, u = solver.primitive()
    good = profile_metrics(case, solver.x, h, u, solver.time)
    assert verdict(good)["pass"]
    corrupted = h.copy()
    corrupted[n // 2] += 0.02 * case.h_ref          # a 2 % mid-domain spike
    bad = profile_metrics(case, solver.x, corrupted, u, solver.time)
    assert not verdict(bad)["monotone"]
    shifted = profile_metrics(case, solver.x, np.full_like(h, case.hR), u,
                             solver.time)
    assert not verdict(shifted)["pass"]


@pytest.mark.parametrize("case", DEFAULT_CASES, ids=lambda c: c.name)
def test_solver_matches_the_exact_solution_at_the_accepted_grid(case):
    """Accepted grid: n = 3200.  Tolerance checks also hold at n = 1600."""
    record = run_case(case, n=3200)
    assert record["verdict"]["pass"], record["verdict"]
    assert record["metrics"]["depth_l1"] < TOLERANCE
    assert record["discretisation"]["clipped_cells"] == 0
    assert record["discretisation"]["dry_momentum_flips"] == 0
    # transmissive ends carry a flux, so mass is *not* conserved here: the
    # analytic budget t*(hL uL - hR uR) must be reproduced instead.
    assert record["metrics"]["volume_budget_relative"] < 1e-10
    coarse = run_case(case, n=1600)
    assert coarse["verdict"]["failed"] == [], coarse["verdict"]


def test_no_wave_reaches_the_far_boundaries():
    """The budget identity above is only valid if the ends stay undisturbed."""
    for case in DEFAULT_CASES:
        sol = case.solution()
        reach = sol.max_wave_speed * case.t_end
        assert reach < min(abs(case.x_min), abs(case.x_max)) - 0.5, (case.name, reach)


def test_monotonicity_clause_needs_the_accepted_grid():
    """Honest record: at n = 1600 one case still shows a 2e-4 extremum."""
    case = next(c for c in DEFAULT_CASES if c.name == "shock-rarefaction")
    coarse = run_case(case, n=1600)
    assert coarse["verdict"]["failed"] == []                 # tolerances met
    assert coarse["verdict"]["monotone"] is False            # 1 extremum
    assert coarse["metrics"]["extremum_count"] == 1
    assert coarse["metrics"]["extremum_amplitude"] < 5e-4
    assert run_case(case, n=3200)["verdict"]["monotone"] is True


def test_coarse_grid_fails_so_the_gate_is_not_vacuous():
    series = convergence_series(
        next(c for c in DEFAULT_CASES if c.name == "rarefaction-rarefaction"),
        resolutions=(400, 1600))
    assert series["rows"][0]["pass"] is False
    assert series["rows"][1]["pass"] is True
    assert series["rows"][1]["depth_l1"] < series["rows"][0]["depth_l1"]


def test_depth_l1_converges_at_first_order_in_the_shock_cases():
    series = convergence_series(
        next(c for c in DEFAULT_CASES if c.name == "shock-shock"),
        resolutions=(400, 800, 1600))
    orders = series["observed_order_depth_l1"]
    assert all(0.5 < order < 2.0 for order in orders), orders


def test_certification_bundle_is_json_safe_and_complete():
    import json

    bundle = certification(resolutions=(20_001, 40_001))
    text = json.dumps(bundle, allow_nan=False)          # strict JSON
    assert len(text) > 1000
    for case in DEFAULT_CASES:
        entry = bundle["cases"][case.name]
        assert entry["pattern"] == PATTERNS[case.name]
        assert len(entry["weak_form_refinement"]) == 2
    assert "stoker_h1_0.5" in bundle["degenerate_limits"]


def test_metrics_are_finite_for_every_case():
    for case in DEFAULT_CASES:
        record = run_case(case, n=800)
        metrics = record["metrics"]
        for key in ("depth_l1", "velocity_l1", "depth_linf"):
            assert math.isfinite(metrics[key])
        assert metrics["h_min"] > 0.0
        assert metrics["h_max_over_href"] < 1.5

@pytest.mark.parametrize("boost", [-12., 12.])
def test_boosted_wet_wave_profiles(boost, record_property):
    error = 0.
    for case in DEFAULT_CASES:
        base = case.solution()
        moved = solve_riemann(base.hL, base.uL+boost, base.hR, base.uR+boost, base.gravity)
        x = np.linspace(base.span[0]-1, base.span[1]+1, 511)
        h, u = sample_exact(base, x, 1.)
        hm, um = sample_exact(moved, x+boost, 1.)
        error = max(error, np.max(abs(h-hm)), np.max(abs(u+boost-um)))
    assert error < 1e-12
    record_property("max_galilean_discrepancy", float(error))


def test_generated_dry_interval_and_finite_volume_refinement(record_property):
    from tensorfvm.shallow_water import ShallowWater1D, dry_riemann_state

    # g=1, cL=1, cR=1/2. Tail speeds are -1 and +1, heads -4 and +2.5.
    data = (1., -3., .25, 2.)
    xi = np.array([-4., -2.5, -1., 0., 1., 1.75, 2.5])
    h, u = dry_riemann_state(xi, 1., *data, gravity=1.)
    np.testing.assert_allclose(h, [1., .25, 0., 0., 0., .0625, .25])
    np.testing.assert_allclose(u, [-3., -2., 0., 0., 0., 1.5, 2.])
    errors = []
    cell_depth_errors = []
    discharge_errors = []
    for n in (200, 400, 800):
        solver = ShallowWater1D(n, -2., 2., gravity=1., cfl=.4, order=2)
        solver.set_state(*dry_riemann_state(solver.x, 0., *data, gravity=1.))
        volume = solver.total_volume()
        solver.advance(.25)
        h, u = solver.primitive()
        href, _ = dry_riemann_state(solver.x, solver.time, *data, gravity=1.)
        errors.append(float(np.mean(abs(h-href))))
        edges = np.linspace(-2., 2., n+1)
        havg, qavg = exact.dry_riemann_cell_averages(edges, solver.time, *data, gravity=1.)
        discharge_errors.append(float(np.mean(abs(h*u-qavg))))
        cell_depth_errors.append(float(np.mean(abs(h-havg))))
        record_property(f"cell_mean_depth_error_{n}", cell_depth_errors[-1])
        record_property(f"cell_mean_discharge_error_{n}", discharge_errors[-1])
        budget = abs(solver.total_volume()-volume - solver.time*(-3.-.5))
        record_property(f"mean_depth_error_{n}", errors[-1])
        record_property(f"mass_budget_error_{n}", float(budget))
        assert h.min() >= 0 and budget < 1e-11
    for series in (errors, cell_depth_errors, discharge_errors):
        assert all(fine < .8*coarse for coarse,fine in zip(series, series[1:]))


def test_dry_reference_moving_one_sided_beds_and_touching_fronts():
    from tensorfvm.shallow_water import dry_riemann_state

    x = np.linspace(-2, 2, 129)
    h, u = dry_riemann_state(x, .5, 1., .5, 0., -8., gravity=1., x0=.25)
    base_h, base_u = exact.ritter_state(x-.25, .5, 1., gravity=1., x_dam=.25)
    np.testing.assert_allclose(h, base_h)
    np.testing.assert_allclose(u, np.where(h>0, .5+base_u, 0.))
    right_h, right_u = dry_riemann_state(-x, .5, 0., 8., 1., -.5, gravity=1., x0=-.25)
    np.testing.assert_allclose(right_h, h)
    np.testing.assert_allclose(right_u, -u)
    assert dry_riemann_state(0., 1., 1., -2., 1., 2., gravity=1.) == (0., 0.)
    with pytest.raises(ValueError, match="interacting wet waves"):
        dry_riemann_state(0., 1., 1., 0., 1., 0.)


@pytest.mark.parametrize("data", [(1., -3., .25, 2.), (1., .3, 0., 0.),
                                  (0., 0., 1., -.3), (1., -2., 1., 2.)])
def test_dry_cell_means_integrate_boundary_fluxes(data, record_property):
    # Shifted discontinuity falls inside a nonuniform cell. All waves remain
    # inside the domain, so the two unchanged exterior states fix both budgets.
    edges = np.array([-4., -1.2, -.41, -.11, .02, .33, .55, 1.1, 3.])
    x0, t = .07, .24
    hL, uL, hR, uR = data
    h, q = exact.dry_riemann_cell_averages(edges, t, *data, gravity=1., x0=x0)
    left, right = x0-edges[0], edges[-1]-x0
    mass = left*hL + right*hR + t*(hL*uL-hR*uR)
    momentum = left*hL*uL + right*hR*uR + t*(hL*uL**2+.5*hL**2-hR*uR**2-.5*hR**2)
    widths = np.diff(edges)
    error = max(abs(widths@h-mass), abs(widths@q-momentum))
    record_property("maximum_integral_discrepancy", float(error))
    assert error < 5e-14
    h0, q0 = exact.dry_riemann_cell_averages(edges, 0., *data, gravity=1., x0=x0)
    fraction = (x0-.02)/(.33-.02)
    assert h0[4] == pytest.approx(fraction*hL + (1-fraction)*hR)
    assert q0[4] == pytest.approx(fraction*hL*uL + (1-fraction)*hR*uR)


def test_dry_cell_means_resolve_partial_fan_and_vacuum():
    # For Ritter with g=hL=1 at t=1, h=(2-x)^2/9 and
    # hu=2*(1+x)*(2-x)^2/27 in the fan. The second cell straddles x=2.
    h, q = exact.dry_riemann_cell_averages([0., 1., 3., 4.], 1., 1., 0., 0., 0., gravity=1.)
    np.testing.assert_allclose(h, [7/27, 1/54, 0.], atol=1e-15, rtol=0)
    np.testing.assert_allclose(q, [13/54, 1/36, 0.], atol=1e-15, rtol=0)


@pytest.mark.parametrize("case", DEFAULT_CASES, ids=lambda case: case.name)
def test_wet_control_volume_budgets_and_refinement(case, record_property):
    from tensorfvm.shallow_water import riemann_cell_averages

    x0, t = .23, .2
    sol = solve_riemann(case.hL, case.uL, case.hR, case.uR, case.gravity, x0)
    lo, hi = sol.span
    edges = x0+t*np.array([lo-2, lo-.4, lo+.31*(hi-lo), hi+.2, hi+2])
    h, q = riemann_cell_averages(sol, edges, t)
    h0, q0 = riemann_cell_averages(sol, edges, 0.)
    widths = np.diff(edges)
    mass_flux = sol.hL*sol.uL - sol.hR*sol.uR
    force = sol.hL*sol.uL**2-sol.hR*sol.uR**2 + .5*sol.gravity*(sol.hL**2-sol.hR**2)
    balance = max(abs(widths@(h-h0)-t*mass_flux), abs(widths@(q-q0)-t*force))
    record_property("integral_balance_error", float(balance))
    assert balance < 2e-13
    errors = []
    for n in (200, 400, 800):
        metrics = run_case(case, n=n)["metrics"]
        row = [metrics["cell_mean_depth_l1"], metrics["cell_mean_discharge_l1"]]
        for name, value in zip(("depth", "discharge"), row):
            record_property(f"normalized_{name}_error_{n}", value)
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(f < c for c, f in zip(coarse, fine))


def test_wet_cell_cut_by_moving_shock_has_volume_weighted_jump():
    from tensorfvm.shallow_water import riemann_cell_averages

    sol = solve_riemann(1., 1., .7, -1., gravity=1., x0=.25)
    assert sol.right_kind == "shock"
    t = .125
    position = sol.x0+t*sol.right_shock_speed
    width = .4*t*(sol.right_head-sol.left_tail)
    edges = [position-.3*width, position+.7*width]
    h, hu = riemann_cell_averages(sol, edges, t)
    assert h[0] == pytest.approx(.3*sol.h_star+.7*sol.hR, abs=1e-14, rel=0.)
    assert hu[0] == pytest.approx(.3*sol.h_star*sol.u_star+.7*sol.hR*sol.uR, abs=1e-14, rel=0.)


@pytest.mark.parametrize("boost", [-8., 8.])
def test_shifted_moving_star_and_wave_metrics(boost, record_property):
    from dataclasses import replace

    case = replace(DEFAULT_CASES[1], x0=1.17, uL=-1.+boost, uR=1.+boost)
    t = .2
    xs = np.linspace(-3, 5, 4001)
    h, u = case.exact(xs,t)
    m = profile_metrics(case, xs, h, u, t)
    assert m["plateau"]["resolved"] and m["plateau"]["depth_rel_error"] == 0
    worst = max(m[f"wave_{side}"]["rel_error"] for side in ("left","right"))
    assert worst < 1e-5
    record_property("maximum_relative_wave_error", worst)
    u += .2
    m = profile_metrics(case, xs, h, u, t)
    assert m["plateau"]["velocity_rel_error"] == pytest.approx(.2/abs(case.solution().u_star))


def test_off_center_dam_initialization_and_stationary_shock(record_property):
    from dataclasses import replace

    base = DEFAULT_CASES[0]
    speed = base.solution().right_shock_speed
    case = replace(base, x0=.37, uL=-speed, uR=-speed, t_end=.25)
    h0, u0 = case.initial_cell_state(np.array([0.,1.]))
    assert h0[0] == pytest.approx(.37*base.hL + .63*base.hR)
    assert h0[0]*u0[0] == pytest.approx(-speed*h0[0])
    m = run_case(case, n=800)["metrics"]
    shock = m["wave_right"]
    assert shock["x_exact"] == pytest.approx(case.x0)
    assert shock["position_error_scale"] == pytest.approx(case.c_ref*case.t_end)
    assert shock["rel_error"] < .05
    record_property("stationary_shock_position_error", abs(shock["x_measured"]-case.x0))
    record_property("relative_volume_budget_error", m["volume_budget_relative"])
    record_property("relative_momentum_budget_error", m["momentum_budget_relative"])
    assert m["momentum_budget_relative"] < 1e-12
    assert m["volume_budget_relative"] < 1e-12


def test_unresolved_star_plateau_fails_grading():
    case = DEFAULT_CASES[2]
    x = np.array([-10., 10.])
    h, u = case.exact(x, .2)
    metrics = profile_metrics(case, x, h, u, .2)
    assert not metrics["plateau"]["resolved"]
    assert "plateau_velocity" in verdict(metrics)["failed"]


def test_integrated_flux_across_shock_agrees_with_arrival_time(record_property):
    from tensorfvm.shallow_water import riemann_integrated_flux

    sol = solve_riemann(1., 0., .5, 0., x0=.37)
    t, arrival = .75, .45
    x = sol.x0 + arrival*sol.right_shock_speed
    def flux(h, u):
        return np.array([h*u, h*u*u + .5*sol.gravity*h*h])
    expected = arrival*flux(sol.hR, sol.uR) + (t-arrival)*flux(sol.h_star, sol.u_star)
    actual = riemann_integrated_flux(sol, x, t)
    error = np.max(abs(actual-expected))
    record_property("maximum_integrated_flux_error", float(error))
    assert error < 1e-14


@pytest.mark.parametrize("case", DEFAULT_CASES, ids=lambda c: c.name)
def test_integrated_flux_balances_cells_cut_by_waves(case, record_property):
    from tensorfvm.shallow_water import riemann_cell_averages, riemann_integrated_flux

    sol = case.solution()
    edges = np.linspace(-.9, 1.12, 18)
    t = .25
    initial = riemann_cell_averages(sol, edges, 0.)
    final = riemann_cell_averages(sol, edges, t)
    fluxes = riemann_integrated_flux(sol, edges, t)
    for name, start, end, flux in zip(("mass", "momentum"), initial, final, fluxes):
        residual = float(np.max(abs((end-start)*np.diff(edges) + np.diff(flux))))
        record_property(f"max_cell_{name}_residual", residual)
        assert residual < 1e-13
    budget = run_case(case, n=400)["metrics"]["momentum_budget_relative"]
    record_property("normalized_numerical_momentum_residual", budget)
    assert budget < 1e-12


def test_ritter_integrated_flux_during_first_wetting(record_property):
    from tensorfvm.shallow_water import dry_riemann_integrated_flux
    from scipy.integrate import quad

    # Unit Ritter problem. At x=1 the dry front arrives at t=1/2.
    x, end = 1., 1.3
    data = (1., 0., 0., 0.)
    for t in (0., .4, .5):
        assert dry_riemann_integrated_flux(x, t, *data, gravity=1.) == (0., 0.)
    def flux(t):
        h = (2-x/t)**2/9
        u = 2*(1+x/t)/3
        return h*u, h*u**2+h**2/2
    reference = [quad(lambda t: flux(t)[i], .5, end)[0] for i in (0,1)]
    actual = dry_riemann_integrated_flux(x, end, *data, gravity=1.)
    error = float(np.max(np.abs(actual-np.asarray(reference))))
    record_property("maximum_time_quadrature_error", error)
    assert error < 2e-14
    reflected = dry_riemann_integrated_flux(-x, end, 0., 0., 1., 0., gravity=1.)
    np.testing.assert_allclose(reflected, (-actual[0], actual[1]), rtol=0., atol=1e-15)


def test_generated_vacuum_flux_after_drying_and_subdomain_budget(record_property):
    from tensorfvm.shallow_water import ShallowWater1D, dry_riemann_cell_averages, dry_riemann_integrated_flux

    data = (1., -3., .25, 2.)
    end = .25
    positions = np.array([-.125, .125])
    # After the fronts at x=-t,+t pass, local flux stops; the cumulative
    # impulse still includes what crossed each face before it dried.
    at_front = dry_riemann_integrated_flux(positions, .125, *data, gravity=1.)
    after = dry_riemann_integrated_flux(positions, end, *data, gravity=1.)
    np.testing.assert_allclose(after, at_front, atol=1e-15, rtol=0)
    np.testing.assert_allclose(after, [[-.125,.03125],[.375,.0625]], atol=1e-15, rtol=0)
    cuts = [-.625, .375]
    mass, impulse = dry_riemann_integrated_flux(cuts, end, *data, gravity=1.)
    errors = []
    for n in (128, 256, 512):
        solver = ShallowWater1D(n, -2., 2., gravity=1., cfl=.4, order=2)
        edges = np.linspace(-2., 2., n+1)
        h0, q0 = dry_riemann_cell_averages(edges, 0., *data, gravity=1.)
        solver.set_state(h0, q0/h0)
        solver.advance(end)
        h, u = solver.primitive()
        inside = (solver.x>cuts[0]) & (solver.x<cuts[1])
        row = []
        for name, change, transport in (("mass",h-h0,mass), ("momentum",h*u-q0,impulse)):
            error = abs(solver.dx*change[inside].sum() - (transport[0]-transport[1]))
            record_property(f"{name}_subdomain_error_{n}", float(error))
            row.append(error)
        errors.append(row)
    for coarse, fine in zip(errors, errors[1:]):
        assert all(b < a for a,b in zip(coarse, fine))
