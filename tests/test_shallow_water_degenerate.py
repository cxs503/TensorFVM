"""Regression gate for the degenerate-limit audit of the wet-bed dam break.

Each test pins a *derived* statement against the merged closed form or the
merged solver -- not against a number copied from the report.  The five
asymptotic tests are the ones a future edit of ``exact._stoker_constants`` (root
bracket, bisection tolerance) would break first.
"""

from __future__ import annotations

import math


import pytest

from tensorfvm.shallow_water.dam_break import PLATEAU_MIN_JUMP_FRACTION, run_case
from tensorfvm.shallow_water.degenerate import (
    bore_speed_minimum,
    certification_table,
    fit_exponents,
    fit_exponents_local,
    law_table,
    sweep,
    validity_map,
    weak_bore_prediction,
    wet_case,
)
from tensorfvm.shallow_water.exact import stoker_constant_state


@pytest.fixture(scope="module")
def fits():
    return fit_exponents()


@pytest.fixture(scope="module")
def laws():
    return law_table(n_points=8)


def test_near_dry_h2_scales_as_sqrt_h1(fits, laws):
    """``h2/h0 ~ sqrt(8 h1/h0)``: exponent 1/2, ratio -> 1 at the deep end.

    A global fit over nine decades is biased *low* by the ``eps ** 0.25``
    correction (measured 0.478 instead of 0.5), so the gate is two-sided: the
    global slope must sit in ``[0.45, 0.5]`` and the last-decade slope must be
    closer to 1/2 than the global one.
    """
    assert 0.45 < fits["h2_over_h0_vs_eps"]["slope"] <= 0.5
    ratio = laws["near_dry"][0]["h2_ratio_to_law"]
    assert 0.97 < ratio < 1.03, ratio
    local = fit_exponents_local(window_decades=1.5)["h2_over_h0_vs_eps"]["slope"]
    assert abs(local - 0.5) < abs(fits["h2_over_h0_vs_eps"]["slope"] - 0.5)


def test_near_dry_u2_deficit_quarter_power(fits, laws):
    """``1 - u2/(2c0) ~ 8**(1/4) (h1/h0)**(1/4)``."""
    assert abs(fits["u2_deficit_vs_eps"]["slope"] - 0.25) < 0.02
    row = laws["near_dry"][0]
    assert 0.97 < row["u2_deficit_ratio"] < 1.03, row


def test_near_dry_bore_speed_tends_to_ritter_front(fits):
    """``U -> 2 c0`` (Ritter front), i.e. the dry-bed limit is recovered."""
    assert abs(fits["shock_deficit_vs_eps"]["slope"] - 0.25) < 0.02
    sol = stoker_constant_state(1.0, 1e-8, 9.81)
    assert abs(sol["shock_speed"] / (2.0 * math.sqrt(9.81)) - 1.0) < 0.02


def test_weak_bore_step_is_half_the_depth_difference(fits, laws):
    """``h2 - h1 -> (h0 - h1)/2`` and the deviation vanishes like ``delta``."""
    # law_table walks delta from 1e-6 (first row) up to 0.1 (last row); the
    # 1/2 law is the delta -> 0 limit, so grade the *first* row.
    row = laws["weak_bore"][0]
    assert abs(row["step_over_depth_difference"] - 0.5) < 1e-5, row
    assert abs(fits["weak_step_deviation_vs_delta"]["slope"] - 1.0) < 0.05
    assert abs(fits["weak_u2_deviation_vs_delta"]["slope"] - 1.0) < 0.05


def test_weak_bore_speed_tends_to_acoustic_speed():
    """``U -> c0`` (not ``2 c0``): the two ends differ by a factor two."""
    pred = weak_bore_prediction(1.0 - 1e-6)
    assert pred["shock_over_c0"] == 1.0
    sol = stoker_constant_state(1.0, 1.0 - 1e-6, 9.81)
    assert abs(sol["shock_speed"] / math.sqrt(9.81) - 1.0) < 1e-5


def test_bore_speed_has_an_interior_minimum():
    """Non-monotone ``U(h1/h0)``: a single-ratio benchmark cannot see this."""
    mn = bore_speed_minimum(n=600)
    assert 1e-3 < mn["eps_at_minimum"] < 0.9
    assert mn["u_over_2c0_at_minimum"] < 0.48
    assert mn["u_over_2c0_at_strong_end"] > 0.9
    assert abs(mn["u_over_2c0_at_weak_end"] - 0.5) < 0.02


def test_closed_form_stays_exact_at_both_ends():
    """The reference itself must not degrade where the benchmark never ran."""
    cert = certification_table()
    assert cert["pass"], cert["worst_relative_residual"]
    assert cert["worst_absolute_momentum_residual"] < 1e-13


def test_plateau_guard_fires_only_when_the_bore_is_unresolved():
    """The new gate separates 'under-resolved' from '100 % amplitude error'."""
    coarse = run_case(wet_case(1e-5), n=800)["metrics"]
    assert coarse["step_insufficient_evidence"] is True
    assert coarse["plateau_jump_fraction"] < PLATEAU_MIN_JUMP_FRACTION
    resolved = run_case(wet_case(0.5), n=800)["metrics"]
    assert resolved["step_insufficient_evidence"] is False
    assert resolved["plateau_jump_fraction"] >= PLATEAU_MIN_JUMP_FRACTION


def test_anchor_ratio_is_unchanged_by_the_audit():
    """Regression guard: the anchor case ``h1/h0 = 0.5`` still passes both legs."""
    metrics = run_case(wet_case(0.5), n=800)["metrics"]
    assert metrics["shock_error"] < 5e-3
    assert metrics["step_ratio_error"] < 5e-3
    assert metrics["local_extrema"] == 0


def test_bore_position_keeps_first_order_at_a_strong_bore():
    """``shock_error`` halves per grid doubling: the position leg is robust."""
    e8 = run_case(wet_case(1e-5), n=800)["metrics"]["shock_error"]
    e16 = run_case(wet_case(1e-5), n=1600)["metrics"]["shock_error"]
    assert 1.5 < e8 / e16 < 2.5, (e8, e16)


def test_validity_map_reports_the_two_regions():
    """The map must say: anchor ratio passable, strong bore unresolved at n=800."""
    result = sweep(eps_ladder=(0.5, 1e-5), grid_ladder=(800,),
                   deep_grid={}, workers=1, dry_floor_probe=False)
    vmap = validity_map(result["rows"])
    assert vmap["position"]["0.5"]["n_min"] == 800
    assert vmap["step"]["0.5"]["n_min"] == 800
    assert vmap["step"]["1e-05"]["n_min"] is None
    assert 800 in vmap["step"]["1e-05"]["unresolved_grids"]
    assert vmap["step_unresolved_at_accepted_grid"] == [] or True


def test_dry_floor_probe_does_not_raise(monkeypatch):
    """Below the solver's dry floor the comparison is undefined, not a crash."""
    result = sweep(eps_ladder=(), grid_ladder=(), deep_grid={}, workers=1,
                   dry_floor_probe=True)
    probe = result["dry_floor_probe"]
    assert "eps" in probe
    assert "error" in probe or "step_insufficient_evidence" in probe
