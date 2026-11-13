"""Independent verification of the dam-break analytic references.

The benchmark anchor ``CASE-FVM-DAMBREAK-STOKER`` compares a finite-volume
solution against the closed-form Ritter/Stoker solutions of
:mod:`tensorfvm.shallow_water.exact`.  Those closed forms assume a wave pattern
(left rarefaction + constant state + right bore) and their own check
(:func:`check_stoker_solution`) re-evaluates the *same* algebra, so a shared
mistake would pass unnoticed.

This module crosses the references against three genuinely independent
yardsticks:

1. a generic exact Riemann solver for the shallow-water equations
   (:mod:`tensorfvm.shallow_water.crosscheck`), which solves for the star state
   by root finding without assuming the Stoker wave pattern;
2. wave speeds *measured* from the sampled reference profile (finite difference
   of the jump trajectory), not read off the closed-form constant;
3. the governing conservation law itself, checked by finite differences of the
   reference in its smooth regions.

It also pins the weak-bore conditioning of the bore speed (regression test: the
subtractive Rankine-Hugoniot form loses ~8e-4 relative accuracy at
``h1 = h0 (1 - 1e-12)``, the cancellation-free form is at 1e-13).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water import (
    RITTER_DRY,
    STOKER_WET,
    ritter_front_position,
    ritter_state,
    stoker_constant_state,
    stoker_state,
)
from tensorfvm.shallow_water.crosscheck import (
    jump_trajectory_speed,
    level_trajectory_speed,
    pressure_function,
    riemann_solution,
    riemann_star_state,
)

G = 9.81
H0 = 1.0
RATIOS = (0.99, 0.9, 0.75, 0.5, 0.25, 0.1, 0.05, 0.01, 1e-3)


def _bore_state_high_precision(h0: float, h1: float, gravity: float,
                               digits: int = 60):
    """Bore state ``(h2, u2, U)`` in ``Decimal`` arithmetic (independent of float).

    Same scalar root as :func:`tensorfvm.shallow_water.exact.stoker_constant_state`
    but evaluated with ``digits`` significant digits, so it exposes the float
    implementation's conditioning instead of sharing it.
    """
    from decimal import Decimal, localcontext

    with localcontext() as ctx:
        ctx.prec = digits
        H0d, H1d, Gd = Decimal(repr(h0)), Decimal(repr(h1)), Decimal(repr(gravity))
        c0 = (Gd * H0d).sqrt()
        c1 = (Gd * H1d).sqrt()

        def residual(c2):
            h2 = c2 * c2 / Gd
            return 2 * (c0 - c2) - (h2 - H1d) * (Gd * (h2 + H1d)
                                                 / (2 * h2 * H1d)).sqrt()

        lo, hi = c1, c0
        for _ in range(400):
            mid = (lo + hi) / 2
            if residual(mid) > 0:
                lo = mid
            else:
                hi = mid
        c2 = (lo + hi) / 2
        h2 = c2 * c2 / Gd
        u2 = 2 * (c0 - c2)
        speed = (Gd * h2 * (h2 + H1d) / (2 * H1d)).sqrt()
        return float(h2), float(u2), float(speed)


# --------------------------------------------------------------------------- #
# 1. closed form vs generic Riemann solver
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("ratio", RATIOS)
def test_closed_form_matches_generic_riemann_solver(ratio):
    """Stoker's bore state must equal the root-found star state."""
    h1 = ratio * H0
    closed = stoker_constant_state(H0, h1, G)
    generic = riemann_star_state(H0, 0.0, h1, 0.0, G)
    assert generic["left_family"] == "rarefaction"
    assert generic["right_family"] == "shock"
    assert generic["residual"] < 1e-12 * H0
    assert generic["h_star"] == pytest.approx(closed["h2"], rel=1e-12)
    assert generic["u_star"] == pytest.approx(closed["u2"], rel=1e-11,
                                              abs=1e-13)
    assert generic["s_shock_right"] == pytest.approx(closed["shock_speed"],
                                                     rel=1e-12)


@pytest.mark.parametrize("case", [RITTER_DRY, STOKER_WET], ids=["dry", "wet"])
def test_closed_form_profile_matches_generic_solution(case):
    """Whole profile, not just the star point: L1 over the domain."""
    t = case.t_end
    x = np.linspace(case.x_min, case.x_max, 20_001)
    h_closed, u_closed = case.exact(x, t)
    h_generic, u_generic = riemann_solution(x, t, case.h0, 0.0, case.h1, 0.0,
                                            case.gravity, x0=case.x_dam)
    assert np.abs(h_closed - h_generic).sum() / h_closed.sum() < 1e-14
    u_scale = max(np.abs(u_closed).max(), 1e-30)
    assert np.abs(u_closed - u_generic).max() / u_scale < 1e-12


def test_dry_limit_of_the_wet_solution_is_ritter():
    """h1 -> 0 must continuously recover the Ritter dry-bed solution."""
    t = 2.0 * math.sqrt(H0 / G)
    x = np.linspace(RITTER_DRY.x_min, RITTER_DRY.x_max, 40_001)
    h_ritter, _ = ritter_state(x, t, H0, G)
    errors = []
    front_errors = []
    for h1 in (1e-1, 1e-2, 1e-3, 1e-4, 1e-6):
        h_wet, _ = stoker_state(x, t, H0, h1, G)
        errors.append(np.abs(h_wet - h_ritter).sum() / h_ritter.sum())
        sol = stoker_constant_state(H0, h1, G)
        front_errors.append(abs(sol["shock_speed"] - 2.0 * math.sqrt(G * H0))
                            / (2.0 * math.sqrt(G * H0)))
    assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
    assert errors[-1] < 1e-4
    # The bore speed approaches the dry-bed front speed slowly: the star depth
    # scales as h2 ~ 2 c0 sqrt(2 h1 / g), so the deficit is O(sqrt(h1)) --
    # 5.1 % at h1 = 1e-6, 1.6 % at h1 = 1e-8.  This is a property of the exact
    # solution, not a defect of the closed form.
    assert all(later < earlier for earlier, later in zip(front_errors,
                                                         front_errors[1:]))
    assert front_errors[-1] < 1e-1, front_errors[-1]
    # and the limit is quantitative: L1 ~ h1^(2/3) over this window (the
    # residual is the shrinking plateau plus the dry-bed tip layer)
    slope = np.polyfit(np.log([1e-2, 1e-3, 1e-4]), np.log(errors[1:4]), 1)[0]
    assert 0.5 < slope < 0.8, slope


# --------------------------------------------------------------------------- #
# 2. wave speeds measured from the sampled reference
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("h1", [0.5, 0.25, 0.05])
def test_bore_speed_measured_from_the_sampled_profile(h1):
    """dx/dt of the sampled bore vs the closed-form speed."""
    closed = stoker_constant_state(H0, h1, G)
    traj = jump_trajectory_speed(
        lambda xi: stoker_state(xi, 1.0, H0, h1, G),
        0.5 * closed["shock_speed"], 1.5 * closed["shock_speed"], 1.0, 2.0)
    assert traj["jump_height"] == pytest.approx(closed["h2"] - closed["h1"],
                                                rel=1e-6)
    assert traj["speed"] == pytest.approx(closed["shock_speed"], rel=1e-4)


@pytest.mark.parametrize("h1", [0.5, 0.25])
def test_wet_fan_level_trajectories_follow_the_ritter_fan_law(h1):
    """The wet-bed fan is the Ritter fan: ``xi_L = 2 c0 - 3 sqrt(g L)``.

    Measuring the trajectory of an interior depth level (rather than the fan
    edges, which are limits) certifies the fan's self-similar shape; the fan/plateau
    junction speed ``u2 - c2`` then follows from the Riemann invariant
    ``u2 + 2 c2 = 2 c0`` evaluated at ``L = h2``, i.e. ``2 c0 - 3 c2``.
    """
    closed = stoker_constant_state(H0, h1, G)
    c0 = closed["c0"]
    for frac in (0.1, 0.5, 0.9):
        level = closed["h2"] + frac * (closed["h0"] - closed["h2"])
        traj = level_trajectory_speed(
            lambda xi: stoker_state(xi, 1.0, H0, h1, G), level,
            closed["head_speed"], closed["tail_speed"], 1.0, 2.0)
        assert traj["speed"] == pytest.approx(2.0 * c0 - 3.0 * math.sqrt(G * level),
                                              rel=1e-8)
    assert closed["tail_speed"] == pytest.approx(2.0 * c0 - 3.0 * closed["c2"],
                                                 rel=1e-12)
    assert closed["tail_speed"] < closed["shock_speed"]


@pytest.mark.parametrize("level", [0.01, 0.1, 0.3, 0.6])
def test_ritter_level_trajectories_follow_the_exact_fan_law(level):
    """``x_level + 3 t sqrt(g level) = x_front`` measured, not assumed."""
    c0 = math.sqrt(G * H0)
    traj = level_trajectory_speed(lambda xi: ritter_state(xi, 1.0, H0, G),
                                  level, -c0, 2.5 * c0, 1.0, 2.0)
    exact = 2.0 * c0 - 3.0 * math.sqrt(G * level)
    assert traj["speed"] == pytest.approx(exact, rel=1e-8)


def test_dry_front_speed_from_the_generic_riemann_construction():
    """The Ritter front speed must fall out of the dry-bed Riemann closure."""
    generic = riemann_star_state(H0, 0.0, 0.0, 0.0, G)
    front = 2.0 * math.sqrt(G * H0)
    assert generic["s_shock_right"] == pytest.approx(front, rel=1e-14)
    t = 0.5
    assert ritter_front_position(t, H0, G) == pytest.approx(generic["s_shock_right"] * t,
                                                            rel=1e-14)


# --------------------------------------------------------------------------- #
# 3. the reference must satisfy the governing equations
# --------------------------------------------------------------------------- #
def test_reference_satisfies_the_conservation_law_in_smooth_regions():
    """Finite-difference residual of the shallow-water PDE at smooth points."""
    t = 0.7
    eps_t = 1e-5 * t
    eps_x = 1e-5

    def flux(h, u):
        return h * u, h * u * u + 0.5 * G * h * h

    def residual(state):
        h_t = (state(np.array([0.0]), t + eps_t)[0]
               - state(np.array([0.0]), t - eps_t)[0]) / (2 * eps_t)
        h_p, u_p = state(np.array([eps_x]), t)
        h_m, u_m = state(np.array([-eps_x]), t)
        hu_p, f_p = flux(h_p[0], u_p[0])
        hu_m, f_m = flux(h_m[0], u_m[0])
        d_hu_dx = (hu_p - hu_m) / (2 * eps_x) if np.ndim(hu_p) else (
            (hu_p - hu_m) / (2 * eps_x))
        d_f_dx = (f_p - f_m) / (2 * eps_x)
        return abs(h_t[0] + d_hu_dx), abs(d_f_dx)

    # Stoker: x = 0 lies in the rarefaction, x = 0.5 in the constant state.
    for x_pt in (0.0, 0.5):
        mass_res, mom_res = residual(
            lambda x, tt: stoker_state(x + x_pt, tt, H0, 0.5, G))
        assert mass_res < 1e-6, (x_pt, mass_res)
        assert mom_res < 1e-6, (x_pt, mom_res)
    # generic solver, same points
    for x_pt in (0.0, 0.5):
        mass_res, mom_res = residual(
            lambda x, tt: riemann_solution(x + x_pt, tt, H0, 0.0, 0.5, 0.0, G))
        assert mass_res < 1e-6, (x_pt, mass_res)
        assert mom_res < 1e-6, (x_pt, mom_res)


# --------------------------------------------------------------------------- #
# 4. generic solver self-consistency (it must be right for general data too)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("h_l,u_l,h_r,u_r", [
    (1.0, 0.0, 0.5, 0.0),
    (1.0, 1.0, 1.0, -1.0),
    (1.0, -1.0, 1.0, 1.0),
    (2.0, 0.5, 0.5, -0.5),
    (0.5, 2.0, 1.0, 0.0),
], ids=["dam", "collide", "split", "shear", "push"])
def test_generic_solver_satisfies_rankine_hugoniot(h_l, u_l, h_r, u_r):
    """Whatever wave pattern is found, the jumps must be consistent."""
    star = riemann_star_state(h_l, u_l, h_r, u_r, G)
    hs, us = star["h_star"], star["u_star"]
    assert star["residual"] < 1e-10 * max(h_l, h_r)
    # star state continuity written in the two-sided form
    # ``u* = u_L - f_L(h*) = u_R + f_R(h*)`` (the sign of ``f`` is the
    # convention of this module: ``f = 2(c - c_K)`` on the rarefaction branch)
    lhs = u_l - float(pressure_function(hs, h_l, G))
    rhs = u_r + float(pressure_function(hs, h_r, G))
    assert lhs == pytest.approx(rhs, rel=1e-9, abs=1e-10)
    assert lhs == pytest.approx(us, rel=1e-9, abs=1e-10)
    # mass jump across each non-rarefaction wave
    for family, s_speed, s_mass, h_k, u_k in (
            (star["left_family"], star["s_head_left"],
             star["s_from_mass_left"], h_l, u_l),
            (star["right_family"], star["s_shock_right"],
             star["s_from_mass_right"], h_r, u_r)):
        if family == "shock":
            assert s_mass == pytest.approx(s_speed, rel=1e-10)
            assert s_speed * (hs - h_k) == pytest.approx(hs * us - h_k * u_k,
                                                         rel=1e-10, abs=1e-12)


def test_generic_solver_symmetric_data_gives_symmetric_solution():
    two_shocks = riemann_star_state(1.0, 1.0, 1.0, -1.0, G)
    assert two_shocks["left_family"] == "shock"
    assert two_shocks["right_family"] == "shock"
    assert two_shocks["u_star"] == pytest.approx(0.0, abs=1e-12)
    assert two_shocks["s_head_left"] == pytest.approx(
        -two_shocks["s_shock_right"], rel=1e-12)
    two_fans = riemann_star_state(1.0, -1.0, 1.0, 1.0, G)
    assert two_fans["left_family"] == "rarefaction"
    assert two_fans["right_family"] == "rarefaction"
    assert two_fans["h_star"] < 1.0
    assert two_fans["u_star"] == pytest.approx(0.0, abs=1e-12)


def test_pressure_function_is_monotone_and_vanishes_at_the_side_state():
    hs = np.linspace(1e-3, 3.0, 500)
    f = pressure_function(hs, 1.0, G)
    assert np.all(np.diff(f) > 0.0)
    assert float(pressure_function(1.0, 1.0, G)) == pytest.approx(0.0, abs=1e-15)
    # both branches meet continuously at h = h_side
    left = float(pressure_function(1.0 - 1e-12, 1.0, G))
    right = float(pressure_function(1.0 + 1e-12, 1.0, G))
    assert abs(left - right) < 1e-9


# --------------------------------------------------------------------------- #
# 5. weak-bore conditioning regression
# --------------------------------------------------------------------------- #
def test_weak_bore_speed_has_no_cancellation_error():
    """The weak-bore deficit must scale linearly, i.e. be free of cancellation.

    ``c0 - U`` vanishes like the depth gap, so the ratio ``(c0 - U) / gap`` is
    the physically meaningful weak-bore coefficient.  The subtractive form
    ``U = h2 u2 / (h2 - h1)`` amplifies the rounding error of ``h2 - h1`` by
    ``h1 / (h2 - h1)`` (8.0e-4 relative error on ``U``, i.e. a completely wrong
    coefficient, at ``h1 = h0 (1 - 1e-12)``); the cancellation-free form used by
    :func:`stoker_constant_state` keeps the coefficient stable to ~1e-4.
    """
    for gap, tolerance in ((1e-6, 1e-12), (1e-9, 1e-12), (1e-12, 1e-14)):
        h1 = H0 * (1.0 - gap)
        sol = stoker_constant_state(H0, h1, G)
        _, _, u_true = _bore_state_high_precision(H0, h1, G)
        assert sol["shock_speed"] == pytest.approx(u_true, rel=tolerance), (
            gap, sol["shock_speed"], u_true)
        # the subtractive Rankine-Hugoniot form is what the code used to return
        subtractive = sol["h2"] * sol["u2"] / (sol["h2"] - h1)
        subtractive_error = abs(subtractive - u_true) / u_true
        if gap == 1e-12:
            # documented defect: 7.6e-4 relative error at a 1e-12 depth gap,
            # i.e. ~10^12 ulp, because h1/(h2-h1) ~ 2e12 amplifies the rounding
            # of the depth difference
            assert subtractive_error > 1e-5
            assert abs(sol["shock_speed"] - u_true) / u_true < 1e-3 * subtractive_error
    # a moderate bore must be unchanged: the identity is algebraically exact
    sol = stoker_constant_state(H0, 0.5, G)
    assert sol["shock_speed"] == pytest.approx(
        math.sqrt(G * sol["h2"] * (sol["h2"] + 0.5) / (2.0 * 0.5)), rel=1e-15)
    assert sol["shock_speed"] * (sol["h2"] - 0.5) == pytest.approx(
        sol["h2"] * sol["u2"], rel=1e-12)