"""Gate for the Sod (1978) reproduction at t = 0.14 (``REPRO-FVM-SOD1978``).

The pytest-sized version keeps the same criteria as
``scripts/repro_fvm_sod1978.py`` but runs three grids so it stays inside the
routine gate budget; the artifact-producing run uses four grids.
"""

import math

import pytest

from tensorfvm.verification.sod1978 import (
    PUBLISHED,
    T_TARGET,
    X0,
    exact_star_state,
    reference_cross_check,
    run_case,
    run_smooth_wave,
    star_plateaus,
    wave_positions,
)

GRIDS = (100, 200, 400)
REF_NX = 400


@pytest.fixture(scope="module")
def reference():
    return exact_star_state()


@pytest.fixture(scope="module")
def profile():
    return run_case(REF_NX, order=2, t_end=T_TARGET)


def test_reference_anchor_is_independently_confirmed():
    """The grading reference agrees with the published star state to 4 decimals.

    Two implementations (bisection here, Newton in ``ExactRiemann``) plus the
    published table: a disagreement would invalidate every comparison below.
    """
    check = reference_cross_check()
    for key in ("p_star", "u_star", "rho_star_left", "rho_star_right",
                "shock_speed", "contact_speed"):
        assert check[key]["independent_vs_published_pct"] < 5e-3, key
        assert check[key]["tensorfvm_vs_published_pct"] < 5e-3, key


def test_star_region_within_5pct(profile, reference):
    """Criterion 2: rho*_L, rho*_R, u*, p* within 5 % of the published values."""
    plateau = star_plateaus(profile["x"], profile, reference, profile["t_end"])
    for key in ("rho_star_left", "rho_star_right", "u_star", "p_star"):
        rel = abs(plateau[key] - PUBLISHED[key]) / abs(PUBLISHED[key])
        assert rel < 0.05, f"{key}: {plateau[key]:.5f} vs {PUBLISHED[key]:.5f}"


def test_wave_positions_within_2pct(profile, reference):
    """Criterion 1: the four wave positions at t = 0.14 within 2 % of x0 + S t."""
    plateau = star_plateaus(profile["x"], profile, reference, profile["t_end"])
    waves = wave_positions(profile["x"], profile, reference, profile["t_end"], plateau)
    for key, speed in (("rarefaction_head_x", "rarefaction_head_speed"),
                       ("rarefaction_tail_x", "rarefaction_tail_speed"),
                       ("contact_x", "contact_speed"),
                       ("shock_x", "shock_speed")):
        exact_x = X0 + reference[speed] * T_TARGET
        assert waves[key] == pytest.approx(exact_x, rel=0.02), key


def test_sod_reproduction_is_convergent_and_conservative():
    """Criterion 3 (literal): the L1 error must fall with refinement.

    The observed rate on a discontinuous solution is reported by the driver; here
    the gate only asks that refinement helps and that no positivity repair or
    conservation drift was needed, i.e. the solution is not silently broken.
    """
    runs = [run_case(nx, order=2, t_end=T_TARGET) for nx in (100, 200)]
    earlier, later = runs
    for key in ("rho", "u", "p"):
        exact = _exact(later, key)
        e_late = abs(later[key] - exact).mean()
        exact = _exact(earlier, key)
        e_early = abs(earlier[key] - exact).mean()
        assert e_late < e_early, key
    assert max(r["positivity_fixes"] for r in runs) == 0
    assert abs(runs[-1]["budget_drift"]["mass"]) < 5e-3


def test_smooth_wave_has_design_order():
    """Criterion 3 (design): the scheme is >= 1st order on smooth data.

    A TVD scheme cannot exceed L1 order ~1 across the Sod discontinuities, so the
    design order is certified on a periodic entropy wave with the same solver:
    donor cell -> 1, MUSCL + MC -> >= 1.5 (limiters lose order at smooth extrema).
    """
    # declared band: observed rate within 2 % of the design order (1 and 2); a
    # first-order upwind rate approaches 1.0 from below (0.99 at these grids)
    # nx starts at 100: at 50 -> 100 the second-order rate is still pre-asymptotic
    # (1.94 there vs 2.13/2.09/2.05 on 100 -> 800 in the full reproduction)
    for order in (1, 2):
        runs = [run_smooth_wave(nx, order=order) for nx in (100, 200, 400)]
        assert 0.0 < runs[-1]["l1_rho"] < runs[0]["l1_rho"]
        rate = min(math.log(a["l1_rho"] / b["l1_rho"]) / math.log(b["nx"] / a["nx"])
                   for a, b in zip(runs[:-1], runs[1:]))
        assert rate >= order * 0.98, f"order {order}: observed {rate:.2f}"


def test_published_positions_scale_with_the_output_time():
    """Self-similarity guard: the star state is time-invariant, positions are not.

    Reporting a t = 0.14 run against t = 0.2 positions (or vice versa) is a 43 %
    error on the shock position, so the time scaling must stay explicit.
    """
    star = exact_star_state()
    x_t14 = X0 + star["shock_speed"] * 0.14
    x_t20 = X0 + star["shock_speed"] * 0.20
    assert x_t20 - x_t14 == pytest.approx(0.06 * star["shock_speed"], rel=1e-12)
    assert (x_t20 - x_t14) / x_t14 > 0.05


def _exact(run, key):
    import numpy as np
    from tensorfvm.verification.sod1978 import exact_sample
    return exact_sample((run["x"] - X0) / run["t_end"], exact_star_state())[key]