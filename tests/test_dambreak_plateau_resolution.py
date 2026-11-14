"""F9: the data-driven wet-bed plateau detector must never be silently wrong.

Background
----------
``data_driven_plateau`` (the F4 fix: measure the Stoker plateau/bore from the
numerical profile without using analytic positions) delimits its window with a
cut ``thresh = 1e-2 * max|dh/dx|`` and used to claim grid independence because
"both |dh/dx| scale with 1/dx".

That claim is false for the *fan*: the fan is a smooth feature, so its physical
slope is a constant (0.298 -> 0.308 m/m for n = 800 ... 12800 in the solver)
while ``max|dh/dx|`` doubles per refinement.  The ratio fan/jump therefore falls
like ``dx`` and crosses the 1e-2 cut: on n = 6400 the walk ran through the whole
rarefaction fan and reported ``h2`` **15.5 %** high and ``bore_x`` 0.85 cell off,
while still reporting ``bore_method == "mid_level_crossing"``.  See
``docs/case-fvm-dambreak-stoker-hires/REPORT.md`` (finding F9).

What this file locks
--------------------
1. the invariant: the detector either returns an amplitude within 1 % of the
   exact plateau **or** declares ``plateau_unbounded`` -- never a confident
   wrong number (checked on the analytic Stoker profile for n = 400 ... 12800,
   where the fan/jump gradient ratio spans 5e-2 ... 3e-3);
2. the mechanism: the cut is tightened by declared decades
   (``PLATEAU_THRESH_MAX_DECADES``) and every attempt is recorded;
3. the auxiliary quality flag: ``flatness``/``bore_valid`` separates a genuinely
   flat plateau from a coarse-grid ramp;
4. the solver side: at n = 6400 (where the old code failed) the wet-bed bore
   amplitude and sub-cell position are within their criteria again.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    PLATEAU_FLAT_FRACTION,
    PLATEAU_THRESH_FRACTION,
    PLATEAU_THRESH_MAX_DECADES,
    STOKER_WET,
    data_driven_plateau,
    run_case,
)
from tensorfvm.shallow_water.exact import stoker_constant_state

GRIDS = (400, 800, 1600, 3200, 6400, 12800)
H0, H1, GRAVITY = 1.0, 0.5, 9.81


def _analytic_profile(n: int, t: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Analytic Stoker profile on a uniform grid (a *synthetic* sharp jump).

    Using the exact solution keeps the test independent of the solver: the jump
    gradient is ``(h2 - h1) / dx`` (grows like 1/dx) while the fan slope is the
    physical constant, so this reproduces the F9 situation by construction.
    """
    x_min, x_max = -5.0, 5.0
    dx = (x_max - x_min) / n
    x = x_min + (np.arange(n) + 0.5) * dx
    h, _ = STOKER_WET.exact(x, t)
    sol = stoker_constant_state(H0, H1, GRAVITY)
    return x, np.asarray(h, dtype=float), sol


def _fan_slope(x: np.ndarray, h: np.ndarray, sol: dict, t: float) -> float:
    """Measured physical slope of the rarefaction fan (diagnostic, no oracle)."""
    x_tail = (sol["u2"] - sol["c2"]) * t
    x_head = -sol["c0"] * t
    lo, hi = min(x_tail, x_head), max(x_tail, x_head)
    dx = x[1] - x[0]
    band = (x[:-1] > lo + 0.1 * (hi - lo)) & (x[:-1] < hi - 0.1 * (hi - lo))
    grad = np.abs(np.diff(h)) / dx
    return float(grad[band].mean())


def test_fan_swallowing_never_returns_a_confident_wrong_amplitude():
    """F9 invariant: |h2 error| < 1 % whenever a value is returned."""
    t = 0.5
    ratios = {}
    for n in GRIDS:
        x, h, sol = _analytic_profile(n, t)
        dd = data_driven_plateau(x, h)
        dx = x[1] - x[0]
        max_grad = float(np.abs(np.diff(h)).max() / dx)
        ratios[n] = _fan_slope(x, h, sol, t) / max_grad
        if dd["bore_method"] == "plateau_unbounded":
            continue                                        # honest refusal
        assert dd["bore_method"] == "mid_level_crossing", (n, dd["bore_method"])
        assert abs(dd["h2"] - sol["h2"]) / sol["h2"] < 1e-2, (n, dd["h2"], sol["h2"])
    # the situation the old code failed on really is present in this fixture:
    # the fan/jump ratio crosses the declared 1e-2 cut within the sweep
    assert ratios[GRIDS[0]] > PLATEAU_THRESH_FRACTION
    assert ratios[GRIDS[-1]] < PLATEAU_THRESH_FRACTION
    assert ratios[GRIDS[-1]] < ratios[GRIDS[0]] / 10.0


def test_threshold_retry_is_declared_and_bounded():
    """The retry loop is a resolution loop, not a fit: decades, bounded count."""
    x, h, _ = _analytic_profile(12800, 0.5)
    dd = data_driven_plateau(x, h)
    attempts = dd["attempts"]
    assert 1 <= len(attempts) <= PLATEAU_THRESH_MAX_DECADES + 1
    assert attempts[0]["thresh_frac"] == pytest.approx(PLATEAU_THRESH_FRACTION)
    for prev, cur in zip(attempts, attempts[1:]):
        assert cur["thresh_frac"] == pytest.approx(prev["thresh_frac"] / 10.0)
        assert cur["thresh"] < prev["thresh"]
    assert attempts[-1]["bounded"] is True
    # the first (declared) cut could not bound the plateau at this resolution
    assert attempts[0]["bounded"] is False, attempts[0]
    assert dd["thresh_frac_used"] < PLATEAU_THRESH_FRACTION
    # the plateau width is a physical length (the full plateau is
    # x_shock - x_fan_end = 3.0 m; the walk stops where the fan steepens, so the
    # detected width is ~2.35 m -- what matters is that it does not scale with
    # dx: the same 2.35 m is read at n = 3200 and at n = 12800)
    assert 2.0 < dd["plateau_width"] < 3.3, dd["plateau_width"]
    x_coarse, h_coarse, _ = _analytic_profile(3200, 0.5)
    coarse = data_driven_plateau(x_coarse, h_coarse)
    assert abs(coarse["plateau_width"] - dd["plateau_width"]) < 0.15


def test_flatness_flag_separates_flat_plateau_from_coarse_ramp():
    """``bore_valid`` is the auxiliary flag; a ramp plateau is not "valid"."""
    t = 0.5
    x_f, h_f, sol = _analytic_profile(1600, t)              # genuinely flat
    flat = data_driven_plateau(x_f, h_f)
    assert flat["bore_valid"] is True
    assert flat["flatness"] < PLATEAU_FLAT_FRACTION

    # same jump, but the plateau is a linear ramp carrying 10 % of the jump:
    # the amplitude estimate stays usable, the "flat plateau" premise does not.
    x, h, _ = _analytic_profile(1600, t)
    x_lo, x_hi = -1.0, 1.8
    band = (x > x_lo) & (x < x_hi)
    h = h.copy()
    h[band] = h[band] + 0.1 * (h0_jump := (sol["h2"] - H1)) * (x[band] - x_lo) / (x_hi - x_lo)
    ramped = data_driven_plateau(x, h)
    assert ramped["bore_method"] == "mid_level_crossing"
    assert ramped["flatness"] > flat["flatness"]
    assert ramped["bore_valid"] is False
    assert abs(ramped["h2"] - sol["h2"]) / sol["h2"] < 5e-2               # still usable


def test_degenerate_profiles_are_undefined_not_wrong():
    x = np.linspace(-5.0, 5.0, 400, endpoint=False) + 0.0125
    assert data_driven_plateau(x, np.zeros_like(x))["bore_method"] == "undefined"
    assert data_driven_plateau(x[:3], np.zeros(3))["bore_method"] == "undefined"
    # constant profile: no jump at all -> no plateau may be quoted
    flat = data_driven_plateau(x, np.full_like(x, 0.7))
    assert flat["bore_method"] in ("undefined", "plateau_unbounded")
    assert math.isnan(flat["h2"])


def test_solver_n6400_wet_bore_subcell_is_finite_and_within_criteria():
    """The grid on which the pre-F9 detector read 15.5 % high (n = 6400)."""
    run = run_case(STOKER_WET, n=6400)
    dd = run["metrics"]["dd"]
    sol = stoker_constant_state(H0, H1, GRAVITY)
    dx = run["discretisation"]["dx"]
    assert dd["bore_method"] == "mid_level_crossing", dd["bore_method"]
    assert dd["bore_valid"] is True
    assert abs(dd["h2"] - sol["h2"]) / sol["h2"] < 5e-2
    assert abs(dd["step_ratio"] - sol["step_ratio"]) / sol["step_ratio"] < 5e-2
    cells_off = dd["bore_x_error"] * run["metrics"]["shock_exact"] / dx
    assert cells_off < 0.5, cells_off