"""Depth-distribution (oracle-free) wet-bed reading: acceptance and falsification.

``plateau_from_depth_modes`` reads the plateau depth, the still-water depth and
the bore position from the **distribution** of the numerical depth profile: each
constant state of the wet-bed dam break is a mode of that distribution, so no
analytic position, level or amplitude enters the reading (``AUDIT.md`` §6
residual, F4/§7.4 there).

The tests below are written to be able to *fail*:

* synthetic three-level profiles with levels that are **not** Stoker values show
  that the detector reads the profile, not the closed form;
* a plateau translated sideways leaves the amplitude reading untouched (it must,
  ``h2`` is an amplitude) while the position reading moves -- i.e. the two
  metrics are independent evidence and neither replaces the other (F4
  consequence B);
* a profile with a wrong plateau amplitude is caught by the amplitude metric
  while the position stays put (the other direction);
* a profile with **no** constant intermediate state, a constant profile, and
  length-mismatched input are *withheld* (``no_plateau``), never quoted;
* on the real solver the depth-mode reading agrees with the structurally
  different gradient walk (:func:`data_driven_plateau`) to within 1 % in
  amplitude and 0.05 m in position, and it respects the predeclared half-cell
  position criterion;
* the committed artifact ``depth_modes.json`` is reproducible from the code in
  this tree (no stale numbers).
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    ACCEPTED_MIN_N,
    BORE_SUBCELL_MAX_CELLS,
    STOKER_WET,
    data_driven_plateau,
    plateau_from_depth_modes,
)
from tensorfvm.shallow_water.exact import (
    stoker_constant_state,
    t_star_to_time,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D

ARTIFACT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "src", "tensorfvm", "verification", "artifacts", "shallow-water-dam-break")


def synthetic_profile(h0=1.0, h2=0.62, h1=0.3, plateau=(0.0, 1.2),
                      fan=(-1.2, 0.0), x_min=-4.0, x_max=6.0, n=1001,
                      smear=0.02):
    """Three-level profile: upstream ``h0``, rarefaction ramp, plateau ``h2``,
    bore of width ``smear`` metres, still water ``h1``.

    The levels are free parameters -- nothing here comes from the closed form.
    """
    x = np.linspace(x_min, x_max, n)
    h = np.full_like(x, float(h1))
    h[x < fan[0]] = h0
    in_fan = (x >= fan[0]) & (x < fan[1])
    h[in_fan] = h0 + (h2 - h0) * (x[in_fan] - fan[0]) / (fan[1] - fan[0])
    h[(x >= plateau[0]) & (x <= plateau[1])] = h2
    bore = (x > plateau[1]) & (x <= plateau[1] + smear)
    frac = (x[bore] - plateau[1]) / smear
    h[bore] = h2 + (h1 - h2) * frac
    return x, h


def _solver_profile(n: int):
    t_end = t_star_to_time(STOKER_WET.t_star_end, STOKER_WET.h0,
                           STOKER_WET.gravity)
    solver = ShallowWater1D(n, STOKER_WET.x_min, STOKER_WET.x_max,
                            gravity=STOKER_WET.gravity, cfl=0.4, order=2)
    solver.set_state(np.where(solver.x < STOKER_WET.x_dam,
                              STOKER_WET.h0, STOKER_WET.h1))
    solver.advance(t_end)
    return solver.x.copy(), solver.h.copy(), solver.dx, solver.time


def test_synthetic_levels_that_are_not_stoker_values_are_recovered():
    """The reading comes from the profile: arbitrary levels are recovered."""
    x, h = synthetic_profile(h0=1.0, h2=0.62, h1=0.3)
    read = plateau_from_depth_modes(x, h)
    assert read["valid"] is True
    assert read["method"] == "depth_mode"
    assert read["n_modes"] == 3
    assert read["h2"] == pytest.approx(0.62, rel=5e-3)
    assert read["h1"] == pytest.approx(0.30, rel=5e-3)
    assert read["h0"] == pytest.approx(1.00, rel=5e-3)
    # bore centre = right edge of the plateau + half the smeared jump
    assert read["bore_x"] == pytest.approx(1.21, abs=0.02)
    # plateau extent: the cells on h2 between fan end and bore
    assert read["plateau_cells"] > 100
    # the band tolerance (5 % of the jump) also admits the ramp cell at the
    # fan end and the first smeared cell, so the run is ~5 cells wider
    assert read["plateau_width"] == pytest.approx(1.2, abs=0.1)


def test_translated_plateau_is_invisible_to_amplitude_and_caught_by_position():
    """Shift the whole constant state sideways: ``h2`` cannot see it, the bore
    position must.  Both metrics are needed; neither replaces the other."""
    x, h = synthetic_profile(plateau=(0.0, 1.2), fan=(-1.2, 0.0))
    moved, h_moved = synthetic_profile(plateau=(0.4, 1.6), fan=(-0.8, 0.4))
    keep = plateau_from_depth_modes(x, h)
    shift = plateau_from_depth_modes(moved, h_moved)
    # amplitude reading: blind to the translation, by construction
    assert shift["h2"] == pytest.approx(keep["h2"], rel=1e-6)
    assert abs(shift["h2"] - 0.62) / 0.62 < 5e-3
    # position reading: moves with the plateau, 0.4 m = 30 % of the bore abscissa
    delta = shift["bore_x"] - keep["bore_x"]
    assert delta == pytest.approx(0.4, abs=0.02)
    assert abs(delta) / keep["bore_x"] > 0.2


def test_wrong_plateau_amplitude_is_caught_by_the_amplitude_metric():
    """Complementary direction: +5 % on the plateau level, position unchanged."""
    x, h = synthetic_profile(h2=0.62)
    x_bad, h_bad = synthetic_profile(h2=0.62 * 1.05)
    good = plateau_from_depth_modes(x, h)
    bad = plateau_from_depth_modes(x_bad, h_bad)
    assert abs(bad["h2"] - 0.62) / 0.62 > 0.04        # amplitude metric fails
    assert abs(bad["step_ratio"] - 0.62 / 0.30) / (0.62 / 0.30) > 0.03
    assert bad["bore_x"] == pytest.approx(good["bore_x"], abs=0.05)


def test_profile_without_intermediate_constant_state_is_withheld():
    """A pure two-level ramp has no interior mode -> no plateau may be quoted."""
    x = np.linspace(-4.0, 6.0, 1001)
    h = np.where(x < -1.0, 1.0, np.where(x > 1.0, 0.5,
                                         1.0 - 0.5 * (x + 1.0) / 2.0))
    read = plateau_from_depth_modes(x, h)
    assert read["valid"] is False
    assert read["method"] == "no_plateau"
    assert not np.isfinite(read["h2"])
    assert read["n_modes"] < 3


def test_constant_and_degenerate_profiles_are_withheld():
    x = np.linspace(-4.0, 6.0, 201)
    for h in (np.full(x.size, 0.5), np.zeros(x.size), np.full(x.size, np.nan)):
        read = plateau_from_depth_modes(x, h)
        assert read["valid"] is False
        assert read["method"] in ("undefined", "no_plateau")
        assert not np.isfinite(read["h2"])
    # length mismatch must not be silently broadcast either
    read = plateau_from_depth_modes(x[:10], np.full(11, 0.5))
    assert read["valid"] is False and read["method"] == "undefined"


def test_solver_reading_respects_half_cell_position_criterion():
    """n >= 800: bore position within half a cell (``BORE_SUBCELL_MAX_CELLS``)."""
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1,
                                STOKER_WET.gravity)
    for n in (ACCEPTED_MIN_N, 2 * ACCEPTED_MIN_N):
        x, h, dx, t = _solver_profile(n)
        read = plateau_from_depth_modes(x, h)
        assert read["valid"] is True and read["method"] == "depth_mode"
        x_shock = STOKER_WET.x_dam + sol["shock_speed"] * t
        assert abs(read["bore_x"] - x_shock) / dx <= BORE_SUBCELL_MAX_CELLS
        assert abs(read["h2"] - sol["h2"]) / sol["h2"] < 1e-2
        assert abs(read["h1"] - sol["h1"]) / sol["h1"] < 1e-6


def test_depth_mode_agrees_with_the_structurally_different_gradient_walk():
    """Two unrelated mechanisms on the same state: amplitude within 1 %,
    position within 0.05 m.  Agreement here is cross-mechanism evidence, not a
    tautology (the walk retries a gradient cut, the modes do not look at dx)."""
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1,
                                STOKER_WET.gravity)
    x, h, dx, t = _solver_profile(ACCEPTED_MIN_N)
    mode = plateau_from_depth_modes(x, h)
    walk = data_driven_plateau(x, h)
    assert walk["bore_method"] == "mid_level_crossing"
    assert abs(mode["h2"] - walk["h2"]) / sol["h2"] < 1e-2
    assert abs(mode["bore_x"] - walk["bore_x"]) < 0.05
    # and the plateau extent recovered from the distribution is not degenerate
    assert mode["plateau_width"] > 0.5 * (walk["plateau_width"])


def test_committed_artifact_is_reproducible_from_this_tree():
    path = os.path.join(ARTIFACT, "depth_modes.json")
    with open(path, encoding="utf-8") as fh:
        study = json.load(fh)
    assert study["case"]["key"] == "stoker-wet"
    rows = {row["n"]: row for row in study["rows"]}
    assert set(rows) >= {200, 1600}
    for n in (200, 1600):
        x, h, dx, t = _solver_profile(n)
        read = plateau_from_depth_modes(x, h)
        row = rows[n]["depth_mode"]
        assert read["valid"] is True
        assert read["h2"] == pytest.approx(row["h2"], rel=1e-9)
        assert read["bore_x"] == pytest.approx(row["bore_x"], rel=1e-9)
        assert read["plateau_cells"] == row["plateau_cells"]
    # the published worst-case numbers on the accepted grid stay inside 5 %
    worst = study["max_error_depth_mode_accepted"]
    assert worst["h2_error"] < study["tolerance"] / 50.0
    assert worst["bore_x_error"] < study["tolerance"]
    assert study["accepted_min_n"] == ACCEPTED_MIN_N