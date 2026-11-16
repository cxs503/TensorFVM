"""Oracle-free grading of the wet-bed ledger quantities (AUDIT.md F4 / §9.5).

``docs/case-fvm-dambreak-stoker/AUDIT.md`` recorded two things that are graded
here:

* **F4 residual (§6)**: the graded wet-bed quantities are read inside a window
  placed from the closed-form ``x_fan_end``/``x_shock`` and at the analytic mid
  level ``(h1 + h2) / 2`` -- the *measurement* contains the answer, so a pass is
  "near tautological".
* **§9.5 open decision**: an oracle-free reading
  (:func:`tensorfvm.shallow_water.dam_break.plateau_from_depth_modes`) had been
  implemented and cross-checked, but deliberately **not** wired into the verdict
  ("是否升级为 headline 判据留给下一次独立复核决定").

These tests are the outcome of that review: the reading is now graded as two new
keys (``bore_position_modes<5%`` / ``step_ratio_modes<5%``), every pre-existing
key keeps its meaning, and the tests pin

1. the *independence* claim -- the two measurements of the same quantity agree to
   well below any threshold, so the oracle dependency had no numerical effect;
2. the *gate* claim -- both new keys pass at the accepted grid and are graded;
3. the *withholding* claim -- a profile whose depth distribution has lost its
   interior mode fails as *insufficient evidence* instead of being quoted;
4. the *sensitivity* claim -- shifting the plateau moves the new position
   reading by > 20 % while the amplitude keys stay blind (< 1e-6), reproducing
   the AUDIT §9.4 can-fail probe through the verdict entry point.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    ACCEPTED_MIN_N,
    STOKER_WET,
    _oracle_free_verdict_entries,
    _oracle_free_wet_readings,
    plateau_from_depth_modes,
    run_case,
)
from tensorfvm.shallow_water.exact import stoker_constant_state
from tensorfvm.verification.case_pool_dambreak import PRIMARY_CRITERIA

T_STAR = STOKER_WET.t_star_end


def _t_end() -> float:
    return T_STAR * math.sqrt(STOKER_WET.h0 / STOKER_WET.gravity)


class _Profile:
    """Minimal solver stand-in: the readings only need ``x``, ``h``, ``time``."""

    def __init__(self, x, h, time):
        self.x = np.asarray(x, dtype=float)
        self.h = np.asarray(h, dtype=float)
        self.time = float(time)


@pytest.fixture(scope="module")
def accepted_run():
    return run_case(STOKER_WET, n=ACCEPTED_MIN_N)


# --------------------------------------------------------------------------- #
# 1. independence
# --------------------------------------------------------------------------- #
def test_oracle_free_reading_matches_the_oracle_window_reading(accepted_run):
    """The oracle dependency (F4) had no numerical effect on the graded value.

    Two *different* measurements of the bore position -- one at a level taken
    from the analytic constants inside a window placed from the analytic
    positions, the other from the depth distribution alone -- agree to ~1e-9
    relative.  So the "near tautology" is structurally real but numerically
    irrelevant: the pass was not manufactured by the window.
    """
    metrics = accepted_run["metrics"]
    modes = metrics["modes"]
    assert modes["valid"] is True
    assert modes["method"] == "depth_mode"
    assert modes["n_modes"] == 3
    assert modes["agreement_bore_vs_oracle_window"] < 1e-6
    # and each agrees with the analytic reference at the same level (the two
    # readings differ by < 1e-6 relative, i.e. far below their own resolution)
    assert modes["bore_position_error"] < 1e-3
    assert metrics["shock_error"] < 1e-3
    assert abs(modes["bore_position_error"] - metrics["shock_error"]) < 1e-6


def test_oracle_free_reading_recovers_the_still_water_state(accepted_run):
    """``h1`` must come back as the undisturbed depth *without being told it*."""

    modes = accepted_run["metrics"]["modes"]
    assert modes["h1"] == pytest.approx(STOKER_WET.h1, rel=0, abs=1e-12)
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    assert modes["h2"] == pytest.approx(sol["h2"], rel=1e-4)
    assert modes["step_ratio"] == pytest.approx(sol["step_ratio"], rel=1e-4)


# --------------------------------------------------------------------------- #
# 2. the gate
# --------------------------------------------------------------------------- #
def test_new_keys_are_graded_and_pass_at_the_accepted_grid(accepted_run):
    verdict = accepted_run["verdict"]
    for key in ("bore_position_modes<5%", "step_ratio_modes<5%"):
        entry = verdict[key]
        assert entry["measured"] is True
        assert entry["method"] == "depth_mode"
        assert entry["pass"] is True
        assert entry["value"] is not None and entry["value"] <= entry["limit"]
    assert verdict["all_pass"] is True


def test_new_keys_are_primary_in_the_pool_entry():
    """A graded key that is not primary would not be visible to the case pool."""

    primary = PRIMARY_CRITERIA["stoker-wet"]
    assert "bore_position_modes<5%" in primary
    assert "step_ratio_modes<5%" in primary
    assert "depth distribution only" in primary["bore_position_modes<5%"]


def test_every_grid_in_the_series_measures_the_still_water_step(accepted_run):
    """The detector is not tuned per grid: it must read every resolution."""

    for n in (100, 200):
        modes = run_case(STOKER_WET, n=n)["metrics"]["modes"]
        assert modes["valid"] is True, n
        assert modes["method"] == "depth_mode", n
        assert modes["bore_position_error"] < 0.05, n
        assert modes["step_ratio_error"] < 0.05, n


# --------------------------------------------------------------------------- #
# 3. withholding: insufficient evidence must fail, never be quoted
# --------------------------------------------------------------------------- #
def test_profile_without_interior_mode_is_withheld_and_fails():
    """Amplitude-preserving structure destroyer.

    The plateau is replaced by a linear ramp whose *middle-half mean is still
    exactly* ``h2`` (the quantity ``plateau_h2<5%`` grades) and whose mid-level
    crossing is still at the analytic bore; only the *constant state* is gone.
    An amplitude read off a window mean cannot see that, so it must not be able
    to pass this shape.
    """
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    t = _t_end()
    n = 1600
    x = np.linspace(STOKER_WET.x_min, STOKER_WET.x_max, n)
    dx = float(x[1] - x[0])
    x_fan_end = STOKER_WET.x_dam + (sol["u2"] - sol["c2"]) * t
    x_shock = STOKER_WET.x_dam + sol["shock_speed"] * t
    h = np.full_like(x, STOKER_WET.h1)
    h[x < x_fan_end] = STOKER_WET.h0
    ramp = (x >= x_fan_end) & (x <= x_shock)
    # linear ramp h1 -> (2 h2 - h1): its mean over any centred window is h2
    h[ramp] = STOKER_WET.h1 + (sol["h2"] - STOKER_WET.h1) * (
        (x[ramp] - x_fan_end) / (x_shock - x_fan_end)) * 2.0

    # old graded amplitude quantity, read the way _wet_metrics reads it
    lo = x_fan_end + 0.25 * (x_shock - x_fan_end)
    hi = x_fan_end + 0.75 * (x_shock - x_fan_end)
    window = (x >= lo) & (x <= hi)
    h2_window_mean = float(h[window].mean())
    amplitude_error = abs(h2_window_mean - sol["h2"]) / sol["h2"]
    # measured 1.1e-4 relative: >= 46x inside the 5 % gate, so the old amplitude
    # keys pass this profile and cannot see the destroyed constant state
    assert amplitude_error < 5e-3, (
        "probe construction: the window mean must be amplitude-exact")

    reading = _oracle_free_wet_readings(STOKER_WET, _Profile(x, h, t))
    assert reading["n_modes"] < 3
    assert reading["valid"] is False
    assert reading["method"] == "no_plateau"
    assert not math.isfinite(reading["bore_x"])

    entries = _oracle_free_verdict_entries({"modes": reading})
    for key, entry in entries.items():
        assert entry["pass"] is False, key
        assert entry["value"] is None, key
        assert entry["measured"] is False, key

    # measured (not assumed): what the *pre-existing* position key reads on the
    # same profile, so the report can state the residual detection power honestly
    from tensorfvm.shallow_water.dam_break import _level_crossing
    oracle_x = _level_crossing(x, h, 0.5 * (STOKER_WET.h1 + sol["h2"]))
    assert abs(oracle_x - x_shock) < dx  # the ramp preserves the mid-level point


def test_two_level_profile_is_not_quoted():
    """A dry-bed-like profile has no interior state: reading withholds."""

    x = np.linspace(-4.0, 6.0, 400)
    h = np.where(x < 0.0, 1.0, 0.0)
    out = plateau_from_depth_modes(x, h)
    assert out["method"] == "no_plateau"
    assert out["valid"] is False
    assert not math.isfinite(out["h2"])


# --------------------------------------------------------------------------- #
# 4. sensitivity: the new position key moves when the plateau moves
# --------------------------------------------------------------------------- #
def test_shifted_plateau_moves_the_new_position_key_but_not_the_amplitudes():
    """AUDIT §9.4 can-fail probe, now through the verdict entry point."""

    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    t = _t_end()
    n = 1600
    x = np.linspace(STOKER_WET.x_min, STOKER_WET.x_max, n)
    x_fan_end = STOKER_WET.x_dam + (sol["u2"] - sol["c2"]) * t
    x_shock = STOKER_WET.x_dam + sol["shock_speed"] * t
    shift = 0.4
    h = np.full_like(x, STOKER_WET.h1)
    # exact solution shapes, translated: the bore is moved by `shift` and the
    # plateau follows it, so every *amplitude* is untouched
    h[x < x_fan_end + shift] = STOKER_WET.h0
    plateau = (x >= x_fan_end + shift) & (x < x_shock + shift)
    h[plateau] = sol["h2"]

    base = _oracle_free_wet_readings(
        STOKER_WET, _Profile(x, h - 0.0, t))
    assert base["valid"] is True
    moved = abs(base["bore_x"] - x_shock)
    assert moved / x_shock > 0.2  # > 20 % as recorded in AUDIT §9.4
    assert base["step_ratio_error"] < 1e-9  # amplitude blind, as recorded
    entries = _oracle_free_verdict_entries({"modes": base})
    assert entries["bore_position_modes<5%"]["pass"] is False
    assert entries["step_ratio_modes<5%"]["pass"] is True