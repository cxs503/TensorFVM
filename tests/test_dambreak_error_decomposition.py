"""Tests of the dam-break D1/D2 *diagnostic* decomposition.

The gates themselves are covered by ``tests/test_twophase_gap.py`` and are not
touched here: these tests pin the invariant that the diagnostics (a) recover a
known phase lag / rate deficit / time-scale error from synthetic curves, (b)
never change a gate verdict, and (c) report the cell-quantisation floor that a
real leg actually has.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.twophase_gap import (
    evaluate_gates,
    quantisation_floor,
    run_dambreak_leg,
)
from tensorfvm.twophase_gap.dambreak_gate import (
    decompose_dambreak_error,
    resolvable_error_floor,
)


def _curves(shift=0.0, scale=1.0, gain=1.0):
    """A monotone reference and a curve that is shifted/scaled/gained."""
    t = np.linspace(0.0, 4.0, 161)
    ref = 1.0 + 0.8 * t
    got = gain * (1.0 + 0.8 * (t / scale - shift))
    return t, got, ref


def test_synthetic_phase_lag_is_recovered():
    t, got, ref = _curves(shift=0.5)
    d = decompose_dambreak_error(t, got, ref)
    assert d["phase_lag_t_star"] == pytest.approx(0.5, abs=0.03)
    assert d["speed_ratio"] == pytest.approx(1.0, abs=0.02)
    assert d["verdict"] == "phase_lag"
    # the shift must explain essentially all of the misfit
    assert d["rms_at_best_shift"] < 0.05 * d["rms_at_zero_shift"]


def test_synthetic_rate_deficit_is_flagged():
    t, got, ref = _curves(gain=0.5)
    d = decompose_dambreak_error(t, got, ref)
    assert d["speed_ratio"] == pytest.approx(0.5, abs=0.02)
    assert d["verdict"] == "rate_deficit"


def test_synthetic_time_scale_error_is_detectable():
    """A sqrt(2) normalisation clash must be absorbed by a *scale*, not a shift.

    The diagnostic reports ``s`` such that ``ref(t / s)`` matches ``got(t)``; a
    curve running at ``1/sqrt(2)`` of the reference speed is therefore recovered
    as ``s = 1/sqrt(2)``.
    """
    t, got, ref = _curves(scale=1.0 / np.sqrt(2.0))
    d = decompose_dambreak_error(t, got, ref)
    assert d["time_scale_zero_shift"] == pytest.approx(1.0 / np.sqrt(2.0),
                                                       abs=0.02)
    t, got, ref = _curves()
    d = decompose_dambreak_error(t, got, ref)
    assert d["time_scale_zero_shift"] == pytest.approx(1.0, abs=0.02)


def test_degeneracies_are_reported_not_raised():
    t = np.linspace(0.0, 4.0, 21)
    d = decompose_dambreak_error(t, np.full(21, np.nan), t)
    assert d["verdict"] == "unavailable"
    assert d["reason"]
    d = decompose_dambreak_error(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                                 np.array([0.0, 1.0]))
    assert d["verdict"] == "unavailable"


def test_resolvable_error_floor_is_quantum_over_reference():
    t = np.array([0.3, 1.0, 3.0])
    ref = np.array([1.0, 0.5, 0.1])
    out = resolvable_error_floor(t, ref, quantum=0.05, gate=0.15)
    assert out["floor_at_window_start"] == pytest.approx(0.05)
    assert out["floor_max"] == pytest.approx(0.5)
    # only the last sample has a floor above the 15 % gate
    assert out["violating_fraction"] == pytest.approx(1.0 / 3.0)
    assert out["samples"] == 3


def test_quantisation_floor_counts_gate_levels():
    out = quantisation_floor(dx=0.005, dy=0.0025, a=0.05715, h0=0.1143,
                             gate=0.15, ref_mean=2.0)
    assert out["height_quantum_rel"] == pytest.approx(0.0025 / 0.1143)
    assert out["height_gate_levels"] == pytest.approx(0.15 * 0.1143 / 0.0025)
    assert out["front_quantum_star"] == pytest.approx(0.005 / 0.05715)
    assert out["front_gate_levels"] == pytest.approx(
        0.15 * 2.0 / (0.005 / 0.05715))


@pytest.fixture(scope="module")
def coarse_leg():
    # t* = 3 keeps at least three samples above LATE_T_STAR so the
    # slope-based part of the diagnostic is defined
    return run_dambreak_leg(48, 16, t_star_end=3.0, control_steps=2)


def test_real_leg_carries_diagnostics_and_keeps_the_honest_fail(coarse_leg):
    rec = coarse_leg["record"]
    diag = rec["diagnostics"]
    assert set(diag) >= {"front", "height", "quantisation", "resolvable"}
    # the cell-quantisation floor of the height observable is exactly dy/h0:
    # cell_size_ratio_dy_over_a = dy/a and the anchor column has h0 = 2 a,
    # so the height quantum in h* units is exactly half that ratio
    assert diag["quantisation"]["height_quantum_rel"] == pytest.approx(
        rec["cell_size_ratio_dy_over_a"] / 2.0, rel=1e-9)
    # ... and the recorded height history lives on the quantum lattice: whole
    # multiples of dy/h0 (levels may be skipped, never subdivided)
    h = np.asarray(coarse_leg["history"]["height_star"], dtype=float)
    step = diag["quantisation"]["height_quantum_rel"]
    # the observable is a cell-centre height, so the lattice is offset by half a
    # quantum and consecutive readings differ by exactly one quantum
    q = h / step
    assert np.allclose(np.round(q - 0.5) + 0.5, q, atol=1e-6)
    jumps = np.diff(np.unique(np.round(q, 9)))
    assert jumps.size > 0 and np.allclose(jumps, 1.0, atol=1e-6)
    # the gate itself is unchanged: this grid still fails D1/D2 honestly
    assert rec["front_max_rel_error"] > 0.15
    assert diag["front"]["verdict"] in {"phase_lag", "rate_deficit", "mixed"}


def test_diagnostics_do_not_change_gate_verdicts(coarse_leg):
    rec = dict(coarse_leg["record"])
    with_diag = dict(rec, grid=[48, 16])
    without = {k: v for k, v in with_diag.items() if k != "diagnostics"}
    without["grid"] = [48, 16]
    g1, m1 = evaluate_gates({"dambreak_48x16": with_diag})
    g2, m2 = evaluate_gates({"dambreak_48x16": without})
    assert [(g["id"], g["passed"]) for g in g1] == [
        (g["id"], g["passed"]) for g in g2]
    assert m1["dambreak_front_error"] == pytest.approx(
        m2["dambreak_front_error"])
    assert m1["dambreak_height_error"] == pytest.approx(
        m2["dambreak_height_error"])
