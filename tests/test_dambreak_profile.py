"""Tests of the dam-break error-profile decomposition.

Two classes of assertions:

* **synthetic** -- the estimators recover a mechanism that was injected on
  purpose (a known delay, a known slope deficit, a known crossing time);
* **artefact-driven honesty guards** -- run against the committed
  ``docs/GAP-FVM-TWOPHASE/artifacts/leg_dambreak_*.json`` and assert that the
  new diagnostics (a) reproduce the graded values rather than a re-tuned
  variant, (b) confirm the failure survives in the developed phase, and
  (c) leave the graded gate constants untouched.  A future "fix" that quietly
  relaxes ``FRONT_ERROR_MAX`` or the window breaks these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.vof_phase0.anchors import (
    MARTIN_MOYCE_FRONT_T,
    MARTIN_MOYCE_FRONT_X,
)
from tensorfvm.twophase_gap import campaign, dambreak_gate
from tensorfvm.twophase_gap.dambreak_profile import (
    DEVELOPED_MIN_T,
    SEGMENTS,
    build_profile,
    developed_phase_stats,
    front_speed_ratio,
    load_legs,
    profile_leg,
    rel_error,
    segment_profile,
    startup_lag,
    time_to_front_star,
)

ARTIFACTS = Path(__file__).resolve().parents[1] / "docs" / "GAP-FVM-TWOPHASE" / "artifacts"


def _ref(t: np.ndarray) -> np.ndarray:
    return np.interp(t, MARTIN_MOYCE_FRONT_T, MARTIN_MOYCE_FRONT_X)


def _synthetic(delay: float = 0.0, scale: float = 1.0, n: int = 400,
               t_max: float = 4.0) -> list[list[float]]:
    """Rows whose front is ``scale * ref(max(t - delay, 0))`` and whose height
    follows the reference curve.  Used to inject a known mechanism."""
    t = np.linspace(0.31, t_max, n)
    front = scale * _ref(np.maximum(t - delay, 0.0))
    ref = _ref(t)
    height = np.ones_like(t)
    return [[float(a), float(b), float(b), float(c), float(d)]
            for a, b, c, d in zip(t, front, ref, height)]


# --------------------------------------------------------------------------- #
# synthetic estimator recovery
# --------------------------------------------------------------------------- #
def test_rel_error_matches_definition():
    got = np.array([0.0, 1.0, 2.0])
    ref = np.array([0.0, 2.0, 1.0])
    e = rel_error(got, ref)
    # ref == 0 is floored, so the first sample is 0/1e-6 -> 0 only because got==0.
    assert e[1] == pytest.approx(0.5)
    assert e[2] == pytest.approx(1.0)
    # A zero reference with a non-zero numerator must not raise or return inf.
    assert np.isfinite(rel_error(np.array([1e-3]), np.array([0.0]))).all()


def test_startup_lag_recovers_injected_delay():
    rows = _synthetic(delay=0.5)
    fit = startup_lag(rows)
    assert fit["shift"]["best_t_star_delay"] == pytest.approx(0.5, abs=0.03)
    assert fit["shift_wins"] is True
    assert fit["shift"]["max_rel_error"] < 0.02


def test_startup_lag_recovers_injected_time_scaling():
    # got(t) = ref(t / 1.6): a *slowed* solution, which a pure delay cannot mimic
    # as well once the curve is strongly curved.
    t = np.linspace(0.31, 4.0, 400)
    rows = [[float(a), float(_ref(a / 1.6)), float(_ref(a / 1.6)),
             float(_ref(a)), 1.0] for a in t]
    fit = startup_lag(rows)
    assert fit["timescale"]["best_alpha"] == pytest.approx(1.6, abs=0.05)


def test_front_speed_ratio_detects_slope_deficit():
    t = np.linspace(0.4, 4.0, 200)
    rows = [[float(a), float(0.5 * _ref(a)), float(0.5 * _ref(a)),
             float(_ref(a)), 1.0] for a in t]
    speed = front_speed_ratio(rows, t_lo=1.0, t_hi=4.0)
    assert speed["ratio"] == pytest.approx(0.5, rel=1e-6)


def test_time_to_front_star_interpolates():
    t = np.array([0.5, 1.0, 1.5, 2.0])
    front = np.array([0.4, 0.8, 1.2, 1.6])
    rows = [[float(a), float(b), float(b), float(b), 1.0]
            for a, b in zip(t, front)]
    out = time_to_front_star(rows, target=1.0)
    assert out["front_half"] == pytest.approx(1.25)
    assert out["delay_t_star"] == pytest.approx(0.0)


def test_segment_profile_boundaries_are_half_open_left_closed_right():
    # A single sample at exactly t* = 0.5 must land in the (0.3, 0.5] segment,
    # matching the gate's ``t* > WINDOW[0]`` convention.
    # Each boundary time must be counted in the segment that *ends* on it.
    rows = [[t, 1.0, 1.0, 1.0, 1.0]
            for t in (0.5, 0.51, 1.0, 1.0001, 2.0, 2.0001)]
    segs = segment_profile(rows)
    counts = {(s["t_from"], s["t_to"]): s["n_samples"] for s in segs}
    assert counts[(0.3, 0.5)] == 1          # t*=0.5 closes segment 1
    assert counts[(0.5, 1.0)] == 2          # t*=0.51 and t*=1.0
    assert counts[(1.0, 2.0)] == 2          # t*=1.0001 and t*=2.0 (closed right end)
    assert counts[(2.0, 4.0)] == 1          # only t*=2.0001, not t*=2.0


def test_segment_profile_reports_time_of_maximum():
    rows = [[0.4, 0.0, 0.0, 1.0, 1.0], [0.45, 0.5, 0.5, 1.0, 1.0]]
    seg = segment_profile(rows)[0]
    assert seg["front_max_rel_error"] == pytest.approx(1.0)
    assert seg["front_max_at_t"] == pytest.approx(0.4)


# --------------------------------------------------------------------------- #
# honesty guards on the committed artefacts
# --------------------------------------------------------------------------- #
requires_artifacts = pytest.mark.skipif(
    not ARTIFACTS.exists() or not list(ARTIFACTS.glob("leg_dambreak_*.json")),
    reason="dam-break leg artefacts not present")


def test_graded_gate_constants_unchanged():
    """The decomposition must not have moved the goalposts."""
    assert campaign.FRONT_ERROR_MAX == pytest.approx(0.15)
    assert campaign.HEIGHT_ERROR_MAX == pytest.approx(0.15)
    assert dambreak_gate.FRONT_ERROR_MAX == pytest.approx(0.15)
    assert dambreak_gate.WINDOW == (0.3, 8.0)
    assert DEVELOPED_MIN_T > SEGMENTS[0][0]
    # the gates that grade a leg still look up the un-decomposed scalar
    keys = {row[2] for row in campaign.GATES}
    assert "front_max_rel_error" in keys
    assert "height_max_rel_error" in keys


@requires_artifacts
def test_profile_reproduces_graded_error_values():
    legs = load_legs(ARTIFACTS)
    assert legs, "expected at least one dam-break leg"
    for name, leg in legs.items():
        prof = profile_leg(leg)
        # the profile's developed-phase max is bounded by the graded max ...
        assert prof["developed"]["front_max_rel_error"] <= (
            prof["gated_front_max_rel_error"] + 1e-12), name
        # ... and the graded value is reproduced from the rows, not re-typed.
        rows = np.asarray(leg["comparison"], dtype=float)
        recomputed = rel_error(rows[:, 1], rows[:, 3]).max()
        assert recomputed == pytest.approx(leg["front_max_rel_error"], rel=1e-9), name


@requires_artifacts
def test_error_peaks_in_the_startup_segment():
    prof = build_profile(load_legs(ARTIFACTS))
    assert prof["verdicts"]["error_peaks_at_startup"] is True
    for name, leg in prof["legs"].items():
        values = [s["front_max_rel_error"] for s in leg["segments"]]
        assert values == sorted(values, reverse=True), name
        assert leg["segments"][0]["front_max_at_t"] < SEGMENTS[0][1] + 1e-9, name


@requires_artifacts
def test_developed_phase_still_fails_the_gate():
    """Honest-FAIL guard: the failure is not an artefact of the startup window."""
    prof = build_profile(load_legs(ARTIFACTS))
    assert prof["verdicts"]["developed_phase_still_exceeds_gate"] is True
    assert prof["verdicts"]["developed_phase_still_exceeds_height_gate"] is True
    for name, leg in prof["legs"].items():
        assert leg["developed"]["front_max_rel_error"] > campaign.FRONT_ERROR_MAX, name


@requires_artifacts
def test_height_error_peaks_mid_transient_not_at_startup():
    """The opposite trend to the front: D2's maximum is *not* a startup artefact.

    Front error decays in time, column-height error peaks around t* ~ 2 (the
    drainage lags), so a single "everything is a startup artefact" story would be
    wrong and must not be written into the report.
    """
    prof = build_profile(load_legs(ARTIFACTS))
    assert prof["verdicts"]["height_error_peaks_mid_transient"] is True
    for name, leg in prof["legs"].items():
        assert leg["height_peak_segment_index"] >= 1, name
        # front peak is always in the first segment, height peak never is
        fronts = [s["front_max_rel_error"] for s in leg["segments"]]
        heights = [s["height_max_rel_error"] for s in leg["segments"]]
        assert int(np.argmax(fronts)) == 0, name
        assert int(np.argmax(heights)) > 0, name


@requires_artifacts
def test_committed_profile_json_matches_recomputation():
    """The committed JSON must be a faithful product of the committed legs."""
    path = ARTIFACTS / "dambreak_profile.json"
    if not path.exists():
        pytest.skip("dambreak_profile.json not generated yet")
    with path.open() as fh:
        stored = json.load(fh)
    fresh = build_profile(load_legs(ARTIFACTS))
    assert set(stored["legs"]) == set(fresh["legs"])
    for name in fresh["legs"]:
        assert (stored["legs"][name]["gated_front_max_rel_error"]
                == pytest.approx(fresh["legs"][name]["gated_front_max_rel_error"],
                                 rel=1e-12))
        assert (stored["legs"][name]["front_speed"]["ratio"]
                == pytest.approx(fresh["legs"][name]["front_speed"]["ratio"],
                                 rel=1e-12))


@requires_artifacts
def test_developed_front_speed_is_not_the_deficit():
    """The mechanism label the report relies on must hold on the real legs."""
    prof = build_profile(load_legs(ARTIFACTS))
    for name, leg in prof["legs"].items():
        ratio = leg["front_speed"]["ratio"]
        assert 0.9 <= ratio <= 1.25, (name, ratio)
    assert prof["verdicts"]["mechanism"].startswith("delayed collapse")