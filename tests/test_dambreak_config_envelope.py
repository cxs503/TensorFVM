"""Tests for the numerical-configuration envelope of the dam-break pool entry.

Three things are pinned:

1. **no regression of the published numbers** -- the new ``dry_depth`` argument
   of :func:`run_case` defaults to the solver default, so every pool gate value
   is bit-identical to the entry's;
2. **the instruments answer calibration questions** -- the depth-floor
   diagnostic is compared with closed-form quadrature, and its validity domain
   (tip thinner than a cell) is pinned rather than assumed;
3. **the certificate logic is not a rubber stamp** -- the envelope extractor is
   exercised on synthetic rows, and the hard-failure detector on synthetic
   discretisation records, including the negative controls the study uses.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET, run_case
from tensorfvm.shallow_water.exact import ritter_state
from tensorfvm.verification.dambreak_config_envelope import (
    BASELINE,
    FLOOR_POINTS,
    analytic_tip_length,
    axis_envelope,
    floor_diagnostics,
    hard_failure,
    hidden_mass_instrument_error,
    run_point,
    sweep_cfl,
    sweep_floor,
)


# --------------------------------------------------------------------------- #
# 1. the published pool numbers are untouched
# --------------------------------------------------------------------------- #
def test_default_floor_is_the_solver_default_and_reproduces_the_pool_numbers():
    for case in (RITTER_DRY, STOKER_WET):
        default = run_case(case, n=400)["verdict"]
        explicit = run_case(case, n=400, dry_depth=1e-12)["verdict"]
        assert default == explicit
        assert run_case(case, n=400)["discretisation"]["dry_depth"] == 1e-12


def test_run_point_reports_gates_hard_flags_and_floor_reading():
    row = run_point(RITTER_DRY, n=400)
    assert row["ok"] is True
    assert row["hard"]["any"] is False
    assert row["gates"]["front_fan<5%"]["pass"] is True
    assert row["floor"]["dry_depth"] == pytest.approx(1e-12)
    assert row["floor"]["analytic_tip_length"] > 0.0
    # signed reading, not only the relative error
    assert row["reading"]["front_exact"] == pytest.approx(4.0, rel=1e-6)
    assert row["reading"]["front_recession"] == pytest.approx(
        row["reading"]["front_exact"] - row["reading"]["front_wetted"])


# --------------------------------------------------------------------------- #
# 2. instruments: the depth-floor diagnostic and its validity domain
# --------------------------------------------------------------------------- #
def test_analytic_tip_length_is_the_root_of_the_exact_profile():
    """``3 t sqrt(g eps)`` is the distance behind the front where h = eps."""
    t, h0, g, eps = 0.638551, 1.0, 9.81, 1e-3
    tip = analytic_tip_length(eps, t, h0, g)
    x_front = 2.0 * math.sqrt(g * h0) * t
    h_at_tip, _ = ritter_state(np.array([x_front - tip]), t, h0, g, 0.0)
    assert float(h_at_tip[0]) == pytest.approx(eps, rel=1e-9)
    # decreasing eps makes the strip thinner like sqrt(eps)
    assert analytic_tip_length(eps / 4.0, t, h0, g) == pytest.approx(tip / 2.0, rel=1e-12)
    assert analytic_tip_length(0.0, t, h0, g) == 0.0


def test_hidden_mass_instrument_is_calibrated_against_quadrature():
    """Validity domain: the reading is trustworthy once the tip spans cells."""
    coarse_tip = hidden_mass_instrument_error(1e-4, t=0.638551, h0=1.0, n=800)
    fine_tip = hidden_mass_instrument_error(1e-2, t=0.638551, h0=1.0, n=800)
    assert fine_tip["relative_difference"] < 1e-3
    assert coarse_tip["relative_difference"] > fine_tip["relative_difference"]
    # below one cell the discrete reading cannot resolve the strip at all
    subcell = hidden_mass_instrument_error(1e-6, t=0.638551, h0=1.0, n=800)
    assert subcell["analytic_tip_length"] < 1e-2  # < one cell at n = 800
    assert subcell["relative_difference"] > 0.5


def test_raising_the_floor_dries_more_of_the_wet_tip():
    run_lo = run_case(RITTER_DRY, n=400, dry_depth=1e-12)
    run_hi = run_case(RITTER_DRY, n=400, dry_depth=1e-3)
    lo = floor_diagnostics(RITTER_DRY, run_lo, 1e-12)
    hi = floor_diagnostics(RITTER_DRY, run_hi, 1e-3)
    assert hi["unjustified_dry_cells"] > lo["unjustified_dry_cells"]
    assert hi["hidden_mass_fraction"] > lo["hidden_mass_fraction"]
    # the recession is far larger than the geometric floor crossing
    assert hi["analytic_tip_length"] > 0.0
    assert hi["floored_length"] > hi["analytic_tip_length"]


def test_floor_reading_is_zero_for_the_wet_case_below_its_depth():
    run = run_case(STOKER_WET, n=400, dry_depth=1e-3)
    diag = floor_diagnostics(STOKER_WET, run, 1e-3)
    assert diag["floored_cells"] == 0
    assert diag["hidden_mass_fraction"] == 0.0


# --------------------------------------------------------------------------- #
# 3. hard-failure detector and envelope extraction
# --------------------------------------------------------------------------- #
def _gates(value: float, limit: float = 0.05) -> dict:
    return {"g<5%": {"value": value, "limit": limit, "pass": value <= limit}}


def test_hard_failure_detects_mass_leak_and_positivity_clipping():
    clean = {"volume_initial": 1.0, "volume_final": 1.0,
             "clipped_cells": 0, "dry_momentum_flips": 0}
    assert hard_failure(clean, _gates(1e-4))["any"] is False

    leaky = dict(clean, volume_final=1.0 + 1e-6)
    report = hard_failure(leaky, _gates(1e-4))
    assert report["mass_leak"] is True and report["any"] is True

    clipped = dict(clean, clipped_cells=3)
    assert hard_failure(clipped, _gates(1e-4))["positivity_clipping"] is True

    # an undefined gate value is a failure, not a pass
    undefined = _gates(1e-4)
    undefined["g<5%"]["value"] = float("nan")
    report = hard_failure(clean, undefined)
    assert report["nan_in_gates"] is True and report["any"] is True


def test_hard_failure_carries_the_exception():
    report = hard_failure({}, {}, error="FloatingPointError: boom")
    assert report["exception"] is True
    assert report["any"] is True
    assert "boom" in report["error"]


def test_run_point_turns_a_broken_configuration_into_data():
    row = run_point(RITTER_DRY, n=3)  # solver needs at least 4 cells
    assert row["ok"] is False
    assert row["hard"]["exception"] is True
    assert row["hard"]["any"] is True


def _row(param: float, value: float, *, hard: bool = False,
         parameter: str = "p") -> dict:
    return {parameter: param, "ok": True, "gates": _gates(value),
            "hard": {"any": hard}}


def test_axis_envelope_extracts_the_certified_band_and_the_binding_gate():
    rows = [_row(1.0, 0.01), _row(2.0, 0.03), _row(4.0, 0.06), _row(8.0, 0.09)]
    env = axis_envelope(rows, "p")
    assert env["certified"] == [1.0, 2.0]
    assert env["failed_points"] == [4.0, 8.0]
    assert env["all_points_certified"] is False
    assert env["binding"] == {"gate": "g<5%", "first_break": 4.0}
    assert env["gates"]["g<5%"]["worst_at"] == 8.0


def test_axis_envelope_treats_a_hard_failure_as_uncertified_even_if_gates_pass():
    rows = [_row(1.0, 0.01), _row(2.0, 0.01, hard=True), _row(3.0, 0.01)]
    env = axis_envelope(rows, "p")
    assert env["certified"] == [1.0, 3.0]
    assert env["failed_points"] == [2.0]
    assert env["hard_failures"][0]["p"] == 2.0


def test_axis_envelope_on_a_fully_green_axis_reports_no_binding_gate():
    env = axis_envelope([_row(0.1, 1e-4), _row(0.4, 1e-4)], "p")
    assert env["all_points_certified"] is True
    assert env["binding"] is None


# --------------------------------------------------------------------------- #
# 4. end-to-end behaviour at a cheap resolution
# --------------------------------------------------------------------------- #
def test_cfl_axis_is_positive_and_certified_at_both_ends():
    rows = sweep_cfl(RITTER_DRY, cfls=(0.2, 0.9), n=200)
    assert all(row["ok"] and not row["hard"]["any"] for row in rows)
    assert all(item["pass"] for row in rows for item in row["gates"].values())
    # the headline reading is not Δt-contaminated in the pooled band only: the
    # CFL 0.9 run is at least as sharp as the CFL 0.2 run (measured trend)
    assert rows[1]["gates"]["front_fan<5%"]["value"] <= rows[0]["gates"]["front_fan<5%"]["value"]


def test_floor_axis_front_wetted_reading_grows_with_the_floor():
    floors = (1e-15, 1e-12, 1e-6, 1e-2)
    rows = sweep_floor(RITTER_DRY, floors=floors, n=400)
    wetted = [row["gates"]["front_wetted<5%"]["value"] for row in rows]
    assert wetted == sorted(wetted)
    assert wetted[0] < 0.05 < wetted[-1]
    assert set(floors) == {row["dry_depth"] for row in rows}
    assert BASELINE["dry_depth"] in FLOOR_POINTS