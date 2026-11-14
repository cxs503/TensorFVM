"""Oracle-free **position** evidence for the dry-bed dam break.

Why this file exists
--------------------
``CASE-FVM-DAMBREAK-STOKER`` is accepted on a *position* statement (``x_f(t)``
vs the analytic solution, ``< 5 %``), but the graded dry-bed headline
``front_fan`` is built on the exact similarity identity
``x_f = x_i + 3 t sqrt(g h_i)``: it certifies the fan **shape** and locates the
front from the closed form, so on its own it cannot show the profile sits at the
right place.  The remaining position-only definitions were the wetted edge (a
depth-floor detector, non-convergent) and ``level_scaled`` (which adds the exact
fan offset, i.e. imports the answer).

This file locks the missing evidence, from four sides:

1. the closed form used as reference is verified through an **independent code
   path** (``ritter_state``) rather than restated, and it refuses degenerate
   levels;
2. the *measurement* provably depends on the profile and the level only
   (translation equivariance + identity with ``_level_crossing``), so it cannot
   be a disguised restatement of ``x_f``;
3. the contour positions converge at first order and pass the 5 % gate on the
   accepted grids -- and the committed artefact is reproduced, so the published
   numbers cannot drift from the code;
4. the negative control stays visible: the wetted-edge error does **not**
   converge, which is why a contour metric was needed at all.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    LEVEL_POSITION_LEVELS,
    LEVEL_POSITION_SERIES_LEVEL,
    LEVEL_SERIES_MIN_CELLS,
    RITTER_DRY,
    TOLERANCE,
    _level_crossing,
    level_position_metrics,
    level_position_study,
)
from tensorfvm.shallow_water.exact import ritter_level_position, ritter_state
from tensorfvm.shallow_water.swe1d import ShallowWater1D

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                        "artifacts", "shallow-water-dam-break-level-position",
                        "study.json")

#: grids of this test: a subset of the published study, fast enough to run here
TEST_RESOLUTIONS = (100, 200, 400, 800)


@pytest.fixture(scope="module")
def study():
    return level_position_study(TEST_RESOLUTIONS)


def _solver(n: int) -> ShallowWater1D:
    solver = ShallowWater1D(n, RITTER_DRY.x_min, RITTER_DRY.x_max,
                            gravity=RITTER_DRY.gravity, cfl=0.4, order=2)
    solver.set_state(np.where(solver.x < RITTER_DRY.x_dam, RITTER_DRY.h0,
                              RITTER_DRY.h1))
    solver.advance(RITTER_DRY.t_end)
    return solver


# --------------------------------------------------------------------------
# 1. the reference closed form is verified, not restated
# --------------------------------------------------------------------------
def test_closed_form_matches_the_independent_ritter_state():
    """``h(x_level) = level * h0`` when evaluated through ``ritter_state``.

    ``ritter_level_position`` is algebra; ``ritter_state`` is a separate code
    path (piecewise similarity solution with the fan bounds).  Checking the
    former against the latter means the reference is not a transcription of the
    formula it is supposed to corroborate.
    """
    t = RITTER_DRY.t_end
    for level in (0.5, 0.25, 0.1, 0.04, 0.01, 0.75):
        x_level = ritter_level_position(level, t, RITTER_DRY.h0,
                                        RITTER_DRY.gravity, RITTER_DRY.x_dam)
        h, _ = ritter_state(np.array([x_level]), t, RITTER_DRY.h0,
                            RITTER_DRY.gravity, RITTER_DRY.x_dam)
        assert h[0] / RITTER_DRY.h0 == pytest.approx(level, rel=1e-12)


def test_closed_form_is_a_contour_and_not_the_tip():
    """The contour sits strictly behind the tip, and ahead of the fan head."""
    t = RITTER_DRY.t_end
    tip = RITTER_DRY.x_dam + 2.0 * t * math.sqrt(RITTER_DRY.gravity
                                                 * RITTER_DRY.h0)
    head = RITTER_DRY.x_dam - t * math.sqrt(RITTER_DRY.gravity * RITTER_DRY.h0)
    for level in LEVEL_POSITION_LEVELS:
        x_level = ritter_level_position(level, t, RITTER_DRY.h0,
                                        RITTER_DRY.gravity, RITTER_DRY.x_dam)
        assert head < x_level < tip


@pytest.mark.parametrize("level", (0.0, 1.0, -0.1, 1.5))
def test_closed_form_rejects_degenerate_levels(level):
    """``level = 1`` is the undisturbed tail (no strict crossing); 0 is the tip."""
    with pytest.raises(ValueError):
        ritter_level_position(level, 1.0, 1.0)


# --------------------------------------------------------------------------
# 2. the measurement imports no analytic quantity
# --------------------------------------------------------------------------
def test_measurement_is_translation_equivariant_in_the_profile():
    """Shifting the depth array shifts the reading by exactly the same amount.

    This is the operational meaning of "oracle-free": the measured abscissa is a
    functional of ``(x, h, level)`` alone.  A detector that consulted the exact
    solution could not move its answer when only the profile is translated.
    """
    solver = _solver(400)
    k = 7
    shifted = np.concatenate([np.repeat(solver.h[0], k), solver.h[:-k]])
    for level in LEVEL_POSITION_LEVELS:
        base = _level_crossing(solver.x, solver.h, level * RITTER_DRY.h0)
        moved = _level_crossing(solver.x, shifted, level * RITTER_DRY.h0)
        assert moved - base == pytest.approx(k * solver.dx, rel=1e-12)


def test_measured_entry_is_exactly_the_profile_crossing():
    """``level_position_metrics`` must not post-process the measured abscissa."""
    solver = _solver(400)
    metrics = level_position_metrics(solver, RITTER_DRY)
    for level in LEVEL_POSITION_LEVELS:
        entry = metrics["levels"][f"{level:g}"]
        assert entry["measured"] == _level_crossing(solver.x, solver.h,
                                                    level * RITTER_DRY.h0)
        # ... while the reference is the independent closed form
        assert entry["exact"] == pytest.approx(
            ritter_level_position(level, solver.time, RITTER_DRY.h0,
                                  RITTER_DRY.gravity, RITTER_DRY.x_dam), rel=1e-15)


# --------------------------------------------------------------------------
# 3. convergence, gate, and no drift of the committed numbers
# --------------------------------------------------------------------------
#: contours shallow enough that the numerical tip smearing is inside the
#: contour itself; they are pre-asymptotic on the grids of this test (measured
#: orders 0.75 -> 1.89 -> 1.34 for n = 200 ... 800) and only reach order one at
#: n = 3200 in the published artefact.  Asserting first order for them *here*
#: would be asserting something the evidence does not say.
SHALLOW_LEVELS = ("0.04", "0.01")


def test_contour_positions_converge_at_first_order(study):
    """A contour of a profile smeared over ``O(dx)`` converges at order one.

    The deep contours (``h/h0 <= 0.25``) are in the asymptotic range from
    ``n = 200``; the shallow ones are held to "no worse than half order" on this
    grid range, with the asymptotic statement left to the artefact (``n =
    3200``), where every declared contour measures order ~1.0.
    """
    rows = study["rows"]
    for level in LEVEL_POSITION_LEVELS:
        key = f"{level:g}"
        errors = [row["levels"][key]["error"] for row in rows]
        assert all(b <= a for a, b in zip(errors, errors[1:])), (
            f"level {key} error is not monotone under refinement: {errors}")
    for coarse, fine in zip(rows, rows[1:]):
        for level in LEVEL_POSITION_LEVELS:
            key = f"{level:g}"
            order = fine["levels"][key].get("observed_order")
            if key in SHALLOW_LEVELS:
                assert order > 0.5, (
                    f"level {key}: order {order:.3f} at n = {fine['n']} stagnates")
            else:
                assert order == pytest.approx(1.0, abs=0.35), (
                    f"level {key}: observed order {order:.3f} at n = {fine['n']}")


def test_gate_holds_on_the_accepted_grids(study):
    """``< 5 %`` on both the final contour positions and the ``x(t)`` series."""
    accepted = [row for row in study["rows"] if row["n"] >= 800]
    assert accepted, "no accepted grid in the study"
    for row in accepted:
        assert row["max_error"] <= TOLERANCE, (row["n"], row["max_error"])
        assert row["series_max_error"] <= TOLERANCE, (row["n"],
                                                      row["series_max_error"])


def test_the_series_is_not_passing_by_measuring_nothing(study):
    """Enough of the ``x(t)`` samples must be graded for the max to mean anything.

    Applied to the grids whose ``x(t)`` error is graded (``n >= 800``).  At
    ``n = 100`` only 6 of 21 samples clear the 20-cell error budget -- the
    budget covers the late window only -- so a fraction floor there would be a
    statement about ``dx``, not about the evidence.
    """
    accepted = [row for row in study["rows"] if row["n"] >= 800]
    assert accepted
    for row in accepted:
        graded = row["series_graded"] / row["series_samples"]
        assert graded >= 0.5, (row["n"], graded)
        assert row["series_graded"] >= 5


def test_declared_series_resolution_criterion_is_an_error_budget():
    """The criterion is ``1 / tolerance`` cells -- declared, not fitted."""
    assert LEVEL_SERIES_MIN_CELLS == pytest.approx(1.0 / TOLERANCE)
    assert 0.0 < LEVEL_POSITION_SERIES_LEVEL < 1.0


def test_committed_artifact_matches_the_code(study):
    """Regenerating a subset must reproduce the published artefact (drift gate)."""
    if not os.path.exists(ARTIFACT):
        pytest.skip("artefact not generated in this checkout")
    with open(ARTIFACT, encoding="utf-8") as handle:
        committed = json.load(handle)["study"]
    by_n = {row["n"]: row for row in committed["rows"]}
    for row in study["rows"]:
        ref = by_n.get(row["n"])
        assert ref is not None, f"n = {row['n']} missing from the artefact"
        assert row["series_max_error"] == pytest.approx(
            ref["series_max_error"], rel=1e-9, abs=1e-12)
        for level in LEVEL_POSITION_LEVELS:
            key = f"{level:g}"
            mine, theirs = row["levels"][key], ref["levels"][key]
            assert mine["measured"] == pytest.approx(theirs["measured"],
                                                     rel=1e-9, abs=1e-12)
            assert mine["error"] == pytest.approx(theirs["error"],
                                                  rel=1e-9, abs=1e-12)


def test_committed_artifact_records_the_gate_outcome():
    """The artefact must state a passing, internally consistent headline."""
    if not os.path.exists(ARTIFACT):
        pytest.skip("artefact not generated in this checkout")
    with open(ARTIFACT, encoding="utf-8") as handle:
        study = json.load(handle)["study"]
    assert study["max_error_at_accepted_grid"] <= study["tolerance"]
    assert study["max_series_error_at_accepted_grid"] <= study["tolerance"]
    assert study["min_observed_order_at_accepted_grid"] >= 0.8


# --------------------------------------------------------------------------
# 4. negative control: the legacy position metric still does not converge
# --------------------------------------------------------------------------
def test_wetted_edge_remains_non_convergent(study):
    """The depth-floor detector must stay visibly worse than the contour metric.

    If this ever passed, either the wetted edge started converging (a real
    improvement in the metric) or the contour metric stopped being sharper --
    both are reviewable events, so fail loudly rather than silently.
    """
    fine = [row for row in study["rows"] if row["n"] >= 400]
    wetted = [row["front_wetted_error"] for row in fine]
    contour = [row["max_error"] for row in fine]
    # refinement does not drive the wetted-edge reading to zero: it grows
    # again between n = 400 and n = 800 (1.56 % -> 2.66 %) while the contour
    # metric halves (2.77 % -> 1.39 %).
    assert not all(b < a for a, b in zip(wetted, wetted[1:])), wetted
    assert wetted[-1] > 1.5 * contour[-1], (
        f"wetted edge {wetted[-1]:.4f} is no longer the weak metric vs contour "
        f"{contour[-1]:.4f} -- re-review the metric choice if this changes")