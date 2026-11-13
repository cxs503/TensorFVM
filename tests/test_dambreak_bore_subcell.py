"""Sub-cell bore position for the Stoker (wet-bed) dam-break case (F7).

Why this file exists
--------------------
``docs/case-fvm-dambreak-stoker/AUDIT.md`` F4 recorded that the oracle-guided
wet-bed metrics are blind to a misplaced bore; the fix (``data_driven_plateau``,
locked by ``test_dambreak_data_driven_plateau.py``) anchored the window to the
*measured* bore.  That fix read the bore position as the **cell centre of the
maximum gradient** -- the same defect class as the dry-bed wetted edge (F1): the
reading is quantised to ``dx``, so its refinement error is non-monotone with an
``O(dx/2)`` floor (0.73 % -> 0.068 % -> 0.399 % -> 0.233 % for n = 200 ... 1600)
and it structurally cannot express a sub-cell displacement.

This file locks the repair: ``bore_x`` is the linearly interpolated crossing of
the mid level ``(h2 + h1) / 2`` between the two **measured** states (no analytic
value enters), while ``bore_x_cell`` keeps the legacy cell-centre reading as the
documented negative control.

Predeclared criteria (module constants / this file, not fitted to a run):

* ``BORE_SUBCELL_MAX_CELLS = 0.5``  -- the sub-cell reading must resolve the
  bore to better than half a cell at *every* refinement rung, otherwise it is a
  cell-centre detector in disguise;
* translation tolerance ``0.1 dx``   -- an imposed fractional-cell translation of
  the depth profile must be recovered to within a tenth of a cell;
* independence -- a pure translation must leave ``h2`` unchanged and a pure
  plateau-amplitude change must leave ``bore_x`` unchanged, so
  amplitude (``h2``/``step_ratio``) and position (``bore_x``) are independent
  evidence rather than two near-tautologies.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    BORE_SUBCELL_MAX_CELLS,
    STOKER_WET,
    data_driven_plateau,
    run_case,
)

ARTIFACT = (Path(__file__).resolve().parents[1]
            / "src/tensorfvm/verification/artifacts/shallow-water-dam-break"
            / "bore_subcell.json")

TRANSLATION_TOL_CELLS = 0.1
X_SHOCK = STOKER_WET.exact_front()


@pytest.fixture(scope="module")
def runs():
    return {n: run_case(STOKER_WET, n=n) for n in (200, 400, 800)}


def _profile(run):
    return (np.asarray(run["profile"]["x"], dtype=float),
            np.asarray(run["profile"]["h_numeric"], dtype=float))


def test_subcell_bore_resolves_below_half_a_cell(runs):
    """Predeclared: the position reading must beat ``0.5 dx`` at every rung."""
    for n, run in runs.items():
        dx = run["discretisation"]["dx"]
        dd = run["metrics"]["dd"]
        assert dd["bore_method"] == "mid_level_crossing", n
        error_cells = dd["bore_x_error"] * X_SHOCK / dx
        assert error_cells <= BORE_SUBCELL_MAX_CELLS, (n, error_cells)
        # it is still a cross-check, not a re-derivation of the closed form
        assert dd["bore_x_error"] < 0.005, (n, dd["bore_x_error"])


def test_legacy_cell_centre_reading_is_quantised(runs):
    """Negative control: the legacy reading cannot resolve below a cell.

    On the accepted grid the cell-centre error is several times the sub-cell
    one, and the *sequence* over refinement is non-monotone -- the signature of
    a reading pinned to cell centres rather than to the wave.
    """
    errors = []
    for n, run in runs.items():
        dx = run["discretisation"]["dx"]
        dd = run["metrics"]["dd"]
        errors.append(dd["bore_x_cell_error"] * X_SHOCK / dx)
    assert max(errors) > BORE_SUBCELL_MAX_CELLS, errors
    # the legacy error is not monotonically decreasing under refinement
    deltas = [errors[i + 1] - errors[i] for i in range(len(errors) - 1)]
    assert any(delta > 0.0 for delta in deltas), errors
    # while the sub-cell reading keeps shrinking (n = 200 -> 800)
    sub = [runs[n]["metrics"]["dd"]["bore_x_error"] for n in sorted(runs)]
    assert sub[-1] < sub[0], sub


def test_subcell_recovers_fractional_cell_translation(runs):
    """A sub-cell displacement of the bore must be *measured*, not rounded away."""
    run = runs[800]
    x, h = _profile(run)
    dx = x[1] - x[0]
    base = data_driven_plateau(x, h)
    worst_sub, worst_cell = 0.0, 0.0
    for frac in (0.2, 0.4, 0.5, 0.6, 0.8):
        delta = frac * dx
        shifted = np.interp(x - delta, x, h)       # translate the profile right
        dd = data_driven_plateau(x, shifted)
        worst_sub = max(worst_sub,
                        abs(dd["bore_x"] - base["bore_x"] - delta) / dx)
        worst_cell = max(worst_cell,
                         abs(dd["bore_x_cell"] - base["bore_x_cell"] - delta) / dx)
        # a pure translation leaves the plateau amplitude alone
        assert abs(dd["h2"] - base["h2"]) / base["h2"] < 1e-4, frac
    assert worst_sub < TRANSLATION_TOL_CELLS, worst_sub
    assert worst_cell > 2.0 * TRANSLATION_TOL_CELLS, worst_cell


def test_position_and_amplitude_are_independent(runs):
    """``h2`` (amplitude) and ``bore_x`` (position) must not be redundant.

    Suppressing one defect class alone: an amplitude error must not be
    absorbed by the position reading, and vice versa.
    """
    x, h = _profile(runs[800])
    dx = x[1] - x[0]
    base = data_driven_plateau(x, h)
    # (a) amplitude defect, position untouched
    plateau = (x > base["bore_x"] - 1.0) & (x < base["bore_x"])
    scaled = h.copy()
    scaled[plateau] *= 1.015
    dd = data_driven_plateau(x, scaled)
    assert abs(dd["h2"] - base["h2"]) / base["h2"] > 1e-3
    assert abs(dd["bore_x"] - base["bore_x"]) / dx < 0.05
    # (b) position defect, amplitude untouched: one cell of translation
    shifted = np.interp(x - dx, x, h)
    dd = data_driven_plateau(x, shifted)
    assert abs(dd["bore_x"] - base["bore_x"] - dx) / dx < TRANSLATION_TOL_CELLS
    assert abs(dd["h2"] - base["h2"]) / base["h2"] < 1e-4


def test_committed_artifact_records_the_subcell_result():
    """The published study must carry the fix and pass its own criteria."""
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    ref = payload["refinement"]
    assert ref["limit_cells"] == BORE_SUBCELL_MAX_CELLS
    assert ref["pass"] and ref["subcell_worst_cells"] <= BORE_SUBCELL_MAX_CELLS
    assert ref["improvement_factor"] > 2.0
    assert payload["translation"]["pass"]
    assert payload["amplitude_mutation"]["pass"]
    for row in ref["rows"]:
        assert row["bore_method"] == "mid_level_crossing"
        assert row["error_cells"] <= BORE_SUBCELL_MAX_CELLS
        # the accepted grid (n >= 400) passes the case verdict; the coarsest
        # rung is *recorded as failing* on the monotonicity check rather than
        # dropped -- an overshoot of amplitude > 1e-4 h0 at n = 200 is a real
        # (coarse-grid) defect of the scheme, not of the metric.
        if row["n"] >= 400:
            assert row["verdict_all_pass"], (row["n"], row["verdict_failing"])
        else:
            assert row["verdict_failing"] == ["monotone(no_extrema)"], row["verdict_failing"]


def test_degenerate_input_reports_undefined_subcell_fields():
    x = np.linspace(0.0, 1.0, 11)
    dd = data_driven_plateau(x, np.zeros_like(x))
    assert dd["bore_method"] == "undefined"
    assert np.isnan(dd["bore_x"]) and np.isnan(dd["bore_x_cell"])
    assert np.isnan(dd["bore_level"])