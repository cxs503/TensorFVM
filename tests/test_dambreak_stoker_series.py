"""The ``x_f(t)`` **series** of the Ritter/Stoker dam-break case, not just its end state.

Why this file exists
--------------------
The case acceptance is a statement about the wave position *as a function of
time* (``x_f(t)`` vs the closed form, ``< 5 %``), plus the still-water level
step at ``t* = 2``.  Before this file the case was graded only on the single
final state: ``run_case`` did record a ``time_to_front_series`` list, but

1. nothing graded it, and
2. it was built with the **wetted-edge** estimator on the dry bed -- the
   depth-floor detector that ``docs/verification-dam-break-front-metric``
   documents as non-convergent.  On the first sample (``t* = 0.005``) that
   estimator reads a ``212 %`` error, i.e. the acceptance quantity was, in the
   only place it existed, wrong by a factor 40 *and* grid-independent.

The fix (``front_sample`` now carries ``headline``/``headline_error`` next to
the legacy fields, ``run_case`` stores ``metrics["front_series"]``) is locked
here from both sides: the graded series must pass at the accepted grid, and the
legacy estimator's failure must stay visible so nobody "simplifies" the code
back to quoting it.

Thresholds are the case's own (``TOLERANCE = 5 %``); the resolution criterion
(``SERIES_MIN_CELLS``) is declared in the module, not fitted here.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import (
    RITTER_DRY,
    SERIES_MIN_CELLS,
    STOKER_WET,
    TOLERANCE,
    series_metrics,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT = os.path.join(REPO_ROOT, "src", "tensorfvm", "verification",
                        "artifacts", "shallow-water-dam-break")


@pytest.fixture(scope="module")
def runs():
    from tensorfvm.shallow_water.dam_break import run_case

    return {
        ("dry", 400): run_case(RITTER_DRY, n=400),
        ("dry", 800): run_case(RITTER_DRY, n=800),
        ("wet", 400): run_case(STOKER_WET, n=400),
        ("wet", 800): run_case(STOKER_WET, n=800),
    }


@pytest.mark.parametrize("kind", ["dry", "wet"])
def test_series_passes_at_the_accepted_grid(runs, kind):
    """``x_f(t)`` error over the whole transient, at the grid the case accepts."""
    series = runs[(kind, 800)]["metrics"]["front_series"]
    assert series["samples_graded"] >= 20
    assert series["max_error"] == pytest.approx(series["max_error"])
    assert series["max_error"] < TOLERANCE, series
    assert series["pass"] is True
    # the graded samples must cover the late transient, i.e. the criterion is
    # not satisfied by grading only the first few well-resolved samples
    assert series["max_error_t_star"] > 0.02


def test_series_carries_both_definitions(runs):
    """Every sample states which estimator is graded and which is legacy."""
    sample = runs[("dry", 800)]["metrics"]["time_to_front_series"][10]
    for key in ("t", "t_star", "position", "exact", "error", "headline",
                "headline_error", "resolved", "travelled_cells"):
        assert key in sample
    # dry bed: the legacy position is the wetted edge, the headline is the fan
    # estimator, and the two are *different* measurements of "the front"
    assert sample["position"] != sample["headline"]
    assert sample["headline_error"] < sample["error"]


def test_dry_wetted_series_defect_is_locked(runs):
    """The legacy series is 200 % off on the dry bed and does not converge.

    Locked deliberately: this is the evidence that grading the series with the
    wetted edge was a defect, not a choice of convention.  If somebody makes
    the wetted edge converge, this test should be *updated with the new
    evidence*, not deleted.
    """
    coarse = runs[("dry", 400)]["metrics"]["front_series"]
    fine = runs[("dry", 800)]["metrics"]["front_series"]
    assert coarse["legacy_max_error"] > 2.0
    assert fine["legacy_max_error"] > 2.0
    # grid independent to within a rounding of the sampling instants => the
    # reading is set by the depth floor, not by dx
    assert fine["legacy_max_error"] == pytest.approx(coarse["legacy_max_error"],
                                                     rel=1e-6)
    # while the graded estimator on the same samples is a discretisation error
    assert fine["max_error"] < 0.02
    assert fine["legacy_max_error"] > 50.0 * fine["max_error"]


def test_dry_series_converges(runs):
    coarse = runs[("dry", 400)]["metrics"]["front_series"]["max_error"]
    fine = runs[("dry", 800)]["metrics"]["front_series"]["max_error"]
    assert fine < 0.5 * coarse, (coarse, fine)


def test_wet_early_transient_bias_is_reported_not_hidden(runs):
    """The first bore samples are under-resolved: they are reported, not graded.

    At ``t* <~ 0.02`` the bore has moved less than ``SERIES_MIN_CELLS`` cells,
    so its mid-level crossing sits inside the smearing of the initial
    discontinuity.  The bias there is ~4.5 % and *grid independent* (it is the
    smeared ramp of the initial jump, not a discretisation error), so it must
    not silently contribute to a converging series -- but it must be visible.
    """
    series = runs[("wet", 800)]["metrics"]["front_series"]
    assert series["min_cells"] == SERIES_MIN_CELLS
    assert series["unresolved_samples"] >= 1
    assert series["unresolved_max_error"] == pytest.approx(0.045, abs=5e-3)
    assert series["unresolved_max_error_t_star"] < 0.05
    # honest bookkeeping: inside tolerance, but with only a ~10 % margin, and
    # the graded series is an order of magnitude better
    assert series["unresolved_max_error"] < TOLERANCE
    assert series["max_error"] < 0.2 * series["unresolved_max_error"]


def test_series_metrics_guards_empty_input():
    empty = series_metrics(RITTER_DRY, [])
    assert empty["samples"] == 0
    assert empty["max_error"] is None
    assert empty["pass"] is False


def test_committed_artifact_publishes_the_headline_metric():
    """The published summary must quote the estimator it can defend.

    ``SUMMARY.md`` used to print the wetted edge as "front position" (2.891 % at
    n = 1600) while the module's graded headline was the fan estimator (0.017 %):
    a factor 170 statement about the solver's accuracy.  The generator now
    prints both, the headline first.
    """
    summary = open(os.path.join(ARTIFACT, "SUMMARY.md"), encoding="utf-8").read()
    assert "front position x_f (fan, headline)" in summary
    assert "wetted edge, diagnostic" in summary
    assert "wave position **series** x_f(t)" in summary
    metrics = json.load(open(os.path.join(ARTIFACT, "metrics.json"), encoding="utf-8"))
    for key in ("ritter-dry", "stoker-wet"):
        series = metrics["cases"][key]["front_series_summary"]
        assert series["pass"] is True, key
        assert series["max_error"] < TOLERANCE, key
        assert series["tolerance"] == TOLERANCE