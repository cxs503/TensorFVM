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


def test_dry_series_is_a_discretisation_error(runs):
    """The graded dry series error must be a *discretisation* error.

    This test used to assert ``err(800) < 0.5 * err(400)``.  That "convergence"
    was an artefact: the n = 400 maximum came from a sample whose fan window
    held only three cells, where the fan estimator degenerates to a cell-centre
    locator and reads ``0.493 dx`` -- a half cell, i.e. a resolution
    *invariant* relative error (6.613 % at both n = 200 and n = 400).  Once such
    samples are excluded by ``SERIES_MIN_FAN_CELLS`` the remaining error no
    longer obeys a factor-2 law (the sample instants differ per grid), so the
    honest invariants are:

    1. every grid clears the acceptance with a wide margin (``< 1 %``, i.e. 5x
       inside the 5 % tolerance) -- previously n = 200/400 *failed* at 6.6 %;
    2. the finest grid is at least as good as every coarser one;
    3. the coarse-grid samples that used to produce that spurious failure are
       still visible in the reason inventory as ``fan_cells``.
    """
    err400 = runs[("dry", 400)]["metrics"]["front_series"]["max_error"]
    err800 = runs[("dry", 800)]["metrics"]["front_series"]["max_error"]
    assert err400 < 0.01 and err800 < 0.01, (err400, err800)
    assert err800 <= err400, (err400, err800)
    reasons = runs[("dry", 400)]["metrics"]["front_series"]["unresolved_reasons"]
    assert reasons.get("fan_cells", 0) >= 1, reasons


def test_dry_three_cell_fan_artefact_is_locked(runs):
    """Lock the defect the fan-cell floor removes: the ungated metric is broken.

    Re-grading the *same* samples with only the travel criterion (the pre-fix
    gate) must reproduce the spurious ``> 5 %`` failure, and the offending
    sample must be the 3-cell one.  If someone deletes ``SERIES_MIN_FAN_CELLS``
    this test still documents why that is not allowed; if the solver's tip
    resolution changes, the numbers move and the test tells us to re-measure.
    """
    from tensorfvm.shallow_water.dam_break import SERIES_MIN_FAN_CELLS

    ungated = {}
    for n in (400, 800):
        samples = runs[("dry", n)]["metrics"]["time_to_front_series"]
        graded = [s for s in samples
                  if s["travelled_resolved"] and s["headline_error"] == s["headline_error"]]
        worst = max(graded, key=lambda s: s["headline_error"])
        ungated[n] = (worst["headline_error"], worst["headline_cells"])
    # the defect: > 5 % on a well-resolved front, from a 3-cell fan window
    assert ungated[400][0] > TOLERANCE, ungated
    assert ungated[400][1] < SERIES_MIN_FAN_CELLS, ungated
    # and after the floor the same run is comfortably inside tolerance
    assert runs[("dry", 400)]["metrics"]["front_series"]["max_error"] < 0.01


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

# --------------------------------------------------------------------------- #
# the series gate is *in the verdict*, and it is not vacuous
# --------------------------------------------------------------------------- #
def test_series_gate_is_a_verdict_entry_for_both_cases(runs):
    """The acceptance quantity must be graded by the machine verdict.

    ``metrics["front_series"]`` existed but nothing gated it: the dry branch
    reported ``all_pass`` at n = 400 while its own graded series error was
    6.61 % > 5 %.  The gate now lives in the verdict for both cases.
    """
    for kind in ("dry", "wet"):
        verdict = runs[(kind, 800)]["verdict"]
        entry = verdict["front_series<5%"]
        assert entry["limit"] == TOLERANCE
        assert entry["pass"] is True
        # the entry carries the counts, so the gate can be audited
        assert entry["samples"] == runs[(kind, 800)]["metrics"]["front_series"]["samples"]
        assert entry["samples_graded"] >= entry["graded_required"]
        assert entry["value"] == pytest.approx(
            runs[(kind, 800)]["metrics"]["front_series"]["max_error"], rel=1e-12)
        # every other entry keeps its numeric schema
        assert all(isinstance(item["pass"], bool)
                   for key, item in verdict.items() if key != "all_pass")


def test_series_gate_fails_where_it_should(runs):
    """Falsification: the gate must be able to fail.

    This test used to point at the n = 400 dry run, whose 6.6 % "failure" was
    a three-cell-fan artefact (that run now passes legitimately at 0.68 %).
    The gate is therefore falsified directly: the same graded samples are
    perturbed by a known >5 % front offset and the gate must reject them, so a
    regression that makes the entry unconditional is caught.
    """
    from tensorfvm.shallow_water.dam_break import series_metrics

    run = runs[("dry", 400)]
    samples = [dict(s) for s in run["metrics"]["time_to_front_series"]]
    gate = series_metrics(RITTER_DRY, samples)
    assert gate["pass"] is True
    for shift in (+0.06, -0.06):
        biased = [dict(s, headline=s["headline"] * (1.0 + shift),
                       headline_error=abs(s["headline"] * (1.0 + shift) - s["exact"])
                       / s["exact"]) for s in samples]
        failed = series_metrics(RITTER_DRY, biased)
        assert failed["pass"] is False, (shift, failed["max_error"])
        assert failed["max_error"] > TOLERANCE
    assert runs[("dry", 800)]["verdict"]["all_pass"] is True


def test_series_gate_rejects_a_vacuous_series():
    """A gate that grades almost nothing measures nothing -> insufficient evidence."""
    from tensorfvm.shallow_water.dam_break import (
        SERIES_MIN_GRADED_FRACTION,
        _series_verdict_entry,
    )

    # tiny error, but only 2 of 25 samples resolvable: not evidence
    vacuous = _series_verdict_entry({
        "front_series": {"samples": 25, "samples_graded": 2,
                         "max_error": 1e-4, "unresolved_max_error": 4.5e-2}})
    assert vacuous["pass"] is False
    assert vacuous["graded_required"] == int(np.ceil(SERIES_MIN_GRADED_FRACTION * 25))
    # no series at all, and a series with no gradable sample
    assert _series_verdict_entry({})["pass"] is False
    assert _series_verdict_entry({})["value"] is None
    assert _series_verdict_entry(
        {"front_series": {"samples": 25, "samples_graded": 25,
                          "max_error": None}})["pass"] is False
    # exactly at the floor with a passing error -> passes
    ok = _series_verdict_entry(
        {"front_series": {"samples": 25, "samples_graded": 13,
                          "max_error": TOLERANCE / 2}})
    assert ok["pass"] is True and ok["graded_required"] == 13
