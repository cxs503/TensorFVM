"""Robustness envelope of the ``CASE-FVM-DAMBREAK-STOKER`` graded numbers.

These tests grade the *perturbation axes* of
:mod:`tensorfvm.shallow_water.robustness`, not the dam-break solver again (that
is ``test_shallow_water_dam_break.py``).  They answer, with numbers, the
question a reviewer of a published benchmark always has: **is the published
error an accident of the declared geometry?**  Every assertion here is a
predeclared budget from the module (``ALIGNMENT_MAX_PP``, ``BOUNDARY_MAX_PP``,
``DOMAIN_MAX_PP``, ``SIMILARITY_REL_TOL``, ``DEPTH_RATIO_MAX_ERROR``); the
resolution is deliberately small (n = 150 ... 200) so the file stays a guard
rail, while the graded evidence at n = 400 ... 1600 is produced by
``docs/case-fvm-dambreak-stoker-robustness/run_robustness_sweep.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET, run_case
from tensorfvm.shallow_water.robustness import (
    ALIGNMENT_MAX_PP,
    BOUNDARY_MAX_PP,
    DEPTH_RATIO_MAX_ERROR,
    DOMAIN_MAX_PP,
    SIMILARITY_REL_TOL,
    _extremum_amplitude,
    headline_error,
    run_sweep,
    sweep_verdict,
    variant_cases,
)

N = 150


def _by_key(sweep: dict) -> dict:
    return {(r["base"], r["variant"], r["sweep_n"]): r for r in sweep["runs"]}


@pytest.fixture(scope="module")
def small_sweep() -> dict:
    return run_sweep(ns=(N,))


def _delta_pp(probe: dict, ref: dict) -> float:
    return 100.0 * (probe["graded"]["headline"] - ref["graded"]["headline"])


@pytest.mark.parametrize("base", ["ritter-dry", "stoker-wet"])
@pytest.mark.parametrize("variant", ["translated_quarter", "translated_half"])
def test_translation_is_not_a_robustness_axis(small_sweep, base, variant):
    """Translating dam **and** grid must not move the graded error.

    The initial condition is sampled at cell centres, so the discrete problem is
    translation invariant by construction; the test pins that claim (a solver
    that keyed anything to an absolute coordinate, e.g. ``x_dam == 0`` or an
    even/odd cell index, would fail here) *and* the consequence for the
    published number: no discontinuity-on-a-cell-face accident exists.
    """
    index = _by_key(small_sweep)
    probe = index[(base, variant, N)]
    ref = index[(base, "baseline", N)]
    assert probe["graded"]["dx"] == pytest.approx(ref["graded"]["dx"], rel=1e-12)
    assert abs(_delta_pp(probe, ref)) <= ALIGNMENT_MAX_PP
    # the wave itself must have moved with the geometry, not stayed put
    shift = (probe["graded"]["headline_position"]
             - ref["graded"]["headline_position"])
    dx = ref["graded"]["dx"]
    expected = {"translated_quarter": 0.25, "translated_half": 0.5}[variant] * dx
    assert shift == pytest.approx(expected, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("base", ["ritter-dry", "stoker-wet"])
def test_domain_length_does_not_change_the_answer(small_sweep, base):
    """Doubling the domain at the same ``dx`` must not move the graded error."""
    index = _by_key(small_sweep)
    probe = index[(base, "wide_domain", N)]
    ref = index[(base, "baseline", N)]
    assert probe["graded"]["dx"] == pytest.approx(ref["graded"]["dx"], rel=1e-12)
    assert abs(_delta_pp(probe, ref)) <= DOMAIN_MAX_PP


@pytest.mark.parametrize("base", ["ritter-dry", "stoker-wet"])
def test_downstream_boundary_proximity(small_sweep, base):
    """An outflow 8 cells behind the wave must stay inside its budget."""
    index = _by_key(small_sweep)
    probe = index[(base, "tight_downstream", N)]
    ref = index[(base, "baseline", N)]
    # ``n`` is re-derived from the shortened domain so that dx matches; the
    # integer rounding is reported, not hidden
    assert probe["graded"]["dx"] == pytest.approx(ref["graded"]["dx"], rel=5e-3)
    assert abs(_delta_pp(probe, ref)) <= BOUNDARY_MAX_PP


@pytest.mark.parametrize("base", ["ritter-dry", "stoker-wet"])
def test_similitude_reproduces_the_nondimensional_answer(small_sweep, base):
    """``x -> 2x``, ``h -> 2h`` at fixed ``n`` and ``t*`` must change nothing.

    This is the check that catches a scale-dependent constant in the code (an
    absolute depth floor, an absolute limiter tolerance, a hard-coded length):
    every graded quantity is non-dimensional, so the drift has to be round-off.
    """
    index = _by_key(small_sweep)
    probe = index[(base, "similarity_x2", N)]["graded"]
    ref = index[(base, "baseline", N)]["graded"]
    # the geometry really was doubled
    assert probe["exact_position"] == pytest.approx(2.0 * ref["exact_position"],
                                                    rel=1e-12)
    assert abs(probe["position_ratio"] / ref["position_ratio"] - 1.0) <= SIMILARITY_REL_TOL
    assert abs(probe["depth_l1"] - ref["depth_l1"]) <= SIMILARITY_REL_TOL * ref["depth_l1"]


@pytest.mark.parametrize("ratio", ["depth_ratio_0.2", "depth_ratio_0.8"])
def test_added_depth_ratios_meet_the_anchor_tolerance(small_sweep, ratio):
    """Bore strength off the graded 0.5 must still meet the 5 % acceptance."""
    index = _by_key(small_sweep)
    graded = index[("stoker-wet", ratio, N)]["graded"]
    assert graded["headline"] <= DEPTH_RATIO_MAX_ERROR
    assert graded["h_min"] >= 0.0
    assert graded["mass_drift"] <= 1e-10


def test_variant_cases_preserve_dx_and_are_complete():
    """The variant list must be the declared one and keep ``dx`` where claimed."""
    specs = variant_cases(100)
    names = {s["variant"] for s in specs}
    assert {"baseline", "translated_quarter", "translated_half",
            "depth_ratio_0.2", "depth_ratio_0.8"} <= names
    for spec in specs:
        case, n = spec["case"], spec["n"]
        dx = (case.x_max - case.x_min) / float(n)
        if spec["variant"] in ("tight_downstream", "tight_dx_rounding",
                              "wide_domain"):
            # domain changed: n was re-derived, dx is preserved to rounding
            assert dx == pytest.approx(10.0 / 100.0, rel=5e-3)
            assert spec["ref_n"] == 100
        elif spec["variant"] in ("translated_quarter", "translated_half"):
            assert dx == pytest.approx(10.0 / 100.0, rel=1e-12)
            assert spec["ref_n"] == 100


def test_dx_control_is_a_stronger_effect_than_the_boundary(small_sweep):
    """The control must show why ``dx`` has to be held exactly.

    ``tight_dx_rounding`` perturbs the same short domain as
    ``tight_downstream`` but rounds the cell count, leaving ``dx`` off by a
    fraction of a percent.  Where that control is active, its shift of the
    graded error must be *larger* than the (exactly zero) shift of the
    ``dx``-preserving variant -- otherwise the sweep's zero would be luck.
    """
    checks = sweep_verdict(small_sweep)["checks"]
    active = [c for c in checks.values() if "max_abs_delta_pp" in c and c["active"]]
    assert active, "the dx control must be active on at least one bed type"
    for check in active:
        assert check["max_dx_rel"] > 1e-4
        assert check["max_abs_delta_pp"] > 0.0
    boundary = [c for k, c in checks.items() if k.endswith("tight_downstream")]
    assert all(c["worst_delta_pp"] == pytest.approx(0.0, abs=1e-9)
               for c in boundary)


def test_extremum_amplitude_detector():
    """The monotonicity diagnostic must measure the bump, not just count it."""
    h0 = 1.0
    flat = np.ones(9)
    assert _extremum_amplitude(flat, h0) == 0.0
    bump = flat.copy()
    bump[4] += 3e-3
    assert _extremum_amplitude(bump, h0) == pytest.approx(3e-3, rel=1e-12)
    # a bump below the reported floor is not an oscillation
    tiny = flat.copy()
    tiny[4] += 5e-5
    assert _extremum_amplitude(tiny, h0) == 0.0


def test_verdict_flags_a_planted_violation(small_sweep):
    """The gate must be able to fail: inflate one variant and re-judge."""
    verdict = sweep_verdict(small_sweep)
    assert verdict["all_pass"] is True

    planted = {**small_sweep, "runs": [dict(r) for r in small_sweep["runs"]]}
    for run in planted["runs"]:
        if run["variant"] == "translated_half":
            graded = dict(run["graded"])
            graded["headline"] = graded["headline"] + 1e-2  # +1 pp
            run["graded"] = graded
    planted_verdict = sweep_verdict(planted)
    assert planted_verdict["all_pass"] is False
    failed = [k for k, c in planted_verdict["checks"].items() if not c["pass"]]
    assert failed and all(k.endswith("translated_half") for k in failed)


def test_headline_error_is_the_graded_metric():
    """``headline_error`` must be the metric the verdict grades, per bed type."""
    dry = run_case(RITTER_DRY, n=100)
    wet = run_case(STOKER_WET, n=100)
    assert headline_error("dry", dry["metrics"]) == pytest.approx(
        dry["verdict"]["front_fan<5%"]["value"], rel=1e-12)
    assert headline_error("wet", wet["metrics"]) == pytest.approx(
        wet["verdict"]["shock_position<5%"]["value"], rel=1e-12)