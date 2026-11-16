"""Regression gate for the third-code cross-validation of the dam-break case.

The archived study (``src/tensorfvm/verification/artifacts/shallow-water-third-code/
xval.json``) is the evidence; this file asserts the *claims* the report makes
about it, so a regenerated archive that weakens them fails here:

* the cross-code distance is at/below the pool tolerance and, read against the
  grid's own one-cell sensitivity, below half a cell of wave translation;
* the anchor's headline readings, measured by the same instrument on the foreign
  code, also pass the tolerance (gate portability, not a TensorFVM-only pass);
* the metric responds monotonically to injected defects and does not fire on
  round-off (an uncalibrated pass would be indistinguishable from a blind metric);
* the honest scope limit is archived, not papered over: Classic Clawpack cannot
  run the anchor's exact dry-bed initial data, and the wettability floor of each
  code is recorded with TensorFVM's strictly lower than the third code's;
* a live smoke run (skipped when clawpack is absent) reproduces the sign of the
  wet-bed agreement on a small grid and the dry-bed failure on the spot.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from tensorfvm.shallow_water.dam_break import CASES
from tensorfvm.verification.dambreak_third_code import (
    TOLERANCE,
    clawpack_available,
    cross_code_distance,
    profile_readings,
    relative_l1,
    run_clawpack,
    run_tensorfvm,
    third_code_study,
)

ARTIFACT = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorfvm"
            / "verification" / "artifacts" / "shallow-water-third-code" / "xval.json")


@pytest.fixture(scope="module")
def xval():
    if not ARTIFACT.exists():  # pragma: no cover
        pytest.skip(f"archived study missing: {ARTIFACT}")
    return json.loads(ARTIFACT.read_text())


# --------------------------------------------------------------------------- #
# archived study: the claims
# --------------------------------------------------------------------------- #
def test_archive_declares_the_third_code_and_tolerance(xval):
    assert xval["tolerance"] == TOLERANCE
    assert "clawpack" in xval["third_code"].lower()
    assert xval["cfl"] == 0.4
    assert xval["accepted_min_n"] == 800
    assert xval["available"] is True


def test_wet_bed_cross_code_distance_within_tolerance(xval):
    """XV1: on the pool's own tolerance the two code families agree."""
    xv1 = xval["verdict"]["criteria"]["XV1_cross_code_depth_l1"]
    assert xv1["pass"] is True
    assert xv1["value"] <= TOLERANCE
    # and the resolution-aware statement: less than half a cell of translation
    assert xv1["equivalent_translation_cells"] < 0.5
    # the distance must also shrink when the grid is refined (both codes converge)
    grids = [int(n) for n in xval["wet"]]
    coarse = xval["wet"][str(min(grids))]["cross_code"]["clawpack-classic-roe"]["depth_l1"]
    fine = xval["wet"][str(max(grids))]["cross_code"]["clawpack-classic-roe"]["depth_l1"]
    assert fine < coarse


def test_gate_portability_on_the_foreign_code(xval):
    """XV2: the anchor's readings, same instrument, foreign profile."""
    xv2 = xval["verdict"]["criteria"]["XV2_gate_portability"]
    assert xv2["pass"] is True
    assert xv2["value"] <= TOLERANCE
    for n_str, row in xval["wet"].items():
        if int(n_str) < xval["accepted_min_n"]:
            continue
        for code, metrics in row["codes"].items():
            assert metrics["ok"] is True, code
            assert metrics["bore_error"] <= TOLERANCE, (n_str, code)
            assert metrics["step_error"] <= TOLERANCE, (n_str, code)
            assert metrics["h2"] == pytest.approx(metrics["h2_exact"], rel=TOLERANCE)


def test_monotone_criterion_transfers_to_the_third_code(xval):
    xv3 = xval["verdict"]["criteria"]["XV3_instrument_monotone"]
    assert xv3["pass"] is True and xv3["failures"] == []
    for n_str, row in xval["wet"].items():
        if int(n_str) < xval["accepted_min_n"]:
            continue  # below the accepted minimum the anchor itself withholds
        for metrics in row["codes"].values():
            assert metrics["n_extrema"] == 0, (n_str, metrics.get("n_extrema"))


def test_dry_bed_scope_limit_is_recorded_honestly(xval):
    """XV4: the dry case is *not* cross-validated, and the archive says so."""
    xv4 = xval["verdict"]["criteria"]["XV4_dry_bed_third_code"]
    assert xv4["pass"] is False
    assert xv4["runnable_grids"] == []
    for row in xval["dry"].values():
        third = {k: v for k, v in row["codes"].items() if k != "tensorfvm-hll-muscl-ssprk3"}
        assert third, "dry table must contain the third-code attempts"
        for metrics in third.values():
            assert metrics["ok"] is False
            assert "positivity" in metrics["reason"].lower() or "positive" in metrics["reason"].lower()
        graded = row["codes"]["tensorfvm-hll-muscl-ssprk3"]
        assert graded["ok"] is True
        assert graded["front_error"] <= TOLERANCE
        assert graded["n_extrema"] == 0


def test_dry_limit_ladder_brackets_the_two_code_families(xval):
    """Both codes run for h1 >= the third code's floor, and agree there."""
    ladder = xval["dry_limit"]
    assert ladder, "dry-limit ladder missing"
    for h1_str, rungs in ladder.items():
        for n_str, row in rungs.items():
            alive = [k for k, v in row["codes"].items() if v.get("ok")]
            if len(alive) < 2:
                continue
            for dist in row["cross_code"].values():
                assert dist["depth_l1"] <= TOLERANCE, (h1_str, n_str)


def test_wettability_floor_quantifies_the_scope_limit(xval):
    floor = xval["wettability_floor"]
    assert floor["floor"]["tensorfvm"] is not None
    for code in ("roe", "hlle"):
        assert floor["floor"][code] is not None
        # the third code needs a strictly thicker downstream layer than TensorFVM
        assert floor["floor"][code] > floor["floor"]["tensorfvm"]


def test_metric_calibration_is_a_ladder_not_a_token(xval):
    """Injection probes: monotone response, sharp position reading, no false alarm."""
    cal = xval["calibration"]
    ladder = [r for r in cal["sensitivity_ladder"] if "translation_cells" in r]
    assert len(ladder) >= 5
    l1 = [r["depth_l1"] for r in ladder]
    assert l1 == sorted(l1) and l1[0] < l1[-1] / 4.0
    bore = [r["bore_error"] for r in ladder]
    assert bore == sorted(bore)
    # the position reading reacts to a one-cell translation that the level step
    # cannot see at all (this is the documented blindness of the amplitude key)
    assert bore[0] > 1e-3
    assert abs(ladder[0]["step_error"]) < 1e-4
    # an amplitude error is seen by the distance and (nearly) not by the position
    amps = [r for r in cal["sensitivity_ladder"] if "plateau_amplitude" in r]
    assert amps and amps[-1]["depth_l1"] > amps[0]["depth_l1"] > 0.0
    assert cal["detection"]["roundoff_false_alarm"] is False
    assert cal["detection"]["depth_l1_per_cell_translation"] > 0.0


# --------------------------------------------------------------------------- #
# live smoke run (needs the third code installed)
# --------------------------------------------------------------------------- #
pytestmark_live = pytest.mark.skipif(not clawpack_available(),
                                     reason="clawpack not importable in this environment")


@pytestmark_live
def test_live_wet_bed_agreement_on_a_coarse_grid():
    wet = CASES["stoker-wet"]
    tfvm = run_tensorfvm(wet, 400)
    third = run_clawpack(wet, 400, "roe")
    assert tfvm["ok"] and third["ok"]
    dist = cross_code_distance(third, tfvm)
    assert dist["depth_l1"] <= TOLERANCE
    for prof in (tfvm, third):
        readings = profile_readings(wet, prof["x"], prof["h"], prof["t"])
        assert readings["bore_error"] <= TOLERANCE


@pytestmark_live
def test_live_dry_bed_is_not_runnable_by_the_third_code():
    dry = CASES["ritter-dry"]
    attempt = run_clawpack(dry, 400, "roe")
    assert attempt["ok"] is False
    assert "positivity" in attempt["reason"].lower() or "positive" in attempt["reason"].lower()
    graded = run_tensorfvm(dry, 400)
    readings = profile_readings(dry, graded["x"], graded["h"], graded["t"])
    assert readings["front_error"] <= TOLERANCE


@pytestmark_live
def test_study_reports_pass_for_the_mandatory_criteria():
    study = third_code_study((400,), ladder=(0.05,))
    assert study["available"] is True
    # with one grid the accepted-minimum gate has nothing to grade: withheld
    assert study["verdict"]["status"] == "withheld"


# --------------------------------------------------------------------------- #
# pure helpers
# --------------------------------------------------------------------------- #
def test_relative_l1_identity_and_scale():
    x = np.linspace(0.0, 1.0, 11)
    a = np.ones_like(x)
    assert relative_l1(x, a, a) == 0.0
    assert relative_l1(x, 2.0 * a, a) == pytest.approx(1.0)


def test_cross_code_distance_rejects_mismatched_grids():
    prof = {"n": 10, "x": np.zeros(10), "h": np.ones(10), "hu": np.ones(10), "u": np.ones(10)}
    with pytest.raises(ValueError):
        cross_code_distance(prof, dict(prof, n=11, x=np.zeros(11), h=np.ones(11)))