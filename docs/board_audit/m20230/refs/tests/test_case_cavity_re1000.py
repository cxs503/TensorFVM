"""Machine gate for CASE-CAVITY-RE1000-FVM (lid-driven cavity, Re = 1000, Ghia 1982).

Two layers, both re-derived from evidence rather than trusted:

1. **Bundle layer** -- every number is read back from
   ``docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json`` (written by
   ``python -m tensorfvm.case_cavity_re1000``) and re-scored here against the
   committed Ghia transcription in :mod:`tensorfvm.lid_cavity_reference`; the
   stored verdict is recomputed, never taken on faith.  Skips cleanly when the
   bundle is absent (a fresh checkout has no run artefacts).
2. **Run layer** -- a deliberately tiny, fast run of the *same* implementation
   (n = 16, a few hundred steps) so the entry point, the bundle writer and the
   scoring path are executed by the suite and not merely imported.

The reference numbers are never retyped: they come from
:func:`tensorfvm.lid_cavity_reference.table` and ``VORTEX_CENTRE``.
"""

from __future__ import annotations

import json
import os

import pytest

from tensorfvm.case_cavity_re1000 import (
    CASE_SLUG,
    CRITERIA,
    REYNOLDS,
    TOLERANCE_FRACTION,
    build_bundle,
    judge,
    ladder_row,
    run_case,
    write_bundle,
)
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(ROOT, "docs", "CASE-CAVITY-RE1000-FVM", "artifacts", "summary.json")
LADDER_DIR = os.path.join(ROOT, "src", "tensorfvm", "verification", "artifacts",
                          "verification_ladder")
TOL = TOLERANCE_FRACTION


def _bundle() -> dict:
    if not os.path.exists(BUNDLE):
        pytest.skip(f"no artefact bundle at {BUNDLE}")
    with open(BUNDLE) as fh:
        return json.load(fh)


# --------------------------------------------------------------- run layer
def test_implementation_runs_and_writes_a_bundle(tmp_path):
    """The entry point really runs the solver and leaves a self-describing bundle."""
    summary = build_bundle(out_dir=tmp_path, grids=(16,), control_grids=(),
                           reynolds=REYNOLDS, tolerance=1e-3, max_steps=200,
                           check_every=50, threads=1, continuation_check=False)
    files = write_bundle(summary, out_dir=tmp_path)
    on_disk = json.load(open(files["summary"]))
    assert on_disk["case"] == CASE_SLUG
    assert on_disk["reynolds"] == REYNOLDS
    case = on_disk["cases"][0]
    assert case["n"] == 16
    assert case["solver"]["steps"] > 0
    # the run is scored against the committed table, not against itself
    ref_u = dict(table(REYNOLDS, "u"))
    assert case["u"]["extremum"]["min_reference"][1] == pytest.approx(
        ref_u[case["u"]["extremum"]["min_reference"][0]])
    # every criterion is reported individually
    assert [c["id"] for c in case["verdict"]["checks"]] == [c["id"] for c in CRITERIA]


def test_solver_run_is_deterministic(tmp_path):
    """Two identical short runs agree bit for bit (no hidden state, no RNG)."""
    a, _ = run_case(16, out_dir=tmp_path / "a", tolerance=1e-3, max_steps=120,
                    check_every=40, threads=1)
    b, _ = run_case(16, out_dir=tmp_path / "b", tolerance=1e-3, max_steps=120,
                    check_every=40, threads=1)
    assert a["u"]["max_abs_error"] == pytest.approx(b["u"]["max_abs_error"], abs=0)
    assert a["v"]["max_abs_error"] == pytest.approx(b["v"]["max_abs_error"], abs=0)


# ------------------------------------------------------------ bundle layer
def test_bundle_is_the_anchor_case():
    bundle = _bundle()
    assert bundle["case"] == CASE_SLUG
    assert bundle["reynolds"] == REYNOLDS
    assert bundle["tolerance_fraction"] == TOL
    assert "Ghia" in bundle["reference"]
    assert bundle["cases"], "no cases in the bundle"


def test_every_reported_row_is_converged_and_divergence_free():
    for case in _bundle()["cases"]:
        assert case["solver"]["converged"] is True, f"n={case['n']} not converged"
        assert case["solver"]["residual_rate"] <= 1e-6
        assert case["max_divergence"] < 1e-10


def test_stored_verdict_is_reproducible_from_the_stored_numbers():
    """``verdict`` must follow from the metrics it reports -- recomputed here."""
    for case in _bundle()["cases"]:
        recomputed = judge(case)
        stored = case["verdict"]
        assert recomputed["pass"] is stored["pass"], f"n={case['n']} verdict drifted"
        for got, want in zip(recomputed["checks"], stored["checks"]):
            assert got["id"] == want["id"]
            assert got["pass"] is want["pass"]
            assert float(got["value"]) == pytest.approx(float(want["value"]), rel=1e-12)


def test_key_quantities_against_the_ghia_tables_inside_five_percent():
    """The anchor's acceptance: u_min, v_max, vortex centre, both profiles < 5 %."""
    bundle = _bundle()
    ref_u = dict(table(REYNOLDS, "u"))
    ref_v = dict(table(REYNOLDS, "v"))
    gx, gy = VORTEX_CENTRE[REYNOLDS]
    production = [c for c in bundle["cases"]
                  if not c.get("role", "").startswith("control")]
    assert production, "no production-resolution row in the bundle"
    for case in production:
        y_ref = case["u"]["extremum"]["min_reference"][0]
        x_ref = case["v"]["extremum"]["max_reference"][0]
        # the stored "reference" values must be the committed transcription
        assert case["u"]["extremum"]["min_reference"][1] == pytest.approx(ref_u[y_ref])
        assert case["v"]["extremum"]["max_reference"][1] == pytest.approx(ref_v[x_ref])
        assert abs(case["u"]["extremum"]["min_computed"][1] - ref_u[y_ref]) / abs(ref_u[y_ref]) < TOL
        assert abs(case["v"]["extremum"]["max_computed"][1] - ref_v[x_ref]) / abs(ref_v[x_ref]) < TOL
        vx, vy = case["vortex_centre"]["computed"]
        assert max(abs(vx - gx), abs(vy - gy)) < TOL
        assert case["u"]["max_abs_error"] < TOL
        assert case["v"]["max_abs_error"] < TOL
        assert case["verdict"]["pass"] is True


def test_control_rows_stay_reported_and_do_not_pass():
    """An under-resolved grid must not be allowed to hide inside the bundle."""
    for case in _bundle()["cases"]:
        if not case.get("role", "").startswith("control"):
            continue
        assert case["verdict"]["pass"] is False


def test_two_resolutions_with_the_finer_one_not_worse():
    """Section 5 of the report spec: at least two resolutions, refinement helps."""
    bundle = _bundle()
    production = sorted((c for c in bundle["cases"]
                         if not c.get("role", "").startswith("control")),
                        key=lambda c: c["n"])
    assert len(production) >= 2, "need at least two production resolutions"
    coarse, fine = production[0], production[-1]
    assert fine["n"] > coarse["n"]
    assert fine["u"]["max_abs_error"] <= coarse["u"]["max_abs_error"] + 1e-12
    assert fine["v"]["max_abs_error"] <= coarse["v"]["max_abs_error"] + 1e-12


def test_continuation_check_reaches_the_same_steady_state():
    """Grid continuation is a cost device: it must not move the steady state."""
    check = _bundle().get("continuation_check")
    if not check:
        pytest.skip("bundle has no continuation check")
    assert check["max_abs_du"] < 1e-6
    assert check["max_abs_dv"] < 1e-6
    assert check["same_steady_state"] is True
    assert check["continued_verdict_pass"] is True
    # the check must be a real second solve: it may not resume the cold run's
    # checkpoint, and it must write a different field file
    assert check["independent_of_cold_run"] is True
    assert check["continued_fields_file"] != check["cold_fields_file"]


# --------------------------------------------------------- integration seam
def test_ladder_consumes_the_run():
    """The repository's C2 cavity criterion reads this run (integration caller).

    ``verification.ladder.lid_cavity_verdict`` scans
    ``lid_cavity_Re<Re>_n<n>.json``; the row emitted by
    :func:`tensorfvm.case_cavity_re1000.ladder_row` must be shaped exactly so that
    the Re = 1000 criterion is evaluated on the finest committed grid.
    """
    path = os.path.join(LADDER_DIR, "lid_cavity_Re1000_n128.json")
    if not os.path.exists(path):
        pytest.skip(f"no ladder row at {path}")
    from tensorfvm.verification.ladder import lid_cavity_verdict

    verdicts = lid_cavity_verdict()
    cavity = [v for v in verdicts if v.cid == "C2"]
    assert cavity, "ladder has no C2 criterion"
    evaluated = cavity[0].detail.get("evaluated_on", "")
    # the ladder picks the finest Re = 1000 row available, i.e. the one this case wrote
    assert "Re1000_n128" in evaluated, evaluated
    # whatever the ladder concludes must follow mechanically from its own numbers
    # (its bar is 2 % of the lid speed, stricter than this case's 5 %) -- the point
    # of the integration test is that the ladder *reads* our row, not that it passes
    expected = "PASS" if float(cavity[0].measured) <= float(cavity[0].threshold) else "FAIL"
    assert cavity[0].verdict == expected, cavity[0].note
    # and the row on disk carries exactly the schema ladder_row produces
    row = json.load(open(path))
    assert set(ladder_row({"reynolds": 1000.0, "n": 128,
                           "solver": {"steps": 0, "converged": True,
                                      "residual_rate": 0.0, "wall_seconds": 0.0},
                           "max_divergence": 0.0, "u": {}, "v": {},
                           "vortex_centre": {"computed": [0, 0]}})) == set(row)