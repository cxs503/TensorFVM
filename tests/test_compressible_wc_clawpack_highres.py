"""Regression gate for the Clawpack high-resolution cross-validation.

Guards the *defect* that hid the third-code reference for three rounds: PyClaw's
``ClawSolver.max_steps`` default (10000) silently truncates the Woodward-Colella
double blast at ``nx >= 3200`` (12799 steps needed), returning a partially
evolved solution with no exception.  ``wc_pyclaw_xval.py`` recorded that
truncation as "CFL rejection / Roe positivity failure at the collision" and
dropped the rung; the attribution is refuted here and the rung is recovered.

The stored-artifact tests never run a solver; the reproduction test replays the
*old* call site (one rung, ~6 s) and skips when Clawpack is absent.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

DOCDIR = Path(__file__).resolve().parents[1] / "docs" / "woodward-colella-double-blast"
PYCLAW_DIR = DOCDIR / "pyclaw"
HIGHRES = PYCLAW_DIR / "highres"
RESULTS = HIGHRES / "results.json"
PROFILES = HIGHRES / "profiles.npz"

pytestmark = pytest.mark.skipif(not RESULTS.exists(),
                                reason="highres artifacts not stored")

T_END = 0.038
DTOL = 1e-12


@pytest.fixture(scope="module")
def res():
    return json.loads(RESULTS.read_text())


@pytest.fixture(scope="module")
def prof():
    with np.load(PROFILES) as d:
        return {k: d[k] for k in d.keys()}


def test_every_rung_reached_end_time(res):
    """No stored rung may be a truncated run (the old silent failure mode)."""
    assert res["rows"], "no rungs stored"
    for row in res["rows"]:
        assert row["truncated"] is False
        assert abs(row["t"] - T_END) < DTOL, (
            f"nx={row['nx']} {row['flux']} stopped at t={row['t']}")


def test_rungs_are_physical(res):
    """Positivity and finiteness: the state is admissible everywhere."""
    for row in res["rows"]:
        assert row["finite"] is True
        assert row["rho_min"] > 0.0
        assert row["p_min"] > 0.0
        # closed tube: mass is conserved to the quadrature error
        assert abs(row["mass"] - 1.0) < 5e-4


def test_step_count_exceeds_library_default(res):
    """Regression: the fix must be exercised (steps > 10000 at nx>=3200)."""
    big = [r for r in res["rows"] if r["nx"] >= 3200]
    assert big, "no nx>=3200 rung stored"
    assert any(r["steps"] > 10000 for r in big), (
        "no rung took more than PyClaw's default max_steps=10000; the defect "
        "fix is not exercised by these artifacts")


def test_defect_is_recorded_with_root_cause(res):
    """Provenance: the errata must ship with the artifacts."""
    d = res["defect"]
    assert "max_steps" in d["root_cause"]
    assert "10000" in d["root_cause"]
    assert "REFUTED" in d["previous_attribution"]


def test_third_code_self_convergence_monotone(res):
    """The pseudo truth must improve with refinement, and its peak settle."""
    sc = res["analysis"]["self_convergence"]["roe"]
    assert sc, "no self-convergence rows"
    for row in sc:
        assert 0.0 < row["coarse_vs_fine_l1_percent"] < 5.0
        assert row["peak_rel_percent"] < 0.5


def test_graded_ladder_beats_ledger_tolerance(res):
    """The ledger needs <10 %; the owner's reject line is 15 %."""
    tf = {r["nx"]: r for r in res["analysis"]["tfvm_vs_truth"]["roe"]}
    assert 3200 in tf
    assert tf[3200]["peak_percent"] < 10.0
    assert tf[3200]["l1_percent"] < 10.0
    assert tf[3200]["peak_percent"] < 15.0
    assert tf[3200]["l1_percent"] < 15.0
    # ... and honestly: the finest rung is well inside 3 %
    assert tf[3200]["peak_percent"] < 3.0
    assert tf[3200]["l1_percent"] < 3.0
    # smooth functional: mass is the tightest measure
    assert tf[3200]["window_mass_percent"] < 0.5


def test_residual_is_reference_side(res):
    """The digitised curve is further from truth than the graded solver is."""
    tf = {r["nx"]: r for r in res["analysis"]["tfvm_vs_truth"]["roe"]}
    dig = res["analysis"]["vs_digitised"]["calibrated"]
    assert dig["l1_percent"] > tf[3200]["l1_percent"], (
        "reference floor is no longer above the solver's own error; the "
        "round-2 attribution would need revisiting")


def test_cross_code_matched_grid_agreement(res):
    """Two independent codes at the same grid: an absolute error bound."""
    cross = res["analysis"]["cross_code"]
    assert cross
    for row in cross:
        assert row["l1_percent"] < 2.0
        assert row["peak_rel_percent"] < 3.0


def test_profiles_are_stored_for_each_rung(res, prof):
    for row in res["rows"]:
        key = f"{row['nx']}_{row['flux']}"
        assert f"x_{key}" in prof and f"rho_{key}" in prof
        assert prof[f"x_{key}"].size == row["nx"]
        assert np.isfinite(prof[f"rho_{key}"]).all()


def test_truth_is_finer_than_the_digitised_production_grid(res):
    """The point of the exercise: out-resolve the printed curve's 3200 grid."""
    assert res["analysis"]["truth"]["roe"]["nx"] >= 6400


@pytest.mark.skipif(
    not (PYCLAW_DIR / "wc_clawpack_highres.py").exists(),
    reason="runner script missing")
def test_truncation_is_reproducible_and_no_longer_silent():
    """Replay the old call site: default max_steps truncates, no exception."""
    try:
        import clawpack.pyclaw  # noqa: F401
    except Exception:  # pragma: no cover - clawpack absent
        pytest.skip("clawpack not installed")
    sys.path.insert(0, str(PYCLAW_DIR))
    from wc_clawpack_highres import claw_run, decay_probe

    old = decay_probe(3200, max_steps=None)     # library default (10000)
    assert old["requested_max_steps"] == 10000
    assert old["t_reached"] < 0.9 * T_END, (
        "the truncation did not reproduce; the defect note needs updating")
    assert old["finite"] is True          # nothing pathological happened
    assert old["p_min"] > 0.0             # ... the state stayed admissible

    # The fixed call either reaches t_end or raises; it must never return a
    # truncated profile silently.  One rung only (~9 s) to keep the gate cheap.
    x, rho, p, u, info = claw_run(3200, "roe")
    assert info["truncated"] is False
    assert abs(info["t"] - T_END) < DTOL
    assert info["steps"] > 10000