"""Tests for the self-similarity (``xi``-collapse) criterion of the dam break.

The criterion adds the axis the other CASE-FVM-DAMBREAK-STOKER checks do not
cover: Ritter's and Stoker's solutions are *self-similar*, so the numerical
profile must be the same curve in ``xi = (x - x_dam)/t`` at every time.

What is pinned here
-------------------
* the closed form is exactly self-similar (so the criterion has an exact anchor);
* the ``xi``-space error of the exact profile is zero and a 1 % uniform defect is
  read as 1 %;
* **the collapse residual alone is vacuous**: a wrong-but-self-similar profile
  (the exact fan of a +1 % ``h0``) collapses perfectly and is only caught by the
  ``xi`` error -- neither half of the pair can be dropped;
* the measured law ``E = C dx/(c0 t)``: first order in ``dx``, ``~1/t`` in time,
  CFL-insensitive, constant ``C`` between resolutions;
* the early-time bore reading is a function of travel in cells ``x_s/dx`` alone
  (the invariant behind AUDIT.md F5) while at a *common* ``t*`` it is grid
  dependent; and the committed artifact does not drift.
"""

from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorfvm.shallow_water import similarity as sim
from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET
from tensorfvm.shallow_water.exact import ritter_state, stoker_state

ARTIFACT = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorfvm"
            / "verification" / "artifacts" / "shallow-water-similarity"
            / "summary.json")

CASES = {"ritter-dry": RITTER_DRY, "stoker-wet": STOKER_WET}


def _closed_form(case, t: float):
    xi = sim.xi_grid(case)
    x = case.x_dam + xi * t
    if case.kind == "dry":
        h, _ = ritter_state(x, t, case.h0, case.gravity, case.x_dam)
    else:
        h, _ = stoker_state(x, t, case.h0, case.h1, case.gravity, case.x_dam)
    return h / case.h0


@pytest.mark.parametrize("key", list(CASES))
def test_closed_form_is_self_similar(key):
    """Evaluating the closed form at three times gives the same profile in xi."""
    case = CASES[key]
    ref = sim.exact_xi_profile(case)["h"]
    for t in (0.05, 0.3, 1.7):
        assert np.max(np.abs(_closed_form(case, t) - ref)) < 1e-12


@pytest.mark.parametrize("key", list(CASES))
def test_xi_error_of_exact_profile_is_zero(key):
    case = CASES[key]
    exact = sim.exact_xi_profile(case)
    assert sim.xi_error(exact, exact) == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("key", list(CASES))
def test_xi_error_reads_a_uniform_defect(key):
    """A +1 % depth error must read as 1 %, not as "small" or "huge"."""
    case = CASES[key]
    exact = sim.exact_xi_profile(case)
    scaled = {"xi": exact["xi"], "h": exact["h"] * 1.01}
    assert sim.xi_error(scaled, exact) == pytest.approx(0.01, rel=1e-9)


@pytest.mark.parametrize("key", list(CASES))
def test_collapse_alone_is_vacuous_for_a_self_similar_wrong_profile(key):
    """Non-vacuity: a self-similar but wrong profile passes the collapse check.

    The exact fan of ``h0 * 1.01`` is a genuine similarity solution -- of a
    different case.  Its profiles collapse perfectly (residual 0), so a
    collapse-only criterion would accept a 1 % wrong solution; the ``xi`` error
    sees it.  This is why the module ships the pair, not the collapse.
    """
    case = CASES[key]
    exact = sim.exact_xi_profile(case)
    scaled = {"xi": exact["xi"], "h": exact["h"] * 1.01}
    assert sim.collapse_residual([scaled], reference=scaled) < 1e-15
    assert sim.xi_error(scaled, exact) > 0.005


@pytest.mark.parametrize("key", list(CASES))
def test_collapse_detects_a_non_self_similar_defect(key):
    """A defect that is inconsistent between times is caught by the residual."""
    case = CASES[key]
    exact = sim.exact_xi_profile(case)
    xi = exact["xi"]
    lo, hi = sim.similarity_support(case)
    bump = np.exp(-0.5 * ((xi - 0.3 * (lo + hi)) / (0.05 * abs(hi))) ** 2)
    defected = {"xi": xi, "h": exact["h"] + 0.05 * bump}
    residual = sim.collapse_residual([exact, defected])
    assert residual > 1e-3
    # and the residual is of the same order as the injected local error
    assert residual == pytest.approx(sim.xi_error(defected, exact), rel=0.5)


@pytest.mark.parametrize("key", list(CASES))
def test_xi_error_is_first_order_in_dx(key):
    """Halving dx halves the xi-space error; the normalised constant is stable."""
    case = CASES[key]
    coarse = sim.similarity_row(case, 400, t_stars=(2.0,))
    fine = sim.similarity_row(case, 800, t_stars=(2.0,))
    e0 = coarse["rows"][-1]["xi_error"]
    e1 = fine["rows"][-1]["xi_error"]
    assert 1.7 < e0 / e1 < 2.3
    c0 = coarse["rows"][-1]["C"]
    c1 = fine["rows"][-1]["C"]
    assert abs(c0 - c1) / (0.5 * (c0 + c1)) < 0.05


def test_xi_error_decays_like_one_over_time():
    """E ~ 1/t: the transient error at t* = 0.05 is far above the final one."""
    case = STOKER_WET
    row = sim.similarity_row(case, 400, t_stars=(0.05, 2.0))
    early, late = row["rows"][0]["xi_error"], row["rows"][-1]["xi_error"]
    assert early > 10.0 * late


def test_xi_error_is_cfl_insensitive():
    """The temporal axis is not the error source at the declared cfl = 0.4."""
    case = STOKER_WET
    lo = sim.similarity_row(case, 400, cfl=0.2, t_stars=(2.0,))["xi_error_final"]
    mid = sim.similarity_row(case, 400, cfl=0.4, t_stars=(2.0,))["xi_error_final"]
    hi = sim.similarity_row(case, 400, cfl=0.8, t_stars=(2.0,))["xi_error_final"]
    assert abs(lo - mid) / mid < 0.05
    assert abs(hi - mid) / mid < 0.05


def test_bore_reading_is_a_function_of_travel_in_cells():
    """AUDIT.md F5: the early-time bore error depends on x_s/dx alone.

    The two ``(n, t*)`` couples below share ``x_s/dx`` by construction (n and t*
    both doubled), so a reading that is a pure function of that group must give
    the same percentage -- the 4.505 % of F5 is reproduced here, and it is a
    statement about the sample schedule, not about the physics.
    """
    study = sim.bore_reading_study(combos=[(400, 0.01), (800, 0.005)])
    rows = study["rows"]
    assert rows[0]["xs_over_dx"] == pytest.approx(rows[1]["xs_over_dx"], rel=1e-12)
    assert abs(rows[0]["error"] - rows[1]["error"]) < 1e-9
    # ``error`` is a fraction: 0.0450472 = the 4.505 % quoted by AUDIT.md F5
    assert rows[0]["error"] == pytest.approx(0.045047, abs=1e-5)


def test_bore_reading_is_grid_dependent_at_a_common_time():
    """The same reading at a *common* t* is not grid independent (F5 wording)."""
    study = sim.bore_reading_study(combos=[(400, 0.02), (800, 0.02)])
    e0, e1 = study["rows"][0]["error"], study["rows"][1]["error"]
    assert abs(e0 - e1) > 1e-2


def test_committed_artifact_does_not_drift():
    """The graded number in the artifact is reproducible on this commit."""
    if not ARTIFACT.exists():           # pragma: no cover - artifact is committed
        pytest.skip("artifact missing")
    payload = json.loads(ARTIFACT.read_text())
    committed = payload["study"]["cases"]["stoker-wet"]["final_xi_error"]["800"]
    row = sim.similarity_row(STOKER_WET, 800, t_stars=(2.0,))
    # ``rel=1e-3``, not machine precision: the artifact advances through the
    # sample times in one trajectory, this test takes one single step sequence
    # to t* = 2, and the two step histories differ at the 2e-4 level.
    assert row["xi_error_final"] == pytest.approx(committed, rel=1e-3)
    # the declared law constants are quoted in the artifact as measured values
    for key in CASES:
        block = payload["study"]["cases"][key]
        assert block["C_plateau"]["min"] < block["C_plateau"]["mean"] < block["C_plateau"]["max"]
        assert block["min_cells_for_gate"] == pytest.approx(
            block["C_plateau"]["mean"] / 0.05, rel=1e-12)


def test_support_matches_the_exact_wave_speeds():
    """The xi window is the exact support: [-c0, 2c0] dry, [-c0, U] wet."""
    c0 = math.sqrt(RITTER_DRY.gravity * RITTER_DRY.h0)
    lo, hi = sim.similarity_support(RITTER_DRY)
    assert (lo, hi) == pytest.approx((-c0, 2.0 * c0))
    grid = sim.xi_grid(RITTER_DRY)
    assert grid[0] < lo < hi < grid[-1]     # margin on both sides
    lo_w, hi_w = sim.similarity_support(STOKER_WET)
    from tensorfvm.shallow_water.exact import stoker_constant_state
    sol = stoker_constant_state(STOKER_WET.h0, STOKER_WET.h1, STOKER_WET.gravity)
    assert hi_w == pytest.approx(sol["shock_speed"])