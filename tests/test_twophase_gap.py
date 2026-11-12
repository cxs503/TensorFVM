"""Tests of the ``GAP-FVM-TWOPHASE`` acceptance face.

The tests are split so that the fast ones run on every CI pass (coarse
resolution, quarter revolution) while the gate semantics -- gate evaluation is
a pure function of the committed leg artefacts -- is covered without any long
solve.
"""

from __future__ import annotations

import json

import pytest

from tensorfvm.twophase_gap import (
    DEFAULTS,
    GATES,
    evaluate_gates,
    run_campaign,
    run_dambreak_leg,
    run_zalesak_leg,
    summarise,
)

from tensorfvm.twophase_gap.zalesak_gate import convergence_order


def test_zalesak_leg_reports_shape_and_conservation():
    leg = run_zalesak_leg(32, "plic", revolutions=1.0)
    rec = leg["record"]
    assert rec["case"] == "zalesak"
    assert rec["cells"] == 32
    # one rigid rotation of a smooth shape must be conservative to round-off
    assert rec["mass_drift_relative"] < 1e-10
    assert abs(rec["bound_violation"]) <= 1e-12
    # a shape metric exists, is finite, and is not trivially zero
    assert 0.0 <= rec["geometric_error"] < 1.0
    assert rec["reference"]["geometric_error_exact"] == 0.0
    assert leg["field"].shape == (32, 32)
    assert leg["exact"].shape == (32, 32)


def test_cicsam_leg_is_conservative():
    leg = run_zalesak_leg(32, "cicsam", revolutions=1.0)
    rec = leg["record"]
    assert rec["scheme"] == "cicsam"
    assert rec["mass_drift_relative"] < 1e-10
    # algebraic transport keeps alpha bounded
    assert abs(rec["bound_violation"]) <= 1e-12


def test_dambreak_leg_matches_martin_moyce_and_control():
    leg = run_dambreak_leg(48, 16, t_star_end=1.5, control_steps=5)
    rec = leg["record"]
    assert not rec["diverged"]
    assert rec["volume_drift"] < 1e-3
    # The comparison against the experiment is a known honest FAIL of this
    # capability (see docs/GAP-FVM-TWOPHASE/README.md); the leg must still
    # *report* finite numbers and stay conservative.
    assert 0.0 < rec["front_max_rel_error"] < 2.0
    assert 0.0 < rec["height_max_rel_error"] < 2.0
    assert rec["front_error_convention"] == "half-cell"
    # zero-gravity negative control: the interface must be frozen
    assert rec["control_max_abs_change"] <= 1e-14


def test_convergence_order_definition():
    assert convergence_order(0.4, 0.1, 2.0) == pytest.approx(2.0, abs=1e-12)
    assert convergence_order(0.0, 0.0, 2.0) == 0.0


def test_gate_evaluation_is_a_function_of_artefacts(tmp_path):
    """A hand-written pair of legs must produce the verdict without solving."""
    plic64 = run_zalesak_leg(32, "plic", revolutions=1.0)["record"]
    plic64 = dict(plic64, cells=64, declared=False, geometric_error=0.09)
    plic128 = dict(plic64, cells=128, declared=True, geometric_error=0.04)
    legs = {"zalesak_plic_64": plic64, "zalesak_plic_128": plic128}
    gates, metrics = evaluate_gates(legs)
    by_id = {g["id"]: g for g in gates}
    assert by_id["Z1"]["passed"] is True            # 4 % <= 5 %
    assert metrics["zalesak_convergence_order"] == pytest.approx(
        convergence_order(0.09, 0.04, 2.0))
    assert by_id["Z5"]["got"] >= 0.8
    # dam-break gates are reported as not_run, never as silently passed
    assert by_id["D1"]["status"] == "not_run"
    assert by_id["D1"]["passed"] is False

    plic128_fail = dict(plic128, geometric_error=0.07)
    gates, _ = evaluate_gates({"zalesak_plic_128": plic128_fail})
    assert {g["id"]: g for g in gates}["Z1"]["passed"] is False


def test_gate_table_and_defaults_are_consistent():
    ids = [g[0] for g in GATES]
    # every gate row is (id, case, metric, op, limit, description)
    # and the ids are unique
    assert len(ids) == len(set(ids))
    assert all(len(g) == 6 for g in GATES)
    # the declared faces must be present verbatim (Zalesak, dam break, Hysing)
    assert {"Z1", "Z2", "Z3", "Z4", "Z5",
            "D1", "D2", "D3", "D4", "D5"} <= set(ids)
    assert {"B1", "B2", "B3", "B4"} <= set(ids)
    assert 64 in DEFAULTS["zalesak_cells"]
    assert 128 in DEFAULTS["zalesak_cells"]


def test_campaign_writes_resumable_artefacts(tmp_path):
    out = tmp_path / "gap"
    summary = run_campaign(
        out, case="zalesak", zalesak_cells=(32,), zalesak_schemes=("plic",),
        revolutions=1.0,
    )
    assert (out / "gap_twophase.json").exists()
    assert (out / "GAP-FVM-TWOPHASE.md").exists()
    record = json.loads((out / "leg_zalesak_plic_32.json").read_text())
    assert record["case"] == "zalesak"
    # summarise() must reproduce the same verdict from disk alone
    again = summarise(out)
    assert again["all_passed"] == summary["all_passed"]
    assert again["metrics"] == summary["metrics"]


def test_solver_self_check_uses_installed_scheme():
    from tensorfvm.vof_phase0.solver import Phase0TwoPhaseSolver

    solver = Phase0TwoPhaseSolver(nx=16, ny=16, length=0.2, height=0.2)
    record = solver.verify_interface_transport(cells=32, revolutions=1.0)
    assert record["solver"]["scheme_class"]
    assert record["graded"] is False          # 32 != declared 128
    assert record["passed"] is None
    assert record["field_shape"] == [32, 32]

def test_hysing_b4_uses_own_convergence_order_not_zalesak():
    """B4 must read the Hysing coarse->fine pair, never the Zalesak order.

    Regression 1 (m16xxx): ``_METRIC_KEY`` mapped ``convergence_order`` ->
    ``zalesak_convergence_order`` for ALL gates, so B4 silently reported the
    Zalesak value.  Regression 2 (m16901): B4 was then wired to an observed
    *order* compared with ``>= 0.05``, which any >= 3.5 % change satisfies.
    B4 is now the relative change of U_T with a ``<= 5 %`` criterion.
    """
    from tensorfvm.twophase_gap.campaign import evaluate_gates, _METRIC_KEY

    assert _METRIC_KEY["hysing_convergence_change"] == "hysing_convergence_change"
    legs = {
        "zalesak_plic_64": {
            "case": "zalesak", "scheme": "plic", "cells": 64, "declared": False,
            "geometric_error": 0.10, "mass_drift_relative": 1e-14,
            "perimeter_error": 0.12, "bound_violation": 0.0,
        },
        "zalesak_plic_128": {
            "case": "zalesak", "scheme": "plic", "cells": 128, "declared": True,
            "geometric_error": 0.04, "mass_drift_relative": 6e-16,
            "perimeter_error": 0.06, "bound_violation": 0.0,
        },
        "hysing_8": {
            "case": "hysing", "cells_per_diameter": 8, "declared": False,
            "terminal_velocity": 0.200, "terminal_velocity_rel_error": 0.15,
            "circularity_rel_error": 1.0, "worst_step_drift": 5e-7,
        },
        "hysing_16": {
            "case": "hysing", "cells_per_diameter": 16, "declared": True,
            "terminal_velocity": 0.250, "terminal_velocity_rel_error": 0.07,
            "circularity_rel_error": 0.9, "worst_step_drift": 1e-8,
        },
    }
    gates, metrics = evaluate_gates(legs)
    b4 = next(g for g in gates if g["id"] == "B4")
    # |0.250 - 0.200| / 0.250 = 0.20 -- the Hysing pair, not the Zalesak order
    assert b4["got"] == pytest.approx(0.20, rel=1e-9)
    assert b4["passed"] is False
    assert b4["comparator"] == "<="
    assert metrics["hysing_convergence_change"] == pytest.approx(0.20, rel=1e-9)
