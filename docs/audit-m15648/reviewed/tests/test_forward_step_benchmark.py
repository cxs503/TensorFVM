"""Tests for the CASE-FORWARD-STEP benchmark grading surface.

The invariants pinned here are the ones the benchmark verdict rests on:

1. a field *built from* the exact Rankine-Hugoniot / isentropic state grades
   green, so the grader cannot be failing on its own definitions;
2. a field that misses the jump grades red, so the grader is not vacuous;
3. the post-shock sample never lands in the cell that straddles the numerical
   transition (the defect that made the pre-repair measurement report a state
   between pre- and post-shock);
4. the troubled-cell repair restores a poisoned cell from its neighbours and
   leaves the embedded wall alone;
5. the grid-convergence helper recovers the order of a synthetic error series.
"""

from __future__ import annotations

import math

import pytest
import torch

from tensorfvm.compressible.forward_step import (
    ForwardStepConfig,
    _sample_behind,
    normal_shock_reference,
)
from tensorfvm.compressible.forward_step_benchmark import (
    GRADED_KEYS,
    TOL,
    convergence,
    grade_run,
    table_markdown,
)
from tensorfvm.compressible.eos import StiffenedGasMixture
from tensorfvm.compressible.solver import EulerSolver


def _exact_field(cfg: ForwardStepConfig, shock_x: float = 0.3,
                 flap: float = 1.0, pocket: bool = True):
    """A field that satisfies the exact normal-shock jump at ``shock_x``.

    ``flap`` scales the measured jummp (used to build a failing field);
    ``pocket`` fills the two columns before the step face with the wall
    stagnation state so the stagnation criterion has the right reference.
    """
    ref = normal_shock_reference(cfg.mach, cfg.gamma, cfg.rho, cfg.p)
    xc, _ = cfg.centers()
    rho = torch.full((cfg.nx, cfg.ny), cfg.rho, dtype=torch.float64)
    u = torch.full((cfg.nx, cfg.ny), cfg.u_in, dtype=torch.float64)
    p = torch.full((cfg.nx, cfg.ny), cfg.p, dtype=torch.float64)
    post = xc >= shock_x
    rho[post] = cfg.rho + flap * (ref["rho2"] - cfg.rho)
    p[post] = cfg.p + flap * (ref["p2"] - cfg.p)
    u[post] = cfg.u_in + flap * (ref["u2"] - cfg.u_in)
    if pocket:
        rho[cfg.i_step - 2:cfg.i_step, : cfg.j_step] = ref["rho02"]
        p[cfg.i_step - 2:cfg.i_step, : cfg.j_step] = ref["p02"]
        u[cfg.i_step - 2:cfg.i_step, : cfg.j_step] = 0.0
    v = torch.zeros_like(u)
    return {"rho": rho, "u": u, "v": v, "p": p}


def _meta(drift: float = 0.0):
    return {"nsteps": 1, "budget": {"mass_drift": drift, "energy_drift": drift}}


def test_exact_field_grades_green():
    cfg = ForwardStepConfig(nx=120, ny=40, t_end=4.0)
    grading = grade_run(_exact_field(cfg), cfg, _meta())
    assert grading["all_graded_pass"], table_markdown(grading)
    for item in grading["items"]:
        if item["tolerance"] == TOL:
            assert abs(item["rel_error"]) < 1e-6, item


def test_wrong_jump_grades_red():
    cfg = ForwardStepConfig(nx=120, ny=40, t_end=4.0)
    grading = grade_run(_exact_field(cfg, flap=0.5), cfg, _meta())
    graded = [i for i in grading["items"] if i["tolerance"] == TOL]
    assert not grading["all_graded_pass"]
    for label in ("p2/p1", "rho2/rho1", "u2/u1"):
        row = next(i for i in graded if i["quantity"] == label)
        assert not row["pass"]
        assert abs(row["rel_error"]) > 0.3  # a half-sized jump is 30-140 % off


def test_budget_identity_fails_when_drift_is_large():
    cfg = ForwardStepConfig(nx=60, ny=20, t_end=0.5)
    grading = grade_run(_exact_field(cfg), cfg, _meta(drift=1e-3))
    row = next(i for i in grading["items"] if i["quantity"] == "mass_budget_drift")
    assert not row["pass"]


def test_post_shock_sample_skips_the_transition_cell():
    """Coarse grid: the sampling window is narrower than one cell.

    The profile below is one pre-shock cell, one cell that straddles the jump,
    then the post-shock plateau.  The half-jump crossing is placed in the left
    half of the straddling cell, which is the case where a "centre >= crossing"
    rule picks the straddling cell itself and reports a state between pre- and
    post-shock (the defect the repair fixed).  The sample must instead be the
    first cell whose upstream face is behind the crossing.
    """
    cfg = ForwardStepConfig(nx=60, ny=20, t_end=0.5)
    xc, _ = cfg.centers()
    xvals = xc.tolist()
    ny = cfg.ny
    rho = torch.full((cfg.nx, ny), 5.4, dtype=torch.float64)
    rho[:10] = 1.4
    rho[10] = 3.6                      # straddling cell
    u = torch.full((cfg.nx, ny), 0.8, dtype=torch.float64)
    u[:10] = 3.0
    u[10] = 1.9
    p = torch.full((cfg.nx, ny), 10.3, dtype=torch.float64)
    p[:10] = 1.0
    p[10] = 5.7
    prof = {"rho": rho[:, 4], "u": u[:, 4], "v": torch.zeros(cfg.nx, dtype=torch.float64),
            "p": p[:, 4]}
    crossing = xvals[10] - 0.3 * cfg.dx   # left half of the straddling cell
    out = _sample_behind(prof, xvals, crossing, cfg, crossing + cfg.dx,
                         crossing + 0.5 * cfg.dx, ("rho", "p", "u"))
    old_pick = [k for k, x in enumerate(xvals) if x >= crossing][0]
    assert old_pick == 10                      # the straddling cell
    assert out["n_cells_behind"] == 1
    assert out["x_window"][0] == pytest.approx(xvals[11], abs=1e-12)
    assert out["rho_at_shock"] == pytest.approx(5.4, rel=1e-12)
    assert out["p_at_shock"] == pytest.approx(10.3, rel=1e-12)


def test_troubled_repair_restores_cell_and_skips_solid():
    cfg = ForwardStepConfig(nx=40, ny=20, t_end=0.1)
    eos = StiffenedGasMixture([cfg.gamma, cfg.gamma], names=["air", "air"])
    solver = EulerSolver(
        nx=cfg.nx, ny=cfg.ny, lx=cfg.lx, ly=cfg.ly, eos=eos,
        bc=(("inflow", {"rho": cfg.rho, "u": cfg.u_in, "v": 0.0, "p": cfg.p, "Y": 0.0}),
            "outflow", "reflective", "reflective"),
        flux_mode="hllc", order=2, cfl=cfg.cfl, solid=cfg.solid_mask(),
    )
    solver.set_uniform(cfg.rho, cfg.u_in, 0.0, cfg.p)
    solver.step(1e-4)
    good = solver.U[:, 5, 12].clone()
    solver.U[:, 6, 12] = torch.tensor([float("nan")] * 5, dtype=torch.float64)
    solver.U[:, 7, 12] = torch.tensor([-1.0, 0.0, 0.0, -1.0, 0.0], dtype=torch.float64)
    solid_before = solver.U[:, cfg.i_step:, : cfg.j_step].clone()
    solver._enforce_positivity()
    assert bool(torch.isfinite(solver.U).all())
    assert solver.troubled_repairs >= 2
    assert torch.allclose(solver.U[:, cfg.i_step:, : cfg.j_step], solid_before)
    # repaired cells carry usable state: positive density and pressure
    prim = solver.primitive()
    assert float(prim["rho"].min()) > 0.0
    assert float(prim["p"].min()) > 0.0
    assert solver.U[0, 6, 12] > 0.5 * good[0]


def test_convergence_recovers_first_order():
    ref = {"p_ratio": 10.333333333333334}
    grades = []
    for dx, factor in ((0.05, 0.2), (0.025, 0.1), (0.0125, 0.05)):
        err = factor  # halving dx halves the error -> order 1
        grades.append({
            "grid": [1, 1], "dx": dx,
            "items": [{"quantity": "p2/p1", "reference": ref["p_ratio"],
                       "got": ref["p_ratio"] * (1 + err), "rel_error": err,
                       "tolerance": TOL, "pass": False}],
        })
    # pad the item list so the index of p2/p1 matches GRADED_KEYS
    for g in grades:
        g["items"] = g["items"] + [dict(g["items"][0]) for _ in range(len(GRADED_KEYS) - 1)]
    conv = convergence(grades)["per_quantity"]["p2/p1"]
    assert conv["order"] == pytest.approx(1.0, abs=1e-9)
    assert abs(conv["extrapolated_error"]) < 1e-9