"""Gate for the grid-of-record artefacts of ``CASE-FORWARD-STEP``.

Pinned here: (1) the committed field dump is re-graded and must reproduce every
closed-form number in ``artifacts/grid_triple.json`` to 1e-9, so the verdict
cannot be hand-edited; (2) the two withdrawn criteria (``u2/u1``, ``M2``) are
still *reported* with status ``withdrawn``, only excluded from the verdict;
(3) the troubled-cell fallback restores a poisoned cell and leaves the embedded
wall alone (its conservation is an open defect, see DESIGN-ADDENDUM-r2.md).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from tensorfvm.compressible.forward_step import ForwardStepConfig, build_case
from tensorfvm.compressible.forward_step_benchmark import WITHDRAWN, grade_run

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "docs" / "repro-fvm-forward-step"
GRID_TRIPLE = ART / "artifacts" / "grid_triple.json"
RECORD_FIELDS = ART / "run" / "fields_480x160.npz"
CLOSED_FORM = ("p2/p1", "rho2/rho1", "u2/u1", "M2", "p_stag/p1", "rho_stag/rho1")


def test_troubled_cell_repair_still_restores_a_poisoned_cell():
    solver, cfg = build_case(ForwardStepConfig(nx=60, ny=20, t_end=0.1))
    for i, j in ((30, 6), (31, 6), (30, 7)):
        solver.U[0, i, j] = 0.5 * solver.rho_floor
        solver.U[3, i, j] = 1e-14
    solid_before = solver.U[:, cfg.i_step:, : cfg.j_step].clone()
    solver._enforce_positivity()
    assert solver.troubled_repairs >= 3
    assert bool(torch.isfinite(solver.U).all())
    assert torch.allclose(solver.U[:, cfg.i_step:, : cfg.j_step], solid_before)
    prim = solver.primitive()
    assert float(prim["rho"].min()) > 0.0 and float(prim["p"].min()) > 0.0


@pytest.mark.skipif(not (GRID_TRIPLE.exists() and RECORD_FIELDS.exists()),
                    reason="grid-of-record artefacts are not in this checkout")
def test_grid_of_record_artefact_reproduces_the_verdict():
    import numpy as np

    data = json.loads(GRID_TRIPLE.read_text())
    record = data["record"]
    assert record is not None, "the grid of record was never graded"
    with np.load(RECORD_FIELDS) as npz:
        prim = {k: torch.as_tensor(npz[k]).double() for k in ("rho", "u", "v", "p")}
    nx, ny = (int(v) for v in record["grid"])
    cfg = ForwardStepConfig(nx=nx, ny=ny, t_end=record["t_end"])
    again = grade_run(prim, cfg, {})
    for item in record["items"]:
        row = next(i for i in again["items"] if i["quantity"] == item["quantity"])
        assert row["status"] == item["status"], item["quantity"]
        if item["quantity"] in CLOSED_FORM:
            assert row["got"] == pytest.approx(item["got"], rel=1e-9), item["quantity"]
    assert set(data["withdrawn"]) == set(WITHDRAWN) == {"u2/u1", "M2"}
    assert record["verdict"]["withdrawn"] == ["M2", "u2/u1"]
    assert record["verdict"]["n_graded"] == record["verdict"]["n_graded_pass"]
