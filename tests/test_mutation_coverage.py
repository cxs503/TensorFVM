"""Coverage tests derived from ``scripts/audit_mutation_survival.py``.

The audit applies one defect at a time to the production 2-D kernels and records
whether the suite reacts. Three mutants survived the pre-existing suite; this
module adds a criterion-based test for the one that could be turned into a
non-tautological check, and documents the two that could not within the scope of
this audit (see ``ACCEPTED_GAPS``). Nothing is weakened to make the record look
complete.

The Rhie--Chow criterion below is physical, not a snapshot: the damping term is
a diffusion on the collocated pressure field, so a cell whose pressure exceeds
its neighbours' mean must export mass. A sign flip reverses that response and is
detected regardless of mesh, tolerances or solution history.
"""
import json
from pathlib import Path

import pytest
import torch

from tensorfvm.body_fitted import BodyFittedSolver
from tensorfvm.solver import SolverConfig
from tensorfvm.verification.mutation_catalogue import CATALOGUE

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "src/tensorfvm/verification/artifacts/mutation_survival/mutation-survival.json"

#: Mutants this audit could not convert into a criterion-based test. Each entry
#: records what was attempted and what the remaining obstacle is; the audit
#: manifest keeps them visible instead of deleting them from the catalogue.
ACCEPTED_GAPS = {
    "CP-WEIGHT-SWAP": (
        "Attempted: affine field must be reproduced exactly at interior face centres of a "
        "stretched three-cell probe. The probe failed on the unmutated source, so the "
        "assumed owner/neighbour geometry of projected-linear weights is not yet "
        "established; the test was removed rather than tuned until it passed. Remaining "
        "work: compare against a face value whose exactness is independent of the weight "
        "convention (e.g. a solver-produced affine pressure on a stretched multi-element "
        "mesh)."
    ),
    "STAG-UPWIND": (
        "Attempted: face interpolation of an affine velocity field must be exact, which "
        "central averaging satisfies and upwinding violates. The staggered solver could "
        "not be instantiated from SolverConfig through the public constructor in this "
        "audit, so no probe was built. Remaining work: drive the staggered solver through "
        "its documented entry point and compare cell-centre velocities against an affine "
        "field."
    ),
}


def test_rhie_chow_damping_is_diffusive_in_the_pressure_field():
    """Mass must leave a cell whose pressure exceeds its neighbours' mean."""
    solver = BodyFittedSolver(SolverConfig(mesh_type="body-fitted", nx=24, ny=8))
    count = solver.p.numel()
    velocity = torch.zeros(count, 2, dtype=solver.p.dtype)
    velocity[:, 0] = 1.0
    pressure = torch.randn(count, dtype=solver.p.dtype,
                           generator=torch.Generator().manual_seed(20261103))
    pressure -= pressure.mean()
    damping = torch.ones(count, dtype=solver.p.dtype)

    flux, _ = solver._rhie_chow(velocity, pressure, damping)
    owner, neighbour = solver.o[solver.f], solver.ni
    divergence = torch.zeros(count, dtype=solver.p.dtype)
    divergence.index_add_(0, owner, flux[solver.f])
    divergence.index_add_(0, neighbour, -flux[solver.f])

    total = torch.zeros(count, dtype=solver.p.dtype)
    tally = torch.zeros(count, dtype=solver.p.dtype)
    total.index_add_(0, owner, pressure[neighbour])
    tally.index_add_(0, owner, torch.ones_like(total[owner]))
    total.index_add_(0, neighbour, pressure[owner])
    tally.index_add_(0, neighbour, torch.ones_like(total[neighbour]))
    boundary_cells = torch.zeros(count, dtype=torch.bool)
    exterior = torch.ones(len(solver.o), dtype=torch.bool)
    exterior[solver.f] = False
    boundary_cells[solver.o[exterior]] = True
    cells = (tally > 0) & ~boundary_cells
    assert int(cells.sum()) > 10, "probe needs interior cells away from every boundary"

    excess = (pressure - total / tally)[cells]
    response = divergence[cells]
    correlation = float((excess * response).sum() / (excess.norm() * response.norm()))
    agree = float(((excess * response) > 0).to(torch.float64).mean())
    # Skewed O-grid cells adjacent to the wall tangent reduce the per-cell fraction
    # (measured 0.861 on the unmutated source); a sign flip drives it to ~0.14.
    assert correlation > 0.5, f"damping response anti-correlates with local excess: r={correlation:.3f}"
    assert agree > 0.80, f"only {agree:.1%} of cells export mass above their neighbours' mean"


def test_catalogue_anchors_are_unique_and_survivors_are_declared():
    """Keep the audit anchored: unique anchors, and every survivor declared."""
    for entry in CATALOGUE:
        source = (REPO / entry["file"]).read_text()
        assert source.count(entry["find"]) == 1, f"{entry['id']}: anchor is not unique"
        mutated = source.replace(entry["find"], entry["replace"])
        assert mutated != source, f"{entry['id']}: mutation is a no-op"
        compile(mutated, entry["file"], "exec")
    if not MANIFEST.exists():
        pytest.skip("mutation-survival manifest not generated in this checkout")
    manifest = json.loads(MANIFEST.read_text())
    records = {record["id"]: record for record in manifest["mutants"]}
    assert set(records) == {entry["id"] for entry in CATALOGUE}
    survivors = {key for key, record in records.items() if record["status"] != "killed"}
    undeclared = survivors - set(ACCEPTED_GAPS)
    assert not undeclared, f"undocumented surviving mutants: {sorted(undeclared)}"
    assert all(records[key]["status"] == "survived" for key in ACCEPTED_GAPS)
    assert all(ACCEPTED_GAPS[key].strip() for key in ACCEPTED_GAPS)
    assert all(record["source_restored"] for record in manifest["mutants"])