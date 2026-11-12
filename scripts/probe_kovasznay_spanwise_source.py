#!/usr/bin/env python3
"""Locate the O(h) spanwise (w) driver of the extruded Kovasznay FVM discretisation.

Debt being audited
------------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger, open) wants an observed L2 order
>= 1.8 at Re = 40.  The delivered round failed at 1.684; m16396 split the norm
and showed the failure is *entirely* carried by the spanwise component ``w``
(exact value identically zero), whose error converges at only **1.106**.  That
audit stopped at attribution-by-norm and explicitly recorded the open question:

    which *term* of the z-momentum discretisation produces the O(h) drive?

This script answers that question with a **term-by-term consistency probe**: it
does not re-solve the case, it evaluates the discrete momentum system at the
*exact* state and looks at the residual of every term and every unknown block
(cell dofs, face dofs) for each velocity component.  A discretisation that is
correct for ``w = 0`` must return an O(h^2) residual there, because no term of
the z-momentum equation is O(h) for the exact field.

Two kinds of probe are run:

``exact-state residual``
    state = exact cell values (at cell centres) + exact face values (at face
    centres); p = exact pressure.  Residual per block/component, and the split
    ``viscous`` / ``convection`` / ``pressure``.
``annihilation probes``
    minimal states that any consistent discretisation must annihilate exactly:
    (a) uniform velocity (free-stream preservation), (b) pressure linear in one
    coordinate only (exact gradient recovery), (c) ``w = 0`` with in-plane exact
    velocity -- the case at hand.

Every number printed is computed here from the solver's own operators; nothing
is hand-copied.  ``--output`` writes ``summary.json`` for the report generator.

Usage (one grid per invocation keeps the cost of a level independent):

    PYTHONPATH=src python scripts/probe_kovasznay_spanwise_source.py \
        --output runs/kovasznay-spanwise/n16 --n 16
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.verification.native_hybrid_ns import kovasznay_fields, kovasznay_solver

COMPONENTS = ('u', 'v', 'w')


def _blocks(solver):
    """Row slices of the state vector: cell (3N) then face (3F), each u,v,w."""
    N, F = solver.N, solver.F
    blocks = {}
    for j, c in enumerate(COMPONENTS):
        blocks[f'cell-{c}'] = np.arange(j, 3 * N, 3)
        blocks[f'face-{c}'] = 3 * N + np.arange(j, 3 * F, 3)
    return blocks


def exact_state(solver, mesh, fields):
    """Exact cell and face values (pointwise at centres) with prescribed entries."""
    exact_cell = fields(mesh.cell_centers)['velocity']
    exact_face = fields(mesh.face_centers)['velocity']
    state = np.empty(3 * (solver.N + solver.F))
    state[:3 * solver.N] = exact_cell.ravel()
    state[3 * solver.N:] = exact_face.ravel()
    # Prescribed entries must carry the solver's own (area-mean) data.
    prescribed = np.ravel(solver.prescribed)
    state[prescribed != 0] = prescribed[prescribed != 0]
    return state, exact_cell, exact_face


def exact_pressure(solver, mesh, fields):
    p = fields(mesh.cell_centers)['pressure'].copy()
    if solver.gauge:
        p = p - p[0] + solver.gauge_value
    return p


def residual_terms(solver, state, p):
    """Split the momentum residual into viscous / convection / pressure terms."""
    conv, source = solver.convection.hybrid_system(
        solver.volume_flux(state), solver.face_mean.matrix, solver.velocity_gradient(state))
    visc = solver.viscous.full_matrix @ state
    pressure = -solver.divergence.T @ p
    load = -solver.load
    return dict(viscous=visc, convection=conv @ state + source, pressure=pressure,
                load=load, total=visc + conv @ state + source + pressure + load)


def block_norms(solver, residual):
    """L2 norm of the residual restricted to each (cell/face, component) block."""
    blocks = _blocks(solver)
    return {name: float(np.linalg.norm(residual[idx])) for name, idx in blocks.items()}


def free_stream_probe(solver, mesh):
    """Uniform velocity must have zero momentum residual everywhere it is free.

    Any nonzero residual is an inconsistency in the z-momentum (or tangential
    face-momentum) path that does not vanish as h -> 0 for the exact field.
    """
    out = {}
    for label, value in (('u=1', np.array([1., 0., 0.])),
                         ('v=1', np.array([0., 1., 0.])),
                         ('w=1', np.array([0., 0., 1.]))):
        state = np.concatenate((np.tile(value, solver.N), np.tile(value, solver.F)))
        # keep prescribed entries: a uniform field satisfies Dirichlet data only
        # for its own component; zero the others to avoid mixing probes.
        free = np.ravel(solver.prescribed) == 0
        state = np.where(free, state, 0.)
        p = np.zeros(solver.N)
        residual = residual_terms(solver, state, p)['total']
        nm = block_norms(solver, residual)
        out[label] = {k: v for k, v in nm.items() if v > 1e-14}
    return out


def pressure_linear_probe(solver, mesh):
    """p = x (resp. z) must be recovered exactly by the discrete traction.

    Interior cell rows must see a unit pressure gradient and nothing else; a
    z-linear pressure must produce *no* in-plane traction and exactly unit
    z-traction.  Any z-traction leak from an x-linear pressure (or vice versa)
    is a cross-coupling defect of the extruded mesh metric.
    """
    out = {}
    for axis, name in ((0, 'p=x'), (1, 'p=y'), (2, 'p=z')):
        p = mesh.cell_centers[:, axis] - mesh.cell_centers[0, axis]
        traction = -solver.divergence.T @ p
        # per-block mean traction magnitude (a unit gradient means |t| = 1 per vol)
        cell = np.zeros(3 * solver.N)
        for j, c in enumerate(COMPONENTS):
            cell[j::3] = traction[:3 * solver.N][j::3]
        per_comp = {c: float(np.max(np.abs(cell[j::3] - (mesh.cell_volumes if j == axis else 0.))))
                    for j, c in enumerate(COMPONENTS)}
        face = np.zeros(3 * solver.F)
        for j in range(3):
            face[j::3] = traction[3 * solver.N:][j::3]
        per_comp_face = {c: float(np.max(np.abs(face[j::3])))
                         for j, c in enumerate(COMPONENTS)}
        out[name] = dict(cell_max_abs_error_vs_exact=per_comp,
                         face_max_abs_traction=per_comp_face)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--reynolds', type=float, default=40.)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    mesh, solver, _ = kovasznay_solver(args.n, args.reynolds, symmetry_extrusion=True,
                                      coupled_linear_method='reused_lu')
    fields = lambda pts: kovasznay_fields(pts, args.reynolds)
    state, exact_cell, exact_face = exact_state(solver, mesh, fields)
    p = exact_pressure(solver, mesh, fields)

    terms = residual_terms(solver, state, p)
    summary = dict(
        n=int(args.n), reynolds=float(args.reynolds), cells=int(solver.N), faces=int(solver.F),
        h=1.0 / args.n,
        exact_state=dict(
            total=block_norms(solver, terms['total']),
            viscous=block_norms(solver, terms['viscous']),
            convection=block_norms(solver, terms['convection']),
            pressure=block_norms(solver, terms['pressure']),
        ),
        free_stream=free_stream_probe(solver, mesh),
        pressure_linear=pressure_linear_probe(solver, mesh),
        max_exact_normwise_w=float(np.max(np.abs(exact_face[:, 2].ravel()))),
        seconds=time.time() - t0,
    )
    payload = json.dumps(summary, indent=2, default=float)
    print(payload)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'summary.json').write_text(payload + '\n')


if __name__ == '__main__':
    main()