#!/usr/bin/env python3
"""Controlled variants of the extruded Kovasznay case used to locate the O(h) w drive.

Built on top of ``scripts/probe_kovasznay_spanwise_source.py`` (which shows the
exact state satisfies every *cell* momentum row, so the spanwise error cannot be
a truncation error of the z-momentum equation).  The remaining candidate is the
**hybrid face closure on the extruded mesh**, and the three variants below
separate it cleanly:

``symmetry``   delivered setup: ``n x n x 1`` extrusion, z-planes free in the
               tangential (u, v) components -- a symmetry plane, so those face
               values are *independent momentum unknowns*.
``zdiri``      same mesh, but the z-plane *tangential* face values are
               prescribed to the exact area means (Dirichlet everywhere).  This
               removes the free tangential unknowns without touching the
               discretisation: if the O(h) w survives, the free tangential
               dofs are not the cause.
``nz<K>``      ``n x n x K`` extrusion with symmetry planes: the same symmetry
               treatment on a mesh that actually resolves the z direction.  If
               the w order recovers as K grows, the defect is a thin-layer
               artefact of the single-cell extrusion, not a symmetry BC error.
``planar``     ``n x n x n`` cube, all faces Dirichlet (the D10 report's
               ``planar`` control, unresolved there); w must be ~0.

Each variant solves the case, then reports the same three error budgets as the
D10 audit (in-plane, spanwise, total) plus the location of the spanwise error.

Usage:

    PYTHONPATH=src python scripts/probe_kovasznay_extrusion_variants.py \
        --variant symmetry --n 8 --output runs/kovasznay-spanwise/sym-n8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.native_hybrid_ns3d import NativeHybridNavierStokes3D
from tensorfvm.native_manufactured3d import exact_boundary_velocity_means
from tensorfvm.verification.native_hybrid_flow import channel_mesh
from tensorfvm.verification.native_hybrid_ns import kovasznay_fields


def build(n, reynolds, variant):
    if variant == 'planar':
        divisions = (n, n, n)
    elif variant.startswith('nz'):
        divisions = (n, n, int(variant[2:]))
    else:
        divisions = (n, n, 1)
    mesh = channel_mesh(divisions, triangulate_faces=True)
    fields = lambda pts: kovasznay_fields(pts, reynolds)
    exact = fields(mesh.cell_centers)
    boundary = exact_boundary_velocity_means(mesh, fields)
    fixed = np.repeat((mesh.neighbor < 0)[:, None], 3, axis=1)
    if variant in ('symmetry', 'planar') or variant.startswith('nz'):
        z = mesh.face_centers[:, 2]
        planes = (mesh.neighbor < 0) & ((z < 1e-10) | (z > 1 - 1e-10))
        if variant != 'planar':
            fixed[planes, :2] = False          # symmetry: tangential free
    solver = NativeHybridNavierStokes3D(mesh, fixed, boundary, viscosity=1. / reynolds,
                                       body_force=0., gauge_value=exact['pressure'][0],
                                       convection_order=2, coupled_linear_method='reused_lu')
    return mesh, solver, exact


def budgets(solver, mesh, exact):
    N, F = solver.N, solver.F
    st = solver.state
    cell = st[:3 * N].reshape(N, 3)
    face = st[3 * N:].reshape(F, 3)
    v = mesh.cell_volumes
    ref = exact['velocity']
    eu, ev, ew = (cell - ref).T
    l2 = lambda e: float(np.sqrt(np.sum(v * e ** 2) / np.sum(v)))
    span_share = float(np.sum(v * ew ** 2) / max(np.sum(v * (eu ** 2 + ev ** 2 + ew ** 2)), 1e-300))
    biggest = np.argsort(-np.abs(ew))[:3]
    zf = np.flatnonzero((mesh.neighbor < 0) &
                        ((mesh.face_centers[:, 2] < 1e-10) | (mesh.face_centers[:, 2] > 1 - 1e-10)))
    return dict(
        inplane_l2=l2(np.hypot(eu, ev)), spanwise_l2=l2(ew), max_abs_w=float(np.abs(ew).max()),
        face_max_abs_w=float(np.abs(face[:, 2]).max()),
        span_share_of_squared_error=span_share,
        worst_cell=dict(center=mesh.cell_centers[int(biggest[0])].tolist(),
                        w=float(ew[biggest[0]])),
        zplane_tangential_face_range=(
            [float(face[zf, 0].min()), float(face[zf, 0].max())] if len(zf) else None),
        zplane_tangential_exact_range=(
            [float(ref[mesh.owner[zf], 0].min()), float(ref[mesh.owner[zf], 0].max())]
            if len(zf) else None),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--reynolds', type=float, default=40.)
    ap.add_argument('--output', type=Path, default=None)
    args = ap.parse_args()

    t0 = time.time()
    mesh, solver, exact = build(args.n, args.reynolds, args.variant)
    result = solver.solve(max_outer_iterations=80, tolerance=1e-8, relaxation=.7)
    summary = dict(variant=args.variant, n=int(args.n), reynolds=float(args.reynolds),
                   cells=int(solver.N), faces=int(solver.F),
                   divisions=[int(np.rint(a)) for a in (mesh.cell_centers.max(0) + .5)],
                   converged=bool(result['numerically_converged']),
                   outer_iterations=len(solver.outer_history),
                   budgets=budgets(solver, mesh, exact),
                   seconds=time.time() - t0)
    payload = json.dumps(summary, indent=2, default=float)
    print(payload)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'summary.json').write_text(payload + '\n')


if __name__ == '__main__':
    main()