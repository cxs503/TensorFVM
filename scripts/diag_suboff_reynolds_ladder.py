#!/usr/bin/env python3
"""Reynolds-continuation ladder for the native SIMPLE+SST SUBOFF path.

Runs momentum sweeps at increasing Reynolds number, restarting from the
previous level's state.  The point is to separate two questions:

* does the ledger-consistent SIMPLE iteration converge at all on this mesh;
* is the surviving growth confined to a few sliver cells at the aft tip.

Every level records the maximum speed, the speed growth, the L2 distance from
the freestream, and the count of cells above 1.2 U, so a localized hot spot is
visible separately from a global divergence.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_rans3d import NativeRansSettings, NativeSimpleSSTRansSolver3D
from tensorfvm.native_sst3d import inflow_from_intensity
from tensorfvm.native_projection3d import NativeProjection3D


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reynolds', type=float, nargs='+', default=[1e4, 1e5, 1e6, 1e7])
    parser.add_argument('--sweeps', type=int, default=25)
    parser.add_argument('--relaxation', type=float, default=.5)
    parser.add_argument('--pressure-relaxation', type=float, default=.1)
    parser.add_argument('--maximum-speed', type=float, default=20.)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty')
    args.output.mkdir(parents=True, exist_ok=True)
    mesh = NativeMesh3D.from_gmsh(args.mesh, triangulate_faces=True)
    fixed = (mesh.neighbor < 0) & (mesh.patch_labels != 'outlet')
    velocity = np.tile([1., 0., 0.], (len(mesh.owner), 1))
    velocity[mesh.patch_labels == 'wall'] = 0.
    k0, omega0 = inflow_from_intensity(1., .005, .05)
    summary = dict(mesh=str(args.mesh), sweeps=args.sweeps,
        relaxation=args.relaxation, pressure_relaxation=args.pressure_relaxation,
        levels=[])
    state = None
    for reynolds in args.reynolds:
        settings = NativeSteadySettings(viscosity=1 / reynolds, max_iterations=1,
            nonlinear_tolerance=1e-8, relaxation=args.relaxation,
            bounded_convection=True, convection_order=1,
            pressure_relaxation=args.pressure_relaxation)
        solver = NativeSimpleSSTRansSolver3D(mesh, fixed, velocity,
            pressure_dirichlet=mesh.patch_labels == 'outlet', settings=settings,
            rans_settings=NativeRansSettings(max_outer_iterations=1, wall_treatment='spalding'))
        if state is None:
            solver.u[:, 0] = 1.
            projection = NativeProjection3D(mesh, mesh.patch_labels == 'outlet')
            _, projected = projection.project(solver.velocity_flux_matrix @ solver.u.ravel() + solver.velocity_flux_source)
            solver.initialize_state(solver.u, solver.p, projected)
        else:
            solver.initialize_state(*state)
        solver.k = np.full(solver.N, k0); solver.omega = np.full(solver.N, omega0)
        solver.k_boundary = np.full(solver.F, k0); solver.omega_boundary = np.full(solver.F, omega0)
        solver._update_momentum_coefficients(solver.current_closure())
        level = dict(reynolds=reynolds, trajectory=[])
        stopped = None
        for sweep in range(args.sweeps):
            try:
                solver.solve()
            except Exception as error:  # noqa: BLE001 - archive every outcome
                stopped = f'{type(error).__name__}: {error}'
                break
            row = solver.history[-1]
            speed = np.linalg.norm(solver.u, axis=1)
            entry = dict(sweep=sweep + 1, max_speed=float(speed.max()),
                l2_deviation=float(np.linalg.norm(solver.u - np.array([1., 0., 0.]), axis=1).mean()),
                cells_above_1_2=float(np.mean(speed > 1.2)),
                velocity_change=row['velocity_change'], flux_change=row['flux_change'],
                momentum_residual=row['momentum_residual'],
                native_divergence_max=row['native_divergence_max'],
                face_flux_consistency_max=row['face_flux_consistency_max'],
                ledger_identity_defect_max=row['ledger_identity_defect_max'])
            level['trajectory'].append(entry)
            print(json.dumps(dict(reynolds=reynolds, **entry)), flush=True)
            if row['velocity_change'] < 1e-7:
                break
            if speed.max() > args.maximum_speed:
                stopped = 'speed guard'
                break
        converged = bool(level['trajectory']) and level['trajectory'][-1]['velocity_change'] < 1e-6
        level.update(stopped=stopped, converged=bool(converged))
        summary['levels'].append(level)
        state = (solver.u.copy(), solver.p.copy(), solver.flux.copy())
        (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
        if stopped: break
    print(json.dumps({level['reynolds']: level['converged'] for level in summary['levels']},
        allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
