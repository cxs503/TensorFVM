#!/usr/bin/env python3
"""Freeze a native SST-2003 SUBOFF run; physical qualification stays explicit."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_rans3d import NativeRansSettings, NativeSSTRansSolver3D
from tensorfvm.native_sst3d import inflow_from_intensity, VARIANT
from run_native_suboff_benchmark import write, digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reynolds', type=float, required=True)
    parser.add_argument('--intensity', type=float, default=.005)
    parser.add_argument('--length-scale', type=float, default=.05)
    parser.add_argument('--outer-iterations', type=int, default=100)
    parser.add_argument('--inner-iterations', type=int, default=80)
    parser.add_argument('--pseudo-dt', type=float, default=1.)
    parser.add_argument('--linear-method', choices=['direct', 'iterative', 'block', 'schur'], default='direct')
    parser.add_argument('--convection-order', type=int, choices=[1, 2], default=2)
    parser.add_argument('--relaxation', type=float, default=.6)
    parser.add_argument('--lu-ordering', choices=['COLAMD', 'MMD_ATA', 'MMD_AT_PLUS_A', 'NATURAL'], default='COLAMD')
    parser.add_argument('--initial-velocity', choices=['freestream', 'rest'], default='freestream')
    args = parser.parse_args()
    if not np.isfinite(args.reynolds) or args.reynolds <= 0:
        parser.error('positive finite Reynolds number required')
    out = args.output
    if out.exists() and any(out.iterdir()):
        parser.error('output must be empty to preserve evidence')
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    repo = Path(__file__).resolve().parents[1]
    shutil.copy2(args.mesh, out / 'input.msh')
    sources = {Path(__file__).resolve(), Path(__file__).with_name('run_native_suboff_benchmark.py')} | {
        Path(m.__file__).resolve() for name, m in list(sys.modules.items())
        if (name == 'tensorfvm' or name.startswith('tensorfvm.')) and getattr(m, '__file__', None)
        and Path(m.__file__).suffix == '.py'}
    bindings = {}
    for source in sorted(sources):
        relative = source.relative_to(repo); target = out / 'source-snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target)
        bindings[str(relative)] = digest(target)
    settings = NativeSteadySettings(viscosity=1 / args.reynolds,
        max_iterations=args.inner_iterations, nonlinear_tolerance=1e-8,
        relaxation=args.relaxation, linear_method=args.linear_method, bounded_convection=True,
        convection_order=args.convection_order, lu_ordering=args.lu_ordering)
    rans_settings = NativeRansSettings(max_outer_iterations=args.outer_iterations,
        pseudo_time_step=args.pseudo_dt, nonlinear_tolerance=1e-7)
    k0, omega0 = inflow_from_intensity(1., args.intensity, args.length_scale)
    write(out / 'config.json', dict(kind='native SST SUBOFF diagnostic',
        variant=VARIANT, dimensionless_units=dict(L=1., U=1., rho=1., Re_L=args.reynolds),
        momentum_settings=asdict(settings), rans_settings=asdict(rans_settings),
        initial_velocity=args.initial_velocity, outlet_backflow_velocity=[1., 0., 0.],
        inlet=dict(intensity=args.intensity, length_scale_L=args.length_scale,
            k=float(k0), omega=float(omega0), provenance='explicit modeling assumptions, not measured experimental inflow'),
        trip='fully turbulent low-Re SST; no explicit experimental trip geometry or transition model',
        physical_validation_passed=False, source_sha256=bindings, mesh_sha256=digest(out / 'input.msh'),
        environment=dict(python=sys.version, executable=sys.executable, platform=platform.platform(), device='CPU scipy; GPU untested')))
    mesh = NativeMesh3D.from_gmsh(out / 'input.msh', triangulate_faces=True)
    fixed = (mesh.neighbor < 0) & (mesh.patch_labels != 'outlet')
    velocity = np.tile([1., 0., 0.], (len(mesh.owner), 1)); velocity[mesh.patch_labels == 'wall'] = 0.
    solver = NativeSSTRansSolver3D(mesh, fixed, velocity,
        pressure_dirichlet=mesh.patch_labels == 'outlet', settings=settings, rans_settings=rans_settings)
    if args.initial_velocity == 'freestream':solver.u[:, 0] = 1.
    print(f'OPERATOR_READY cells={len(mesh.cells)} elapsed={time.perf_counter()-start:.3f}', flush=True)
    history = []; momentum_history = []
    def momentum_checkpoint(row):
        momentum_history.append(row)
        with (out / 'momentum-history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        print(json.dumps(dict(stage='momentum', **row), allow_nan=False), flush=True)
    def checkpoint(row):
        history.append(row)
        with (out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        np.savez_compressed(out / 'current-state.npz', velocity=solver.u,
            pressure=solver.p - 2 / 3 * solver.k, modified_pressure=solver.p,
            k=solver.k, omega=solver.omega, flux=solver.flux)
        print(json.dumps(row, allow_nan=False), flush=True)
    failure = None
    try:
        result = solver.solve_rans(k0, omega0, inlet_k=k0, inlet_omega=omega0,
            callback=checkpoint, momentum_callback=momentum_checkpoint)
        converged = result['numerically_converged']
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error)); converged = False
    forces = solver.forces()
    wall = mesh.patch_labels == 'wall'
    area = np.linalg.norm(mesh.area_vectors[wall], axis=1)
    normal = mesh.area_vectors[wall] / area[:, None]
    traction = forces['anisotropic_stress_traction'][wall] / area[:, None]
    tangent = traction - normal * np.sum(traction * normal, axis=1)[:, None]
    friction_velocity = np.sqrt(np.linalg.norm(tangent, axis=1))
    yplus = friction_velocity * solver.wall_distance[mesh.owner[wall]] * args.reynolds
    np.savez_compressed(out / 'fields.npz', velocity=solver.u,
        pressure=solver.p - 2 / 3 * solver.k, modified_pressure=solver.p,
        k=solver.k, omega=solver.omega, flux=solver.flux,
        pressure_faces=forces['physical_pressure_faces'],
        anisotropic_stress_traction=forces['anisotropic_stress_traction'],
        isotropic_reynolds_stress_traction=forces['isotropic_reynolds_stress_traction'],
        wall_distance=solver.wall_distance, wall_yplus=yplus, wall_face_indices=np.flatnonzero(wall))
    summary = dict(cells=len(mesh.cells), Re_L=args.reynolds, numerically_converged=converged,
        physical_validation_passed=False, spatial_convergence_qualified=False,
        solution_state='coupled_iterate' if history else 'momentum_only' if momentum_history else 'initialization_only',
        completed_momentum_iterations=len(momentum_history),
        low_Re_wall_resolution=dict(maximum_yplus=float(yplus.max()),
            area_weighted_mean_yplus=float(np.sum(yplus * area) / area.sum()),
            maximum_yplus_below_one=bool(yplus.max() < 1),
            qualified=bool(converged and failure is None and yplus.max() < 1)),
        forces_dimensionless={key: value.tolist() for key, value in forces.items() if key.endswith('force')},
        history=history, momentum_history=momentum_history, failure=failure, wall_seconds=time.perf_counter()-start,
        limitations=['Experimental total drag and Cp are not yet compared by this runner.',
            'True nonlinear momentum, k and omega convergence is required independently.',
            'Actual computed y+ is necessary for wall-resolved SST acceptance.',
            'Inflow turbulence and tripping remain modeling assumptions.',
            'Single grid does not establish spatial convergence.'])
    write(out / 'result.json', summary)
    (out / 'report.md').write_text(f'''# Native SUBOFF SST-2003 diagnostic

Cells: {len(mesh.cells):,}; Re_L: {args.reynolds:g}; model: {VARIANT}.
Numerically converged: **{converged}**. Experimental validation passed: **false**.
Computed maximum wall y+: **{yplus.max():.6g}**.
This wall-resolved model requires a suitable first-cell distance; a set mesh
height alone is not a wall-resolution qualification. No wall function is used.

Physical pressure is converted from pi=p+2rho k/3, and the reported force
includes both anisotropic and isotropic Reynolds stress consistently.
The input, numerical sources, configuration, current nonlinear residuals,
raw fields, actual wall y+ and force components are frozen in this directory.
Experimental metrics and a mesh-refinement study remain outstanding.

Failure: `{failure}`. CPU wall time: {summary['wall_seconds']:.3f}s.
''')
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if converged else 2


if __name__ == '__main__':
    raise SystemExit(main())
