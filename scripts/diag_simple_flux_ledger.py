#!/usr/bin/env python3
"""Flux-ledger diagnostic for the native SIMPLE pressure/velocity sweep.

Reproduces the SUBOFF momentum/RANS iteration with explicit knobs and logs the
face-flux ledger at every accepted step:

* ``face_flux_consistency_max``: stored flux vs flux reconstructed from the
  accepted (u, p) state through the current Rhie--Chow mobility;
* ``pressure_increment_flux_response_max``: |diag(alpha) @ pressure_flux_defect
  @ dp| / |S|, the ledger pressure term moved by one full pressure correction;
* ``ledger_response_defect_max``: the part of the above that the accepted state
  update leaves inconsistent, i.e. |corr @ (dp - dp_applied)| / |S|.  The
  legacy under-relaxed-pressure update (p += alpha_p * dp while the flux takes
  the full response) leaves (1 - alpha_p) * corr @ dp off the manifold.

Modes:
* ``momentum``: one turbulence-closure/mobility update (outer iteration 1 of
  the interleaved run) followed by N SIMPLE sweeps without SST transport.
* ``rans``: full interleaved SIMPLE+SST loop, identical to
  ``scripts/run_native_suboff_rans.py`` scheduling.
"""
import argparse
import json
from pathlib import Path
import sys
import time

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
    parser.add_argument('--reynolds', type=float, default=1.2e7)
    parser.add_argument('--mode', choices=['momentum', 'rans'], default='momentum')
    parser.add_argument('--sweeps', type=int, default=4, help='SIMPLE sweeps in momentum mode')
    parser.add_argument('--outer', type=int, default=20, help='RANS outer iterations in rans mode')
    parser.add_argument('--inner', type=int, default=1)
    parser.add_argument('--relaxation', type=float, default=.05)
    parser.add_argument('--pressure-relaxation', type=float, default=.03)
    parser.add_argument('--pressure-velocity-algorithm', choices=['simple', 'simplec'], default='simple')
    parser.add_argument('--convection-order', type=int, choices=[1, 2], default=1)
    parser.add_argument('--pseudo-dt', type=float, default=.0001, help='SST pseudo time step in rans mode')
    parser.add_argument('--turbulence-relaxation', type=float, default=.5)
    parser.add_argument('--intensity', type=float, default=.005)
    parser.add_argument('--length-scale', type=float, default=.05)
    parser.add_argument('--initial-velocity', choices=['freestream', 'rest'], default='freestream')
    parser.add_argument('--project-initial-flux', action='store_true')
    parser.add_argument('--initial-run', type=Path, help='Frozen fields.npz restart on identical mesh')
    parser.add_argument('--maximum-speed', type=float, default=5.)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty to preserve evidence')
    args.output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    config = dict(vars(args))
    config['output'] = str(config['output'])
    config['mesh'] = str(config['mesh'])
    if config.get('initial_run') is not None:
        config['initial_run'] = str(config['initial_run'])
    config['note'] = ('ledger response defect is max|corr@(dp-dp_applied)|/|S|; '
        'legacy under-relaxed pressure applies dp_applied=alpha_p*dp to p while the '
        'flux takes the full response, so it leaves (1-alpha_p)*corr@dp inconsistent')
    (args.output / 'config.json').write_text(json.dumps(config, indent=2, allow_nan=False))

    mesh = NativeMesh3D.from_gmsh(args.mesh, triangulate_faces=True)
    fixed = (mesh.neighbor < 0) & (mesh.patch_labels != 'outlet')
    velocity = np.tile([1., 0., 0.], (len(mesh.owner), 1))
    velocity[mesh.patch_labels == 'wall'] = 0.
    settings_kwargs = dict(viscosity=1 / args.reynolds,
        max_iterations=args.inner if args.mode == 'rans' else args.sweeps,
        nonlinear_tolerance=1e-8, relaxation=args.relaxation,
        bounded_convection=True, convection_order=args.convection_order,
        pressure_relaxation=args.pressure_relaxation,
        pressure_velocity_algorithm=args.pressure_velocity_algorithm)
    import dataclasses
    supported = {f.name for f in dataclasses.fields(NativeSteadySettings)}
    settings = NativeSteadySettings(**{k: v for k, v in settings_kwargs.items() if k in supported})
    rans_settings = NativeRansSettings(max_outer_iterations=args.outer,
        pseudo_time_step=args.pseudo_dt, turbulence_relaxation=args.turbulence_relaxation,
        nonlinear_tolerance=1e-7, wall_treatment='spalding')
    k0, omega0 = inflow_from_intensity(1., args.intensity, args.length_scale)
    solver = NativeSimpleSSTRansSolver3D(mesh, fixed, velocity,
        pressure_dirichlet=mesh.patch_labels == 'outlet',
        settings=settings, rans_settings=rans_settings)
    if args.initial_run is not None:
        initial = np.load(args.initial_run, allow_pickle=False)
        solver.initialize_state(initial['velocity'], initial['modified_pressure'], initial['flux'])
    elif args.initial_velocity == 'freestream':
        solver.u[:, 0] = 1.
    if args.project_initial_flux:
        projection = NativeProjection3D(mesh, mesh.patch_labels == 'outlet')
        tentative = solver.velocity_flux_matrix @ solver.u.ravel() + solver.velocity_flux_source
        _, projected = projection.project(tentative)
        solver.initialize_state(solver.u, solver.p, projected)

    norms = np.linalg.norm(mesh.area_vectors, axis=1)
    ledger = []
    previous_p = None

    def record(row, stage):
        # Recompute the current reconstruction response at the accepted state so
        # the ledger is measured with the same operators the next sweep uses.
        nonlocal previous_p
        block, target, _, correction, source = solver._system(solver.flux)
        consistent = solver.velocity_flux_matrix @ solver.u.ravel() + correction @ solver.p + source
        state_increment = np.zeros(solver.N) if previous_p is None else solver.p - previous_p
        step_response = np.abs(correction @ state_increment) / norms
        row = dict(row, stage=stage, elapsed_seconds=time.perf_counter() - start,
            maximum_speed=float(np.linalg.norm(solver.u, axis=1).max()),
            ledger_recomputed_consistency_max=float(np.max(np.abs(consistent - solver.flux) / norms)),
            state_pressure_step_response_max=float(step_response.max()),
            log_entry=len(ledger) + 1)
        previous_p = solver.p.copy()
        ledger.append(row)
        with (args.output / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        shown = {k: row.get(k) for k in (
            'iteration', 'outer_iteration', 'face_flux_consistency_max',
            'state_pressure_step_response_max',
            'current_reconstructed_divergence_max', 'native_divergence_max',
            'momentum_residual', 'maximum_speed')}
        print(json.dumps(dict(stage=stage, **shown), allow_nan=False), flush=True)
        if args.maximum_speed is not None and row['maximum_speed'] > args.maximum_speed:
            raise RuntimeError(f'divergence guard: max speed {row["maximum_speed"]:g} > {args.maximum_speed:g}')

    failure = None
    try:
        if args.mode == 'momentum':
            solver.k = np.broadcast_to(np.asarray(k0, dtype=float), (solver.N,)).copy()
            solver.omega = np.broadcast_to(np.asarray(omega0, dtype=float), (solver.N,)).copy()
            solver.k_boundary = np.broadcast_to(np.asarray(k0, dtype=float), (solver.F,)).copy()
            solver.omega_boundary = np.broadcast_to(np.asarray(omega0, dtype=float), (solver.F,)).copy()
            closure = solver.current_closure()
            solver._update_momentum_coefficients(closure)
            solver.solve(callback=lambda row: record(row, 'momentum'))
        else:
            solver.solve_rans(k0, omega0, inlet_k=k0, inlet_omega=omega0,
                callback=lambda row: record(row, 'rans'),
                momentum_callback=lambda row: record(row, 'momentum'))
    except Exception as error:  # noqa: BLE001 - diagnostic must archive failures
        failure = dict(type=type(error).__name__, message=str(error))
    summary = dict(cells=len(mesh.cells), mode=args.mode, failure=failure,
        ledger_entries=len(ledger),
        final_face_flux_consistency_max=(ledger[-1].get('face_flux_consistency_max') if ledger else None),
        final_maximum_speed=(ledger[-1].get('maximum_speed') if ledger else None),
        wall_seconds=time.perf_counter() - start)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if failure is None else 2


if __name__ == '__main__':
    raise SystemExit(main())
