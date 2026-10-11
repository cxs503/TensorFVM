#!/usr/bin/env python3
"""Run and freeze a native mixed-cell 3D manufactured or laminar diagnostic.

Manufactured acceptance is independent of experimental SUBOFF acceptance.
The laminar mode has no experimental reference and cannot pass a physical gate.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import scipy
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_collocated3d import NativeSteadySettings, NativeSteadySolver3D
from tensorfvm.native_manufactured3d import abc_fields, exact_boundary_flux, volume_weighted_error


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['manufactured', 'laminar'], default='manufactured')
    p.add_argument('--nu', type=float, default=.1)
    p.add_argument('--max-iterations', type=int, default=100)
    p.add_argument('--tolerance', type=float, default=1e-7)
    p.add_argument('--relaxation', type=float, default=.7)
    p.add_argument('--linear-method', choices=['direct', 'iterative', 'block', 'schur'], default='direct')
    args = p.parse_args()
    out = args.output
    if out.exists() and any(out.iterdir()):
        p.error('output must be empty to preserve evidence')
    if args.nu <= 0:
        p.error('positive kinematic viscosity required')
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    start = time.perf_counter()
    shutil.copy2(args.mesh, out / 'input.msh')
    sources = {Path(__file__).resolve()} | {
        Path(m.__file__).resolve() for name, m in list(sys.modules.items())
        if (name == 'tensorfvm' or name.startswith('tensorfvm.'))
        and getattr(m, '__file__', None) and Path(m.__file__).suffix == '.py'
    }
    bindings = {}
    for source in sorted(sources):
        relative = source.relative_to(repo)
        target = out / 'source-snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        bindings[str(relative)] = digest(target)
    settings = NativeSteadySettings(viscosity=args.nu, max_iterations=args.max_iterations,
        nonlinear_tolerance=args.tolerance, relaxation=args.relaxation,
        linear_method=args.linear_method, bounded_convection=args.mode == 'laminar')
    write(out / 'config.json', dict(mode=args.mode, settings=asdict(settings),
        mesh_sha256=digest(out / 'input.msh'), source_sha256=bindings,
        experimental_accuracy_qualified=False,
        manufactured_definition='independent continuum ABC velocity, trigonometric pressure and analytic NS forcing; all velocity boundaries manufactured including hull',
        error_definition='volume-weighted relative L2 at actual cell volume centroids; scalar pressure mean removed',
        acceptance='converged AND each of u_x,u_y,u_z,vector velocity,pressure errors strictly <0.03',
        physical_laminar_definition='no-slip bare hull, U=(1,0,0), pressure outlet=0, Re_L=1/nu; no experimental comparison'))
    write(out / 'environment.json', dict(python=sys.version, executable=sys.executable,
        platform=platform.platform(), numpy=np.__version__, scipy=scipy.__version__,
        git_base=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        device='CPU scipy sparse reference; GPU not tested'))
    mesh = NativeMesh3D.from_gmsh(out / 'input.msh', triangulate_faces=True)
    boundary = mesh.neighbor < 0
    if args.mode == 'manufactured':
        field = lambda points: abc_fields(points, nu=args.nu)
        exact = field(mesh.cell_centers)
        solver = NativeSteadySolver3D(mesh, boundary, field(mesh.face_centers)['velocity'],
            settings=settings, body_force=exact['acceleration'],
            boundary_flux=exact_boundary_flux(mesh, field), gauge_value=exact['pressure'][0])
    else:
        fixed = boundary & (mesh.patch_labels != 'outlet')
        velocity = np.tile([1., 0., 0.], (len(mesh.owner), 1))
        velocity[mesh.patch_labels == 'wall'] = 0
        solver = NativeSteadySolver3D(mesh, fixed, velocity,
            pressure_dirichlet=mesh.patch_labels == 'outlet', settings=settings)
        exact = None
    print(f'OPERATOR_READY cells={len(mesh.cells)} elapsed={time.perf_counter()-start:.3f}', flush=True)
    history = []
    def checkpoint(row):
        history.append(row)
        with (out / 'history.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
        np.savez_compressed(out / 'current-state.npz', velocity=solver.u,
            pressure=solver.p, flux=solver.flux)
        print(json.dumps(row), flush=True)
    failure = None
    try:
        result = solver.solve(callback=checkpoint)
    except Exception as error:
        failure = dict(type=type(error).__name__, message=str(error))
        result = dict(converged=False, velocity=solver.u, pressure=solver.p, flux=solver.flux)
    fields = dict(velocity=result['velocity'], pressure=result['pressure'], flux=result['flux'])
    forces = solver.forces()
    fields.update(pressure_faces=forces['pressure_faces'],
        viscous_traction_integrated=forces['viscous_traction_integrated'])
    errors = {}
    if exact is not None:
        fields.update(reference_velocity=exact['velocity'], reference_pressure=exact['pressure'])
        errors['velocity'] = volume_weighted_error(result['velocity'], exact['velocity'], mesh.cell_volumes)
        for axis in range(3):
            errors[f'velocity_{"xyz"[axis]}'] = volume_weighted_error(result['velocity'][:, axis], exact['velocity'][:, axis], mesh.cell_volumes)
        errors['pressure'] = volume_weighted_error(result['pressure'], exact['pressure'], mesh.cell_volumes, remove_mean=True)
    np.savez_compressed(out / 'fields.npz', **fields)
    summary = dict(mode=args.mode, cells=len(mesh.cells), geometry=mesh.audit(),
        converged=result['converged'], relative_L2_errors=errors,
        manufactured_pointwise_accuracy_passed=bool(exact is not None and result['converged'] and all(x < .03 for x in errors.values())),
        mesh_convergence_qualified=False, experimental_accuracy_qualified=False,
        wall_forces={key: forces[key].tolist() for key in ['pressure_force', 'viscous_force', 'total_force']},
        history=history, failure=failure, wall_seconds=time.perf_counter()-start,
        extended_pressure_stencils=getattr(solver.pressure, 'extended_pressure_stencils', None),
        limitations=['Single grid cannot establish spatial convergence.',
            'Manufactured hull boundary is not physical no-slip SUBOFF flow.',
            'No turbulence model or experimental drag/Cp comparison in this runner.',
            'Full nonorthogonal stress matrix is not guaranteed positive definite.',
            'Cell-center error metric does not equal error in exact cell averages.'])
    write(out / 'result.json', summary)
    rows = '\n'.join(f'| {key} | {100*value:.6g}% | {value < .03} |' for key, value in errors.items())
    (out / 'report.md').write_text(f'''# Native 3D SUBOFF {args.mode} calculation

Actual mixed-cell body-fitted mesh: {len(mesh.cells):,} control volumes.
Converged: **{result['converged']}**. Independent manufactured field gate: **{summary['manufactured_pointwise_accuracy_passed']}**.
Experimental drag/Cp qualification: **false**. Spatial refinement qualification: **false**.

| Relative volume-weighted L2 metric | Error | Strictly below 3% |
|---|---:|---|
{rows}

The manufactured solution prescribes nonzero velocity at the hull and analytic
continuum forcing; it is a discretization verification, not physical SUBOFF flow.
Pressure is solved from coupled momentum and continuity, not a velocity potential.
See `config.json` for normalization, `result.json` for current reconstructed mass
and momentum residuals, and `history.jsonl` for every completed nonlinear update.
The input mesh and imported numerical sources are frozen with SHA256 bindings.
`fields.npz` contains raw cell fields and native face fluxes/tractions.

Failure, if any: `{failure}`.
CPU wall time: {summary['wall_seconds']:.3f} seconds. GPU has not been tested.
''')
    print(json.dumps(summary, allow_nan=False), flush=True)
    return 0 if result['converged'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
