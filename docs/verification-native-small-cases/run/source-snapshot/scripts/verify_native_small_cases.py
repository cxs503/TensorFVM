#!/usr/bin/env python3
"""Run tiny native solver defect diagnostics; exit 2 if any requirement fails."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.test_native_collocated3d import skew_mesh
from tests.test_native_rans3d import channel
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_simple3d import NativeSimpleSolver3D
from tensorfvm.native_rans3d import NativeSimpleSSTRansSolver3D, NativeRansSettings
from tensorfvm.verification.native_energy import two_tetrahedra, energy_diagnostic
from tensorfvm.wall_functions import wall_traction


def relaxation_case():
    mesh = skew_mesh(3)
    x, y, z = mesh.cell_centers.T
    force = np.column_stack((np.sin(3*y), np.cos(2*x), x*z))
    fields, runs = [], []
    for relaxation in (.3, .8):
        solver = NativeSimpleSolver3D(mesh, mesh.neighbor < 0, np.zeros((len(mesh.owner), 3)),
            body_force=force, settings=NativeSteadySettings(viscosity=.1, convection=False,
                relaxation=relaxation, max_iterations=1500, nonlinear_tolerance=1e-9))
        result = solver.solve()
        fields.append((solver.u.copy(), solver.p.copy()))
        runs.append(dict(relaxation=relaxation, converged=result['converged'],
                         iterations=len(result['history']), last=result['history'][-1]))
    errors = [float(np.linalg.norm(fields[0][i]-fields[1][i])/np.linalg.norm(fields[1][i])) for i in (0, 1)]
    return dict(cells=27, runs=runs, relative_velocity_difference=errors[0],
                relative_pressure_difference=errors[1],
                passed=all(r['converged'] for r in runs) and max(errors) < 1e-8)


def interrupted_wall_case():
    mesh = channel()
    wall = mesh.patch_labels == 'wall'
    boundary = np.tile([1., 0., 0.], (len(mesh.owner), 1))
    boundary[wall] = 0.
    solver = NativeSimpleSSTRansSolver3D(mesh, (mesh.neighbor < 0)&(mesh.patch_labels != 'outlet'),
        boundary, mesh.patch_labels == 'outlet',
        settings=NativeSteadySettings(viscosity=.001, max_iterations=1, convection_order=1, relaxation=.3),
        rans_settings=NativeRansSettings(wall_treatment='spalding'))
    solver.u[:, 0] = 1.
    def stop(row):
        raise StopIteration('interrupt after accepted momentum step')
    interrupted = False
    try:
        solver.solve_rans(.001, 1., inlet_k=.001, inlet_omega=1., momentum_callback=stop)
    except StopIteration:
        interrupted = True
    velocity, matrix = solver.u.copy(), solver._diffusion.copy()
    forces = solver.forces()
    area = np.linalg.norm(mesh.area_vectors[wall], axis=1)
    normal = mesh.area_vectors[wall]/area[:, None]
    def tangent(value):
        return value-normal*np.sum(value*normal, axis=1)[:, None]
    expected = wall_traction(tangent(solver.u[mesh.owner[wall]]),
        solver.wall_distance[mesh.owner[wall]], 1., .001)['traction_pa']*area[:, None]
    actual = tangent(forces['anisotropic_stress_traction'][wall])
    frozen = tangent(forces['linearized_anisotropic_stress_traction'][wall])
    error = float(np.linalg.norm(actual-expected)/np.linalg.norm(expected))
    frozen_error = float(np.linalg.norm(frozen-expected)/np.linalg.norm(expected))
    unchanged = np.array_equal(velocity, solver.u) and (matrix != solver._diffusion).nnz == 0
    return dict(cells=27, interrupted=interrupted, relative_shear_error=error,
                frozen_relative_shear_error=frozen_error, state_and_matrix_unchanged=unchanged,
                passed=interrupted and unchanged and error < 1e-12 and frozen_error > 1e-8)


def rejected_state_case(mode):
    import tensorfvm.native_simple3d as module
    original = module.splu
    mesh = skew_mesh(2)
    solver = NativeSimpleSolver3D(mesh, mesh.neighbor < 0, np.zeros((len(mesh.owner), 3)),
        body_force=np.tile([.3, .2, .1], (len(mesh.cells), 1)),
        settings=NativeSteadySettings(viscosity=.1, convection=False, max_iterations=1))
    solver.initialize_state(np.zeros((solver.N, 3)), np.zeros(solver.N), np.zeros(solver.F))
    before = [a.copy() for a in (solver.u, solver.p, solver.flux)]
    class PerturbedFactor:
        def __init__(self, matrix, **kwargs):
            self.factor, self.size = original(matrix, **kwargs), matrix.shape[0]
        def solve(self, rhs):
            value = self.factor.solve(rhs)
            if self.size == solver.N:
                value = np.full(solver.N, np.nan) if mode == 'nonfinite' else value+np.linspace(0., 1., solver.N)
            return value
    failure = None
    with patch.object(module, 'splu', PerturbedFactor):
        try:
            solver.solve()
        except RuntimeError as error:
            failure = str(error)
    unchanged = all(np.array_equal(a, b) for a, b in zip(before, (solver.u, solver.p, solver.flux)))
    return dict(cells=8, injection=mode, failure=failure, accepted_state_unchanged=unchanged,
                accepted_steps=len(solver.history), rejected_linear_solve=solver.last_rejected_linear_solve,
                passed=bool(failure and 'linear solve' in failure and unchanged and not solver.history))


def figures(output, sweep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 4.6))
    for index, skew in enumerate((0., 1.)):
        mesh = two_tetrahedra(skew)
        ax = fig.add_subplot(1, 2, index+1, projection='3d')
        faces = [mesh.vertices[list(face)] for face in mesh.face_vertices]
        ax.add_collection3d(Poly3DCollection(faces, facecolor='#7bb4db', alpha=.25, edgecolor='black'))
        p = mesh.vertices
        ax.auto_scale_xyz(p[:, 0], p[:, 1], p[:, 2])
        ax.set_box_aspect(np.maximum(np.ptp(p, axis=0), .1))
        for i, point in enumerate(p):
            ax.text(*point, str(i))
        ax.set(xlabel='x', ylabel='y', zlabel='z', title=f'Two tetrahedra: skew={skew:g}')
    fig.tight_layout(); fig.savefig(output/'two-cell-mesh.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([s['skew'] for s in sweep], [s['kinetic_energy_rate'] for s in sweep], 'o-')
    ax.axhline(0, color='black', linewidth=.8)
    ax.set(xlabel='Geometric deformation parameter', ylabel='dE/dt, fixed velocity witness',
           title='Viscous suboperator, zero boundary velocity, mu=rho=1')
    ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(output/'energy-sweep.png', dpi=160); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    results = dict(energy_control=energy_diagnostic(two_tetrahedra(0.)),
                   energy_skew=energy_diagnostic(two_tetrahedra()))
    sweep = [dict(skew=float(s), **energy_diagnostic(two_tetrahedra(s))) for s in np.linspace(0, 1, 21)]
    print('Two-cell energy cases complete', flush=True)
    results['relaxation'] = relaxation_case()
    results['interrupted_wall'] = interrupted_wall_case()
    results['rejected_inaccurate'] = rejected_state_case('inaccurate')
    results['rejected_nonfinite'] = rejected_state_case('nonfinite')
    figures(output, sweep)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    for name, value in results.items():
        write(name+'.json', value)
    write('energy-sweep.json', sweep)
    sources = {Path(__file__).resolve()}
    sources.update(Path(m.__file__).resolve() for name, m in sys.modules.items()
                   if name.startswith(('tensorfvm.', 'tests.test_native_')) and getattr(m, '__file__', '').endswith('.py'))
    bindings = {}
    for source in sorted(sources):
        relative = source.relative_to(REPO)
        destination = output/'source-snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        bindings[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    write('sources.json', bindings)
    summary = dict(requirements={k: v['passed'] for k, v in results.items()},
                   all_requirements_passed=all(v['passed'] for v in results.values()),
                   physical_benchmark_passed=False)
    write('summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary['all_requirements_passed'] else 2


if __name__ == '__main__':
    sys.exit(main())
