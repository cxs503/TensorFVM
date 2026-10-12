#!/usr/bin/env python3
"""Energy-based hybrid FV verification and frozen SIMPLE pressure diagnosis."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import sys
from unittest.mock import patch

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tests.test_native_hybrid_viscous3d import thin_mixed_mesh
from tests.test_native_rans3d import box
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.native_linear_diagnostics import linear_system_diagnostics
from tensorfvm.native_collocated3d import NativeSteadySettings
from tensorfvm.native_rans3d import NativeSimpleSSTRansSolver3D, NativeRansSettings
from tensorfvm.verification.native_energy import two_tetrahedra


def metrics(result):
    return {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}


def energy_comparison(mesh, velocity, output, name):
    original = NativeViscous3D(mesh, mesh.neighbor < 0)
    rate = float(np.sum(velocity*original.op.integrate(original.momentum_flux(velocity))))
    hybrid = NativeHybridViscous3D(mesh)
    result = hybrid.equilibrate_faces(velocity, linear_method='cg')
    np.savez_compressed(output/(name+'-fields.npz'), velocity=velocity,
                        face_velocity=result['face_values'], face_traction=result['face_flux'])
    return dict(cells=len(mesh.cells), original_kinetic_energy_rate=rate,
                hybrid=metrics(result), hybrid_unknowns=hybrid.matrix.shape[0],
                hybrid_nonzeros=hybrid.matrix.nnz,
                passed=result['kinetic_energy_rate'] <= 1e-12 and result['energy_balance_error'] < 1e-9
                       and result['interface_flux_defect_max'] < 1e-9)


def convergence():
    cases = {}
    for components in (1,3):
        rows = []
        for n in (2,4,8,16):
            mesh = box(n)
            sn = np.sin(np.pi*mesh.cell_centers); cs = np.cos(np.pi*mesh.cell_centers)
            phi = np.prod(sn, axis=1)
            if components == 1:
                exact = phi[:, None]; source = 3*np.pi**2*exact
            else:
                exact = np.column_stack((phi, np.zeros((len(phi),2))))
                source = np.pi**2*np.column_stack((10/3*phi, -cs[:,0]*cs[:,1]*sn[:,2]/3,
                                                   -cs[:,0]*sn[:,1]*cs[:,2]/3))
            result = NativeHybridViscous3D(mesh, components=components).solve(source, linear_method='cg')
            error = float(np.sqrt(np.sum(mesh.cell_volumes[:,None]*(result['cell_values']-exact)**2)))
            row = dict(n=n, cells=len(mesh.cells), l2_error=error,
                       observed_order=float(np.log2(rows[-1]['l2_error']/error)) if rows else None,
                       linear_relative_residual=result['linear_relative_residual'],
                       interface_flux_defect_max=result['interface_flux_defect_max'])
            rows.append(row)
            print('refinement', components, n, error, flush=True)
        cases[str(components)] = dict(rows=rows, passed=rows[-1]['observed_order'] > 1.8)
    return cases


def pressure_audit(baseline, output):
    import tensorfvm.native_simple3d as module
    config = json.loads((baseline/'config.json').read_text())
    mesh = NativeMesh3D.from_gmsh(baseline/'input.msh', triangulate_faces=True)
    fields = np.load(baseline/'fields.npz', allow_pickle=False)
    boundary = np.tile([1.,0.,0.], (len(mesh.owner),1)); boundary[mesh.patch_labels == 'wall'] = 0.
    solver = NativeSimpleSSTRansSolver3D(mesh, (mesh.neighbor < 0)&(mesh.patch_labels != 'outlet'),
        boundary, mesh.patch_labels == 'outlet', settings=NativeSteadySettings(**config['momentum_settings']),
        rans_settings=NativeRansSettings(**config['rans_settings']))
    solver.initialize_state(fields['velocity'], fields['modified_pressure'], fields['flux'])
    before = [v.copy() for v in (solver.u, solver.p, solver.flux)]
    captured = {}
    class Capture:
        def __init__(self, matrix, **kwargs):
            self.matrix, self.factor = matrix, splu(matrix, **kwargs)
        def solve(self, rhs):
            answer = self.factor.solve(rhs)
            if self.matrix.shape[0] == solver.N:
                captured.update(matrix=self.matrix.copy(), rhs=rhs.copy(), solution=answer.copy())
            return answer
    failure = None
    with patch.object(module, 'splu', Capture):
        try:
            solver.solve_rans(config['inlet']['k'], config['inlet']['omega'],
                              inlet_k=config['inlet']['k'], inlet_omega=config['inlet']['omega'])
        except RuntimeError as error:
            failure = str(error)
    if not failure or not captured:
        raise RuntimeError('Expected frozen baseline rejection was not reproduced; update the diagnostic')
    matrix, rhs, candidate = (captured[k] for k in ('matrix','rhs','solution'))
    sparse.save_npz(output/'pressure-matrix.npz', matrix)
    np.savez_compressed(output/'pressure-candidate.npz', rhs=rhs, solution=candidate)
    original = linear_system_diagnostics(matrix, rhs, candidate, estimate_condition=True)
    row_scale = 1/abs(matrix).max(axis=1).toarray().ravel()
    row_scaled = sparse.diags(row_scale) @ matrix
    column_scale = 1/abs(row_scaled).max(axis=0).toarray().ravel()
    scaled = (row_scaled @ sparse.diags(column_scale)).tocsc()
    factor = splu(scaled)
    solve = lambda b: column_scale*factor.solve(row_scale*b)
    value = solve(rhs); trials = []
    extended_matrix = matrix.astype(np.longdouble)
    for step in range(3):
        trials.append(dict(refinement_steps=step, **linear_system_diagnostics(matrix, rhs, value)))
        residual = rhs.astype(np.longdouble)-extended_matrix @ value.astype(np.longdouble)
        value += solve(np.asarray(residual, float))
    unchanged = all(np.array_equal(a,b) for a,b in zip(before,(solver.u,solver.p,solver.flux)))
    return dict(failure=failure, original=original, scaled_trials=trials,
                rejected_solve=solver.last_rejected_linear_solve,
                accepted_state_unchanged=unchanged, accepted_steps=len(solver.history),
                pressure_tolerance=max(5*solver.settings.linear_tolerance,1e-9),
                longdouble_epsilon=float(np.finfo(np.longdouble).eps),
                passed=False, qualification='Diagnostic only; no tolerance change or candidate acceptance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=Path('docs/verification-native-solver-audit/suboff-state-preserved'))
    parser.add_argument('--witness', type=Path, default=Path('docs/verification-native-solver-audit/after/viscous-energy-witness.npz'))
    args = parser.parse_args(); output = args.output
    if output.exists() and any(output.iterdir()):
        parser.error('output must be empty')
    output.mkdir(parents=True, exist_ok=True)
    def write(name, value):
        (output/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    inputs = [args.baseline/p for p in ('input.msh','config.json','fields.npz')]+[args.witness]
    write('inputs.json', {str(p): digest(p) for p in inputs})
    write('environment.json', dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                                  numpy=np.__version__, device='CPU', hybrid_integrated_into_simple=False))
    energy = {}
    for name, mesh in [('two-cell',two_tetrahedra()),('thin-mixed',thin_mixed_mesh())]:
        velocity = np.zeros((len(mesh.cells),3)); velocity[0,0] = 1.
        energy[name] = energy_comparison(mesh, velocity, output, name)
    mesh = NativeMesh3D.from_gmsh(args.baseline/'input.msh', triangulate_faces=True)
    velocity = np.load(args.witness, allow_pickle=False)['velocity']
    energy['suboff-witness'] = energy_comparison(mesh, velocity, output, 'suboff-witness')
    write('energy.json', energy); print('ENERGY COMPLETE', flush=True)
    refinement = convergence(); write('convergence.json', refinement)
    pressure = pressure_audit(args.baseline, output); write('pressure.json', pressure)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7,4))
    for components, label in [('1','Scalar diffusion'),('3','Symmetric deviatoric stress')]:
        rows = refinement[components]['rows']
        ax.loglog([1/r['n'] for r in rows], [r['l2_error'] for r in rows], 'o-', label=label)
    h = np.array([1/16,1/8,1/4,1/2])
    ax.loglog(h, h*h, '--', label='slope 2')
    ax.set(xlabel='h', ylabel='Volume-weighted L2 error', title='Hybrid FV manufactured diffusion, homogeneous Dirichlet')
    ax.grid(True, which='both', alpha=.25); ax.legend(); fig.tight_layout()
    fig.savefig(output/'convergence.png', dpi=160); plt.close(fig)
    sources = {Path(__file__).resolve()}
    sources.update(Path(m.__file__).resolve() for name,m in sys.modules.items()
                   if name.startswith(('tensorfvm.','tests.test_native_')) and getattr(m,'__file__','').endswith('.py'))
    bindings = {}
    for source in sorted(sources):
        relative = source.relative_to(REPO); destination = output/'source-snapshot'/relative
        destination.parent.mkdir(parents=True,exist_ok=True); destination.write_bytes(source.read_bytes())
        bindings[str(relative)] = digest(source)
    write('sources.json', bindings)
    summary = dict(hybrid_subsystem_checks_passed=all(r['passed'] for r in energy.values()) and all(r['passed'] for r in refinement.values()),
                   original_viscous_defect_fixed=False, pressure_failure_fixed=False,
                   hybrid_integrated_into_simple=False, physical_benchmark_passed=False)
    write('summary.json',summary); print(json.dumps(summary,indent=2),flush=True)
    return 2  # Production flow validation remains incomplete.


if __name__ == '__main__':
    sys.exit(main())
