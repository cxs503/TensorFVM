#!/usr/bin/env python3
"""Native hybrid Stokes verification; never awards SUBOFF/RANS qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra, energy_diagnostic
from tensorfvm.verification.native_hybrid_flow import channel_solver, channel_metrics


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output
    if out.exists() and any(out.iterdir()): parser.error('output must be empty')
    out.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]; hashes = {}
    sources = {Path(__file__).resolve()} | {Path(m.__file__).resolve() for name,m in list(sys.modules.items())
        if name.startswith('tensorfvm') and getattr(m,'__file__',None) and Path(m.__file__).suffix=='.py'}
    for source in sorted(sources):
        relative = source.relative_to(repo); target = out/'source-snapshot'/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    write(out/'sources.json',hashes)
    start = time.perf_counter(); rows = []; comparisons = []; samples = []
    for skew in np.linspace(0, 1, 21):
        mesh = two_tetrahedra(float(skew))
        old = energy_diagnostic(mesh)
        velocity = np.zeros((2, 3)); velocity[0, 0] = 1.
        op = NativeHybridViscous3D(mesh)
        result = op.equilibrate_faces(velocity)
        # Exhaust all six cell-velocity directions by forming the small Schur
        # complement; this prevents accepting only one favorable test velocity.
        a = op.matrix.toarray(); cc = op.cell_dofs
        schur = a[:cc, :cc]-a[:cc, cc:] @ np.linalg.solve(a[cc:, cc:], a[cc:, :cc])
        samples.append(dict(skew=float(skew), old_energy_rate=old['kinetic_energy_rate'],
            hybrid_energy_rate=result['kinetic_energy_rate'],
            hybrid_symmetric_minimum_eigenvalue=float(np.linalg.eigvalsh((schur+schur.T)/2)[0]),
            energy_balance_error=result['energy_balance_error'],
            interface_flux_defect_max=result['interface_flux_defect_max']))
    write(out/'energy-sweep.json', samples)
    final = None
    for n in (2, 4, 8):
        reference = None
        for algorithm in ('coupled', 'simple', 'simplec'):
            mesh, solver = channel_solver(n)
            result = solver.solve(algorithm=algorithm, tolerance=1e-10, max_iterations=6000)
            row = dict(n=n, algorithm=algorithm, **channel_metrics(mesh, result))
            row['elapsed_seconds'] = time.perf_counter()-start
            rows.append(row)
            write(out/f'{n}-{algorithm}-history.json', result['history'])
            np.savez_compressed(out/f'{n}-{algorithm}-fields.npz',
                vertices=mesh.vertices, cells=np.asarray(mesh.cells), cell_centers=mesh.cell_centers,
                cell_volumes=mesh.cell_volumes, face_centers=mesh.face_centers,
                owner=mesh.owner, neighbor=mesh.neighbor, area_vectors=mesh.area_vectors,
                **{key:result[key] for key in ('velocity','face_velocity','pressure','flux','total_traction_integrated')})
            if reference is None: reference = result
            else:
                comparisons.append(dict(n=n, algorithm=algorithm,
                    maximum_absolute_difference={key:float(np.max(np.abs(result[key]-reference[key])))
                        for key in ('velocity','face_velocity','pressure','flux')}))
            print(json.dumps(row), flush=True)
            final = result
    write(out/'cases.json', rows); write(out/'algorithm-comparison.json', comparisons)
    fine = [r for r in rows if r['n'] == 8]
    passed = bool(all(r['numerically_converged'] for r in rows)
        and all(max(r['velocity_relative_l2'], r['pressure_relative_l2'], r['wall_force_relative_error']) < .03 for r in fine)
        and all(s['hybrid_symmetric_minimum_eigenvalue'] > 0 and s['hybrid_energy_rate'] < 0 for s in samples)
        and all(max(c['maximum_absolute_difference'].values()) < 1e-7 for c in comparisons))
    write(out/'summary.json', dict(native_stokes_verification_passed=passed,
        legacy_collocated_viscosity_fixed=False, legacy_suboff_fixed=False,
        sst_integrated=False, physical_suboff_benchmark_passed=False,
        gpu_tested=False, runtime_seconds=time.perf_counter()-start,
        scope='native hybrid Stokes only; no OpenFOAM fields or solver'))
    # Uniform-grid cut through the lower central cell layer. Each image pixel
    # represents an actual computed cell value; no smoothed contour fit.
    n = 8; layer = n//2-1
    p = final['pressure'].reshape(n,n,n)[layer]
    u = final['velocity'].reshape(n,n,n,3)[layer]
    fig, axes = plt.subplots(1,3, figsize=(12,3.7), constrained_layout=True)
    for axis in axes: axis.set(xlim=(0,1), ylim=(0,1), xlabel='x', ylabel='y', aspect='equal')
    for coordinate in np.linspace(0,1,n+1):
        axes[0].axvline(coordinate, color='.4', lw=.6); axes[0].axhline(coordinate, color='.4', lw=.6)
    axes[0].set_title('Actual cell cross section (512 cells)')
    for axis, data, title in zip(axes[1:], [p, np.linalg.norm(u,axis=2)], ['Cell pressure', 'Cell speed']):
        picture = axis.imshow(data, origin='lower', extent=(0,1,0,1), interpolation='nearest')
        axis.set_title(title); fig.colorbar(picture, ax=axis)
    fig.savefig(out/'mesh-fields.png', dpi=180); fig.savefig(out/'mesh-fields.pdf'); plt.close(fig)
    center = (np.arange(n)+.5)/n; dense = np.linspace(0,1,300)
    fig, axes = plt.subplots(1,3, figsize=(12,3.6), constrained_layout=True)
    axes[0].plot(dense, dense*(1-dense), label='analytic'); axes[0].plot(center,u[:,n//2,0], 'o', label='native SIMPLEC')
    axes[0].set(xlabel='y', ylabel='u_x', title='Velocity profile'); axes[0].legend()
    axes[1].plot(dense,2*(1-dense), label='analytic'); axes[1].plot(center,p[n//2,:], 'o', label='native SIMPLEC')
    axes[1].set(xlabel='x', ylabel='p', title='Pressure profile'); axes[1].legend()
    selected = [r for r in rows if r['algorithm']=='simplec']
    for key,label in [('velocity_relative_l2','Velocity'),('pressure_relative_l2','Pressure'),('wall_force_relative_error','Wall force')]:
        axes[2].loglog([r['cells'] for r in selected],[100*r[key] for r in selected], 'o-', label=label)
    axes[2].axhline(3,color='k',linestyle='--',label='3%')
    axes[2].set(xlabel='Cells',ylabel='Relative error (%)',title='Refinement'); axes[2].legend()
    fig.savefig(out/'profiles-refinement.png',dpi=180); fig.savefig(out/'profiles-refinement.pdf'); plt.close(fig)
    for relative, expected in hashes.items():
        if hashlib.sha256((repo/relative).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Source changed during verification: {relative}')
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
