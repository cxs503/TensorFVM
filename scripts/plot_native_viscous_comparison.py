#!/usr/bin/env python3
"""Audited same-mesh viscous-operator comparison with common colour scales."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from plot_native_suboff_fields import load_plot_mesh, cell_section_polygons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plane-z', type=float, default=.002857978102883826)
    parser.add_argument('--x-limits', nargs=2, type=float, default=[.97, 1.025])
    parser.add_argument('--y-limits', nargs=2, type=float, default=[-.015, .015])
    args = parser.parse_args(); out = args.output
    if out.exists(): parser.error('Output must be new')
    if not np.isfinite(args.plane_z): parser.error('Finite section plane required')
    for limits in (args.x_limits, args.y_limits):
        if not np.isfinite(limits).all() or limits[0] >= limits[1]: parser.error('Finite increasing display limits required')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    bindings = {}; configs = []; fields = []; results = []
    for run in (args.baseline, args.candidate):
        audit = json.loads((run/'audit/audit.json').read_text())
        if audit.get('passed') is not True: raise ValueError('Independent manufactured audits must pass')
        for name in ('config.json', 'input.msh', 'fields.npz', 'result.json'):
            path = run/name; matches = [v for k,v in audit['producer_input_sha256'].items() if Path(k).name == name]
            if matches != [sha(path)]: raise ValueError('Independent audit binding mismatch: '+str(path))
            bindings[str(path)] = sha(path)
        bindings[str(run/'audit/audit.json')] = sha(run/'audit/audit.json')
        configs.append(json.loads((run/'config.json').read_text()))
        results.append(json.loads((run/'result.json').read_text()))
        with np.load(run/'fields.npz') as data:
            fields.append({k: data[k].copy() for k in ('pressure', 'velocity', 'reference_pressure', 'reference_velocity')})
    first, second = configs
    for key in ('mesh_sha256', 'mode', 'reconstruction', 'settings', 'merge_coplanar_faces', 'quadratic_linear_solver', 'manufactured_definition'):
        if first.get(key) != second.get(key): raise ValueError('Same-mesh/operator-control contract differs: '+key)
    if first['mode'] != 'manufactured' or [c.get('viscous_reconstruction', 'quadratic') for c in configs] != ['quadratic', 'cubic']:
        raise ValueError('Requires quadratic versus cubic viscosity in manufactured flow')
    for key in ('reference_pressure', 'reference_velocity'):
        np.testing.assert_array_equal(fields[0][key], fields[1][key])
    m, _, _ = load_plot_mesh(args.baseline/'input.msh', args.baseline/'fields.npz')
    polygons, ids = cell_section_polygons(m, value=args.plane_z)
    visible = np.array([p[:,0].max() >= args.x_limits[0] and p[:,0].min() <= args.x_limits[1]
                        and p[:,1].max() >= args.y_limits[0] and p[:,1].min() <= args.y_limits[1] for p in polygons])
    polygons = [p for p, keep in zip(polygons, visible) if keep]; ids = ids[visible]
    if not len(ids): raise ValueError('Empty native section')
    align = lambda p: p-np.dot(p, m.cell_volumes)/m.cell_volumes.sum()
    reference = align(fields[0]['reference_pressure'])
    pressures = [align(f['pressure']) for f in fields]
    errors = [abs(p-reference) for p in pressures]
    pressure_limits = (min(p[ids].min() for p in [reference]+pressures), max(p[ids].max() for p in [reference]+pressures))
    error_limit = max(e[ids].max() for e in errors)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    out.mkdir(parents=True)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), layout='constrained')
    for ax, values, title, error in zip(axes.ravel(), [reference, *pressures, None, *errors],
        ['Analytic reference pressure', 'Quadratic viscous gradient', 'Cubic viscous gradient', 'Actual cell intersections', 'Quadratic absolute pressure error', 'Cubic absolute pressure error'], [False, False, False, False, True, True]):
        if values is None:
            col = PolyCollection(polygons, facecolors='white', edgecolors='#444444', linewidths=.25)
        else:
            col = PolyCollection(polygons, array=values[ids], cmap='viridis' if error else 'coolwarm', edgecolors='#777777', linewidths=.08)
            col.set_clim(*( (0, error_limit) if error else pressure_limits)); fig.colorbar(col, ax=ax, label='Solver pressure units')
        ax.add_collection(col); ax.set(xlim=args.x_limits, ylim=args.y_limits, xlabel='x/L', ylabel='y/L', title=title); ax.set_aspect('equal')
    fig.suptitle(f'Same {len(m.cells):,}-cell mesh, z/L={args.plane_z:g}; shared pressure/error scales\nActual centroid samples, global gauges only; MMS, physical SUBOFF unqualified')
    fig.savefig(out/'pressure-comparison.png', dpi=190); fig.savefig(out/'pressure-comparison.pdf'); plt.close(fig)
    wall = np.unique(m.owner[m.patch_labels == 'wall']); x = m.cell_centers[wall, 0]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), layout='constrained', sharex=True)
    axes[0].scatter(x, reference[wall], s=9, color='#555555', label='Analytic reference')
    for p, e, label, color in zip(pressures, errors, ['Quadratic viscosity', 'Cubic viscosity'], ['#cb7020', '#167a84']):
        axes[0].scatter(x, p[wall], s=5, color=color, alpha=.6, label=label)
        axes[1].scatter(x, e[wall]/.25, s=7, color=color, alpha=.6, label=label)
    axes[0].set(ylabel='Pressure − global volume mean', title='All wall-adjacent cell centroids, all azimuths (not wall Cp)')
    axes[1].set(xlabel='x/L', ylabel='Absolute pressure error / (rho U²)', yscale='log')
    for ax in axes: ax.grid(alpha=.2); ax.legend()
    fig.savefig(out/'wall-adjacent-pressure.png', dpi=190); fig.savefig(out/'wall-adjacent-pressure.pdf'); plt.close(fig)
    np.savetxt(out/'wall-adjacent-pressure.csv', np.column_stack((wall,m.cell_centers[wall],reference[wall],pressures[0][wall],pressures[1][wall],errors[0][wall],errors[1][wall])), delimiter=',',
        header='cell,x,y,z,reference_global_aligned,quadratic_global_aligned,cubic_global_aligned,quadratic_absolute_error,cubic_absolute_error', comments='')
    history = results[1]['history']
    keys = ['velocity_change', 'pressure_change', 'flux_change', 'momentum_residual', 'native_divergence_max', 'current_reconstructed_divergence_max', 'face_flux_consistency_max']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for index, key in enumerate(keys):
        ax = axes[int(index >= 3)]
        ax.semilogy([row['iteration'] for row in history], [max(row[key], 1e-18) for row in history], marker='.', label=key)
    for ax in axes:
        ax.axhline(second['settings']['nonlinear_tolerance'], color='black', linestyle='--', label='Original tolerance')
        ax.set(xlabel='Accepted nonlinear update', ylabel='Recorded gate value'); ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.suptitle('Cubic viscosity: all seven convergence gates; initialized from prior numerical solution')
    fig.savefig(out/'nonlinear-convergence.png', dpi=190); fig.savefig(out/'nonlinear-convergence.pdf'); plt.close(fig)
    np.savetxt(out/'nonlinear-history.csv', np.array([[row['iteration']]+[row[k] for k in keys] for row in history]), delimiter=',', header=','.join(['iteration']+keys), comments='')
    np.savez_compressed(out/'section.npz', polygon_vertices=np.concatenate(polygons), polygon_offsets=np.r_[0,np.cumsum([len(p) for p in polygons])],
        cell_ids=ids, reference_pressure=reference[ids], quadratic_pressure=pressures[0][ids], cubic_pressure=pressures[1][ids],quadratic_error=errors[0][ids],cubic_error=errors[1][ids])
    summary = dict(scope='same-grid operator change, not a mesh-convergence series or physical validation', cells=len(m.cells), plane_z=args.plane_z,
        display_window=dict(x=args.x_limits,y=args.y_limits), pressure_limits=pressure_limits, error_limits=(0,error_limit), global_gauge_only=True,
        representation='exact cell section polygons coloured by actual cell centroid values; no smoothing or plane interpolation', input_sha256=bindings)
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    for source in (Path(__file__), Path(__file__).with_name('plot_native_suboff_fields.py')): shutil.copy2(source, out/source.name)
    (out/'manifest.json').write_text(json.dumps({p.name: sha(p) for p in out.iterdir() if p.is_file()},indent=2)+'\n')


if __name__ == '__main__': main()
