#!/usr/bin/env python3
"""Plot audited SUBOFF refinement errors without fitting an observed order."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(runs, local_audit, source_review=None):
    local = json.loads((local_audit / 'result.json').read_text())
    manifest = json.loads((local_audit / 'manifest.json').read_text())
    if manifest.get('result.json') != digest(local_audit / 'result.json'):
        raise ValueError('Local audit result hash mismatch')
    rows, policies, profiles, bindings = [], [], [], {}
    windows = None
    for run in runs:
        config = json.loads((run / 'config.json').read_text())
        profiles.append({k: v for k, v in config.get('source_sha256', {}).items()
                         if k.startswith('src/tensorfvm/')})
        audit_path = run / 'audit/audit.json'
        audit = json.loads(audit_path.read_text())
        if not audit.get('passed') or config['mode'] != 'manufactured':
            raise ValueError('Comparison requires independently passed manufactured runs')
        for name in ('config.json', 'input.msh', 'fields.npz', 'result.json'):
            path = run / name
            matches = [v for k, v in audit['producer_input_sha256'].items()
                       if Path(k).name == name]
            if len(matches) != 1 or matches[0] != digest(path):
                raise ValueError('Independent audit binding mismatch: ' + str(path))
            bindings[str(path.resolve())] = matches[0]
        bindings[str(audit_path.resolve())] = digest(audit_path)
        evidence = local[run.name]
        for key, name in (('mesh_sha256', 'input.msh'), ('fields_sha256', 'fields.npz')):
            if evidence[key] != digest(run / name):
                raise ValueError('Local audit binding mismatch: ' + str(run))
        current_windows = evidence['fixed_spatial_windows_L']
        if windows is not None and current_windows != windows:
            raise ValueError('Fixed spatial windows must match across runs')
        windows = current_windows
        settings = config['settings'].copy()
        settings.pop('max_iterations', None)
        # Historical runs used SuperLU's implicit COLAMD default. Preserve
        # rejection of genuinely different ordering choices in comparisons.
        settings.setdefault('lu_ordering', 'COLAMD')
        policies.append(dict(settings=settings, reconstruction=config.get('reconstruction', 'linear'),
                             quadratic_linear_solver=config.get('quadratic_linear_solver', 'matrix-free'),
                             merge_coplanar_faces=config.get('merge_coplanar_faces', False),
                             manufactured_definition=config['manufactured_definition']))
        row = dict(case=run.name, cells=evidence['cells'], faces=evidence['faces'],
                   velocity_L2_percent=100 * audit['relative_L2_errors']['velocity'])
        for region in ('all', 'wall_adjacent', 'near_body_fixed_window', 'near_tail_fixed_window'):
            stats = evidence['regions'][region]
            row[region + '_pressure_L2_percent'] = 100 * stats['pressure_relative_L2_global_gauge_aligned']
            row[region + '_maximum_error_percent_rho_U2'] = 100 * stats['maximum_error_over_rho_U2']
            row[region + '_rms_error_percent_rho_U2'] = 100 * stats['rms_error_over_rho_U2']
            row[region + '_cells_above_3percent_rho_U2'] = stats['cells_above_3percent_rho_U2']
        rows.append(row)
    if len(rows) < 2 or len({r['cells'] for r in rows}) != len(rows):
        raise ValueError('At least two distinct cell counts are required')
    if any(policy != policies[0] for policy in policies[1:]):
        raise ValueError('Numerical settings and face policies must match; compare experiments separately')
    if any(profile != profiles[0] for profile in profiles[1:]):
        if source_review is None:
            raise ValueError('Numerical source versions differ; an explicit bound compatibility probe is required')
        ledger = json.loads((source_review / 'manifest.json').read_text())
        documents = {}
        for name in ('result.json', 'historical.json', 'current.json', 'historical.npz', 'current.npz'):
            path = source_review / name
            if ledger.get(name) != digest(path):
                raise ValueError('Source compatibility artifact hash mismatch')
            bindings[str(path.resolve())] = digest(path)
            if name.endswith('.json'):
                documents[name] = json.loads(path.read_text())
        if documents['result.json'].get('passed') is not True:
            raise ValueError('Source compatibility probe failed')
        reviewed = [documents[name]['frozen_import_sha256'] for name in ('historical.json', 'current.json')]
        if any(profile not in reviewed for profile in profiles):
            raise ValueError('Compatibility probe does not cover these exact numerical source versions')
    return sorted(rows, key=lambda r: r['cells']), policies[0], windows, bindings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--local-audit', type=Path, required=True)
    parser.add_argument('--source-compatibility-review', type=Path,
                        help='Bound operator probe required when numerical source hashes differ')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be new to preserve evidence')
    rows, policy, windows, bindings = collect(args.runs, args.local_audit, args.source_compatibility_review)
    args.output.mkdir(parents=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), layout='constrained')
    n = [r['cells'] for r in rows]
    regions = [('all', 'Whole domain'), ('wall_adjacent', 'Wall owners'),
               ('near_body_fixed_window', 'Fixed near-body window'),
               ('near_tail_fixed_window', 'Fixed tail window')]
    for ax, suffix, title in zip(axes,
            ('_pressure_L2_percent', '_rms_error_percent_rho_U2', '_maximum_error_percent_rho_U2'),
            ('Reference-relative pressure L2 (%)', 'Pressure RMS / rho U² (%)', 'Maximum pressure error / rho U² (%)')):
        for region, label in regions:
            ax.loglog(n, [r[region + suffix] for r in rows], 'o-', label=label)
        ax.axhline(3, color='black', linestyle=':', linewidth=1, label='3% reference line')
        ax.set(xlabel='Actual volume cells', ylabel=title)
        ax.grid(alpha=.25, which='both')
    axes[0].legend(fontsize=8)
    fig.suptitle('Same numerical settings; global pressure gauge only\nMMS refinement diagnostic; no experimental or mesh-independence qualification')
    fig.savefig(args.output / 'pressure-refinement.png', dpi=190)
    fig.savefig(args.output / 'pressure-refinement.pdf')
    plt.close(fig)
    # Histories are bound by the independent audit together with the final
    # fields. Plot the worst of every original convergence gate, not a selected
    # residual that could hide unconverged pressure or reconstructed continuity.
    gates = ('velocity_change', 'pressure_change', 'flux_change', 'momentum_residual',
             'native_divergence_max', 'current_reconstructed_divergence_max', 'face_flux_consistency_max')
    history_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout='constrained')
    for run in args.runs:
        producer = json.loads((run / 'result.json').read_text())
        history = producer['history']
        iteration = [h['iteration'] for h in history]
        worst = [max(h[key] for key in gates) for h in history]
        axes[0].semilogy(iteration, worst, 'o-', label=f"{producer['cells']:,} cells")
        axes[1].semilogy(iteration, [h['pressure_change'] for h in history], 'o-', label=f"{producer['cells']:,} cells")
        for h, maximum in zip(history, worst):
            history_rows.append(dict(case=run.name, iteration=h['iteration'], maximum_original_gate=maximum,
                                     **{key: h[key] for key in gates}))
    for ax, label in zip(axes, ('Maximum of all seven original gates', 'Relative pressure change')):
        ax.axhline(policy['settings']['nonlinear_tolerance'], color='black', linestyle=':', label='Original tolerance')
        ax.set(xlabel='Accepted nonlinear update', ylabel=label)
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle('Actual archived nonlinear convergence; unchanged stopping criteria')
    fig.savefig(args.output / 'nonlinear-convergence.png', dpi=190)
    fig.savefig(args.output / 'nonlinear-convergence.pdf')
    plt.close(fig)
    with (args.output / 'nonlinear-history.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history_rows[0]))
        writer.writeheader()
        writer.writerows(history_rows)
    with (args.output / 'metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result = dict(rows=rows, numerical_policy=policy, fixed_spatial_windows_L=windows,
                  mesh_convergence_qualified=False, experimental_accuracy_qualified=False,
                  limitations='Changing local cell sizes and first-layer thickness is not uniform h refinement. No observed order or GCI is fitted. Fixed windows select centroids; wall-owner volume changes with refinement. The 3% lines refer to three distinct norms, not interchangeable acceptance gates.')
    (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    shutil.copy2(__file__, args.output / 'executed-plot.py')
    for name in ('result.json', 'manifest.json'):
        path = args.local_audit / name
        bindings[str(path.resolve())] = digest(path)
    artifacts = {p.name: digest(p) for p in args.output.iterdir() if p.is_file()}
    (args.output / 'manifest.json').write_text(json.dumps(dict(input_sha256=bindings, artifact_sha256=artifacts), indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
