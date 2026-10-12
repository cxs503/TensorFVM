#!/usr/bin/env python3
"""Compare archived, numerically converged SUBOFF grid-study reports.

Surface Cp differences are grid sensitivity, not experimental validation.
Fixed first-cell height means this is not a formal isotropic GCI study.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--reports', type=Path, nargs='+', required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
edges = np.linspace(0, 1, 101)
x = (edges[:-1] + edges[1:]) / 2
rows, profiles = [], []
baseline = None
for report in a.reports:
    status = json.loads((report / 'status.json').read_text())
    audit = json.loads((report / 'convergence-audit.json').read_text())
    if not audit['numerically_converged'] or not status['numerically_converged']:
        raise ValueError(f'grid not numerically converged: {report}')
    provenance = json.loads((report / 'evidence/provenance.json').read_text())
    configuration = {key: provenance[key] for key in [
        'stl_sha256', 'reynolds', 'first_layer_thickness_L', 'layers_requested',
        'growth_requested', 'turbulence_inlet', 'geometry', 'domain_L',
    ]}
    for name in ['system/fvSchemes', 'system/fvSolution',
                 'constant/momentumTransport', 'constant/physicalProperties']:
        configuration[name] = (report / 'evidence/case-input' / name).read_text()
    if baseline is None:
        baseline = configuration
    elif configuration != baseline:
        raise ValueError(f'geometry or physical/numerical settings differ: {report}')
    cp = np.loadtxt(report / 'evidence/wall-cp-pilot.csv.gz', delimiter=',', skiprows=1)
    wall = np.loadtxt(report / 'evidence/wall-yplus.csv.gz', delimiter=',', skiprows=1)
    np.testing.assert_allclose(cp[:, :3], wall[:, :3], rtol=0, atol=1e-12)
    weights = wall[:, -1]
    area = np.histogram(cp[:, 0], bins=edges, weights=weights)[0]
    weighted = np.histogram(cp[:, 0], bins=edges, weights=weights * cp[:, 3])[0]
    profile = np.divide(weighted, area, out=np.full_like(area, np.nan), where=area > 0)
    profiles.append(profile)
    row = {key: status[key] for key in [
        'cells', 'wall_faces', 'last_iteration', 'last_cd', 'reference_cd',
        'cd_pressure', 'cd_viscous', 'provisional_drag_relative_error_percent',
        'area_fraction_with_layers', 'area_fraction_six_layers',
        'last_yplus', 'max_nonorthogonality_deg', 'extended_mesh_ok',
    ]}
    row.update(label=report.name, cd_mean_last_200=audit['force_windows']['cd']['mean'],
               numerically_converged=True, physical_benchmark_passed=False)
    rows.append(row)

comparisons = []
for i in range(1, len(rows)):
    valid = np.isfinite(profiles[i]) & np.isfinite(profiles[i-1])
    difference = profiles[i][valid] - profiles[i-1][valid]
    pair = dict(from_grid=rows[i-1]['label'], to_grid=rows[i]['label'],
                cp_common_bins=int(valid.sum()), cp_bin_width_L=.01,
                cp_rmse_absolute=float(np.sqrt(np.mean(difference**2))),
                cp_max_absolute_difference=float(np.max(np.abs(difference))))
    for name in ['last_cd', 'cd_pressure', 'cd_viscous']:
        pair[name + '_relative_change_percent'] = float(100 * (rows[i][name] / rows[i-1][name] - 1))
    comparisons.append(pair)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
for row, profile in zip(rows, profiles):
    label = f"{row['label']} ({row['cells']:,})"
    axes[0].plot(x, profile, label=label, lw=1.2)
    axes[1].plot(x, profile, label=label, lw=1.2)
axes[0].set(xlabel='x/L', ylabel='Area-weighted bin mean Cp', title='Surface pressure: grid sensitivity')
axes[1].set(xlabel='x/L', ylabel='Area-weighted bin mean Cp', title='Stern detail', xlim=(.8, 1))
axes[0].legend(fontsize=7)
fig.savefig(a.output / 'cp-grid-comparison.png', dpi=180)
fig.savefig(a.output / 'cp-grid-comparison.pdf')
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
cells = np.array([row['cells'] for row in rows]) / 1e6
for key, label in [('last_cd', 'Total Cd'), ('cd_pressure', 'Pressure Cd'), ('cd_viscous', 'Viscous Cd')]:
    axes[0].plot(cells, [row[key] for row in rows], 'o-', label=label)
axes[0].axhline(rows[0]['reference_cd'], color='k', ls='--', label='Total drag reference')
axes[0].set(xlabel='Cells (million)', ylabel='Cd (frontal area)', title='Converged drag components')
axes[0].legend(fontsize=8)
axes[1].plot(cells, [row['provisional_drag_relative_error_percent'] for row in rows], 'o-')
axes[1].axhline(3, color='k', ls='--', label='3% target')
axes[1].set(xlabel='Cells (million)', ylabel='Absolute relative error (%)', title='Total drag vs provisional reference')
axes[1].legend()
fig.savefig(a.output / 'drag-grid-comparison.png', dpi=180)
fig.savefig(a.output / 'drag-grid-comparison.pdf')
plt.close(fig)
np.savetxt(a.output / 'cp-bin-profiles.csv', np.column_stack([x, *profiles]), delimiter=',',
           header='x_L,' + ','.join(row['label'] for row in rows), comments='')
(a.output / 'comparison.json').write_text(json.dumps(dict(
    grids=rows, adjacent_comparisons=comparisons,
    cp_metric='Absolute Cp differences between common 0.01L area-weighted surface bins; no experimental Cp comparison',
    study='Engineering grid sensitivity at fixed first-layer height; no formal GCI claim',
    physical_benchmark_passed=False), indent=2, allow_nan=False) + '\n')
print(json.dumps(comparisons, indent=2))
