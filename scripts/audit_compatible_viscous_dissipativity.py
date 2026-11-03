"""Skew-family dissipativity audit of the viscous assemblies (no solver involved).

For every geometry of the ``two_tetrahedra`` family this records, for the legacy
fused assembly at three non-orthogonal-correction dampings and for the
energy-compatible (Galerkin) assembly, the symmetric-part minimum eigenvalue of
the momentum operator and the discrete kinetic-energy change rate of the
prescribed field (1,0,0) in cell 0.

Outputs: sweep.json, sweep.csv, report-table.md and dissipativity-sweep.png
next to this script's --out directory.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from tensorfvm.native_compatible_viscous3d import compatible_momentum_operators
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra


def legacy(mesh, mask, damping, velocity):
    viscous = NativeViscous3D(mesh, mask, 1., correction_damping=damping)
    matrix = viscous.momentum_diffusion_matrix.toarray()
    # Repo diagnostic convention: E = u^T (D A G) u, E < 0 means dissipative;
    # lambda_min is reported for -D A G so that lambda_min > 0 certifies dissipation.
    eigenvalue = float(np.linalg.eigvalsh(-.5 * (matrix + matrix.T))[0])
    return float(velocity.ravel() @ (matrix @ velocity.ravel())), eigenvalue


def compatible(mesh, mask, velocity):
    operators = compatible_momentum_operators(mesh, mask)
    dense = operators['matrix'].toarray()
    # Energy form K is PSD and approximates -div(2 mu S); in the fusion slot above
    # the equivalent matrix is -K, so E = -u^T K u and lambda_min(K) > 0 certify dissipation.
    eigenvalue = float(np.linalg.eigvalsh(.5 * (dense + dense.T))[0])
    return -float(velocity.ravel() @ (dense @ velocity.ravel())), eigenvalue


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='docs/verification-native-compatible-viscous')
    parser.add_argument('--points', type=int, default=41)
    arguments = parser.parse_args()
    out = Path(arguments.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for skew in np.linspace(0, 1, arguments.points):
        mesh = two_tetrahedra(float(skew)); mask = mesh.neighbor < 0
        velocity = np.zeros((len(mesh.cells), 3)); velocity[0, 0] = 1.
        from tensorfvm.native_viscous3d import NativeViscous3D as _  # noqa: F401
        angle = float(np.degrees(np.arccos(np.clip(
            np.sum(mesh.area_vectors * NativeViscous3D(mesh, mask, 1.).op.d, axis=1)
            / (np.linalg.norm(mesh.area_vectors, axis=1)
               * np.linalg.norm(NativeViscous3D(mesh, mask, 1.).op.d, axis=1)), -1, 1)).max()))
        row = {'skew': float(skew), 'max_nonorthogonality_degrees': angle}
        for damping in (1., .5, 0.):
            rate, eigenvalue = legacy(mesh, mask, damping, velocity)
            tag = f'legacy_damping_{damping:.1f}'
            row[f'{tag}_rate'] = rate; row[f'{tag}_lambda_min'] = eigenvalue
        rate, eigenvalue = compatible(mesh, mask, velocity)
        row['compatible_rate'] = rate; row['compatible_lambda_min'] = eigenvalue
        rows.append(row)
    (out / 'sweep.json').write_text(json.dumps(rows, indent=1))
    with (out / 'sweep.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows(rows)
    failures = [r for r in rows if r['legacy_damping_1.0_rate'] > 0]
    lines = ['| skew | max angle deg | legacy E (damping 1) | legacy E (damping 0) | compatible E | legacy lambda_min | compatible lambda_min |',
             '|---|---|---|---|---|---|---|']
    for row in rows:
        lines.append('| {:.3f} | {:.2f} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} |'.format(
            row['skew'], row['max_nonorthogonality_degrees'], row['legacy_damping_1.0_rate'],
            row['legacy_damping_0.0_rate'], row['compatible_rate'],
            row['legacy_damping_1.0_lambda_min'], row['compatible_lambda_min']))
    (out / 'report-table.md').write_text('\n'.join(lines) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    skews = [r['skew'] for r in rows]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for damping, style in ((1., '-o'), (.5, '-s'), (0., '-^')):
        axes[0].plot(skews, [r[f'legacy_damping_{damping:.1f}_rate'] for r in rows], style,
                     markersize=2.6, linewidth=1.1, label=f'legacy, damping={damping:.1f}')
    axes[0].plot(skews, [r['compatible_rate'] for r in rows], '-k', linewidth=1.6,
                 label='energy-compatible (Galerkin)')
    axes[0].axhline(0., color='grey', linewidth=.8)
    axes[0].set_yscale('symlog', linthresh=1e-3)
    axes[0].set_xlabel('geometry skew parameter'); axes[0].set_ylabel('discrete kinetic energy rate E')
    axes[0].set_title('E > 0 means the operator creates kinetic energy (dissipative: E < 0)')
    axes[0].legend(fontsize=7); axes[0].grid(alpha=.3)
    axes[1].plot(skews, [r['legacy_damping_1.0_lambda_min'] for r in rows], '-o', markersize=2.6,
                 linewidth=1.1, label='legacy (damping 1.0)')
    axes[1].plot(skews, [r['compatible_lambda_min'] for r in rows], '-k', linewidth=1.6,
                 label='energy-compatible')
    axes[1].axhline(0., color='grey', linewidth=.8)
    axes[1].set_yscale('symlog', linthresh=1e-6)
    axes[1].set_xlabel('geometry skew parameter'); axes[1].set_ylabel('min eigenvalue of symmetric part')
    axes[1].set_title('lambda_min > 0 certifies dissipativity')
    axes[1].legend(fontsize=7); axes[1].grid(alpha=.3)
    figure.tight_layout(); figure.savefig(out / 'dissipativity-sweep.png', dpi=150)
    figure.set_size_inches(7.2, 2.9); figure.savefig(out / 'dissipativity-sweep-report.png', dpi=72)
    print(json.dumps({'points': len(rows), 'legacy_energy_producing_points': len(failures),
                      'first_failing_skew': failures[0]['skew'] if failures else None,
                      'first_failing_angle': round(failures[0]['max_nonorthogonality_degrees'], 3) if failures else None,
                      'compatible_min_rate': min(r['compatible_rate'] for r in rows),
                      'compatible_min_lambda': min(r['compatible_lambda_min'] for r in rows),
                      'legacy_damping0_min_rate': min(r['legacy_damping_0.0_rate'] for r in rows),
                      'legacy_damping0_energy_producing_points': sum(1 for r in rows if r['legacy_damping_0.0_rate'] > 0),
                      'legacy_rate_at_skew1': rows[-1]['legacy_damping_1.0_rate'],
                      'legacy_rate_at_skew1_damping0': rows[-1]['legacy_damping_0.0_rate']}, indent=1))


if __name__ == '__main__':
    main()
