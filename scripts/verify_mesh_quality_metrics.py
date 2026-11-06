#!/usr/bin/env python3
"""Generate the mesh-quality metric evidence bundle (JSON, markdown, figure).

Every number in the emitted markdown is generated from the same audit run, so
the report table is never transcribed by hand.
"""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorfvm.verification.mesh_quality_audit import GATE_DEGREES, audit


def figure(evidence, path):
    rows = evidence['shear_closed_form']
    figure_, axes = plt.subplots(1, 2, figsize=(9.6, 4.), constrained_layout=True)
    shear = [row['shear_deg'] for row in rows]
    axes[0].plot(shear, [row['measured_nonorthogonality_deg'] for row in rows], 'o', ms=5,
                 label='measured (mesh_quality)')
    axes[0].plot([0, 80], [0, 80], 'k--', lw=.8, label='arctan(gamma) closed form')
    axes[0].axhline(GATE_DEGREES, color='crimson', lw=.9, ls=':',
                    label=f'gate {GATE_DEGREES:.0f}$^\\circ$')
    axes[0].set(xlabel='shear angle (deg)', ylabel='non-orthogonality (deg)',
                title='Metric reproduces the closed form exactly')
    axes[0].legend(fontsize=7, loc='upper left')
    axes[1].plot(shear, [row['measured_min_projected_distance_m'] * 1e3 for row in rows], 's-', ms=4,
                 label='minimum projected distance')
    axis = axes[1].twinx()
    axis.plot(shear, [1. if row['gate_passed'] else 0. for row in rows], 'v', color='crimson', ms=5,
              label='gate passed')
    axis.set(ylabel='gate (1/0)', yticks=[0, 1])
    axes[1].set(xlabel='shear angle (deg)', ylabel='min projected distance (mm)',
                title='Gate flips at %.1f$^\\circ$' % evidence['gate_flip']['flip_angle_deg'])
    figure_.savefig(path, dpi=200)
    plt.close(figure_)


def markdown(evidence):
    rows = evidence['shear_closed_form']
    lines = ['| shear (deg) | measured non-orth. (deg) | closed form error (deg) | skewness | min projected dist. (mm) | gate |',
             '|---|---|---|---|---|---|']
    for row in rows:
        lines.append('| %.0f | %.6f | %.2e | %.1e | %.5f | %s |' % (
            row['shear_deg'], row['measured_nonorthogonality_deg'], row['absolute_error_deg'],
            row['measured_skewness'], row['measured_min_projected_distance_m'] * 1e3,
            'pass' if row['gate_passed'] else 'reject'))
    curved = evidence['curved_channel']['shipped']
    lines += ['', '| curved-channel item | shipped value | independent recomputation | |Δ| |', '|---|---|---|---|']
    for key in ('nonorthogonality_max_deg', 'nonorthogonality_p95_deg', 'internal_skewness_max',
                'minimum_face_projected_distance_m', 'minimum_cell_area_m2'):
        lines.append('| %s | %.15g | %.15g | %.1e |' % (
            key, curved[key], evidence['curved_channel']['independent'][key],
            evidence['curved_channel']['differences'][key]))
    lines += ['', '| control | observed | verdict |', '|---|---|---|']
    flip = evidence['gate_flip']
    lines.append('| gate flip angle | %.10f deg | documented %.0f deg, |Δ| = %.2e deg |' % (
        flip['flip_angle_deg'], flip['documented_gate_deg'], flip['absolute_error_deg']))
    face = evidence['inverted_face_control']
    lines.append('| one flipped face orientation | non-orth %.1f deg, positive projected distances = %s | gate %s |' % (
        face['nonorthogonality_max_deg'], face['positive_projected_distances'],
        'passed (WRONG)' if face['gate_passed'] else 'rejected'))
    cell = evidence['inverted_cell_control']
    lines.append('| one negative cell area (%.3e m2) | positive cell areas = %s | gate %s |' % (
        cell['inverted_cell_area_m2'], cell['positive_cell_areas'],
        'passed (WRONG)' if cell['gate_passed'] else 'rejected'))
    lines.append('| rigid rotation 0/17/43/90 deg | non-orth. spread %.2e deg | invariant |' % (
        max(abs(row['nonorthogonality_max_deg'] - evidence['rotation_invariance'][0]['nonorthogonality_max_deg'])
            for row in evidence['rotation_invariance'])))
    lines += ['', '**verdict: %s**' % ('PASS' if evidence['verdict'] else 'FAIL')]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('scripts/verification-mesh-quality'))
    parser.add_argument('--nx', type=int, default=8)
    parser.add_argument('--ny', type=int, default=4)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)
    evidence = audit(arguments.nx, arguments.ny)
    priced = json.dumps(evidence, sort_keys=True, indent=2)
    (arguments.out / 'results.json').write_text(priced + '\n')
    (arguments.out / 'report-table.md').write_text(markdown(evidence))
    figure(evidence, arguments.out / 'nonorthogonality-vs-shear.png')
    summary = dict(verdict=evidence['verdict'],
                   gate_flip_deg=evidence['gate_flip']['flip_angle_deg'],
                   max_closed_form_error_deg=max(row['absolute_error_deg'] for row in evidence['shear_closed_form']),
                   independent_recompute_difference=evidence['curved_channel']['max_absolute_difference'],
                   sha256=hashlib.sha256(priced.encode()).hexdigest())
    print(json.dumps(summary, indent=2))
    if not evidence['verdict']:
        raise SystemExit('mesh quality audit failed')


if __name__ == '__main__':
    main()