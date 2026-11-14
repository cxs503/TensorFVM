#!/usr/bin/env python3
"""CASE-FVM-CHANNEL-KMM180: grade Spalart--Allmaras on the anchor.

The anchor matrix (``docs/case-fvm-channel-kmm180/benchmark.json``) grades SST,
k-omega and k-epsilon.  This driver adds the repository's fourth closure, SA
(:mod:`tensorfvm.verification.kmm180_sa`), on exactly the same reference, the
same two grading windows and the same grading code, then writes a self-contained
artifact set (report + machine-readable rows + figures + manifest) that never
re-types a number by hand.

Run::

    PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180_sa.py \
        --out docs/case-fvm-channel-kmm180-sa
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from tensorfvm.verification.kmm180_sa import (self_check, solve_sa_channel)  # noqa: E402
from tensorfvm.verification.kmm1987 import (load_kmm1987_reference,  # noqa: E402
                                            validate_kmm1987)

MATRIX = ROOT / 'docs' / 'case-fvm-channel-kmm180' / 'benchmark.json'
GRIDS = (64, 128, 256)
FIRST_CELL_Y_PLUS = 0.3
SCHEMA = 'tensorfvm.case-fvm-channel-kmm180-sa/1'


def grade(n, dns=None):
    """Solve SA at one resolution and grade it exactly as the matrix does."""
    dns = load_kmm1987_reference() if dns is None else dns
    result = solve_sa_channel(n, re_tau=180.0, first_cell_y_plus=FIRST_CELL_Y_PLUS)
    validation = validate_kmm1987(result['y_plus'], result['u_plus'], result['Cf'],
                                  result['bulk_u_plus'], dns=dns)
    return result, validation


def rows_from_matrix(data):
    """Every graded row of the pinned matrix, whatever the container layout is."""
    out = {}
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            if {'log_region_rms', 'log_layer_rms'} <= set(item) and 'model' in item:
                out['%s_n%s' % (item['model'], item.get('n'))] = item
            else:
                stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='docs/case-fvm-channel-kmm180-sa')
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dns = load_kmm1987_reference()

    cases = {}
    profiles = {}
    for n in GRIDS:
        result, validation = grade(n, dns=dns)
        cases['sa_n%d' % n] = dict(
            model='sa', n=n, converged=bool(result['converged']),
            iterations=result['iterations'],
            relative_local_residual=result['relative_local_residual'],
            first_cell_y_plus=result['first_cell_y_plus'],
            Cf=result['Cf'], bulk_u_plus=result['bulk_u_plus'],
            log_region_rms=validation['log_region']['rms_relative'],
            log_region_max=validation['log_region']['max_relative'],
            log_layer_rms=validation['log_region_narrow']['rms_relative'],
            log_layer_max=validation['log_region_narrow']['max_relative'],
            cf_relative_error=validation['cf']['relative_error'],
            bulk_relative_error=validation['bulk_velocity']['relative_error'],
            verdict=validation['verdict'], verdict_log_layer=validation['verdict_log_layer'],
            overall_verdict=validation['overall_verdict'],
            self_check=self_check(result))
        profiles[n] = result

    matrix = json.loads(MATRIX.read_text())
    published = rows_from_matrix(matrix)
    reference = dict(re_tau=dns['re_tau'], source=str(dns.get('path', 'docs/rans-turbulence-validation-matrix/dns/0180')),
                     provenance=matrix.get('reference') if isinstance(matrix.get('reference'), dict) else None)

    # ---- machine artefacts -------------------------------------------------
    with (out / 'metrics.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['case', 'n', 'metric', 'value', 'unit'])
        for key, row in cases.items():
            for metric in ('log_region_rms', 'log_region_max', 'log_layer_rms', 'log_layer_max',
                           'cf_relative_error', 'bulk_relative_error', 'Cf', 'bulk_u_plus',
                           'iterations', 'relative_local_residual'):
                w.writerow([key, row['n'], metric, repr(row[metric]), '-'])

    n_fine = GRIDS[-1]
    with (out / 'profiles.csv').open('w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['n', 'y_plus', 'u_plus_sa', 'u_plus_dns', 'nut_plus', 'shear_plus', 'total_target'])
        ref = load_kmm1987_reference()
        for n in GRIDS:
            r = profiles[n]
            u_dns = np.interp(r['y_plus'], ref['y_plus'], ref['u_plus'])
            for i in range(len(r['y_plus'])):
                w.writerow([n, '%.6g' % r['y_plus'][i], '%.6g' % r['u_plus'][i],
                            '%.6g' % u_dns[i], '%.6g' % (r['nut'][i] * 180.0),
                            '%.6g' % r['shear'][i], '%.6g' % (1.0 - r['y'][i])])

    # ---- figures -----------------------------------------------------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    r = profiles[n_fine]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(dns['y_plus'], dns['u_plus'], 'k-', lw=1.6, label='Lee & Moser 2015 DNS')
    ax.semilogx(r['y_plus'], r['u_plus'], 'C3--', lw=1.6, label='SA, n=%d' % n_fine)
    ax.axvspan(30, 0.2 * 180, color='C0', alpha=.18, label='physical log layer y+<=0.2 Re_tau')
    ax.set_xlabel('y+'); ax.set_ylabel('u+'); ax.set_title('SA channel at Re_tau=180 vs DNS')
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(out / 'fig_velocity_profile.png', dpi=140); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for n in GRIDS:
        rr = profiles[n]
        mask = (rr['y_plus'] >= 1) & (rr['y_plus'] <= 0.9 * 180)
        u_dns = np.interp(rr['y_plus'], dns['y_plus'], dns['u_plus'])
        ax.semilogx(rr['y_plus'][mask], 100 * (rr['u_plus'][mask] - u_dns[mask]) / u_dns[mask],
                    label='n=%d' % n, lw=1.3)
    ax.axhline(5, color='k', ls=':', lw=1); ax.axhline(-5, color='k', ls=':', lw=1)
    ax.axvline(30, color='C0', ls='--', lw=1)
    ax.set_xlabel('y+'); ax.set_ylabel('100 (u+_SA - u+_DNS)/u+_DNS')
    ax.set_title('SA deviation from the DNS reference'); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out / 'fig_deviation.png', dpi=140); plt.close(fig)

    # ---- verdict summary ---------------------------------------------------
    fine = cases['sa_n%d' % n_fine]
    rows = {k: v for k, v in published.items() if k.split('_n')[0] in ('sst', 'komega', 'kepsilon')}
    summary = dict(
        schema=SCHEMA, case='CASE-FVM-CHANNEL-KMM180', closure='Spalart-Allmaras (1994)',
        re_tau=180.0, first_cell_y_plus=FIRST_CELL_Y_PLUS, grids=list(GRIDS),
        tolerance=0.05, reference=reference,
        cases=cases,
        published_matrix_rows={k: dict(log_region_rms=v.get('log_region_rms'),
                                       log_layer_rms=v.get('log_layer_rms'),
                                       verdict=v.get('verdict'),
                                       verdict_log_layer=v.get('verdict_log_layer'))
                               for k, v in sorted(rows.items())},
        sa_verdict=dict(log_region_rms=fine['log_region_rms'],
                        log_layer_rms=fine['log_layer_rms'],
                        legacy=fine['verdict'], log_layer=fine['verdict_log_layer']),
        environment=dict(python=platform.python_version(), numpy=np.__version__,
                         machine=platform.machine()),
        repro_cmd='PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180_sa.py --out %s' % out,
    )
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True))

    manifest = dict(schema=SCHEMA, artifacts={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in sorted(out.iterdir())
                                              if p.is_file() and p.name != 'manifest.json'})
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps(summary['sa_verdict'], indent=1))
    print(json.dumps({k: (v['log_region_rms'], v['log_layer_rms']) for k, v in cases.items()}, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())