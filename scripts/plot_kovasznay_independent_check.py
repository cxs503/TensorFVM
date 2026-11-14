"""Figure for the independent Kovasznay reference audit (CASE-FVM-KOVASZNAY-ORDER).

Reads the JSON produced by ``scripts/verify_kovasznay_reference_independent.py``
and the committed field dumps; writes a 4-panel PNG.  No number is typed by hand:
every curve comes from the JSON / npz files.

    PYTHONPATH=src python scripts/plot_kovasznay_independent_check.py \
        --json docs/CASE-FVM-KOVASZNAY-ORDER/independent_reference_check.json \
        --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
        --png docs/CASE-FVM-KOVASZNAY-ORDER/independent-reference-check.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', type=Path, required=True)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--png', type=Path, required=True)
    args = ap.parse_args(argv)

    report = json.loads(args.json.read_text())
    levels = report['levels']
    h = np.array([row['h'] for row in levels])
    est = report['independent_estimates']

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.5))

    # (a) convergence: independent recompute, h^2 reference, GCI of the finest
    ax = axes[0][0]
    ax.loglog(h, est['velocity_full']['per_level'], 'o-', color='tab:blue',
              label=f"velocity L2 (slope {est['velocity_full']['slope']:.4f})")
    ax.loglog(h, est['pressure_meanfree']['per_level'], 's-', color='tab:red',
              label=f"pressure L2, mean-free (slope {est['pressure_meanfree']['slope']:.4f})")
    ref = est['velocity_full']['per_level'][0] * (h / h[0]) ** 2
    ax.loglog(h, ref, 'k--', lw=1.0, label='nominal second order (h^2)')
    gci = report['richardson']['velocity_full']['gci_finest']
    ax.errorbar([h[-1]], [est['velocity_full']['per_level'][-1]],
                yerr=[[0.0], [gci]], fmt='none', ecolor='tab:blue', capsize=4,
                label=f'GCI(Fs=1.25) on finest = {gci:.2e}')
    ax.set(xlabel='cell size h', ylabel='relative L2 error',
           title='(a) independent recompute of the 32/64/128 ladder')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    # (b) sampling bias and spurious spanwise share (error budget)
    ax = axes[0][1]
    if report['sampling_bias'] is not None:
        ax.loglog(h, report['sampling_bias']['per_level'], '^-', color='tab:green',
                  label=f"centre-sampling bias (order {report['sampling_bias']['order']:.4f})")
    ax.loglog(h, est['velocity_spanwise']['per_level'], 'v-', color='tab:purple',
              label='spurious spanwise share (round-off)')
    ax.set(xlabel='cell size h', ylabel='error share',
           title='(b) error budget: sampling truncation is O(h^2), w stays at round-off')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    # (c) analytic reference field + discrete field at the finest grid
    ax = axes[1][0]
    finest = max(levels, key=lambda row: row['n'])['n']
    npz = np.load(args.runs / f'n{finest}' / f'n{finest}-fields.npz')
    centres = npz['cell_centers']
    x, y = centres[:, 0], centres[:, 1]
    n = finest
    uu = npz['velocity'][:, 0].reshape(n, n)
    xs = np.unique(np.round(x, 12))
    ys = np.unique(np.round(y, 12))
    cs = ax.contourf(xs, ys, uu.T, levels=20, cmap='viridis')
    fig.colorbar(cs, ax=ax, label='u_x (discrete, n=%d)' % n)
    ax.set(xlabel='x', ylabel='y', title=f'(c) discrete u_x at the finest grid (n={n})')
    ax.set_aspect('equal')

    # (d) scoreboard: published vs independent slope, threshold line
    ax = axes[1][1]
    keys = ['velocity_full', 'velocity_inplane', 'pressure_meanfree']
    published = report['cross_check_vs_summary']
    xpos = np.arange(len(keys))
    ax.bar(xpos - 0.2, [published[key]['slope'] for key in keys], width=0.4,
           label='published summary')
    ax.bar(xpos + 0.2, [est[key]['slope'] for key in keys], width=0.4,
           label='independent recompute')
    ax.axhline(1.8, color='k', ls='--', lw=1.0, label='acceptance threshold 1.8')
    for i, key in enumerate(keys):
        ax.annotate(f"|diff|={published[key]['slope_diff']:.1e}",
                    (i, max(published[key]['slope'], est[key]['slope']) + 0.02),
                    ha='center', fontsize=8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(['velocity (3D)', 'velocity (in-plane)', 'pressure (mean-free)'])
    ax.set(ylabel='log-log slope', ylim=(0, 2.3), title='(d) acceptance criterion, two code paths')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    fig.suptitle('CASE-FVM-KOVASZNAY-ORDER: independent audit of the 1948 reference and the '
                 'measured order', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    args.png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.png, dpi=130)
    print(f'wrote {args.png}')


if __name__ == '__main__':
    main()