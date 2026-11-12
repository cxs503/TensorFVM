#!/usr/bin/env python3
"""Figure + number table for docs/verification-kovasznay-order-closure.

Every number is read from the raw verification payloads written by
``python -m tensorfvm.verification.kovasznay_order_closure`` / the ladder probe
(``runs/kovasznay-order-closure/{zero_gradient.json,control_n8.json}``).
Nothing is typed by hand into the report tables.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RAW = HERE / 'runs'          # in-repo mirror (runs/ is outside the sparse checkout)
if not RAW.is_dir():
    RAW = REPO / 'runs/kovasznay-order-closure'
GATE = 1.8


def main():
    tied = json.loads((RAW / 'zero_gradient.json').read_text())
    control = json.loads((RAW / 'control_n8.json').read_text())
    levels = sorted((int(k) for k in tied))
    inplane = np.array([tied[str(n)]['inplane_l2'] for n in levels])
    span = np.array([tied[str(n)]['spanwise_l2'] for n in levels])
    pairs = [(a, b, math.log(tied[str(a)]['inplane_l2'] / tied[str(b)]['inplane_l2']) / math.log(b / a))
             for a, b in zip(levels, levels[1:])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(levels, inplane, 'o-', label='in-plane (u,v), tied zero-gradient closure')
    ax.loglog(levels, np.maximum(span, 1e-20), 's--', label='spanwise w (exact value 0)')
    ax.loglog([8], [control['symmetry']['inplane_l2']], 'k*', ms=11,
              label='in-plane, delivered free-traction closure (n=8)')
    ax.loglog([8], [control['symmetry']['spanwise_l2']], 'kx', ms=9,
              label='spanwise, delivered free-traction closure (n=8)')
    ax.loglog([8], [control['dirichlet']['inplane_l2']], 'k+', ms=10,
              label='in-plane, exact-tangential Dirichlet control (n=8)')
    ideal = inplane[0] * (levels[0] / np.array(levels, float)) ** 2
    ax.loglog(levels, ideal, ':', color='grey', label='O(h^2) reference slope')
    ax.set_xlabel('n  (n x n x 1 extruded mesh)')
    ax.set_ylabel('volume weighted L2 error')
    ax.set_title('Kovasznay Re=40: tie closure removes the O(h) spanwise artefact')
    ax.grid(True, which='both', alpha=.3)
    ax.legend(fontsize=6.5, loc='lower left')

    ax = axes[1]
    ax.axhline(GATE, color='r', ls='--', label='acceptance gate 1.8')
    if pairs:
        ax.bar([f'{a}->{b}' for a, b, _ in pairs], [p for _, _, p in pairs], width=.5,
               label='in-plane observed order (tied closure)')
        for i, (_, _, p) in enumerate(pairs):
            ax.text(i, p + .04, f'{p:.3f}', ha='center', fontsize=9)
    ax.axhline(1.808, color='grey', ls=':', label='delivered closure, in-plane, 16->32 (m16396)')
    ax.set_ylim(0, 2.3)
    ax.set_ylabel('observed order')
    ax.set_title('order on the available ladder pairs')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_order_closure.png', dpi=150)

    lines = ['### 3.1 ladder under the tied zero-gradient closure (raw payload)', '',
             '| n | cells | faces | total L2 | in-plane L2 | spanwise L2 | max\\|w\\| | tie residual | Picard conv | s |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|']
    for n in levels:
        r = tied[str(n)]
        lines.append('| %d | %d | %d | %.4e | %.4e | %.4e | %.4e | %.1e | %s | %.1f |' % (
            n, r['cells'], r['faces'], r['total_l2'], r['inplane_l2'], r['spanwise_l2'],
            r['max_abs_w'], r['tie_residual'], r['converged'], r['seconds']))
    lines += ['', '### 3.2 boundary-closure control at n = 8 (same mesh, same discretisation)', '',
              '| closure | in-plane L2 | spanwise L2 | max\\|w\\| |', '|---|---:|---:|---:|']
    for name in ('symmetry', 'dirichlet', 'zero_gradient'):
        r = control[name]
        lines.append('| %s | %.4e | %.4e | %.4e |' % (name, r['inplane_l2'], r['spanwise_l2'], r['max_abs_w']))
    lines += ['', '### 3.3 observed orders (tied closure, in-plane; total == in-plane because w == 0)', '',
              '| pair | in-plane (== total) order | vs 1.8 gate |', '|---|---:|---|']
    for a, b, p in pairs:
        lines.append('| %d -> %d | %.3f | %s |' % (a, b, p, 'PASS' if p >= GATE else 'FAIL'))
    lines.append('')
    (HERE / 'numbers.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
