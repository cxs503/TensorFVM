#!/usr/bin/env python3
"""Round-3 (m16838): merge the per-level zero-gradient payloads into one ladder,
evaluate the pre-declared C1-C8 criteria, and render table + figure from JSON only.

Inputs (raw payloads written by the instrument itself):
  * ``runs/lower_8_16/summary.json``  -- ``--levels 8 16 --closure zero_gradient``
  * ``runs/level_32/summary.json``    -- ``--levels 32 --closure zero_gradient``
  * ``runs/control_n8.json``          -- the free-traction / exact-Dirichlet control

Outputs:
  * ``runs/ladder_round3.json``  -- merged native payload + machine verdict
  * ``numbers_round3.md``        -- the report tables (zero hand-typed numbers)
  * ``fig_round3_ladder.png``    -- error ladder + observed orders vs the 1.8 gate
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification.kovasznay_order_closure import (
    evaluate, merge_tables, observed_orders)

HERE = Path(__file__).resolve().parent
RAW = HERE / 'runs'
GATE = 1.8


def load(path):
    return json.loads(Path(path).read_text())


def main():
    lower = load(RAW / 'lower_8_16' / 'summary.json')
    fine = load(RAW / 'level_32' / 'summary.json')
    control = load(RAW / 'control_n8.json')

    table = merge_tables(lower['table'], fine['table'])
    # the free-traction / exact-Dirichlet control is only archived as three scalar rows
    table['symmetry'] = [dict(n=int(control['symmetry']['n']), closure='symmetry',
                              inplane_l2=float(control['symmetry']['inplane_l2']),
                              spanwise_l2=float(control['symmetry']['spanwise_l2']),
                              max_abs_w=float(control['symmetry']['max_abs_w']))]
    tied = sorted(table['zero_gradient'], key=lambda r: r['n'])
    orders = observed_orders(tied)
    verdict = evaluate(table)
    payload = dict(levels=[int(r['n']) for r in tied], closures=sorted(table),
                   sources=[str(RAW / 'lower_8_16' / 'summary.json'),
                            str(RAW / 'level_32' / 'summary.json')],
                   table=table, orders=orders, verdict=verdict)
    (RAW / 'ladder_round3.json').write_text(json.dumps(payload, indent=2, default=float) + '\n')

    # ---- tables -------------------------------------------------------------
    lines = ['### 3.1 tied zero-gradient ladder (raw payload, one process per level)', '',
             '| n | cells | faces | dofs | total L2 | in-plane L2 | spanwise L2 | max\\|w\\| | tie residual | Picard | s | peak RSS MB |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|']
    for r in tied:
        lines.append('| %d | %d | %d | %s | %.6e | %.6e | %.1e | %.1e | %.1e | %s | %.1f | %.1f |' % (
            r['n'], r['cells'], r['faces'], r.get('dofs', 'n/a'), r['total_l2'],
            r['inplane_l2'], r['spanwise_l2'], r['max_abs_w'], r['tie_residual'],
            r['converged'], r['seconds'], r.get('peak_rss_mb', float('nan'))))
    lines += ['', '### 3.2 observed orders on every available pair', '',
              '| pair | total | in-plane | pressure (mean-free) | vs 1.8 gate |',
              '|---|---:|---:|---:|---|']
    for o in orders:
        lines.append('| %s | %.4f | %.4f | %.4f | %s |' % (
            o['pair'], o['total'], o['inplane'], o['pressure_mean_free'],
            'PASS' if o['total'] >= GATE else 'FAIL'))
    fit = verdict.get('three_level_fit')
    if fit:
        lines += ['', '### 3.3 three-level fit ``f(h) = L + C h^p`` (levels %s)' %
                  '/'.join(str(n) for n in fit['levels']), '',
                  '| p | stderr | convergence | monotone | asymptotic | C extrapolated | GCI |',
                  '|---:|---:|---|---|---|---:|---:|',
                  '| %.4f | %s | %s | %s | %s | %.6e | %.4f |' % (
                      fit['order'], ('nan (0 dof)' if not math.isfinite(fit['order_stderr'])
                                     else '%.3f' % fit['order_stderr']),
                      fit['convergence'], fit['monotone'], fit['asymptotic'],
                      fit['extrapolated'], fit['gci'])]
    lines += ['', '### 3.4 machine verdict (pre-declared criteria)', '',
              '| id | statement | value | verdict |', '|---|---|---|---|']
    for c in verdict['checks']:
        value = c['value']
        if isinstance(value, float):
            value = '%.6g' % value
        lines.append('| %s | %s | %s | **%s** |' % (c['id'], c['statement'], value, c['verdict']))
    lines += ['', 'anchor_closed: **%s**; not_run: %s' % (verdict['anchor_closed'],
                                                         verdict['not_run'])]
    (HERE / 'numbers_round3.md').write_text('\n'.join(lines) + '\n')

    # ---- figure -------------------------------------------------------------
    levels = np.array([r['n'] for r in tied], float)
    inplane = np.array([r['inplane_l2'] for r in tied])
    span = np.array([r['spanwise_l2'] for r in tied])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(levels, inplane, 'o-', label='in-plane (u,v), tied zero-gradient closure')
    ax.loglog(levels, np.maximum(span, 1e-20), 's--', label='spanwise w (exact value 0)')
    ax.loglog([8], [control['symmetry']['spanwise_l2']], 'kx', ms=9,
              label='spanwise w, delivered free-traction closure (n=8)')
    ideal = inplane[0] * (levels[0] / levels) ** 2
    ax.loglog(levels, ideal, ':', color='grey', label='O(h^2) reference slope')
    ax.set_xlabel('n  (n x n x 1 extruded mesh)')
    ax.set_ylabel('volume weighted L2 error')
    ax.set_title('Kovasznay Re=40: two-order link at n=32 (tied closure)')
    ax.grid(True, which='both', alpha=.3)
    ax.legend(fontsize=7, loc='lower left')

    ax = axes[1]
    ax.axhline(GATE, color='r', ls='--', label='acceptance gate 1.8')
    labels = [o['pair'] for o in orders]
    ax.bar(labels, [o['total'] for o in orders], width=.45,
           label='total velocity L2 order (tied closure)')
    for i, o in enumerate(orders):
        ax.text(i, o['total'] + .04, '%.3f' % o['total'], ha='center', fontsize=9)
    ax.axhline(1.808, color='grey', ls=':', label='delivered closure, in-plane, 16->32 (m16396)')
    ax.set_ylim(0, 2.3)
    ax.set_ylabel('observed order')
    ax.set_title('observed order per grid pair')
    ax.legend(fontsize=7)
    ax.grid(True, axis='y', alpha=.3)
    fig.tight_layout()
    fig.savefig(HERE / 'fig_round3_ladder.png', dpi=150)

    print('\n'.join(lines))
    print('wrote', RAW / 'ladder_round3.json')


if __name__ == '__main__':
    main()