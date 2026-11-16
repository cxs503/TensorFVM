#!/usr/bin/env python3
"""Figure + numbers for the round-off grading note (``m23320``).

Reads only committed artifacts -- ``runs-32-64-128-zero_gradient/summary.json``
and the levels' ``cases.json`` -- and emits

    m23320-roundoff-grading.png   two panels: order ladder, spanwise budget
    (stdout)                      the numbers table, printed from the same data

    PYTHONPATH=src python docs/CASE-FVM-KOVASZNAY-ORDER/make_m23320_roundoff_figure.py

Panel (b) makes the grading rule visible: the three ``zero_gradient`` spanwise
shares sit 14--15 orders of magnitude below the round-off band, while a
``free``-closure level (documented value, this checkout has no such run) sits
*five* orders above the finest grid's real velocity error.  A single threshold
criterion therefore separates "the closure did its job" from "there is a real
spurious mode", and no fitted exponent is needed to do it.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs-32-64-128-zero_gradient'
SUMMARY = json.loads((RUNS / 'summary.json').read_text())
#: Documented ``free``-closure spurious spanwise velocity at n=32 (max|w|),
#: from ``m20571-artifact-schema-drift-diagnosis.md`` §2.  The run itself is not
#: part of this checkout, so it is drawn as an annotation, never as data.
FREE_CLOSURE_N32_MAX_W = 4.53e-03


def numbers():
    levels = SUMMARY['levels']
    fits = SUMMARY['fits']
    audit = SUMMARY['roundoff_grading']['velocity_spanwise']
    rows = []
    for level in levels:
        rows.append(dict(n=level['n'], h=level['h'],
                         velocity_full=level['velocity_full'],
                         pressure_meanfree=level['pressure_meanfree'],
                         spanwise_share=level['velocity_spanwise'],
                         max_abs_w=level['max_abs_z_velocity']))
    return dict(rows=rows, fits=fits, audit=audit, verdict=SUMMARY['verdict'])


def figure(path):
    data = numbers()
    rows = data['rows']
    h = np.array([row['h'] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)

    ax = axes[0]
    for key, style, label in (
            ('velocity_full', 'o-', 'velocity L2 (full 3D u,v,w)'),
            ('pressure_meanfree', '^-.', 'pressure L2 (mean-free gauge)')):
        ax.loglog(h, [row[key] for row in rows], style,
                  label=f"{label} (slope {data['fits'][key]['slope']:.3f})")
    ax.loglog(h, rows[0]['velocity_full'] * (h / h[0]) ** 2, 'k:',
              label='nominal second order ($h^2$)')
    ax.invert_xaxis()
    ax.set(xlabel='cell size h', ylabel='relative L2 error',
           title='Kovasznay Re=40, zero_gradient closure: order ladder 32/64/128')
    ax.legend(fontsize=7)

    ax = axes[1]
    shares = np.array([row['spanwise_share'] for row in rows])
    ax.loglog(h, shares, 'v-', color='tab:purple', label='spanwise $w$ RMS / $|u_{exact}|$')
    ax.loglog(h, [row['velocity_full'] for row in rows], 's--', color='tab:blue',
              label='in-plane/full velocity L2')
    band = data['audit']['limit']
    ax.axhline(band, color='tab:red', ls='--', lw=1,
               label=f'round-off band {band:.0e} (grading threshold)')
    ax.axhline(FREE_CLOSURE_N32_MAX_W, color='tab:orange', ls=':', lw=1,
               label=f'free closure, documented n=32 ({FREE_CLOSURE_N32_MAX_W:.2e})')
    for row in rows:
        ax.annotate(f"n={row['n']}", (row['h'], row['spanwise_share']),
                    textcoords='offset points', xytext=(4, 6), fontsize=7)
    ax.invert_xaxis()
    ax.set(xlabel='cell size h', ylabel='share of $|u_{exact}|$',
           ylim=(min(shares) / 10, FREE_CLOSURE_N32_MAX_W * 5),
           title='Spanwise budget is graded on size, not on an order\n'
                 f"max share {data['audit']['max_share']:.2e} "
                 f"<= {band:.0e} -> machine zero (n/a)")
    ax.legend(fontsize=7, loc='center left')
    fig.savefig(path, dpi=140)
    print(f'wrote {path}')


def main():
    figure(HERE / 'm23320-roundoff-grading.png')
    data = numbers()
    print('| n | h | vel L2 (full) | p L2 (mean-free) | spanwise share | max|w| |')
    print('|---:|---:|---:|---:|---:|---:|')
    for row in data['rows']:
        print(f"| {row['n']} | {row['h']:.5f} | {row['velocity_full']:.6e} | "
              f"{row['pressure_meanfree']:.6e} | {row['spanwise_share']:.6e} | "
              f"{row['max_abs_w']:.6e} |")
    print()
    for key, fit in data['fits'].items():
        print(f"{key}: slope={fit['slope']:.6f} r2={fit['r_squared']} "
              f"machine_zero={fit.get('machine_zero')} graded={fit.get('graded')}")
    verdict = data['verdict']
    print(f"verdict: pass={verdict['pass']} velocity_full={verdict['velocity_full_slope']:.6f} "
          f"spanwise_machine_zero={verdict['spanwise_machine_zero']} "
          f"spanwise_order_graded={verdict['spanwise_order_graded']}")


if __name__ == '__main__':
    main()