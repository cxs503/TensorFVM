#!/usr/bin/env python3
"""Figure for the 2026-11-16 close-out sync (m21068 line, mission m22605).

Nothing in this figure is read from ``summary.json``: the L2 errors are
recomputed here from each level's committed ``n<N>-fields.npz`` with
``tensorfvm.native_manufactured3d.volume_weighted_error`` (the same helper the
summarizer uses), so the plot is an independent read of the same raw fields.

Left panel  — velocity / mean-free pressure relative L2 vs h on log-log axes,
              with least-squares slopes and h^1 / h^2 reference guides.
Right panel — n=128 centreline u_x at y≈0.5 against the Kovasznay solution
              (point values, not interpolated).

Writes ``closeout-regen-m22605.png`` next to this file.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.native_manufactured3d import volume_weighted_error

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs-32-64-128-zero_gradient'
GRIDS = (32, 64, 128)


def level_errors(n):
    fields = np.load(RUNS / f'n{n}' / f'n{n}-fields.npz')
    volumes = fields['cell_volumes']
    return dict(
        h=1.0 / n,
        velocity=volume_weighted_error(fields['velocity'], fields['exact_velocity'], volumes),
        pressure=volume_weighted_error(fields['pressure'], fields['exact_pressure'], volumes,
                                       remove_mean=True),
    )


def slope(errors, key):
    h = np.log([e['h'] for e in errors])
    e = np.log([e[key] for e in errors])
    return float(np.polyfit(h, e, 1)[0])


def main():
    errors = [level_errors(n) for n in GRIDS]
    summary = json.loads((RUNS / 'summary.json').read_text())

    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.0, 4.4))

    h = np.array([e['h'] for e in errors])
    for key, label, marker in (('velocity', 'velocity L2 (u,v,w)', 'o'),
                               ('pressure', 'pressure L2 (mean-free)', 's')):
        e = np.array([x[key] for x in errors])
        ax.loglog(h, e, marker=marker, label=f'{label}: slope {slope(errors, key):.4f}')
        for x, y in zip(h, e):
            ax.annotate(f'{y:.2e}', (x, y), textcoords='offset points', xytext=(6, -10),
                        fontsize=7)
    guide = errors[0]['velocity'] * (h / h[0]) ** 2
    ax.loglog(h, guide, 'k--', lw=1.0, label=r'reference $h^2$')
    ax.loglog(h, errors[0]['velocity'] * (h / h[0]), 'k:', lw=1.0, label=r'reference $h^1$')
    ax.set_xlabel('h')
    ax.set_ylabel('relative L2 error')
    ax.set_title('Kovasznay Re=40: error vs h (recomputed from npz)')
    ax.invert_xaxis()
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=8)

    fields = np.load(RUNS / 'n128' / 'n128-fields.npz')
    centers = fields['cell_centers']
    band = np.abs(centers[:, 1] - 0.5) <= 0.5 * (1.0 / 128)
    order = np.argsort(centers[band, 0])
    bx.plot(centers[band, 0][order], fields['velocity'][band, 0][order], 'o', ms=3,
            label='FVM n=128 (cell values)')
    bx.plot(centers[band, 0][order], fields['exact_velocity'][band, 0][order], '-', lw=1.2,
            label='Kovasznay (1948) analytic')
    bx.set_xlabel('x')
    bx.set_ylabel('$u_x$ at y=0.5')
    bx.set_title('n=128 centreline velocity')
    bx.grid(True, alpha=0.3)
    bx.legend(fontsize=8)
    bx.text(0.02, 0.04,
            f"committed summary slope {summary['fits']['velocity_full']['slope']:.4f}",
            transform=bx.transAxes, fontsize=8)

    fig.tight_layout()
    out = HERE / 'closeout-regen-m22605.png'
    fig.savefig(out, dpi=130)
    print('wrote', out)
    for e in errors:
        print(e)


if __name__ == '__main__':
    main()