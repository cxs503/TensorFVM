#!/usr/bin/env python3
"""Field and error maps for the CASE-FVM-KOVASZNAY-ORDER level dumps.

Reads a committed ``n<N>-fields.npz`` (cell centres, volumes, discrete and
analytic velocity) and draws the streamwise velocity, the analytic solution and
the pointwise error, so the reported convergence order has a picture attached to
the same numbers the gate recomputes.  Nothing is hand-copied: every pixel comes
from the npz the acceptance test reads.

Usage::

    PYTHONPATH=src python scripts/plot_kovasznay_field_error.py \
        --fields docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n128/n128-fields.npz \
        --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n128/fields-error.png

The 2-D case is a symmetry-extruded n x n x 1 mesh, so the cell centres are
re-sorted onto the (z, y, x) index lattice before being reshaped; the script
refuses a level whose centres do not form that lattice instead of silently
plotting a scrambled field.
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def lattice(centers, shape=None):
    """Re-order cell centres onto their (z, y, x) index lattice."""
    nz = len(np.unique(np.round(centers[:, 2], 12)))
    ny = len(np.unique(np.round(centers[:, 1], 12)))
    nx = len(np.unique(np.round(centers[:, 0], 12)))
    if nx * ny * nz != len(centers):
        raise SystemExit(f'cell centres do not form a structured lattice: '
                         f'{nx}x{ny}x{nz} != {len(centers)}')
    order = np.lexsort((centers[:, 0], centers[:, 1], centers[:, 2]))
    if shape is not None and (nx, ny, nz) != tuple(shape):
        raise SystemExit(f'lattice {nx}x{ny}x{nz} != expected {tuple(shape)}')
    return order, (nz, ny, nx)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fields', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--component', type=int, default=0,
                        help='velocity component to plot (0=x, 1=y, 2=z)')
    args = parser.parse_args()

    data = np.load(args.fields)
    centers = data['cell_centers']
    order, (nz, ny, nx) = lattice(centers)
    velocity = data['velocity'][order, args.component].reshape(nz, ny, nx)[0]
    exact = data['exact_velocity'][order, args.component].reshape(nz, ny, nx)[0]
    pressure = data['pressure'][order].reshape(nz, ny, nx)[0]
    exact_pressure = data['exact_pressure'][order].reshape(nz, ny, nx)[0]
    x = np.unique(np.round(centers[:, 0], 12))
    y = np.unique(np.round(centers[:, 1], 12))
    if len(x) != nx or len(y) != ny:
        raise SystemExit('non-uniform spacing: this plotter assumes a uniform lattice')
    dx, dy = x[1] - x[0], y[1] - y[0]
    extent = (x[0] - dx / 2, x[-1] + dx / 2, y[0] - dy / 2, y[-1] + dy / 2)

    error = np.abs(velocity - exact)
    levels = np.linspace(min(velocity.min(), exact.min()), max(velocity.max(), exact.max()), 41)

    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), constrained_layout=True)
    for ax, field, title, kw in [
        (axes[0], velocity, f'native FVM $u_x$, n={nx}', dict(cmap='viridis', levels=levels)),
        (axes[1], exact, 'Kovasznay 1948 analytic $u_x$', dict(cmap='viridis', levels=levels)),
        (axes[2], error, '$|u_x - u_x^{exact}|$', dict(cmap='magma', levels=41)),
        (axes[3], pressure - exact_pressure, '$p - p^{exact}$ (fixed gauge)',
         dict(cmap='coolwarm', levels=41)),
    ]:
        mesh = ax.contourf(*np.meshgrid(x, y, indexing='ij'), field.T, **kw)
        fig.colorbar(mesh, ax=ax, shrink=.85)
        ax.set(xlabel='x', ylabel='y', title=title, aspect='equal')
    rel = error.max() / np.abs(exact).max()
    axes[2].text(.03, 1.03, f'max|$\\Delta u_x$| = {error.max():.3e} '
                            f'({100 * rel:.3f}% of max $|u_x|$)',
                 transform=axes[2].transAxes, fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f'wrote {args.output} (n={nx}, max|du_x|={error.max():.6e})')


if __name__ == '__main__':
    main()