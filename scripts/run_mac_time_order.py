"""Publish the periodic MAC temporal-order study (floors, orders, figures).

Writes ``scripts/evidence/periodic-mac-temporal-order/study.json`` plus the
figures embedded by the report.  The artefact sits under ``scripts/`` because the
solver-development worktree is a sparse cone (``src/ tests/ scripts/``): a path
under ``docs/`` cannot be committed from it, and evidence that only exists in a
full checkout is evidence nobody can re-derive.  Every number in the JSON comes straight from
:func:`tensorfvm.verification.mac_time_order.study`; nothing is transcribed.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from tensorfvm.verification.mac_time_order import (
    TemporalStudyConfig,
    run_midpoint,
    study,
    taylor_green_exact_velocity,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fig_orders(summary, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for grid in summary['grids']:
        dts = grid['time_steps_s']
        steps = [dt / dts[0] for dt in dts[:-1]]  # one difference per dt level pair
        for key, label in (('velocity', 'face velocity'), ('pressure', 'projection pressure'),
                           ('first_order_control_velocity', 'forward-Euler control')):
            values = grid[key]['differences']
            axes[0].loglog(steps, values, 'o-', label=f"n={grid['n']} {label}")
        axes[1].semilogx(steps, [grid['velocity']['differences'][0] * s ** -2 for s in steps],
                         'k--', lw=1, label='slope -2 reference')
    axes[0].set_xlabel('dt / dt_coarsest')
    axes[0].set_ylabel(r'$||u_{dt}-u_{dt/2}||_2/||u_{dt/2}||_2$')
    axes[0].set_title('Self-convergence differences (floor-free)')
    axes[0].grid(True, which='both', alpha=0.3)
    axes[0].legend(fontsize=6)

    labels, values, colours = [], [], []
    for grid in summary['grids']:
        for key, colour in (('velocity', 'tab:blue'), ('pressure', 'tab:orange'),
                            ('first_order_control_velocity', 'tab:green')):
            labels.append(f"n={grid['n']}\n{key.replace('_velocity','').replace('first_order_control','Euler')}")
            values.append(grid[key]['richardson_order'])
            colours.append(colour)
    axes[1].bar(range(len(values)), values, color=colours)
    axes[1].axhline(2.0, color='k', ls=':', lw=1)
    axes[1].axhline(1.0, color='k', ls='--', lw=1)
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_xticklabels(labels, fontsize=5, rotation=90)
    axes[1].set_ylabel('observed order (Richardson)')
    axes[1].set_title('Observed temporal order')
    axes[1].grid(True, axis='y', alpha=0.3)
    figure.tight_layout()
    figure.savefig(out / 'temporal-order.png', dpi=160)
    plt.close(figure)


def fig_floor(summary, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(5.6, 4.0))
    for grid in summary['grids']:
        dts = grid['time_steps_s']
        axes.plot(dts, [row['velocity_error_vs_exact'] for row in grid['error_vs_exact']],
                  'o-', label=f"n={grid['n']} velocity")
        axes.plot(dts, [row['kinetic_energy_error_vs_exact'] for row in grid['error_vs_exact']],
                  's--', label=f"n={grid['n']} kinetic energy")
    axes.axhline(0.03, color='r', ls=':', label='3% criterion')
    axes.set_xscale('log')
    axes.set_yscale('log')
    axes.set_xlabel('dt [s]')
    axes.set_ylabel('relative error vs exact Taylor-Green')
    axes.set_title('Error vs exact solution saturates: the spatial floor')
    axes.grid(True, which='both', alpha=0.3)
    axes.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(out / 'error-floor.png', dpi=160)
    plt.close(figure)


def fig_field(summary, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    config = TemporalStudyConfig(n=32, planes=4, final_time_s=1.0, time_steps_s=(0.05,))
    solver = run_midpoint(config, 0.05)
    exact = taylor_green_exact_velocity(solver, config.viscosity_pa_s, config.final_time_s,
                                        config.amplitude)
    x, y, _ = solver.centers()
    figure, axes = plt.subplots(1, 4, figsize=(15.0, 3.6))
    panels = [
        (solver.velocity[0].mean(0), r'computed $u_x$'),
        (exact[0].mean(0), r'exact $u_x$'),
        ((solver.velocity[0] - exact[0]).mean(0), r'$u_x$ difference'),
        (solver.pressure.mean(0) - solver.pressure.mean() * 0, r'projection pressure $p$'),
    ]
    for axis, (field, title) in zip(axes, panels):
        image = axis.pcolormesh(x.mean(0), y.mean(0), field, shading='auto', cmap='RdBu_r')
        axis.set_title(title, fontsize=9)
        axis.set_aspect('equal')
        figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle('Taylor-Green fields at t = 1.0 s, n = 32, dt = 0.05 s (z-mean slice)', fontsize=10)
    figure.tight_layout()
    figure.savefig(out / 'fields.png', dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('scripts/evidence/periodic-mac-temporal-order'))
    parser.add_argument('--figures', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    summary = study()
    summary['cfl_note'] = ('grids refine dt in step with h, so dt/h and the convective CFL '
                           'number are held fixed; n=32 with dt=0.2 s is rejected by the '
                           'Picard iteration (residual NaN) and is never reported as a result')
    summary['velocity_order_passed'] = all(
        1.95 < grid['velocity']['richardson_order'] < 2.05 for grid in summary['grids'])
    summary['pressure_first_order_confirmed'] = all(
        0.9 < grid['pressure']['richardson_order'] < 1.15 for grid in summary['grids'])
    summary['control_first_order_confirmed'] = all(
        0.9 < grid['first_order_control_velocity']['richardson_order'] < 1.15
        for grid in summary['grids'])
    summary['physical_accuracy_3_percent_passed'] = all(
        row['velocity_error_vs_exact'] < 0.03 and row['kinetic_energy_error_vs_exact'] < 0.03
        for grid in summary['grids'] for row in grid['error_vs_exact'])
    if args.figures:
        fig_orders(summary, args.output)
        fig_floor(summary, args.output)
        fig_field(summary, args.output)
        summary['figures'] = {name: sha(args.output / name)
                             for name in ('temporal-order.png', 'error-floor.png', 'fields.png')}
    (args.output / 'study.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({key: summary[key] for key in
                      ('velocity_order_passed', 'pressure_first_order_confirmed',
                       'control_first_order_confirmed', 'physical_accuracy_3_percent_passed')},
                     indent=2))
    for grid in summary['grids']:
        print(f"n={grid['n']} velocity order {grid['velocity']['richardson_order']:.4f} "
              f"pressure order {grid['pressure']['richardson_order']:.4f} "
              f"control {grid['first_order_control_velocity']['richardson_order']:.4f} "
              f"u_floor {grid['error_vs_exact'][-1]['velocity_error_vs_exact']:.3e}")


if __name__ == '__main__':
    main()