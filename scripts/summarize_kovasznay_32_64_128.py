#!/usr/bin/env python3
"""Close CASE-FVM-KOVASZNAY-ORDER with a 32/64/128 grid-refinement study.

The feature-ledger acceptance for this anchor is

    "L2 convergence order >= 1.8 (nominal second order) on the three grids
     32/64/128, measured as the log-log slope"

so this script does three things, all from raw run output:

1. loads every per-grid ``cases.json`` written by
   ``scripts/verify_kovasznay_order.py`` and *recomputes* the relative L2
   errors from the saved cell-centred fields (``n<N>-fields.npz``).  Any
   disagreement with ``cases.json`` is a hard error, so the numbers printed in
   the report can never drift from the solution that produced them;
2. evaluates the criterion as a least-squares slope of ``log e`` against
   ``log h`` over the three levels, for the full 3D velocity, its in-plane
   (u, v) and spanwise (w) budgets, and for the pressure in both gauges;
3. writes ``summary.json``, ``RESULTS.md`` and the convergence/field figure
   straight from those numbers (no hand transcription).

Spanwise budget convention: the exact spanwise velocity is identically zero,
so a plain relative L2 error is undefined.  It is reported as the RMS of w
normalised by the exact *full* velocity norm,
``sqrt(sum V w^2 / sum V |u_exact|^2)``, i.e. the share of the total velocity
error carried by the spurious spanwise mode.

Reference: Kovasznay, L. I. G. (1948), "Laminar flow behind a two-dimensional
grid", Proc. Cambridge Philos. Soc. 44:58-62, doi:10.1017/S0305004100023999.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tensorfvm.native_manufactured3d import volume_weighted_error
from tensorfvm.verification.native_run import write_json

THRESHOLD = 1.8
RECOMPUTE_RTOL = 1e-10
#: Settings that must be identical on every level of a refinement sequence.  A
#: log-log slope is only a property of the *scheme* if the scheme is the same on
#: every level, so a mixed sequence is refused instead of silently fitted.
CONFIG_KEYS = ('reynolds', 'symmetry_closure', 'convection_order', 'algorithm')


def load_level(directory):
    """Load one level: the recorded row plus the fields needed to re-derive it."""
    directory = Path(directory)
    row = json.loads((directory / 'cases.json').read_text())[0]
    fields = np.load(directory / f"n{row['n']}-fields.npz")
    volumes = fields['cell_volumes']
    velocity, pressure = fields['velocity'], fields['pressure']
    exact_velocity, exact_pressure = fields['exact_velocity'], fields['exact_pressure']

    full = volume_weighted_error(velocity, exact_velocity, volumes)
    inplane = volume_weighted_error(velocity[:, :2], exact_velocity[:, :2], volumes)
    spanwise_rms = float(np.sqrt(np.sum(volumes * velocity[:, 2] ** 2)
                                 / np.sum(volumes * np.sum(exact_velocity ** 2, axis=1))))
    fixed = volume_weighted_error(pressure, exact_pressure, volumes)
    meanfree = volume_weighted_error(pressure, exact_pressure, volumes, remove_mean=True)

    # Anti-hand-transcription gate: the recorded numbers must be reproducible
    # from the dump alone.
    for name, recomputed in (('velocity_l2', full), ('pressure_l2', fixed),
                             ('pressure_mean_free_l2', meanfree)):
        recorded = float(row[name])
        if not np.isclose(recorded, recomputed, rtol=RECOMPUTE_RTOL, atol=1e-15):
            raise RuntimeError(f"n={row['n']} {name}: cases.json says {recorded!r} "
                               f"but the saved fields give {recomputed!r}")
    return dict(n=int(row['n']), h=float(row['h']), cells=int(row['cells']),
                converged=bool(row['numerically_converged']),
                configuration={key: row.get(key) for key in CONFIG_KEYS},
                outer_iterations=int(row['outer_iterations']),
                divergence_max=float(row['divergence_max']),
                global_momentum_imbalance=float(row['global_momentum_imbalance']),
                max_abs_z_velocity=float(row['max_abs_z_velocity']),
                velocity_full=full, velocity_inplane=inplane,
                velocity_spanwise=spanwise_rms,
                pressure_fixed=fixed, pressure_meanfree=meanfree,
                sampling_bias=row.get('sampling_bias'),
                solve_seconds=row.get('solve_seconds'),
                directory=str(directory))


def validate_levels(levels):
    """Refuse to fit a slope across levels that used different discretisations.

    ``free`` (traction-free) and ``zero_gradient`` (symmetry) closures of the
    extrusion planes give different spurious-spanwise budgets, so a sequence that
    mixes them -- e.g. a historical ``free`` level left in the same directory as
    a ``zero_gradient`` rerun -- produces a number that is not the order of any
    scheme.  The guard is also applied to Reynolds number, convection order and
    the linear-solve algorithm, which are equally able to change the observed
    order.  Returns the (single) configuration shared by all levels.
    """
    if not levels:
        raise SystemExit('no levels loaded')
    reference = dict(levels[0]['configuration'])
    for level in levels[1:]:
        other = level['configuration']
        diff = {key: (reference.get(key), other.get(key))
                for key in CONFIG_KEYS if other.get(key) != reference.get(key)}
        if diff:
            raise SystemExit(f"n={level['n']} was solved with a different configuration "
                             f"(reference n={levels[0]['n']}): {diff}")
    return reference


QUANTITIES = {
    'velocity_full': 'velocity L2 (full 3D u,v,w)',
    'velocity_inplane': 'velocity L2 (in-plane u,v)',
    'velocity_spanwise': 'spanwise w RMS / |u_exact|',
    'pressure_fixed': 'pressure L2 (fixed gauge)',
    'pressure_meanfree': 'pressure L2 (mean-free gauge)',
}


def observed_orders(levels, key):
    out = []
    for coarse, fine in zip(levels, levels[1:]):
        ratio = coarse['h'] / fine['h']
        out.append(dict(coarse=coarse['n'], fine=fine['n'], ratio=float(ratio),
                        order=float(np.log(coarse[key] / fine[key]) / np.log(ratio))))
    return out


def fit_slope(levels, key):
    x = np.log([level['h'] for level in levels])
    y = np.log([level[key] for level in levels])
    slope, intercept = np.polyfit(x, y, 1)
    predicted = slope * x + intercept
    ss_res = float(np.sum((y - predicted) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return dict(slope=float(slope), intercept=float(intercept),
                r_squared=float(1 - ss_res / ss_tot) if ss_tot > 0 else None)


def figure(levels, orders, fits, threshold, png_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    finest = levels[-1]
    fields = np.load(Path(finest['directory']) / f"n{finest['n']}-fields.npz")
    n = finest['n']
    velocity = fields['velocity'].reshape(n, n, 3)
    pressure = fields['pressure'].reshape(n, n)
    exact_u = fields['exact_velocity'].reshape(n, n, 3)
    centres = fields['cell_centers']

    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)

    ax = axes[0, 0]
    for key, style in (('velocity_full', 'o-'), ('velocity_inplane', 's--'),
                       ('pressure_meanfree', '^-.')):
        ax.loglog([level['h'] for level in levels], [level[key] for level in levels], style,
                  label=f"{QUANTITIES[key]} (slope {fits[key]['slope']:.3f})")
    h = np.array([level['h'] for level in levels])
    reference = levels[0]['velocity_full'] * (h / h[0]) ** 2
    ax.loglog(h, reference, 'k:', label='nominal second order (h^2)')
    ax.invert_xaxis()
    ax.set(xlabel='cell size h', ylabel='relative L2 error',
           title=f'Kovasznay Re=40 grid refinement {levels[0]["n"]}/'
                 f'{levels[1]["n"]}/{levels[-1]["n"]}')
    ax.legend(fontsize=7)

    ax = axes[0, 1]
    ax.semilogx([level['h'] for level in levels], [level['velocity_spanwise'] for level in levels],
                'v-', label='spanwise w (spurious)')
    ax.semilogx([level['h'] for level in levels], [level['velocity_inplane'] for level in levels],
                's--', label='in-plane u,v')
    ax.invert_xaxis()
    ax.set(xlabel='cell size h', ylabel='error share', title='Error budget: spanwise vs in-plane')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    speed = np.linalg.norm(velocity, axis=2)
    image = ax.imshow(speed, origin='lower', extent=(0, 1, 0, 1))
    ax.set(title=f'Computed speed |u|, n={n}', xlabel='x', ylabel='y')
    fig.colorbar(image, ax=ax)

    ax = axes[1, 1]
    dense = np.linspace(0, 1, 400)
    row = n // 2
    y = float(centres.reshape(n, n, 3)[row, 0, 1])
    from tensorfvm.verification.native_hybrid_ns import kovasznay_fields
    analytic = kovasznay_fields(np.column_stack((dense, np.full_like(dense, y),
                                                 np.full_like(dense, 0.5))), 40.)['velocity'][:, 0]
    x = (np.arange(n) + 0.5) / n
    ax.plot(dense, analytic, 'k-', label='Kovasznay 1948 analytic $u_x$')
    ax.plot(x, velocity[row, :, 0], 'o', ms=3, label=f'native FVM n={n}')
    ax.plot(x, exact_u[row, :, 0], 'x', ms=3, alpha=.5, label='exact at cell centres')
    ax.set(xlabel='x', ylabel='$u_x$', title=f'Centreline y={y:.4f}, z=0.5')
    ax.legend(fontsize=8)

    fig.savefig(png_path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=Path, required=True,
                        help='directory holding n32/, n64/, n128/ level directories')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--markdown', type=Path, default=None)
    parser.add_argument('--figure', type=Path, default=None)
    parser.add_argument('--title', default='CASE-FVM-KOVASZNAY-ORDER 32/64/128 order study')
    parser.add_argument('--require-grids', default='32,64,128')
    args = parser.parse_args()

    required = [int(value) for value in args.require_grids.split(',')]
    directories = sorted(Path(args.runs).glob('n*'),
                         key=lambda path: int(path.name.lstrip('n')) if path.name[1:].isdigit() else 0)
    levels = [load_level(directory) for directory in directories
              if directory.name[1:].isdigit() and int(directory.name[1:]) in required]
    levels = sorted({level['n']: level for level in levels}.values(), key=lambda level: level['n'])
    if [level['n'] for level in levels] != required:
        raise SystemExit(f'need levels {required}, got {[level["n"] for level in levels]}')
    configuration = validate_levels(levels)

    orders = {key: observed_orders(levels, key) for key in QUANTITIES}
    fits = {key: fit_slope(levels, key) for key in QUANTITIES}
    verdict = {
        'threshold': args.threshold,
        'quantity': 'L2 relative error log-log least-squares slope over '
                    + '/'.join(str(level['n']) for level in levels),
        'velocity_full_slope': fits['velocity_full']['slope'],
        'velocity_full_pass': bool(fits['velocity_full']['slope'] >= args.threshold),
        'velocity_inplane_slope': fits['velocity_inplane']['slope'],
        'velocity_inplane_pass': bool(fits['velocity_inplane']['slope'] >= args.threshold),
        'pressure_meanfree_slope': fits['pressure_meanfree']['slope'],
        'pressure_meanfree_pass': bool(fits['pressure_meanfree']['slope'] >= args.threshold),
        'all_levels_converged': bool(all(level['converged'] for level in levels)),
    }
    verdict['pass'] = bool(verdict['velocity_full_pass'] and verdict['all_levels_converged'])
    verdict['pass_inplane'] = bool(verdict['velocity_inplane_pass']
                                   and verdict['all_levels_converged'])

    args.output.mkdir(parents=True, exist_ok=True)
    summary = dict(title=args.title, levels=levels, observed_orders=orders, fits=fits,
                   verdict=verdict,
                   provenance=dict(runner='scripts/verify_kovasznay_order.py',
                                   summarizer='scripts/summarize_kovasznay_32_64_128.py',
                                   errors_recomputed_from='n<N>-fields.npz',
                                   recompute_rtol=RECOMPUTE_RTOL,
                                   configuration=configuration))
    write_json(args.output / 'summary.json', summary)

    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure(levels, orders, fits, args.threshold, args.figure)

    lines = [f'## {args.title}', '',
             'Verdict: ' + ('PASS' if verdict['pass'] else 'FAIL')
             + f" (full-3D velocity slope {fits['velocity_full']['slope']:.4f}"
             + f" vs threshold {args.threshold}); in-plane velocity slope "
             + f"{fits['velocity_inplane']['slope']:.4f}", '',
             '| n | h | cells | vel L2 (full) | vel L2 (u,v) | w RMS share | p L2 (mean-free) | converged |',
             '|---:|---:|---:|---:|---:|---:|---:|:--|']
    for level in levels:
        lines.append(f"| {level['n']} | {level['h']:.5f} | {level['cells']} | "
                     f"{level['velocity_full']:.6e} | {level['velocity_inplane']:.6e} | "
                     f"{level['velocity_spanwise']:.6e} | {level['pressure_meanfree']:.6e} | "
                     f"{'yes' if level['converged'] else 'NO'} |")
    lines += ['', '| quantity | slope | R^2 | verdict (>= '
              f'{args.threshold}) |', '|---|---:|---:|:--|']
    for key, label in QUANTITIES.items():
        fit = fits[key]
        passed = 'PASS' if fit['slope'] >= args.threshold else 'FAIL'
        lines.append(f"| {label} | {fit['slope']:.4f} | {fit['r_squared']:.6f} | {passed} |")
    markdown = '\n'.join(lines) + '\n'
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown)
    print(markdown)
    print('VERDICT', json.dumps(verdict))
    return 0 if verdict['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())