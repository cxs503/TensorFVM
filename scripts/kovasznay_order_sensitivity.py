#!/usr/bin/env python3
"""Sensitivity (negative) control for the Kovasznay 32/64/128 order verdict.

``scripts/summarize_kovasznay_32_64_128.py`` answers the ledger question ("is the
log-log slope over 32/64/128 >= 1.8?").  It cannot, on its own, answer the
question a reviewer actually asks about that number: *can this estimator say
no?*  A slope measurement that returns ~2 for any scheme is worthless, so the
verdict has to be accompanied by a deliberately degraded run of the **same**
case, the **same** mesh family and the **same** closure, in which only the
expected order changes.

This script pairs two committed series:

``second order`` (reference)
    ``convection_order=2``, ``zero_gradient`` extrusion closure -- the series that
    carries the acceptance verdict.
``first order`` (control)
    ``convection_order=1`` (upwind), everything else identical.  Classical
    first-order upwind must show an O(h) velocity error, i.e. a slope near 1,
    and must therefore *fail* the >= 1.8 gate.

It refits both slopes from the raw per-grid field dumps (``n<N>-fields.npz``),
so neither verdict can drift from the solution that produced it, and it refuses
to report a discrimination margin unless the two series differ **only** in the
convection order (a control that also changed the closure or the Reynolds number
would prove nothing).

Exit status is 0 only when all of the following hold, which is what makes this a
gate rather than a plot:

1. the reference series reaches the ledger threshold on the full 3D velocity;
2. the first-order control is *rejected* by that same threshold;
3. the two slopes are separated by at least ``--min-margin`` (default 0.5).

If the control fails to separate, the honest reading is that the acceptance
number is not evidence of second order at all; the script then writes the
artifacts and exits 1.

Reference: Kovasznay, L. I. G. (1948), "Laminar flow behind a two-dimensional
grid", Proc. Cambridge Philos. Soc. 44:58-62, doi:10.1017/S0305004100023999.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from summarize_kovasznay_32_64_128 import (CONFIG_KEYS, QUANTITIES, fit_slope, load_level,
                                          observed_orders, validate_levels)
from tensorfvm.verification.native_run import write_json

THRESHOLD = 1.8
#: Slope separation that a genuine scheme-order change must produce.  Half an
#: order of magnitude is far outside the fitting noise of these three levels
#: (R^2 > 0.9999 on the reference series) and far inside the first-to-second
#: order gap of upwind vs central differencing.
MIN_MARGIN = .5
#: A first-order control above this slope would not be rejected by the >= 1.8
#: gate, i.e. the gate would be blind to a real first-order defect.
CONTROL_CEILING = 1.5


def load_series(directory, grids, label):
    """Load and cross-check one refinement series from its raw field dumps."""
    directory = Path(directory)
    missing = [n for n in grids if not (directory / f'n{n}' / 'cases.json').is_file()]
    if missing:
        raise SystemExit(f'{label}: levels {missing} not run in {directory}')
    levels = [load_level(directory / f'n{n}') for n in grids]
    if [level['n'] for level in levels] != list(grids):
        raise SystemExit(f'{label}: grid sequence {[level["n"] for level in levels]} '
                         f'does not match {list(grids)}')
    configuration = validate_levels(levels)
    return dict(label=label, directory=str(directory), levels=levels,
                configuration=configuration)


def compare(reference, control, threshold, min_margin):
    """Refit both slopes and decide whether the estimator discriminates."""
    differing = {key: (reference['configuration'].get(key), control['configuration'].get(key))
                 for key in CONFIG_KEYS
                 if reference['configuration'].get(key) != control['configuration'].get(key)}
    premise_ok = differing == {'convection_order': (2, 1)}
    def safe_slope(levels, key):
        """Slope only where the quantity is defined and positive on every level.

        A quantity that is exactly zero (e.g. the spurious spanwise velocity under
        the zero-gradient closure, or a round-off-level value that can come out
        negative) has no log, so ``None`` is recorded instead of a silent NaN.
        """
        values = [level[key] for level in levels]
        if any(not np.isfinite(value) or value <= 0 for value in values):
            return None
        return float(fit_slope(levels, key)['slope'])

    slopes = {series['label']: {key: safe_slope(series['levels'], key) for key in QUANTITIES}
              for series in (reference, control)}
    orders = {series['label']: {key: observed_orders(series['levels'], key)
                                for key in ('velocity_full', 'pressure_meanfree')}
              for series in (reference, control)}
    ref_slope = slopes[reference['label']]['velocity_full']
    ctl_slope = slopes[control['label']]['velocity_full']
    margin = ref_slope - ctl_slope
    converged = all(level['converged'] for series in (reference, control)
                    for level in series['levels'])
    verdict = {
        'threshold': threshold,
        'min_margin': min_margin,
        'control_ceiling': CONTROL_CEILING,
        'quantity': 'full-3D velocity relative L2 error, log-log least-squares slope',
        'configurations_differ_only_in': differing,
        'control_premise_ok': bool(premise_ok),
        'all_levels_converged': bool(converged),
        'reference_slope': float(ref_slope),
        'control_slope': float(ctl_slope),
        'margin': float(margin),
        'reference_passes_threshold': bool(ref_slope >= threshold),
        'control_rejected_by_threshold': bool(ctl_slope < threshold),
        'control_below_ceiling': bool(ctl_slope <= CONTROL_CEILING),
        'margin_reached': bool(margin >= min_margin),
    }
    verdict['pass'] = bool(verdict['control_premise_ok'] and verdict['all_levels_converged']
                           and verdict['reference_passes_threshold']
                           and verdict['control_rejected_by_threshold']
                           and verdict['control_below_ceiling']
                           and verdict['margin_reached'])
    return dict(slopes=slopes, orders=orders, verdict=verdict)


def figure(reference, control, result, png_path, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    ax = axes[0]
    colours = {reference['label']: 'tab:blue', control['label']: 'tab:red'}
    for series in (reference, control):
        h = np.array([level['h'] for level in series['levels']])
        e = np.array([level['velocity_full'] for level in series['levels']])
        slope = result['slopes'][series['label']]['velocity_full']
        ax.loglog(h, e, 'o-', color=colours[series['label']],
                  label=f"{series['label']} (slope {slope:.3f})")
    h = np.array([level['h'] for level in reference['levels']])
    e0 = reference['levels'][0]['velocity_full']
    for power, style in ((1, 'k--'), (2, 'k:')):
        ax.loglog(h, e0 * (h / h[0]) ** power, style, label=f'h^{power} reference')
    ax.invert_xaxis()
    ax.set(xlabel='cell size h', ylabel='relative L2 error (full 3D velocity)',
           title='Kovasznay Re=40: scheme-order sensitivity')
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = [reference['label'], control['label']]
    values = [result['slopes'][label]['velocity_full'] for label in labels]
    bars = ax.bar(labels, values, color=[colours[label] for label in labels], alpha=.85)
    ax.axhline(threshold, color='k', ls='--', label=f'ledger threshold {threshold}')
    ax.set(ylabel='fitted velocity slope', ylim=(0, max(2.2, max(values) * 1.15)),
           title=f"margin {result['verdict']['margin']:.3f} "
                 f"(need >= {result['verdict']['min_margin']})")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + .03, f'{value:.3f}',
                ha='center', fontsize=9)
    ax.legend(fontsize=8)
    fig.savefig(png_path, dpi=170)
    plt.close(fig)


def markdown(reference, control, result):
    verdict = result['verdict']
    lines = ['## Kovasznay order estimator: negative control', '',
             'Verdict: ' + ('PASS' if verdict['pass'] else 'FAIL')
             + f" -- reference slope {verdict['reference_slope']:.4f}, control slope "
               f"{verdict['control_slope']:.4f}, margin {verdict['margin']:.4f}", '',
             '| series | convection order | closure | velocity slope | p mean-free slope | '
             'reaches >= ' + f"{verdict['threshold']} |",
             '|---|:--:|:--:|---:|---:|:--|']
    for series in (reference, control):
        label = series['label']
        lines.append(f"| {label} | {series['configuration'].get('convection_order')} | "
                     f"{series['configuration'].get('symmetry_closure')} | "
                     f"{result['slopes'][label]['velocity_full']:.4f} | "
                     f"{result['slopes'][label]['pressure_meanfree']:.4f} | "
                     f"{'yes' if result['slopes'][label]['velocity_full'] >= verdict['threshold'] else 'no'} |")
    lines += ['', '| series | n | h | vel L2 (full) | p L2 (mean-free) | converged |',
              '|---|---:|---:|---:|---:|:--|']
    for series in (reference, control):
        for level in series['levels']:
            lines.append(f"| {series['label']} | {level['n']} | {level['h']:.6f} | "
                         f"{level['velocity_full']:.6e} | {level['pressure_meanfree']:.6e} | "
                         f"{'yes' if level['converged'] else 'NO'} |")
    lines += ['', 'Gates: ' + json.dumps({key: verdict[key] for key in
                                          ('control_premise_ok', 'all_levels_converged',
                                           'reference_passes_threshold',
                                           'control_rejected_by_threshold',
                                           'control_below_ceiling', 'margin_reached')})]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--reference', type=Path, required=True,
                        help='convection_order=2 series (the acceptance evidence)')
    parser.add_argument('--control', type=Path, required=True,
                        help='convection_order=1 series (the negative control)')
    parser.add_argument('--grids', default='32,64,128')
    parser.add_argument('--threshold', type=float, default=THRESHOLD)
    parser.add_argument('--min-margin', type=float, default=MIN_MARGIN)
    parser.add_argument('--output', type=Path, required=True,
                        help='JSON summary path')
    parser.add_argument('--markdown', type=Path, default=None)
    parser.add_argument('--figure', type=Path, default=None)
    args = parser.parse_args()

    grids = tuple(int(value) for value in args.grids.split(','))
    reference = load_series(args.reference, grids, 'second order (convection_order=2)')
    control = load_series(args.control, grids, 'first order (convection_order=1)')
    result = compare(reference, control, args.threshold, args.min_margin)

    summary = dict(title='Kovasznay 32/64/128 order estimator: negative control',
                   reference=dict(label=reference['label'],
                                  directory=reference['directory'],
                                  configuration=reference['configuration'],
                                  levels=reference['levels']),
                   control=dict(label=control['label'],
                                directory=control['directory'],
                                configuration=control['configuration'],
                                levels=control['levels']),
                   slopes=result['slopes'], observed_orders=result['orders'],
                   verdict=result['verdict'],
                   provenance=dict(
                       runner='scripts/verify_kovasznay_order.py',
                       summarizer='scripts/kovasznay_order_sensitivity.py',
                       errors_recomputed_from='n<N>-fields.npz',
                       reference='"Kovasznay (1948) exact solution; second order is the '
                                 'nominal scheme order under test"',
                       control='"same case, upwind convection (order 1); a first-order '
                               'defect the gate must reject"'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, summary)
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(reference, control, result))
    if args.figure:
        args.figure.parent.mkdir(parents=True, exist_ok=True)
        figure(reference, control, result, args.figure, args.threshold)
    print(markdown(reference, control, result))
    print('VERDICT', json.dumps(result['verdict']))
    return 0 if result['verdict']['pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())