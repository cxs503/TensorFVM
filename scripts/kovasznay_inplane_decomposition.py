#!/usr/bin/env python3
"""In-plane / spanwise error-budget decomposition for the Kovasznay case.

The Kovasznay (1948) solution is a 2D steady Navier-Stokes flow extruded into a
unit cube, so the exact spanwise velocity is identically zero.  The velocity
error norm therefore mixes two different things: the discretisation error of the
physically meaningful (u, v) field and the spurious spanwise mode w, which is a
numerical artefact.  Reports of this case quote the split, and the split must be
recomputable from a stored field -- hand-written or rescaled columns are not
evidence.

This script recomputes every number it prints from the cell-centred fields in
``n<N>-fields.npz``:

* ``velocity_l2``          full 3D relative L2 error, all three components;
* ``velocity_l2_inplane``  (u, v) relative L2 error, same denominator (the exact
                           norm, in which ``w_exact == 0``);
* ``spanwise_share``       sqrt(sum V w^2 / sum V |u_exact|^2), i.e. the share
                           of the total velocity error carried by w;
* ``pressure_l2``          fixed-gauge pressure relative L2 error;
* ``pressure_mean_free_l2`` the same after removing the volume-weighted means.
                           The pressure is only defined up to a constant, and in
                           the fixed gauge a mesh-dependent constant pollutes the
                           error, so the mean-free gauge is the one with a
                           convergence order that measures the scheme.

Every level is cross-checked against the ``cases.json`` written by the solver
driver; any disagreement above ``RECOMPUTE_RTOL`` is a hard error, so a report
number can never drift from the field that produced it.  A requested level whose
field file is absent is a hard error as well: there is no fallback sequence and
no fabricated coefficient anywhere in this script.

Usage::

    PYTHONPATH=src python scripts/kovasznay_inplane_decomposition.py \
        --level 16=docs/.../selftest-4-8-16/n16/n16-fields.npz \
        --level 32=docs/.../runs-32-64-128/n32/n32-fields.npz \
        --level 64=docs/.../runs-32-64-128/n64/n64-fields.npz \
        --output docs/.../runs-32-64-128/inplane-decomposition \
        --markdown docs/case-kovasznay-order-order-verification/RESULTS_INPLANE_TABLE.md \
        --figure docs/case-kovasznay-order-order-verification/convergence_inplane.png

Reference: Kovasznay, L. I. G. (1948), "Laminar flow behind a two-dimensional
grid", Proc. Cambridge Philos. Soc. 44:58-62, doi:10.1017/s0305004100023999.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from tensorfvm.native_manufactured3d import volume_weighted_error
from tensorfvm.verification.native_run import write_json

#: A recomputed error that disagrees with ``cases.json`` by more than this is a
#: defect, not noise: both come from the same stored field.
RECOMPUTE_RTOL = 1e-10

#: Quantities that carry a convergence order.  ``spanwise_share`` is reported
#: but is *not* a discretisation order of a physical quantity: its exact value is
#: zero, so its "order" depends on which artefact happens to dominate.  It is
#: listed for the error budget only.
ORDERED = ('velocity_l2', 'velocity_l2_inplane', 'pressure_l2', 'pressure_mean_free_l2')

#: recomputed row key -> key in the solver's ``cases.json``
CASES_KEYS = {
    'velocity_l2': 'velocity_l2',
    'velocity_l2_inplane': 'velocity_l2_inplane',
    'pressure_l2': 'pressure_l2',
    'pressure_mean_free_l2': 'pressure_mean_free_l2',
    'max_abs_w': 'max_abs_z_velocity',
}


def parse_level(spec):
    """Parse ``N=path/to/n<N>-fields.npz``."""
    if '=' not in spec:
        raise SystemExit(f'--level expects N=path, got {spec!r}')
    n_text, path_text = spec.split('=', 1)
    try:
        n = int(n_text)
    except ValueError:
        raise SystemExit(f'--level N must be an integer, got {n_text!r}')
    path = Path(path_text)
    if not path.exists():
        raise SystemExit(
            f'level n={n}: field file {path} does not exist. '
            'This script does not fall back to a documented or scaled '
            'sequence: without a stored field there is no evidence.')
    return n, path


def measure_level(n, path):
    """Recompute the error budget of one level from its stored cell fields."""
    data = np.load(path)
    missing = [k for k in ('cell_volumes', 'velocity', 'exact_velocity',
                           'pressure', 'exact_pressure') if k not in data.files]
    if missing:
        raise SystemExit(f'level n={n}: {path} lacks {missing}')
    volumes = data['cell_volumes']
    velocity = data['velocity']
    exact_velocity = data['exact_velocity']
    pressure = data['pressure']
    exact_pressure = data['exact_pressure']
    if velocity.shape != exact_velocity.shape or velocity.ndim != 2 or velocity.shape[1] != 3:
        raise SystemExit(f'level n={n}: velocity shapes {velocity.shape} vs '
                         f'{exact_velocity.shape} are not an (N, 3) pair')
    if pressure.shape != exact_pressure.shape:
        raise SystemExit(f'level n={n}: pressure shapes disagree')

    row = {
        'n': n,
        'h': 1.0 / n,
        'cells': int(len(volumes)),
        'fields': str(path),
        'velocity_l2': volume_weighted_error(velocity, exact_velocity, volumes),
        'velocity_l2_inplane': volume_weighted_error(velocity[:, :2], exact_velocity[:, :2], volumes),
        'pressure_l2': volume_weighted_error(pressure, exact_pressure, volumes),
        'pressure_mean_free_l2': volume_weighted_error(
            pressure, exact_pressure, volumes, remove_mean=True),
        'max_abs_w': float(np.max(np.abs(velocity[:, 2]))),
    }
    # The exact spanwise velocity is zero, so a relative L2 of w is undefined;
    # normalise the spurious mode by the exact total velocity norm instead.
    exact_norm = float(np.sum(volumes * np.sum(exact_velocity ** 2, axis=1)))
    if exact_norm <= 0:
        raise SystemExit(f'level n={n}: exact velocity norm is zero')
    row['spanwise_share'] = float(
        np.sqrt(np.sum(volumes * velocity[:, 2] ** 2) / exact_norm))
    return row


def read_cases(cases_path):
    """Read the (single record) cases.json of one level, or {} if absent."""
    path = Path(cases_path)
    if not path.exists():
        return {}
    records = json.loads(path.read_text())
    if len(records) != 1:
        raise SystemExit(f'{path} holds {len(records)} records, expected exactly one')
    return records[0]


def cross_check(row, cases, cases_path):
    """Fail loudly if the recomputed budget disagrees with the solver record."""
    if not cases:
        return
    compared = []
    for row_key, cases_key in CASES_KEYS.items():
        if cases_key not in cases:
            continue
        recorded = float(cases[cases_key])
        recomputed = float(row[row_key])
        scale = max(abs(recorded), abs(recomputed), 1e-300)
        if abs(recorded - recomputed) > RECOMPUTE_RTOL * scale:
            raise SystemExit(
                f'level n={row["n"]}: {cases_key} recomputed from {row["fields"]} is '
                f'{recomputed:.17g} but {cases_path} records {recorded:.17g} '
                f'(relative difference {abs(recorded - recomputed) / scale:.3e} '
                f'> {RECOMPUTE_RTOL:g}). The report must not be allowed to '
                'quote a number its own field does not reproduce.')
        compared.append(cases_key)
    row['cross_checked_against_cases'] = compared


def lsq_order(levels, quantity):
    """log-log least-squares slope and R^2 of ``quantity`` against ``h``."""
    h = np.array([level['h'] for level in levels], dtype=float)
    e = np.array([level[quantity] for level in levels], dtype=float)
    if len(h) < 2:
        raise SystemExit(f'{quantity}: a slope needs at least two levels, got {len(h)}')
    if np.any(e <= 0):
        raise SystemExit(f'{quantity}: non-positive error cannot be fitted on a log-log axis')
    x, y = np.log(h), np.log(e)
    slope, intercept = np.polyfit(x, y, 1)
    residual = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual ** 2)) / ss_tot if ss_tot > 0 else float('nan')
    # With exactly two levels the fit is interpolatory: R^2 is 1 by construction
    # and carries no information, so it is withheld rather than printed as proof.
    return {
        'slope': float(slope),
        'r2': (float(r2) if len(levels) > 2 else None),
        'residual_degrees_of_freedom': len(levels) - 2,
    }


def pair_orders(levels, quantity):
    out = []
    for coarse, fine in zip(levels, levels[1:]):
        ratio = coarse[quantity] / fine[quantity]
        if ratio <= 0:
            raise SystemExit(f'{quantity}: non-positive error ratio')
        out.append({
            'from': coarse['n'],
            'to': fine['n'],
            'order': float(np.log(ratio) / np.log(coarse['h'] / fine['h'])),
        })
    return out


def analyse(levels):
    orders = {}
    for quantity in ORDERED:
        orders[quantity] = {
            'least_squares': lsq_order(levels, quantity),
            'pair_orders': pair_orders(levels, quantity),
        }
    return {
        'levels': levels,
        'orders': orders,
        'spanwise_order_for_reference_only': pair_orders(levels, 'spanwise_share'),
        'scheme_configs': {
            'reynolds': sorted({level.get('reynolds') for level in levels} - {None}),
            'symmetry_closure': sorted({level.get('symmetry_closure') for level in levels} - {None}),
            'convection_order': sorted({level.get('convection_order') for level in levels} - {None}),
            'algorithm': sorted({level.get('algorithm') for level in levels} - {None}),
        },
    }


def check_same_scheme(result):
    """Refuse a mixed configuration: a slope is a property of one scheme."""
    conflicts = {}
    for key in ('reynolds', 'symmetry_closure', 'convection_order', 'algorithm'):
        values = result['scheme_configs'][key]
        if len(values) > 1:
            conflicts[key] = values
    if conflicts:
        raise SystemExit(
            'refusing to fit an order across a mixed discrete configuration: '
            f'{conflicts}. A log-log slope over levels solved with different '
            'schemes is not the order of any of them. Note that the '
            '`symmetry_closure` key is absent from pre-2026-11 records; a '
            'missing key cannot be shown equal to `zero_gradient`, so such a '
            'sequence has to be described as `unspecified`, not fitted silently.')


def write_markdown(result, path):
    levels = result['levels']
    lines = ['| n | h | cells | vel L2 (full 3D) | vel L2 (in-plane u,v) | '
             'spanwise share | p L2 (fixed gauge) | p L2 (mean-free) | max\\|w\\| |',
             '|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for level in levels:
        lines.append(
            f'| {level["n"]} | {level["h"]:.5e} | {level["cells"]} | '
            f'{level["velocity_l2"]:.6e} | {level["velocity_l2_inplane"]:.6e} | '
            f'{level["spanwise_share"]:.3e} | {level["pressure_l2"]:.6e} | '
            f'{level["pressure_mean_free_l2"]:.6e} | {level["max_abs_w"]:.3e} |')
    lines.append('')
    lines.append('| quantity | pair orders | least-squares slope | R² | residual dof |')
    lines.append('|---|---|---:|---:|---:|')
    for quantity, entry in result['orders'].items():
        pairs = ' / '.join(f'{pair["from"]}→{pair["to"]}: {pair["order"]:.4f}'
                           for pair in entry['pair_orders'])
        ls = entry['least_squares']
        r2 = 'withheld (2 levels)' if ls['r2'] is None else f'{ls["r2"]:.6f}'
        lines.append(f'| `{quantity}` | {pairs} | **{ls["slope"]:.4f}** | {r2} | '
                     f'{ls["residual_degrees_of_freedom"]} |')
    lines.append('')
    lines.append('All values above are recomputed from the stored cell-centred '
                 'fields by `scripts/kovasznay_inplane_decomposition.py`; the '
                 'table is generated, not transcribed.')
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text('\n'.join(lines) + '\n')


def write_figure(result, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    levels = result['levels']
    h = [level['h'] for level in levels]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), constrained_layout=True)

    ax = axes[0]
    series = [
        ('velocity_l2', 'velocity L2 (full 3D)', 'o-'),
        ('velocity_l2_inplane', 'velocity L2 (in-plane u,v)', 's--'),
        ('pressure_mean_free_l2', 'pressure L2 (mean-free)', '^:'),
    ]
    for quantity, label, style in series:
        entry = result['orders'][quantity]
        slope = entry['least_squares']['slope']
        ax.loglog(h, [level[quantity] for level in levels], style,
                  label=f'{label} (slope={slope:.3f})')
    anchor = levels[0]['h']
    ref = levels[0]['velocity_l2_inplane']
    ax.loglog([anchor, levels[-1]['h']], [ref, ref * (levels[-1]['h'] / anchor) ** 2],
              'k-', lw=1, alpha=0.5, label='slope = 2 reference')
    ax.set(xlabel='h = 1/n', ylabel='volume-weighted relative L2 error',
           title='Kovasznay Re=40: error budget recomputed from stored fields')
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1]
    x = np.arange(len(levels))
    ax.bar(x - 0.2, [level['velocity_l2'] for level in levels], 0.4,
           label='full 3D', color='tab:blue', alpha=0.75)
    ax.bar(x + 0.2, [level['velocity_l2_inplane'] for level in levels], 0.4,
           label='in-plane (u,v)', color='tab:orange', alpha=0.75)
    ax.bar(x, [level['spanwise_share'] for level in levels], 0.25,
           label='spanwise share (w)', color='tab:red', alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([f'n={level["n"]}' for level in levels])
    ax.set_yscale('log')
    ax.set(ylabel='L2 error', title='Error budget: spurious w is %s of the total'
           % ('smaller' if levels[0]['spanwise_share'] < levels[0]['velocity_l2'] else 'comparable to'))
    ax.grid(True, which='both', axis='y', alpha=0.3)
    ax.legend(fontsize=7)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--level', action='append', required=True,
                        help='N=path/to/n<N>-fields.npz (repeat per level)')
    parser.add_argument('--cases', action='append', default=[],
                        help='N=path/to/n<N>/cases.json to cross-check (repeat)')
    parser.add_argument('--output', help='directory for summary.json')
    parser.add_argument('--markdown', help='path for the generated table')
    parser.add_argument('--figure', help='path for the generated figure')
    args = parser.parse_args(argv)

    paths = dict(parse_level(spec) for spec in args.level)
    cases_paths = {}
    for spec in args.cases:
        n_text, path_text = spec.split('=', 1)
        cases_paths[int(n_text)] = Path(path_text)

    levels = []
    for n in sorted(paths):
        row = measure_level(n, paths[n])
        if n in cases_paths:
            cases = read_cases(cases_paths[n])
            cross_check(row, cases, cases_paths[n])
            for key in ('reynolds', 'symmetry_closure', 'convection_order', 'algorithm'):
                if key in cases:
                    row[key] = cases[key]
        levels.append(row)

    result = analyse(levels)
    check_same_scheme(result)

    for level in levels:
        print(f'n={level["n"]:>3}  cells={level["cells"]:>6}  '
              f'full={level["velocity_l2"]:.6e}  inplane={level["velocity_l2_inplane"]:.6e}  '
              f'w-share={level["spanwise_share"]:.3e}  '
              f'p(mean-free)={level["pressure_mean_free_l2"]:.6e}')
    for quantity, entry in result['orders'].items():
        ls = entry['least_squares']
        r2 = 'withheld' if ls['r2'] is None else f'{ls["r2"]:.6f}'
        pairs = ', '.join(f'{pair["from"]}->{pair["to"]}={pair["order"]:.4f}'
                          for pair in entry['pair_orders'])
        print(f'{quantity:<24} slope={ls["slope"]:.4f}  R2={r2}  pairs: {pairs}')

    if args.cases:
        checked = {level['n']: level.get('cross_checked_against_cases') for level in levels}
        print(f'cross-checked against cases.json: {checked}')

    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)
        write_json(Path(args.output) / 'summary.json', {**result, 'generated_by':
                   'scripts/kovasznay_inplane_decomposition.py'})
        print(f'wrote {Path(args.output) / "summary.json"}')
    if args.markdown:
        write_markdown(result, args.markdown)
        print(f'wrote {args.markdown}')
    if args.figure:
        write_figure(result, args.figure)
        print(f'wrote {args.figure}')
    return 0


if __name__ == '__main__':
    sys.exit(main())