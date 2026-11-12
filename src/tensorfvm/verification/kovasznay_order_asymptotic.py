"""Asymptotic Kovasznay (1948) order ladder, one grid per process, resumable.

Why this module exists
----------------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger) asks for an observed velocity L2
order ``>= 1.8`` on the Kovasznay analytic solution at Re = 40.  The history is

* m16130 delivered the case and recorded 1.684 -> honest FAIL;
* m16396 split the norm: in-plane 1.808 (passes), spanwise ``w`` 1.106 (the
  shortfall), ``w`` being analytically identically zero;
* m16524 located the cause: the *free tangential face dofs* on the symmetry
  (z-)planes of the extruded mesh drive an O(h) ``tau_xz``;
* m16695 added the solver-level ``zero_gradient_faces`` closure (face component
  tied to its owner cell value, i.e. the strong form ``d u_t / d n = 0``) and
  showed ``w`` collapses to *exactly* 0 at n = 8/16 with the in-plane error
  unchanged -- but the ladder stopped at two levels, so the anchor criterion
  (order on the finest pair) was still open (8->16 gave 1.588, pre-asymptotic).

This module is the continuation: it turns the ladder into **one grid per
process** with a JSON record per level, so the expensive levels can be run on
the compute fleet independently and re-merged later, and it grades the
pre-declared criteria over *every* available level pair plus a least-squares
slope over the whole ladder.

Pre-declared criteria (fixed before any grid of this round was run)
------------------------------------------------------------------
C1  constraint manifold: ``max|u_face - u_owner_cell|`` on the tied dofs
    ``<= 1e-12`` at every level.
C2  spanwise exactness: ``spanwise L2 == 0`` and ``max|w| == 0`` at every level
    (``w`` is analytically zero; the tied closure must not re-introduce it).
C3  **anchor criterion**: total velocity L2 observed order ``>= 1.8`` on the
    finest available level pair.
C4  in-plane L2 observed order ``>= 1.8`` on the finest pair.
C5  least-squares slope of total L2 over *all* levels ``>= 1.8`` (robust to a
    single noisy pair).
C6  no inconclusive gate: Picard reports numerical convergence at every level.
C7  (informational) pressure (mean-free) order on the finest pair -- the anchor
    asks for velocity only; recorded so the pressure path cannot silently rot.
C8  (informational) if the same level was solved with two linear-solver
    configurations, their total L2 must agree to ``<= 1e-6`` relative, i.e. the
    linear tolerance must not contaminate the measured order.

Usage
-----
One grid per process (each is independent, results are merged afterwards)::

    PYTHONPATH=src python -m tensorfvm.verification.kovasznay_order_asymptotic \
        --level 32 --output runs/kovasznay-asymptotic
    PYTHONPATH=src python -m tensorfvm.verification.kovasznay_order_asymptotic \
        --merge --output runs/kovasznay-asymptotic

``--merge`` writes ``summary.json`` in the output directory and prints the
orders plus every criterion verdict; it never re-solves anything.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .kovasznay_extrusion import error_budgets, extrusion_solver, observed_order
from .kovasznay_order_closure import pressure_budget

LEVELS = (16, 32, 64, 128)
GATE = 1.8
TIE_GATE = 1e-12
LINEAR_GATE = 1e-6


def level_path(directory, n, linear_method='reused_lu', closure='zero_gradient'):
    """File name carries every key the merge filters on (closure and solver).

    Two records that differ only in the boundary closure or the linear
    configuration must not overwrite each other -- the free-traction control at
    the same ``n`` is exactly such a case.
    """
    return Path(directory) / f'level-n{int(n)}-{closure}-{linear_method}.json'


def run_level(n, *, reynolds=40., convection_order=2, linear_method='reused_lu',
              tolerance=1e-8, relaxation=.7, max_outer_iterations=80, gate=GATE):
    """Solve one grid of the ladder and return its machine-checkable record."""
    started = time.time()
    mesh, solver, exact = extrusion_solver(
        n, reynolds, tangential='zero_gradient', convection_order=convection_order,
        linear_method=linear_method)
    result = solver.solve(max_outer_iterations=max_outer_iterations,
                          tolerance=tolerance, relaxation=relaxation)
    solve_seconds = time.time() - started
    budgets = error_budgets(mesh, solver, exact)
    space = solver.viscous.velocity_space
    linear = solver.coupled_linear_history[-1] if solver.coupled_linear_history else {}
    outer = solver.outer_history[-1] if solver.outer_history else {}
    record = dict(
        n=int(n), h=1. / n, reynolds=reynolds, convection_order=convection_order,
        closure='zero_gradient', linear_method=linear_method,
        cells=int(solver.N), faces=int(solver.F),
        unknowns=int(len(space.free)),
        converged=bool(result['numerically_converged']),
        outer_iterations=len(solver.outer_history),
        final_momentum_residual=float(outer.get('current_momentum_residual',
                                                 outer.get('momentum_relative_residual',
                                                           float('nan')))),
        final_divergence_max=float(outer.get('divergence_max', float('nan'))),
        inner_iterations=outer.get('inner_iterations'),
        seconds=solve_seconds,
        tie_residual=float(space.constraint_error(solver.state - solver.prescribed)),
        tied_components=int(space.zero_gradient_components.sum()),
        spanwise_l2=float(budgets['spanwise_l2']),
        max_abs_w=float(budgets['max_abs_w']),
        inplane_l2=float(budgets['inplane_l2']),
        total_l2=float(budgets['total_l2']),
        linear_true_relative_residual=(None if 'true_relative_residual' not in linear
                                       else float(linear['true_relative_residual'])),
        linear_gmres_iterations=linear.get('gmres_iterations'),
        gate=gate,
    )
    record.update(pressure_budget(mesh, solver, exact))
    return record


def write_level(directory, record):
    path = level_path(directory, record['n'], record['linear_method'], record['closure'])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=float) + '\n')
    return path


def load_records(directory, linear_method='reused_lu', closure='zero_gradient'):
    """Load every level record of one (closure, linear method) family."""
    directory = Path(directory)
    records = []
    for path in directory.glob('level-n*.json'):
        record = json.loads(path.read_text())
        if record.get('closure') == closure and record.get('linear_method') == linear_method:
            records.append(record)
    return sorted(records, key=lambda r: r['n'])


def pair_orders(records):
    """Two-level observed orders for every consecutive pair (h ratio 2)."""
    rows = []
    for coarse, fine in zip(records, records[1:]):
        # observed_order takes the grid-refinement ratio (h_coarse / h_fine)
        ratio = coarse['h'] / fine['h']
        rows.append(dict(
            pair=f"{coarse['n']}->{fine['n']}",
            h_ratio=float(ratio),
            total=observed_order(coarse['total_l2'], fine['total_l2'], ratio),
            inplane=observed_order(coarse['inplane_l2'], fine['inplane_l2'], ratio),
            pressure_mean_free=observed_order(coarse['pressure_mean_free_l2'],
                                             fine['pressure_mean_free_l2'], ratio),
        ))
    return rows


def fit_order(records, key):
    """Least-squares slope of ``log(error)`` against ``log(h)`` over all levels."""
    usable = [r for r in records if np.isfinite(r[key]) and r[key] > 0]
    if len(usable) < 2:
        return None
    h = np.log([r['h'] for r in usable])
    e = np.log([r[key] for r in usable])
    slope, intercept = np.polyfit(h, e, 1)
    fitted = slope * h + intercept
    residual = float(np.max(np.abs(np.asarray(e) - fitted)))
    return dict(slope=float(slope), max_log_residual=residual, levels=len(usable))


def _linear_independence(records):
    """C8: same level solved with two linear configurations must agree."""
    rows = []
    groups = {}
    for record in records:
        groups.setdefault(record['n'], []).append(record)
    for n, group in sorted(groups.items()):
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                scale = max(a['total_l2'], b['total_l2'], np.finfo(float).tiny)
                rows.append(dict(
                    n=int(n), methods=[a['linear_method'], b['linear_method']],
                    total_l2=[a['total_l2'], b['total_l2']],
                    relative_difference=float(abs(a['total_l2'] - b['total_l2']) / scale)))
    return rows


def evaluate(records, *, gate=GATE, linear_records=None):
    """Grade the pre-declared criteria on the levels that are available."""
    checks = []

    def add(identifier, statement, value, verdict):
        checks.append(dict(id=identifier, statement=statement, value=value, verdict=verdict))

    if not records:
        add('C0', 'at least two ladder levels available', 0, 'FAIL')
        return dict(checks=checks, orders=[], fits={}, anchor_closed=False)

    orders = pair_orders(records)
    fits = {key: fit_order(records, key) for key in
            ('total_l2', 'inplane_l2', 'pressure_mean_free_l2')}
    finest = orders[-1] if orders else {}

    worst_tie = max(r['tie_residual'] for r in records)
    add('C1', 'tied-dof residual max|u_face-u_cell| <= %.0e at every level' % TIE_GATE,
        worst_tie, 'PASS' if worst_tie <= TIE_GATE else 'FAIL')
    span = max(r['max_abs_w'] for r in records)
    span_exact = all(r['spanwise_l2'] == 0. and r['max_abs_w'] == 0. for r in records)
    add('C2', 'spanwise L2 and max|w| exactly 0 at every level (w exact = 0)',
        span, 'PASS' if span_exact else 'FAIL')
    total = finest.get('total')
    add('C3', 'ANCHOR: total velocity L2 order >= %.1f on the finest pair %s'
        % (gate, finest.get('pair')), total,
        'PASS' if (total is not None and total >= gate) else 'FAIL')
    inplane = finest.get('inplane')
    add('C4', 'in-plane L2 order >= %.1f on the finest pair' % gate, inplane,
        'PASS' if (inplane is not None and inplane >= gate) else 'FAIL')
    lsq = (fits['total_l2'] or {}).get('slope')
    add('C5', 'least-squares total L2 slope over all levels >= %.1f' % gate, lsq,
        'PASS' if (lsq is not None and lsq >= gate) else 'FAIL')
    converged = [r['converged'] for r in records]
    add('C6', 'Picard reported numerical convergence at every level', converged,
        'PASS' if all(converged) else 'FAIL')
    add('C7', 'informational: pressure (mean-free) order on the finest pair',
        finest.get('pressure_mean_free'), 'INFO')
    rows = _linear_independence(list(records) + list(linear_records or []))
    worst = max([r['relative_difference'] for r in rows], default=None)
    add('C8', 'informational: linear-configuration independence of total L2 <= %.0e'
        % LINEAR_GATE, worst, 'INFO' if worst is None
        else ('PASS' if worst <= LINEAR_GATE else 'FAIL'))
    # C7/C8 are informational by declaration: they are recorded, never gating
    gating = [c for c in checks if c['verdict'] in ('PASS', 'FAIL') and c['id'] not in ('C7', 'C8')]
    return dict(checks=checks, orders=orders, fits=fits,
                anchor_closed=bool(gating) and all(c['verdict'] == 'PASS' for c in gating))


def merge(directory, linear_method='reused_lu'):
    directory = Path(directory)
    records = load_records(directory, linear_method=linear_method)
    others = [r for r in load_records(directory, closure='zero_gradient')
              if r['linear_method'] != linear_method]
    verdict = evaluate(records, linear_records=others)
    payload = dict(levels=[r['n'] for r in records], linear_method=linear_method,
                   records=records, ledger_others=others, **verdict)
    (directory / 'summary.json').write_text(json.dumps(payload, indent=2, default=float) + '\n')
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--level', type=int, default=None,
                        help='run this single grid and write its level record')
    parser.add_argument('--merge', action='store_true',
                        help='merge the level records already in --output')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--linear-method', default='reused_lu',
                        choices=('reused_lu', 'direct', 'block_amg'))
    parser.add_argument('--reynolds', type=float, default=40.)
    parser.add_argument('--convection-order', type=int, default=2)
    parser.add_argument('--tolerance', type=float, default=1e-8)
    parser.add_argument('--relaxation', type=float, default=.7)
    parser.add_argument('--max-outer-iterations', type=int, default=80)
    args = parser.parse_args()
    if args.level is not None:
        record = run_level(args.level, reynolds=args.reynolds,
                           convection_order=args.convection_order,
                           linear_method=args.linear_method,
                           tolerance=args.tolerance, relaxation=args.relaxation,
                           max_outer_iterations=args.max_outer_iterations)
        path = write_level(args.output, record)
        print(json.dumps(record, indent=2, default=float))
        print('wrote', path)
    if args.merge:
        payload = merge(args.output, linear_method=args.linear_method)
        print(json.dumps(dict(levels=payload['levels'], orders=payload['orders'],
                              fits=payload['fits']), indent=2, default=float))
        for check in payload['checks']:
            print(f"{check['id']} {check['verdict']:5s} {check['statement']} -> {check['value']}")
        print('anchor_closed:', payload['anchor_closed'])
    if args.level is None and not args.merge:
        parser.error('nothing to do: pass --level N and/or --merge')


if __name__ == '__main__':
    main()