"""Kovasznay (1948) order ladder under an asymptotically consistent symmetry closure.

Why this module exists
----------------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger, open) wants an observed velocity
L2 order ``>= 1.8`` at ``Re = 40``.  Delivered round m16130 measured 1.684 and
recorded an honest FAIL; m16396 (budget split) and m16524 (term probe) then
showed that the whole shortfall is a first-order spurious *spanwise* velocity
that comes from the only boundary closure the solver offered on the symmetry
(z-)planes: the tangential face components were left as free momentum unknowns
with a zero-traction row, so their discrete value drifts away from the owner
cell value and drives ``tau_xz``.  Since ``w`` is analytically identically zero,
that artefact can only pollute the norm; it says nothing about the interior
discretisation.

The physical symmetry condition is ``d u_t / d n = 0``.  m16695 added that as a
*solver* boundary condition (``zero_gradient_faces``: the face component is
tied to its owner cell value, its equation is replaced by the constraint, the
dof leaves the unknown set -- see ``NativeVelocitySpace3D``).  No analytic data
is injected on those components, so the closure is usable on any symmetry plane
and not only on a manufactured case.

Pre-declared criteria (fixed before the ladder was run, all from raw output)
---------------------------------------------------------------------------
C1  constraint manifold: at every level the accepted state satisfies
    ``|u_face - u_owner_cell|`` on the tied dofs ``<= 1e-12``.
C2  spanwise exactness: ``spanwise L2 == 0`` and ``max|w| == 0`` at every
    level (``w`` analytically zero; the tied closure must not re-introduce it).
C3  no accuracy loss on transfer: at the coarsest level the in-plane L2 under
    the tied closure is ``<= 1.02 x`` the same quantity under the delivered
    free-traction closure (both levels measured in this run).
C4  **anchor criterion**: total velocity L2 observed order ``>= 1.8`` on the
    finest available level pair.  (The 8->16 pair is pre-asymptotic; the ladder
    reports every pair, the criterion is graded on the finest pair and the
    reported uncertainty is the spread over the available pairs.)
C5  in-plane L2 observed order ``>= 1.8`` on the finest pair.
C6  no inconclusive gate: the Picard iteration must report numerical
    convergence at every level (otherwise the order is undefined, not FAIL).

C7 and C8 were added in the round that ran the deeper levels (the 8->16 pair
alone is pre-asymptotic, so C4/C5 graded on one pair cannot separate "order is
>= 1.8" from "the two level values happened to fall that way"):

C7  asymptotic consistency: **every** level pair with ``n_coarse >= 16`` (i.e.
    pairs that are not the pre-asymptotic 8->16 one) reports an order
    ``>= 1.8``; a ladder is not allowed to pass on one lucky pair.
C8  three-level identifiability: the three-level fit ``f(h) = L + C h^p``
    (``verification.multi_grid.multi_grid_order``, equal refinement ratio)
    reports ``p >= 1.8`` on a consistent (monotone) ladder.  With exactly three
    levels the fit has no residual degrees of freedom, so the linearised order
    standard error is undefined (``nan``) and is *reported*, not graded; when a
    fourth level exists (dof >= 1) the standard error must also be ``<= 0.1``.

Usage
-----
    PYTHONPATH=src python -m tensorfvm.verification.kovasznay_order_closure \\
        --levels 8 16 32 --output runs/kovasznay-order-closure

Heavy levels can be run in separate invocations and merged (each payload keeps
its own raw records, so the merge is arithmetic on measured numbers only):

    PYTHONPATH=src python -m tensorfvm.verification.kovasznay_order_closure \\
        --levels 64 --closure zero_gradient --output runs/l64
    PYTHONPATH=src python -m tensorfvm.verification.kovasznay_order_closure \\
        --merge runs/main/summary.json runs/l64/summary.json --output runs/merged
"""
import argparse
import json
import math
import resource
import time
from pathlib import Path

import numpy as np

from .kovasznay_extrusion import error_budgets, extrusion_solver, observed_order
from .multi_grid import multi_grid_order

LEVELS = (8, 16, 32)
CLOSURES = ('zero_gradient', 'symmetry')
ASYMPTOTIC_MIN_N = 16      # pairs with n_coarse >= this are graded by C7
GATE = 1.8


def pressure_budget(mesh, solver, exact):
    """Volume weighted relative L2 of the gauge-consistent, mean-free pressure."""
    volumes = mesh.cell_volumes
    pressure = np.asarray(solver.p, float)
    error = pressure - exact['pressure']
    reference = exact['pressure']
    weights = volumes / volumes.sum()
    scale = np.sqrt(np.sum(weights * reference ** 2))
    mean_free = error - np.sum(weights * error)
    return dict(
        pressure_l2=float(np.sqrt(np.sum(weights * error ** 2)) / scale),
        pressure_mean_free_l2=float(np.sqrt(np.sum(weights * mean_free ** 2)) / scale),
    )


def one_level(n, closure, reynolds=40., convection_order=2):
    started = time.time()
    mesh, solver, exact = extrusion_solver(n, reynolds, tangential=closure,
                                          convection_order=convection_order)
    result = solver.solve(max_outer_iterations=80, tolerance=1e-8, relaxation=.7)
    budgets = error_budgets(mesh, solver, exact)
    record = dict(
        n=int(n), closure=closure, cells=int(solver.N), faces=int(solver.F),
        dofs=int(3 * (solver.N + solver.F)),
        converged=bool(result['numerically_converged']),
        outer_iterations=len(solver.outer_history), seconds=time.time() - started,
        peak_rss_mb=float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.),
        total_l2=budgets['total_l2'], inplane_l2=budgets['inplane_l2'],
        spanwise_l2=budgets['spanwise_l2'], max_abs_w=budgets['max_abs_w'],
        tie_residual=float(solver.viscous.velocity_space.constraint_error(
            solver.state - solver.prescribed)),
        tied_components=int(solver.viscous.velocity_space.zero_gradient_components.sum()),
        prescribed_tangential_on_planes=float(np.max(np.abs(
            solver.prescribed.reshape(-1, 3)[solver.N:][
                solver.viscous.velocity_space.zero_gradient_components]))),
    )
    record.update(pressure_budget(mesh, solver, exact))
    return record


def ladder(levels=LEVELS, closures=CLOSURES, reynolds=40.):
    """Whole ladder in one process: the free-traction control only at the finest-free
    level it needs (the transfer comparison is a coarsest-level check)."""
    table = {}
    for closure in closures:
        selected = levels if closure == 'zero_gradient' else levels[:1]
        table[closure] = [one_level(n, closure, reynolds) for n in selected]
    return table


def solve_levels(pairs, reynolds=40.):
    """Solve an explicit list of ``(n, closure)`` pairs (one process per level)."""
    table = {}
    for n, closure in pairs:
        table.setdefault(closure, []).append(one_level(int(n), closure, reynolds))
    for records in table.values():
        records.sort(key=lambda r: r['n'])
    return table


def merge_tables(*tables):
    """Union of level tables keyed by ``(closure, n)``; the first record wins."""
    merged, seen = {}, set()
    for table in tables:
        for closure, records in table.items():
            for record in records:
                key = (record['closure'], int(record['n']))
                if key in seen:
                    continue
                seen.add(key)
                merged.setdefault(closure, []).append(record)
    return {closure: sorted(records, key=lambda r: r['n'])
            for closure, records in merged.items()}


def observed_orders(records):
    pairs = []
    for coarse, fine in zip(records, records[1:]):
        pairs.append(dict(
            pair=f"{coarse['n']}->{fine['n']}",
            coarse_n=int(coarse['n']),
            total=observed_order(coarse['total_l2'], fine['total_l2']),
            inplane=observed_order(coarse['inplane_l2'], fine['inplane_l2']),
            spanwise=observed_order(coarse['spanwise_l2'], fine['spanwise_l2']),
            pressure_mean_free=observed_order(coarse['pressure_mean_free_l2'],
                                             fine['pressure_mean_free_l2']),
        ))
    return pairs


def three_level_fit(records, key='total_l2'):
    """``f(h) = L + C h^p`` over the three finest levels (fine -> coarse order)."""
    usable = [r for r in records if r.get(key) and r[key] > 0.]
    if len(usable) < 3:
        return None
    fine_to_coarse = list(reversed(usable[-3:]))
    spacings = [1. / r['n'] for r in fine_to_coarse]
    values = [float(r[key]) for r in fine_to_coarse]
    if not all(math.isfinite(v) and v > 0. for v in values):
        return None
    study = multi_grid_order(spacings, values, theory_order=2.)
    return dict(levels=[int(r['n']) for r in fine_to_coarse],
                order=float(study.order),
                order_stderr=float(study.order_stderr),
                convergence=str(study.convergence),
                monotone=bool(study.monotone),
                asymptotic=bool(study.asymptotic),
                order_ratio=float(study.order_ratio),
                extrapolated=float(study.extrapolated),
                gci=float(study.gci))


def evaluate(table):
    tied = sorted(table.get('zero_gradient', []), key=lambda r: r['n'])
    control = sorted(table.get('symmetry', []), key=lambda r: r['n'])
    orders = observed_orders(tied)
    finest = orders[-1] if orders else {}
    checks = []

    def add(identifier, statement, value, verdict):
        checks.append(dict(id=identifier, statement=statement, value=value, verdict=verdict))

    if not tied:
        raise ValueError('no zero_gradient level in the table')
    worst_tie = max(r['tie_residual'] for r in tied)
    add('C1', 'constraint manifold max|u_face-u_cell| on tied dofs <= 1e-12',
        worst_tie, 'PASS' if worst_tie <= 1e-12 else 'FAIL')
    span = max(r['max_abs_w'] for r in tied)
    add('C2', 'spanwise L2 and max|w| exactly 0 at every level (w exact = 0)',
        span, 'PASS' if all(r['spanwise_l2'] == 0. and r['max_abs_w'] == 0. for r in tied) else 'FAIL')
    if control:
        coarse = tied[0]
        match = [r for r in control if r['n'] == coarse['n']]
        if match:
            ratio = coarse['inplane_l2'] / match[0]['inplane_l2']
            add('C3', 'in-plane L2 (tied) <= 1.02 x in-plane L2 (delivered free traction) at n=%d'
                % coarse['n'], ratio, 'PASS' if ratio <= 1.02 else 'FAIL')
        else:
            add('C3', 'in-plane L2 transfer check needs the control at n=%d' % coarse['n'],
                None, 'NOT_RUN')
    else:
        add('C3', 'in-plane L2 transfer check needs a free-traction control level',
            None, 'NOT_RUN')
    total = finest.get('total')
    add('C4', 'anchor: total velocity L2 order >= 1.8 on the finest pair %s'
        % finest.get('pair'), total,
        'PASS' if (total is not None and total >= GATE) else 'FAIL')
    inplane = finest.get('inplane')
    add('C5', 'in-plane L2 order >= 1.8 on the finest pair',
        inplane, 'PASS' if (inplane is not None and inplane >= GATE) else 'FAIL')
    add('C6', 'Picard reported numerical convergence at every level',
        [r['converged'] for r in tied],
        'PASS' if all(r['converged'] for r in tied) else 'FAIL')
    asymptotic = [o for o in orders if o['coarse_n'] >= ASYMPTOTIC_MIN_N]
    if asymptotic:
        worst = min(o['total'] for o in asymptotic)
        add('C7', 'every level pair with n_coarse >= %d reports total order >= 1.8'
            % ASYMPTOTIC_MIN_N, worst, 'PASS' if worst >= GATE else 'FAIL')
    else:
        add('C7', 'asymptotic pairs (n_coarse >= %d) available' % ASYMPTOTIC_MIN_N,
            None, 'NOT_RUN')
    fit = three_level_fit(tied, 'total_l2')
    if fit is None:
        add('C8', 'three-level fit needs >= 3 converged levels', None, 'NOT_RUN')
    else:
        stderr = fit['order_stderr']
        # dof == 0 (three levels) carries no order uncertainty estimate at all;
        # grade it only when the fit actually has residual degrees of freedom.
        identified = (not math.isfinite(stderr)) or stderr <= .1
        ok = bool(math.isfinite(fit['order']) and fit['order'] >= GATE
                  and identified and fit['convergence'] == 'monotone')
        add('C8', 'three-level f(h)=L+C h^p gives p >= 1.8 with stderr <= 0.1 and a consistent ladder',
            dict(order=fit['order'], order_stderr=stderr, convergence=fit['convergence']),
            'PASS' if ok else 'FAIL')
    graded = [c for c in checks if c['verdict'] in ('PASS', 'FAIL')]
    return dict(checks=checks, orders=orders, three_level_fit=fit,
                anchor_closed=bool(graded) and all(c['verdict'] == 'PASS' for c in graded),
                not_run=[c['id'] for c in checks if c['verdict'] == 'NOT_RUN'])


def run(levels=LEVELS, output=None, closures=CLOSURES):
    payload = dict(levels=[int(n) for n in levels], closures=list(closures), table={},
                   orders={}, verdict={})
    table = ladder(tuple(levels), tuple(closures))
    payload['table'] = table
    payload['orders'] = observed_orders(sorted(table.get('zero_gradient', []),
                                               key=lambda r: r['n']))
    payload['verdict'] = evaluate(table)
    if output is not None:
        write_payload(payload, output)
    return payload


def write_payload(payload, output):
    path = Path(output)
    path.mkdir(parents=True, exist_ok=True)
    (path / 'summary.json').write_text(json.dumps(payload, indent=2, default=float) + '\n')


def merge_payloads(paths, output=None):
    documents = [json.loads(Path(path).read_text()) for path in paths]
    table = merge_tables(*[document['table'] for document in documents])
    payload = dict(levels=sorted({int(r['n']) for records in table.values() for r in records}),
                   closures=sorted(table), table=table,
                   orders=observed_orders(sorted(table.get('zero_gradient', []),
                                                 key=lambda r: r['n'])),
                   sources=[str(path) for path in paths])
    payload['verdict'] = evaluate(table)
    if output is not None:
        write_payload(payload, output)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, nargs='+', default=None,
                        help='grid levels; omit with --closure to solve the full default ladder')
    parser.add_argument('--closure', action='append', choices=list(CLOSURES), default=None,
                        help='restrict to these closures (default: the full ladder)')
    parser.add_argument('--merge', nargs='+', default=None,
                        help='merge existing summary.json payloads instead of solving')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.merge:
        payload = merge_payloads(args.merge, args.output)
    elif args.levels and args.closure:
        closures = tuple(args.closure)
        table = solve_levels([(n, closure) for closure in closures for n in args.levels])
        payload = dict(levels=[int(n) for n in args.levels], closures=list(closures),
                       table=table, orders=observed_orders(sorted(
                           table.get('zero_gradient', []), key=lambda r: r['n'])))
        payload['verdict'] = evaluate(table)
        if args.output is not None:
            write_payload(payload, args.output)
    else:
        payload = run(tuple(args.levels or LEVELS),
                      args.output, tuple(args.closure or CLOSURES))
    print(json.dumps(payload['orders'], indent=2))
    for check in payload['verdict']['checks']:
        print(f"{check['id']} {check['verdict']:7s} {check['statement']} -> {check['value']}")
    print('anchor_closed:', payload['verdict']['anchor_closed'],
          'not_run:', payload['verdict']['not_run'])
    if payload['verdict'].get('three_level_fit'):
        print('three_level_fit:', json.dumps(payload['verdict']['three_level_fit']))


if __name__ == '__main__':
    main()