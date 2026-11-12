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

The physical symmetry condition is ``d u_t / d n = 0``.  This round adds that as
a *solver* boundary condition (``zero_gradient_faces``: the face component is
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

Usage
-----
    PYTHONPATH=src python -m tensorfvm.verification.kovasznay_order_closure \\
        --levels 8 16 32 --output runs/kovasznay-order-closure
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .kovasznay_extrusion import error_budgets, extrusion_solver, observed_order

LEVELS = (8, 16, 32)
CLOSURES = ('zero_gradient', 'symmetry')


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
        converged=bool(result['numerically_converged']),
        outer_iterations=len(solver.outer_history), seconds=time.time() - started,
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
    table = {}
    for closure in closures:
        # the free-traction control only has to establish the transfer comparison
        selected = levels if closure == 'zero_gradient' else levels[:1]
        table[closure] = [one_level(n, closure, reynolds) for n in selected]
    return table


def observed_orders(records):
    pairs = []
    for coarse, fine in zip(records, records[1:]):
        pairs.append(dict(
            pair=f"{coarse['n']}->{fine['n']}",
            total=observed_order(coarse['total_l2'], fine['total_l2']),
            inplane=observed_order(coarse['inplane_l2'], fine['inplane_l2']),
            spanwise=observed_order(coarse['spanwise_l2'], fine['spanwise_l2']),
            pressure_mean_free=observed_order(coarse['pressure_mean_free_l2'],
                                             fine['pressure_mean_free_l2']),
        ))
    return pairs


def evaluate(table):
    tied = table['zero_gradient']
    coarse_control = table['symmetry'][0]
    coarse = tied[0]
    orders = observed_orders(tied)
    finest = orders[-1] if orders else {}
    checks = []

    def add(identifier, statement, value, verdict):
        checks.append(dict(id=identifier, statement=statement, value=value, verdict=verdict))

    worst_tie = max(r['tie_residual'] for r in tied)
    add('C1', 'constraint manifold max|u_face-u_cell| on tied dofs <= 1e-12',
        worst_tie, 'PASS' if worst_tie <= 1e-12 else 'FAIL')
    span = max(r['max_abs_w'] for r in tied)
    add('C2', 'spanwise L2 and max|w| exactly 0 at every level (w exact = 0)',
        span, 'PASS' if all(r['spanwise_l2'] == 0. and r['max_abs_w'] == 0. for r in tied) else 'FAIL')
    ratio = coarse['inplane_l2'] / coarse_control['inplane_l2']
    add('C3', 'in-plane L2 (tied) <= 1.02 x in-plane L2 (delivered free traction) at n=%d' % coarse['n'],
        ratio, 'PASS' if ratio <= 1.02 else 'FAIL')
    total = finest.get('total')
    add('C4', 'anchor: total velocity L2 order >= 1.8 on the finest pair %s'
        % finest.get('pair'), total,
        'PASS' if (total is not None and total >= 1.8) else 'FAIL')
    inplane = finest.get('inplane')
    add('C5', 'in-plane L2 order >= 1.8 on the finest pair',
        inplane, 'PASS' if (inplane is not None and inplane >= 1.8) else 'FAIL')
    add('C6', 'Picard reported numerical convergence at every level',
        [r['converged'] for r in tied],
        'PASS' if all(r['converged'] for r in tied) else 'FAIL')
    return dict(checks=checks, orders=orders,
                anchor_closed=all(c['verdict'] == 'PASS' for c in checks))


def run(levels=LEVELS, output=None):
    payload = dict(levels=[int(n) for n in levels], closures=list(CLOSURES), table={},
                   orders={}, verdict={})
    table = ladder(levels)
    payload['table'] = table
    payload['orders'] = observed_orders(table['zero_gradient'])
    payload['verdict'] = evaluate(table)
    if output is not None:
        path = Path(output)
        path.mkdir(parents=True, exist_ok=True)
        (path / 'summary.json').write_text(json.dumps(payload, indent=2, default=float) + '\n')
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--levels', type=int, nargs='+', default=list(LEVELS))
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    payload = run(tuple(args.levels), args.output)
    print(json.dumps(payload['orders'], indent=2))
    for check in payload['verdict']['checks']:
        print(f"{check['id']} {check['verdict']:5s} {check['statement']} -> {check['value']}")
    print('anchor_closed:', payload['verdict']['anchor_closed'])


if __name__ == '__main__':
    main()