#!/usr/bin/env python3
"""Error-budget audit of the Kovasznay (1948) order study: *where* does order 1.68 come from?

Background / the debt being audited
-----------------------------------
``CASE-FVM-KOVASZNAY-ORDER`` (feature ledger, still open) asks for an observed L2
convergence order >= 1.8 on the Kovasznay analytic solution at Re = 40.  The
delivered round measured 1.684 for velocity (16/32/64) and recorded an honest
FAIL (``docs/case-kovasznay-order-order-verification/RESULTS.md``).  A FAIL is
only useful once it is *attributed*: the reported velocity norm mixes two very
different discretisation paths

* ``u, v`` -- the in-plane analytic field, for which the mesh is the natural one, and
* ``w``    -- the spanwise component, whose **exact value is identically zero** and
  which on the ``n x n x 1`` extruded mesh is transported by the weakest part of
  the discretisation (a single cell across the extrusion, Dirichlet at both
  z-planes, no in-plane neighbours in z).

This script does not re-score the case; it splits the measured error into those
budgets and re-measures the order of each separately, for four configurations
that isolate the candidate causes:

==============  =========================  ==========================================
config          mesh                       isolates
==============  =========================  ==========================================
``extruded``    ``n x n x 1`` + symmetry   the delivered round's setup (baseline)
``planar``      ``n x n`` planar (z = 0)   the extrusion-direction DOF (w block)
``extruded-o1`` extruded, first-order      the convection-order contribution
``planar-o1``   planar, first-order        convection on the planar path
==============  =========================  ==========================================

Every number written out is computed here from the solver's own state; nothing
is hand-copied.  ``summary.json`` carries the pairwise/least-squares orders per
budget so the conclusion is machine-checkable.

Usage (one grid per invocation keeps the cost of a level independent):

    PYTHONPATH=src python scripts/audit_kovasznay_order_components.py \
        --output runs/kovasznay-order/extruded-n32 --config extruded --n 32
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.verification.native_hybrid_ns import kovasznay_fields, kovasznay_solver

CONFIGS = {
    # name: (symmetry_extrusion, convection_order)
    'extruded': (True, 2),
    'planar': (False, 2),
    'extruded-o1': (True, 1),
    'planar-o1': (False, 1),
}


def _rel_l2(error, reference, volumes):
    w = volumes.reshape((len(volumes),) + (1,) * (np.asarray(error).ndim - 1))
    return float(np.sqrt(np.sum(w * np.square(error)) / np.sum(w * np.square(reference))))


def decompose(velocity, exact_velocity, volumes):
    """Split ||u_h - u||_L2 into the in-plane budget and the spurious-spanwise budget.

    ``w_exact == 0`` identically for this solution, so the spanwise budget is a pure
    error term.  The two returned budgets add to the total in quadrature by
    construction of the volume-weighted norm; ``closure`` reports that identity.
    """
    e_uv = velocity[:, :2] - exact_velocity[:, :2]
    e_w = velocity[:, 2:3] - exact_velocity[:, 2:3]
    total = _rel_l2(velocity - exact_velocity, exact_velocity, volumes)
    inplane = _rel_l2(e_uv, exact_velocity, volumes)
    span = _rel_l2(e_w, exact_velocity, volumes)
    return dict(
        velocity_l2=total,
        inplane_l2=inplane,
        spanwise_l2=span,
        quadrature_closure=abs(np.hypot(inplane, span) - total) / total,
        max_abs_spanwise=float(np.max(np.abs(velocity[:, 2]))),
        max_abs_inplane=float(np.max(np.abs(velocity[:, :2]))),
        inplane_u_l2=_rel_l2(velocity[:, 0] - exact_velocity[:, 0], exact_velocity[:, 0], volumes),
        inplane_v_l2=_rel_l2(velocity[:, 1] - exact_velocity[:, 1], exact_velocity[:, 1], volumes),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--config', choices=sorted(CONFIGS), required=True)
    p.add_argument('--n', type=int, required=True)
    p.add_argument('--reynolds', type=float, default=40.)
    p.add_argument('--outer-relaxation', type=float, default=.7)
    p.add_argument('--tolerance', type=float, default=1e-8)
    args = p.parse_args()

    symmetry_extrusion, convection_order = CONFIGS[args.config]
    tick = time.perf_counter()
    mesh, solver, exact = kovasznay_solver(
        args.n, args.reynolds, symmetry_extrusion=symmetry_extrusion,
        convection_order=convection_order, coupled_linear_method='reused_lu')
    build_seconds = time.perf_counter() - tick
    volumes = np.asarray(mesh.cell_volumes)

    solve_tick = time.perf_counter()
    result = solver.solve(algorithm='coupled', outer_relaxation=args.outer_relaxation,
                          tolerance=args.tolerance, max_outer_iterations=200,
                          callback=lambda row: print('outer', row, flush=True))
    solve_seconds = time.perf_counter() - solve_tick

    velocity = np.asarray(result['velocity'])
    pressure = np.asarray(result['pressure'])
    row = dict(
        config=args.config, n=args.n, h=1. / args.n, reynolds=args.reynolds,
        symmetry_extrusion=symmetry_extrusion, convection_order=convection_order,
        cells=len(mesh.cells), faces=len(mesh.owner),
        numerically_converged=bool(result['numerically_converged']),
        boundary_velocity_error=float(result['boundary_velocity_error']),
        divergence_max=float(result['outer_history'][-1]['divergence_max']),
        outer_iterations=len(result['outer_history']),
        pressure_l2=_rel_l2(pressure - exact['pressure'], exact['pressure'], volumes),
        pressure_mean_free_l2=_rel_l2(
            (pressure - exact['pressure']) - np.sum(volumes * (pressure - exact['pressure'])) / volumes.sum(),
            exact['pressure'] - np.sum(volumes * exact['pressure']) / volumes.sum(), volumes),
        build_seconds=build_seconds, solve_seconds=solve_seconds,
        **decompose(velocity, np.asarray(exact['velocity']), volumes),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'cases.json').write_text(json.dumps([row], indent=2) + '\n')
    print('RESULT ' + json.dumps({k: v for k, v in row.items()}, default=str), flush=True)


if __name__ == '__main__':
    main()