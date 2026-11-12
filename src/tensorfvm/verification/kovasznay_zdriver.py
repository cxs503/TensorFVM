"""Locate the O(h) spanwise driver in the extruded Kovasznay case (D10 follow-up).

The ``CASE-FVM-KOVASZNAY-ORDER`` attribution (m16396) split the velocity L2
error into an in-plane (u, v) budget (order 1.808, passes the 1.8 gate) and a
spanwise ``w`` budget (order 1.106, fails) whose exact value is identically
zero.  The open question was *which* term of the z-momentum equation produces
the O(h) ``w``.

This driver answers it with the exact-state residual probe: evaluate every
term (viscous / convection / pressure / load) of the momentum equation at the
*exact* Kovasznay state (``w == 0``) and report the residual per unknown block
(cell-u/v/w and face-u/v/w).  A discretisation that is correct for the exact
field must return an O(h^2) residual in every *cell* row of the z-momentum
equation (the exact ``w`` is identically zero, so no term of that equation is
O(h)); residuals in prescribed rows are reactions, not defects.

Finding (n = 8, symmetry-extrusion, order-2 convection):
  * every *cell-w* row has residual 0.0 for every term -- the z-momentum cell
    equation is satisfied identically by ``w == 0``;
  * the only non-zero spanwise residuals live in the *face-w* rows (the
    shared-face equilibrium), dominated by the pressure term (3.8e-2 at n=8);
  * therefore the O(h) ``w`` is NOT a defect of the z-momentum cell
    discretisation: it is a *coupling* artefact -- the cell-w unknown is
    driven by the u, v, p through the shared-face / pressure coupling, even
    though its own equation would give ``w == 0``.

Consequence for verification practice: the ``w`` budget should be scored as a
coupling artefact (separate it, or impose the analytic tangential data), not
as evidence of a first-order z-momentum scheme.

Everything here is a *verification* driver: no solver code is modified.
"""

import json
from pathlib import Path

import numpy as np

from tensorfvm.verification.kovasznay_extrusion import (
    extrusion_solver,
    exact_state_residual,
)

COMPONENTS = ('u', 'v', 'w')


def run(n=8, reynolds=40., *, tangential='symmetry', convection_order=2,
        output=None):
    """Run the exact-state residual probe and return the per-term/block table.

    ``tangential`` and ``convection_order`` mirror ``extrusion_solver``; the
    default is the delivered round's setup (symmetry-extrusion, order-2).
    """
    mesh, solver, _ = extrusion_solver(
        n, reynolds, tangential=tangential, convection_order=convection_order)
    residual = exact_state_residual(mesh, solver, reynolds)

    # cell-w rows: the z-momentum cell equation.  A correct discretisation
    # must give 0 here for every term (w == 0 exactly).
    cell_w = {term: residual[term]['cell-w'] for term in residual}
    # face-w rows: the shared-face equilibrium (the only non-zero spanwise
    # residuals live here).
    face_w = {term: residual[term]['face-w'] for term in residual}

    result = dict(
        n=n, reynolds=reynolds, tangential=tangential,
        convection_order=convection_order,
        cell_w_residual=cell_w,
        face_w_residual=face_w,
        cell_w_all_zero=all(abs(v) == 0.0 for v in cell_w.values()),
        # which term dominates the face-w (the coupling channel)
        face_w_dominant_term=max(face_w, key=lambda k: abs(face_w[k])),
        face_w_dominant_value=float(max(abs(v) for v in face_w.values())),
    )
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--reynolds', type=float, default=40.)
    ap.add_argument('--tangential', default='symmetry',
                    choices=('symmetry', 'dirichlet'))
    ap.add_argument('--convection-order', type=int, default=2)
    ap.add_argument('--output', default=None)
    args = ap.parse_args(argv)
    r = run(args.n, args.reynolds, tangential=args.tangential,
            convection_order=args.convection_order, output=args.output)
    print(json.dumps(r, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
