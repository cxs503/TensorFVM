#!/usr/bin/env python3
"""One-refinement-level runs for the natural-convection cavity ladder.

The archived ladder (``docs/incompressible-baselines/cavity_n0*.json``) stops at
``n = 96``.  That is enough to show that ``Nu`` is grid converged to a couple of
percent but *not* enough to decide whether the stream-function anchor
(``|psi|/alpha`` vs de Vahl Davis) is a discretisation error or a
normalisation mismatch: on the archived grids the metric is still marching
(11.17 -> 10.33 -> 10.08 at ``Ra = 1e5``).

.. warning::
   The archived ladder was produced on the **frozen** deferred-correction path
   (``dc_relax = 1``, record stage name ``deferred_1``).  The current
   :func:`run_fvm_incompressible_baselines.run_one` default is the *damped*
   path (``dc_relax = 0.05``, stage ``limited_1``) and that path blows up
   (``NaN`` continuity, then ``uref = inf`` -> ``ZeroDivisionError``) on the
   cavity at ``n = 32, Ra = 1e5`` with both Poisson backends.  Extending the
   ladder therefore requires ``dc_relax = 1.0`` explicitly; with it the archived
   ``n = 32, Ra = 1e5`` record is reproduced to machine precision (see
   ``docs/verification-natural-convection-metrics/audit.json``).

This script runs exactly one level beyond the archive with the *same* solver
settings as the archive (cavity build, relaxation 0.5/0.2/0.7,
``momentum_sweeps=5``, ``p_cycles=3``, one frozen deferred-correction cycle,
``tol=1e-7``) and writes the same JSON record schema, so the new point can be
consumed by ``scripts/audit_natural_convection_metrics.py`` without a special
case.

Usage::

    python3 scripts/run_cavity_refinement.py --n 128 --rayleigh 1e4 --rayleigh 1e5
    python3 scripts/run_cavity_refinement.py --n 32 --rayleigh 1e5 --out /tmp/refcheck
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import run_fvm_incompressible_baselines as base  # noqa: E402  (path set above)

# Iteration budgets of the archived ladder, extended by one level.  The ``*1.4``
# factor for ``Ra > 1e5`` is the same rule the archive used (the Ra = 1e6 flow
# needs more outer iterations to settle).
BUDGET = {32: 20000, 64: 40000, 96: 60000, 128: 120000}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--rayleigh", type=float, action="append", required=True)
    ap.add_argument("--out", default="runs/incompressible-refinement")
    ap.add_argument("--poisson", default="multigrid", choices=("sparse", "multigrid"))
    ap.add_argument("--max-iter", type=int, default=0)
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    for ra in args.rayleigh:
        tag = f"cavity_n{args.n:03d}_ra{ra:.0e}".replace("+", "")
        budget = args.max_iter or int(
            BUDGET[args.n] * (1.4 if ra > 1e5 else 1.0))
        job = dict(case="natural_convection_cavity", level=args.n, rayleigh=ra,
                   tag=tag, budget=dict(max_iter=int(budget)), init_from=None)
        print(f"[{tag}] start (max_iter={budget}, poisson={args.poisson})", flush=True)
        record = base.run_one(job, args.out, poisson_solver=args.poisson,
                              init_dir=None, deferred=1, dc_relax=1.0)
        print(f"[{tag}] done in {record['wall_seconds']}s "
              f"metrics={json.dumps(record['metrics'], default=float)[:400]}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())