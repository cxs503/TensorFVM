#!/usr/bin/env python3
"""Round 23 driver: why did the refined rungs of the ladder diverge?

Round 22 pre-registered a grid ladder for ``CASE-FVM-PERIODIC-HILLS``
(closure-free, ``Re = 2800``, CFL held fixed) and submitted the four rungs.
All four jobs returned ``rc = 0``, but the three refined artefacts
(``144x72``, ``192x96``, ``240x120``) carry ``samples = 0`` and
``config.diverged_at_step`` = 508 / 538 / 652 -- the solver's internal
non-finite guard ``break`` s out of the loop, so the exit code says nothing.
Wall-clock confirms it: 9 / 17 / 31 s instead of the hours a 30k-step run
needs.

That leaves the pre-registered attribution **unmeasurable** rather than
"inconclusive because the ladder is short", and it raises a concrete question
this script answers with a cheap, self-contained experiment:

    is the startup blow-up controlled by the time step (a stability limit that
    tightens with the mesh) or is it mesh-intrinsic at any ``dt``?

Method: step the *same* solver family (:func:`build_solver`, closure-free) one
step at a time, recording ``max|u|`` before the guard fires, for a small matrix
of ``(nx, ny, dt)``.  Nothing is time-averaged -- these are 700-2000 step runs,
seconds each -- so the script is safe to run on the control plane.

Reported per case: the step and physical time at which the guard fired, the
last finite ``max|u|``, the growth rate of ``max|u|`` over the last 100 finite
steps, the advective CFL and the explicit viscous number
``nu*dt/min(dx,dy)**2`` (the classic parabolic stability parameter of the
explicit diffusion term).

Usage
-----
    PYTHONPATH=src python scripts/round23_divergence_probe.py \
        --out docs/case-periodic-hills/round23_probe.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_round11 import build_solver

REYNOLDS = 2800.0
BULK = 1.0
NU = BULK * 1.0 / REYNOLDS  # U_b * H / Re with H = 1

#: ``(label, nx, ny, dt, max_steps)``.  ``dt = 0.0133`` is the round-22 rung B
#: step; the halvings test the stability limit at fixed mesh, the extra mesh
#: tests whether the limit moves with ``dx``.
CASES = [
    ("B_dt0133", 144, 72, 0.01330, 700),
    ("B_dt0067", 144, 72, 0.00665, 900),
    ("B_dt0033", 144, 72, 0.00333, 900),
    ("C_dt0100", 192, 96, 0.01000, 700),
    ("C_dt0025", 192, 96, 0.00250, 900),
    ("A_dt0200", 96, 48, 0.02000, 700),
]


def _cell_size(solver):
    dx = solver.config.length_m / solver.config.nx
    dy = solver.config.channel_top_m / solver.config.ny
    return float(dx), float(dy)


def probe_case(label, nx, ny, dt, max_steps):
    solver = build_solver("laminar", nx=nx, ny=ny, reynolds=REYNOLDS, dt=dt)
    dx, dy = _cell_size(solver)
    hist_step, hist_max_u = [], []
    diverged_at = None
    for step in range(1, max_steps + 1):
        try:
            solver.advance(1)
        except FloatingPointError:
            diverged_at = step
            break
        u = np.asarray(solver.u)
        if not np.all(np.isfinite(u)):
            diverged_at = step
            break
        if step % 5 == 0 or step <= 5:
            hist_step.append(step)
            hist_max_u.append(float(np.max(np.abs(u))))
        # cheap sanity bound: an incompressible run at U_b = 1 does not reach
        # max|u| = 100 within a startup transient; stop before 1e300 poisons
        # the JSON.
        if hist_max_u and hist_max_u[-1] > 100.0:
            diverged_at = step
            break
    max_u = np.asarray(hist_max_u, dtype=float)
    growth = None
    if max_u.size >= 20:
        tail = max_u[-20:]
        if np.all(tail > 0) and np.all(np.isfinite(tail)):
            growth = float(
                (math.log(tail[-1]) - math.log(tail[0]))
                / max(hist_step[-1] - hist_step[-20], 1)
            )
    cfl = float(BULK * dt / min(dx, dy))
    visc = float(NU * dt / min(dx, dy) ** 2)
    return {
        "label": label,
        "nx": int(nx),
        "ny": int(ny),
        "dt": float(dt),
        "max_steps": int(max_steps),
        "steps_run": int(hist_step[-1] if hist_step else 0),
        "diverged_at_step": diverged_at,
        "t_diverged": (None if diverged_at is None else float(diverged_at * dt)),
        "max_abs_u_last": (float(max_u[-1]) if max_u.size else None),
        "max_abs_u_peak": (float(np.max(max_u)) if max_u.size else None),
        "log_growth_per_step": growth,
        "dx_over_H": float(dx),
        "dy_over_H": float(dy),
        "advective_cfl": cfl,
        "explicit_viscous_number": visc,
        "bulk_velocity_last": float(solver.bulk_velocity()),
        "body_force_last": float(solver.body_force),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/case-periodic-hills/round23_probe.json")
    ap.add_argument("--cases", nargs="*", default=None,
                    help="subset of labels to run (default: all)")
    args = ap.parse_args(argv)

    cases = [c for c in CASES if args.cases is None or c[0] in args.cases]
    rows = []
    for label, nx, ny, dt, steps in cases:
        row = probe_case(label, nx, ny, dt, steps)
        rows.append(row)
        print("{label:9s} {nx}x{ny} dt={dt:.5f} steps={steps_run} "
              "diverged_at={diverged_at_step} t={t_diverged} "
              "max|u|={max_abs_u_last} cfl={advective_cfl:.3f} "
              "visc={explicit_viscous_number:.4f}".format(**row))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "case": "CASE-FVM-PERIODIC-HILLS round 23 (startup instability probe)",
        "closure": "laminar (nu_t == 0)",
        "reynolds": REYNOLDS,
        "nu": NU,
        "note": ("Short single-core runs; no time averaging.  Used to decide "
                 "whether the refined ladder rungs can be re-run at a smaller "
                 "time step or whether the instability is mesh-intrinsic."),
        "cases": rows,
    }
    out.write_text(json.dumps(payload, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())