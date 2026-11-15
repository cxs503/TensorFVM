#!/usr/bin/env python3
"""Round 23 probe: *why* do the refined ladder levels of the periodic hill diverge?

Round 22 queued the four-level ladder of ``CASE-FVM-PERIODIC-HILLS`` and left the
verdict open because the jobs had not returned.  They have now returned, and the
answer is not a convergence number at all::

    tag        diverged_at_step   samples    x_r/H
    96x48 dt.01      --            1201     8.230
    144x72           508              0     --      (max|u| -> 1e3 at step ~500)
    192x96           538              0     --
    240x120          652              0     --

i.e. **every refined level blows up during the averaging window**, while the base
mesh survives 40 000 steps at twice the time step.  A grid-convergence study is
therefore impossible until the instability is attributed.  This probe is the
attribution: one configuration per process, only the named knob changes.

The candidate mechanisms that scale *the wrong way* under refinement (i.e. that
can make a finer mesh less stable at fixed CFL):

``blend``       momentum convection is explicit and blended central/upwind
                (``convection_blend * central + (1-blend) * upwind``); the
                central part is anti-diffusive, so the stable CFL depends on how
                well resolved the shear layer is -- refining *creates* the
                high-cell-Peclet content that the coarse mesh smeared away.
``sweeps``      the non-orthogonal defect correction of :meth:`compact_gradient`
                is a fixed number of sweeps; the size of the defect relative to
                the orthogonal part does not vanish under refinement.
``dt``          the time step of the ladder was chosen at fixed CFL, so ``dt``
                shrinks with ``h``: if the instability is a physical one (not a
                CFL one) shrinking ``dt`` must not help.

Usage
-----
    PYTHONPATH=src python scripts/round23_stability_probe.py \\
        --nx 144 --ny 72 --dt 0.0133 --steps 1200 [--blend 0.75] [--sweeps 2] \\
        [--out /tmp/probe.json]
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver

#: a state is called diverged when ``max|u|`` passes this multiple of the bulk
#: velocity; nothing physical in this case exceeds ~2.5 U_b (round 20 artefacts),
#: so 50 U_b is safe against a false positive while still tripping long before
#: the guard in ``run_time_averaged_case`` (which waits for non-finite values).
DIVERGED_FACTOR = 50.0


def run(nx: int, ny: int, dt: float, steps: int, blend: float | None,
        sweeps: int | None, reynolds: float = 2800.0, seed_steps: int = 0):
    cfg_kwargs = dict(nx=nx, ny=ny, reynolds=reynolds, time_step_s=dt)
    if blend is not None:
        cfg_kwargs["convection_blend"] = blend
    cfg = PeriodicHillConfig(**cfg_kwargs)
    solver = PeriodicHillSolver(cfg)
    if sweeps is not None:
        solver.compact_gradient = lambda field, _s=int(sweeps), _b=solver: (
            _b.__class__.compact_gradient(_b, field, nonorthogonal_sweeps=_s))

    m = solver.mesh
    dn_mid = 0.5 * (m.dn_n[:-1] + m.dn_n[1:])            # per eta row
    trace, div_step, div_where = [], None, None
    t0 = time.time()
    for k in range(1, steps + 1):
        solver.advance(1)
        umax = float(np.abs(solver.u).max())
        if k % 50 == 0 or k <= 5:
            cfl = np.abs(solver.u) * dt / m.dn_e + np.abs(solver.v) * dt / dn_mid[:, None]
            trace.append({"step": k, "max_abs_u": umax,
                          "max_abs_v": float(np.abs(solver.v).max()),
                          "body_force": float(solver.body_force),
                          "cfl_local_max": float(np.nanmax(cfl))})
        if not np.isfinite(umax) or umax > DIVERGED_FACTOR * cfg.bulk_velocity:
            div_step = k
            j, i = np.unravel_index(np.argmax(np.abs(solver.u)), solver.u.shape)
            div_where = {
                "x_over_H": float(solver.mesh.xc[j, i] / cfg.height_m),
                "y_over_H": float(solver.mesh.yc[j, i] / cfg.height_m),
            }
            break
    wall = time.time() - t0
    return {
        "nx": nx, "ny": ny, "dt": dt, "steps_requested": steps,
        "steps_done": len(trace) and (div_step or steps),
        "blend": cfg.convection_blend, "sweeps": sweeps,
        "diverged_at_step": div_step,
        "diverged_where": div_where,
        "max_abs_u_final": trace[-1]["max_abs_u"] if trace else None,
        "trace": trace,
        "wall_clock_s": wall,
        "cfl_x": cfg.bulk_velocity * dt / cfg.dx,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nx", type=int, required=True)
    ap.add_argument("--ny", type=int, required=True)
    ap.add_argument("--dt", type=float, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--blend", type=float, default=None)
    ap.add_argument("--sweeps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    res = run(args.nx, args.ny, args.dt, args.steps, args.blend, args.sweeps)
    txt = json.dumps(res, indent=1)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(txt)
    print(f"[{args.nx}x{args.ny} dt={args.dt} blend={res['blend']} sweeps={res['sweeps']}] "
          f"diverged_at={res['diverged_at_step']} "
          f"where={res['diverged_where']} wall={res['wall_clock_s']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())