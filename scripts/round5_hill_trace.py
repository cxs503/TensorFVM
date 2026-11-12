"""Trace of the Re = 2800 hill run with the Spalart--Allmaras closure attached.

Mission 16880, round 5 of ``CASE-FVM-PERIODIC-HILLS``.  The driver
(:mod:`tensorfvm.verification.periodic_hill_round5`) reports the criteria; this
script records *how the answer develops in time*, because the closure does not
reach equilibrium inside the affordable window and a single snapshot would hide
that (round 4 recorded the same trap for the instantaneous laminar trace).

    PYTHONPATH=src python scripts/round5_hill_trace.py [out.json] [steps]
"""
from __future__ import annotations

import json
import sys

import numpy as np

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_turbulence import SpalartAllmaras

OUT = sys.argv[1] if len(sys.argv) > 1 else "docs/case-periodic-hills/round5_hill_trace.json"
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
CHUNK = 250
DT = 0.01

solver = PeriodicHillSolver(PeriodicHillConfig(
    nx=48, ny=24, reynolds=2800.0, time_step_s=DT, body_force_relaxation=1.0))
solver.advance(500)                      # laminar warm-up (round-4 state)
closure = SpalartAllmaras(solver)
solver.turbulence = closure
x = solver.mesh.x_n[0] / solver.config.height_m

trace = []
total = 0
while total < STEPS:
    step = min(CHUNK, STEPS - total)
    solver.advance(step)
    total += step
    tau_b, _ = solver.wall_shear()
    negative = x[tau_b < 0.0]
    separation = float(negative.min()) if negative.size else float("nan")
    reattachment = float(negative.max()) if negative.size else float("nan")
    trace.append({
        "step": int(solver.steps), "t": float(solver.steps * DT),
        "max_abs_velocity": float(np.abs(solver.u).max()),
        "max_nu_t_over_nu": float(solver.nu_t.max() / solver.nu),
        "mean_nu_t_over_nu": float(solver.nu_t.mean() / solver.nu),
        "body_force": float(solver.body_force),
        "bulk_velocity": solver.bulk_velocity(),
        "x_s_over_H": separation, "x_r_over_H": reattachment,
        "min_wall_shear": float(tau_b.min()),
        "max_abs_flux_divergence": float(np.abs(solver.flux_divergence()).max()),
    })

payload = {
    "what": "Re=2800 periodic hill, 48x24, dt=0.01, Spalart-Allmaras attached after 500 laminar steps",
    "nu": solver.nu, "dt": DT, "warmup_steps": 500,
    "trace": trace,
    "final_wall_shear": {"x_over_H": x.tolist(),
                         "tau_b": solver.wall_shear()[0].tolist()},
}
with open(OUT, "w") as fh:
    json.dump(payload, fh, indent=2)
for row in trace:
    print(f"t={row['t']:6.1f} max|u|={row['max_abs_velocity']:.3f} "
          f"max nu_t/nu={row['max_nu_t_over_nu']:7.2f} "
          f"x_s/H={row['x_s_over_H']:.2f} x_r/H={row['x_r_over_H']:.2f} "
          f"G={row['body_force']:.2e}")
