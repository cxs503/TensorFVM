#!/usr/bin/env python
"""Robustness controls for the dam-break front deficit (node P447W2-N01).

The 50x100 Martin & Moyce run under-predicts the front by 40 % at t = 0.1 s.
Before that is reported as a capability gap, the obvious explanations have to
be eliminated by experiment, on a cheap 25x50 grid:

* gas-phase stiffness of ``pressure_form="reduced"`` (rho_l/rho_g = 1000):
  raise rho_g to 10 and 100;
* the pressure form itself: ``"total"`` instead of ``"reduced"``;
* momentum convection: switch ``momentum_advection`` off.

Each variant runs the same geometry/time window and prints the front error at
the three criterion times; the JSON is written to
``artifacts/dam_variants.json`` so the report quotes machine output only.

    PYTHONPATH=src python docs/P447W2-N01-vof-three-case/dam_variants.py
"""

from __future__ import annotations

import json
import math
import os
import time

import torch

torch.set_num_threads(max(1, int(os.environ.get("TP_THREADS", "8"))))

from tensorfvm.two_phase_solver import GridConfig, TwoPhaseConfig, TwoPhaseSolver
from tensorfvm.vof_threecase.dambreak import (
    DamBreakMMConfig,
    front_position,
    initial_alpha,
    martin_moyce_error,
    reduced_hydrostatic_pressure,
)

HERE = os.path.dirname(os.path.abspath(__file__))
VARIANTS = [
    dict(label="baseline (reduced, rho_g=1, advection on)"),
    dict(label="rho_g = 10", rho_g=10.0, mu_g=1.0e-4),
    dict(label="rho_g = 100", rho_g=100.0, mu_g=1.0e-3),
    dict(label="pressure_form = total", form="total", initp="none"),
    dict(label="momentum advection off", advection=False),
]


def run(ny: int, nx: int, label: str, rho_g: float = 1.0, form: str = "reduced",
        advection: bool = True, initp: str = "reduced", mu_g: float = 1.0e-5) -> dict:
    cfg = DamBreakMMConfig(ny=ny, nx=nx, aspect=2.0)
    tp = TwoPhaseConfig(enabled=True, rho_liquid=1000.0, rho_gas=rho_g,
                        mu_liquid=1.0e-3, mu_gas=mu_g, gravity=9.81,
                        pressure_form=form, warm_start_pressure=True,
                        momentum_advection=advection, vof_scheme="plic",
                        top_atmosphere=True, bottom_wall=True)
    solver = TwoPhaseSolver(
        GridConfig(nx=cfg.nx, ny=cfg.ny, length=cfg.length, height=cfg.height), tp)
    solver.set_alpha(initial_alpha(solver, cfg))
    if initp == "reduced":
        solver.p = reduced_hydrostatic_pressure(solver)
    m0 = float(solver.alpha.sum())
    samples, pending = [], [0.10, 0.20, 0.30]
    t0 = time.time()
    while solver.time < 0.30 - 1e-12:
        u_max = float(max(solver.u.abs().max(), solver.v.abs().max()))
        dt = 0.35 * min(solver.dx, solver.dy) / max(
            math.sqrt(9.81 * cfg.depth), u_max, 1e-9)
        if pending and solver.time + dt > pending[0]:
            dt = max(pending[0] - solver.time, 1e-12)
        solver.step(dt)
        while pending and solver.time >= pending[0] - 1e-9:
            t = pending.pop(0)
            front = front_position(solver.alpha, solver.dx)["x_front"]
            rec = martin_moyce_error(t, front, cfg)
            samples.append({"t": t, "x_front": front,
                            "t_star": rec["t_star"], "x_star_sim": rec["x_star_sim"],
                            "relative_error": rec["relative_error"]})
    out = {"label": label, "ny": ny, "nx": nx, "rho_gas": rho_g,
           "pressure_form": form, "momentum_advection": advection,
           "samples": samples,
           "mass_drift": abs(float(solver.alpha.sum()) - m0) / m0,
           "max_u": float(max(solver.u.abs().max(), solver.v.abs().max())),
           "wall_seconds": time.time() - t0}
    print(f"[variant] {label}: " + " ".join(
        f"t={s['t']:.2f}:{100 * s['relative_error']:+.1f}%" for s in samples)
        + f" drift={out['mass_drift']:.1e} ({out['wall_seconds']:.0f}s)", flush=True)
    return out


def main():
    records = [run(25, 50, **v) for v in VARIANTS]
    path = os.path.join(HERE, "artifacts", "dam_variants.json")
    with open(path, "w") as fh:
        json.dump({"case": "dam_break_variants", "grid": "25x50",
                   "records": records}, fh, indent=2, sort_keys=True, default=float)
    print(f"[write] {path}")


if __name__ == "__main__":
    main()