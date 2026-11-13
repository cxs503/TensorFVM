#!/usr/bin/env python3
"""Round 8: lightweight sweep of production_limit for the SA closure on periodic hills.

Anchor: CASE-FVM-PERIODIC-HILLS
Mission: 18492 (selfstart, TensorFVM)

Lightweight version: 500 steps per case, 5 production_limit values.
Results saved to docs/case-periodic-hills/round8_sweep.json.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver, Statistics
from tensorfvm.periodic_hill_turbulence import SpalartAllmaras, SpalartAllmarasConfig

REFERENCE_XR_H = 5.34
REFERENCE_XS_H = 0.227

SWEEP_VALUES = [None, 10.0, 50.0, 100.0, 200.0]
NX, NY = 48, 24
REYNOLDS = 2800.0
STEPS = 500
STATS_START = 250


def run_case(production_limit, nx=NX, ny=NY, reynolds=REYNOLDS, steps=STEPS):
    """Run a single periodic hill case with the given production_limit."""
    config = PeriodicHillConfig(nx=nx, ny=ny, reynolds=reynolds, time_step_s=0.02)
    solver = PeriodicHillSolver(config)
    cfg = SpalartAllmarasConfig(production_limit=production_limit)
    solver.turbulence = SpalartAllmaras(solver, cfg)

    stats = Statistics(solver)
    t0 = time.time()

    for step in range(1, steps + 1):
        solver.advance(1)
        if step >= STATS_START:
            stats.accumulate(solver)

    elapsed = time.time() - t0
    sep, rea = solver.reattachment()

    # Mean statistics
    mean_u, mean_v, mean_G = stats.mean()
    mean_u_b = solver.bulk_velocity(mean_u)

    # Eddy viscosity stats
    nu_t = solver.turbulence.viscosity()
    nu_t_ratio = nu_t / solver.nu
    max_nu_t_ratio = float(nu_t_ratio.max())
    mean_nu_t_ratio = float(nu_t_ratio.mean())

    return {
        "production_limit": production_limit,
        "steps": steps,
        "elapsed_s": round(elapsed, 1),
        "x_s/H": sep if not math.isnan(sep) else None,
        "x_r/H": rea if not math.isnan(rea) else None,
        "bulk_velocity": round(mean_u_b, 6),
        "body_force": round(mean_G, 6),
        "max_nu_t/nu": round(max_nu_t_ratio, 2),
        "mean_nu_t/nu": round(mean_nu_t_ratio, 2),
        "x_r/H_error_pct": round(abs(rea - REFERENCE_XR_H) / REFERENCE_XR_H * 100, 1) if not math.isnan(rea) else None,
    }


def main():
    results = []
    print(f"{'production_limit':>18} | {'x_r/H':>8} | {'max nu_t/nu':>11} | {'error%':>7} | {'time_s':>7}")
    print("-" * 65)

    for pl in SWEEP_VALUES:
        t0 = time.time()
        r = run_case(pl)
        elapsed = time.time() - t0
        results.append(r)
        pl_str = "None" if pl is None else f"{pl:.0f}"
        xr_str = f"{r['x_r/H']:.3f}" if r["x_r/H"] is not None else "nan"
        err_str = f"{r['x_r/H_error_pct']:.1f}" if r["x_r/H_error_pct"] is not None else "nan"
        print(f"{pl_str:>18} | {xr_str:>8} | {r['max_nu_t/nu']:>11.2f} | {err_str:>7} | {elapsed:>7.1f}")

    # Save results
    out_path = Path(__file__).parent.parent / "docs" / "case-periodic-hills" / "round8_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"sweep": results, "reference": {"x_r/H": REFERENCE_XR_H, "x_s/H": REFERENCE_XS_H}}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
