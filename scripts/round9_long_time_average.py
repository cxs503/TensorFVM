#!/usr/bin/env python3
"""Round 9 driver: long time-averaged periodic-hill runs (CASE-FVM-PERIODIC-HILLS).

Why this exists
---------------
Rounds 1--8 each produced an *instantaneous* verdict and each one said the same
thing in a different way: at ``Re_b = 2800`` the 2-D URANS + SA solution keeps
evolving past ``t = 60`` (``max nu_t/nu`` 0.2 -> 36, ``x_r/H`` 2.6 -> 4.1 ->
6.1 -> nan -> 5.6).  Every round's recommendation #1 was the same: run long
enough to time-average.  That is what this driver does -- one configuration per
process, so the three meshes can be sharded on the cluster.

Outputs (per case, into ``docs/case-periodic-hills/``):
    round9_<tag>.json  -- configuration, time series, stationarity report, gates
    round9_<tag>.npz   -- time-averaged u, v, tau_b, tau_t, mesh coordinates

Usage
-----
    PYTHONPATH=src python scripts/round9_long_time_average.py --case-index 0 --case-count 1
    PYTHONPATH=src python scripts/round9_long_time_average.py --list
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver, Statistics
from tensorfvm.periodic_hill_turbulence import SpalartAllmaras, SpalartAllmarasConfig
from tensorfvm.verification.periodic_hill_round6 import (
    BREUER_PROFILE_Y05,
    REFERENCE,
    REFERENCE_TOLERANCE,
)
from tensorfvm.verification.periodic_hill_round9 import (
    averaging_window,
    compare_profile,
    horizontal_profile,
    stationarity_report,
    time_averaged_reattachment,
)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"

#: (tag, nx, ny, steps, sample_every, burn_in_steps, dt)
CASES = [
    ("48x24", 48, 24, 30000, 20, 10000, 0.02),
    ("96x48", 96, 48, 20000, 20, 8000, 0.02),
    ("144x72", 144, 72, 12000, 20, 6000, 0.02),
]


def _finite(obj):
    """Replace non-finite floats by ``None`` so the JSON is strict (no NaN)."""
    if isinstance(obj, dict):
        return {k: _finite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_case(tag, nx, ny, steps, sample_every, burn_in, dt,
             reynolds=2800.0, hill_amplitude=1.0, production_limit=None,
             verbose=True):
    config = PeriodicHillConfig(nx=nx, ny=ny, reynolds=reynolds,
                                time_step_s=dt, hill_amplitude=hill_amplitude)
    solver = PeriodicHillSolver(config)
    solver.turbulence = SpalartAllmaras(
        solver, SpalartAllmarasConfig(production_limit=production_limit))

    stats = Statistics(solver)
    series = {"t": [], "step": [], "x_s_over_H": [], "x_r_over_H": [],
              "body_force": [], "bulk_velocity": [], "max_nu_t_ratio": [],
              "mean_nu_t_ratio": []}
    # block-wise averaging of the wall shear: the metric the gate uses is the
    # reattachment of the *mean* shear, so stationarity must be judged on that
    # metric, not on the (often ill-defined) instantaneous crossing count
    n_blocks = 4
    block_tau = [np.zeros(solver.mesh.tau_bottom.shape[0]) for _ in range(n_blocks)]
    block_count = np.zeros(n_blocks, dtype=int)
    block_width = max((steps - burn_in) / n_blocks, 1.0)
    t0 = time.time()
    for step in range(1, steps + 1):
        solver.advance(1)
        if step >= burn_in and (step - burn_in) % sample_every == 0:
            sep, rea = solver.reattachment()
            nu_t_ratio = solver.turbulence.viscosity() / solver.nu
            series["t"].append(step * dt)
            series["step"].append(step)
            series["x_s_over_H"].append(float(sep))
            series["x_r_over_H"].append(float(rea))
            series["body_force"].append(float(solver.body_force))
            series["bulk_velocity"].append(float(solver.bulk_velocity()))
            series["max_nu_t_ratio"].append(float(nu_t_ratio.max()))
            series["mean_nu_t_ratio"].append(float(nu_t_ratio.mean()))
            stats.accumulate(solver)
            k = min(n_blocks - 1, int((step - burn_in) / block_width))
            block_tau[k] += solver.wall_shear()[0]
            block_count[k] += 1
        if verbose and step % 5000 == 0:
            print(f"[{tag}] step {step}/{steps} t={step*dt:g} "
                  f"elapsed {time.time()-t0:.0f}s x_r={series['x_r_over_H'][-1] if series['x_r_over_H'] else 'nan'}",
                  flush=True)
    wall = time.time() - t0

    mean_u, mean_v, mean_G = stats.mean()
    mean_tau_b, mean_tau_t = stats.mean_wall_shear()
    x_over_h = solver.mesh.xc[0] / config.height_m

    rea = time_averaged_reattachment(mean_tau_b, x_over_h)
    hp = horizontal_profile(mean_u, solver.mesh.yc, x_over_h, 0.5, config.height_m)
    cmp = compare_profile(hp["x_over_h"], hp["u"], bulk_velocity=config.bulk_velocity)

    # --- stationarity of the metrics the gate uses -------------------------
    t = np.asarray(series["t"])
    xr = np.asarray(series["x_r_over_H"])
    finite = np.isfinite(xr)
    stationarity = {
        "x_r_over_H": stationarity_report(t[finite], xr[finite], n_blocks=4),
        "block_window_x_r_over_H": averaging_window(t[finite], xr[finite], n_blocks=8),
        "max_nu_t_ratio": stationarity_report(t, np.asarray(series["max_nu_t_ratio"]), n_blocks=4),
        "body_force": stationarity_report(t, np.asarray(series["body_force"]), n_blocks=4),
        "bulk_velocity": stationarity_report(t, np.asarray(series["bulk_velocity"]), n_blocks=4),
        "snapshots_without_reattachment": int(np.count_nonzero(~finite)),
    }

    coverage = float(finite.mean()) if finite.size else 0.0
    block_xr = []
    for k in range(n_blocks):
        if block_count[k] == 0:
            block_xr.append(None)
            continue
        rep_k = time_averaged_reattachment(block_tau[k] / block_count[k], x_over_h)
        block_xr.append(rep_k["x_r_over_H"])
    finite_blocks = [v for v in block_xr if v is not None and np.isfinite(v)]
    block_spread = (float(max(finite_blocks) - min(finite_blocks))
                    / abs(float(np.mean(finite_blocks)))
                    if len(finite_blocks) == n_blocks else float("inf"))
    stationarity["x_r_over_H_by_block"] = block_xr
    stationarity["x_r_over_H_block_spread_rel"] = block_spread
    stationarity["x_r_over_H_block_samples"] = [int(v) for v in block_count]
    stationarity["reattachment_coverage"] = coverage

    x_r = rea["x_r_over_H"]
    ref_xr = REFERENCE["x_r_over_H"]
    gate = {
        "x_r_over_H": {
            "computed": x_r,
            "reference": ref_xr,
            "rel_error": (abs(x_r - ref_xr) / ref_xr) if np.isfinite(x_r) else None,
            "tolerance": REFERENCE_TOLERANCE,
            "pass": bool(np.isfinite(x_r) and abs(x_r - ref_xr) / ref_xr <= REFERENCE_TOLERANCE),
        },
        "y_H_0.5_profile": {
            "stations": cmp["n_stations"],
            "stations_pass": cmp["n_pass"],
            "worst": cmp["worst"],
            "ok": cmp["ok"],
            "tolerance": REFERENCE_TOLERANCE,
        },
        "stationary_window": bool(block_spread <= REFERENCE_TOLERANCE),
        "reattachment_coverage": coverage,
    }
    gate["pass"] = bool(gate["x_r_over_H"]["pass"] and gate["y_H_0.5_profile"]["ok"]
                        and gate["stationary_window"])

    payload = {
        "case": "CASE-FVM-PERIODIC-HILLS round 9 (time-averaged gate)",
        "tag": tag,
        "config": {
            "nx": nx, "ny": ny, "steps": steps, "sample_every": sample_every,
            "burn_in_steps": burn_in, "time_step_s": dt,
            "end_time": steps * dt, "averaging_window": (burn_in * dt, steps * dt),
            "reynolds": reynolds, "hill_amplitude": hill_amplitude,
            "production_limit": production_limit,
            "convection_blend": solver.config.convection_blend,
            "bulk_velocity": solver.config.bulk_velocity,
            "nu": solver.nu,
        },
        "wall_clock_s": wall,
        "samples": stats.samples,
        "series": {k: [float(x) for x in v] for k, v in series.items()},
        "time_averaged": {
            "x_s_over_H": rea["x_s_over_H"],
            "x_r_over_H": rea["x_r_over_H"],
            "crossings": rea["crossings"],
            "body_force": mean_G,
            "max_nu_t_ratio": float(series["max_nu_t_ratio"][-1]) if series["max_nu_t_ratio"] else None,
            "mean_nu_t_ratio_window": float(stationarity["max_nu_t_ratio"]["mean"]),
        },
        "horizontal_profile_y_H_0.5": {
            "x_over_H": [float(v) for v in hp["x_over_h"]],
            "u_over_Ub": [float(v) for v in hp["u"]],
            "inadmissible_columns": int(len(hp["inadmissible"])),
            "reference": {str(k): v for k, v in BREUER_PROFILE_Y05.items()},
        },
        "profile_comparison": cmp,
        "stationarity": stationarity,
        "gate": gate,
    }
    np.savez_compressed(
        OUTDIR / f"round9_{tag}.npz",
        mean_u=mean_u, mean_v=mean_v, mean_tau_b=mean_tau_b, mean_tau_t=mean_tau_t,
        yc=solver.mesh.yc, xc=solver.mesh.xc, x_n=solver.mesh.x_n[0],
        x_over_h=x_over_h, u=solver.u, v=solver.v,
    )
    (OUTDIR / f"round9_{tag}.json").write_text(
        json.dumps(_finite(payload), indent=1, default=float))
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case-index", type=int, default=0)
    ap.add_argument("--case-count", type=int, default=1)
    ap.add_argument("--tag")
    ap.add_argument("--nx", type=int)
    ap.add_argument("--ny", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--burn-in", type=int)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--reynolds", type=float, default=2800.0)
    ap.add_argument("--production-limit", type=float, default=None)
    args = ap.parse_args(argv)

    if args.list:
        for i, c in enumerate(CASES):
            print(i, c)
        return 0

    OUTDIR.mkdir(parents=True, exist_ok=True)
    results = []
    if args.tag is not None:
        cases = [(args.tag, args.nx or 48, args.ny or 24, args.steps or 30000,
                  args.sample_every, args.burn_in or 10000, args.dt)]
    else:
        cases = [CASES[i] for i in range(len(CASES))
                 if i % max(args.case_count, 1) == args.case_index]
    for c in cases:
        tag, nx, ny, steps, every, burn, dt = c[:7]
        payload = run_case(tag, nx, ny, steps, every, burn, dt,
                           reynolds=args.reynolds,
                           production_limit=args.production_limit)
        results.append({"tag": tag, "gate": payload["gate"],
                        "x_r_over_H": payload["time_averaged"]["x_r_over_H"]})
        print(f"[{tag}] DONE x_r/H={payload['time_averaged']['x_r_over_H']} "
              f"gate_pass={payload['gate']['pass']} wall={payload['wall_clock_s']:.0f}s",
              flush=True)
    print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())