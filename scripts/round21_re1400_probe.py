#!/usr/bin/env python3
"""Round 21 probe: *where* does the closure-free ``Re = 1400`` run diverge?

Round 20's sister-case ladder came back as

    Re = 700   laminar  -> converged, x_r/H = 8.078
    Re = 1400  laminar  -> FloatingPointError at step 3013 (28.6 s of wall clock)
    Re = 2800  laminar  -> converged, x_r/H = 8.264

A blow-up at the *middle* Reynolds number, while both sides converge, is not a
monotone physical trend.  Before it can be quoted as "Re = 1400 is unsteady and
the model cannot follow it", it has to be located: is the run leaving the
explicit-convection stability limit (``CFL = |u| dt / dx > 1``), is the
bulk-velocity controller winding up, or is the velocity field itself diverging
with the controller still nominal?

This probe advances the same configuration step by step, records the
diagnostics every ``--every`` steps and writes them to
``docs/case-periodic-hills/round21_re1400_probe.json``.  It never writes an
``x_r/H``: a diverged run has no reattachment length, and the whole point of
round 19's lesson is that a diverged case must never be scored as "no
crossing".

Usage
-----
    PYTHONPATH=src python scripts/round21_re1400_probe.py --steps 3200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_round11 import build_solver

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def probe(reynolds: float = 1400.0, *, nx: int = 96, ny: int = 48,
          dt: float = 0.02, steps: int = 3200, every: int = 25,
          convection_blend: float | None = None) -> dict:
    """Advance until divergence (or ``steps``) and return the diagnostic trace."""
    solver = build_solver("laminar", nx=nx, ny=ny, reynolds=reynolds, dt=dt,
                          convection_blend=convection_blend)
    cfg = solver.config
    dx_min = float(np.min(np.linalg.norm(
        np.diff(solver.mesh.xc[0, :].ravel()) if False else
        np.diff(np.asarray(solver.mesh.xc)[0, :]), axis=0))) if False else None
    dx = cfg.length_m / nx
    trace = {"step": [], "t": [], "max_abs_u": [], "rms_u": [], "bulk_velocity": [],
             "body_force": [], "cfl": [], "argmax_eta": [], "argmax_x_over_h": []}
    diverged_at = None
    t0 = time.time()
    for step in range(1, steps + 1):
        try:
            solver.advance(1)
        except FloatingPointError as exc:  # the round-10 non-finite guard
            diverged_at = step
            trace["divergence_message"] = str(exc)
            break
        if step % every == 0 or step == 1:
            u = np.asarray(solver.u)
            umax = float(np.max(np.abs(u)))
            trace["step"].append(step)
            trace["t"].append(step * dt)
            trace["max_abs_u"].append(umax)
            trace["rms_u"].append(float(np.sqrt(np.mean(u ** 2))))
            trace["bulk_velocity"].append(float(solver.bulk_velocity()))
            trace["body_force"].append(float(solver.body_force))
            trace["cfl"].append(umax * dt / dx)
            j, i = np.unravel_index(int(np.argmax(np.abs(u))), u.shape)
            trace["argmax_eta"].append(float(solver.mesh.eta_cell[j, i]))
            trace["argmax_x_over_h"].append(float(solver.mesh.xc[j, i] / cfg.height_m))
    wall = time.time() - t0
    payload = {
        "reynolds": float(reynolds), "nx": int(nx), "ny": int(ny), "dt": float(dt),
        "steps_requested": int(steps), "convection_blend": (
            float(solver.config.convection_blend)),
        "diverged_at_step": diverged_at,
        "wall_clock_s": wall, "trace": trace,
        "dx_cells": dx,
    }
    payload["diagnosis"] = diagnose(payload)
    return payload


def diagnose(payload: dict) -> dict:
    """Classify the blow-up from the trace alone (no physics claims)."""
    tr = payload["trace"]
    cfl = np.asarray(tr["cfl"], dtype=float)
    umax = np.asarray(tr["max_abs_u"], dtype=float)
    out = {"diverged": payload["diverged_at_step"] is not None,
           "max_cfl_observed": float(np.nanmax(cfl)) if cfl.size else None,
           "max_abs_u_observed": float(np.nanmax(umax)) if umax.size else None}
    if not out["diverged"]:
        out["class"] = "no_divergence_in_probe_budget"
        return out
    if cfl.size >= 4:
        # exponential growth rate over the last quarter of the trace
        k0 = max(0, int(0.75 * cfl.size))
        k = np.arange(k0, cfl.size)
        finite = np.isfinite(umax[k]) & (umax[k] > 0)
        if finite.sum() >= 3:
            slope = float(np.polyfit(k[finite], np.log(umax[k][finite]), 1)[0])
            out["growth_rate_per_sample"] = slope
            out["sample_interval_steps"] = (tr["step"][k[finite][1]]
                                            - tr["step"][k[finite][0]])
            out["growth_rate_per_step"] = slope / out["sample_interval_steps"]
    out["class"] = ("cfl_violation_before_divergence"
                    if out["max_cfl_observed"] is not None and out["max_cfl_observed"] > 1.0
                    else "divergence_with_cfl_below_one")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reynolds", type=float, default=1400.0)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--steps", type=int, default=3200)
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--out", default=str(OUTDIR / "round21_re1400_probe.json"))
    args = ap.parse_args(argv)
    payload = probe(args.reynolds, nx=args.nx, ny=args.ny, dt=args.dt,
                    steps=args.steps, every=args.every)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: payload[k] for k in
                      ("reynolds", "dt", "diverged_at_step", "wall_clock_s",
                       "diagnosis")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())