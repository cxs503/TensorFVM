"""Run one Hysing (2009) rising-bubble leg and persist the full evidence.

The campaign driver keeps only the summary record per leg; grid-convergence and
reference-provenance audits also need the per-step history (``y_c(t)``,
circularity, ``umax``, volume drift) and the interface snapshots, so this
script dumps both.

Usage::

    PYTHONPATH=src python3 scripts/run_hysing_leg.py --cpd 16 \
        --out docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing

The declared-resolution leg is ``cpd == hysing_gate.DECLARED_CPD`` (16); the
file name carries the resolution so the coarse leg is never mistaken for the
declared one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch

from tensorfvm.twophase_gap.hysing_gate import (
    HysingCase,
    bubble_metrics,
    run_hysing_leg,
)


def run_with_fields(cpd: int, end_time: float, pressure_form: str,
                    snapshot_times: tuple[float, ...], out: pathlib.Path) -> dict:
    """Same integration as :func:`run_hysing_leg`, plus interface snapshots."""
    from tensorfvm.two_phase_solver import TwoPhaseSolver

    case = HysingCase(cells_per_diameter=cpd, end_time=end_time,
                      pressure_form=pressure_form)
    grid, tp = case.solve_config()
    solver = TwoPhaseSolver(grid, tp)
    solver.set_alpha(case.initial_alpha(grid.nx, grid.ny, solver.dx, solver.dy))

    dx, dy = solver.dx, solver.dy
    snaps: dict[str, np.ndarray] = {}
    pending = list(snapshot_times)
    history: list[dict] = []
    while solver.time < end_time:
        dt = solver.stable_time_step()
        if solver.time + dt > end_time:
            dt = end_time - solver.time
        solver.step(dt)
        m = bubble_metrics(solver.alpha, dx, dy)
        m["time"] = float(solver.time)
        m["umax"] = float(max(solver.u.abs().max(), solver.v.abs().max()))
        history.append(m)
        while pending and solver.time >= pending[0] - 1e-9:
            t_snap = pending.pop(0)
            snaps[f"t{t_snap:g}"] = solver.alpha.detach().cpu().numpy().astype(np.float32)
    return {"case": case, "grid": grid, "history": history, "snaps": snaps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpd", type=int, default=16)
    ap.add_argument("--end-time", type=float, default=3.0)
    ap.add_argument("--pressure-form", default="total",
                    choices=("total", "reduced"))
    ap.add_argument("--snapshots", nargs="+", type=float,
                    default=[0.0, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--out", default="docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    res = run_hysing_leg(args.cpd, end_time=args.end_time,
                         pressure_form=args.pressure_form)
    record = res["record"]
    record["wall_seconds"] = time.time() - t0
    (out / f"leg_hysing_{args.cpd}.json").write_text(
        json.dumps({"record": record, "history": res["history"]}, indent=2) + "\n")
    print(f"[leg] cpd={args.cpd} steps={record['steps']} "
          f"peak={record['peak_velocity']:.4f} window={record['terminal_velocity']:.4f} "
          f"cmin={record['min_circularity']:.4f} "
          f"yc_end={res['history'][-1]['yc']:.4f} wall={record['wall_seconds']:.0f}s",
          flush=True)

    # snapshots are an extra (expensive) pass only when asked for
    if args.snapshots:
        t1 = time.time()
        ext = run_with_fields(args.cpd, args.end_time, args.pressure_form,
                              tuple(args.snapshots), out)
        np.savez_compressed(
            out / f"fields_hysing_{args.cpd}.npz",
            **{f"alpha_{k}": v for k, v in ext["snaps"].items()},
            **{"history": np.array([[h["time"], h["yc"], h["xc"],
                                      h["circularity"], h["umax"]]
                                     for h in ext["history"]])},
        )
        (out / f"fields_hysing_{args.cpd}.json").write_text(json.dumps({
            "cpd": args.cpd, "nx": ext["grid"].nx, "ny": ext["grid"].ny,
            "domain": [ext["grid"].length, ext["grid"].height],
            "snapshots": sorted(ext["snaps"]), "wall_seconds": time.time() - t1,
            "note": "alpha: 1=liquid, 0=gas; arrays are cell-centred (ny, nx)",
        }, indent=2) + "\n")
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"[snapshots] {sorted(ext['snaps'])} wall={time.time() - t1:.0f}s",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())