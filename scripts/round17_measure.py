#!/usr/bin/env python3
"""Round 17 confirmatory run: retain the mean wall shear of the SST variant.

The round-16 artefacts (``r16_patankar_96x48.json``) stored the *rising crossing
list* but not the mean bottom-wall shear profile, so ``x_s/H`` of that run cannot
be re-scored post hoc -- exactly the artefact-contract gap that hid the
reattachment.  This driver repeats the model-consistent (Patankar ``omega``.
sink) run at ``96x48`` / 20 000 steps with ``save_fields=True`` so the profile is
retained, then scores it with the certified round-17 operator and records the
measured ``S_P/(beta omega)`` ratio (``1.0`` = Patankar) so the artefact cannot
claim a variant it did not run.

    PYTHONPATH=src python3 scripts/round17_measure.py --nx 96 --ny 48 --steps 20000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from round16_acceptance import measure_sink_ratio  # noqa: E402

from tensorfvm.verification.periodic_hill_round11 import (  # noqa: E402
    build_solver, run_time_averaged_case)
from tensorfvm.verification.periodic_hill_round15 import write_json  # noqa: E402
from tensorfvm.verification.wall_shear_bubbles import (  # noqa: E402
    separation_reattachment)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
WINDOW = (1.9, 7.1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nx", type=int, default=96)
    ap.add_argument("--ny", type=int, default=48)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--burn-in", type=int, default=8000)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--reynolds", type=float, default=2800.0)
    ap.add_argument("--momentum-blend", type=float, default=0.75)
    ap.add_argument("--transport-blend", type=float, default=0.0)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args(argv)

    tag = args.tag or f"r17_patankar_{args.nx}x{args.ny}"
    solver = build_solver(
        "sst", nx=args.nx, ny=args.ny, reynolds=args.reynolds, dt=args.dt,
        convection_blend=args.momentum_blend,
        transport_convection_blend=args.transport_blend)
    sink = measure_sink_ratio(solver)

    t0 = time.time()
    payload = run_time_averaged_case(
        tag, "sst", nx=args.nx, ny=args.ny, steps=args.steps,
        sample_every=20, burn_in_steps=args.burn_in, dt=args.dt,
        reynolds=args.reynolds, solver=solver, out_dir=OUTDIR,
        save_fields=True, verbose=True, progress_every=max(args.steps // 8, 1))
    wall = time.time() - t0

    data = np.load(OUTDIR / f"{tag}.npz")
    tau = np.asarray(data["mean_tau_b"], dtype=float)
    x = np.asarray(data["x_over_h"], dtype=float)
    scored = separation_reattachment(tau, x, window=WINDOW)
    ref = payload["score_hill"]["reference"]["x_r_over_H"]
    x_r = scored["x_r_over_H"]

    out = {
        "round": 17,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "purpose": ("retain the mean bottom-wall shear of the model-consistent "
                    "SST variant and score it with the certified operator"),
        "tag": tag,
        "config": {"nx": args.nx, "ny": args.ny, "steps": args.steps,
                   "dt": args.dt, "reynolds": args.reynolds,
                   "burn_in_steps": args.burn_in,
                   "momentum_convection_blend": args.momentum_blend,
                   "transport_convection_blend": args.transport_blend,
                   "averaging_window": payload.get("config", {}).get(
                       "averaging_window")},
        "sink_coefficient_ratio": sink,
        "wall_clock_s": wall,
        "certified": {
            "status": scored["status"],
            "x_s_over_H": float(scored["x_s_over_H"]),
            "x_r_over_H": float(x_r),
            "out_of_window": bool(scored["out_of_window"]),
            "n_bubbles": int(scored["n_bubbles"]),
            "bubbles": scored["bubbles"],
            "all_crossings": scored["all_crossings"],
        },
        "reference": {"x_r_over_H": ref},
        "x_r_error_vs_ref": (float(abs(x_r - ref) / ref)
                             if np.isfinite(x_r) else None),
        "payload_x_r_over_H": payload.get("time_averaged", {}).get("x_r_over_H"),
        "stationarity": payload.get("stationarity"),
        "profile_comparison": payload.get("profile_comparison"),
        "max_nu_t_ratio": payload.get("time_averaged", {}).get("max_nu_t_ratio"),
        "mean_nu_t_ratio": payload.get("time_averaged", {}).get(
            "mean_nu_t_ratio_window"),
        "body_force": payload.get("time_averaged", {}).get("body_force"),
    }
    write_json(OUTDIR / f"round17_measure_{args.nx}x{args.ny}.json", out)
    print(json.dumps({k: out[k] for k in ("sink_coefficient_ratio",
                                          "certified", "x_r_error_vs_ref",
                                          "wall_clock_s")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())