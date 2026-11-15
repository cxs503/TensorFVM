#!/usr/bin/env python3
"""Round 15 driver for CASE-FVM-PERIODIC-HILLS (one configuration per process).

Two experiments, both shardable:

    # the discriminator: plane channel with the same solver/closure/grid style
    PYTHONPATH=src python scripts/round15_channel_and_rescore.py channel \
        --ny 64 --nx 8 --steps 30000 --burn-in 10000

    # the re-score: the healed operator (rounds 12-13 fixes are in HEAD) on the
    # periodic hill, first-order and second-order transport convection
    PYTHONPATH=src python scripts/round15_channel_and_rescore.py hill --tag r15_96x48
    PYTHONPATH=src python scripts/round15_channel_and_rescore.py hill \
        --tag r15b_96x48 --transport-blend 1.0

Outputs (``docs/case-periodic-hills/``):

    round15_channel_<closure>[_<tag>].json   -- blocks, profiles, gate verdict
    round11_sst_<tag>.json                   -- round-11 schema (hill re-score)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tensorfvm.verification.periodic_hill_round15 import run_channel, write_json

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def channel(args) -> int:
    t0 = time.time()
    payload = run_channel(
        args.closure, ny=args.ny, nx=args.nx, dt=args.dt,
        reynolds=args.reynolds, steps=args.steps, burn_in=args.burn_in,
        block_steps=args.block_steps, chunk=args.chunk,
        transport_convection_blend=args.transport_blend,
        production_limit=args.production_limit, verbose=True)
    tag = f"_{args.tag}" if args.tag else ""
    out = write_json(payload, OUTDIR / f"round15_channel_{args.closure}{tag}.json")
    final = payload["final"]
    print(json.dumps({
        "artefact": str(out),
        "gates": final["gates"],
        "all_gates_pass": final["all_gates_pass"],
        "nu_t_plus_over_log_law": final["nu_t_plus_over_log_law"],
        "re_tau": final["re_tau"],
        "re_tau_dean": final["re_tau_dean"],
        "kappa_eff": final["kappa_eff"],
        "max_nu_t_ratio": payload["blocks"][-1]["profile"]["max_nu_t_ratio"],
        "interpretation": final["interpretation"],
        "wall_seconds": time.time() - t0,
    }, indent=1))
    return 0 if final["all_gates_pass"] else 1


def hill(args) -> int:
    from tensorfvm.verification.periodic_hill_round11 import (
        gate_summary, run_time_averaged_case)

    t0 = time.time()
    payload = run_time_averaged_case(
        args.tag, args.closure, nx=args.nx, ny=args.ny, steps=args.steps,
        dt=args.dt, reynolds=args.reynolds, burn_in_steps=args.burn_in,
        production_limit=args.production_limit,
        transport_convection_blend=args.transport_blend,
        verbose=True, out_dir=OUTDIR, save_fields=True)
    gate = gate_summary(payload)
    print(json.dumps({
        "artefact": str(OUTDIR / f"round11_{args.closure}_{args.tag}.json"),
        "x_r_over_H": gate["x_r_over_H"],
        "reference": gate.get("reference"),
        "profile_pass": f"{gate['profile_stations_pass']}/{gate['profile_stations']}",
        "stationary": gate["stationary"],
        "pass": gate["pass"],
        "reasons": gate["reasons"],
        "wall_seconds": time.time() - t0,
    }, indent=1))
    return 0 if gate["pass"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="kind", required=True)

    ch = sub.add_parser("channel")
    ch.add_argument("--closure", default="sst", choices=("sa", "sst"))
    ch.add_argument("--ny", type=int, default=64)
    ch.add_argument("--nx", type=int, default=8)
    ch.add_argument("--dt", type=float, default=0.01)
    ch.add_argument("--reynolds", type=float, default=2800.0)
    ch.add_argument("--steps", type=int, default=30000)
    ch.add_argument("--burn-in", type=int, default=10000)
    ch.add_argument("--block-steps", type=int, default=5000)
    ch.add_argument("--chunk", type=int, default=250)
    ch.add_argument("--transport-blend", type=float, default=None)
    ch.add_argument("--production-limit", type=float, default=None)
    ch.add_argument("--tag", default=None)

    hi = sub.add_parser("hill")
    hi.add_argument("--closure", default="sst", choices=("sa", "sst"))
    hi.add_argument("--tag", required=True)
    hi.add_argument("--nx", type=int, default=96)
    hi.add_argument("--ny", type=int, default=48)
    hi.add_argument("--dt", type=float, default=0.02)
    hi.add_argument("--reynolds", type=float, default=2800.0)
    hi.add_argument("--steps", type=int, default=20000)
    hi.add_argument("--burn-in", type=int, default=8000)
    hi.add_argument("--transport-blend", type=float, default=None)
    hi.add_argument("--production-limit", type=float, default=None)

    args = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    return channel(args) if args.kind == "channel" else hill(args)


if __name__ == "__main__":
    sys.exit(main())