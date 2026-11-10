#!/usr/bin/env python3
"""Run one dam-break variant of the P475W1-N06 chain on the cluster.

The variant list is fixed in ``VARIANTS`` below so a shard is selected by an
integer index: ``python scripts/vof_chain_dam.py --index 2 --out <dir>``.
Each shard writes ``dam_<index>.json`` (history + criteria) plus the final
liquid-fraction field ``dam_<index>__alpha.npy``; nothing is printed except the
one-line summary, so the job logs stay readable.

Why a separate script instead of the module CLI: the cluster harness passes a
command template only, and the campaign wants every variant of the same case at
the same wall clock so the comparison is not biased by truncation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.verification.two_phase_campaign import (  # noqa: E402
    DamBreakRun,
    evaluate_dam_break,
    run_dam_break_case,
)
from tensorfvm.verification.vof_chain import MASS_DRIFT_TOL  # noqa: E402

#: (label, DamBreakRun overrides).  The grid study keeps everything else at the
#: Martin & Moyce defaults documented in ``two_phase_campaign.DamBreakRun``.
VARIANTS: list[tuple[str, dict]] = [
    ("plic-100x30", dict(nx=100, ny=30, vof_scheme="plic")),
    ("plic-200x60", dict(nx=200, ny=60, vof_scheme="plic")),
    ("plic-400x120", dict(nx=400, ny=120, vof_scheme="plic")),
    ("upwind-100x30", dict(nx=100, ny=30, vof_scheme="upwind")),
    ("upwind-200x60", dict(nx=200, ny=60, vof_scheme="upwind")),
    ("plic-100x30-noadv", dict(nx=100, ny=30, vof_scheme="plic",
                              momentum_advection=False)),
    ("plic-100x30-reduced", dict(nx=100, ny=30, vof_scheme="plic",
                                 pressure_form="reduced")),
    ("plic-200x60-noadv", dict(nx=200, ny=60, vof_scheme="plic",
                               momentum_advection=False)),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--out", default="runs/vof-chain-anchor")
    ap.add_argument("--max-t-star", type=float, default=3.0)
    ap.add_argument("--wall", type=float, default=None)
    args = ap.parse_args()
    label, overrides = VARIANTS[args.index]
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = DamBreakRun(max_t_star=args.max_t_star, **overrides)
    run = run_dam_break_case(cfg, progress=True, max_wall_seconds=args.wall,
                             return_field=True)
    alpha = run.pop("final_alpha")
    run["evaluation"] = evaluate_dam_break(run)
    run["mass_gate"] = {
        "pass": abs(run["final_volume_drift"]) <= MASS_DRIFT_TOL,
        "criterion": f"|dV|/V0 <= {MASS_DRIFT_TOL:g}",
    }
    run["label"] = label
    run["variant_index"] = args.index
    (out / f"dam_{args.index:02d}.json").write_text(
        json.dumps(run, indent=2, sort_keys=True, default=str))
    np.save(out / f"dam_{args.index:02d}__alpha.npy", alpha.double().numpy())
    ev = run["evaluation"]
    print(f"[{label}] steps={run['steps']} t*={run['history'][-1]['t_star']:.2f} "
          f"worst={ev['worst_relative_error']} dV={run['final_volume_drift']:.3e} "
          f"elapsed={run['elapsed_seconds']:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())