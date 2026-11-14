#!/usr/bin/env python
"""Resolution x momentum-scheme ladder for ``CASE-FVM-DAMBREAK-WET``.

Runs one (grid, momentum scheme) configuration, writes
``artifacts/leg_<key>.json`` with the full front history and the graded
checkpoints on the Martin & Moyce axes, and (optionally) the final fields.

    PYTHONPATH=src python docs/CASE-FVM-DAMBREAK-WET/run_ladder.py --list
    PYTHONPATH=src python docs/CASE-FVM-DAMBREAK-WET/run_ladder.py --index 0
    PYTHONPATH=src python docs/CASE-FVM-DAMBREAK-WET/run_ladder.py --index 0 --count 6   # shard i of 6

The configuration list and the acceptance thresholds are declared once below
(``LADDER`` / ``GATE``) so a shard and the merge step cannot disagree.
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

torch.set_num_threads(max(1, int(os.environ.get("TP_THREADS", "4"))))

from tensorfvm.vof_phase0 import anchors_mm
from tensorfvm.vof_threecase.dambreak import DamBreakMMConfig, run_dam_break

HERE = os.path.dirname(os.path.abspath(__file__))
ART = os.path.join(HERE, "artifacts")

COLUMN_WIDTH = 0.05715          # a, the M&M column width [m]
ASPECT = 2.0                    # h0/a of the digitised curve
GRAVITY = 9.81
T_END = 0.172                   # > T=2.96 support end, plus one sample of slack

#: (key, nx, ny, momentum scheme).  dy = height/ny, dx = length/nx, so
#: (100, 50) has 4 mm square cells in the 0.4 m x 0.2 m tank.
LADDER = (
    ("up_g50", 50, 25, "upwind"),
    ("l2_g50", 50, 25, "limited2"),
    ("up_g100", 100, 50, "upwind"),
    ("l2_g100", 100, 50, "limited2"),
    ("up_g200", 200, 100, "upwind"),
    ("l2_g200", 200, 100, "limited2"),
)

#: Acceptance of the anchor ``CASE-FVM-DAMBREAK-WET``: |X_sim - X_ref| / X_ref
#: <= 8 % at *every* predeclared checkpoint of the digitised M&M curve.
GATE = {
    "tolerance": 0.08,
    "checkpoints_T": list(anchors_mm.MM_CHECKPOINTS_T),
    "production_grid": "g100",
}


def config_for(nx: int, ny: int, scheme: str) -> DamBreakMMConfig:
    return DamBreakMMConfig(
        column_width=COLUMN_WIDTH,
        aspect=ASPECT,
        length=0.4,
        height=0.2,
        nx=nx,
        ny=ny,
        gravity=GRAVITY,
        momentum_scheme=scheme,
        convention="martin_moyce",
        t_end=T_END,
        max_steps=200000,
        track_every=2,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fields", action="store_true", help="also store the final alpha field")
    args = ap.parse_args()

    if args.list:
        for i, (key, nx, ny, scheme) in enumerate(LADDER):
            print(i, key, nx, ny, scheme)
        return 0

    idxs = range(len(LADDER)) if args.index is None else [
        i for i in range(len(LADDER)) if i % args.count == args.index % args.count
    ]
    os.makedirs(os.path.join(ART, "logs"), exist_ok=True)
    for i in idxs:
        key, nx, ny, scheme = LADDER[i]
        cfg = config_for(nx, ny, scheme)
        t0 = time.time()
        rec = run_dam_break(cfg)
        wall = time.time() - t0
        alpha = rec.pop("_alpha_final")
        rec.pop("_solver")
        rec.update(
            {
                "key": key,
                "scheme": scheme,
                "dx": cfg.dx,
                "dy": cfg.dy,
                "cells_across_column": cfg.column_width / cfg.dx,
                "wall_clock_s": wall,
                "reference": {
                    "source": "Martin & Moyce (1952) Fig. 3, h0/a=2 (14-point digitisation)",
                    "axes": "T = t sqrt(2 g / a), X = x_f / a, X(0)=1",
                    "module": "tensorfvm.vof_phase0.anchors_mm",
                },
                "gate": GATE,
            }
        )
        path = os.path.join(ART, f"leg_{key}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=1)
        if args.fields:
            os.makedirs(os.path.join(ART, "fields"), exist_ok=True)
            np.savez_compressed(
                os.path.join(ART, "fields", f"{key}.npz"), alpha=alpha.numpy())
        worst = max(s["mm"]["abs_relative_error"] for s in rec["samples"])
        print(
            f"[{key}] {nx}x{ny} {scheme} steps={rec['steps']} wall={wall:.1f}s "
            f"max|err|={100*worst:.2f}% drift={rec['mass_drift_max']:.2e}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())