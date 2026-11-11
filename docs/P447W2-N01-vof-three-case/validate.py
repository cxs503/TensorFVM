#!/usr/bin/env python
"""Shardable pre-flight probe for node P447W2-N01 (one configuration per shard).

    PYTHONPATH=src python docs/P447W2-N01-vof-three-case/validate.py --index 0 --count 4

Used before the full campaign to check that each leg runs on the cluster and
that the dam-break front is in the right *physical* place (the previous attempt
of this node failed the Martin & Moyce comparison by 60 %, so the physics is
verified before the resolution matrix is paid for).
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

torch.set_num_threads(max(1, int(os.environ.get("TP_THREADS", "8"))))

from tensorfvm.vof_threecase.dambreak import DamBreakMMConfig, run_dam_break
from tensorfvm.vof_threecase.statics import StaticBubbleConfig, run_static_bubble
from tensorfvm.vof_threecase.zalesak import NotchConfig, run_long_rotation

CONFIGS = [
    ("dam", DamBreakMMConfig(ny=50, nx=100, aspect=2.0, column_width=0.05715,
                             length=0.4, height=0.2)),
    ("dam", DamBreakMMConfig(ny=100, nx=200, aspect=2.0, column_width=0.05715,
                             length=0.4, height=0.2)),
    ("static", StaticBubbleConfig(cells=64, mu_liquid=1.0e-2)),
    ("zalesak", NotchConfig(cells=128, scheme="cicsam", degrees=360.0)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--count", type=int, default=len(CONFIGS))
    args = ap.parse_args()
    todo = [c for i, c in enumerate(CONFIGS) if i % args.count == args.index]
    for kind, cfg in todo:
        t0 = time.time()
        if kind == "dam":
            rec = run_dam_break(cfg)
            rec.pop("_alpha_final", None)
            rec.pop("_solver", None)
            print(json.dumps({"cfg": f"dam {cfg.ny}x{cfg.nx}",
                              "wall": time.time() - t0,
                              "samples": rec["samples"],
                              "mass_drift_max": rec["mass_drift_max"],
                              "steps": rec["steps"],
                              "max_abs_relative_error": rec["max_abs_relative_error"]},
                             indent=1), flush=True)
        elif kind == "static":
            rec = run_static_bubble(cfg)
            rec.pop("_solver", None)
            keep = {k: rec[k] for k in
                    ("cells", "spurious_ratio", "saturated", "jump_error_relative",
                     "volume_drift", "steps", "tau_mu", "dt", "plateau_ratio")}
            print(json.dumps({"cfg": f"static {cfg.cells} mu={cfg.mu_liquid}", "wall": time.time() - t0,
                              **keep}, indent=1), flush=True)
        else:
            rec = run_long_rotation(cfg)
            rec.pop("_field", None)
            rec.pop("_exact", None)
            keep = {k: rec[k] for k in
                    ("scheme", "cells", "degrees", "notch_error_relative",
                     "area_error_contour_vs_initial", "perimeter_error_analytic",
                     "mass_drift_relative", "peak_retention", "steps")}
            print(json.dumps({"cfg": f"zalesak {cfg.scheme} {cfg.cells} {cfg.degrees}",
                              "wall": time.time() - t0, **keep}, indent=1), flush=True)


if __name__ == "__main__":
    main()