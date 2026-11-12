#!/usr/bin/env python
"""Resolution study + field dump for the Mach-3 forward-facing step.

    python scripts/forward_step_study.py --out docs/repro-fvm-forward-step/run \
        --grids 120x40,240x80,480x160 --t-end 4

Woodward & Colella (1984) *J. Comput. Phys.* **54** 115-173, problem 2: uniform
Mach 3 flow (gamma = 1.4, rho = 1.4, p = 1) in a tunnel of length 3 / height 1
with a reflecting step of height 0.2 starting at x = 0.6, impulsively started
from the uniform state and reported at t = 4.  The paper publishes density
contour plots for this problem and no tables, so no number here is transcribed
from it: every reference value is evaluated in code from the exact
Rankine-Hugoniot / isentropic relations for the stated inflow state (see
:func:`tensorfvm.compressible.forward_step.normal_shock_reference`).

This script only *runs* the case and dumps the raw fields, so the graded
measurements can be recomputed from the saved ``.npz`` without repeating the
integration (``--measure-only``).  Grading itself lives in
:mod:`tensorfvm.compressible.forward_step_benchmark`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from tensorfvm.compressible.forward_step import (  # noqa: E402
    T_END,
    ForwardStepConfig,
    run_case,
)

FIELDS = ("rho", "u", "v", "p")


def parse_grids(text: str) -> list[tuple[int, int]]:
    out = []
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        nx, ny = item.lower().split("x")
        out.append((int(nx), int(ny)))
    if not out:
        raise SystemExit("--grids is empty")
    return out


def run_grid(nx: int, ny: int, t_end: float, outdir: Path,
             threads: int | None) -> dict:
    if threads:
        torch.set_num_threads(threads)
    cfg = ForwardStepConfig(nx=nx, ny=ny, t_end=t_end)
    t0 = time.time()
    result = run_case(cfg)
    wall = time.time() - t0
    prim = result["prim"]
    xc, yc = cfg.centers()
    tag = f"{nx}x{ny}"
    npz = outdir / f"fields_{tag}.npz"
    import numpy as np

    np.savez_compressed(
        npz,
        xc=xc.numpy(), yc=yc.numpy(),
        solid=cfg.solid_mask().numpy().astype(np.int8),
        **{k: prim[k].numpy() for k in FIELDS},
    )
    row = {
        "grid": {"nx": nx, "ny": ny, "dx": cfg.dx, "dy": cfg.dy,
                 "cells": nx * ny, "cells_over_step": cfg.j_step},
        "config": cfg.as_dict(),
        "wall_seconds": wall,
        "nsteps": result["nsteps"],
        "time": result["time"],
        "positivity_fixes": result["positivity_fixes"],
        "budget": result["budget"],
        "fields": npz.name,
        "torch_threads": torch.get_num_threads(),
    }
    (outdir / f"run_{tag}.json").write_text(json.dumps(row, indent=1, sort_keys=True))
    print(f"[forward-step] {tag}: t={result['time']:.4f} steps={result['nsteps']} "
          f"{wall:.1f}s -> {npz}", flush=True)
    return row


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "runs" / "forward_step"))
    ap.add_argument("--grids", default="120x40,240x80,480x160")
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--threads", type=int, default=None)
    args = ap.parse_args(argv)
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for nx, ny in parse_grids(args.grids):
        rows.append(run_grid(nx, ny, args.t_end, outdir, args.threads))
    manifest = {
        "case": "CASE-FORWARD-STEP",
        "reference": "Woodward & Colella, J. Comput. Phys. 54 (1984) 115-173, problem 2",
        "t_end": args.t_end,
        "grids": [r["grid"] for r in rows],
        "runs": rows,
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps([{ "grid": r["grid"]["nx"] * r["grid"]["ny"],
                        "seconds": round(r["wall_seconds"], 1)} for r in rows]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())