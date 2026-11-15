#!/usr/bin/env python3
"""Round 18 probe: per-step wall clock of the SST periodic-hill solver per mesh.

The round-17 next-step list asks for a *grid-refinement ladder*
(96x48 -> 144x72 -> 192x96) so that the +42 % reattachment error can be
attributed either to discretisation or to the 2-D/turbulence-model dimension.
Job budgets are capped, so the affordable number of time steps at each level has
to be measured on the node that will run the ladder instead of extrapolated from
a laptop: this probe times ``--steps`` solver steps at every requested level and
prints the numbers the submission was sized from.

    PYTHONPATH=src python3 scripts/round18_probe.py --steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.verification.periodic_hill_round11 import build_solver  # noqa: E402

LEVELS = ((96, 48), (144, 72), (192, 96))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--reynolds", type=float, default=2800.0)
    args = ap.parse_args(argv)

    rows = []
    for nx, ny in LEVELS:
        solver = build_solver("sst", nx=nx, ny=ny, reynolds=args.reynolds,
                              dt=args.dt, convection_blend=0.75,
                              transport_convection_blend=0.0)
        t0 = time.time()
        solver.advance(steps=args.steps)
        per_step = (time.time() - t0) / max(args.steps, 1)
        rows.append({"nx": nx, "ny": ny, "cells": nx * ny,
                     "per_step_s": round(per_step, 5),
                     "steps_per_7200s": int(7200.0 / per_step)})
        print(f"{nx}x{ny}: {per_step:.4f} s/step -> "
              f"{int(7200.0 / per_step)} steps in a 2 h job", flush=True)
    print("PROBE_JSON " + json.dumps(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())