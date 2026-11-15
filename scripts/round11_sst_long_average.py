#!/usr/bin/env python3
"""Round 11 driver: long *time-averaged* periodic-hill runs, closure-parametric.

Why this exists
---------------
Round 9 time-averaged the SA branch at 96x48 and settled it in the negative (no
mean reattachment; profile 70-112 % low; ``nu_t/nu = 65``).  Round 10 added the
SST closure and verified the *implementation*, but only on a 48x24 short
transient, and said so: the acceptance gate was still "a compute question, not a
model-class one".  This driver is that compute: the same loop, the same meshes,
the same budget and the same metrics as round 9, with the closure selectable, so
SA and SST are compared apples to apples on the time mean.

One configuration per process, so the meshes shard on the cluster.

Outputs (per case, into ``docs/case-periodic-hills/``):
    round11_<closure>_<tag>.json  -- config, series, stationarity, gate, verdict
    round11_<closure>_<tag>.npz   -- time-averaged u, v, tau_b, tau_t, mesh

Usage
-----
    PYTHONPATH=src python scripts/round11_sst_long_average.py --list
    PYTHONPATH=src python scripts/round11_sst_long_average.py --closure sst --case-index 0 --case-count 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorfvm.verification.periodic_hill_round11 import (
    closure_comparison,
    load_payload,
    run_time_averaged_case,
    strict_finite,
)

OUTDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"

#: (tag, nx, ny, steps, sample_every, burn_in_steps) -- exactly round 9's budget
#: for the same three meshes, so the SST artefact is comparable with
#: ``round9_<tag>.json`` row by row.
CASES = [
    ("48x24", 48, 24, 30000, 20, 10000),
    ("96x48", 96, 48, 20000, 20, 8000),
    ("144x72", 144, 72, 12000, 20, 6000),
]


def _finite(obj):
    """Backwards-compatible alias of :func:`strict_finite` (old scripts import it)."""
    return strict_finite(obj)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--case-index", type=int, default=0)
    ap.add_argument("--case-count", type=int, default=1)
    ap.add_argument("--closure", default="sst", choices=("sa", "sst", "laminar"))
    ap.add_argument("--tag")
    ap.add_argument("--nx", type=int)
    ap.add_argument("--ny", type=int)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--sample-every", type=int, default=20)
    ap.add_argument("--burn-in", type=int)
    ap.add_argument("--dt", type=float, default=0.02)
    ap.add_argument("--reynolds", type=float, default=2800.0)
    ap.add_argument("--production-limit", type=float, default=None)
    ap.add_argument("--convection-blend", type=float, default=None,
                    help="momentum convection blend (parent solver)")
    ap.add_argument("--transport-blend", type=float, default=None,
                    help="k/omega transport convection blend (round 15; "
                         "previously unreachable from this driver)")
    ap.add_argument("--freestream-intensity", type=float, default=None)
    ap.add_argument("--compare", action="store_true",
                    help="build round11_comparison.json from the artefacts on disk")
    args = ap.parse_args(argv)

    if args.list:
        for i, (tag, nx, ny, steps, sample_every, burn_in) in enumerate(CASES):
            print(f"{i}: {tag} {nx}x{ny} steps={steps} t={steps * args.dt:g} "
                  f"burn_in={burn_in * args.dt:g} sample_every={sample_every}")
        return 0

    if args.compare:
        payloads = {}
        for closure in ("sa", "sst"):
            for tag, *_ in CASES:
                path = OUTDIR / f"round11_{closure}_{tag}.json"
                if path.exists():
                    payloads[f"{closure}:{tag}"] = load_payload(path)
        for tag, *_ in CASES:  # round-9 SA artefacts are the SA baseline
            path = OUTDIR / f"round9_{tag}.json"
            if path.exists():
                payloads.setdefault(f"sa:round9:{tag}", load_payload(path))
        table = closure_comparison(payloads)
        out = OUTDIR / "round11_comparison.json"
        out.write_text(json.dumps(table, indent=1, allow_nan=False))
        print(json.dumps(table, indent=1, allow_nan=False))
        return 0

    if args.tag:
        tag, nx, ny, steps, sample_every, burn_in = (
            args.tag, args.nx, args.ny, args.steps, args.sample_every, args.burn_in)
        if None in (nx, ny, steps, burn_in):
            ap.error("--tag requires --nx --ny --steps --burn-in")
    else:
        if not 0 <= args.case_index < len(CASES):
            ap.error(f"--case-index must be in 0..{len(CASES) - 1}")
        tag, nx, ny, steps, sample_every, burn_in = CASES[args.case_index]

    payload = run_time_averaged_case(
        tag, args.closure, nx=nx, ny=ny, steps=steps, sample_every=sample_every,
        burn_in_steps=burn_in, dt=args.dt, reynolds=args.reynolds,
        production_limit=args.production_limit,
        convection_blend=args.convection_blend,
        transport_convection_blend=args.transport_blend,
        freestream_intensity=args.freestream_intensity,
        verbose=True, out_dir=OUTDIR, save_fields=True)

    gate = payload["gate"]
    print(f"[{args.closure}/{tag}] samples={payload['samples']} "
          f"wall={payload['wall_clock_s']:.0f}s "
          f"x_r/H={gate['x_r_over_H']} "
          f"profile {gate['profile_stations_pass']}/{gate['profile_stations']} "
          f"stationary={gate['stationary']} pass={gate['pass']}", flush=True)
    for reason in gate["reasons"]:
        print(f"  - {reason}", flush=True)
    return 0 if gate["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())