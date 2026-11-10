#!/usr/bin/env python
"""Shardable sweep of the seakeeping evidence cases.

The case list is data, not logic, so a shard only has to know ``{i}/{n}``::

    python scripts/wave_tank_td/sweep_seakeeping.py --shard 1 --shards 6 \
        --outdir docs/wave-tank-td-seakeeping/artifacts

Every shard writes the JSON files it produced and, so that a lost artifact
directory cannot lose the run, also prints each payload gzip+base85 encoded
between markers on stdout (the job log is the fallback transport).
"""
from __future__ import annotations

import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_seakeeping import GRIDS, run_one  # noqa: E402

from tensorfvm.wave_tank_td import cases as wt_cases  # noqa: E402

# (kind, period, grid, damping scale, remove_beach)
CASES = [
    ("seakeeping", 0.70, "coarse", 1.0, False),
    ("seakeeping", 0.80, "coarse", 1.0, False),
    ("seakeeping", 0.90, "coarse", 1.0, False),
    ("seakeeping", 1.00, "coarse", 1.0, False),
    ("seakeeping", 1.10, "coarse", 1.0, False),
    ("seakeeping", 1.20, "coarse", 1.0, False),
    ("seakeeping", 1.40, "coarse", 1.0, False),
    ("seakeeping", 0.90, "fine", 1.0, False),
    ("seakeeping", 1.10, "fine", 1.0, False),
    ("seakeeping", 1.10, "coarse", 0.7, False),
    ("seakeeping", 1.10, "coarse", 1.3, False),
    ("seakeeping", 1.20, "coarse", 1.0, True),
    ("propagation", 0.90, "coarse", 1.0, False),
    ("propagation", 1.10, "coarse", 1.0, False),
]


def emit(payload: dict, outdir: Path | None) -> None:
    text = json.dumps(payload, indent=2, sort_keys=True, default=float)
    if outdir is not None:
        outdir.mkdir(parents=True, exist_ok=True)
        (outdir / f"{payload['case']}.json").write_text(text)
    blob = base64.b85encode(gzip.compress(text.encode())).decode()
    print(f"===ARTIFACT {payload['case']} {blob}===")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, default=1, help="1-based shard index")
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--outdir", default="artifacts")
    a = ap.parse_args(argv)
    if not 1 <= a.shard <= a.shards:
        raise SystemExit("shard must be in [1, shards]")
    outdir = Path(a.outdir)
    mine = [c for k, c in enumerate(CASES) if k % a.shards == a.shard - 1]
    print(f"shard {a.shard}/{a.shards}: {len(mine)} case(s)")
    for kind, period, grid, scale, no_beach in mine:
        if kind == "propagation":
            nx, ns, spp = GRIDS[grid]
            payload = wt_cases.run_propagation_case(
                period=period, nx=nx, ns=ns, length=14.0, periods=6.0,
                steps_per_period=spp,
                name=f"propagation_{grid}_T{period:.2f}_{scale:.2f}",
                artifact_dir=None)
        else:
            payload = run_one(period, grid, amp=0.02, periods=12.0, damp_scale=scale,
                              outdir=None,
                              label=("control_no_beach" if no_beach else
                                     f"seakeep_{grid}_T{period:.2f}_d{scale:.2f}"),
                              damp_waves=0.0 if no_beach else 1.5)
        emit(payload, outdir)
        print(f"done {payload['case']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())