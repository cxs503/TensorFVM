#!/usr/bin/env python
"""Run the two-phase VOF verification campaign (TensorFVM).

Examples
--------
List the run table (index -> case/scheme/grid)::

    python scripts/run_vof_verification.py --list

Run a shard of the campaign (this is what the cluster job does)::

    python scripts/run_vof_verification.py --index 3 --count 9 \
        --out docs/vof-two-phase-verification/artifacts/runs

Merge the shards, add the hydrostatic sweep and write ``campaign.json``::

    python scripts/run_vof_verification.py --merge \
        --runs docs/vof-two-phase-verification/artifacts/runs \
        --out docs/vof-two-phase-verification/artifacts

``--quick`` replaces the 256^2 level with 96^2 so the whole surface can be run
in a few minutes for a smoke check (never used for the reported numbers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from tensorfvm.vof_verification import campaign  # noqa: E402

# The PLIC sweeps are many small tensor ops: torch threading saturates at a few
# threads, so keep it small and predictable (override with TFVM_THREADS).
torch.set_num_threads(int(os.environ.get("TFVM_THREADS", "4")))


def _parse_index_list(text: str) -> list[int]:
    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, hi = chunk.split("-")
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(chunk))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="print the run table")
    ap.add_argument("--index", default=None,
                    help="run index (or comma list / range, e.g. 0,3-5)")
    ap.add_argument("--shard", type=int, default=None,
                    help="shard index i (0-based) for strided execution")
    ap.add_argument("--shards", type=int, default=None,
                    help="number of shards n: run every n-th spec starting at i")
    ap.add_argument("--out", default="artifacts/runs",
                    help="output directory for run_*.json")
    ap.add_argument("--runs", default="artifacts/runs",
                    help="directory to read run_*.json from when merging")
    ap.add_argument("--merge", action="store_true",
                    help="merge shards into campaign.json")
    ap.add_argument("--hydrostatic", action="store_true",
                    help="also run the hydrostatic rest-state sweep")
    ap.add_argument("--hydrostatic-only", action="store_true",
                    help="run only the hydrostatic sweep and write "
                         "hydrostatic.json (no transport specs)")
    ap.add_argument("--skip-hydrostatic", action="store_true",
                    help="when merging, reuse a previously written hydrostatic "
                         "section instead of running it")
    ap.add_argument("--deformation-sweep", action="store_true",
                    help="run the deformation-window sweep (attribution of the "
                         "tearing-anchor error) and write "
                         "deformation_sensitivity.json")
    ap.add_argument("--quick", action="store_true",
                    help="shrink the finest grid (smoke test only)")
    args = ap.parse_args(argv)

    if args.quick:
        quick = {}
        for k, g in campaign.GRIDS.items():
            fine = (96, 48) if k == "double_vortex" else (96, 96)
            quick[k] = [g[0], g[1], fine]
        campaign.GRIDS = quick
        campaign.SPECS = campaign.build_specs()

    if args.list:
        for i, spec in enumerate(campaign.SPECS):
            print(f"{i:3d}  {spec['case']:14s} {spec['scheme']:6s} "
                  f"{spec['cells'][0]}x{spec['cells'][1]}")
        return 0

    if args.deformation_sweep:
        out_dir = args.out.rsplit("/runs", 1)[0]
        os.makedirs(out_dir, exist_ok=True)
        sweep = campaign.run_deformation_sweep()
        target = os.path.join(out_dir, "deformation_sensitivity.json")
        with open(target, "w") as fh:
            json.dump(sweep, fh, indent=1, sort_keys=True)
        print(f"wrote {target} ({len(sweep['rows'])} runs)")
        return 0

    if args.hydrostatic_only:
        out_dir = args.out.rsplit("/runs", 1)[0]
        os.makedirs(out_dir, exist_ok=True)
        hydro = campaign.run_hydrostatic_sweep()
        target = os.path.join(out_dir, "hydrostatic.json")
        with open(target, "w") as fh:
            json.dump(hydro, fh, indent=1, sort_keys=True)
        worst_wb = max(c["well_balanced"]["pressure_oscillation"]
                       for c in hydro["cases"])
        worst_ship = max(c["shipped"]["pressure_oscillation"]
                         for c in hydro["cases"])
        print(f"wrote {target}: worst oscillation shipped={worst_ship:.3e} "
              f"well_balanced={worst_wb:.3e}")
        return 0

    if args.index is not None:
        indices = _parse_index_list(args.index)
    elif args.shard is not None and args.shards:
        indices = list(range(args.shard, len(campaign.SPECS), args.shards))
    elif args.merge:
        indices = []          # merge only: never re-run the campaign
    else:
        indices = list(range(len(campaign.SPECS)))

    if indices:
        campaign.run_indices(indices, args.out)
    if not args.merge:
        return 0

    runs = campaign.load_runs(args.runs)
    hydrostatic = None
    hydrostatic_path = os.path.join(args.out.rsplit("/runs", 1)[0],
                                    "hydrostatic.json")
    if args.hydrostatic:
        hydrostatic = campaign.run_hydrostatic_sweep()
        with open(hydrostatic_path, "w") as fh:
            json.dump(hydrostatic, fh, indent=1, sort_keys=True)
    elif args.skip_hydrostatic and os.path.exists(hydrostatic_path):
        with open(hydrostatic_path) as fh:
            hydrostatic = json.load(fh)

    report = campaign.merge_runs(runs, hydrostatic)
    os.makedirs(args.out, exist_ok=True)
    target = os.path.join(args.out.rsplit("/runs", 1)[0], "campaign.json")
    with open(target, "w") as fh:
        json.dump(report, fh, indent=1, sort_keys=True)
    print(f"wrote {target} from {len(runs)} runs")
    for key, crit in report["criteria"].items():
        print(f"{key}: {crit['passed']} value={crit.get('value')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())