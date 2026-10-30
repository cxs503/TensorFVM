#!/usr/bin/env python3
"""Recompute the metrics of an already-converged run from its stored artefacts."""
import argparse, json, sys
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO))
from validation_tmr.postprocess import flat_plate_metrics, backstep_metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, choices=("flatplate", "backstep"))
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viscosity", type=float, required=True)
    args = parser.parse_args()
    reference = REPO / "validation_tmr/reference/tmr"
    with np.load(args.output / f"{args.label}-fields.npz") as handle:
        fields = {name: handle[name] for name in handle.files}
    with np.load(args.output / f"{args.label}-wall.npz") as handle:
        wall = {name: handle[name] for name in handle.files}
    path = args.output / f"{args.label}-metrics.json"
    payload = (json.loads(path.read_text()) if path.exists()
               else dict(state=json.loads(
                   (args.output / f"{args.label}-history.json").read_text())[-1])
               )
    kernel = flat_plate_metrics if args.case == "flatplate" else backstep_metrics
    payload["metrics"] = kernel(fields, wall, reference, args.viscosity)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print({k: v for k, v in payload["metrics"].items() if not isinstance(v, (list, dict))})

if __name__ == "__main__":
    main()
