"""Command-line driver: run one numerical-wave-tank case and write its JSON.

Usage::

    python scripts/wave_tank/run_cases.py geometry
    python scripts/wave_tank/run_cases.py wave   --amp 0.03 --period 1.0 --nx 200 --ny 60
    python scripts/wave_tank/run_cases.py control --amp 0.03 --period 1.0 --nx 200 --ny 60
    python scripts/wave_tank/run_cases.py body   --amp 0.03 --period 1.1 --nx 200 --ny 60

The JSON lands in ``outputs/wave-tank/<case>.json``; the report is assembled from
those files, never from hand-copied numbers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorfvm.wave_tank.cases import (run_body_case, run_geometry_gate,  # noqa: E402
                                       run_wave_case)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=["geometry", "wave", "control", "body"])
    ap.add_argument("--amp", type=float, default=0.03)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ny", type=int, default=60)
    ap.add_argument("--length", type=float, default=10.0)
    ap.add_argument("--height", type=float, default=1.4)
    ap.add_argument("--depth", type=float, default=1.0)
    ap.add_argument("--periods", type=float, default=6.0)
    ap.add_argument("--steps-per-period", type=int, default=120)
    ap.add_argument("--out", default="outputs/wave-tank")
    args = ap.parse_args(argv)

    common = dict(amp=args.amp, period=args.period, nx=args.nx, ny=args.ny,
                  length=args.length, height=args.height, depth=args.depth,
                  periods=args.periods, steps_per_period=args.steps_per_period,
                  artifact_dir=args.out)
    if args.case == "geometry":
        out = run_geometry_gate(args.out)
    elif args.case == "wave":
        out = run_wave_case(name="wave_damped", **common)
    elif args.case == "control":
        out = run_wave_case(name="wave_negative_control", damping=False, **common)
    else:
        out = run_body_case(name="body", **common)
    print(json.dumps(out, indent=2, sort_keys=True)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
