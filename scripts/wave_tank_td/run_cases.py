#!/usr/bin/env python
"""Run the sigma-grid NWT acceptance cases and write evidence JSON.

Examples
--------
python scripts/wave_tank_td/run_cases.py geometry
python scripts/wave_tank_td/run_cases.py wave        --nx 200 --ns 24
python scripts/wave_tank_td/run_cases.py control     --nx 200 --ns 24
python scripts/wave_tank_td/run_cases.py propagation --nx 200 --ns 24
python scripts/wave_tank_td/run_cases.py rao         --period 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorfvm.wave_tank_td import cases  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case", choices=["geometry", "wave", "control", "propagation", "rao"])
    ap.add_argument("--amp", type=float, default=0.02)
    ap.add_argument("--period", type=float, default=1.0)
    ap.add_argument("--nx", type=int, default=200)
    ap.add_argument("--ns", type=int, default=24)
    ap.add_argument("--length", type=float, default=12.0)
    ap.add_argument("--depth", type=float, default=1.0)
    ap.add_argument("--periods", type=float, default=8.0)
    ap.add_argument("--spp", type=int, default=100, help="steps per period")
    ap.add_argument("--blend", type=float, default=0.15)
    ap.add_argument("--name", default=None)
    ap.add_argument("--body-length", type=float, default=0.6)
    ap.add_argument("--body-draft", type=float, default=0.2)
    ap.add_argument("--body-x", type=float, default=5.0)
    ap.add_argument("--added-mass-33", type=float, default=300.0)
    ap.add_argument("--added-mass-55", type=float, default=3.5)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)
    kw = dict(amp=a.amp, period=a.period, nx=a.nx, ns=a.ns, length=a.length,
              depth=a.depth, blend=a.blend, artifact_dir=a.outdir)
    if a.case == "geometry":
        out = cases.run_geometry_gate(a.outdir)
        print(json.dumps(out, indent=2, sort_keys=True)[:1200])
        return 0
    if a.case == "rao":
        out = cases.run_rao_case(name=a.name or "rao", periods=a.periods,
                                 steps_per_period=a.spp, body_length=a.body_length,
                                 body_draft=a.body_draft, body_x=a.body_x,
                                 added_mass=(a.added_mass_33, a.added_mass_55), **kw)
        print(json.dumps({k: v for k, v in out.items() if k != "series"},
                         indent=2, sort_keys=True))
        return 0
    if a.case == "wave":
        out = cases.run_wave_case(name=a.name or "wave", periods=a.periods,
                                  steps_per_period=a.spp, **kw)
    elif a.case == "control":
        out = cases.run_reflection_control(name=a.name or "control_no_beach",
                                           periods=a.periods, steps_per_period=a.spp,
                                           **kw)
    else:
        out = cases.run_propagation_case(name=a.name or "propagation",
                                         periods=a.periods, steps_per_period=a.spp,
                                         **kw)
    print(json.dumps({k: v for k, v in out.items() if k not in ("series", "series_x")},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())