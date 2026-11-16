"""Regenerate ``docs/swe-dry-front-attribution/metrics.json`` (no cluster needed).

    PYTHONPATH=src python docs/swe-dry-front-attribution/run_audit.py --n 40

Every number in ``REPORT.md`` is read from this file; none is typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorfvm.verification import swe_dry_front as df

OUT = Path(__file__).with_name("metrics.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--periods", type=float, default=2.0)
    ap.add_argument("--ladder", type=int, nargs="*", default=[40])
    args = ap.parse_args()

    payload = {
        "island": df.island_invariants(),
        "attribution": df.attribution(n=args.n, periods=args.periods),
        "cures": df.cure_study(n=args.n, periods=args.periods),
        "ladder": df.resolution_ladder(tuple(args.ladder), periods=args.periods),
        "settings": {"n": args.n, "periods": args.periods,
                     "samples_mass_budget": 41},
    }
    OUT.write_text(json.dumps(payload, indent=2))
    att = payload["attribution"]
    print("island invariants hold:", payload["island"]["enclosed_invariants_hold"])
    print("spike %.3f m/s at t=%.4f cell=%s h/H=%.3e depth_contrast=%.1fx"
          % (att["spike"]["speed"], att["spike"]["t"], att["spike"]["cell"],
             att["spike"]["h_over_H"], att["spike"]["depth_contrast"]))
    for row in payload["cures"]:
        print("%-26s vmax=%8.3f steps=%4d L2=%6.2f%% delta_mean=%6.3f"
              % (row["variant"], row["vmax"], row["steps"], 100 * row["l2"],
                 row["delta_mean"]))
    for row in payload["ladder"]:
        print("n=%-4d delta_mean=%.4f L2=%.2f%% predicted=%.2f%% explained=%.0f%%"
              % (row["n"], row["delta_mean"], 100 * row["l2"],
                 100 * row["l2_predicted"], 100 * row["fraction_explained"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())