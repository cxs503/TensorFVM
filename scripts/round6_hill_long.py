"""Round-6 run driver for CASE-FVM-PERIODIC-HILLS (one configuration per process).

Usage (each line is an independent, shardable unit):

    PYTHONPATH=src python scripts/round6_hill_long.py hill sa-upwind
    PYTHONPATH=src python scripts/round6_hill_long.py hill sa-second
    PYTHONPATH=src python scripts/round6_hill_long.py hill laminar
    PYTHONPATH=src python scripts/round6_hill_long.py channel

Writes ``docs/case-periodic-hills/round6_<name>.json``.  The 48x24 SA runs cost
~23 ms/step (~12 min per 30000 steps), the channel ~1 h at ``ny = 64``; this is
why the driver exists as separate invocations rather than a single script.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from tensorfvm.verification import periodic_hill_round6 as r6

OUT = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"


def _log(record):
    print(f"  t={record['t']:7.1f}  x_r/H={record['x_r_over_H']:6.3f}  "
          f"x_s/H={record['x_s_over_H']:5.3f}  "
          f"max nu_t/nu={record['max_nu_t_over_nu']:7.2f}  "
          f"|u|max={record['max_abs_velocity']:.3f}", flush=True)


def hill(name, **kwargs):
    t0 = time.time()
    print(f"[round6] hill configuration {name}", flush=True)
    result = r6.run_blocked(config=name, progress=_log, **kwargs)
    result["wall_seconds"] = time.time() - t0
    out = OUT / f"round6_{name}.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"config": name, "final": result["final"],
                      "criteria": result["criteria"],
                      "nu_t_plateau": {k: result["nu_t_plateau"][k] for k in
                                       ("end", "equilibrated", "asymptote_relative_error")},
                      "x_r_plateau": {k: result["x_r_plateau"][k] for k in
                                      ("end", "equilibrated", "asymptote_relative_error")},
                      "profile_worst_error": result["profile_y_over_h_half"]["error"]["worst_error"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))
    return result


def channel(**kwargs):
    t0 = time.time()
    print("[round6] plane-channel convergence (round-5 C4 revisit)", flush=True)
    result = r6.channel_convergence(**kwargs)
    result["wall_seconds"] = time.time() - t0
    out = OUT / "round6_channel.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({"c4_verdict": result["c4_verdict"],
                      "last": result["blocks"][-1],
                      "y_plus_requirement": result["y_plus_requirement"],
                      "wall_seconds": result["wall_seconds"]}, indent=2))
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: round6_hill_long.py {hill <config>|channel}")
    kind = argv[0]
    OUT.mkdir(parents=True, exist_ok=True)
    if kind == "hill":
        name = argv[1] if len(argv) > 1 else "sa-upwind"
        total = int(argv[2]) if len(argv) > 2 else 30000
        hill(name, total_steps=total)
    elif kind == "channel":
        channel()
    else:
        raise SystemExit(f"unknown kind {kind!r}")


if __name__ == "__main__":
    main()
