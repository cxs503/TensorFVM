"""Run CASE-FVM-THACKER-BASIN: 2-D well-balanced shallow water vs Thacker (1981).

    PYTHONPATH=src python scripts/run_thacker_basin.py \
        --out docs/case-fvm-thacker-basin/artifacts        # ≈ 1 min, no cluster

Writes ``metrics.json``, ``lake_at_rest.json``, ``negative_control.json``,
``exactness.json``, ``convergence.json``, ``profiles.json``, ``SUMMARY.md`` and
``fig_thacker_basin.png``.  Every number in the report is read from these
artefacts, none is typed by hand.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tensorfvm.verification import thacker_basin as tb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/case-fvm-thacker-basin/artifacts")
    ap.add_argument("--resolution", type=int, default=160)
    ap.add_argument("--resolutions", type=int, nargs="*", default=[40, 80, 160])
    ap.add_argument("--periods", type=float, default=tb.DEFAULTS["periods"],
                    help="integration window in oscillation periods")
    ap.add_argument("--skip-convergence", action="store_true")
    args = ap.parse_args()

    payload = {}
    payload["exactness"] = tb.exactness_check()
    print("C2 closed-form residual = %.3g"
          % payload["exactness"]["max_relative_residual"])

    payload["lake_at_rest"] = tb.lake_at_rest_check(n=80)
    print("C1 lake at rest max|u| = %.3g  drift = %.3g"
          % (payload["lake_at_rest"]["max_velocity"],
             payload["lake_at_rest"]["surface_drift"]))

    payload["negative_control"] = tb.negative_control(n=80)
    print("C8 naive source control max|u| = %.3g (must be large)"
          % payload["negative_control"]["max_velocity"])

    if args.skip_convergence:
        payload["convergence"] = {"rows": [], "orders_centre": [],
                                  "orders_shore": [], "observed_order": 0.0,
                                  "monotone": False, "pass": False}
    else:
        payload["convergence"] = tb.convergence_study(args.resolutions,
                                                      periods=args.periods)
        print("C7 orders =", payload["convergence"]["orders_centre"])

    run = tb.run_case(n=args.resolution, periods=args.periods)
    payload["metrics"] = run
    print("C4 centre L2 = %.4f %%   C5 shoreline mean = %.4f %%   C6 sym = %.4f %%"
          % (100.0 * run["centre_depth"]["l2_relative"],
             100.0 * run["shoreline"]["mean_relative"],
             100.0 * run["symmetry"]["normalised"]))

    basin = tb.build_basin()
    payload["profiles"] = tb.profile_at(basin, args.resolution)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fig_path = out / "fig_thacker_basin.png"
    payload["figure"] = tb.figure_thacker(run, payload["profiles"],
                                          payload["convergence"],
                                          payload["lake_at_rest"], fig_path)
    written = tb.write_artifacts(out, payload)
    print(json.dumps(written, indent=2))
    print((out / "SUMMARY.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
