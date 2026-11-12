"""A/B experiment: ``pressure_form`` of the Hysing leg.

``TwoPhaseConfig.pressure_form`` documents ``"reduced"`` as the form to use
whenever ``gravity != 0`` and a free surface is present, and both the historic
Zalesak/dam-break legs and :func:`run_hysing_leg` leave it at the ``"total"``
default.  The Hysing leg is the only leg with both gravity *and* a closed
liquid domain, so the pressure split is a first-order suspect for the
``y_c(t=3)`` deficit reported in
``docs/GAP-FVM-TWOPHASE/HYSING-LEG-REPAIR.md`` (-23.3 % vs Hysing 2009).

This script runs the *same* resolution (``--cpd``) and end time with the two
pressure forms and dumps both the summary record and the per-step history
(``yc``, circularity, ``umax``, volume drift) so the difference is attributable
step by step.

Usage::

    PYTHONPATH=src python3 docs/GAP-FVM-TWOPHASE/ab_pressure_form.py \
        --cpd 8 --end-time 3 --out docs/GAP-FVM-TWOPHASE/artifacts
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

from tensorfvm.twophase_gap.hysing_gate import HYSING_REF_UT, HYSING_REF_YC, run_hysing_leg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpd", type=int, default=8)
    ap.add_argument("--end-time", type=float, default=3.0)
    ap.add_argument("--forms", nargs="+", default=["total", "reduced"])
    ap.add_argument("--out", default="docs/GAP-FVM-TWOPHASE/artifacts/ab_pressure_form")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    summary = {}
    for form in args.forms:
        t0 = time.time()
        res = run_hysing_leg(args.cpd, end_time=args.end_time, pressure_form=form)
        wall = time.time() - t0
        rec, hist = res["record"], res["history"]
        # keep the history light: it is only used for the attribution figure
        slim = [
            {k: h[k] for k in ("time", "yc", "xc", "circularity", "volume_drift", "umax")}
            for h in hist
        ]
        payload = {"record": rec, "history": slim, "wall_seconds": wall}
        (out / f"hysing_{form}_cpd{args.cpd}.json").write_text(
            json.dumps(payload, indent=2) + "\n")
        summary[form] = {
            "wall_seconds": wall,
            "steps": rec["steps"],
            "peak_velocity": rec["peak_velocity"],
            "terminal_velocity": rec["terminal_velocity"],
            "min_circularity": rec["min_circularity"],
            "yc_end": slim[-1]["yc"] if slim else None,
            "umax_end": slim[-1]["umax"] if slim else None,
            "total_volume_drift": rec["total_volume_drift"],
            "worst_step_drift": rec["worst_step_drift"],
            "diverged": rec["diverged"],
        }
        print(f"[{form}] wall={wall:.0f}s steps={rec['steps']} "
              f"peak_u={rec['peak_velocity']:.4f} yc_end={summary[form]['yc_end']:.4f} "
              f"umax_end={summary[form]['umax_end']:.4f}", flush=True)

    ref = {"peak_velocity": HYSING_REF_UT, "yc_end": HYSING_REF_YC}
    (out / f"ab_cpd{args.cpd}.json").write_text(
        json.dumps({"cpd": args.cpd, "end_time": args.end_time,
                    "reference": ref, "forms": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())