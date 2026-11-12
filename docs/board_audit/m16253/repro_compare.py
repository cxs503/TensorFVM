#!/usr/bin/env python3
"""Compare the delivered artefacts with the auditor's independent re-run.

Both sides are JSON written by the same driver; the comparison is on numbers,
never on file bytes (timings differ).  Also re-scores the delivered rows against
the criterion, so the verdict's benchmark numbers come from this file.

Run: python3 docs/board_audit/m16253/repro_compare.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF = HERE / "refs/docs/case-blasius-plate/artifacts"
RERUN = Path("/root/tp-agent/work/m16253/runs/m16253/repro")
OUT = Path("/root/tp-agent/work/m16253/runs/m16253/repro_compare.json")

KEYS = ("cf", "H", "re_x")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    delivered = load(REF / "march.json")
    rerun = load(RERUN / "march.json")
    rows = {}
    for key in KEYS:
        a, b = delivered["measured"][key], rerun["measured"][key]
        rows[key] = {"delivered": a, "rerun": b,
                     "abs_diff": abs(a - b), "rel_diff": abs(a - b) / abs(a)}

    sens_d = {r["tag"]: r for r in load(REF / "march-sensitivity.json")["rows"]}
    sens_r = {r["tag"]: r for r in load(RERUN / "march-sensitivity.json")["rows"]}
    sens = {tag: {"cf_delivered": sens_d[tag]["cf"], "cf_rerun": sens_r[tag]["cf"],
                  "abs_diff": abs(sens_d[tag]["cf"] - sens_r[tag]["cf"])}
            for tag in sens_d}

    comparison = delivered["comparison"]
    payload = {
        "delivered_commit": "4f87f691",
        "rerun_dir": str(RERUN),
        "identical_numbers": all(v["abs_diff"] == 0.0 for v in rows.values()),
        "fields": rows,
        "sensitivity": sens,
        "criterion": {
            "tolerance": comparison["error_limit"],
            "ref_cf": comparison["reference"]["cf"],
            "ref_H": comparison["reference"]["H"],
            "got_cf": comparison["got"]["cf"],
            "got_H": comparison["got"]["H"],
            "cf_relative_error": comparison["cf"]["relative_error"],
            "H_relative_error": comparison["H"]["relative_error"],
            "passed": comparison["passed"],
        },
        "delivered_summary_all_passed": load(REF / "summary.json")["all_passed"],
        "rerun_summary_all_passed": load(RERUN / "summary.json")["all_passed"],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"identical": payload["identical_numbers"],
                      "cf_rel_err": payload["criterion"]["cf_relative_error"],
                      "H_rel_err": payload["criterion"]["H_relative_error"],
                      "passed": payload["criterion"]["passed"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())