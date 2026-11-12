"""Compare the auditor's clean-dir reproduction against the committed artifacts.

Emits runs/m16204/repro_compare.json so every number in the audit report is
machine-produced (no hand transcription).
"""

from __future__ import annotations

import json
from pathlib import Path

CANON = Path("/root/tp-agent/work/m16182/docs/case-blasius-plate/artifacts")
CLEAN = Path("/root/tp-agent/work/m16204/runs/m16204/repro-clean")
OUT = Path("/root/tp-agent/work/m16204/runs/m16204/repro_compare.json")


def walk(a, b, path, diffs):
    if isinstance(a, dict):
        for key in a:
            walk(a[key], (b or {}).get(key) if isinstance(b, dict) else None,
                 f"{path}/{key}", diffs)
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            diffs.append({"path": path, "canon": len(a) if isinstance(a, list) else a,
                          "clean": len(b) if isinstance(b, list) else b})
            return
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", diffs)
    else:
        if isinstance(a, float) and isinstance(b, float):
            if abs(a - b) > 1e-12 * max(1.0, abs(a)):
                diffs.append({"path": path, "canon": a, "clean": b})
        elif a != b:
            diffs.append({"path": path, "canon": a, "clean": b})


def main() -> int:
    report = {}
    for name in ("march.json", "march-sensitivity.json", "summary.json"):
        canon = json.loads((CANON / name).read_text())
        clean = json.loads((CLEAN / name).read_text())
        diffs: list = []
        walk(canon, clean, name, diffs)
        report[name] = {"n_diffs": len(diffs), "diffs": diffs[:10],
                        "bitwise_identical": not diffs}
    art = json.loads((CANON / "march.json").read_text())
    report["headline_numbers"] = {
        "cf": art["measured"]["cf"],
        "cf_relative_error": art["comparison"]["cf"]["relative_error"],
        "H": art["measured"]["H"],
        "H_relative_error": art["comparison"]["H"]["relative_error"],
        "ref_cf": art["comparison"]["reference"]["cf"],
        "ref_H": art["comparison"]["reference"]["H"],
        "error_limit": art["comparison"]["error_limit"],
        "passed": art["comparison"]["passed"],
    }
    sens = json.loads((CANON / "march-sensitivity.json").read_text())
    report["sensitivity"] = [
        {"tag": r["tag"], "ny": r["ny"], "stations": r["stations"],
         "cf_relative_error": r["cf_relative_error"],
         "H_relative_error": r["H_relative_error"], "passed": r["passed"]}
        for r in sens["rows"]]
    summaryonly = json.loads((CLEAN / "summary-summaryonly-only.json").read_text())
    report["summary_only_in_fresh_dir"] = {
        "rows": len(summaryonly["rows"]),
        "all_passed": summaryonly["all_passed"],
        "note": "CLI --summary-only skips execution and only merges existing "
                "*.json; in an empty --out directory it reports rows=[] and "
                "exit code 2, so the mission's quoted check line is only valid "
                "on a pre-populated out directory"}
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: (v if k != "march.json" else {"bitwise": v["bitwise_identical"]})
                      for k, v in report.items()}, indent=1)[:1600])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())