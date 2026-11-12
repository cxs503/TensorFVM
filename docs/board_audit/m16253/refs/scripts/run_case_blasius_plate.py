"""Run CASE-FVM-BLASIUS-PLATE and write machine-readable artifacts.

Usage (inside a repo checkout, artifacts land under ``--out``)::

    python scripts/run_case_blasius_plate.py --configs march
    python scripts/run_case_blasius_plate.py --configs fvm-baseline --max-iterations 800
    python scripts/run_case_blasius_plate.py --configs march --summary-only

Each configuration writes ``<tag>.json``; ``summary.json`` merges every ``*.json``
already present in the output directory, so cluster jobs can be dispatched one
configuration per job (and one node each) without losing the others.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

from tensorfvm.case_blasius_plate import (
    CASE_ID, ERROR_LIMIT, FVM_CONFIGS, LITERATURE, REFERENCE_RE_X,
    run_fvm_case, run_march_case, run_march_sensitivity, similarity_crosscheck,
)


def run_one(tag: str, max_iterations: int, threads: int, tolerance: float,
            trace_path=None, device: str = "cached"):
    if tag == "march":
        return run_march_case()
    if tag == "march-sensitivity":
        return run_march_sensitivity()
    if tag not in FVM_CONFIGS:
        raise SystemExit(f"unknown configuration {tag!r}; choose from "
                         f"{sorted(FVM_CONFIGS)} or 'march'")
    options = dict(FVM_CONFIGS[tag])
    return run_fvm_case(max_iterations=max_iterations, tolerance=tolerance,
                         threads=threads, trace_path=trace_path, device=device,
                         **options)


def summarise(directory: Path) -> dict:
    rows = []
    for path in sorted(directory.glob("*.json")):
        if path.name == "summary.json":
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if "comparison" not in data:
            continue
        comparison = data["comparison"]
        rows.append({
            "artifact": path.name,
            "device": data["device"],
            "cf": data["measured"]["cf"],
            "H": data["measured"]["H"],
            "re_x": data["measured"]["re_x"],
            "cf_relative_error": comparison["cf"]["relative_error"],
            "H_relative_error": comparison["H"]["relative_error"],
            "cf_passed": comparison["cf"]["passed"],
            "H_passed": comparison["H"]["passed"],
            "converged": data.get("run", {}).get("converged"),
            "iterations": data.get("run", {}).get("iterations"),
            "elapsed_s": data.get("run", {}).get("elapsed_s"),
        })
    summary = {
        "case": CASE_ID,
        "error_limit": ERROR_LIMIT,
        "reference_re_x": REFERENCE_RE_X,
        "literature": LITERATURE,
        "similarity_crosscheck": similarity_crosscheck(),
        "rows": rows,
        "all_passed": bool(rows) and all(r["cf_passed"] and r["H_passed"] for r in rows),
    }
    (directory / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("runs/blasius-plate"))
    parser.add_argument("--configs", default="march",
                        help="comma separated tags from %s plus 'march'" % sorted(FVM_CONFIGS))
    parser.add_argument("--max-iterations", type=int, default=2500)
    parser.add_argument("--device", default="cached",
                        choices=("cached", "simple"),
                        help="linear-algebra backend of the FVM device")
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    if not args.summary_only:
        for tag in [t.strip() for t in args.configs.split(",") if t.strip()]:
            started = time.time()
            result = run_one(tag, args.max_iterations, args.threads, args.tolerance,
                             trace_path=args.out / f"{tag}.trace.jsonl",
                             device=args.device)
            result["tag"] = tag
            path = args.out / f"{tag}.json"
            path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                            encoding="utf-8")
            comparison = result.get("comparison")
            if comparison is None:
                print(f"[{tag}] {time.time() - started:.1f}s  sensitivity rows: "
                      + ", ".join(f"{r['tag']} cf {100 * r['cf_relative_error']:+.2f}% "
                                  f"H {100 * r['H_relative_error']:+.2f}%"
                                  for r in result["rows"]), flush=True)
                continue
            print(f"[{tag}] {time.time() - started:.1f}s  "
                  f"cf={result['measured']['cf']:.6g} "
                  f"(err {100 * comparison['cf']['relative_error']:+.2f}%)  "
                  f"H={result['measured']['H']:.4f} "
                  f"(err {100 * comparison['H']['relative_error']:+.2f}%)  "
                  f"-> {'PASS' if comparison['passed'] else 'FAIL'}", flush=True)
    summary = summarise(args.out)
    for row in summary["rows"]:
        print(f"{row['artifact']:24s} cf err {100 * row['cf_relative_error']:+7.2f}%  "
              f"H err {100 * row['H_relative_error']:+7.2f}%  "
              f"{'PASS' if row['cf_passed'] and row['H_passed'] else 'FAIL'}")
    cross = summary["similarity_crosscheck"]
    print(f"literature audit: 2 f''(0) = {cross['two_fpp0']:.6f} vs "
          f"{LITERATURE['cf_sqrt_re_x']} ({100 * cross['cf_constant_relative_deviation']:.3f}%), "
          f"H_ode = {cross['H']:.4f} vs {LITERATURE['shape_factor']} "
          f"({100 * cross['shape_factor_relative_deviation']:.3f}%)")
    finite = all(math.isfinite(row["cf"]) and math.isfinite(row["H"])
                 for row in summary["rows"])
    return 0 if (summary["all_passed"] and finite) else 2


if __name__ == "__main__":
    raise SystemExit(main())