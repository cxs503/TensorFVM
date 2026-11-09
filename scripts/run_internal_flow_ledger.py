"""Internal-flow momentum-ledger campaign driver.

Usage::

    python scripts/run_internal_flow_ledger.py single --family expansion \
        --area-ratio 2 --level coarse --out /tmp/ledger
    python scripts/run_internal_flow_ledger.py all --out artifacts/internal_flow \
        --workers 8
    python scripts/run_internal_flow_ledger.py report --out artifacts/internal_flow \
        --docs docs/internal-flow-momentum-ledger

``single`` runs one case and writes ``<case>_<level>_<mode>.json``; ``all`` runs
the full case matrix in parallel processes (grid continuation inside a worker) and
then builds ``results.json``, the figures and the markdown report.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.internal_flow import cases, pipeline  # noqa: E402

LEVELS = ("coarse", "medium", "fine")


def _record_path(out: str, case) -> str:
    return os.path.join(out, f"{case.name}_{case.level}_{case.outlet_mode}.json")


def run_job(job: dict) -> list[str]:
    """Run a chained list of case specs in one process (warm start carries over)."""
    out = job["out"]
    written = []
    warm = None
    for spec in job["specs"]:
        fields: dict = {}
        build_kwargs = {k: v for k, v in spec.items()
                        if k not in ("tolerance", "max_iterations", "figure",
                                     "continue", "job")}
        case = cases.build(**build_kwargs)
        figure_dir = os.path.join(out, "figures") if spec.get("figure") else None
        record = pipeline.run_case(case, tolerance=spec.get("tolerance", 1e-6),
                                   max_iterations=spec.get("max_iterations", 60000),
                                   warm=warm, figure_dir=figure_dir,
                                   fields_out=fields)
        record["spec"] = spec
        path = _record_path(out, case)
        with open(path, "w") as handle:
            json.dump(record, handle, indent=1, default=float)
        written.append(path)
        # carry the converged field to the next case of the same job (Ra sweep)
        warm = fields if spec.get("continue") else None
        print(f"[done] {case.name} {case.level} {case.outlet_mode} "
              f"it={record['solver']['iterations']} "
              f"conv={record['solver']['converged']}", flush=True)
    return written


def case_matrix() -> list[dict]:
    specs: list[dict] = []
    # ---- expansion: three grids per area ratio, plus the outlet sensitivity set
    for ar in (2.0, 3.0):
        for level in LEVELS:
            specs.append({"family": "expansion", "area_ratio": ar, "level": level,
                          "outlet_mode": "dirichlet", "figure": level == "fine",
                          "job": f"expansion_ar{ar:g}"})
    for mode in ("extended", "traction"):
        specs.append({"family": "expansion", "area_ratio": 2.0, "level": "medium",
                      "outlet_mode": mode, "job": f"expansion_outlet_{mode}"})
    # ---- backward-facing step
    for re in (100.0, 400.0):
        for level in LEVELS:
            specs.append({"family": "step", "reynolds": re, "level": level,
                          "outlet_mode": "dirichlet", "figure": level == "fine",
                          "job": f"step_re{re:g}"})
    for mode in ("extended", "traction"):
        specs.append({"family": "step", "reynolds": 400.0, "level": "medium",
                      "outlet_mode": mode, "job": f"step_outlet_{mode}"})
    # ---- natural-convection cavity (Ra continuation inside one worker)
    for level in LEVELS:
        for i, ra in enumerate((1.0e4, 1.0e5, 1.0e6)):
            specs.append({"family": "cavity", "rayleigh": ra, "level": level,
                          "continue": True,
                          "figure": level == "fine" and ra == 1.0e6,
                          "job": f"cavity_{level}"})
    return specs


def jobs_from_matrix(out: str, *, include: str | None = None) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for spec in case_matrix():
        if include and include not in spec["job"]:
            continue
        grouped.setdefault(spec["job"], []).append(spec)
    jobs = []
    for name, specs in grouped.items():
        chained = []
        for spec in specs:
            spec = dict(spec)
            spec.pop("job", None)
            chained.append(spec)
        jobs.append({"name": name, "specs": chained, "out": out})
    return jobs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("single")
    one.add_argument("--family", required=True)
    one.add_argument("--level", default="coarse")
    one.add_argument("--area-ratio", type=float, default=2.0)
    one.add_argument("--reynolds", type=float, default=100.0)
    one.add_argument("--rayleigh", type=float, default=1.0e4)
    one.add_argument("--outlet-mode", default="dirichlet")
    one.add_argument("--tolerance", type=float, default=1e-6)
    one.add_argument("--max-iterations", type=int, default=60000)
    one.add_argument("--out", required=True)
    one.add_argument("--figure", action="store_true")
    many = sub.add_parser("all")
    many.add_argument("--out", required=True)
    many.add_argument("--workers", type=int, default=8)
    many.add_argument("--include", default=None)
    rep = sub.add_parser("report")
    rep.add_argument("--out", required=True)
    rep.add_argument("--docs", required=True)
    args = parser.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    if args.command == "single":
        spec = {"family": args.family, "level": args.level,
                "outlet_mode": args.outlet_mode}
        if args.family == "expansion":
            spec["area_ratio"] = args.area_ratio
        elif args.family == "step":
            spec["reynolds"] = args.reynolds
        else:
            spec["rayleigh"] = args.rayleigh
        spec["tolerance"] = args.tolerance
        spec["max_iterations"] = args.max_iterations
        spec["figure"] = args.figure
        run_job({"specs": [spec], "out": args.out})
        return 0

    if args.command == "all":
        jobs = jobs_from_matrix(args.out, include=args.include)
        print(f"{len(jobs)} jobs, {args.workers} workers", flush=True)
        with mp.get_context("spawn").Pool(args.workers) as pool:
            for written in pool.imap_unordered(run_job, jobs):
                print(f"[job] {len(written)} records", flush=True)
        return 0

    from tensorfvm.internal_flow import ledger_report
    ledger_report.build(args.out, docs_dir=args.docs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())