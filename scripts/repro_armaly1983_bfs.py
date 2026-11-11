#!/usr/bin/env python3
"""Reproduce Armaly et al. (1983) backward-facing step flow with TensorFVM.

Runs the reproduction matrix (Reynolds number x grid level x downstream length)
through :mod:`tensorfvm.inc_bench.armaly`, which drives the existing steady
SIMPLE kernel, and writes one JSON artifact per point plus an aggregate file.

    PYTHONPATH=src python scripts/repro_armaly1983_bfs.py \
        --re 100 389 --levels coarse medium --downstream 12 40 \
        --out docs/repro-armaly1983-bfs/artifacts --pressure-iterations 300

Every number written into the artifact is produced by the run itself; nothing is
copied from the paper except the two experimental anchors, which carry their
provenance and digitisation band in ``tensorfvm.inc_bench.armaly``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402

from tensorfvm.inc_bench.armaly import (  # noqa: E402
    ARMALY_1983_CITATION, ReproductionRun, experiment_point)
from tensorfvm.inc_bench.solver import SolverOptions  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", type=float, nargs="+", default=[100.0, 389.0],
                        help="Reynolds numbers based on D_h = 2 H_in (Armaly basis)")
    parser.add_argument("--levels", nargs="+", default=["coarse"],
                        help="grid levels of tensorfvm.inc_bench.cases.BFS_LEVELS")
    parser.add_argument("--downstream", type=float, nargs="+", default=[12.0],
                        help="downstream channel length in step heights")
    parser.add_argument("--out", default="docs/repro-armaly1983-bfs/artifacts")
    parser.add_argument("--pressure-iterations", type=int, default=300)
    parser.add_argument("--pressure-tolerance", type=float, default=1e-8)
    parser.add_argument("--max-iterations", type=int, default=4000)
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--high-order", dest="high_order", action="store_true",
                        default=True)
    parser.add_argument("--first-order", dest="high_order", action="store_false")
    parser.add_argument("--alpha-u", type=float, default=0.7)
    parser.add_argument("--alpha-p", type=float, default=0.3)
    parser.add_argument("--threads", type=int, default=0,
                        help="torch intra-op threads (0 = leave the default)")
    parser.add_argument("--tag", default="", help="suffix for artifact file names")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    options = SolverOptions(pressure_iterations=args.pressure_iterations,
                            pressure_tolerance=args.pressure_tolerance,
                            tolerance=args.tolerance, high_order=args.high_order,
                            alpha_u=args.alpha_u, alpha_p=args.alpha_p)
    aggregate: list[dict[str, object]] = []
    started = time.time()
    for reynolds in args.re:
        for level in args.levels:
            for downstream in args.downstream:
                print(f"[armaly] Re_Dh={reynolds:g} level={level} "
                      f"L_down={downstream:g} S ...", flush=True)
                run = ReproductionRun(reynolds, level, downstream, options=options,
                                     max_iterations=args.max_iterations)
                record = run.run()
                record["experiment"] = _experiment_block(reynolds)
                tag = args.tag or f"re{reynolds:g}_{level}_ld{downstream:g}"
                path = out / f"armaly1983_{tag}.json"
                path.write_text(json.dumps(record, indent=1, sort_keys=True))
                aggregate.append(record)
                print(f"[armaly]   -> L_r/S={record['reattachment_over_step']:.4f} "
                      f"(exp {record['experiment_over_step']}) "
                      f"converged={record['converged']} "
                      f"iters={record['iterations']} "
                      f"outlet_dev={record['profile_outlet_column']['max_rel']:.3f} "
                      f"{record['seconds']}s", flush=True)
    summary = {
        "external_anchor": ARMALY_1983_CITATION,
        "generated_by": "scripts/repro_armaly1983_bfs.py",
        "arguments": vars(args),
        "wall_seconds": round(time.time() - started, 1),
        "runs": aggregate,
    }
    suffix = f"_{args.tag}" if args.tag else ""
    (out / f"armaly1983_matrix{suffix}.json").write_text(
        json.dumps(summary, indent=1, sort_keys=True))
    _print_table(aggregate)
    return 0


def _experiment_block(reynolds: float) -> dict[str, object] | None:
    try:
        point = experiment_point(reynolds)
    except ValueError:
        return None
    return {"reynolds_dh": point.reynolds_dh,
            "reattachment_over_step": point.reattachment_over_step,
            "band_over_step": point.band_over_step,
            "note": point.note,
            "citation": ARMALY_1983_CITATION}


def _print_table(records: list[dict[str, object]]) -> None:
    header = (f"{'Re_Dh':>7} {'level':>7} {'L_down':>7} {'nx':>5} {'ny':>4} "
              f"{'L_r/S':>8} {'exp':>6} {'err%':>7} {'conv':>5} {'iters':>6} "
              f"{'outlet_dev%':>11} {'dev_pass':>8}")
    print(header)
    print("-" * len(header))
    for r in records:
        exp = r["experiment_over_step"]
        err = r["reattachment_rel_error"]
        print(f"{r['reynolds_dh']:7.0f} {r['level']:>7} "
              f"{r['downstream_step_heights']:7.0f} {r['nx']:5d} {r['ny']:4d} "
              f"{r['reattachment_over_step']:8.4f} "
              f"{'n/a' if exp is None else f'{exp:6.1f}'} "
              f"{'n/a' if err is None else f'{100 * err:6.2f}'} "
              f"{str(r['converged']):>5} {r['iterations']:6d} "
              f"{100 * r['profile_outlet_column']['max_rel']:11.2f} "
              f"{str(r['profile_outlet_pass_5pct']):>8}")


if __name__ == "__main__":
    raise SystemExit(main())