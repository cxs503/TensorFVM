#!/usr/bin/env python3
"""Driver for the independent incompressible FVM verification surface.

Usage::

    python scripts/run_inc_bench.py run --case backward_facing_step --level coarse
    python scripts/run_inc_bench.py summary
    python scripts/run_inc_bench.py figures

``run`` solves one case level and writes ``artifacts/inc_bench/<case>_<level>.json``.
``summary`` aggregates the per-level JSON into ``artifacts/inc_bench/summary.json``
with the machine-checkable criteria; ``figures`` renders the PNG set used by the
report.  Everything is regenerated from the JSON, so no number in the report is
hand-copied.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.inc_bench import cases  # noqa: E402
from tensorfvm.inc_bench.solver import SolverOptions  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "inc_bench"

DEFAULT_ITERATIONS = {
    ("backward_facing_step", "coarse"): 6000,
    ("backward_facing_step", "medium"): 8000,
    ("backward_facing_step", "fine"): 8000,
    ("sudden_expansion", "coarse"): 6000,
    ("sudden_expansion", "medium"): 8000,
    ("sudden_expansion", "fine"): 8000,
    ("natural_cavity", "coarse"): 8000,
    ("natural_cavity", "medium"): 12000,
    ("natural_cavity", "fine"): 12000,
    ("natural_cavity", "reference"): 15000,
}


def build(case: str, level: str, rayleigh: float | None) -> cases.CaseSetup:
    if case == "backward_facing_step":
        return cases.backward_step(level)
    if case == "sudden_expansion":
        return cases.sudden_expansion(level)
    if case == "natural_cavity":
        if rayleigh is None:
            raise SystemExit("natural_cavity needs --rayleigh")
        return cases.natural_cavity(rayleigh, level)
    raise SystemExit(f"unknown case {case!r}")


def run_one(case: str, level: str, rayleigh: float | None,
            iterations: int | None, tolerance: float,
            pressure_solver: str = "cg", pressure_iterations: int = 300,
            alpha_u: float | None = None, alpha_p: float | None = None) -> dict:
    setup = build(case, level, rayleigh)
    options = SolverOptions(max_iterations=iterations or DEFAULT_ITERATIONS.get(
        (case, level), 6000), tolerance=tolerance, check_every=200,
        min_iterations=300, pressure_solver=pressure_solver,
        pressure_iterations=int(pressure_iterations))
    if alpha_u is not None:
        options.alpha_u = float(alpha_u)
        options.alpha_u_warmup = min(options.alpha_u_warmup, float(alpha_u))
    if alpha_p is not None:
        options.alpha_p = float(alpha_p)
        options.alpha_p_warmup = min(options.alpha_p_warmup, float(alpha_p))
    solver = setup.solver(options)
    started = time.time()
    record = solver.solve()
    elapsed = time.time() - started
    metrics = cases.METRICS[case](solver, setup)
    payload = {
        "case": case, "level": level, "rayleigh": rayleigh,
        "setup": setup.summary(),
        "options": {"tolerance": tolerance, "high_order": options.high_order,
                    "alpha_u": options.alpha_u, "alpha_p": options.alpha_p,
                    "alpha_t": options.alpha_t,
                    "momentum_sweeps": options.momentum_sweeps,
                    "energy_sweeps": options.energy_sweeps,
                    "max_iterations": int(options.max_iterations)},
        "convergence": {"iterations": record["iterations"],
                        "converged": bool(record["converged"]),
                        "change": float(record["change"]),
                        "final": record["final"]},
        "wall_clock_seconds": elapsed,
        "metrics": metrics,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = f"{case}_{level}" + (f"_ra{rayleigh:g}" if rayleigh else "")
    path = OUT / f"{tag}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"wrote {path}  ({record['iterations']} it, {elapsed:.1f} s, "
          f"converged={record['converged']})")
    return payload


def load(tag: str) -> dict | None:
    path = OUT / f"{tag}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def richardson(values: list[float], refinement: float = 2.0) -> dict:
    """Two-term Richardson estimate from three monotone grid values."""
    usable = [v for v in values if isinstance(v, float) and math.isfinite(v)]
    if len(usable) < 3:
        return {"estimate": float("nan"), "order": float("nan"),
                "note": "three finite grid values required"}
    coarse, medium, fine = usable[-3], usable[-2], usable[-1]
    denominator = medium - coarse
    if abs(denominator) < 1e-12:
        return {"estimate": fine, "order": float("nan"),
                "note": "degenerate: coarse and medium agree"}
    ratio = (fine - medium) / denominator
    if not (0.0 < ratio < 1.0):
        return {"estimate": fine, "order": float("nan"),
                "note": f"non-monotone convergence (ratio {ratio:.3f})"}
    order = math.log(ratio) / math.log(1.0 / refinement)
    estimate = fine + (fine - medium) / (refinement ** order - 1.0)
    return {"estimate": estimate, "order": order,
            "note": "fine + (fine - medium) / (r^p - 1)"}


def summary() -> dict:
    levels = ("coarse", "medium", "fine")
    payload: dict = {"criteria": [], "cases": {}}
    bfs = {level: load(f"backward_facing_step_{level}") for level in levels}
    bfs = {k: v for k, v in bfs.items() if v}
    if bfs:
        metrics = {k: v["metrics"] for k, v in bfs.items()}
        verdict = cases.backward_step_verdict(metrics)
        extrapolation = richardson([metrics[level]["reattachment_over_step"]
                                    for level in levels if level in metrics])
        payload["cases"]["backward_facing_step"] = {
            "levels": {level: {
                "L_r_over_S": metrics[level]["reattachment_over_step"],
                "friction_min": metrics[level]["friction_min"],
                "friction_far": metrics[level]["friction_far"],
                "friction_far_analytic": metrics[level]["friction_far_analytic"],
                "friction_far_rel_error": metrics[level]["friction_far_rel_error"],
                "nx": bfs[level]["setup"]["nx"], "ny": bfs[level]["setup"]["ny"],
                "cells": bfs[level]["setup"]["cells"],
                "iterations": bfs[level]["convergence"]["iterations"],
                "converged": bfs[level]["convergence"]["converged"],
                "change": bfs[level]["convergence"]["change"],
                "expansion_ratio_mask": bfs[level]["setup"]["geometry"]["expansion_ratio_mask"],
                "expansion_ratio_rel_error":
                    bfs[level]["setup"]["geometry"]["expansion_ratio_rel_error"],
                "wall_clock_seconds": bfs[level]["wall_clock_seconds"],
            } for level in levels if level in metrics},
            "richardson": extrapolation,
            "verdict": verdict,
        }
        payload["criteria"].extend(verdict["criteria"])
    expansion = {level: load(f"sudden_expansion_{level}") for level in levels}
    expansion = {k: v for k, v in expansion.items() if v}
    if expansion:
        metrics = {k: v["metrics"] for k, v in expansion.items()}
        setups = {k: cases.sudden_expansion(k) for k in expansion}
        verdict = cases.expansion_verdict(metrics, setups)
        payload["cases"]["sudden_expansion"] = {
            "levels": {level: {
                "L_r_over_h": metrics[level]["reattachment_over_inlet_height"],
                "K": metrics[level]["loss_coefficient"],
                "K_borda_carnot": metrics[level]["loss_coefficient_borda_carnot"],
                "K_excess": metrics[level]["loss_excess_over_borda_carnot"],
                "mass_imbalance": metrics[level]["mass_imbalance"],
                "cells": expansion[level]["setup"]["cells"],
                "iterations": expansion[level]["convergence"]["iterations"],
                "converged": expansion[level]["convergence"]["converged"],
                "change": expansion[level]["convergence"]["change"],
                "wall_clock_seconds": expansion[level]["wall_clock_seconds"],
            } for level in levels if level in metrics},
            "richardson": richardson([metrics[level]["reattachment_over_inlet_height"]
                                      for level in levels if level in metrics]),
            "verdict": verdict,
        }
        payload["criteria"].extend(verdict["criteria"])
    cavity: dict[str, dict] = {}
    for rayleigh in (1e4, 1e5, 1e6):
        entry = None
        for level in ("medium", "fine", "coarse", "reference"):
            tag = f"natural_cavity_{level}_ra{rayleigh:g}"
            entry = load(tag)
            if entry:
                break
        if entry:
            cavity[f"{rayleigh:g}"] = entry
    if cavity:
        metrics = {key: value["metrics"] for key, value in cavity.items()}
        verdict = cases.cavity_verdict(metrics)
        payload["cases"]["natural_cavity"] = {
            "levels": {key: {
                "rayleigh": float(key),
                "nusselt": metrics[key]["nusselt_hot"],
                "nusselt_reference": metrics[key]["nusselt_reference"],
                "nusselt_rel_error": metrics[key]["nusselt_rel_error"],
                "psi_max": metrics[key]["psi_max"],
                "psi_max_reference": metrics[key]["psi_max_reference"],
                "psi_max_rel_error": metrics[key]["psi_max_rel_error"],
                "u_max": metrics[key]["u_max"],
                "u_max_rel_error": metrics[key]["u_max_rel_error"],
                "energy_balance_rel": metrics[key]["energy_balance_rel"],
                "cells": cavity[key]["setup"]["cells"],
                "iterations": cavity[key]["convergence"]["iterations"],
                "converged": cavity[key]["convergence"]["converged"],
                "change": cavity[key]["convergence"]["change"],
                "wall_clock_seconds": cavity[key]["wall_clock_seconds"],
            } for key in metrics},
            "verdict": verdict,
        }
        payload["criteria"].extend(verdict["criteria"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    failed = [item["id"] for item in payload["criteria"] if not item["passed"]]
    print(f"summary written: {len(payload['criteria'])} criteria, "
          f"{len(failed)} failed: {failed}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "summary"])
    parser.add_argument("--case", default="backward_facing_step")
    parser.add_argument("--level", default="coarse")
    parser.add_argument("--rayleigh", type=float, default=None)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--tolerance", type=float, default=1e-7)
    parser.add_argument("--pressure", default="cg", choices=["cg", "sor"])
    parser.add_argument("--pressure-iterations", type=int, default=300)
    parser.add_argument("--alpha-u", type=float, default=None)
    parser.add_argument("--alpha-p", type=float, default=None)
    args = parser.parse_args()
    if args.command == "run":
        run_one(args.case, args.level, args.rayleigh, args.iterations, args.tolerance,
                args.pressure, args.pressure_iterations, args.alpha_u, args.alpha_p)
    else:
        summary()


if __name__ == "__main__":
    main()