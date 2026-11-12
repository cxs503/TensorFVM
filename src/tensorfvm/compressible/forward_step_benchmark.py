"""Benchmark grading for the Mach-3 forward-facing step (CASE-FORWARD-STEP).

Woodward & Colella, *J. Comput. Phys.* **54** (1984) 115-173, problem 2
("a Mach 3 wind tunnel with a step"): tunnel ``3 x 1``, reflecting step of
height ``0.2`` starting at ``x = 0.6``, uniform inflow ``M = 3``,
``gamma = 1.4``, ``rho = 1.4``, ``p = 1``, impulsively started from the uniform
state and reported at ``t = 4``.  The paper publishes density *contours* for
this problem and no table, so no reference number is transcribed from it: the
graded reference values are computed in code from the exact Rankine-Hugoniot
and isentropic relations for the paper's stated inflow state
(:func:`tensorfvm.compressible.forward_step.normal_shock_reference`).

This module is the *grading* half of the benchmark: it reads the field dumps
written by ``scripts/forward_step_study.py`` (or the ones inside
``docs/repro-fvm-forward-step/run``), re-measures the graded quantities with
:mod:`tensorfvm.compressible.forward_step` so a grader can recompute every
number without repeating the integration, applies the pre-registered
tolerances, and reports the grid-convergence study.  It is also the case's
entry point into the package CLI::

    python -m tensorfvm.compressible --legs forward_step --output docs/repro-fvm-forward-step/run

and can be run directly::

    python -m tensorfvm.compressible.forward_step_benchmark --runs docs/repro-fvm-forward-step/run

Reference page note: the case definition is the paper's problem 2 (pp. 161-166,
figures 6-7 for the t = 4 density contours).  Because those are contours, the
8 % tolerance is applied to the closed-form shock/stagnation quantities, as
pre-registered in ``docs/repro-fvm-forward-step/DESIGN.md``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .forward_step import (
    ForwardStepConfig,
    measure_shock_jump,
    normal_shock_reference,
    stagnation_state,
)

__all__ = [
    "CASE",
    "TOL",
    "GRADED_KEYS",
    "load_run",
    "grade_run",
    "grade_dir",
    "convergence",
    "table_markdown",
    "write_grading",
    "main",
]

#: mission tolerance on every graded number (pre-registered, docs/.../DESIGN.md §4)
TOL = 0.08

#: grid of record, pre-registered before the production run
GRID_OF_RECORD = (480, 160)

#: benchmark registry entry (kept in code so the platform can read it from the repo)
CASE = {
    "anchor_slug": "CASE-FORWARD-STEP",
    "domain": "TensorFVM",
    "title": "Mach 3 wind tunnel with a forward-facing step (Woodward & Colella problem 2)",
    "reference": ("Woodward, P. & Colella, P., 'The numerical simulation of two-dimensional "
                  "fluid flow with strong shocks', J. Comput. Phys. 54 (1984) 115-173, "
                  "problem 2 (pp. 161-166, figs. 6-7: t = 4 density contours; no tables)"),
    "reference_kind": ("exact Rankine-Hugoniot / isentropic relations for the paper's stated "
                       "inflow state, evaluated in code (the paper publishes contours only)"),
    "geometry": {"length": 3.0, "height": 1.0, "step_x": 0.6, "step_h": 0.2},
    "state": {"mach": 3.0, "gamma": 1.4, "rho": 1.4, "p": 1.0, "t_end": 4.0},
    "tolerance": TOL,
    "grid_of_record": list(GRID_OF_RECORD),
    "solver": "tensorfvm.compressible.solver.EulerSolver (MUSCL-Hancock + HLLC, CFL 0.4)",
    "repro_cmd": ("python scripts/repro_fvm_forward_step.py --out docs/repro-fvm-forward-step/run "
                  "--grids 120x40,240x80,480x160 --t-end 4"),
}

#: keys graded against the closed-form reference, mapped to their reference value
GRADED_KEYS = (
    ("p2/p1", "p_ratio", "shock", "p_ratio"),
    ("rho2/rho1", "rho_ratio", "shock", "rho_ratio"),
    ("u2/u1", "u_ratio", "shock", "u_ratio"),
    ("M2", "mach2", "shock", "mach2"),
    ("p_stag/p1", "p_stag_ratio", "stagnation", "p_stag_ratio"),
    ("rho_stag/rho1", "rho_stag_ratio", "stagnation", "rho_stag_ratio"),
)

BUDGET_KEYS = (
    ("mass_budget_drift", "mass_drift", 1e-6),
    ("energy_budget_drift", "energy_drift", 1e-2),
)


def _cfg_from_dict(d: Dict) -> ForwardStepConfig:
    keys = ("nx", "ny", "lx", "ly", "step_x", "step_h", "mach", "gamma",
            "rho", "p", "t_end", "cfl", "order", "limiter", "project_wall_flux")
    return ForwardStepConfig(**{k: d[k] for k in keys if k in d})


def load_run(run_dir: Path, tag: Optional[str] = None) -> Tuple[Dict[str, torch.Tensor], ForwardStepConfig, Dict]:
    """Load one saved run: ``fields_<nx>x<ny>.npz`` + ``run_<nx>x<ny>.json``."""
    run_dir = Path(run_dir)
    if tag is None:
        grid = [p.stem.split("_", 1)[1] for p in sorted(run_dir.glob("fields_*.npz"))]
        if not grid:
            raise FileNotFoundError(f"no fields_*.npz under {run_dir}")
        grid.sort(key=lambda g: int(g.split("x")[0]) * int(g.split("x")[1]))
        tag = grid[-1]
    fields_path = run_dir / f"fields_{tag}.npz"
    meta_path = run_dir / f"run_{tag}.json"
    if not fields_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"missing {fields_path.name} or {meta_path.name} under {run_dir}")
    meta = json.loads(meta_path.read_text())
    cfg = _cfg_from_dict(meta["config"])
    data = torch.load(fields_path, weights_only=False) if fields_path.suffix == ".pt" else None
    if data is None:
        import numpy as np

        with np.load(fields_path) as npz:
            data = {k: torch.as_tensor(npz[k]) for k in npz.files}
    prim = {k: data[k].double() for k in ("rho", "u", "v", "p")}
    return prim, cfg, meta


def _reference(cfg: ForwardStepConfig) -> Dict[str, float]:
    ref = normal_shock_reference(cfg.mach, cfg.gamma, cfg.rho, cfg.p)
    ref = dict(ref)
    ref["p_stag_ratio"] = ref["p02"] / cfg.p
    ref["rho_stag_ratio"] = ref["rho02"] / cfg.rho
    return ref


def grade_run(prim: Dict[str, torch.Tensor], cfg: ForwardStepConfig,
              meta: Dict, tolerance: float = TOL) -> Dict:
    """Grade one run: the six closed-form criteria plus the two budget identities."""
    ref = _reference(cfg)
    jump = measure_shock_jump(prim, cfg)
    stag = stagnation_state(prim, cfg)
    got = {"p_ratio": jump["p_ratio"], "rho_ratio": jump["rho_ratio"],
           "u_ratio": jump["u_ratio"], "mach2": jump["mach2"],
           "p_stag_ratio": stag["p_ratio"], "rho_stag_ratio": stag["rho_ratio"]}
    items = []
    for label, got_key, _group, ref_key in GRADED_KEYS:
        g, r = got[got_key], ref[ref_key]
        err = g / r - 1.0
        items.append({
            "quantity": label, "reference": r, "got": g, "rel_error": err,
            "tolerance": tolerance, "pass": bool(abs(err) <= tolerance),
        })
    budget = meta.get("budget", {})
    for label, key, tol in BUDGET_KEYS:
        value = float(budget.get(key, float("nan")))
        items.append({
            "quantity": label, "reference": 0.0, "got": value,
            "rel_error": value, "tolerance": tol,
            "pass": bool(abs(value) <= tol),
        })
    graded = [i for i in items if i["tolerance"] == tolerance]
    return {
        "grid": [cfg.nx, cfg.ny],
        "dx": cfg.dx,
        "t_end": cfg.t_end,
        "nsteps": meta.get("nsteps"),
        "wall_seconds": meta.get("wall_seconds"),
        "positivity_fixes": meta.get("positivity_fixes"),
        "items": items,
        "shock_sampling": {
            "y": jump["y"], "x_shock": jump["x_shock"], "standoff": jump["standoff"],
            "n_cells_behind": jump["n_cells_behind"], "window": jump["window"],
            "sensitivity": jump["sensitivity"],
        },
        "all_graded_pass": all(i["pass"] for i in graded),
        "all_pass": all(i["pass"] for i in items),
    }


def _tags(run_dir: Path) -> List[str]:
    tags = [p.stem.split("_", 1)[1] for p in Path(run_dir).glob("fields_*.npz")]
    tags.sort(key=lambda g: int(g.split("x")[0]) * int(g.split("x")[1]))
    return tags


def convergence(grades: Sequence[Dict]) -> Dict:
    """Observed error order and Richardson-extrapolated values over the grid triple.

    Reported, *not* graded: the pre-registered criterion is the grid of record.
    For a refinement ratio ``r = dx_coarse / dx_fine`` and errors ``e`` against
    the closed-form reference, the observed order is
    ``p = ln|e_coarse/e_fine| / ln r`` and the grid-independent estimate is
    ``f0 = f_f / (1 - (e_f/e_c - 1))`` written in value form below.  A missing
    monotone decay is reported as ``order: null`` rather than fitted.
    """
    out = {}
    if len(grades) < 2:
        return {"per_quantity": {}, "note": "need at least two grids"}
    for idx, (label, got_key, _group, ref_key) in enumerate(GRADED_KEYS):
        rows = [(g["dx"], g["items"][idx]["got"], g["items"][idx]["rel_error"]) for g in grades]
        fine, coarse = rows[-1], rows[-2]
        r = coarse[0] / fine[0]
        e_f, e_c = fine[2], coarse[2]
        entry = {"values": [v for _, v, _ in rows], "errors": [e for _, _, e in rows],
                 "refinement_ratio": r}
        if e_f != 0.0 and e_c != 0.0 and (e_f * e_c) > 0 and abs(e_c) > abs(e_f):
            order = math.log(abs(e_c / e_f)) / math.log(r)
            entry["order"] = order
            ref = grades[-1]["items"][idx]["reference"]
            entry["extrapolated"] = ref * (1.0 + e_f * (1.0 - 1.0 / (r ** order - 1.0)))
            entry["extrapolated_error"] = entry["extrapolated"] / grades[-1]["items"][idx]["reference"] - 1.0
        else:
            entry["order"] = None
            entry["extrapolated"] = None
            entry["extrapolated_error"] = None
        out[label] = entry
    return {"per_quantity": out}


def table_markdown(grading: Dict) -> str:
    """Render the ref-vs-got table of the graded run (markdown, tick/cross)."""
    lines = [
        "| quantity | reference | got | rel. error | tolerance | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for item in grading["items"]:
        mark = "✅" if item["pass"] else "❌"
        rel = f"{item['rel_error'] * 100:+.2f} %" if item["tolerance"] == TOL else f"{item['rel_error']:.2e}"
        lines.append(
            f"| {item['quantity']} | {item['reference']:.6g} | {item['got']:.6g} | "
            f"{rel} | {item['tolerance']:g} | {mark} |")
    grid = grading["grid"]
    lines.append("")
    lines.append(f"grid {grid[0]}x{grid[1]} (dx = {grading['dx']:.6g}), "
                 f"t = {grading['t_end']:g}, {grading['nsteps']} steps, "
                 f"all graded pass = {grading['all_graded_pass']}")
    return "\n".join(lines)


def grade_dir(run_dir: Path, tolerance: float = TOL) -> Dict:
    """Grade every saved grid in ``run_dir``; the grid of record is graded."""
    run_dir = Path(run_dir)
    grades = []
    for tag in _tags(run_dir):
        prim, cfg, meta = load_run(run_dir, tag)
        grades.append(grade_run(prim, cfg, meta, tolerance=tolerance))
    if not grades:
        raise FileNotFoundError(f"no fields_*.npz under {run_dir}")
    record = next((g for g in grades if tuple(g["grid"]) == GRID_OF_RECORD), grades[-1])
    return {
        "case": CASE,
        "tolerance": tolerance,
        "grids": [g["grid"] for g in grades],
        "grid_of_record": record["grid"],
        "is_grid_of_record": tuple(record["grid"]) == GRID_OF_RECORD,
        "record": record,
        "per_grid": grades,
        "convergence": convergence(grades),
        "all_graded_pass": record["all_graded_pass"] and all(
            i["pass"] for i in record["items"]),
    }


def write_grading(run_dir: Path, tolerance: float = TOL) -> Dict:
    run_dir = Path(run_dir)
    grading = grade_dir(run_dir, tolerance=tolerance)
    (run_dir / "grading.json").write_text(json.dumps(grading, indent=1, sort_keys=True))
    (run_dir / "table.md").write_text(table_markdown(grading["record"]) + "\n")
    return grading


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Grade the CASE-FORWARD-STEP benchmark run")
    ap.add_argument("--runs", default="docs/repro-fvm-forward-step/run",
                    help="directory holding fields_<nx>x<ny>.npz + run_<nx>x<ny>.json")
    ap.add_argument("--tolerance", type=float, default=TOL)
    ap.add_argument("--json", action="store_true", help="print the whole grading dict")
    args = ap.parse_args(argv)
    grading = write_grading(Path(args.runs), tolerance=args.tolerance)
    if args.json:
        print(json.dumps(grading, indent=1, sort_keys=True))
    else:
        print(table_markdown(grading["record"]))
        conv = grading["convergence"].get("per_quantity", {})
        for label, entry in conv.items():
            ordr = entry.get("order")
            ex = entry.get("extrapolated")
            if ordr is None or ex is None:
                print(f"[convergence] {label}: order n/a, extrapolated n/a")
            else:
                print(f"[convergence] {label}: order {ordr:.2f}, "
                      f"extrapolated {ex:.6g} ({entry['extrapolated_error'] * 100:+.2f} %)")
    return 0 if grading["all_graded_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())