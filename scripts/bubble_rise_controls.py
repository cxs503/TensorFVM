"""Control experiments for the 2-D bubble-rise benchmark.

The benchmark answers "how far is the terminal velocity from the ellipsoidal-
bubble correlation?".  These controls answer "**why**", with three one-factor
variations on a coarse grid:

1. ``static_bubble`` -- gravity switched off, convection off: whatever velocity
   remains is the surface-tension (CSF) spurious current.  It is the floor of the
   velocity error budget and it must stay well below the anchor.
2. ``density_ratio`` -- 10 vs 100 (air/water is ~833).  The gas phase is
   ``1/rho_g`` amplified in both the CSF force and the pressure correction, so
   this measures how much of the residual spurious motion is a light-phase
   artefact.
3. ``confinement`` -- domain width 4d vs 8d.  A 2-D bubble between close walls
   carries an extra drag that no 3-D correlation contains.

Usage::

    PYTHONPATH=src python scripts/bubble_rise_controls.py --output docs/bubble-rise-2d
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from tensorfvm.verification.bubble_rise import (
    BubbleCase, TERMINAL_VELOCITY_TOLERANCE, fan_tsuchiya_velocity, run_case,
)

BASE = dict(name="control", eotvos=1.23, morton=2.63e-11, density_ratio=10.0,
            viscosity_ratio=55.0, cells_per_diameter=12, end_time=1.5)


def _variant(**kwargs) -> BubbleCase:
    params = dict(BASE)
    params.update(kwargs)
    return BubbleCase(**params)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("docs/bubble-rise-2d"))
    parser.add_argument("--cells", type=int, default=12)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    started = time.time()
    cases = {
        "baseline_n12": _variant(cells_per_diameter=args.cells),
        "static_bubble_no_gravity": _variant(cells_per_diameter=args.cells,
                                             momentum_advection=False, end_time=0.5),
        "density_ratio_100": _variant(cells_per_diameter=args.cells,
                                      density_ratio=100.0),
        "domain_width_8d": _variant(cells_per_diameter=args.cells,
                                    domain_width=8.0, end_time=2.0),
    }
    results = {}
    for name, case in cases.items():
        print(f"[{name}] n={case.cells_per_diameter} "
              f"width={case.domain_width}d ratio={case.density_ratio}", flush=True)
        if name == "static_bubble_no_gravity":
            # g = 0 is not a BubbleCase field: patch the solver config instead.
            grid, tp = case.solve_config()
            tp.gravity = 0.0
            from tensorfvm.two_phase_solver import TwoPhaseSolver
            solver = TwoPhaseSolver(grid, tp)
            bubble = case.initial_alpha(grid.nx, grid.ny, solver.dx, solver.dy)
            solver.set_alpha(1.0 - bubble)
            peak, drift0 = 0.0, float((1.0 - solver.alpha).sum())
            while solver.time < case.end_time:
                solver.step(solver.stable_time_step())
                peak = max(peak, float(max(solver.u.abs().max(), solver.v.abs().max())))
            vol = float((1.0 - solver.alpha).sum())
            results[name] = {
                "case": dataclasses.asdict(case),
                "peak_spurious_velocity": peak,
                "relative_to_anchor": peak / fan_tsuchiya_velocity(case.eotvos),
                "volume_drift": (vol - drift0) / drift0,
                "steps": len(solver.history),
            }
        else:
            run = run_case(case, progress=True)
            results[name] = {
                "case": run["case"],
                "terminal_velocity": run["terminal_velocity"],
                "anchor": run["terminal_velocity_anchor"],
                "relative_error": (run["terminal_velocity"]
                                   - run["terminal_velocity_anchor"])
                                  / run["terminal_velocity_anchor"],
                "aspect_ratio": run["shape"]["aspect_ratio"],
                "circularity": run["shape"]["circularity"],
                "reynolds": run["dimensionless"]["reynolds"],
                "worst_step_drift": run["conservation"]["worst_step_relative_drift"],
                "steps": run["cost"]["steps"],
            }
        print(f"  -> {json.dumps({k: v for k, v in results[name].items() if k != 'case'}, default=str)}",
              flush=True)

    base_u = results["baseline_n12"]["terminal_velocity"]
    findings = {
        "spurious_current_floor": results["static_bubble_no_gravity"]["relative_to_anchor"],
        "density_ratio_effect": (results["density_ratio_100"]["terminal_velocity"]
                                 - base_u) / base_u,
        "confinement_effect": (results["domain_width_8d"]["terminal_velocity"]
                               - base_u) / base_u,
        "declared_tolerance": TERMINAL_VELOCITY_TOLERANCE,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    payload = {"controls": results, "findings": findings,
               "elapsed_seconds": round(time.time() - started, 1)}
    (args.output / "controls.json").write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(findings, indent=2))
    _figure(payload, args.output)
    return 0


def _figure(payload: dict, output_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [k for k in payload["controls"] if k != "static_bubble_no_gravity"]
    values = [payload["controls"][k]["terminal_velocity"] for k in names]
    anchor = payload["controls"]["baseline_n12"]["anchor"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar(range(len(names)), values, color="C0", label="measured U_T")
    ax.axhline(anchor, color="k", ls="--", label="Fan-Tsuchiya anchor")
    ax.axhspan(anchor * (1 - TERMINAL_VELOCITY_TOLERANCE),
               anchor * (1 + TERMINAL_VELOCITY_TOLERANCE), color="k", alpha=0.08,
               label=f"+/-{TERMINAL_VELOCITY_TOLERANCE:.0%} band")
    ax.axhline(payload["controls"]["static_bubble_no_gravity"]["peak_spurious_velocity"],
               color="r", ls=":", label="spurious-current floor (g=0)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=12, fontsize=8)
    ax.set_ylabel("U sqrt(gd)")
    ax.set_title("Bubble-rise controls: what limits the terminal velocity")
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_dir / "fig5-controls.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
