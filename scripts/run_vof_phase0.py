#!/usr/bin/env python
"""Phase-0 acceptance campaign for the TensorFVM two-phase VOF capability.

Runs the machine-checkable gates of node ``p233w1-N03`` and writes a single
JSON plus the report figures:

    AC2  dam-break surge front vs Martin & Moyce (1952) at t* = 1, 2, 3
    AC3  density ratio 1000:1, >= 1000 time steps, no divergence
    AC4  total liquid-volume residual <= 1e-6
    AC5  three-grid GCI study (error must decrease monotonically)
    AC6  figures: free-surface snapshots, front comparison, GCI, mass ledger

Usage::

    python scripts/run_vof_phase0.py --out docs/vof-phase0/artifacts
    python scripts/run_vof_phase0.py --quick          # coarse smoke run
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorfvm.vof_phase0.anchors import (  # noqa: E402
    martin_moyce_front,
    martin_moyce_height,
    ritter_front,
)
from tensorfvm.vof_phase0.dam_break import (  # noqa: E402
    DamBreakConfig,
    run_dam_break,
)
from tensorfvm.vof_phase0.solver import Phase0TwoPhaseSolver  # noqa: E402

GATE_FRONT_TOL = 0.10       # AC2: 10 % relative front error
GATE_MASS_RESIDUAL = 1.0e-6  # AC4
GATE_DENSITY_RATIO = 1000.0  # AC3
GATE_DENSITY_STEPS = 1000    # AC3
SAMPLE_TIMES = (1.0, 2.0, 3.0)


def _sample(t_star: np.ndarray, values: np.ndarray, target: float) -> float:
    i = int(np.argmin(np.abs(t_star - target)))
    return float(values[i])


def run_case(nx: int, ny: int, t_star_end: float, surface_model: str = "void",
             pressure_init: str = "hydrostatic") -> dict:
    cfg = DamBreakConfig(nx=nx, ny=ny, t_star_end=t_star_end,
                         pressure_init=pressure_init)
    solver = Phase0TwoPhaseSolver(
        nx=nx, ny=ny, length=cfg.length, height=cfg.height,
        rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas, gravity=cfg.gravity,
        alpha_cfl=cfg.cfl, vof_bounds=cfg.vof_bounds, surface_model=surface_model,
    )
    result = run_dam_break(cfg, solver=solver)
    return result


def front_errors(result: dict, times=SAMPLE_TIMES) -> list[dict]:
    rows = []
    for target in times:
        i = int(np.argmin(np.abs(result["t_star"] - target)))
        sim = float(result["front_star"][i])
        ref = martin_moyce_front(target)
        rows.append({
            "t_star": float(result["t_star"][i]),
            "x_front_m": float(result["front"][i]),
            "x_star_sim": sim,
            "x_star_ref": ref,
            "rel_error": (sim - ref) / ref,
            "abs_rel_error": abs(sim - ref) / ref,
            "ritter_x_star": (ritter_front(float(result["t"][i]), result["h0"],
                                           result["config"]["gravity"],
                                           result["config"]["a"])
                              - result["config"]["a"]) / result["config"]["a"],
        })
    return rows


def richardson_positions(coarse: float, medium: float, fine: float,
                         ratio: float = 2.0) -> dict:
    """Observed order, Richardson extrapolation and Roache GCI (3 grids).

    Uses the three front positions themselves, which is the standard
    self-convergence use of GCI: the reference is the grid limit, not the
    experiment, so the number says how far the discretisation is from
    converging, not whether the model agrees with the data.
    """
    d1, d2 = medium - coarse, fine - medium
    if d2 == 0.0 or d1 / d2 <= 0.0:
        return {"order": None, "extrapolated": None, "gci_fine": None}
    p = math.log(abs(d1 / d2)) / math.log(ratio)
    p = max(0.1, min(6.0, p))
    denom = ratio ** p - 1.0
    extrapolated = fine + (fine - medium) / denom
    gci = 1.25 * abs(medium - fine) / denom / abs(fine)
    return {"order": p, "extrapolated": extrapolated, "gci_fine": gci}


def richardson(medium: float, fine: float, ratio: float = 2.0) -> dict:
    """Observed order of accuracy and Roache GCI from a coarse-fine pair."""
    if fine == 0.0:
        return {"order": None, "gci_fine": None}
    if medium == fine:
        return {"order": None, "gci_fine": 0.0}
    # iterative observed order (constant refinement ratio assumed)
    p = 1.0
    for _ in range(200):
        denom = ratio ** p - 1.0
        if denom == 0.0:
            break
        p_new = math.log(abs(medium / fine)) / math.log(ratio) \
            if medium != 0.0 and fine != 0.0 else p
        p_new = max(0.1, min(6.0, p_new)) if math.isfinite(p_new) else p
        if abs(p_new - p) < 1e-8:
            p = p_new
            break
        p = p_new
    gci = 1.25 * abs(medium - fine) / (ratio ** p - 1.0) / abs(fine)
    return {"order": p, "gci_fine": gci}


def stability_case(nx: int, ny: int, steps: int,
                   surface_model: str = "two_phase") -> dict:
    """1000:1 density-ratio stability run with the fully coupled projection."""
    cfg = DamBreakConfig(nx=nx, ny=ny, t_star_end=1.0e9, max_steps=steps)
    solver = Phase0TwoPhaseSolver(
        nx=nx, ny=ny, length=cfg.length, height=cfg.height,
        rho_liquid=cfg.rho_liquid, rho_gas=cfg.rho_gas,
        mu_liquid=cfg.mu_liquid, mu_gas=cfg.mu_gas, gravity=cfg.gravity,
        alpha_cfl=cfg.cfl, surface_model=surface_model,
    )
    from tensorfvm.vof_phase0.dam_break import initial_alpha

    solver.set_alpha(initial_alpha(cfg))
    solver.set_hydrostatic_pressure(cfg.h0, cfg.a)
    v0 = solver.liquid_volume()
    finite = True
    alpha_min, alpha_max = float(solver.alpha.min()), float(solver.alpha.max())
    speed_max = 0.0
    started = time.perf_counter()
    for _ in range(steps):
        dt = solver.stable_dt()
        info = solver.step(dt)
        if not (math.isfinite(info["speed_max"])):
            finite = False
            break
        alpha_min = min(alpha_min, info["alpha_min"])
        alpha_max = max(alpha_max, info["alpha_max"])
        speed_max = max(speed_max, info["speed_max"])
        if not np.isfinite(solver.alpha.cpu().numpy()).all():
            finite = False
            break
    volume = solver.liquid_volume()
    return {
        "surface_model": surface_model,
        "density_ratio": solver.rho_l / solver.rho_g,
        "steps_requested": steps,
        "steps_completed": solver.steps,
        "final_time_s": solver.time,
        "wall_s": time.perf_counter() - started,
        "finite": finite,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "speed_max": speed_max,
        "volume_initial": v0,
        "volume_final": volume,
        "volume_residual": abs(volume - v0),
        "volume_residual_rel": abs(volume - v0) / v0,
        "cg_last": solver.cg_last,
    }


def hydrostatic_equilibrium(nx: int = 40, ny: int = 40, steps: int = 200) -> dict:
    """Static stratified pool: the spurious-current / exact-equilibrium check."""
    import torch

    length, height, depth = 0.4, 0.4, 0.2
    out = {}
    for model in ("void", "two_phase"):
        solver = Phase0TwoPhaseSolver(
            nx=nx, ny=ny, length=length, height=height, surface_model=model)
        ys = (torch.arange(ny) + 0.5) * solver.dy
        alpha = (ys[:, None] < depth).to(torch.float64).expand(ny, nx).clone()
        solver.set_alpha(alpha)
        head = torch.clamp(depth - ys, min=0.0)[:, None].expand(ny, nx)
        solver.p = torch.where(alpha > 0.5,
                               1000.0 * 9.81 * head,
                               torch.zeros_like(head)).clone()
        v0 = solver.liquid_volume()
        for _ in range(steps):
            solver.step(solver.stable_dt())
        out[model] = {
            "spurious_u_max": float(solver.u.abs().max()),
            "spurious_v_max": float(solver.v.abs().max()),
            "alpha_min": float(solver.alpha.min()),
            "alpha_max": float(solver.alpha.max()),
            "volume_residual_rel": abs(solver.liquid_volume() - v0) / v0,
        }
    return out


def figures(out_dir: Path, main: dict, errors: list[dict],
            gci_rows: list[dict]) -> list[dict]:
    """Render the report figures; returns [{name, path, caption}]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = []
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1. free-surface snapshots
    alpha = np.asarray(main["alpha_final"])
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    im = ax.imshow(alpha, origin="lower", aspect="auto", cmap="Blues",
                   extent=[0.0, main["config"]["length"], 0.0,
                           main["config"]["height"]], vmin=0.0, vmax=1.0)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title(f"Volume fraction, t* = {main['t_star'][-1]:.1f} "
                 f"({main['config']['nx']}x{main['config']['ny']})")
    fig.colorbar(im, ax=ax, label=r"$\alpha$")
    fig.tight_layout()
    path = fig_dir / "free_surface_field.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "free_surface_field", "path": str(path),
                    "caption": "Liquid volume fraction at the end of the run "
                               "(algebraic VOF interface)."})

    # 2. front trajectory vs Martin & Moyce and Ritter
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(main["t_star"], main["front_star"], "o-", ms=3, lw=1.2,
            label="TensorFVM VOF (simulation)")
    mm = np.linspace(0.1, main["t_star"][-1], 200)
    ax.plot(mm, [martin_moyce_front(t) for t in mm], "s--", ms=3, lw=1.4,
            label="Martin & Moyce (1952)")
    ax.plot(main["t_star"],
            [(ritter_front(t, main["h0"], main["config"]["gravity"],
                           main["config"]["a"]) - main["config"]["a"])
             / main["config"]["a"] for t in main["t"]],
            ":", lw=1.0, label="Ritter (1892) shallow-water limit")
    ax.set_xlabel(r"$t^* = t\sqrt{g/a}$")
    ax.set_ylabel(r"$x^* = (x_f - a)/a$")
    ax.set_title("Surge front: simulation vs experiment vs analytic limit")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = fig_dir / "front_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "front_comparison", "path": str(path),
                    "caption": "Surge-front trajectory against the digitised "
                               "Martin & Moyce curve and the Ritter limit."})

    # 3. GCI / grid convergence
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    dx = [row["dx_mm"] for row in gci_rows]
    err = [abs(row["error_at_t2"]) * 100.0 for row in gci_rows]
    ax.plot(dx, err, "o-")
    for row in gci_rows:
        ax.annotate(f"{row['error_at_t2']*100:+.1f}%",
                    (row["dx_mm"], abs(row["error_at_t2"]) * 100.0),
                    textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("cell size dx [mm]")
    ax.set_ylabel("|front error| at t* = 2 [%]")
    ax.set_title("Grid convergence of the surge-front error")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = fig_dir / "grid_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "grid_convergence", "path": str(path),
                    "caption": "Surge-front error versus cell size (three "
                               "grids, refinement ratio 2)."})

    # 4. mass ledger
    fig, ax = plt.subplots(figsize=(5.4, 3.2))
    drift = np.asarray(main["volume"]) - main["volume_initial"]
    ax.semilogy(main["t_star"], np.maximum(np.abs(drift), 1e-20), "o-", ms=2)
    ax.set_xlabel(r"$t^*$")
    ax.set_ylabel("|V - V0| [m$^2$]")
    ax.set_title("Liquid volume ledger (flux-form VOF)")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = fig_dir / "mass_ledger.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    figures.append({"name": "mass_ledger", "path": str(path),
                    "caption": "Absolute liquid-volume residual over the run."})
    return figures


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/vof-phase0/artifacts")
    ap.add_argument("--grids", default="60x20,120x40,240x80")
    ap.add_argument("--t-star-end", type=float, default=4.0)
    ap.add_argument("--quick", action="store_true",
                    help="coarse grids and short runs (smoke test)")
    ap.add_argument("--skip-stability", action="store_true")
    args = ap.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    grids = []
    for spec in args.grids.split(","):
        nx, ny = (int(v) for v in spec.lower().split("x"))
        grids.append((nx, ny))
    if args.quick:
        grids = [(40, 14), (60, 20), (80, 27)]
        args.t_star_end = min(args.t_star_end, 2.0)

    report: dict = {
        "node": "p233w1-N03",
        "package": "tensorfvm.vof_phase0",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gates": {},
        "runs": [],
    }

    # ---- grid study / AC2 / AC4 -------------------------------------- #
    gci_rows = []
    main_result = None
    for nx, ny in grids:
        result = run_case(nx, ny, args.t_star_end)
        errors = front_errors(result)
        row = {
            "nx": nx, "ny": ny,
            "dx_mm": result["config"]["length"] / nx * 1000.0,
            "steps": result["steps"],
            "wall_s": result["elapsed_s"],
            "max_volume_residual": result["max_volume_residual"],
            "alpha_bounds": list(result["alpha_bounds"]),
            "diverged": result["diverged"],
            "errors": errors,
            "error_at_t2": errors[1]["rel_error"] if len(errors) > 1 else None,
        }
        report["runs"].append(row)
        gci_rows.append(row)
        if nx == max(g[0] for g in grids):
            main_result = result
        print(f"[grid {nx}x{ny}] steps={row['steps']} wall={row['wall_s']:.0f}s "
              f"mass_res={row['max_volume_residual']:.2e} "
              f"err(t*=2)={row['error_at_t2']*100:+.1f}%", flush=True)

    # AC2: front error at t* = 1, 2, 3 on the finest grid
    finest = [r for r in report["runs"] if r["nx"] == max(g[0] for g in grids)][0]
    ac2_rows = finest["errors"]
    ac2_pass = all(r["abs_rel_error"] <= GATE_FRONT_TOL for r in ac2_rows)
    report["gates"]["AC2_front_vs_martin_moyce"] = {
        "criterion": f"relative front error <= {GATE_FRONT_TOL:.0%} at t* = 1, 2, 3",
        "grid": f"{finest['nx']}x{finest['ny']}",
        "rows": ac2_rows,
        "max_abs_rel_error": max(r["abs_rel_error"] for r in ac2_rows),
        "passed": ac2_pass,
    }

    # AC4
    ac4_value = max(r["max_volume_residual"] for r in report["runs"])
    report["gates"]["AC4_mass_residual"] = {
        "criterion": f"|V - V0| <= {GATE_MASS_RESIDUAL:g} m^2",
        "max_residual": ac4_value,
        "passed": ac4_value <= GATE_MASS_RESIDUAL,
    }

    # AC5: GCI on the front position at t* = 2
    rows_by_dx = sorted(gci_rows, key=lambda r: -r["dx_mm"])
    monotone = all(abs(rows_by_dx[i + 1]["error_at_t2"])
                   <= abs(rows_by_dx[i]["error_at_t2"]) + 1e-12
                   for i in range(len(rows_by_dx) - 1))
    positions = [r["errors"][1]["x_star_sim"] for r in rows_by_dx]
    ratio = rows_by_dx[0]["dx_mm"] / rows_by_dx[1]["dx_mm"]
    rich = richardson_positions(*positions, ratio=ratio)
    report["gates"]["AC5_grid_convergence"] = {
        "criterion": "|front error| decreases monotonically on 3 grids; "
                     "observed order and GCI reported",
        "errors_at_t2": [r["error_at_t2"] for r in rows_by_dx],
        "positions_at_t2": positions,
        "dx_mm": [r["dx_mm"] for r in rows_by_dx],
        "ref_at_t2": martin_moyce_front(2.0),
        "monotone": monotone,
        "observed_order": rich["order"],
        "gci_fine": rich["gci_fine"],
        "richardson_extrapolated": rich["extrapolated"],
        "extrapolated_rel_error": (
            (rich["extrapolated"] - martin_moyce_front(2.0))
            / martin_moyce_front(2.0)) if rich["extrapolated"] else None,
        "passed": bool(monotone),
    }

    # AC3: density-ratio stability
    if not args.skip_stability:
        stab = stability_case(60, 20, GATE_DENSITY_STEPS)
        report["gates"]["AC3_density_ratio_stability"] = {
            "criterion": f"density ratio {GATE_DENSITY_RATIO:g}:1 stable for "
                         f">= {GATE_DENSITY_STEPS} steps without divergence",
            **stab,
            "passed": bool(stab["finite"]
                           and stab["steps_completed"] >= GATE_DENSITY_STEPS
                           and stab["alpha_min"] >= -1e-3
                           and stab["alpha_max"] <= 1.0 + 1e-3
                           and stab["volume_residual_rel"] <= GATE_MASS_RESIDUAL),
        }
        report["hydrostatic_equilibrium"] = hydrostatic_equilibrium()
        print(f"[AC3] steps={stab['steps_completed']} finite={stab['finite']} "
              f"alpha=[{stab['alpha_min']:.2e},{stab['alpha_max']:.3f}] "
              f"vol_res={stab['volume_residual_rel']:.2e}", flush=True)

    # AC6: figures + report table
    figs = figures(out_dir, main_result, ac2_rows, gci_rows)
    report["figures"] = figs
    report["gates"]["AC6_report"] = {
        "criterion": "paper-quality report with free-surface field, front "
                     "comparison curve, convergence and mass ledger",
        "figures": [f["name"] for f in figs],
        "passed": len(figs) >= 4,
    }
    report["gates"]["AC1_vof_unit_tests"] = {
        "criterion": ">= 10 pytest cases covering 1D/2D advection, rotation, shear",
        "suite": "tests/test_vof_phase0_advection.py",
        "passed": None,  # filled in by the caller after running pytest
        "note": "run `pytest tests/test_vof_phase0_advection.py -q`",
    }

    # main-run history for the report
    report["main_run"] = {
        "grid": f"{main_result['config']['nx']}x{main_result['config']['ny']}",
        "steps": main_result["steps"],
        "wall_s": main_result["elapsed_s"],
        "t_star": np.asarray(main_result["t_star"]).tolist()[-1],
        "alpha_bounds": list(main_result["alpha_bounds"]),
        "max_divergence": float(np.max(main_result["divergence_max"])),
        "mean_cg_iterations": float(np.mean(
            [c["iterations"] for c in main_result["cg"]])),
        "height_error_at_t1": None,
    }
    i1 = int(np.argmin(np.abs(main_result["t_star"] - 1.0)))
    ref_h = martin_moyce_height(1.0)
    report["main_run"]["height_error_at_t1"] = float(
        (main_result["height_star"][i1] - ref_h) / ref_h)

    out_json = out_dir / "phase0_report.json"
    out_json.write_text(json.dumps(report, indent=2, default=float))
    print(f"wrote {out_json}")
    for key, gate in report["gates"].items():
        print(f"  {key}: passed={gate.get('passed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())