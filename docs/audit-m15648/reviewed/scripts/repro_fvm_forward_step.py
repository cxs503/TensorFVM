#!/usr/bin/env python
"""Reproduce Woodward & Colella (1984) problem 2 (Mach 3 forward-facing step).

    python scripts/repro_fvm_forward_step.py --out docs/repro-fvm-forward-step

Runs the impulsively started Mach-3 wind-tunnel-with-a-step case on a grid
triple with TensorFVM's compressible Euler solver (embedded solid wall), grades
the shock-jump criteria against the exact normal-shock relations for the
paper's inflow state, writes ``artifacts/forward_step.json`` + the finest-grid
fields (``.npz``) + four figures, and exits non-zero when a criterion misses
its tolerance.  The case lives in
:mod:`tensorfvm.compressible.forward_step`; this file is only the CLI/driver so
that the reproduction has one documented command.

Reference
    Woodward, P. & Colella, P., "The numerical simulation of two-dimensional
    fluid flow with strong shocks", *J. Comput. Phys.* **54** (1984) 115-173,
    problem 2: uniform Mach 3 flow (gamma = 1.4, rho = 1.4, p = 1) enters a
    tunnel of length 3 and height 1 through a step of height 0.2 that begins at
    x = 0.6; all walls reflecting, the run is impulsively started and reported
    at t = 4.  That paper publishes density contours for this problem, not
    tables, so no number in this file is transcribed from it: the graded
    reference values are evaluated in code from the Rankine-Hugoniot and
    isentropic relations for the stated inflow state.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.compressible.forward_step import (  # noqa: E402
    T_END,
    ForwardStepConfig,
    mass_flux_balance,
    measure_shock_jump,
    normal_shock_reference,
    run_case,
    stagnation_state,
    step_force,
    wall_profiles,
)

#: tolerance of the mission (mission text: every graded number within 8 %)
TOL = 0.08
GRADED = ("p_ratio", "rho_ratio", "u_ratio", "mach2")
DEFAULT_GRIDS = ((120, 40), (240, 80), (480, 160))


def _grid_run(nx: int, ny: int, t_end: float) -> dict:
    cfg = ForwardStepConfig(nx=nx, ny=ny, t_end=t_end)
    t0 = time.time()
    result = run_case(cfg)
    wall = time.time() - t0
    prim = result["prim"]
    jump = measure_shock_jump(prim, cfg)
    row = {
        "grid": {"nx": nx, "ny": ny, "dx": cfg.dx, "dy": cfg.dy},
        "wall_seconds": wall,
        "nsteps": result["nsteps"],
        "time": result["time"],
        "positivity_fixes": result["positivity_fixes"],
        "shock": jump,
        "stagnation": stagnation_state(prim, cfg),
        "wall_pressure": wall_profiles(prim, cfg),
        "mass_flux": mass_flux_balance(prim, cfg),
        "step_force": step_force(prim, cfg),
        "budget": result["budget"],
    }
    return row, prim, cfg


def _criteria(rows: list) -> dict:
    """Grade the finest grid against the exact relations (tolerance predeclared)."""
    fine = rows[-1]
    jump, stag = fine["shock"], fine["stagnation"]
    ref = normal_shock_reference()
    spec = (
        ("p2/p1", ref["p_ratio"], jump["p_ratio"], "shock jump, pressure"),
        ("rho2/rho1", ref["rho_ratio"], jump["rho_ratio"], "shock jump, density"),
        ("u2/u1", ref["u_ratio"], jump["u_ratio"], "shock jump, velocity"),
        ("M2", ref["mach2"], jump["mach2"], "shock jump, downstream Mach"),
        ("p_stag/p1", ref["p02"] / ref["p1"], stag["p_ratio"], "wall stagnation pressure"),
        ("rho_stag/rho1", ref["rho02"] / ref["rho1"], stag["rho_ratio"],
         "wall stagnation density"),
    )
    items = []
    for name, refval, got, note in spec:
        err = got / refval - 1.0
        items.append({"quantity": name, "reference": refval, "got": got,
                      "rel_error": err, "tolerance": TOL,
                      "pass": bool(abs(err) < TOL), "note": note})
    budget = fine["budget"]
    items.append({"quantity": "mass_budget_drift", "reference": 0.0,
                  "got": budget["mass_drift"], "rel_error": budget["mass_drift"],
                  "tolerance": 1e-6, "pass": bool(abs(budget["mass_drift"]) < 1e-6),
                  "note": "fluid mass + open-end outflow ledger vs initial"})
    items.append({"quantity": "energy_budget_drift", "reference": 0.0,
                  "got": budget["energy_drift"], "rel_error": budget["energy_drift"],
                  "tolerance": 1e-2, "pass": bool(abs(budget["energy_drift"]) < 1e-2),
                  "note": "fluid energy + open-end outflow ledger vs initial"})
    graded = [i for i in items if i["tolerance"] == TOL]
    return {
        "grid_of_record": fine["grid"],
        "tolerance": TOL,
        "items": items,
        "all_graded_pass": all(i["pass"] for i in graded),
        "all_pass": all(i["pass"] for i in items),
        "ungraded": {
            "mass_flux_imbalance": fine["mass_flux"]["rel_diff"],
            "note": ("quasi-steady indicator only: the recirculation behind the step is "
                     "unsteady at t = 4, so this is not an exact reference value"),
        },
    }


def _ablation(t_end: float = 0.5) -> dict:
    """Consequence of the wall-flux projection (same grid, same end time)."""
    out = {}
    for flag in (False, True):
        cfg = ForwardStepConfig(nx=60, ny=20, t_end=t_end, project_wall_flux=flag)
        res = run_case(cfg)
        out["projected" if flag else "mirror_only"] = {
            "mass_drift": res["budget"]["mass_drift"],
            "energy_drift": res["budget"]["energy_drift"],
            "p_ratio": measure_shock_jump(res["prim"], cfg)["p_ratio"],
            "t_end": t_end, "grid": [60, 20],
        }
    return out


def _figures(outdir: Path, rows: list, prim, cfg) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    made = []

    xc, yc = cfg.centers()
    xs, ys = xc.numpy(), yc.numpy()
    rho = prim["rho"].numpy().T
    solid = cfg.solid_mask().numpy().T

    # 1: density contours at t_end with the block masked out
    fig, ax = plt.subplots(figsize=(9, 3.4), dpi=140)
    rho_masked = np.ma.masked_array(rho, mask=solid)
    levels = np.linspace(float(np.min(rho)), float(np.max(rho)), 31)
    cs = ax.contourf(xs, ys, rho_masked, levels=levels, cmap="inferno", extend="both")
    ax.contour(xs, ys, rho_masked, levels=levels[::3], colors="k", linewidths=0.3, alpha=0.5)
    ax.set_aspect("equal")
    ax.add_patch(plt.Rectangle((cfg.step_x, 0.0), cfg.lx - cfg.step_x, cfg.step_h,
                               facecolor="0.35", edgecolor="k", lw=0.5, zorder=3))
    ax.set_xlim(0.0, cfg.lx)
    ax.set_ylim(0.0, cfg.ly)
    ax.set_xlabel("x / tunnel height")
    ax.set_ylabel("y / tunnel height")
    ax.set_title(f"Mach 3 forward-facing step, t = {cfg.t_end:g}: density (finest grid)")
    fig.colorbar(cs, ax=ax, label=r"$\rho$")
    path = figdir / "forward_step_density.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    made.append(str(path.relative_to(outdir)))

    # 2: centre-line profiles with the exact normal-shock levels
    ref = normal_shock_reference()
    probe_y = cfg.step_h / 2.0
    j = int(torch.argmin(torch.abs(yc - probe_y)).item())
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=140)
    for ax, key, refval, label in (
        (axes[0], "rho", ref["rho2"], r"$\rho$"),
        (axes[1], "p", ref["p2"], "p"),
        (axes[2], "u", ref["u2"], "u"),
    ):
        ax.plot(xc.numpy(), prim[key][:, j].numpy(), "-", lw=1.2, label="TensorFVM")
        ax.axhline(refval, color="crimson", ls="--", lw=1.0,
                   label="Rankine-Hugoniot post-shock")
        ax.axvline(cfg.step_x, color="grey", ls=":", lw=1.0)
        ax.set_xlabel("x")
        ax.set_title(f"{label} on the stagnation streamline y = {probe_y:g}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Near-normal shock jump vs exact normal-shock state")
    path = figdir / "forward_step_centreline.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    made.append(str(path.relative_to(outdir)))

    # 3: wall pressure distribution resolved along the wall
    fine = rows[-1]
    wp = fine["wall_pressure"]
    fig, ax = plt.subplots(figsize=(7.5, 3.4), dpi=140)
    ax.plot(wp["floor"]["x"], wp["floor"]["p"], label="upstream floor (y = 0)")
    ax.plot([cfg.step_x] * len(wp["step_face"]["p"]), wp["step_face"]["p"],
            label="step face (x = 0.6)")
    ax.plot(wp["roof"]["x"], wp["roof"]["p"], label="step roof (y = 0.2)")
    ax.axhline(ref["p2"], color="crimson", ls="--", lw=1.0,
               label="p2 (normal shock, M = 3)")
    ax.set_xlabel("x")
    ax.set_ylabel("wall pressure p")
    ax.set_title("Wall pressure (finest grid)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    path = figdir / "forward_step_wall_pressure.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    made.append(str(path.relative_to(outdir)))

    # 4: error of the four graded quantities against grid spacing
    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=140)
    dxs = [r["grid"]["dx"] for r in rows]
    for key, refval in (("p_ratio", ref["p_ratio"]), ("rho_ratio", ref["rho_ratio"]),
                        ("u_ratio", ref["u_ratio"]), ("mach2", ref["mach2"])):
        errs = [abs(r["shock"][key] / refval - 1.0) * 100.0 for r in rows]
        ax.loglog(dxs, errs, "o-", label=key)
    ax.axhline(TOL * 100.0, color="crimson", ls="--", lw=1.0, label="8 % criterion")
    ax.set_xlabel("cell size dx = dy")
    ax.set_ylabel("|rel. error| [%]")
    ax.set_title("Shock-jump error vs mesh size")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    path = figdir / "forward_step_error_vs_dx.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    made.append(str(path.relative_to(outdir)))
    return made


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "repro-fvm-forward-step"))
    ap.add_argument("--t-end", type=float, default=T_END)
    ap.add_argument("--grids", default=",".join(f"{a}x{b}" for a, b in DEFAULT_GRIDS),
                    help="comma separated nx x ny list, coarse to fine")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    grids = []
    for item in args.grids.split(","):
        nx, ny = item.lower().split("x")
        grids.append((int(nx), int(ny)))
    outdir = Path(args.out)
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)

    rows, fine_prim, fine_cfg = [], None, None
    t_start = time.time()
    for nx, ny in grids:
        print(f"[forward-step] running {nx}x{ny} to t = {args.t_end:g} ...", flush=True)
        row, prim, cfg = _grid_run(nx, ny, args.t_end)
        print(f"[forward-step]   {row['nsteps']} steps, {row['wall_seconds']:.1f} s, "
              f"shock at x = {row['shock']['x_shock']:.4f}, "
              f"p2/p1 = {row['shock']['p_ratio']:.4f} "
              f"({100 * row['shock']['errors']['p_ratio']:+.2f} %)", flush=True)
        rows.append(row)
        fine_prim, fine_cfg = prim, cfg

    criteria = _criteria(rows)
    ablation = _ablation()
    ref = normal_shock_reference()
    payload = {
        "case": "woodward-colella-1984-problem-2-forward-facing-step",
        "reproduction": "REPRO-FVM-FORWARD-STEP",
        "reference": {
            "citation": "Woodward & Colella, J. Comput. Phys. 54 (1984) 115-173, problem 2",
            "reported_as": "density contour plots at fixed times (no tables in the paper)",
            "reference_numbers_computed_from": (
                "exact Rankine-Hugoniot / isentropic relations for the paper's "
                "stated inflow state (M = 3, gamma = 1.4, rho = 1.4, p = 1)"
            ),
            "values": ref,
        },
        "setup": {
            "lx": fine_cfg.lx, "ly": fine_cfg.ly, "step_x": fine_cfg.step_x,
            "step_h": fine_cfg.step_h, "t_end": args.t_end, "cfl": fine_cfg.cfl,
            "order": fine_cfg.order, "limiter": fine_cfg.limiter,
            "bc": ["inflow (M=3 uniform)", "outflow (zero gradient)", "reflective", "reflective"],
            "wall_representation": "embedded solid mask on cell faces (no staircase error)",
            "scheme": "MUSCL-Hancock (MC limiter) + HLLC",
        },
        "repro_cmd": ("python scripts/repro_fvm_forward_step.py --out "
                      "docs/repro-fvm-forward-step --t-end 4"),
        "grids": [row["grid"] | {"nsteps": row["nsteps"], "wall_seconds": row["wall_seconds"]}
                  for row in rows],
        "results": rows,
        "criteria": criteria,
        "wall_treatment_ablation": ablation,
        "wall_seconds_total": time.time() - t_start,
        "ecosystem_friction": [
            "control-node load average was >300 on 128 cores during development, so the "
            "grid triple was run on the cluster queue instead of the control node",
            "the platform knowledge-base search endpoint (:8021) answered 'engine_offline', "
            "so the same-domain prior-art check had to be done with grep in the repository",
            "EulerSolver had no embedded-wall support: the step had to be added as a "
            "mirrored solid mask (solid=) before the case could be run at all",
        ],
    }

    figures = [] if args.no_figures else _figures(outdir, rows, fine_prim, fine_cfg)
    payload["figures"] = figures
    (outdir / "artifacts" / "forward_step.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False))

    # finest grid fields, for downstream plotting / reuse
    import numpy as np

    np.savez_compressed(
        outdir / "artifacts" / "forward_step_fields.npz",
        x=fine_cfg.centers()[0].numpy(), y=fine_cfg.centers()[1].numpy(),
        rho=fine_prim["rho"].numpy(), u=fine_prim["u"].numpy(),
        v=fine_prim["v"].numpy(), p=fine_prim["p"].numpy(),
        solid=fine_cfg.solid_mask().numpy(),
    )

    print(json.dumps({"criteria": criteria["items"]}, indent=2))
    return 0 if criteria["all_pass"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())