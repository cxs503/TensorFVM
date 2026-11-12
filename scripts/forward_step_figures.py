#!/usr/bin/env python
"""Figures for the CASE-FORWARD-STEP benchmark run.

    python scripts/forward_step_figures.py --runs docs/repro-fvm-forward-step/run

Reads the field dumps + ``grading.json`` written by
``scripts/forward_step_study.py`` and
``python -m tensorfvm.compressible.forward_step_benchmark``, and writes

* ``figures/density.png``      -- t = 4 density field (finest grid present)
* ``figures/centreline.png``   -- rho/p/u on the stagnation streamline with the
  exact Rankine-Hugoniot levels and the sampling window marked
* ``figures/convergence.png``  -- relative error vs dx for the graded numbers
* ``figures/shock_zoom.png``   -- the numerical shock transition at every grid,
  which is what the post-shock sampling definition has to stay clear of.

Every figure writes the numbers it plots, so the report can quote them without
retyping.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from tensorfvm.compressible.forward_step import normal_shock_reference  # noqa: E402
from tensorfvm.compressible.forward_step_benchmark import load_run  # noqa: E402

LABELS = {
    "p_ratio": r"$p_2/p_1$",
    "rho_ratio": r"$\rho_2/\rho_1$",
    "u_ratio": r"$u_2/u_1$",
    "mach2": r"$M_2$",
    "p_stag_ratio": r"$p_{02}/p_1$",
    "rho_stag_ratio": r"$\rho_{02}/\rho_1$",
}


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", default=str(ROOT / "runs" / "forward_step"))
    args = ap.parse_args(argv)
    run_dir = Path(args.runs)
    figdir = run_dir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    grading = json.loads((run_dir / "grading.json").read_text())
    tags = [f"{g['grid'][0]}x{g['grid'][1]}" for g in grading["per_grid"]]
    made = []

    runs = {}
    for tag in tags:
        prim, cfg, meta = load_run(run_dir, tag)
        runs[tag] = (prim, cfg, meta)
    finest = tags[-1]
    prim, cfg, _ = runs[finest]
    xc, yc = cfg.centers()
    xs = xc.numpy()
    ys = yc.numpy()
    solid = cfg.solid_mask().numpy()

    # --- 1: density field -------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 3.5), dpi=140)
    rho = np.ma.masked_array(prim["rho"].numpy().T, mask=solid.T)
    levels = np.linspace(float(rho.min()), float(rho.max()), 41)
    cs = ax.contourf(xs, ys, rho, levels=levels, cmap="inferno", extend="both")
    ax.contour(xs, ys, rho, levels=levels[::4], colors="w", linewidths=0.3, alpha=0.55)
    ax.add_patch(plt.Rectangle((cfg.step_x, 0.0), cfg.lx - cfg.step_x, cfg.step_h,
                               facecolor="0.3", edgecolor="k", lw=0.6, zorder=3))
    ax.set_aspect("equal")
    ax.set_xlim(0, cfg.lx)
    ax.set_ylim(0, cfg.ly)
    ax.set_xlabel("x / tunnel height")
    ax.set_ylabel("y / tunnel height")
    ax.set_title(f"Mach 3 forward-facing step (Woodward & Colella 1984, problem 2), "
                 f"t = {cfg.t_end:g}, grid {finest}")
    fig.colorbar(cs, ax=ax, label=r"$\rho$")
    fig.tight_layout()
    path = figdir / "density.png"
    fig.savefig(path)
    plt.close(fig)
    made.append(path.name)

    # --- 2: stagnation-streamline profiles --------------------------------
    ref = normal_shock_reference(cfg.mach, cfg.gamma, cfg.rho, cfg.p)
    record = grading["record"]
    y_probe = record["shock_sampling"]["y"]
    j = int(np.argmin(np.abs(ys - y_probe)))
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2), dpi=140)
    for ax, key, refval, label in (
        (axes[0], "rho", ref["rho2"], r"$\rho$"),
        (axes[1], "p", ref["p2"], "p"),
        (axes[2], "u", ref["u2"], "u"),
    ):
        ax.plot(xs, prim[key][:, j].numpy(), "-", lw=1.1, color="C0",
                label=f"TensorFVM {finest}")
        ax.axhline(refval, color="crimson", ls="--", lw=1.0,
                   label="exact normal-shock state")
        ax.axvline(record["shock_sampling"]["x_shock"], color="k", ls=":", lw=1.0,
                   label=r"half-jump $x_s$")
        ax.set_xlim(0.15, cfg.step_x + 0.05)
        ax.set_xlabel("x")
        ax.set_title(f"{label} on y = {y_probe:.3g}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Stagnation-streamline profile vs the exact post-shock state")
    fig.tight_layout()
    path = figdir / "centreline.png"
    fig.savefig(path)
    plt.close(fig)
    made.append(path.name)

    # --- 3: convergence ---------------------------------------------------
    conv = grading["convergence"]["per_quantity"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    for key, entry in conv.items():
        dxs = [g["dx"] for g in grading["per_grid"]]
        errs = [abs(e) for e in entry["errors"]]
        ax.loglog(dxs, errs, "o-", lw=1.1, ms=4, label=LABELS.get(key, key))
    ax.axhline(grading["tolerance"], color="crimson", ls="--", lw=1.0,
               label=f"tolerance {grading['tolerance']:g}")
    ax.set_xlabel("dx")
    ax.set_ylabel("relative error vs exact state")
    ax.set_title("Grid convergence of the graded quantities")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    path = figdir / "convergence.png"
    fig.savefig(path)
    plt.close(fig)
    made.append(path.name)

    # --- 4: shock transition zoom ----------------------------------------
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=140)
    x_s = record["shock_sampling"]["x_shock"]
    for tag in tags:
        p2, c2, _ = runs[tag]
        _, y2 = c2.centers()
        jj = int(np.argmin(np.abs(y2.numpy() - c2.step_h / 2.0)))
        xv = c2.centers()[0].numpy()
        m = (xv > x_s - 0.12) & (xv < x_s + 0.12)
        ax.plot(xv[m], p2["rho"][:, jj].numpy()[m], "o-", ms=3, lw=1.0,
                label=f"{tag} (dx = {c2.dx:.4g})")
    ax.axhline(ref["rho2"], color="crimson", ls="--", lw=1.0, label=r"exact $\rho_2$")
    ax.axhline(cfg.rho, color="grey", ls=":", lw=1.0, label=r"$\rho_1$")
    ax.axvline(x_s, color="k", ls=":", lw=1.0)
    ax.set_xlabel("x")
    ax.set_ylabel(r"$\rho$ on the stagnation streamline")
    ax.set_title("Numerical shock transition: the graded sample must sit behind it")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = figdir / "shock_zoom.png"
    fig.savefig(path)
    plt.close(fig)
    made.append(path.name)

    (figdir / "manifest.json").write_text(json.dumps(
        {"figures": made, "finest_grid": finest, "t_end": cfg.t_end}, indent=1))
    print("\n".join(str(figdir / name) for name in made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())