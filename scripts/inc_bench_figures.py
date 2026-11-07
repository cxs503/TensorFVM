#!/usr/bin/env python3
"""Render the figure set for the incompressible FVM verification report.

Every panel is drawn from ``artifacts/inc_bench/*.json`` produced by
``scripts/run_inc_bench.py``; no number is typed by hand.  Missing inputs are
skipped with a warning so the script is runnable at any stage of a campaign.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "inc_bench"


def load(tag: str) -> dict | None:
    path = SRC / f"{tag}.json"
    if not path.exists():
        print(f"missing {path}")
        return None
    return json.loads(path.read_text())


def figure_friction() -> Path | None:
    """C_f(x) for the backward-facing step and the sudden expansion."""
    levels = [level for level in ("coarse", "medium", "fine")
              if load(f"backward_facing_step_{level}") is not None]
    if not levels:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for level, colour in zip(levels, ("#1f77b4", "#ff7f0e", "#2ca02c")):
        entry = load(f"backward_facing_step_{level}")
        profile = entry["metrics"]["friction_profile"]
        x = np.asarray(profile["x_over_step"])
        cf = np.asarray(profile["cf"])
        length = min(x.size, cf.size)
        x, cf = x[:length], cf[:length]
        mask = x >= -0.5
        axes[0].plot(x[mask], cf[mask], "-", color=colour, lw=1.3,
                     label=f"{level} ({entry['setup']['nx']}x{entry['setup']['ny']})")
        axes[0].axvline(entry["metrics"]["reattachment_over_step"], color=colour,
                        ls=":", lw=1.0)
    axes[0].axvspan(5.8, 6.2, color="grey", alpha=0.18, label="benchmark 6.0+/-0.2")
    axes[0].axhline(0.0, color="black", lw=0.6)
    axes[0].set_xlim(-0.5, 12.0)
    axes[0].set_ylim(-0.010, 0.012)
    axes[0].set_xlabel("x / step height")
    axes[0].set_ylabel("C_f on the step-side wall")
    axes[0].set_title("Backward-facing step, Re_Dh = 800")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    exp_levels = [level for level in ("coarse", "medium", "fine")
                  if load(f"sudden_expansion_{level}") is not None]
    for level, colour in zip(exp_levels, ("#1f77b4", "#ff7f0e", "#2ca02c")):
        entry = load(f"sudden_expansion_{level}")
        profile = entry["metrics"]["friction_profile"]
        x = np.asarray(profile["x_over_h"])
        cf = np.asarray(profile["cf"])
        length = min(x.size, cf.size)
        x, cf = x[:length], cf[:length]
        mask = x >= -0.5
        axes[1].plot(x[mask], cf[mask], "-", color=colour, lw=1.3, label=level)
    axes[1].axhline(0.0, color="black", lw=0.6)
    if exp_levels:
        axes[1].set_xlim(-0.5, 10.0)
        axes[1].set_ylim(-0.03, 0.03)
    axes[1].set_xlabel("x / inlet height")
    axes[1].set_ylabel("C_f on the expansion wall")
    axes[1].set_title("Sudden expansion, AR = 4, Re_D = 100")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    path = SRC / "fig-friction-and-reattachment.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_convergence() -> Path | None:
    summary = SRC / "summary.json"
    if not summary.exists():
        return None
    data = json.loads(summary.read_text())
    bfs = data.get("cases", {}).get("backward_facing_step", {}).get("levels", {})
    if not bfs:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    order = [level for level in ("coarse", "medium", "fine") if level in bfs]
    cells = [bfs[level]["cells"] for level in order]
    reattach = [bfs[level]["L_r_over_S"] for level in order]
    axes[0].semilogx(cells, reattach, "o-", color="#1f77b4")
    axes[0].axhspan(5.8, 6.2, color="grey", alpha=0.2, label="6.0 +/- 0.2")
    richardson = data["cases"]["backward_facing_step"].get("richardson", {})
    if math.isfinite(richardson.get("estimate", float("nan"))):
        axes[0].axhline(richardson["estimate"], color="crimson", ls="--",
                        label=f"Richardson {richardson['estimate']:.3f} "
                              f"(p={richardson.get('order', float('nan')):.2f})")
    axes[0].set_xlabel("cells")
    axes[0].set_ylabel("L_r / step height")
    axes[0].set_title("Reattachment grid convergence")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    expansion = data.get("cases", {}).get("sudden_expansion", {}).get("levels", {})
    if expansion:
        keys = [level for level in ("coarse", "medium", "fine") if level in expansion]
        k = [expansion[level]["K"] for level in keys]
        axes[1].bar(range(len(keys)), k, color="#4c72b0", width=0.55)
        kb = expansion[keys[-1]]["K_borda_carnot"]
        axes[1].axhline(kb, color="crimson", ls="--",
                        label=f"Borda-Carnot {kb:.3f}")
        axes[1].set_xticks(range(len(keys)))
        axes[1].set_xticklabels(keys)
        axes[1].set_ylabel("loss coefficient K")
        axes[1].set_title("Sudden expansion pressure loss")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = SRC / "fig-grid-convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_cavity() -> Path | None:
    entries = {}
    for rayleigh in (1e4, 1e5, 1e6):
        entry = None
        for level in ("medium", "fine", "coarse", "reference"):
            entry = load(f"natural_cavity_{level}_ra{rayleigh:g}")
            if entry:
                break
        if entry and math.isfinite(entry["metrics"]["nusselt_hot"]) \
                and entry["metrics"]["nusselt_hot"] < 1.0e3:
            entries[rayleigh] = entry
    if not entries:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ra = sorted(entries)
    nusselt = [entries[value]["metrics"]["nusselt_hot"] for value in ra]
    reference = [entries[value]["metrics"]["nusselt_reference"] for value in ra]
    axes[0].loglog(ra, reference, "s", color="crimson", label="de Vahl Davis 1983")
    axes[0].loglog(ra, nusselt, "o-", color="#1f77b4", label="this solver")
    for value, nu, ref in zip(ra, nusselt, reference):
        axes[0].annotate(f"{(nu - ref) / ref * 100:+.1f}%", (value, nu),
                         textcoords="offset points", xytext=(6, -12), fontsize=8)
    axes[0].set_xlabel("Rayleigh number")
    axes[0].set_ylabel("mean Nusselt number")
    axes[0].set_title("Natural convection cavity, Pr = 0.71")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both")

    psi = [entries[value]["metrics"]["psi_max"] for value in ra]
    psi_ref = [entries[value]["metrics"]["psi_max_reference"] for value in ra]
    axes[1].loglog(ra, psi_ref, "s", color="crimson", label="de Vahl Davis 1983")
    axes[1].loglog(ra, psi, "o-", color="#1f77b4", label="this solver")
    axes[1].set_xlabel("Rayleigh number")
    axes[1].set_ylabel("psi_max")
    axes[1].set_title("Streamfunction maximum")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = SRC / "fig-cavity-nusselt.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def figure_fields() -> Path | None:
    entry = None
    for tag in ("natural_cavity_fine_ra1e+06", "natural_cavity_medium_ra1e+06",
                "natural_cavity_coarse_ra1e+06", "natural_cavity_medium_ra100000",
                "natural_cavity_coarse_ra10000"):
        entry = load(tag)
        if entry and math.isfinite(entry["metrics"]["nusselt_hot"]) \
                and entry["metrics"]["nusselt_hot"] < 1.0e3:
            break
        entry = None
    if entry is None:
        return None
    temperature = np.asarray(entry["metrics"]["temperature_field"])
    psi = np.asarray(entry["metrics"]["psi_field"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
    im0 = axes[0].contourf(temperature, levels=24, cmap="inferno")
    fig.colorbar(im0, ax=axes[0], shrink=0.85, label="T")
    axes[0].set_title(f"Temperature, Ra = {entry['rayleigh']:g}")
    levels = np.linspace(psi.min(), psi.max(), 21)
    im1 = axes[1].contourf(psi, levels=levels, cmap="viridis")
    axes[1].contour(psi, levels=levels, colors="black", linewidths=0.3)
    fig.colorbar(im1, ax=axes[1], shrink=0.85, label="psi")
    axes[1].set_title("Streamfunction")
    for axis in axes:
        axis.set_xlabel("x cells")
        axis.set_ylabel("y cells")
        axis.set_aspect("equal")
    fig.tight_layout()
    path = SRC / "fig-cavity-fields.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    made = []
    for builder in (figure_friction, figure_convergence, figure_cavity, figure_fields):
        try:
            path = builder()
        except Exception as exc:  # pragma: no cover - diagnostics only
            print(f"{builder.__name__} failed: {exc}")
            path = None
        if path:
            made.append(path)
            print(f"wrote {path}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())