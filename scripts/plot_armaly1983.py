#!/usr/bin/env python3
"""Figures for the Armaly (1983) backward-facing-step reproduction.

Reads the per-run JSON artifacts written by ``scripts/repro_armaly1983_bfs.py``
and draws, from the *same* fields whose numbers are quoted in the report:

* ``fig_reattachment.png``  -- L_r/S versus Re_Dh against the experimental anchors
* ``fig_profile.png``       -- outlet velocity profile versus the Poiseuille parabola
* ``fig_development.png``   -- development of the profile deviation with x
* ``fig_field.png``          -- streamwise velocity contour + streamlines
* ``fig_friction.png``       -- wall friction C_f(x) with the reattachment crossing

    python scripts/plot_armaly1983.py --runs runs/armaly_local runs/armaly1983 \
        --out docs/repro-armaly1983-bfs/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(dirs: list[str]) -> list[dict]:
    runs: dict[tuple, dict] = {}
    for directory in dirs:
        for path in sorted(glob.glob(str(Path(directory) / "armaly1983_re*.json"))):
            record = json.loads(Path(path).read_text())
            key = (record["reynolds_dh"], record["level"],
                   record["downstream_step_heights"])
            runs[key] = record          # later directories win
    return [runs[key] for key in sorted(runs)]


def field_grid(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    field = record["field"]
    nx, ny = field["shape"]
    index = np.array(field["fluid_index"], dtype=int)
    rows, cols = np.unravel_index(index, (ny, nx))
    u = np.full((ny, nx), np.nan)
    u[rows, cols] = np.array(field["u"], dtype=float)
    x = np.array(field["x"], dtype=float)
    y = np.array(field["y"], dtype=float)
    return x, y, u


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["runs/armaly_local"])
    parser.add_argument("--out", default="docs/repro-armaly1983-bfs/figures")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    records = load(args.runs)
    if not records:
        raise SystemExit(f"no armaly artifacts found in {args.runs}")
    made: list[str] = []

    # 1 -- reattachment length versus Reynolds number
    fig, ax = plt.subplots(figsize=(5.4, 4.0))
    anchors = sorted({(r["reynolds_dh"],
                       r["experiment"]["reattachment_over_step"],
                       r["experiment"]["band_over_step"])
                      for r in records if r.get("experiment")})
    if anchors:
        re_a = [a[0] for a in anchors]
        lr_a = [a[1] for a in anchors]
        band = [a[2] for a in anchors]
        ax.errorbar(re_a, lr_a, yerr=band, fmt="s", color="k", ms=7, capsize=4,
                    label="Armaly et al. (1983) experiment")
    for level in sorted({r["level"] for r in records}):
        rows = [r for r in records if r["level"] == level
                and r["downstream_step_heights"] == min(
                    x["downstream_step_heights"] for x in records)]
        rows.sort(key=lambda r: r["reynolds_dh"])
        ax.plot([r["reynolds_dh"] for r in rows],
                [r["reattachment_over_step"] for r in rows], "o-",
                label=f"TensorFVM inc_bench ({level})")
    for r in records:
        if r.get("reattachment_rel_error") is not None:
            ax.annotate(f"{100 * r['reattachment_rel_error']:.1f}%",
                        (r["reynolds_dh"], r["reattachment_over_step"]),
                        textcoords="offset points", xytext=(6, 6), fontsize=7)
    ax.set_xlabel(r"$Re_{D_h} = U\,2H_{in}/\nu$")
    ax.set_ylabel(r"$L_r / S$")
    ax.set_title("Backward-facing step reattachment length")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = out / "fig_reattachment.png"
    fig.tight_layout(); fig.savefig(path, dpi=160); plt.close(fig)
    made.append(path.name)

    # 2 -- outlet velocity profile versus the parabola
    outlet = [r for r in records
              if r["downstream_step_heights"] == max(x["downstream_step_heights"]
                                                     for x in records)]
    fig, axes = plt.subplots(1, len(outlet), figsize=(4.2 * len(outlet), 3.8),
                             squeeze=False)
    for ax, record in zip(axes[0], outlet):
        profile = record["profile"]
        y = np.array(profile["y"]); u = np.array(profile["u"])
        fluid = np.array(profile["fluid"], dtype=bool)
        metric = record["profile_outlet_column"]
        eta = y / metric["height"]
        ax.plot(u[fluid], y[fluid], "o-", ms=3, label="TensorFVM outlet")
        ax.plot(6 * metric["bulk_velocity"] * eta * (1 - eta), y,
                "k--", lw=1.2, label="plane Poiseuille (same flux)")
        ax.set_title(f"$Re_{{D_h}}$={record['reynolds_dh']:.0f}, "
                     f"max dev {100 * metric['max_rel']:.2f}%", fontsize=9)
        ax.set_xlabel(r"$u/U_{in}$"); ax.set_ylabel(r"$y/S$")
        ax.grid(alpha=0.3); ax.legend(fontsize=7)
    path3 = out / "fig_profile.png"
    fig.tight_layout(); fig.savefig(path3, dpi=160); plt.close(fig)
    made.append(path3.name)

    # 3 -- development of the deviation with streamwise position
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for record in records:
        if record["downstream_step_heights"] != max(
                x["downstream_step_heights"] for x in records):
            continue
        stations = record["profile_stations"]
        xs = [s["x_over_step"] for s in stations]
        dev = [100 * s["max_rel"] for s in stations]
        ax.plot(xs, dev, "o-", label=f"$Re_{{D_h}}$={record['reynolds_dh']:.0f}")
    ax.axhline(5.0, color="r", ls=":", lw=1.2, label="5 % gate")
    ax.set_xlabel(r"$x/S$ (from the step plane)")
    ax.set_ylabel("max profile deviation [%]")
    ax.set_title("Development toward the Poiseuille parabola")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    dev_path = out / "fig_development.png"
    fig.tight_layout(); fig.savefig(dev_path, dpi=160); plt.close(fig)
    made.append(dev_path.name)

    # 4 -- field contour + streamlines (largest domain, one panel per Re)
    cloud = [r for r in records
             if r["downstream_step_heights"] == max(x["downstream_step_heights"]
                                                    for x in records)]
    fig, axes = plt.subplots(len(cloud), 1, figsize=(7.6, 2.4 * len(cloud)),
                             squeeze=False)
    for ax, record in zip(axes[:, 0], cloud):
        x, y, u = field_grid(record)
        im = ax.pcolormesh(x, y, u, shading="auto", cmap="viridis")
        field = record["field"]
        psi = np.full_like(u, np.nan)
        index = np.array(field["fluid_index"], dtype=int)
        rows, cols = np.unravel_index(index, u.shape)
        psi[rows, cols] = np.array(field["psi"], dtype=float)
        if np.isfinite(psi).any():
            ax.contour(x, y, np.nan_to_num(psi), levels=12, colors="w", linewidths=0.5)
        ax.set_title(f"$u/U_{{in}}$, $Re_{{D_h}}$={record['reynolds_dh']:.0f} "
                     f"({record['level']}, $L_{{down}}$="
                     f"{record['downstream_step_heights']:.0f}$S$)", fontsize=9)
        ax.set_xlabel(r"$x/S$"); ax.set_ylabel(r"$y/S$")
        fig.colorbar(im, ax=ax, pad=0.01)
    fld_path = out / "fig_field.png"
    fig.tight_layout(); fig.savefig(fld_path, dpi=160); plt.close(fig)
    made.append(fld_path.name)

    # 5 -- wall friction and the reattachment crossing
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    for record in records:
        friction = record["friction_profile"]
        ax.plot(friction["x_over_step"], friction["cf"], lw=1.2,
                label=f"$Re_{{D_h}}$={record['reynolds_dh']:.0f} ({record['level']})")
        lr = record["reattachment_over_step"]
        if math.isfinite(lr):
            ax.plot([lr], [0.0], "k*", ms=10)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlim(0, min(20, max(max(r["friction_profile"]["x_over_step"][-1:]) for r in records)))
    ax.set_xlabel(r"$x/S$"); ax.set_ylabel(r"$C_f$ on the lower wall")
    ax.set_title("Wall friction and the reattachment point (star)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    cf_path = out / "fig_friction.png"
    fig.tight_layout(); fig.savefig(cf_path, dpi=160); plt.close(fig)
    made.append(cf_path.name)

    print("\n".join(made))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())