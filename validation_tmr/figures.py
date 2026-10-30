#!/usr/bin/env python3
"""Figures for the NASA-TMR validation report.

Every panel is produced from the artefacts in ``validation_tmr/runs`` plus the
frozen TMR reference files; nothing is copied from another mission.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from validation_tmr.postprocess import (backstep_metrics, flat_plate_metrics,  # noqa: E402
                                        read_tmr, read_tmr_zones)

REFERENCE = REPO / "validation_tmr/reference/tmr"
RUNS = REPO / "validation_tmr/runs"
FIGURES = REPO / "validation_tmr/figures"
VISCOSITY = dict(flatplate=2e-7, backstep=1.0 / 36000.0)
LEVELS = dict(flatplate=("35x25", "69x49", "137x97"), backstep=("5levdn", "4levdn", "3levdn"))
STYLE = dict(flatplate=("0.55", "0.25", "0.0"), backstep=("0.55", "0.25", "0.0"))
MARKERS = ("o", "s", "^")
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})


def load(case, label):
    folder = RUNS / case
    with np.load(folder / f"{label}-fields.npz") as handle:
        fields = {name: handle[name] for name in handle.files}
    with np.load(folder / f"{label}-wall.npz") as handle:
        wall = {name: handle[name] for name in handle.files}
    metrics = json.loads((folder / f"{label}-metrics.json").read_text())["metrics"]
    history = json.loads((folder / f"{label}-history.json").read_text())
    return fields, wall, metrics, history


def available(case):
    return [level for level in LEVELS[case]
            if (RUNS / case / f"{case}-{level}-metrics.json").exists()]


def figure_flatplate_cf(levels):
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    theory = read_tmr(str(REFERENCE / "FlatPlate_validation_cf_K-S.dat"))
    axes[0].plot(theory[:, 0], theory[:, 1], "k-", lw=1.4,
                 label="Karman-Schoenherr (TMR)")
    for index, level in enumerate(levels):
        _, _, metrics, _ = load("flatplate", f"flatplate-{level}")
        table = metrics["cf_table"]
        re_theta = [row["re_theta"] for row in table]
        axes[0].plot(re_theta, [row["cf"] for row in table], MARKERS[index] + "-",
                     ms=3.4, lw=1.0, color=STYLE["flatplate"][index % 3],
                     label=f"TensorFVM SA {level}")
        axes[1].semilogy(re_theta, [row["cf_relative_error"] * 100 for row in table],
                         MARKERS[index] + "-", ms=3.4, lw=1.0,
                         color=STYLE["flatplate"][index % 3], label=f"{level}")
    axes[0].set(xlabel=r"$Re_\theta$", ylabel=r"$C_f$",
                title="2DZP flat plate: skin friction vs $Re_\\theta$")
    axes[1].axhline(5.0, color="r", ls="--", lw=1.0, label="5 % gate")
    axes[1].axhline(15.0, color="r", ls=":", lw=1.0, label="15 % unacceptable")
    axes[1].set(xlabel=r"$Re_\theta$", ylabel="relative error [%]",
                title="Deviation from Karman-Schoenherr")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    figure.savefig(FIGURES / "fig01_flatplate_cf.png")
    plt.close(figure)


def figure_flatplate_uplus(levels):
    cols = read_tmr(str(REFERENCE / "FlatPlate_validation_u+y+.dat"))
    cols = cols[np.argsort(cols[:, 0])]
    figure, axis = plt.subplots(figsize=(4.8, 4.0))
    axis.semilogx(cols[:, 0], cols[:, 1], "k-", lw=1.4,
                  label="Coles law of the wall (TMR)")
    for index, level in enumerate(levels):
        _, _, metrics, _ = load("flatplate", f"flatplate-{level}")
        profile = metrics["u_plus_y_plus"]
        y_plus = np.array(profile["y_plus"])
        keep = y_plus > 0.5
        axis.semilogx(y_plus[keep], np.array(profile["u_plus"])[keep],
                      MARKERS[index] + "-", ms=3.0, lw=1.0,
                      color=STYLE["flatplate"][index % 3],
                      label=f"TensorFVM SA {level} "
                            f"($Re_\\theta$={metrics['profile_station']['re_theta']:.0f})")
    axis.set(xlabel=r"$y^+$", ylabel=r"$u^+$",
             title="2DZP flat plate: $u^+$ vs $y^+$ near $Re_\\theta=10^4$",
             xlim=(1, 3000))
    axis.legend(fontsize=7)
    figure.savefig(FIGURES / "fig02_flatplate_uplus.png")
    plt.close(figure)


def figure_flatplate_field(levels):
    level = levels[-1]
    fields, wall, metrics, _ = load("flatplate", f"flatplate-{level}")
    centers = fields["centers"]
    speed = fields["velocity"][:, 0]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    contour = axes[0].tricontourf(centers[:, 0], centers[:, 1], speed, levels=40,
                                  cmap="viridis")
    axes[0].set(xlabel="x/L", ylabel="y/L", ylim=(0, 0.12),
                title=f"Streamwise velocity, SA, grid {level}")
    figure.colorbar(contour, ax=axes[0], label="u/U")
    profile = metrics["profile"]
    axes[1].plot(profile["u_plus"], np.array(profile["y_plus"]), "o-", ms=2.5, lw=1.0,
                 label="TensorFVM SA")
    cols = read_tmr(str(REFERENCE / "FlatPlate_validation_u+y+.dat"))
    axes[1].plot(cols[:, 1], cols[:, 0], "k-", lw=1.2, label="Coles law")
    axes[1].set(xlabel=r"$u^+$", ylabel=r"$y^+$", yscale="log",
                title=f"$u^+$-$y^+$ at $Re_\\theta$="
                      f"{metrics['profile_station']['re_theta']:.0f}")
    axes[1].legend(fontsize=7)
    figure.savefig(FIGURES / "fig03_flatplate_field.png")
    plt.close(figure)


def figure_backstep_field(levels):
    level = levels[-1]
    fields, _, metrics, _ = load("backstep", f"backstep-{level}")
    centers = fields["centers"]
    speed = np.linalg.norm(fields["velocity"], axis=1)
    zoom = (centers[:, 0] > -6) & (centers[:, 0] < 22)
    figure, axes = plt.subplots(2, 1, figsize=(9.2, 5.6),
                                gridspec_kw=dict(height_ratios=[1.5, 1.0]))
    contour = axes[0].tricontourf(centers[zoom, 0], centers[zoom, 1], speed[zoom],
                                  levels=40, cmap="magma")
    axes[0].axvline(0.0, color="w", ls="--", lw=0.8)
    axes[0].axvline(metrics["reattachment"], color="c", lw=1.4,
                    label=f"computed $x_r/H$={metrics['reattachment']:.2f}")
    axes[0].axvline(metrics["reattachment_experiment"], color="lime", ls=":", lw=1.4,
                    label=f"experiment $x_r/H$={metrics['reattachment_experiment']:.2f}")
    axes[0].set(xlabel="x/H", ylabel="y/H", title=f"|U|/U_ref, SA, grid {level}")
    axes[0].legend(fontsize=7, loc="upper right")
    figure.colorbar(contour, ax=axes[0], label="|U|/U_ref")
    smooth = speed[zoom]
    axes[1].tricontourf(centers[zoom, 0], centers[zoom, 1],
                        fields["velocity"][zoom, 0], levels=40, cmap="RdBu_r",
                        vmin=-0.35, vmax=0.35)
    axes[1].set(xlabel="x/H", ylabel="y/H",
                title="Streamwise velocity (blue = reversed flow)")
    figure.savefig(FIGURES / "fig04_backstep_field.png")
    plt.close(figure)


def figure_backstep_wall(levels):
    exp_cf = read_tmr(str(REFERENCE / "Backstep_validation_cf.exp.dat"))
    zones = read_tmr_zones(str(REFERENCE / "Backstep_validation_cp.expnew.dat"))
    exp_cp = next(data for title, data in zones.items() if "bottom" in title)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    axes[0].errorbar(exp_cf[:, 0], exp_cf[:, 1] * 1e3, yerr=exp_cf[:, 2] * 1e3,
                     fmt="ks", ms=2.6, lw=0.8, label="Driver & Seegmiller (TMR)")
    axes[1].plot(exp_cp[:, 0], exp_cp[:, 1], "ks", ms=2.6,
                 label="Driver & Seegmiller (TMR)")
    for index, level in enumerate(levels):
        _, _, metrics, _ = load("backstep", f"backstep-{level}")
        x = np.array(metrics["cf"]["x"])
        cf = np.array(metrics["cf"]["cf"])
        keep = (x > -6) & (x < 22)
        axes[0].plot(x[keep], cf[keep] * 1e3, "-", lw=1.0,
                     color=STYLE["backstep"][index % 3], label=f"TensorFVM SA {level}")
        cp = np.array(metrics["cp"]["cp"])
        cx = np.array(metrics["cp"]["x"])
        keep = (cx > -6) & (cx < 45)
        axes[1].plot(cx[keep], cp[keep], "-", lw=1.0,
                     color=STYLE["backstep"][index % 3], label=f"{level}")
    axes[0].axhline(0.0, color="0.4", lw=0.7)
    axes[0].set(xlabel="x/H", ylabel=r"$C_f\times10^3$", ylim=(-2, 4),
                title="Bottom-wall skin friction")
    axes[1].set(xlabel="x/H", ylabel=r"$C_p$", title="Bottom-wall pressure coefficient")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    figure.savefig(FIGURES / "fig05_backstep_wall.png")
    plt.close(figure)


def figure_backstep_profiles(levels):
    zones = read_tmr_zones(str(REFERENCE / "Backstep_validation_profiles.exp.dat"))
    stations = [1.0, 4.0, 6.0, 10.0]
    figure, axes = plt.subplots(1, 4, figsize=(11.0, 3.2), sharey=True)
    for axis, station in zip(axes, stations):
        key = [title for title in zones if abs(float(title.split("=")[-1]) - station) < 1e-6]
        if key:
            data = zones[key[0]]
            axis.plot(data[:, 2], data[:, 1], "ks", ms=2.6, label="experiment")
        for index, level in enumerate(levels):
            _, _, metrics, _ = load("backstep", f"backstep-{level}")
            row = [item for item in metrics["profiles"]
                   if abs(item["station"] - station) < 1e-9]
            if not row:
                continue
            row = row[0]
            axis.plot(row["u_over_uref"], row["y"], "-", lw=1.0,
                      color=STYLE["backstep"][index % 3], label=level)
        axis.set(xlabel=r"$u/U_{ref}$", title=f"x/H = {station:g}", xlim=(-0.4, 1.25))
        axis.set_ylim(0, 9)
    axes[0].set_ylabel("y/H")
    axes[0].legend(fontsize=7)
    figure.suptitle("2DBFS velocity profiles (Driver & Seegmiller data)", y=1.02)
    figure.savefig(FIGURES / "fig06_backstep_profiles.png")
    plt.close(figure)


def figure_convergence():
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.5))
    for case in ("flatplate", "backstep"):
        levels = available(case)
        errors, cells = [], []
        for level in levels:
            _, _, metrics, _ = load(case, f"{case}-{level}")
            cells.append(len(json.loads((RUNS / case / f"{case}-{level}-metrics.json")
                                        .read_text())["state"] and
                             np.load(RUNS / case / f"{case}-{level}-fields.npz")["volumes"]))
            if case == "flatplate":
                errors.append(metrics["cf_relative_error_rms"])
            else:
                errors.append(metrics["reattachment_relative_error"])
        axes[0].loglog(cells, errors, "o-", lw=1.2,
                       label=("flat plate $C_f$ RMS" if case == "flatplate"
                              else "backstep $x_r$ error"))
    axes[0].set(xlabel="cells", ylabel="relative error",
                title="Error under grid refinement")
    axes[0].legend(fontsize=7)
    for level in available("flatplate"):
        history = json.loads((RUNS / "flatplate" / f"flatplate-{level}-history.json")
                             .read_text())
        axes[1].semilogy([row["iteration"] for row in history],
                         [row["momentum"] for row in history], lw=1.0,
                         label=f"flat plate {level} momentum")
    axes[1].set(xlabel="SIMPLE iteration", ylabel="normalised residual",
                title="Convergence histories")
    axes[1].legend(fontsize=6)
    figure.savefig(FIGURES / "fig07_convergence.png")
    plt.close(figure)


def main():
    global FIGURES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    FIGURES = args.out
    FIGURES.mkdir(parents=True, exist_ok=True)
    flat = available("flatplate")
    step = available("backstep")
    if flat:
        figure_flatplate_cf(flat)
        figure_flatplate_uplus(flat)
        figure_flatplate_field(flat)
    if step:
        figure_backstep_field(step)
        figure_backstep_wall(step)
        figure_backstep_profiles(step)
    if flat or step:
        figure_convergence()
    print(sorted(path.name for path in FIGURES.glob("*.png")))


if __name__ == "__main__":
    main()