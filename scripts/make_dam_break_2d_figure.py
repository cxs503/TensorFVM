#!/usr/bin/env python
"""Generate the four-panel figure for the 2-D dam-break confirmation report.

Produces ``docs/case-fvm-dambreak-stoker-2d/dam_break_2d.png`` from the
committed artifacts (zero hand-typed numbers).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from tensorfvm.shallow_water.exact import ritter_state, stoker_state, t_star_to_time

ARTIFACTS = os.path.join(REPO, "src", "tensorfvm", "verification",
                         "artifacts", "shallow-water-dam-break-2d")
OUT = os.path.join(REPO, "docs", "case-fvm-dambreak-stoker-2d", "dam_break_2d.png")

G = 9.81
H0 = 1.0
H1_WET = 0.5
T_STAR = 2.0


def _load_profile(key: str):
    path = os.path.join(ARTIFACTS, f"profile_2d_{key}_nx800.csv")
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1]


def _exact_profile(key: str, x):
    t = t_star_to_time(T_STAR, H0, G)
    if key == "ritter-dry":
        h, _ = ritter_state(x, t, H0, G, 0.0)
    else:
        h, _ = stoker_state(x, t, H0, H1_WET, G, 0.0)
    return h


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # ---- Panel 1: Ritter dry profile ----
    ax = axes[0, 0]
    x, h_num = _load_profile("ritter-dry")
    h_ex = _exact_profile("ritter-dry", x)
    ax.plot(x, h_ex, "k-", lw=1.5, label="Exact (Ritter 1892)")
    ax.plot(x, h_num, "b.", ms=3, label="2-D SWE (nx=800, y-avg)")
    ax.axvline(4.0, color="r", ls="--", lw=1, label="x_f = 4.0 m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h [m]")
    ax.set_title("Ritter dry-bed: depth profile at t* = 2")
    ax.legend(fontsize=7)
    ax.set_xlim(-1, 5)
    ax.set_ylim(-0.05, 1.1)

    # ---- Panel 2: Stoker wet profile ----
    ax = axes[0, 1]
    x, h_num = _load_profile("stoker-wet")
    h_ex = _exact_profile("stoker-wet", x)
    ax.plot(x, h_ex, "k-", lw=1.5, label="Exact (Stoker 1957)")
    ax.plot(x, h_num, "b.", ms=3, label="2-D SWE (nx=800, y-avg)")
    with open(os.path.join(ARTIFACTS, "metrics_2d.json")) as f:
        m = json.load(f)
    shock_x = m["stoker-wet"]["metrics"]["bore_x"]
    ax.axvline(shock_x, color="r", ls="--", lw=1, label=f"x_s = {shock_x:.4f} m")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("h [m]")
    ax.set_title("Stoker wet-bed: depth profile at t* = 2")
    ax.legend(fontsize=7)
    ax.set_xlim(-1, 3.5)
    ax.set_ylim(0.4, 1.1)

    # ---- Panel 3: Convergence ----
    ax = axes[1, 0]
    with open(os.path.join(ARTIFACTS, "convergence_2d.json")) as f:
        conv = json.load(f)
    for key, label, color in [("ritter-dry", "Ritter dry", "blue"),
                               ("stoker-wet", "Stoker wet", "red")]:
        rows = conv[key]["rows"]
        nx = [r["nx"] for r in rows]
        l1 = [r["depth_l1"] for r in rows]
        ax.loglog(nx, l1, "o-", color=color, label=f"{label} L1")
    nx_ref = np.array([200, 800])
    ax.loglog(nx_ref, [0.01, 0.0025], "k--", lw=1, label="1st order")
    ax.set_xlabel("nx")
    ax.set_ylabel("depth L1 error")
    ax.set_title("Grid convergence (ny=8 fixed)")
    ax.legend(fontsize=7)

    # ---- Panel 4: Transverse defect ----
    ax = axes[1, 1]
    with open(os.path.join(ARTIFACTS, "modulation_2d.json")) as f:
        mod = json.load(f)
    for key, label, color in [("ritter-dry", "Ritter dry", "blue"),
                               ("stoker-wet", "Stoker wet", "red")]:
        series = mod[key]["series"]
        t_stars = series["t_star"]
        defects = series["modulated_defect"]
        ax.plot(t_stars, defects, "o-", color=color, ms=3, label=f"{key} modulated")
    ax.axhline(1e-2, color="orange", ls="--", lw=1, label="modulation amplitude (1%)")
    ax.set_xlabel("t*")
    ax.set_ylabel("transverse defect / h0")
    ax.set_title("Transverse modulation decay (nx=400)")
    ax.legend(fontsize=7)
    ax.set_yscale("log")

    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
