#!/usr/bin/env python3
"""PEAK-ANOMALY 图（由 peak_anomaly.py 产出的 json + 存量剖面渲染）."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "woodward-colella-double-blast" / "ladder" / "data"
REF_CSV = HERE.parent / "woodward-colella-double-blast" / "reference" / "reference_density_calibrated.csv"
FIG = HERE / "figures"


def load(tag: str):
    d = np.load(DATA / f"ladder_profile_{tag}.npz")
    return d["x"], d["rho"]


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    res = json.loads((HERE / "peak_anomaly.json").read_text())
    fvm = {r["tag"]: r for r in res["fvm_rungs"]}
    w5 = {r["tag"]: r for r in res["weno5_rungs"]}
    w7 = {r["tag"]: r for r in res["weno7_rungs"]}

    # ---- 图 1: 峰值/平台序列 + 剖面放大 + 守恒/正性 ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    fig.suptitle("XVAL-FVM-M18737-ROUND2 — peak anomaly (nx=3200→6400 drop) + nx=6400 conservation/positivity",
                 fontsize=11)

    ax = axes[0, 0]
    fvm_tags = ["nx1600_hllc", "nx3200_hllc", "nx6400_hllc"]
    w5_tags = ["nx3200_weno5", "nx6400_weno5", "nx12800_weno5"]
    ax.semilogy([1600, 3200, 6400], [fvm[t]["peak"] for t in fvm_tags], "o-", label="MUSCL/HLLC peak")
    ax.semilogy([3200, 6400, 12800], [w5[t]["peak"] for t in w5_tags], "s-", label="WENO5 peak (cross-code)")
    ax.semilogy([400, 1600], [w7[t]["peak"] for t in ("nx400_weno7", "nx1600_weno7")], "^--", label="WENO7 peak")
    ax.axhline(res["A6_reference"]["ref_peak"], color="k", ls=":", lw=1,
               label=f"ref (RPJ@3200) peak={res['A6_reference']['ref_peak']:.4f}")
    ax.set_xlabel("nx")
    ax.set_ylabel("density peak")
    ax.set_title("(a) peak density vs resolution (log y)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for t, mk in [("nx1600_hllc", "o"), ("nx3200_hllc", "s"), ("nx6400_hllc", "^")]:
        x, rho = load(t)
        m = (x > 0.62) & (x < 0.92)
        ax.plot(x[m], rho[m], mk + "-", ms=3, lw=0.8, label=f"FVM {t}")
    for t, mk in [("nx6400_weno5", "D"), ("nx12800_weno5", "v")]:
        x, rho = load(t)
        m = (x > 0.62) & (x < 0.92)
        ax.plot(x[m], rho[m], mk + "-", ms=3, lw=0.8, label=f"WENO5 {t}")
    ax.set_xlabel("x")
    ax.set_ylabel("rho")
    ax.set_title("(b) density plateau zoom x∈[0.62,0.92]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    cons = res["A4_conservation"]["all_rungs"]
    tags = list(cons.keys())
    nx = [int(t[2:].split("_")[0]) for t in tags]
    ax.semilogy(nx, [abs(cons[t]["mass_trap_rel"]) for t in tags], "o-", label="|ΔM| trap")
    ax.semilogy(nx, [abs(cons[t]["energy_trap_rel"]) for t in tags], "s-", label="|ΔE|/E0 trap")
    ax.axhline(1e-4, color="r", ls=":", lw=1, label="1e-4 gate")
    ax.set_xlabel("nx")
    ax.set_ylabel("relative error")
    ax.set_title("(c) conservation residuals (trapezoid on cell averages)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.semilogy(nx, [cons[t]["min_rho"] for t in tags], "o-", label="min rho")
    ax.semilogy(nx, [cons[t]["min_p"] for t in tags], "s-", label="min p")
    ax.axhline(0.1, color="r", ls=":", lw=1, label="rho gate 0.1")
    ax.axhline(1.0, color="orange", ls=":", lw=1, label="p gate 1.0")
    ax.set_xlabel("nx")
    ax.set_ylabel("value")
    ax.set_title("(d) positivity margins")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    p1 = FIG / "fig_peak_anomaly_summary.png"
    fig.savefig(p1, dpi=140)
    plt.close(fig)

    # ---- 图 2: 平台统计对比（max vs plateau mean，FVM vs WENO5）----
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 4.5))
    ax.plot([1600, 3200, 6400], [fvm[t]["peak"] for t in fvm_tags], "o-", label="FVM max")
    ax.plot([1600, 3200, 6400], [fvm[t]["plateau_mean"] for t in fvm_tags], "o--", label="FVM plateau mean")
    ax.plot([3200, 6400, 12800], [w5[t]["peak"] for t in w5_tags], "s-", label="WENO5 max")
    ax.plot([3200, 6400, 12800], [w5[t]["plateau_mean"] for t in w5_tags], "s--", label="WENO5 plateau mean")
    ax.axhline(res["A6_reference"]["ref_peak"], color="k", ls=":", lw=1, label="ref peak")
    ax.set_xlabel("nx")
    ax.set_ylabel("rho")
    ax.set_title("peak vs plateau mean (non-monotonicity is in the level, not the max)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = FIG / "fig_peak_anomaly_plateau.png"
    fig.savefig(p2, dpi=140)
    plt.close(fig)
    print("wrote", p1)
    print("wrote", p2)


if __name__ == "__main__":
    main()
