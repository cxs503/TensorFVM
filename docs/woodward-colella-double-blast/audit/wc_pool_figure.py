"""Figures for the ``CASE-WOODWARD-COLELLA`` case-pool registration.

Four panels, all drawn from committed artifacts (no solver call, no hand-typed
number):

(a) density profile at the pool grid against both reference tables,
(b) peak-density error over the committed ladder vs the 10 % tolerance,
(c) the metric that the ground-truth calibration moved: window mass integral
    and near-vacuum trough, biased vs calibrated reference,
(d) the margin of every criterion at the pool grid (value / threshold).

    PYTHONPATH=src python docs/woodward-colella-double-blast/audit/wc_pool_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))
from tensorfvm.compressible import wc_case_pool as pool  # noqa: E402

OUT = Path(__file__).resolve().parent / "fig_wc_pool_acceptance.png"


def main() -> Path:
    cal, com = pool.load_ref("calibrated"), pool.load_ref("committed")
    prof = pool.load_profile(pool.POOL_GRID)
    grids = list(pool.GRID_STEPS)
    lad_cal = {r["nx"]: {c["criterion"]: c["value"] for c in r["criteria"]}
               for r in pool.ladder("calibrated")}
    lad_com = {r["nx"]: {c["criterion"]: c["value"] for c in r["criteria"]}
               for r in pool.ladder("committed")}
    eff = pool.calibration_effect()

    fig, ax = plt.subplots(2, 2, figsize=(13.2, 8.6))

    # (a) profile vs the two references
    a = ax[0, 0]
    w = (prof["x"] >= 0.4) & (prof["x"] <= 1.0)
    a.plot(prof["x"][w], prof["rho"][w], lw=1.4,
           label=f"TensorFVM nx={pool.POOL_GRID} (MUSCL/HLLC)")
    a.plot(cal["x"], cal["rho"], lw=1.0, ls="--", label="reference, calibrated (primary)")
    a.plot(com["x"], com["rho"], lw=1.0, ls=":", label="reference, original digitisation")
    a.set_xlabel("x"); a.set_ylabel(r"$\rho$")
    a.set_title("(a) pool grid vs published reference")
    a.legend(fontsize=8, loc="upper left")
    a.grid(alpha=0.3)

    # (b) peak-density error across the ladder, both references
    b = ax[0, 1]
    dx = [1.0 / n for n in grids]
    b.plot(dx, [lad_cal[n]["peak_density"] for n in grids], "o-",
           label="calibrated")
    b.plot(dx, [lad_com[n]["peak_density"] for n in grids], "s--",
           label="committed")
    b.axhline(pool.POOL_TOLERANCE, color="crimson", lw=1.2)
    b.text(dx[0], pool.POOL_TOLERANCE * 1.04, "ledger tolerance 10 %",
           color="crimson", fontsize=8, ha="left")
    for n in grids:
        b.annotate(str(n), (1.0 / n, lad_cal[n]["peak_density"]),
                   textcoords="offset points", xytext=(4, 6), fontsize=7)
    b.set_xscale("log"); b.set_xlabel(r"$\Delta x$  (labels: nx)")
    b.set_ylabel(r"$|\rho_{max}-\rho_{max}^{ref}|/\rho_{max}^{ref}$  [%]")
    b.set_title("(b) peak-density error: fails below nx=800 on both references")
    b.legend(fontsize=8); b.grid(alpha=0.3, which="both")

    # (c) the two bias-dominated metrics, biased vs calibrated
    c = ax[1, 0]
    idx = np.arange(len(grids))
    width = 0.2
    c.bar(idx - 1.5 * width, [lad_com[n]["window_mass"] for n in grids], width,
          label="window mass, original")
    c.bar(idx - 0.5 * width, [lad_cal[n]["window_mass"] for n in grids], width,
          label="window mass, calibrated")
    c.bar(idx + 0.5 * width, [lad_com[n]["low_plateau"] for n in grids], width,
          label="trough rho_min, original")
    c.bar(idx + 1.5 * width, [lad_cal[n]["low_plateau"] for n in grids], width,
          label="trough rho_min, calibrated")
    c.set_yscale("log")
    c.set_xticks(idx); c.set_xticklabels([f"nx={n}" for n in grids])
    c.set_ylabel("relative error [%]")
    c.set_title("(c) the 1.4 % / 13-14 % gaps were the +0.022017 reference bias")
    c.legend(fontsize=7, ncol=2); c.grid(alpha=0.3, axis="y")

    # (d) margin of each criterion at the pool grid
    d = ax[1, 1]
    entry = pool.case_pool_entry(pool.POOL_GRID)
    rows = [r for r in entry["criteria"]]
    labels = [r["criterion"] for r in rows]
    ratio = [r["value"] / r["threshold"] for r in rows]
    colors = ["tab:blue" if r["primary"] else "tab:gray" for r in rows]
    d.barh(labels, ratio, color=colors)
    d.axvline(1.0, color="crimson", lw=1.2)
    d.text(1.02, -0.4, "FAIL ->", color="crimson", fontsize=8)
    d.set_xlabel("value / threshold  (grey = informational, not gated)")
    d.set_title(f"(d) criterion margin at nx={pool.POOL_GRID}: {entry['verdict']}")
    d.grid(alpha=0.3, axis="x")
    for i, (r, v) in enumerate(zip(rows, ratio)):
        d.text(v + 0.02, i, f"{r['value']:.4g} {r['unit']}", va="center", fontsize=7)

    fig.suptitle("CASE-WOODWARD-COLELLA — case-pool acceptance (all numbers from committed artifacts)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT, dpi=140)
    print(OUT)
    return OUT


if __name__ == "__main__":
    main()