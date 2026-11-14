"""Figure for the asymptotic-range study of CASE-WOODWARD-COLELLA.

Three panels, all built from committed artefacts (no solver call):

(a) density profile on the comparison window at nx = 800/1600/3200 together with
    the calibrated digitised reference and the p = 2 Richardson-extrapolated
    field - the visual statement of "the grids converge *past* the reference";
(b) zoom on the collision peak, where the residual lives;
(c) the distance ladder: successive-grid distances, distance to the reference at
    the two extreme grids, and the extrapolated-field distance, against the 10 %
    acceptance line.

Usage::

    PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_asymptotic_figure.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import tensorfvm.compressible.wc_reference as wcr  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUT = HERE.parent / "artifacts" / "fig_wc_asymptotic_nx3200.png"


def _load_analysis():
    spec = importlib.util.spec_from_file_location("wc_asym_fig", HERE / "wc_asymptotic_analysis.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["wc_asym_fig"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ana = _load_analysis()
    res = json.loads((DATA / "asymptotic_analysis.json").read_text())
    profiles = ana.load_profiles()
    tags = {p["nx"]: t for t, p in profiles.items() if p["family"] == "hllc"}
    ref = wcr.load_reference(ana.CALIBRATED)
    lo, hi = wcr.REFERENCE_X

    ext = ana.extrapolate_field(profiles[tags[3200]], profiles[tags[1600]], 2.0)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6))

    ax = axes[0]
    ax.plot(ref["x"], ref["rho"], "k-", lw=2.0, label="digitised reference (calibrated, RPJ@3200)")
    for nx in (800, 1600, 3200):
        if nx in tags:
            p = profiles[tags[nx]]
            ax.plot(p["x"], p["rho"], lw=1.0, alpha=0.85,
                    label=f"TensorFVM MUSCL/HLLC nx={nx}")
    ax.plot(ext["x"], ext["rho"], "g--", lw=1.4, label="Richardson-extrapolated field (p=2)")
    ax.set_xlim(lo, hi)
    ax.set_xlabel("x")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title("(a) comparison window")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    xw, rw = ana.window_mask(ref["x"], ref["rho"])
    m = (xw >= 0.70) & (xw <= 0.86)
    ax.plot(xw[m], rw[m], "k-", lw=2.2, label="reference (calibrated)")
    for nx in (800, 3200):
        p = profiles[tags[nx]]
        xx, rr = ana.window_mask(p["x"], p["rho"])
        mm = (xx >= 0.70) & (xx <= 0.86)
        ax.plot(xx[mm], rr[mm], lw=1.1, label=f"nx={nx}")
    xx, rr = ana.window_mask(ext["x"], ext["rho"])
    mm = (xx >= 0.70) & (xx <= 0.86)
    ax.plot(xx[mm], rr[mm], "g--", lw=1.4, label="extrapolated (p=2)")
    budget = res["families"]["hllc"]["uncertainty_budget"]["peak_density"]
    ax.set_xlabel("x")
    ax.set_ylabel(r"density $\rho$")
    ax.set_title(f"(b) collision peak: reference {budget['reference']:.3f} vs "
                 f"nx3200 {profiles[tags[3200]]['rho'][(profiles[tags[3200]]['x'] > lo)].max():.3f}")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    d = res["families"]["hllc"]["field_distance_successive"]
    ef = res["families"]["hllc"]["extrapolated_field"]
    sweep = {row["order_used"]: row for row in ef["order_sweep"]}
    labels = [f"D({d[0]['grids'][0]},{d[0]['grids'][1]})",
              f"D({d[1]['grids'][0]},{d[1]['grids'][1]})",
              "nx800→ref",
              "nx3200→ref",
              "extrap(p=2)→ref"]
    vals = [d[0]["percent"], d[1]["percent"],
            ef["distance_nx1600_to_calibrated_percent"],  # nx1600→ref (finest pair use)
            ef["distance_nx3200_to_calibrated_percent"],
            sweep[2.0]["distance_to_calibrated_reference_percent"]]
    vals[2] = ef["distance_nx1600_to_calibrated_percent"]
    colors = ["#4477aa", "#4477aa", "#cc6677", "#cc6677", "#cc6677"]
    bars = ax.bar(range(len(vals)), vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.08, f"{v:.2f}%", ha="center", fontsize=8)
    ax.axhline(wcr.TOLERANCE_PERCENT, color="k", ls=":", lw=1.2, label="10 % acceptance")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("relative L1 distance [%]")
    ax.set_title("(c) distance ladder (blue = grid-to-grid, red = vs reference)")
    ax.set_ylim(0, max(vals) * 1.35)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("CASE-WOODWARD-COLELLA — asymptotic-range verification at nx=3200 "
                 "(two interacting blast waves, t=0.038)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    print(f"figure -> {OUT}")


if __name__ == "__main__":
    main()