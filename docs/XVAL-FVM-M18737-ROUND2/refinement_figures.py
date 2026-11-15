"""Figures + summary.json for the XVAL-FVM-M18737-ROUND2 refinement closure.

Produces (into ``docs/XVAL-FVM-M18737-ROUND2/figures``):

``fig_refine_l1_ladder.png``
    log--log L1 vs grid for the graded MUSCL/HLLC ladder, with the predeclared
    forecast band (PREDECLARATION-refine.md), the 1 % certified bar and the
    observed order.
``fig_refine_residual_attribution.png``
    Upper panel: window density profiles (graded rungs vs the WENO5@12800
    reference).  Lower panel: the *share of the L1 residual* per cell, so the
    question "is the residual a plateau offset or front-localised" is answered
    by the picture instead of by prose.

Run: PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/refinement_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification import wc_refine as wr

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
CERTIFIED = wr.CERTIFIED_PERCENT

#: the predeclared forecast, frozen in PREDECLARATION-refine.md before the
#: nx=6400/12800 graded rungs existed (p_measured = log2(e1600/e3200) = 0.8785)
FORECAST = {"nx_ref": 3200, "err_ref": 1.7685, "order": 0.8785,
            "band_order": (0.79, 1.05), "targets": (6400, 12800)}


def fig_ladder(verdict: dict) -> Path:
    rows = [r for r in verdict["ladder"] if r.get("measured")]
    nx = np.array([r["nx"] for r in rows], float)
    l1 = np.array([r["l1_percent"] for r in rows], float)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.loglog(nx, l1, "o-", lw=1.8, ms=7, color="#1f4e79", label="MUSCL/HLLC vs WENO5@12800 (measured)")

    fx = np.array([FORECAST["nx_ref"]] + list(FORECAST["targets"]), float)
    for p, style, lab in ((FORECAST["band_order"][0], ":", "forecast band p=0.79..1.05"),
                          (FORECAST["order"], "--", "forecast p=0.878 (frozen before run)"),
                          (FORECAST["band_order"][1], ":", None)):
        fy = FORECAST["err_ref"] * (fx / FORECAST["nx_ref"]) ** (-p)
        ax.loglog(fx, fy, style, color="0.45", lw=1.3, label=lab)

    ax.axhline(CERTIFIED, color="#c00000", lw=1.4, ls="-.")
    ax.text(nx.min(), CERTIFIED * 1.06, "certified-<1 % bar", color="#c00000", fontsize=8)

    for r in rows:
        ax.annotate(f'{r["l1_percent"]:.3f} %', (r["nx"], r["l1_percent"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color="#1f4e79")
    if verdict["orders"]:
        p = verdict["orders"][-1]["order"]
        ax.set_title(f"Graded-solution refinement (Woodward-Colella t=0.038, window x=0.4-1)\n"
                     f"observed order p = {p:.3f} on the last measured pair", fontsize=10)
    ax.set_xlabel("nx (cells)")
    ax.set_ylabel("L1 error vs independent WENO5@12800  [%]")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    out = FIG / "fig_refine_l1_ladder.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_residual(verdict: dict, ref_nx: int = 12800) -> Path:
    data = wr.default_data_dir()
    ref = wr.load_profile(data, ref_nx, "weno5")
    rows = [r for r in verdict["ladder"] if r.get("measured")]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.6, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.25, 1.0]})
    xr = ref["x"]
    m = (xr >= wr.WINDOW[0]) & (xr <= wr.WINDOW[1])
    ax0.plot(xr[m], ref["rho"][m], "-", color="0.25", lw=1.1, label=f"WENO5@{ref_nx} (reference)")
    colors = {1600: "#7f7f7f", 3200: "#1f4e79", 6400: "#2e7d32", 12800: "#b8860b"}
    for r in rows:
        prof = wr.load_profile(data, r["nx"], "hllc")
        xp = prof["x"]
        mp = (xp >= wr.WINDOW[0]) & (xp <= wr.WINDOW[1])
        ax0.plot(xp[mp], prof["rho"][mp], "-", lw=1.0,
                 color=colors.get(r["nx"], "C3"), alpha=0.85,
                 label=f'MUSCL/HLLC@{r["nx"]} (L1 {r["l1_percent"]:.3f} %)')

    fin = rows[-1]
    prof = wr.load_profile(data, fin["nx"], "hllc")
    dens = wr.residual_density(prof["rho"], ref["rho"])
    share_pct = 100.0 * dens["share"] / dens["share"].sum()
    ax1.fill_between(dens["x"], 0, share_pct, color="#c00000", alpha=0.55, lw=0)
    ax1.set_ylabel(f'share of L1 residual [% of total]\n(nx={fin["nx"]} vs WENO5@{ref_nx})', fontsize=8.5)
    ax1.grid(alpha=0.25)
    ax0.set_ylabel("density rho  [1]")
    ax0.set_yscale("log")
    ax0.grid(alpha=0.25, which="both")
    ax0.legend(fontsize=7.5, loc="upper left")
    ax0.set_title("Where the certified-<1 % residual actually lives (window x=0.4-1, t=0.038)", fontsize=10)
    ax1.set_xlabel("x")
    fig.tight_layout()
    out = FIG / "fig_refine_residual_attribution.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def main() -> dict:
    FIG.mkdir(parents=True, exist_ok=True)
    verdict = wr.audit(None, 12800)
    figs = [fig_ladder(verdict), fig_residual(verdict)]
    data = wr.default_data_dir()
    prof = wr.load_profile(data, verdict["finest_graded"]["nx"], "hllc")
    ref = wr.load_profile(data, 12800, "weno5")
    regions = [("left_plateau", (0.40, 0.55)), ("collision_fronts", (0.55, 0.72)),
               ("right_contact_and_shock", (0.72, 1.00))]
    summary = {
        "verdict": verdict,
        "regions": wr.region_shares(wr.residual_density(prof["rho"], ref["rho"]), regions),
        "figures": [p.name for p in figs],
        "predeclared_forecast": FORECAST,
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    print(json.dumps({"figures": summary["figures"], "regions": summary["regions"],
                      "finest": verdict["finest_graded"],
                      "certified_lt_1pct": verdict["certified_lt_1pct"],
                      "not_measured": verdict["not_measured"]}, indent=2))
    return summary


if __name__ == "__main__":
    main()