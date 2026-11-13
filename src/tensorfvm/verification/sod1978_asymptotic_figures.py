"""Figures for the asymptotic Sod (1978) convergence study.

Four panels, all generated from ``artifacts/asymptotic_study.json`` only:

``fig1_error_ladder``
    log-log L1(density) vs ``h``: whole field, cell average and the smooth
    entropy wave, with reference slopes ``h^1`` and ``h^2`` drawn so the eye can
    check the rate without reading a table.
``fig2_rates_towards_one``
    pairwise observed order vs the grid pair, with the literal criterion line at
    ``1.0`` -- this is the plot that shows whether the rate approaches one.
``fig3_profiles``
    the finest-grid density/velocity/pressure profiles against the exact
    solution (the paper's ``t`` = 0.14 snapshot).
``fig4_roundoff_floor``
    restricted-window L1 error against the declared round-off floor, so no rate
    is quoted from a measure that has hit machine precision.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _errs(rows: List[Dict], key: str):
    h, e = [], []
    for r in rows:
        v = r[key]
        if v and math.isfinite(v):
            h.append(1.0 / r["nx"])
            e.append(v)
    return np.array(h), np.array(e)


def make_figures(outdir: Path, study: Dict) -> List[Dict[str, str]]:
    outdir = Path(outdir)
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figs: List[Dict[str, str]] = []
    rows = study["rows"]
    orders = sorted(rows, key=int)

    # ---------------- fig 1: error ladder -----------------------------------
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for o in orders:
        h, e = _errs(rows[o], "l1_rho_full")
        ax.loglog(h, e, "o-", label=f"whole field L1, order {o} recon")
        h, ec = _errs(rows[o], "l1_rho_cellavg")
        ax.loglog(h, ec, "s--", alpha=0.75,
                  label=f"cell-average L1, order {o} recon")
    for o in orders:
        sw = study.get("smooth_wave", {}).get(o)
        if not sw:
            continue
        h = np.array([1.0 / r["nx"] for r in sw["rows"]])
        e = np.array([r["l1_rho"] for r in sw["rows"]])
        ax.loglog(h, e, "^-", alpha=0.7,
                  label=f"smooth entropy wave, order {o} recon")
    href = np.array([1.0 / max(r["nx"] for r in rows[orders[0]]), 1.0 / min(r["nx"] for r in rows[orders[0]])])
    # reference slopes anchored to the finest whole-field point
    h0, e0 = _errs(rows[orders[-1]], "l1_rho_full")
    e0, h0 = e0[-1], h0[-1]
    for p, style in ((1.0, ":"), (2.0, "-.")):
        ax.loglog(href, e0 * (href / h0) ** p, color="k", lw=0.9, ls=style,
                  label=f"reference slope h^{p:.0f}")
    ax.set_xlabel("grid spacing h = 1/nx")
    ax.set_ylabel("L1 error of density (mean |rho - rho_exact|)")
    ax.set_title("Sod 1978 t=0.14: error ladder, nx=100 ... 6400")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    f1 = figdir / "fig1_error_ladder.png"
    fig.tight_layout()
    fig.savefig(f1, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig1_error_ladder.png",
                 "caption": "L1 error ladder over nx=100..6400 (whole field, cell average, "
                            "smooth entropy wave) with h^1 and h^2 reference slopes."})

    # ---------------- fig 2: pairwise rates --------------------------------
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    for o in orders:
        succ = study["rate_tables"][o]["l1_rho_full"]["successive"]
        x = np.arange(len(succ))
        ax.plot(x, [s["rate"] for s in succ], "o-", label=f"whole field, order {o}")
        for i, s in enumerate(succ):
            ax.annotate(f"{s['nx_pair'][0]}->{s['nx_pair'][1]}", (i, s["rate"]),
                        textcoords="offset points", xytext=(0, 7), fontsize=6.5,
                        ha="center", rotation=0)
        succ_type = study["rate_tables"][o]["l1_rho_cellavg"]["successive"]
        ax.plot(np.arange(len(succ_type)), [s["rate"] for s in succ_type], "s--",
                alpha=0.7, label=f"cell average, order {o}")
        sw = study.get("smooth_wave", {}).get(o)
        if sw:
            ax.axhline(sw["fitted_exponent_finest3"], color="C2" if o == "1" else "C3",
                       lw=0.8, ls=":",
                       label=f"smooth-wave fitted exponent, order {o}")
    ax.axhline(1.0, color="k", lw=1.2)
    ax.text(0.02, 1.02, "literal criterion: whole-field L1 order >= 1.0",
            transform=ax.get_yaxis_transform(), fontsize=8)
    ax.set_xlabel("grid pair (doubling)")
    ax.set_ylabel("observed order log(e1/e2)/log(2)")
    ax.set_title("Does the whole-field rate approach one as h -> 0?")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    f2 = figdir / "fig2_rates_towards_one.png"
    fig.tight_layout()
    fig.savefig(f2, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig2_rates_towards_one.png",
                 "caption": "Pairwise observed order over the declared ladder against the "
                            "literal >=1.0 criterion line; annotations give the grid pair."})

    # ---------------- fig 3: profiles at the finest grid --------------------
    prof = study.get("profiles") or []
    if prof:
        p = prof[-1]
        x = np.array(p["x"])
        fig, axes = plt.subplots(3, 1, figsize=(8.2, 7.6), sharex=True)
        for ax, key, lab in zip(axes, ("rho", "u", "p"), ("density", "velocity", "pressure")):
            ax.plot(x, p[key], "o", ms=1.6, mfc="none", mew=0.5,
                    label=f"TensorFVM HLLC+MUSCL, nx={p['nx']}")
            ax.plot(x, p[f"{key}_exact"], "-", lw=1.3, color="k", label="exact (Sod/Riemann)")
            ax.set_ylabel(lab)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="best")
        axes[0].set_title(f"Sod t={p['t_end']:.2f} field comparison at nx={p['nx']}")
        axes[-1].set_xlabel("x")
        f3 = figdir / "fig3_profiles.png"
        fig.tight_layout()
        fig.savefig(f3, dpi=140)
        plt.close(fig)
        figs.append({"name": "fig3_profiles.png",
                     "caption": "Density/velocity/pressure at the finest grid against the exact "
                                "Riemann solution (field comparison)."})

    # ---------------- fig 4: round-off floor -------------------------------
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for o in orders:
        h, e = _errs(rows[o], "l1_rho_restricted")
        ax.loglog(h, e, "o-", label=f"restricted (smooth windows), order {o}")
    ax.axhline(study["thresholds_declared_before_run"]["roundoff_floor"], color="r",
               ls="--", label="declared round-off floor 1e-12")
    ax.set_xlabel("grid spacing h")
    ax.set_ylabel("L1 error on smooth windows")
    ax.set_title("Why the smooth-window rate cannot be graded at fine grids")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    f4 = figdir / "fig4_roundoff_floor.png"
    fig.tight_layout()
    fig.savefig(f4, dpi=140)
    plt.close(fig)
    figs.append({"name": "fig4_roundoff_floor.png",
                 "caption": "Restricted-window L1 error vs the declared 1e-12 round-off floor; "
                            "the measure collapses to machine precision before the finest grids."})
    return figs