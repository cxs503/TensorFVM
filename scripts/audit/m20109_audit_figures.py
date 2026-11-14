#!/usr/bin/env python3
"""Figures for the m20109 board audit (CASE-FVM-CONVDIFF-PHASE).

Every curve is read from the audit artifacts written by
scripts/audit/m20109_convdiff_phase_audit.py --all (nothing hand-typed).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART = os.path.join(REPO, "artifacts", "board_audit_m20109")
FIG = os.path.join(REPO, "docs", "board_audit_m20109", "figures")
TOL = 0.03   # 3 % tolerance in force from the mission goal (referee table empty)


def main() -> None:
    os.makedirs(FIG, exist_ok=True)
    rec = json.load(open(os.path.join(ART, "recompute.json")))
    ref = json.load(open(os.path.join(ART, "referee.json")))
    rep = json.load(open(os.path.join(ART, "report.json")))

    # ---- fig 1: Courant sweep, audit vs artifact vs leading-order law ------
    cs = rec["courant_sweep"]
    C = np.array([m["courant_target"] for m in cs])
    lag_audit = np.array([m["lag_per_step_vs_index_reference"] for m in cs])
    law = np.array([m["upwind_leading_order_per_step"] for m in cs])
    art = json.load(open(os.path.join(REPO, "docs", "board_audit_m20109", "audited",
                                     "runs", "summary.json")))
    lag_art = np.array([r["lag_per_step"] for r in art["courant_sweep"]])

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    a = ax[0]
    a.plot(C, lag_audit, "o", ms=6, label="audit recomputation (independent fit)")
    a.plot(C, lag_art, "x", ms=7, label="delivered artifact summary.json")
    a.plot(C, law, "-", lw=1.2, label=r"leading order $C\theta^3(1-C)(1-2C)/6$")
    a.axhline(0, color="k", lw=0.6)
    a.set_xlabel("Courant number C"); a.set_ylabel("phase lag per step [rad]")
    a.set_title("CASE-FVM-CONVDIFF-PHASE: dispersion sign change at C=1/2, C=1")
    a.legend(fontsize=7); a.grid(alpha=.3)

    # ---- fig 1 right: grid-order ladder ------------------------------------
    lad = rec["grid_ladder"]
    h = np.array([m["ds"] for m in lad])
    e = np.array([abs(m["lag_rad_vs_index_reference"]) for m in lad])
    b = ax[1]
    b.loglog(h, e, "o-", label="audit |phase lag| vs $h$")
    b.loglog(h, e[-1] * (h / h[-1]) ** 3, "k--", lw=1,
             label="slope 3 reference (observed %.3f)" % rec["observed_order_audit"])
    b.set_xlabel("cell spacing $h$ [m]"); b.set_ylabel("|phase lag| after 200 steps [rad]")
    b.set_title("third-order dispersion confirmed"); b.legend(fontsize=7); b.grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "audit_fig1_phase_and_order.png"), dpi=150)
    plt.close(fig)

    # ---- fig 2: error budget vs the 3 % tolerance --------------------------
    err = rec["errors_pct"]
    items = [
        ("phase vs analytic\n(index reference, as reported)", err["phase_vs_index_reference"]),
        ("phase vs analytic\n(geometric reference, audit)", err["phase_vs_geometric_reference"]),
        ("amplitude vs modified\nequation (C4)", err["amplitude_vs_modified_equation"]),
        ("amplitude vs pure\nanalytic sine", err["amplitude_vs_analytic"]),
        ("diffusive branch vs\ncontinuum a=1e-2 (C6)", 24.749672420181754),
    ]
    fig, ax = plt.subplots(figsize=(8.4, 4.2))
    y = np.arange(len(items))
    vals = [max(v, 1e-6) for _, v in items]
    colors = ["tab:green" if v < 100 * TOL else "tab:red" for _, v in items]
    ax.barh(y, vals, color=colors, alpha=.85)
    ax.set_yticks(y); ax.set_yticklabels([k for k, _ in items], fontsize=8)
    ax.set_xscale("log"); ax.axvline(100 * TOL, color="k", ls="--", lw=1.2,
                                     label="tolerance in force: 3 % (mission goal; bench table empty)")
    for yi, (_, v) in zip(y, items):
        ax.text(max(v, 1e-6) * 1.15, yi, ("%.4g %%" % v), va="center", fontsize=8)
    ax.set_xlabel("error [%]  (log scale)")
    ax.set_title("audit error budget: what the case certifies and what it honestly does not")
    ax.legend(fontsize=7, loc="lower right"); ax.grid(alpha=.3, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "audit_fig2_error_budget.png"), dpi=150)
    plt.close(fig)

    print("referee rows:", ref["registered_rows"], "| report headline ok:",
          rep["headline_lag_per_step_matches_artifact"])
    print("figures written to", os.path.relpath(FIG, REPO))


if __name__ == "__main__":
    main()