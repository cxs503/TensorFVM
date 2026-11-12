#!/usr/bin/env python3
"""Figures for the m15859 audit (CASE-CHANNEL-PERIODIC, TensorFVM).

Reuses the audit recomputation (docs/reports/m15888_audit_m15859_verify.py) and
the delivered artifacts; writes docs/reports/figures/m15888_audit_m15859.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ART = ROOT / "docs" / "case-channel-periodic"
FIG = HERE / "figures" / "m15888_audit_m15859.png"


def main():
    bench = json.loads((ART / "benchmark.json").read_text())
    audit = json.loads((HERE / "m15888_audit_m15859_verify.json").read_text())
    ref = next(c for c in bench["cases"] if c["role"] == "reference")
    y = np.array(ref["fields"]["y_m"])
    u = np.array(ref["fields"]["u_m_s"])
    ue = np.array(ref["fields"]["u_exact_m_s"])

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    # 1) delivered profile vs analytic, with the O(dy^2) offset amplified
    ax[0].plot(u, y, "o", ms=3, label="solver u(y) (ny=64, half-cell)")
    ax[0].plot(ue, y, "-", lw=1.6, label=r"exact $6\bar U (y/H)(1-y/H)$ (audit recomputed)")
    ax[0].plot(u * 1.25, y, ":", lw=1.0, color="grey", label="solver x1.25 (offset visible)")
    ax[0].set_xlabel("u / (m/s)")
    ax[0].set_ylabel("y / H")
    ax[0].set_title(f"profile: L2 = {audit['recomputed_reference_profile']['velocity_l2_recomputed']:.3e}"
                    f" (peak-normalised), < 2% = "
                    f"{audit['recomputed_reference_profile']['under_2pct']}")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    # 2) grid convergence, recomputed from the delivered series + audit fresh point
    spatial = sorted([c for c in bench["cases"] if c["role"] == "spatial"], key=lambda c: c["ny"])
    ny = np.array([c["ny"] for c in spatial], dtype=float)
    pick = lambda c, n: next(m["relative_error"] for m in c["metrics"] if m["name"] == n)
    l2 = np.array([pick(c, "velocity_l2") for c in spatial])
    ax[1].loglog(1 / ny, l2, "s-", label="delivered velocity L2")
    ax[1].loglog(1 / ny, (1 / ny) ** 2, "k--", lw=1, label=r"$(\Delta y/H)^2$")
    fresh = audit["fresh_case_outside_plan"]
    ax[1].loglog([1 / 48], [fresh["half-cell"]["velocity_l2_rel_peak"]], "*", ms=12,
                 label="audit fresh ny=48, Re_H=250")
    ax[1].set_xlabel(r"$\Delta y / H$")
    ax[1].set_ylabel("relative error")
    ax[1].set_title(f"grid convergence: observed orders {['%.2f' % o for o in bench['observed_orders']['velocity_l2']]}")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")

    # 3) ref-vs-got per metric, recomputed by the audit from raw profiles.csv
    hd = audit["recomputed_headline"]
    names = ["bulk_velocity", "tau_wall_flux", "tau_wall_quadratic", "poiseuille"]
    vals = [hd["bulk_rel_error"], hd["tau_wall_flux_rel_error"],
            hd["tau_wall_quadratic_rel_error"], hd["poiseuille_rel_error"]]
    vals = [max(v, 1e-16) for v in vals]
    ax[2].bar(names, vals, color=["tab:blue", "tab:green", "tab:orange", "tab:red"])
    ax[2].set_yscale("log")
    ax[2].axhline(0.02, color="k", ls="--", label="2% acceptance")
    ax[2].set_ylabel("relative error (audit recomputed)")
    ax[2].set_title("headline metrics vs analytic reference")
    for i, v in enumerate(vals):
        ax[2].text(i, v * 1.6, f"{v:.2e}", ha="center", fontsize=7)
    ax[2].legend(fontsize=7)
    ax[2].tick_params(axis="x", labelrotation=20, labelsize=8)
    ax[2].grid(alpha=0.3, axis="y", which="both")

    fig.suptitle("m15888 audit of m15859 · CASE-CHANNEL-PERIODIC (TensorFVM) · "
                 f"replay diff vs published = {audit['replay']['max_rel_diff_vs_published']:.1e}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG, dpi=140)
    print("wrote", FIG, FIG.stat().st_size)


if __name__ == "__main__":
    main()