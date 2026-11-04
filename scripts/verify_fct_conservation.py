#!/usr/bin/env python3
"""Run the FCT conservation audit; emit JSON + a 3-panel figure.

Usage:  PYTHONPATH=src python3 scripts/verify_fct_conservation.py [out_dir]
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from tensorfvm.verification import fct_conservation_audit as fca  # noqa: E402


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(__file__), "..", "src", "tensorfvm", "verification", "artifacts",
        "fct-conservation-audit")
    os.makedirs(out_dir, exist_ok=True)
    result = fca.run(os.path.join(out_dir, "fct_conservation_audit.json"))
    dec = result["gaussian_decomposition"]
    study = result["compact_study"]

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    labels = ["clamp clip\n(per step)", "boundary flux\n(per step)", "anti-diffusive\n(per step)"]
    vals = [dec["clamp_clip_per_step"], dec["boundary_flux_per_step"],
            dec["anti_contribution_per_step"]]
    colors = ["#2a9d8f", "#e63946", "#457b9d"]
    ax.bar(labels, [abs(v) for v in vals], color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("|contribution| (alpha units)")
    ax.set_title("Gaussian-case drift decomposition\n(n=256, one step)")
    for i, v in enumerate(vals):
        ax.text(i, max(abs(v), 1e-18) * 1.3, f"{v:.1e}", ha="center", fontsize=8)
    ax.text(0.5, -0.16, "clamp never activates; 100% of the leak is boundary",
            transform=ax.transAxes, ha="center", fontsize=8)

    ax = axes[1]
    for key, color, marker in (("fct", "#e63946", "o"), ("upwind", "#2a9d8f", "s")):
        ns = [m["n"] for m in study[key]]
        drift = [m["mass_rel_drift"] for m in study[key]]
        ax.loglog(ns, drift, marker, label=key, lw=1.5)
    ax.axhline(1e-12, color="grey", ls=":", lw=1)
    ax.text(60, 2e-12, "machine precision", fontsize=8, color="grey")
    ax.set_xlabel("grid n")
    ax.set_ylabel("mass rel. drift")
    ax.set_title("Compact profile: conservation\n(upwind exact, FCT converging)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[2]
    for key, color, marker in (("fct", "#e63946", "o"), ("upwind", "#2a9d8f", "s")):
        ns = [m["n"] for m in study[key]]
        l1 = [m["l1_rel"] for m in study[key]]
        ax.loglog(ns, l1, marker, label=key, lw=1.5)
    ax.set_xlabel("grid n")
    ax.set_ylabel("rel. L1 error")
    ax.set_title(f"Compact profile: accuracy\n(FCT p={study['fct_order_l1']:.2f}, "
                 f"upwind p={study['upwind_order_l1']:.2f})")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    fig_path = os.path.join(out_dir, "fct_conservation_audit.png")
    fig.savefig(fig_path, dpi=110)
    plt.close(fig)

    print(json.dumps(result, indent=2))
    print("FIGURE", os.path.abspath(fig_path))


if __name__ == "__main__":
    main()
