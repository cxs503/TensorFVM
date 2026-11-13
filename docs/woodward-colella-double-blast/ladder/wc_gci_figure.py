"""GCI convergence figure for CASE-WOODWARD-COLELLA.

Generates a publication-quality figure showing the three-grid convergence
of peak density, L1 error, and window mass, with GCI uncertainty bands.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from wc_gci_analysis import analyse_gci, PEAK_DENSITY, L1_ERROR, WINDOW_MASS, REFERENCE

OUTDIR = Path(__file__).resolve().parent / "data"


def make_gci_figure():
    """Three-panel GCI convergence figure."""
    results = analyse_gci()
    sizes = np.array([200, 400, 800])
    dx = 1.0 / sizes

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # Panel 1: Peak density convergence
    ax = axes[0]
    r = results["peak_density"]
    ax.plot(dx, [PEAK_DENSITY[n] for n in sizes], "o-", label="TensorFVM HLLC/MUSCL")
    ax.axhline(REFERENCE["peak_density"], color="k", ls="--", lw=1, label="Reference (RPJ@3200)")
    ax.axhline(r.extrapolated, color="r", ls=":", lw=1, label=f"Richardson ext. ({r.extrapolated:.2f})")
    ax.fill_between(dx, r.gci_band_low, r.gci_band_high, alpha=0.15, color="r",
                     label=f"GCI band ({r.gci_percent:.1f}%)")
    ax.set_xlabel("dx = 1/nx")
    ax.set_ylabel(r"$\rho_{\max}$")
    ax.set_title(f"Peak density (p_obs={r.observed_order:.2f})")
    ax.legend(fontsize=7)
    ax.set_xscale("log")

    # Panel 2: L1 error convergence
    ax = axes[1]
    r = results["l1_error"]
    ax.plot(dx, [L1_ERROR[n] for n in sizes], "s-", color="C1", label="L1(ρ) error")
    ax.axhline(r.extrapolated, color="r", ls=":", lw=1, label=f"Richardson ext. ({r.extrapolated:.4f})")
    ax.fill_between(dx, max(0, r.gci_band_low), r.gci_band_high, alpha=0.15, color="r",
                     label=f"GCI band ({r.gci_percent:.1f}%)")
    ax.set_xlabel("dx = 1/nx")
    ax.set_ylabel("L1 relative error")
    ax.set_title(f"L1 error (p_obs={r.observed_order:.2f})")
    ax.legend(fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")

    # Panel 3: Window mass convergence
    ax = axes[2]
    r = results["window_mass"]
    ax.plot(dx, [WINDOW_MASS[n] for n in sizes], "^-", color="C2", label="Window mass ∫ρdx")
    ax.axhline(REFERENCE["window_mass"], color="k", ls="--", lw=1, label="Reference")
    ax.axhline(r.extrapolated, color="r", ls=":", lw=1, label=f"Richardson ext. ({r.extrapolated:.5f})")
    ax.fill_between(dx, r.gci_band_low, r.gci_band_high, alpha=0.15, color="r",
                     label=f"GCI band ({r.gci_percent:.2f}%)")
    ax.set_xlabel("dx = 1/nx")
    ax.set_ylabel(r"$\int_{0.4}^{1.0} \rho \, dx$")
    ax.set_title(f"Window mass (p_obs={r.observed_order:.2f})")
    ax.legend(fontsize=7)
    ax.set_xscale("log")

    fig.suptitle("CASE-WOODWARD-COLELLA: GCI Convergence Analysis (nx=200, 400, 800)",
                 fontsize=11, y=1.02)
    fig.tight_layout()

    path = OUTDIR / "fig_wc_gci_convergence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {path}")
    return str(path)


if __name__ == "__main__":
    make_gci_figure()
