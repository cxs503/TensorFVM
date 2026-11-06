#!/usr/bin/env python3
"""Generate stability analysis figures for the VOF transport report.

Produces:
- amplification_factor.png: |G| vs theta for various C values
- stability_map.png: max|G| vs C for both schemes

Usage: PYTHONPATH=src python3 scripts/verify_vof_stability.py
"""

import math
from pathlib import Path

import numpy as np

# Ensure output directory exists
OUTPUT_DIR = Path(__file__).parent.parent / "docs" / "vof-transport-stability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def G_blend(theta, C):
    real = (1 - C + 2 * C ** 2) + (C - 2 * C ** 2) * math.cos(theta)
    imag = -C * math.sin(theta)
    return math.sqrt(real ** 2 + imag ** 2)


def G_upwind(theta, C):
    real = 1 - C * (1 - math.cos(theta))
    imag = -C * math.sin(theta)
    return math.sqrt(real ** 2 + imag ** 2)


def generate_amplification_plot():
    """Plot |G| vs theta for various C values."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas = np.linspace(0, math.pi, 500)
    C_values = [0.1, 0.25, 1.0 / 3.0, 0.5, 0.75, 1.0]
    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#34495e"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for C, color in zip(C_values, colors):
        g_vals = [G_blend(t, C) for t in thetas]
        label = f"C = {C:.3f}" if C != 1.0 / 3.0 else "C = 1/3 (critical)"
        ax.plot(thetas, g_vals, color=color, linewidth=2, label=label)

    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="|G| = 1 (stability limit)")
    ax.set_xlabel(r"Wavenumber $\theta$", fontsize=12)
    ax.set_ylabel(r"Amplification Factor $|G|$", fontsize=12)
    ax.set_title("VOF Blend Scheme: Amplification Factor vs Wavenumber", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, math.pi)
    ax.set_ylim(0.8, 1.25)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "amplification_factor.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def generate_stability_map():
    """Plot max|G| vs C for both schemes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_values = np.linspace(0.01, 1.0, 200)

    max_blend = []
    max_upwind = []
    thetas = np.linspace(0, math.pi, 500)

    for C in C_values:
        gb = max(G_blend(t, C) for t in thetas)
        gu = max(G_upwind(t, C) for t in thetas)
        max_blend.append(gb)
        max_upwind.append(gu)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(C_values, max_blend, color="#e74c3c", linewidth=2, label="Blend scheme")
    ax.plot(C_values, max_upwind, color="#2ecc71", linewidth=2, label="Upwind scheme")
    ax.axhline(y=1.0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="|G| = 1 (stability limit)")
    ax.axvline(x=1.0 / 3.0, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.7, label="C = 1/3 (blend stability limit)")

    ax.set_xlabel("Courant Number C", fontsize=12)
    ax.set_ylabel(r"Max Amplification Factor $\max_\theta |G|$", fontsize=12)
    ax.set_title("VOF Transport: Stability Map", fontsize=14)
    ax.legend(fontsize=10, loc="upper left")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.95, 1.25)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "stability_map.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


def generate_diffusion_plot():
    """Plot effective numerical diffusion coefficient vs C."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    C_values = np.linspace(0, 1.0, 200)
    D_blend = [1 - 3 * C for C in C_values]
    D_upwind = [1 - C for C in C_values]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(C_values, D_blend, color="#e74c3c", linewidth=2, label="Blend: D = (udx/2)(1-3C)")
    ax.plot(C_values, D_upwind, color="#2ecc71", linewidth=2, label="Upwind: D = (udx/2)(1-C)")
    ax.axhline(y=0, color="black", linestyle="--", linewidth=1, alpha=0.5)
    ax.axvline(x=1.0 / 3.0, color="#e74c3c", linestyle=":", linewidth=1, alpha=0.7, label="C = 1/3 (anti-diffusive threshold)")

    ax.set_xlabel("Courant Number C", fontsize=12)
    ax.set_ylabel(r"Effective Diffusion Coefficient (normalized)", fontsize=12)
    ax.set_title("VOF Transport: Effective Numerical Diffusion", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.5, 1.1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    outpath = OUTPUT_DIR / "diffusion_coefficient.png"
    plt.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {outpath}")


if __name__ == "__main__":
    generate_amplification_plot()
    generate_stability_map()
    generate_diffusion_plot()
    print("All figures generated successfully.")
