#!/usr/bin/env python3
"""Generate publication-quality figures for the JOUR-ICE-STRUCT-GUARD deliverable.

Produces four figures that together document the ice-domain guardrail system:
1. Parameter bounds overview (hard/soft/typical ranges for all 8 ice parameters)
2. Validation decision map (PASS/WARN/FAIL regions for thickness vs strength)
3. Guarded vs unguarded resistance comparison (the silent-extrapolation problem)
4. Clamping behavior demonstration (how out-of-bounds values are repaired)

Usage
-----
    python scripts/generate_ice_guardrail_figures.py [--output-dir DIR]

All figures are saved as PNG files in the output directory (default:
``artifacts/ice_guardrail_figures/``).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure the package is importable from a checkout without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from tensorfvm.ice import (
    ICE_THICKNESS,
    ICE_STRENGTH,
    ICE_FRICTION,
    ICE_SPEED,
    ICE_DENSITY,
    ICE_ELASTIC_MODULUS,
    ICE_POISSON_RATIO,
    ICE_TEMPERATURE,
    GuardrailStatus,
    validate_ice_parameters,
    clamp_ice_parameters,
    guarded_level_ice_resistance,
    IceShipGeometry,
    IceCondition,
    level_ice_resistance_kN,
)


# --------------------------------------------------------------------------- #
# Figure 1: Parameter bounds overview
# --------------------------------------------------------------------------- #

def make_bounds_overview(output_dir: Path) -> Path:
    """Figure 1 — horizontal bar chart of hard/soft/typical ranges for all 8 parameters."""
    params = [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
    ]
    labels = [
        "Thickness (m)", "Strength (MPa)", "Friction", "Speed (m/s)",
        "Density (kg/m³)", "Elastic Mod. (GPa)", "Poisson ν", "Temp (°C)",
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(params))
    height = 0.25

    for i, p in enumerate(params):
        b = p.bounds
        # Normalize to [0,1] for visual comparison (each parameter has different units)
        span = b.hard_max - b.hard_min
        if span == 0:
            span = 1.0
        hard_lo = (b.hard_min - b.hard_min) / span
        hard_hi = (b.hard_max - b.hard_min) / span
        soft_lo = (b.soft_min - b.hard_min) / span
        soft_hi = (b.soft_max - b.hard_min) / span
        typ_lo = (b.typical_min - b.hard_min) / span
        typ_hi = (b.typical_max - b.hard_min) / span

        ax.barh(y[i] + height, hard_hi - hard_lo, height, left=hard_lo,
                color="gray", alpha=0.15, label="Physical bounds" if i == 0 else "")
        ax.barh(y[i], soft_hi - soft_lo, height, left=soft_lo,
                color="steelblue", alpha=0.3, label="Calibration domain" if i == 0 else "")
        ax.barh(y[i] - height, typ_hi - typ_lo, height, left=typ_lo,
                color="forestgreen", alpha=0.4, label="Typical range" if i == 0 else "")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Normalized range [0 = hard_min, 1 = hard_max]", fontsize=9)
    ax.set_title("Ice Parameter Guardrails — Three-Tier Bound Structure", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = output_dir / "fig1_bounds_overview.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/4] {path}")
    return path


# --------------------------------------------------------------------------- #
# Figure 2: Validation decision map (thickness vs strength)
# --------------------------------------------------------------------------- #

def make_decision_map(output_dir: Path) -> Path:
    """Figure 2 — 2-D decision surface: PASS / WARN / FAIL regions."""
    thickness_vals = np.linspace(0, 15, 150)
    strength_vals = np.linspace(0, 40, 150)
    T, S = np.meshgrid(thickness_vals, strength_vals)

    status_map = np.zeros_like(T, dtype=int)  # 0=FAIL, 1=WARN, 2=PASS
    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            result = validate_ice_parameters({
                "ice_thickness": float(T[i, j]),
                "ice_strength": float(S[i, j]),
            }, allow_extrapolation=True)
            if result.status == GuardrailStatus.PASS:
                status_map[i, j] = 2
            elif result.status == GuardrailStatus.WARN:
                status_map[i, j] = 1
            else:
                status_map[i, j] = 0

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = matplotlib.colors.ListedColormap(["#ff6b6b", "#ffd93d", "#6bcb77"])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    im = ax.imshow(status_map, origin="lower", aspect="auto",
                   extent=[0, 15, 0, 40], cmap=cmap, norm=norm)

    # Mark the notorious failure cases
    ax.scatter([50], [0], color="red", s=200, marker="x", linewidths=3,
               label="h=50 m (impossible)", zorder=5, clip_on=False)
    ax.scatter([0], [100], color="red", s=200, marker="x", linewidths=3,
               label="σ_c=100 MPa (impossible)", zorder=5, clip_on=False)
    ax.scatter([1.5], [3.0], color="green", s=150, marker="o", edgecolors="black",
               linewidths=1, label="Typical Arctic (1.5 m, 3 MPa)", zorder=5)

    ax.set_xlabel("Ice thickness h (m)", fontsize=10)
    ax.set_ylabel("Compressive strength σ_c (MPa)", fontsize=10)
    ax.set_title("Validation Decision Map — PASS / WARN / FAIL Regions", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 40)
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["FAIL", "WARN", "PASS"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = output_dir / "fig2_decision_map.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/4] {path}")
    return path


# --------------------------------------------------------------------------- #
# Figure 3: Guarded vs unguarded resistance
# --------------------------------------------------------------------------- #

def make_resistance_comparison(output_dir: Path) -> Path:
    """Figure 3 — Bar chart: guarded vs unguarded resistance for extreme inputs."""
    geometry = IceShipGeometry(
        length_m=150.0, beam_m=25.0, draft_m=8.0,
        block_coefficient=0.75, bow_angle_deg=30.0,
    )

    cases = [
        {"label": "Typical\n(h=1.5 m, σ=3 MPa)",
         "ice": {"ice_thickness": 1.5, "ice_strength": 3.0,
                 "ice_friction": 0.1, "ice_speed": 2.5,
                 "ship_length": 150.0, "ship_beam": 25.0,
                 "ship_draft": 8.0, "ship_block_coefficient": 0.75}},
        {"label": "Extreme h\n(h=8 m, σ=3 MPa)",
         "ice": {"ice_thickness": 8.0, "ice_strength": 3.0,
                 "ice_friction": 0.1, "ice_speed": 2.5,
                 "ship_length": 150.0, "ship_beam": 25.0,
                 "ship_draft": 8.0, "ship_block_coefficient": 0.75}},
        {"label": "Impossible h\n(h=50 m, σ=3 MPa)",
         "ice": {"ice_thickness": 50.0, "ice_strength": 3.0,
                 "ice_friction": 0.1, "ice_speed": 2.5,
                 "ship_length": 150.0, "ship_beam": 25.0,
                 "ship_draft": 8.0, "ship_block_coefficient": 0.75}},
        {"label": "Impossible σ\n(h=1.5 m, σ=100 MPa)",
         "ice": {"ice_thickness": 1.5, "ice_strength": 100.0,
                 "ice_friction": 0.1, "ice_speed": 2.5,
                 "ship_length": 150.0, "ship_beam": 25.0,
                 "ship_draft": 8.0, "ship_block_coefficient": 0.75}},
    ]

    guarded_vals = []
    unguarded_vals = []
    guard_statuses = []

    for case in cases:
        params = case["ice"]
        # Unguarded (raw formula)
        try:
            ice_cond = IceCondition(
                thickness_m=params["ice_thickness"],
                strength_mpa=params["ice_strength"],
                friction_coeff=params["ice_friction"],
                speed_ms=params["ice_speed"],
            )
            unguarded = level_ice_resistance_kN(geometry, ice_cond)
        except (ZeroDivisionError, ValueError):
            unguarded = float("nan")
        unguarded_vals.append(unguarded if math.isfinite(unguarded) else 0)

        # Guarded
        result = guarded_level_ice_resistance(params, mode="clamp")
        guarded_vals.append(result.resistance_kN if result.resistance_kN is not None else 0)
        guard_statuses.append(result.guard.status.value)

    x = np.arange(len(cases))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, guarded_vals, width, label="Guarded (clamped)",
                   color="steelblue", edgecolor="black", linewidth=0.5)
    ax.bar(x + width/2, unguarded_vals, width, label="Unguarded (raw)",
                   color="coral", edgecolor="black", linewidth=0.5)

    # Annotate guard status
    for i, status in enumerate(guard_statuses):
        color = {"pass": "green", "warn": "orange", "fail": "red"}.get(status, "gray")
        ax.text(i, max(guarded_vals[i], unguarded_vals[i]) * 1.05,
                f"[{status.upper()}]", ha="center", fontsize=8,
                color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in cases], fontsize=9)
    ax.set_ylabel("Resistance (kN)", fontsize=10)
    ax.set_title("Guarded vs Unguarded Ice Resistance — The Silent Extrapolation Problem",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()

    path = output_dir / "fig3_resistance_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/4] {path}")
    return path


# --------------------------------------------------------------------------- #
# Figure 4: Clamping behavior
# --------------------------------------------------------------------------- #

def make_clamping_demo(output_dir: Path) -> Path:
    """Figure 4 — Before/after clamping for all 8 ice parameters."""
    params = [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
    ]
    labels = ["h (m)", "σ_c (MPa)", "μ", "v (m/s)", "ρ (kg/m³)", "E (GPa)", "ν", "T (°C)"]

    # Extreme inputs that need clamping
    extreme_values = [50.0, 100.0, 2.0, 20.0, 1000.0, 50.0, 0.6, 10.0]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, (param, label, extreme) in enumerate(zip(params, labels, extreme_values)):
        ax = axes[i]
        b = param.bounds

        # Draw bounds
        ax.axvline(b.hard_min, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(b.hard_max, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.axvline(b.soft_min, color="orange", linestyle="-.", linewidth=1, alpha=0.7)
        ax.axvline(b.soft_max, color="orange", linestyle="-.", linewidth=1, alpha=0.7)

        # Clamp result
        result = clamp_ice_parameters({param.bounds.name: extreme})
        clamped = result.clamped_params.get(param.bounds.name, extreme)

        # Plot
        ax.scatter([extreme], [0.5], color="red", s=100, zorder=5,
                   label=f"Input: {extreme}")
        if math.isfinite(clamped):
            ax.scatter([clamped], [0.5], color="green", s=100, zorder=5,
                       label=f"Clamped: {clamped:.2f}")
        ax.axhline(0.5, color="gray", alpha=0.3)

        ax.set_title(f"{label}", fontsize=10)
        ax.set_xlim(min(b.hard_min, extreme) * 0.8, max(b.hard_max, extreme) * 1.2)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper right")

    fig.suptitle("Clamping Behavior — Extreme Inputs Repaired to Physical Bounds",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    path = output_dir / "fig4_clamping_demo.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/4] {path}")
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate ice guardrail figures")
    parser.add_argument("--output-dir", default="artifacts/ice_guardrail_figures",
                        help="Output directory for PNG files")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating ice guardrail figures...")
    paths = []
    paths.append(make_bounds_overview(output_dir))
    paths.append(make_decision_map(output_dir))
    paths.append(make_resistance_comparison(output_dir))
    paths.append(make_clamping_demo(output_dir))

    # Generate a summary JSON with base64-encoded figures for report embedding
    summary = {"figures": []}
    for p in paths:
        with open(p, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        summary["figures"].append({
            "name": p.stem,
            "path": str(p),
            "png_base64": b64[:80] + "...",  # truncated for readability
        })

    summary_path = output_dir / "figures_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
