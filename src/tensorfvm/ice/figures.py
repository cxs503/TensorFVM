"""Figure generation for ice-domain guardrail documentation.

Thin wrapper around ``scripts/generate_ice_guardrail_figures.py`` so that
``tensorfvm-ice-figures`` works as a CLI entry point after ``pip install``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="tensorfvm-ice-figures",
        description="Generate publication-quality figures for the ice guardrail deliverable.",
    )
    parser.add_argument("--output-dir", default="artifacts/ice_guardrail_figures",
                        help="Output directory for PNG files")
    args = parser.parse_args(argv)

    # Import the actual generation logic (works from installed package or checkout)
    from tensorfvm.ice import (
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
        GuardrailStatus, validate_ice_parameters,
        clamp_ice_parameters, guarded_level_ice_resistance, IceShipGeometry, IceCondition,
        level_ice_resistance_kN,
    )

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Bounds overview
    params_list = [
        ICE_THICKNESS, ICE_STRENGTH, ICE_FRICTION, ICE_SPEED,
        ICE_DENSITY, ICE_ELASTIC_MODULUS, ICE_POISSON_RATIO, ICE_TEMPERATURE,
    ]
    labels = [
        "Thickness (m)", "Strength (MPa)", "Friction", "Speed (m/s)",
        "Density (kg/m³)", "Elastic Mod. (GPa)", "Poisson ν", "Temp (°C)",
    ]
    fig, ax = plt.subplots(figsize=(12, 6))
    y = np.arange(len(params_list))
    height = 0.25
    for i, p in enumerate(params_list):
        b = p.bounds
        span = b.hard_max - b.hard_min or 1.0
        ax.barh(y[i] + height, (b.hard_max - b.hard_min) / span, height, left=0,
                color="gray", alpha=0.15, label="Physical bounds" if i == 0 else "")
        ax.barh(y[i], (b.soft_max - b.soft_min) / span, height, left=(b.soft_min - b.hard_min) / span,
                color="steelblue", alpha=0.3, label="Calibration domain" if i == 0 else "")
        ax.barh(y[i] - height, (b.typical_max - b.typical_min) / span, height, left=(b.typical_min - b.hard_min) / span,
                color="forestgreen", alpha=0.4, label="Typical range" if i == 0 else "")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Normalized range", fontsize=9)
    ax.set_title("Ice Parameter Guardrails — Three-Tier Bound Structure", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    p1 = output_dir / "fig1_bounds_overview.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [1/4] {p1}")

    # Figure 2: Decision map
    thickness_vals = np.linspace(0, 15, 150)
    strength_vals = np.linspace(0, 40, 150)
    T, S = np.meshgrid(thickness_vals, strength_vals)
    status_map = np.zeros_like(T, dtype=int)
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
    norm = matplotlib.colors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    ax.imshow(status_map, origin="lower", aspect="auto", extent=[0, 15, 0, 40], cmap=cmap, norm=norm)
    ax.scatter([1.5], [3.0], color="green", s=150, marker="o", edgecolors="black",
               linewidths=1, label="Typical Arctic (1.5 m, 3 MPa)", zorder=5)
    ax.set_xlabel("Ice thickness h (m)", fontsize=10)
    ax.set_ylabel("Compressive strength σ_c (MPa)", fontsize=10)
    ax.set_title("Validation Decision Map", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 40)
    cbar = plt.colorbar(ax.imshow(status_map, origin="lower", aspect="auto", extent=[0, 15, 0, 40], cmap=cmap, norm=norm), ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["FAIL", "WARN", "PASS"])
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    p2 = output_dir / "fig2_decision_map.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [2/4] {p2}")

    # Figure 3: Resistance comparison
    geometry = IceShipGeometry(length_m=150.0, beam_m=25.0, draft_m=8.0, block_coefficient=0.75, bow_angle_deg=30.0)
    cases = [
        {"label": "Typical", "ice_thickness": 1.5, "ice_strength": 3.0},
        {"label": "Extreme h", "ice_thickness": 8.0, "ice_strength": 3.0},
        {"label": "Impossible h", "ice_thickness": 50.0, "ice_strength": 3.0},
        {"label": "Impossible σ", "ice_thickness": 1.5, "ice_strength": 100.0},
    ]
    guarded_vals, unguarded_vals = [], []
    for c in cases:
        params = {"ice_thickness": c["ice_thickness"], "ice_strength": c["ice_strength"],
                  "ice_friction": 0.1, "ice_speed": 2.5, "ship_length": 150.0,
                  "ship_beam": 25.0, "ship_draft": 8.0, "ship_block_coefficient": 0.75}
        try:
            ice_cond = IceCondition(thickness_m=c["ice_thickness"], strength_mpa=c["ice_strength"],
                                    friction_coeff=0.1, speed_ms=2.5)
            unguarded_vals.append(level_ice_resistance_kN(geometry, ice_cond))
        except (ZeroDivisionError, ValueError):
            unguarded_vals.append(0)
        result = guarded_level_ice_resistance(params, mode="clamp")
        guarded_vals.append(result.resistance_kN or 0)
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(cases))
    ax.bar(x - 0.175, guarded_vals, 0.35, label="Guarded", color="steelblue")
    ax.bar(x + 0.175, unguarded_vals, 0.35, label="Unguarded", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels([c["label"] for c in cases])
    ax.set_ylabel("Resistance (kN)")
    ax.set_title("Guarded vs Unguarded Ice Resistance", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_yscale("log")
    plt.tight_layout()
    p3 = output_dir / "fig3_resistance_comparison.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [3/4] {p3}")

    # Figure 4: Clamping demo
    extreme_values = [50.0, 100.0, 2.0, 20.0, 1000.0, 50.0, 0.6, 10.0]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, (param, label, extreme) in enumerate(zip(params_list, labels, extreme_values)):
        ax = axes[i]
        b = param.bounds
        ax.axvline(b.hard_min, color="red", linestyle="--", alpha=0.7)
        ax.axvline(b.hard_max, color="red", linestyle="--", alpha=0.7)
        result = clamp_ice_parameters({param.bounds.name: extreme})
        clamped = result.clamped_params.get(param.bounds.name, extreme)
        ax.scatter([extreme], [0.5], color="red", s=100, label=f"Input: {extreme}")
        if math.isfinite(clamped):
            ax.scatter([clamped], [0.5], color="green", s=100, label=f"Clamped: {clamped:.2f}")
        ax.set_title(label, fontsize=10)
        ax.set_yticks([])
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Clamping Behavior", fontsize=12, fontweight="bold")
    plt.tight_layout()
    p4 = output_dir / "fig4_clamping_demo.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [4/4] {p4}")

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
