"""Generate publication-quality figures for ice guardrail validation.

Produces:
1. Parameter bounds visualization (all 8 parameters)
2. Validation results for test cases
3. The specific failure cases from the task (h=50 m, σ_c=100 MPa)
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.ice import (
    ICE_THICKNESS,
    ICE_STRENGTH,
    ICE_FRICTION,
    ICE_SPEED,
    ICE_DENSITY,
    ICE_ELASTIC_MODULUS,
    ICE_POISSON_RATIO,
    ICE_TEMPERATURE,
    validate_ice_parameters,
    clamp_ice_parameters,
    get_parameter_report,
)


def make_bounds_figure(output_dir: Path = Path("artifacts/ice_guardrails")):
    """Create a figure showing parameter bounds and test cases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle("Ice Domain Guardrails — Physical Parameter Bounds", fontsize=14, fontweight="bold")

    params = [
        (ICE_THICKNESS, "Ice Thickness (h)", "m"),
        (ICE_STRENGTH, "Compressive Strength (σ_c)", "MPa"),
        (ICE_FRICTION, "Friction Coefficient (μ)", ""),
        (ICE_SPEED, "Ship Speed (v)", "m/s"),
        (ICE_DENSITY, "Ice Density (ρ_ice)", "kg/m³"),
        (ICE_ELASTIC_MODULUS, "Elastic Modulus (E)", "GPa"),
        (ICE_POISSON_RATIO, "Poisson's Ratio (ν)", ""),
        (ICE_TEMPERATURE, "Temperature (T)", "°C"),
    ]

    test_cases = {
        "ice_thickness": [(1.5, "typical", "green"), (6.0, "extrapolation", "orange"), (50.0, "impossible", "red")],
        "ice_strength": [(3.0, "typical", "green"), (18.0, "extrapolation", "orange"), (100.0, "impossible", "red")],
        "ice_friction": [(0.1, "typical", "green"), (0.5, "extrapolation", "orange"), (2.0, "impossible", "red")],
        "ice_speed": [(2.5, "typical", "green"), (10.0, "extrapolation", "orange"), (20.0, "impossible", "red")],
        "ice_density": [(900.0, "typical", "green"), (830.0, "extrapolation", "orange"), (700.0, "impossible", "red")],
        "ice_elastic_modulus": [(5.0, "typical", "green"), (15.0, "extrapolation", "orange"), (100.0, "impossible", "red")],
        "ice_poisson_ratio": [(0.33, "typical", "green"), (0.2, "extrapolation", "orange"), (0.6, "impossible", "red")],
        "ice_temperature": [(-15.0, "typical", "green"), (-50.0, "extrapolation", "orange"), (-70.0, "impossible", "red")],
    }

    for idx, (param, title, unit) in enumerate(params):
        ax = axes[idx // 4, idx % 4]
        b = param.bounds

        # Draw bands
        ax.axhspan(b.hard_min, b.hard_max, alpha=0.1, color="gray", label="Physical bounds")
        ax.axhspan(b.soft_min, b.soft_max, alpha=0.2, color="blue", label="Calibration domain")
        ax.axhspan(b.typical_min, b.typical_max, alpha=0.3, color="green", label="Typical range")

        # Draw boundary lines
        ax.axhline(b.hard_min, color="gray", linestyle="--", linewidth=0.5)
        ax.axhline(b.hard_max, color="gray", linestyle="--", linewidth=0.5)
        ax.axhline(b.soft_min, color="blue", linestyle="-.", linewidth=0.5)
        ax.axhline(b.soft_max, color="blue", linestyle="-.", linewidth=0.5)

        # Mark test cases
        key = list(test_cases.keys())[idx]
        for value, label, color in test_cases[key]:
            ax.scatter(0.5, value, color=color, s=80, zorder=5, edgecolors="black", linewidth=0.5)
            ax.annotate(
                f"{value}",
                (0.55, value),
                fontsize=7,
                color=color,
                fontweight="bold" if label == "impossible" else "normal",
            )

        ax.set_title(title, fontsize=9)
        ax.set_ylabel(unit if unit else "dimensionless", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        if idx == 0:
            ax.legend(loc="upper right", fontsize=6)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = output_dir / "fig_ice_guardrail_bounds.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")

    # Generate validation demo
    demo_cases = [
        {"ice_thickness": 1.5, "ice_strength": 3.0, "ice_friction": 0.1},
        {"ice_thickness": 50.0},
        {"ice_strength": 100.0},
        {"ice_thickness": 6.0, "ice_strength": 2.0},
    ]

    results = []
    for case in demo_cases:
        result = validate_ice_parameters(case)
        results.append({
            "input": case,
            "is_valid": result.is_valid,
            "status": result.status.value,
            "errors": result.errors,
            "warnings": result.warnings,
        })

    json_path = output_dir / "validation_demo.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {json_path}")

    # Generate parameter report for the failure case
    report = get_parameter_report({
        "ice_thickness": 50.0,
        "ice_strength": 100.0,
        "ice_friction": 0.8,
        "ice_speed": 10.0,
    })
    report_path = output_dir / "failure_case_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Saved: {report_path}")

    return png_path, json_path, report_path


def make_validation_summary_figure(output_dir: Path = Path("artifacts/ice_guardrails")):
    """Create a summary figure showing validation results for all test cases."""
    output_dir.mkdir(parents=True, exist_ok=True)

    test_cases = [
        ("Typical Arctic", {"ice_thickness": 1.5, "ice_strength": 3.0, "ice_friction": 0.1, "ice_speed": 2.5}),
        ("h=50 m (impossible)", {"ice_thickness": 50.0}),
        ("σ_c=100 MPa (unrealistic)", {"ice_strength": 100.0}),
        ("h=6 m (extrapolation)", {"ice_thickness": 6.0}),
        ("Combined extremes", {"ice_thickness": 50.0, "ice_strength": 100.0, "ice_speed": 20.0, "ice_friction": 0.8}),
    ]

    fig, ax = plt.subplots(figsize=(10, 4))

    colors = []
    labels = []
    for name, params in test_cases:
        result = validate_ice_parameters(params)
        if result.is_valid:
            colors.append("green" if not result.extrapolation_risk else "orange")
            labels.append("PASS" if not result.extrapolation_risk else "WARN")
        else:
            colors.append("red")
            labels.append("FAIL")

    y_pos = range(len(test_cases))
    ax.barh(y_pos, [1] * len(test_cases), color=colors, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([name for name, _ in test_cases])
    ax.set_xlim(0, 1.2)
    ax.set_xlabel("Validation Result")
    ax.set_title("Ice Guardrail Validation Results")

    for i, (name, label) in enumerate(zip([n for n, _ in test_cases], labels)):
        ax.text(1.05, i, label, va="center", fontweight="bold", color=colors[i])

    ax.set_xticks([])
    plt.tight_layout()
    png_path = output_dir / "fig_validation_summary.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {png_path}")
    return png_path


if __name__ == "__main__":
    make_bounds_figure()
    make_validation_summary_figure()
