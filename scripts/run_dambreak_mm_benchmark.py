#!/usr/bin/env python
"""CASE-FVM-DAMBREAK-WET: Martin & Moyce (1952) dam-break wet-bed benchmark.

Runs the TensorFVM two-phase VOF solver for the canonical dam-break problem
and compares surge-front trajectory and residual column height against the
Martin & Moyce experimental data.

Reference:
    J.C. Martin, W.J. Moyce, "An experimental study of the collapse of liquid
    columns on a rigid horizontal plane", Phil. Trans. R. Soc. Lond. A 244
    (1952) 312-324.

Dimensionless scaling (per Martin & Moyce):
    t* = t * sqrt(g / a),   x* = x / a,   h* = h / a
where a = 0.05715 m (2.25 inches) is the initial column half-width.

Usage:
    python scripts/run_dambreak_mm_benchmark.py [--nx NX] [--ny NY] [--max-time T]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure tensorfvm is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from tensorfvm.verification.dam_break import (
    DamBreakConfig,
    run_dam_break,
    compare_against_mm,
    mm_surge_reference,
    mm_height_reference,
    MM_SURGE_T,
    MM_SURGE_X,
    MM_HEIGHT_T,
    MM_HEIGHT_H,
)


def run_benchmark(nx: int = 200, ny: int = 60, max_time: float = 0.6,
                  output_dir: str = "artifacts") -> dict:
    """Run the dam-break benchmark and return results dict."""
    cfg = DamBreakConfig(
        column_width=0.05715,
        domain_length=0.5,
        domain_height=0.15,
        nx=nx,
        ny=ny,
        gravity=9.81,
        rho_liquid=1000.0,
        rho_gas=1.2,
        mu_liquid=1.0e-3,
        mu_gas=1.8e-5,
        sigma=0.0,
        vof_cfl=0.4,
        max_time=max_time,
        device="cpu",
    )

    print(f"Running dam-break benchmark: nx={nx}, ny={ny}, max_time={max_time}s")
    t0 = time.perf_counter()
    result = run_dam_break(cfg)
    elapsed = time.perf_counter() - t0
    print(f"Simulation completed in {elapsed:.1f}s ({result['steps']} steps)")

    # Compare against Martin & Moyce
    metrics = compare_against_mm(
        result["surge_front"],
        result["column_height"],
        cfg.column_width,
        cfg.gravity,
    )

    # Build detailed comparison tables
    a = cfg.column_width
    g = cfg.gravity
    t_scale = math.sqrt(a / g)

    surge_comparison = []
    for t, x in result["surge_front"]:
        t_star = t / t_scale
        if t_star < 0.1:
            continue
        x_star = x / a
        x_ref = mm_surge_reference(t_star)
        if x_ref > 0:
            err = abs(x_star - x_ref) / x_ref
            surge_comparison.append({
                "t": round(t, 6),
                "t_star": round(t_star, 4),
                "x_star": round(x_star, 4),
                "x_ref": round(x_ref, 4),
                "error_pct": round(err * 100, 2),
            })

    height_comparison = []
    for t, h in result["column_height"]:
        t_star = t / t_scale
        if t_star < 0.1:
            continue
        h_star = h / a
        h_ref = mm_height_reference(t_star)
        if h_ref > 0:
            err = abs(h_star - h_ref) / h_ref
            height_comparison.append({
                "t": round(t, 6),
                "t_star": round(t_star, 4),
                "h_star": round(h_star, 4),
                "h_ref": round(h_ref, 4),
                "error_pct": round(err * 100, 2),
            })

    # Compute summary statistics
    surge_errors = [c["error_pct"] for c in surge_comparison]
    height_errors = [c["error_pct"] for c in height_comparison]

    summary = {
        "case": "CASE-FVM-DAMBREAK-WET",
        "reference": "Martin & Moyce (1952) Phil. Trans. R. Soc. Lond. A 244:312-324",
        "config": {
            "nx": nx,
            "ny": ny,
            "max_time": max_time,
            "column_width_m": cfg.column_width,
            "domain_length_m": cfg.domain_length,
            "domain_height_m": cfg.domain_height,
            "gravity": cfg.gravity,
            "rho_liquid": cfg.rho_liquid,
            "rho_gas": cfg.rho_gas,
            "mu_liquid": cfg.mu_liquid,
            "mu_gas": cfg.mu_gas,
            "vof_cfl": cfg.vof_cfl,
        },
        "runtime_s": round(elapsed, 1),
        "steps": result["steps"],
        "surge_front": {
            "n_points": len(surge_comparison),
            "mean_error_pct": round(float(np.mean(surge_errors)), 2) if surge_errors else None,
            "max_error_pct": round(float(np.max(surge_errors)), 2) if surge_errors else None,
            "comparison": surge_comparison,
        },
        "column_height": {
            "n_points": len(height_comparison),
            "mean_error_pct": round(float(np.mean(height_errors)), 2) if height_errors else None,
            "max_error_pct": round(float(np.max(height_errors)), 2) if height_errors else None,
            "comparison": height_comparison,
        },
        "metrics": [
            {"name": m.name, "value": round(m.value, 4), "limit": m.limit, "pass": m.value < m.limit}
            for m in metrics
        ],
    }

    # Save results
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "dambreak_mm_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def generate_figures(summary: dict, output_dir: str = "artifacts"):
    """Generate comparison figures."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    surge_data = summary["surge_front"]["comparison"]
    height_data = summary["column_height"]["comparison"]

    if not surge_data:
        print("No surge data to plot")
        return

    # Figure 1: Surge front comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    # Martin & Moyce reference
    t_star_ref = np.linspace(0, 8, 200)
    x_star_ref = [mm_surge_reference(t) for t in t_star_ref]
    ax.plot(t_star_ref, x_star_ref, "k-", linewidth=2, label="Martin & Moyce (1952) reference")

    # Reference data points
    ax.scatter(MM_SURGE_T, MM_SURGE_X, c="black", marker="o", s=50, zorder=5,
               label="M&M digitised data")

    # Simulation results
    t_star_sim = [d["t_star"] for d in surge_data]
    x_star_sim = [d["x_star"] for d in surge_data]
    ax.plot(t_star_sim, x_star_sim, "r-", linewidth=1.5, label=f"TensorFVM VOF (nx={summary['config']['nx']}, ny={summary['config']['ny']})")
    ax.scatter(t_star_sim, x_star_sim, c="red", marker="s", s=30, zorder=5)

    ax.set_xlabel(r"Dimensionless time $t^* = t\sqrt{g/a}$", fontsize=12)
    ax.set_ylabel(r"Dimensionless surge front $x^* = x/a$", fontsize=12)
    ax.set_title("Dam-Break Surge Front: TensorFVM vs Martin & Moyce (1952)", fontsize=14)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add error annotation
    mean_err = summary["surge_front"]["mean_error_pct"]
    max_err = summary["surge_front"]["max_error_pct"]
    ax.text(0.95, 0.05, f"Mean error: {mean_err:.1f}%\nMax error: {max_err:.1f}%",
            transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()
    fig.savefig(out_dir / "surge_front_comparison.png", dpi=150)
    plt.close()

    # Figure 2: Surge front error vs time
    if surge_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        errors = [d["error_pct"] for d in surge_data]
        t_stars = [d["t_star"] for d in surge_data]
        ax.plot(t_stars, errors, "b-", linewidth=1.5)
        ax.scatter(t_stars, errors, c="blue", marker="o", s=30)
        ax.axhline(y=8.0, color="red", linestyle="--", linewidth=1, label="8% tolerance")
        ax.axhline(y=15.0, color="orange", linestyle="--", linewidth=1, label="15% tolerance")
        ax.set_xlabel(r"Dimensionless time $t^*$", fontsize=12)
        ax.set_ylabel("Relative error (%)", fontsize=12)
        ax.set_title("Surge Front Position Error vs Time", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        fig.savefig(out_dir / "surge_front_error.png", dpi=150)
        plt.close()

    # Figure 3: Column height comparison
    if height_data:
        fig, ax = plt.subplots(figsize=(10, 6))

        t_star_ref = np.linspace(0, 5, 200)
        h_star_ref = [mm_height_reference(t) for t in t_star_ref]
        ax.plot(t_star_ref, h_star_ref, "k-", linewidth=2, label="Martin & Moyce (1952) reference")

        ax.scatter(MM_HEIGHT_T, MM_HEIGHT_H, c="black", marker="o", s=50, zorder=5,
                   label="M&M digitised data")

        t_star_sim = [d["t_star"] for d in height_data]
        h_star_sim = [d["h_star"] for d in height_data]
        ax.plot(t_star_sim, h_star_sim, "r-", linewidth=1.5, label="TensorFVM VOF")
        ax.scatter(t_star_sim, h_star_sim, c="red", marker="s", s=30, zorder=5)

        ax.set_xlabel(r"Dimensionless time $t^* = t\sqrt{g/a}$", fontsize=12)
        ax.set_ylabel(r"Dimensionless column height $h^* = h/a$", fontsize=12)
        ax.set_title("Dam-Break Residual Column Height: TensorFVM vs Martin & Moyce (1952)", fontsize=14)
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)

        mean_err = summary["column_height"]["mean_error_pct"]
        ax.text(0.95, 0.05, f"Mean error: {mean_err:.1f}%",
                transform=ax.transAxes, fontsize=10, verticalalignment="bottom",
                horizontalalignment="right",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

        plt.tight_layout()
        fig.savefig(out_dir / "column_height_comparison.png", dpi=150)
        plt.close()

    print(f"Figures saved to {out_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Martin & Moyce dam-break benchmark")
    parser.add_argument("--nx", type=int, default=200, help="Grid resolution in x")
    parser.add_argument("--ny", type=int, default=60, help="Grid resolution in y")
    parser.add_argument("--max-time", type=float, default=0.6, help="Max simulation time (s)")
    parser.add_argument("--output-dir", type=str, default="artifacts", help="Output directory")
    args = parser.parse_args()

    summary = run_benchmark(nx=args.nx, ny=args.ny, max_time=args.max_time,
                            output_dir=args.output_dir)
    generate_figures(summary, output_dir=args.output_dir)

    # Print verdict
    print("\n=== VERDICT ===")
    for m in summary["metrics"]:
        status = "PASS" if m["pass"] else "FAIL"
        print(f"  {m['name']}: {m['value']:.4f} (limit: {m['limit']}) [{status}]")

    surge_pass = summary["surge_front"]["mean_error_pct"] < 8.0 if summary["surge_front"]["mean_error_pct"] else False
    print(f"\n  Surge front mean error < 8%: {'PASS' if surge_pass else 'FAIL'} "
          f"({summary['surge_front']['mean_error_pct']:.1f}%)")

    return 0 if surge_pass else 1


if __name__ == "__main__":
    sys.exit(main())
