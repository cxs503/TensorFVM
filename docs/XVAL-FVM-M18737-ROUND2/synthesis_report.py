"""Synthesis report for XVAL-FVM-M18737-ROUND2.

Brings together all rounds of validation for the Woodward-Colella double blast
wave case in TensorFVM: round-1 initial validation, round-2 cross-validation
with error budget, method claim conformance audit, convergence diagnosis,
and the certified-<1% reference path.

Usage::
    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/synthesis_report.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
XVAL = REPO / "docs" / "woodward-colella-double-blast" / "xval"
if str(XVAL) not in sys.path:
    sys.path.insert(0, str(XVAL))

import wc_xval  # noqa: E402

OUTDIR = Path(__file__).resolve().parent / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)


def make_summary_figure(analysis: dict) -> str:
    """Create a 2x2 summary figure for the synthesis report."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Ladder convergence: peak and L1 vs nx
    ax = axes[0, 0]
    profiles = analysis["profiles"]
    fvm_tags = [t for t in profiles if profiles[t]["family"] == "fvm" and "hllc" in t and "minmod" not in t]
    fvm_tags.sort(key=lambda t: profiles[t]["nx"])

    nx_vals = [profiles[t]["nx"] for t in fvm_tags]
    peak_vals = [profiles[t]["peak_percent_calibrated"] for t in fvm_tags]
    l1_vals = [profiles[t]["l1_percent_calibrated"] for t in fvm_tags]

    ax.semilogx(nx_vals, peak_vals, "ro-", label="Peak density error %", markersize=8)
    ax.semilogx(nx_vals, l1_vals, "bs-", label="L1 error %", markersize=8)
    ax.axhline(10, color="gray", linestyle="--", alpha=0.7, label="Ledger tolerance (10%)")
    ax.axhline(3, color="orange", linestyle="--", alpha=0.7, label="Round-1 claim line (3%)")
    ax.set_xlabel("Grid points (nx)")
    ax.set_ylabel("Error (%)")
    ax.set_title("(a) Convergence vs calibrated reference")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    for nx, pk, l1 in zip(nx_vals, peak_vals, l1_vals):
        ax.annotate(f"{pk:.1f}%", (nx, pk), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="red")
        ax.annotate(f"{l1:.1f}%", (nx, l1), textcoords="offset points",
                    xytext=(5, -10), fontsize=7, color="blue")

    # (b) Self-convergence (reference-free)
    ax = axes[0, 1]
    sc = analysis["self_convergence"]["fvm"]
    diffs = sc["differences"]
    coarse_nx = [d["coarse_nx"] for d in diffs]
    l1_win = [d["l1_window_percent"] for d in diffs]

    ax.semilogx(coarse_nx, l1_win, "go-", markersize=8, label="Grid-to-grid L1 (window)")
    ax.set_xlabel("Coarse grid nx")
    ax.set_ylabel("Self-convergence L1 (%)")
    ax.set_title(f"(b) Reference-free self-convergence\norders: {sc['observed_orders']}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) Error budget for finest grid
    ax = axes[1, 0]
    eb = analysis["error_budget_finest"]["l1"]
    categories = ["Total", "Own (discretisation)", "Reference-side"]
    values = [eb["total_percent"], eb["own_percent"], eb["reference_side_percent"]]
    colors = ["#e74c3c", "#3498db", "#2ecc71"]
    bars = ax.bar(categories, values, color=colors, edgecolor="black")
    ax.set_ylabel("Error (%)")
    ax.set_title(f"(c) Error budget (nx={sc['finest_nx']})")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{val:.2f}%", ha="center", fontsize=9, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    # (d) Cross-code convergence
    ax = axes[1, 1]
    cc = analysis["cross_code"]
    cc_nx = sorted(cc.keys())
    cc_l1 = [cc[nx]["l1_between_percent"] for nx in cc_nx]
    cc_peak = [cc[nx]["peak_delta_percent"] for nx in cc_nx]

    ax.semilogx(cc_nx, cc_l1, "mo-", markersize=8, label="L1 between MUSCL & WENO5")
    ax.semilogx(cc_nx, cc_peak, "c^-", markersize=8, label="Peak delta")
    ax.set_xlabel("Grid points (nx)")
    ax.set_ylabel("Inter-code difference (%)")
    ax.set_title("(d) Cross-code convergence\n(independent discretisations)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = OUTDIR / "fig_synthesis_summary.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(png_path.relative_to(OUTDIR.parent.parent))


def make_verdict_table(analysis: dict) -> str:
    """Generate a markdown verdict table."""
    lines = []
    lines.append("| 网格 (nx) | 峰值误差 % | L1 误差 % | 质量积分 % | 最陡面位置误差 | 判定 |")
    lines.append("|---|---|---|---|---|---|")
    profiles = analysis["profiles"]
    fvm_tags = [t for t in profiles if profiles[t]["family"] == "fvm" and "hllc" in t and "minmod" not in t]
    fvm_tags.sort(key=lambda t: profiles[t]["nx"])

    for t in fvm_tags:
        p = profiles[t]
        verdict = "PASS" if p["verdict_calibrated"] == "PASS" else "FAIL"
        lines.append(
            f"| {p['nx']} | {p['peak_percent_calibrated']:.2f} | "
            f"{p['l1_percent_calibrated']:.2f} | {p['mass_percent_calibrated']:.3f} | "
            f"{p['peak_position_error']:.4f} | {verdict} |"
        )

    # Add cross-code entries
    cc = analysis["cross_code"]
    for nx in sorted(cc.keys()):
        e = cc[nx]
        lines.append(
            f"| WENO5@{nx} vs MUSCL | — | {e['l1_between_percent']:.3f} (互差) | "
            f"— | {e['peak_delta_percent']:.3f} (峰值互差) | — |"
        )

    return "\n".join(lines)


def main():
    analysis = wc_xval.run()

    # Save JSON
    json_path = OUTDIR / "synthesis_analysis.json"
    json_path.write_text(json.dumps(analysis, indent=2, default=str))

    # Generate figures
    fig_path = make_summary_figure(analysis)

    # Generate verdict table
    verdict_md = make_verdict_table(analysis)

    # Print summary
    print(f"Analysis JSON: {json_path}")
    print(f"Summary figure: {fig_path}")
    print()
    print("=== Verdict Table ===")
    print(verdict_md)
    print()

    # Key findings
    sc = analysis["self_convergence"]["fvm"]
    eb = analysis["error_budget_finest"]["l1"]
    pe = analysis["peak_extrapolation_fvm"]

    print("=== Key Findings ===")
    print(f"Self-convergence (finest nx={sc['finest_nx']}): L1 = {sc['finest_self_l1_percent']:.3f}%")
    print(f"Error estimate at finest: {sc['finest_error_estimate_percent']:.3f}%")
    print(f"Observed orders: {sc['observed_orders']}")
    print()
    print(f"Error budget (nx={sc['finest_nx']}):")
    print(f"  Total: {eb['total_percent']:.3f}%")
    print(f"  Own discretisation: {eb['own_percent']:.3f}%")
    print(f"  Reference-side: {eb['reference_side_percent']:.3f}%")
    print()
    print(f"Peak extrapolation:")
    print(f"  Order p = {pe['order']:.3f}")
    print(f"  Continuum limit ρ_max = {pe['peak_infinity']:.4f}")
    print(f"  Reference peak = {analysis['reference']['calibrated']['peak']:.4f}")
    gap = 100 * (pe["peak_infinity"] - analysis["reference"]["calibrated"]["peak"]) / analysis["reference"]["calibrated"]["peak"]
    print(f"  Gap vs reference = {gap:.2f}%")


if __name__ == "__main__":
    main()
