#!/usr/bin/env python
"""Assemble full benchmark summary and generate figures."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.internal_flow.reference import DE_VAHL_DAVIS_NU, DE_VAHL_DAVIS_VELOCITY

OUT = Path("artifacts/heated-cavity-benchmark")


def assemble_summary() -> dict:
    summary = {
        "benchmark": "de Vahl Davis (1983) heated cavity",
        "reference": "Int. J. Numer. Methods Fluids 3, 249-264",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "reference_nu": {str(int(k)): v for k, v in DE_VAHL_DAVIS_NU.items()},
        "reference_velocity": {str(int(k)): v for k, v in DE_VAHL_DAVIS_VELOCITY.items()},
        "cases": {},
        "acceptance": {},
    }

    for ra in [1e3, 1e4, 1e5, 1e6]:
        case_path = OUT / f"case_Ra{int(ra)}.json"
        if case_path.exists():
            data = json.loads(case_path.read_text())
            summary["cases"][str(int(ra))] = data["case"]
            summary["acceptance"][str(int(ra))] = data["acceptance"]

    summary_path = OUT / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary written to {summary_path}")
    return summary


def generate_figures(summary: dict):
    """Generate comparison figures."""
    figs = []

    # Figure 1: Nu comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 5))
    ras = [1e3, 1e4, 1e5, 1e6]
    ra_labels = ["10³", "10⁴", "10⁵", "10⁶"]
    nu_refs = [DE_VAHL_DAVIS_NU[ra] for ra in ras]
    nu_gots = []
    for ra in ras:
        acc = summary["acceptance"].get(str(int(ra)), {})
        nu_gots.append(acc.get("nu_computed", 0))

    x = np.arange(len(ras))
    width = 0.35
    bars1 = ax.bar(x - width/2, nu_refs, width, label='de Vahl Davis (1983)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, nu_gots, width, label='TensorFVM (n=128)', color='coral', alpha=0.8)

    ax.set_xlabel('Rayleigh Number (Ra)', fontsize=12)
    ax.set_ylabel('Mean Nusselt Number (Nu)', fontsize=12)
    ax.set_title('Heated Cavity: Mean Nusselt Number Comparison', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Ra={l}' for l in ra_labels])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add error annotations
    for i, (ref, got) in enumerate(zip(nu_refs, nu_gots)):
        err = abs(got - ref) / ref * 100
        ax.annotate(f'{err:.2f}%', xy=(i + width/2, got),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', fontsize=9, color='darkgreen')

    fig.tight_layout()
    fig_path = OUT / "nu_comparison.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    figs.append({"name": "nu_comparison", "path": str(fig_path)})
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Figure 2: Nu error vs mesh resolution
    fig, ax = plt.subplots(figsize=(8, 5))
    meshes = [32, 64, 128]
    colors = {'1000.0': 'blue', '10000.0': 'green', '100000.0': 'orange', '1000000.0': 'red'}
    markers = {'1000.0': 'o', '10000.0': 's', '100000.0': '^', '1000000.0': 'D'}

    for ra_str, case_data in summary["cases"].items():
        ra = float(ra_str)
        ref_nu = DE_VAHL_DAVIS_NU.get(ra, 0)
        errors = []
        for n in meshes:
            mesh_data = case_data["meshes"].get(str(n), {})
            nu_mean = mesh_data.get("nu_mean", 0)
            if ref_nu > 0 and nu_mean > 0:
                errors.append(abs(nu_mean - ref_nu) / ref_nu * 100)
            else:
                errors.append(0)
        ax.plot(meshes, errors, marker=markers.get(ra_str, 'o'),
                color=colors.get(ra_str, 'gray'), label=f'Ra={ra_str}', linewidth=2)

    ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% tolerance')
    ax.axhline(y=3, color='orange', linestyle='--', alpha=0.5, label='3% tolerance')
    ax.set_xlabel('Mesh Resolution (n × n)', fontsize=12)
    ax.set_ylabel('Relative Error in Nu (%)', fontsize=12)
    ax.set_title('Mesh Convergence: Nu Error vs Resolution', fontsize=13)
    ax.set_xscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig_path = OUT / "nu_convergence.png"
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    figs.append({"name": "nu_convergence", "path": str(fig_path)})
    plt.close(fig)
    print(f"Figure saved: {fig_path}")

    # Figure 3: Nu profile along hot wall (for Ra=1e5 as example)
    ra_example = 1e5
    case_data = summary["cases"].get(str(int(ra_example)), {})
    finest_mesh = 128
    mesh_data = case_data.get("meshes", {}).get(str(finest_mesh), {})
    if mesh_data.get("nu_local"):
        fig, ax = plt.subplots(figsize=(8, 5))
        y = np.array(mesh_data["y"])
        nu_local = np.array(mesh_data["nu_local"])
        ax.plot(nu_local, y, 'b-', linewidth=2, label=f'TensorFVM Ra={int(ra_example)}, n={finest_mesh}')
        ax.set_xlabel('Local Nusselt Number', fontsize=12)
        ax.set_ylabel('y/H', fontsize=12)
        ax.set_title(f'Local Nusselt Number Profile on Hot Wall (Ra={int(ra_example)})', fontsize=13)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)

        fig.tight_layout()
        fig_path = OUT / "nu_profile_Ra1e5.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        figs.append({"name": "nu_profile_Ra1e5", "path": str(fig_path)})
        plt.close(fig)
        print(f"Figure saved: {fig_path}")

    # Figure 4: Velocity extrema comparison
    fig, ax = plt.subplots(figsize=(8, 5))
    u_refs = [DE_VAHL_DAVIS_VELOCITY[ra]["u_max"] for ra in ras if ra in DE_VAHL_DAVIS_VELOCITY]
    v_refs = [DE_VAHL_DAVIS_VELOCITY[ra]["v_max"] for ra in ras if ra in DE_VAHL_DAVIS_VELOCITY]
    u_gots = []
    v_gots = []
    valid_ras = []
    for ra in ras:
        acc = summary["acceptance"].get(str(int(ra)), {})
        if acc.get("u_max_reference"):
            u_gots.append(acc["u_max_computed"])
            v_gots.append(acc["v_max_computed"])
            valid_ras.append(ra)

    if valid_ras:
        x = np.arange(len(valid_ras))
        ra_labels_v = [f"10^{int(np.log10(r))}" for r in valid_ras]
        width = 0.35
        ax.bar(x - width/2, u_refs[:len(valid_ras)], width, label='u_max (ref)', color='steelblue', alpha=0.8)
        ax.bar(x + width/2, u_gots, width, label='u_max (TensorFVM)', color='lightblue', alpha=0.8)
        ax.bar(x - width/2, v_refs[:len(valid_ras)], width, label='v_max (ref)', bottom=u_refs[:len(valid_ras)], color='coral', alpha=0.8)
        ax.bar(x + width/2, v_gots, width, label='v_max (TensorFVM)', bottom=u_gots, color='lightsalmon', alpha=0.8)

        ax.set_xlabel('Rayleigh Number', fontsize=12)
        ax.set_ylabel('Velocity Extrema (|u|_max, |v|_max) / (α/H)', fontsize=11)
        ax.set_title('Velocity Extrema Comparison', fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([f'Ra={l}' for l in ra_labels_v])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        fig.tight_layout()
        fig_path = OUT / "velocity_comparison.png"
        fig.savefig(fig_path, dpi=150, bbox_inches='tight')
        figs.append({"name": "velocity_comparison", "path": str(fig_path)})
        plt.close(fig)
        print(f"Figure saved: {fig_path}")

    return figs


def main():
    summary = assemble_summary()

    # Print acceptance table
    print("\n" + "="*80)
    print("ACCEPTANCE TABLE: de Vahl Davis (1983) Heated Cavity Benchmark")
    print("="*80)
    print(f"{'Ra':>8} | {'Nu_ref':>8} | {'Nu_got':>8} | {'Nu_err%':>8} | {'u_err%':>8} | {'v_err%':>8} | Status")
    print("-"*80)
    for ra in [1e3, 1e4, 1e5, 1e6]:
        acc = summary["acceptance"].get(str(int(ra)), {})
        ref_nu = acc.get("nu_reference", 0)
        got_nu = acc.get("nu_computed", 0)
        nu_err = acc.get("nu_relative_error", 0)
        u_err = acc.get("u_max_relative_error", 0)
        v_err = acc.get("v_max_relative_error", 0)
        status = "PASS" if acc.get("nu_within_5_percent") else "FAIL"
        print(f"{int(ra):>8} | {ref_nu:>8.4f} | {got_nu:>8.4f} | {nu_err*100:>7.2f}% | {u_err*100:>7.2f}% | {v_err*100:>7.2f}% | {status}")
    print("="*80)

    figs = generate_figures(summary)
    print(f"\nGenerated {len(figs)} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
