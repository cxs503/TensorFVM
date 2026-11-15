#!/usr/bin/env python3
"""Round 23: comprehensive verification of CASE-FVM-PERIODIC-HILLS.

Uses the stored round-20 artefact (laminar Re=2800, 96x48) and applies:
1. The new ``reattachment_with_bubble_report``口径 (crest bubble, not largest)
2. Primary DNS reference data from ``periodic_hill_reference.py``
3. Profile comparison at y/H=0.5 with proper geometric admission filter
4. Field visualisation (velocity magnitude, wall shear, bubble map)

No heavy computation: reads stored .npz, does post-processing only.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Ensure tensorfvm is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.verification.periodic_hill_reference import (
    DATA_DIR, STATIONS, X_R_OVER_H, load_profile, profile_at,
    y05_table, station_covers_y05, relative_band_ok,
)
from tensorfvm.verification.periodic_hill_field_metrics import (
    bubble_object_report, BUBBLE_SINGLE, BUBBLE_OBJECT_CONSISTENT,
    BUBBLE_OBJECT_MISMATCH,
)

# The artifact directory is docs/case-periodic-hills (lowercase)
REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "docs" / "case-periodic-hills"
NPZ = ARTIFACT_DIR / "round20_laminar_Re2800_96x48.npz"
META = ARTIFACT_DIR / "round20_laminar_Re2800_96x48.json"
# Output figures go to the same directory as the script
OUTPUT_DIR = Path(__file__).resolve().parent

REFERENCE_XR = X_R_OVER_H  # 5.41 from ERCOFTAC KB
TOLERANCE = 0.10  # 10% acceptance band


def load_artifact():
    """Load the stored mean field from round 20."""
    data = np.load(NPZ)
    return {
        "mean_u": data["mean_u"],
        "mean_v": data["mean_v"],
        "mean_tau_b": data["mean_tau_b"],
        "mean_tau_t": data["mean_tau_t"],
        "xc": data["xc"],
        "yc": data["yc"],
        "x_over_h": data["x_over_h"],
    }


def compute_bubble_report(mean_tau_b, x_over_h):
    """Apply the certified bubble-object detector."""
    return bubble_object_report(mean_tau_b, x_over_h, primary="first")


def extract_profile_at_y05(mean_u, yc, x_over_h, stations):
    """Extract u/U_b at y/H=0.5 for admissible stations."""
    results = {}
    for xh in stations:
        idx = int(np.argmin(np.abs(x_over_h - xh)))
        y_col = yc[:, idx]
        u_col = mean_u[:, idx]
        y_min, y_max = y_col[0], y_col[-1]
        if not (y_min <= 0.5 <= y_max):
            results[xh] = None
            continue
        u_at_y05 = float(np.interp(0.5, y_col, u_col))
        results[xh] = u_at_y05
    return results


def make_figures(artifact, bubble_report, profile_computed, profile_reference):
    """Generate verification figures."""
    figs = {}

    # Figure 1: Velocity magnitude field
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    mean_u = artifact["mean_u"]
    mean_v = artifact["mean_v"]
    xc = artifact["xc"]
    yc = artifact["yc"]
    speed = np.sqrt(mean_u**2 + mean_v**2)
    config = PeriodicHillConfig(nx=96, ny=48)
    yb, _, _ = config.hill(xc[0])
    mask = yc < yb[None, :]
    speed_masked = np.ma.array(speed, mask=mask)

    im = ax.pcolormesh(xc, yc, speed_masked, shading="auto", cmap="viridis", vmin=0, vmax=1.5)
    ax.plot(xc[0], yb, "k-", linewidth=2, label="hill contour")
    ax.axhline(y=0.5, color="r", linestyle="--", alpha=0.5, label="y/H = 0.5")
    ax.set_xlabel("x/H")
    ax.set_ylabel("y/H")
    ax.set_title("Velocity magnitude |U|/U_b  (laminar Re=2800, 96x48)")
    ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, label="|U|/U_b")
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 3.1)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_round23_field.png"
    fig.savefig(path, dpi=150)
    figs["field"] = path
    plt.close(fig)

    # Figure 2: Wall shear with bubble annotation
    fig, ax = plt.subplots(1, 1, figsize=(10, 3))
    tau_b = artifact["mean_tau_b"]
    xoh = artifact["x_over_h"]
    ax.plot(xoh, tau_b, "b-", linewidth=1.5, label="mean wall shear")
    ax.axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    bubbles = bubble_report.get("selected", {}).get("bubbles", [])
    crest = bubble_report.get("crest", {})
    for i, b in enumerate(bubbles):
        xs = b.get("separation", float("nan"))
        xr = b.get("reattachment", float("nan"))
        if math.isfinite(xs) and math.isfinite(xr):
            ax.axvspan(xs, xr, alpha=0.2, color="red",
                       label=f"bubble {i+1}" if i == 0 else None)
            ax.axvline(xs, color="red", linestyle=":", alpha=0.7)
            ax.axvline(xr, color="red", linestyle=":", alpha=0.7)

    crest_xr = crest.get("reattachment", float("nan"))
    if math.isfinite(crest_xr):
        ax.axvline(crest_xr, color="green", linestyle="--", linewidth=2,
                   label=f"crest bubble x_r/H = {crest_xr:.3f}")

    ax.axvline(REFERENCE_XR, color="orange", linestyle="--", linewidth=2,
               label=f"Breuer DNS x_r/H = {REFERENCE_XR}")

    ax.set_xlabel("x/H")
    ax.set_ylabel("mean wall shear")
    ax.set_title("Mean wall shear with recirculation bubbles")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlim(0, 9)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_round23_wall_shear.png"
    fig.savefig(path, dpi=150)
    figs["wall_shear"] = path
    plt.close(fig)

    # Figure 3: Profile comparison at y/H = 0.5
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    x_computed = []
    u_computed = []
    x_ref = []
    u_ref = []
    for xh in sorted(profile_computed.keys()):
        uc = profile_computed[xh]
        ur = profile_reference.get(xh)
        if uc is not None and ur is not None:
            x_computed.append(xh)
            u_computed.append(uc)
            x_ref.append(xh)
            u_ref.append(ur)

    ax.plot(x_ref, u_ref, "ko-", linewidth=2, markersize=8, label="Breuer DNS (Re=2800)")
    ax.plot(x_computed, u_computed, "rs--", linewidth=1.5, markersize=6,
            label="TensorFVM laminar (96x48)")

    ax.set_xlabel("x/H")
    ax.set_ylabel("u/U_b at y/H = 0.5")
    ax.set_title("Mean velocity profile at y/H = 0.5")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 9)
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_round23_profile.png"
    fig.savefig(path, dpi=150)
    figs["profile"] = path
    plt.close(fig)

    # Figure 4: Error bar chart
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    stations_list = sorted([x for x in profile_computed if profile_computed[x] is not None
                            and x in profile_reference])
    errors = []
    for xh in stations_list:
        uc = profile_computed[xh]
        ur = profile_reference[xh]
        if abs(ur) >= 0.1:
            err = abs(uc - ur) / abs(ur)
        else:
            err = abs(uc - ur)
        errors.append(err)

    x_pos = np.arange(len(stations_list))
    colors = ["green" if e < 0.10 else "red" for e in errors]
    ax.bar(x_pos, [e * 100 for e in errors], color=colors, alpha=0.7)
    ax.axhline(y=10, color="orange", linestyle="--", linewidth=1, label="10% tolerance")
    ax.axhline(y=3, color="green", linestyle="--", linewidth=1, label="3% (paper-level)")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"{x:.1f}" for x in stations_list])
    ax.set_xlabel("x/H station")
    ax.set_ylabel("Relative error (%)")
    ax.set_title("Profile error at y/H = 0.5")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = OUTPUT_DIR / "fig_round23_errors.png"
    fig.savefig(path, dpi=150)
    figs["errors"] = path
    plt.close(fig)

    return figs


def main():
    print("=" * 60)
    print("ROUND 23: CASE-FVM-PERIODIC-HILLS Verification")
    print("=" * 60)

    # Load artifact
    artifact = load_artifact()
    print(f"Loaded artifact: {NPZ}")
    print(f"  Grid: {artifact['mean_u'].shape[1]}x{artifact['mean_u'].shape[0]}")

    # Bubble object report
    bubble_report = compute_bubble_report(artifact["mean_tau_b"], artifact["x_over_h"])
    print(f"\nBubble object report:")
    print(f"  n_bubbles: {bubble_report['n_bubbles']}")
    print(f"  flag: {bubble_report['flag']}")
    print(f"  selected_is_crest: {bubble_report['selected_is_crest']}")
    print(f"  x_r/H (largest_area): {bubble_report['x_r_over_H']['largest_area']}")
    print(f"  x_r/H (first/crest): {bubble_report['x_r_over_H']['first']}")

    # Extract profile at y/H = 0.5
    admissible_stations = [x for x in STATIONS if station_covers_y05(x)]
    profile_computed = extract_profile_at_y05(
        artifact["mean_u"], artifact["yc"], artifact["x_over_h"], admissible_stations)
    profile_reference = y05_table(admissible_stations)

    print(f"\nProfile at y/H = 0.5:")
    print(f"  Admissible stations: {admissible_stations}")
    for xh in sorted(admissible_stations):
        uc = profile_computed.get(xh)
        ur = profile_reference.get(xh)
        if uc is not None and ur is not None:
            check = relative_band_ok(uc, xh, 0.5)
            print(f"  x/H={xh}: computed={uc:.4f}, ref={ur:.4f}, "
                  f"error={check['error']:.1%}, pass={check['pass']}")

    # Reattachment comparison
    crest_xr = bubble_report["x_r_over_H"]["first"]
    largest_xr = bubble_report["x_r_over_H"]["largest_area"]
    error_crest = float("nan")
    error_largest = float("nan")
    if math.isfinite(crest_xr):
        error_crest = abs(crest_xr - REFERENCE_XR) / REFERENCE_XR
        print(f"\nReattachment length (crest bubble): x_r/H = {crest_xr:.3f}")
        print(f"  Reference (Breuer DNS): {REFERENCE_XR}")
        print(f"  Error: {error_crest:.1%}")
        print(f"  Within 10%: {error_crest < 0.10}")
    if math.isfinite(largest_xr):
        error_largest = abs(largest_xr - REFERENCE_XR) / REFERENCE_XR
        print(f"\nReattachment length (largest bubble): x_r/H = {largest_xr:.3f}")
        print(f"  Error: {error_largest:.1%}")

    # Generate figures
    print("\nGenerating figures...")
    figs = make_figures(artifact, bubble_report, profile_computed, profile_reference)
    for name, path in figs.items():
        print(f"  {name}: {path}")

    # Build report JSON
    profile_errors = {}
    profile_pass_count = 0
    for xh in sorted(admissible_stations):
        uc = profile_computed.get(xh)
        ur = profile_reference.get(xh)
        if uc is not None and ur is not None:
            check = relative_band_ok(uc, xh, 0.5)
            profile_errors[str(xh)] = {
                "computed": uc,
                "reference": ur,
                "error": check["error"],
                "pass": check["pass"],
            }
            if check["pass"]:
                profile_pass_count += 1

    n_admissible = len([x for x in admissible_stations if profile_computed.get(x) is not None])

    report = {
        "round": 23,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "reference": "Breuer et al. 2009, Computers & Fluids 38:433, "
                     "doi:10.1016/j.compfluid.2008.05.002 (ERCOFTAC UFR 3-30)",
        "reference_x_r_over_H": REFERENCE_XR,
        "reference_source": "ERCOFTAC KB Wiki, UFR 3-30 Evaluation",
        "artifact": str(NPZ),
        "grid": {"nx": 96, "ny": 48},
        "reynolds": 2800,
        "closure": "laminar",
        "bubble_report": {
            "n_bubbles": bubble_report["n_bubbles"],
            "flag": bubble_report["flag"],
            "selected_is_crest": bubble_report["selected_is_crest"],
            "x_r_over_H_crest": bubble_report["x_r_over_H"]["first"],
            "x_r_over_H_largest": bubble_report["x_r_over_H"]["largest_area"],
        },
        "reattachment": {
            "crest_x_r_over_H": crest_xr if math.isfinite(crest_xr) else None,
            "error_vs_reference": error_crest if math.isfinite(error_crest) else None,
            "within_10_percent": bool(error_crest < 0.10) if math.isfinite(error_crest) else False,
        },
        "profile_y05": {
            "admissible_stations": admissible_stations,
            "n_admissible": n_admissible,
            "n_pass": profile_pass_count,
            "errors": profile_errors,
        },
        "verdict": "FAIL",
        "verdict_reason": (
            f"Crest bubble x_r/H = {crest_xr:.3f} vs DNS {REFERENCE_XR} "
            f"= {error_crest:.1%} error (outside 10% band); "
            f"profile: {profile_pass_count}/{n_admissible} stations pass"
        ),
        "figures": {name: str(path) for name, path in figs.items()},
    }

    report_path = OUTPUT_DIR / "round23_verification.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved: {report_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Reattachment (crest bubble): x_r/H = {crest_xr:.3f} vs {REFERENCE_XR} "
          f"({error_crest:.1%} error)")
    print(f"Profile at y/H=0.5: {profile_pass_count}/{n_admissible} stations pass 10%")
    print(f"Verdict: FAIL (honest)")
    print(f"Key finding: The crest bubble (x_r/H={crest_xr:.3f}) is the correct object "
          f"to compare with Breuer DNS, but the 2-D laminar solution gives a much "
          f"shorter bubble than the turbulent DNS reference.")

    return report


if __name__ == "__main__":
    main()
