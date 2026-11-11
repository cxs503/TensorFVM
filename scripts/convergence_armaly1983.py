#!/usr/bin/env python3
"""Grid convergence analysis for the Armaly (1983) BFS reproduction.

Reads JSON artifacts from multiple grid levels and computes:
- Grid convergence index (GCI) following Roache (1994)
- Observed order of accuracy
- Extrapolated zero-grid-spacing value
- Error estimates at each level

Usage:
    python scripts/convergence_armaly1983.py --runs runs/armaly_cluster_v2 --out docs/repro-armaly1983-bfs/convergence
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_records(directories: list[str]) -> list[dict]:
    """Load all armaly JSON artifacts from the given directories."""
    records = {}
    for directory in directories:
        for path in sorted(Path(directory).glob("armaly1983_re*.json")):
            record = json.loads(path.read_text())
            key = (record["reynolds_dh"], record["level"],
                   record["downstream_step_heights"])
            records[key] = record
    return [records[key] for key in sorted(records)]


def grid_spacing(level: str) -> float:
    """Return the nominal grid spacing in step heights for each level."""
    # BFS_LEVELS: coarse=(16,17), medium=(33,35), fine=(49,52)
    # dy = 1.0 / n_s where n_s is cells per step height
    mapping = {"coarse": 1.0 / 16, "medium": 1.0 / 33, "fine": 1.0 / 49}
    return mapping.get(level, float("nan"))


def gci_analysis(values: list[float], spacings: list[float],
                 safety_factor: float = 1.25) -> dict:
    """Richardson extrapolation and GCI following Roache (1994).

    Requires at least 3 grid levels with monotonic spacing refinement.
    """
    if len(values) < 3 or len(spacings) < 3:
        return {"status": "insufficient_data", "n_levels": len(values)}

    # Sort by spacing (coarsest first)
    pairs = sorted(zip(spacings, values), key=lambda x: x[0], reverse=True)
    h1, h2, h3 = pairs[0][0], pairs[1][0], pairs[2][0]
    f1, f2, f3 = pairs[0][1], pairs[1][1], pairs[2][1]

    r21 = h1 / h2
    r32 = h2 / h3

    # Observed order of accuracy
    eps21 = f2 - f1
    eps32 = f3 - f2
    q21 = math.log(abs(eps32 / eps21) + 1e-30) / math.log(r21)
    p = q21  # observed order

    # Extrapolated value (fine grid limit)
    f_ext = (r21 ** p * f1 - f2) / (r21 ** p - 1)

    # Approximate relative error
    e_a21 = abs((f1 - f2) / f1) * 100
    e_a32 = abs((f2 - f3) / f2) * 100

    # GCI (fine grid)
    gci_fine = safety_factor * e_a21 / (r21 ** p - 1)

    # GCI (coarse grid)
    gci_coarse = safety_factor * e_a32 * r21 ** p / (r32 ** p - 1)

    # Check asymptotic range
    if abs(eps32 / eps21) > 0:
        asymptotic_check = abs((r32 ** p - 1) / (r21 ** p - 1) * eps21 / eps32 - 1)
    else:
        asymptotic_check = float("nan")

    return {
        "status": "complete",
        "spacings": [h1, h2, h3],
        "values": [f1, f2, f3],
        "refinement_ratios": {"r21": r21, "r32": r32},
        "observed_order": p,
        "extrapolated_value": f_ext,
        "approximate_errors_pct": {"e_a21": e_a21, "e_a32": e_a32},
        "gci_pct": {"fine": gci_fine, "coarse": gci_coarse},
        "asymptotic_range_check": asymptotic_check,
        "is_asymptotic": asymptotic_check < 0.1 if math.isfinite(asymptotic_check) else False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["runs/armaly_cluster_v2",
                                                       "runs/armaly_cluster",
                                                       "docs/repro-armaly1983-bfs/artifacts"])
    parser.add_argument("--out", default="docs/repro-armaly1983-bfs/convergence")
    args = parser.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    records = load_records(args.runs)
    if not records:
        print("No records found. Run the reproduction script first.")
        return 1

    # Group by (Re, downstream_length)
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        key = (r["reynolds_dh"], r["downstream_step_heights"])
        groups.setdefault(key, []).append(r)

    results = {}
    for (re, ld), group in sorted(groups.items()):
        # Sort by grid level
        level_order = {"coarse": 0, "medium": 1, "fine": 2}
        group.sort(key=lambda r: level_order.get(r["level"], 99))

        values = [r["reattachment_over_step"] for r in group]
        spacings = [grid_spacing(r["level"]) for r in group]
        levels = [r["level"] for r in group]

        analysis = gci_analysis(values, spacings)
        analysis["levels"] = levels
        analysis["reynolds_dh"] = re
        analysis["downstream_step_heights"] = ld

        # Add experimental reference
        exp_map = {100.0: 3.0, 389.0: 7.9}
        if re in exp_map:
            analysis["experimental_value"] = exp_map[re]
            if analysis["status"] == "complete":
                ext = analysis["extrapolated_value"]
                analysis["extrapolated_error_pct"] = abs(ext - exp_map[re]) / exp_map[re] * 100

        results[f"Re{re:.0f}_Ld{ld:.0f}S"] = analysis

        # Print summary
        print(f"\n{'='*60}")
        print(f"Re_Dh = {re:.0f}, L_down = {ld:.0f}S")
        print(f"{'='*60}")
        for i, (level, val, sp) in enumerate(zip(levels, values, spacings)):
            exp = exp_map.get(re, float("nan"))
            err = abs(val - exp) / exp * 100 if math.isfinite(exp) else float("nan")
            print(f"  {level:7s} (h={sp:.4f}): L_r/S = {val:.4f}, error = {err:.2f}%")

        if analysis["status"] == "complete":
            print(f"\n  Observed order of accuracy: p = {analysis['observed_order']:.3f}")
            print(f"  Extrapolated value (h→0):   L_r/S = {analysis['extrapolated_value']:.4f}")
            if math.isfinite(analysis.get("experimental_value", float("nan"))):
                ext_err = analysis["extrapolated_error_pct"]
                print(f"  Extrapolated error vs exp:  {ext_err:.2f}%")
            print(f"  GCI (fine):                 {analysis['gci_pct']['fine']:.2f}%")
            print(f"  GCI (coarse):               {analysis['gci_pct']['coarse']:.2f}%")
            print(f"  Asymptotic range:           {'Yes' if analysis['is_asymptotic'] else 'No'}")

    # Save results
    out_path = out / "convergence_analysis.json"
    out_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nResults saved to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
