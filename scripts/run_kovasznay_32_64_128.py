#!/usr/bin/env python3
"""Run the full 32/64/128 Kovasznay convergence study and produce the closure report.

This script is designed to be run when system load is low (load15 < 50).
It runs the three grids sequentially, summarizes the results, and produces
a markdown report and convergence figure.

Usage:
    python scripts/run_kovasznay_32_64_128.py --output runs/kovasznay-32-64-128

Expected runtime: ~10-30 minutes per grid under normal load.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def run_grid(n, output_dir, reynolds=40.0):
    """Run a single grid and return the result."""
    grid_dir = output_dir / f"n{n}"
    grid_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "scripts/verify_kovasznay_order.py",
        "--output", str(grid_dir),
        "--n", str(n),
        "--reynolds", str(reynolds),
        "--convection-order", "2",
        "--outer-relaxation", "0.7",
        "--tolerance", "1e-8",
        "--max-outer-iterations", "120",
        "--max-iterations", "8000",
    ]
    
    print(f"Running n={n}...")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    elapsed = time.time() - t0
    
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode})")
        print(f"  stderr: {result.stderr[-500:]}")
        return None
    
    # Load the result
    cases_file = grid_dir / "cases.json"
    if cases_file.exists():
        with open(cases_file) as f:
            cases = json.load(f)
        print(f"  OK ({elapsed:.0f}s): vel_L2={cases[0]['velocity_l2']:.6e}")
        return cases[0]
    
    print(f"  WARNING: cases.json not found")
    return None


def summarize_and_report(output_dir, results):
    """Run the summarize script and produce the report."""
    runs = [str(output_dir / f"n{r['n']}") for r in results]
    
    cmd = [
        sys.executable, "scripts/summarize_kovasznay_order.py",
        "--runs"] + runs + [
        "--output", str(output_dir / "summary"),
        "--grids", "32", "64", "128",
        "--threshold", "1.8",
        "--markdown", str(output_dir / "RESULTS_32_64_128.md"),
        "--figure", str(output_dir / "convergence_32_64_128.png"),
    ]
    
    print("\nSummarizing results...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    print(result.stdout)
    if result.returncode != 0:
        print(f"stderr: {result.stderr}")
    
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True,
                   help="Output directory for run artifacts")
    p.add_argument("--reynolds", type=float, default=40.0)
    p.add_argument("--skip-summarize", action="store_true")
    args = p.parse_args()
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    grids = [32, 64, 128]
    results = []
    
    for n in grids:
        result = run_grid(n, args.output, args.reynolds)
        if result is not None:
            results.append(result)
        else:
            print(f"Skipping n={n} due to failure")
    
    if len(results) < 3:
        print(f"\nERROR: Only {len(results)}/3 grids completed. Cannot summarize.")
        return 1
    
    if not args.skip_summarize:
        success = summarize_and_report(args.output, results)
        return 0 if success else 2
    
    # Just print the results
    print("\nResults:")
    for r in results:
        print(f"  n={r['n']}: vel_L2={r['velocity_l2']:.6e}, p_L2={r['pressure_mean_free_l2']:.6e}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
