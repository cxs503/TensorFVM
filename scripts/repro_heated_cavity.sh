#!/bin/bash
# Reproduction command for de Vahl Davis (1983) heated cavity benchmark
# Usage: bash scripts/repro_heated_cavity.sh
#
# This script runs the full benchmark suite for the natural convection
# square cavity at Ra = 1e3, 1e4, 1e5, 1e6 with Pr = 0.71.
# Reference: de Vahl Davis, G. (1983), Int. J. Numer. Methods Fluids 3, 249-264.

set -e
cd "$(dirname "$0")/.."

echo "=== de Vahl Davis (1983) Heated Cavity Benchmark ==="
echo "Reference: Int. J. Numer. Methods Fluids 3, 249-264"
echo ""

# Run each Rayleigh number case
for ra in 1000 10000 100000 1000000; do
    echo "--- Running Ra=$ra ---"
    python scripts/run_cavity_case.py --rayleigh $ra --meshes 32,64,128
    echo ""
done

# Assemble summary and generate figures
echo "=== Assembling Summary ==="
python scripts/assemble_cavity_benchmark.py

echo ""
echo "=== Benchmark Complete ==="
echo "Artifacts: artifacts/heated-cavity-benchmark/"
echo "Summary:   artifacts/heated-cavity-benchmark/summary.json"
