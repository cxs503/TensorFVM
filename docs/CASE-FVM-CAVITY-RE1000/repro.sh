#!/usr/bin/env bash
# Reproduce the CASE-FVM-CAVITY-RE1000 evidence bundle from scratch.
#
#   bash docs/CASE-FVM-CAVITY-RE1000/repro.sh            # full campaign (~30 min, 1 core)
#   bash docs/CASE-FVM-CAVITY-RE1000/repro.sh --fast     # wiring smoke test only (~1 min)
#
# Requirements: python with torch/numpy/matplotlib, repository root as CWD.
set -euo pipefail

cd "$(dirname "$0")/../.."
export PYTHONPATH=src

if [[ "${1:-}" == "--fast" ]]; then
    python docs/CASE-FVM-CAVITY-RE1000/run_case.py \
        --grids 16 --cold-control 16 --max-steps 2000 \
        --out-dir /tmp/case_cavity_re1000_smoke
    exit 0
fi

# 1. campaign: cold n = 32, cold n = 64 (continuation control), n = 128 from n = 64
python docs/CASE-FVM-CAVITY-RE1000/run_case.py \
    --grids 32,64,128 --cold-control 32 --tolerance 1e-07 --max-steps 250000

# 2. machine gate: re-derive every criterion from the stored artefacts and fields
python -m pytest tests/test_case_cavity_re1000.py -q

# 3. report + figures (numbers injected from the artefacts)
python docs/CASE-FVM-CAVITY-RE1000/make_report.py