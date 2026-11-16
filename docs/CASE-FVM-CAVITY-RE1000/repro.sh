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

# 1. campaign: cold n = 32, n = 64 prolonged from 32, n = 128 prolonged from 64,
#    cold counter-check re-run at n = 64 — the grid the campaign actually
#    warm-started.  Pointing --cold-control at n = 32 (started from rest) would
#    compare that run with its own repetition and vacate the check, so the value
#    is pinned here and asserted against runs/summary.json's recorded provenance
#    by tests/test_case_cavity_re1000_continuation_control.py.
python docs/CASE-FVM-CAVITY-RE1000/run_case.py \
    --grids 32,64,128 --cold-control 64 --tolerance 1e-07 --max-steps 250000

# 2. machine gate: re-derive every criterion from the stored artefacts and fields,
#    and check the bundle's provenance (grids / control grid / continuation numbers)
python -m pytest tests/test_case_cavity_re1000.py \
    tests/test_case_cavity_re1000_continuation_control.py -q

# 3. report + figures (numbers injected from the artefacts)
python docs/CASE-FVM-CAVITY-RE1000/make_report.py