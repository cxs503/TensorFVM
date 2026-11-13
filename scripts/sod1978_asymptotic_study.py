#!/usr/bin/env python
"""CLI for the Sod (1978) asymptotic-grid convergence study.

    PYTHONPATH=src python scripts/sod1978_asymptotic_study.py \
        --out docs/repro-fvm-sod1978-asymptotic --figures

Writes ``artifacts/asymptotic_study.json`` (setup, declared thresholds, per-grid
measurements, rate tables, round-off floor, verdicts, smooth-wave design order,
profiles) plus four figures.  The implementation lives in
:mod:`tensorfvm.verification.sod1978_asymptotic`; this file is only the CLI so
the study has a single documented command.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.verification.sod1978_asymptotic import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())