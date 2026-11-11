#!/usr/bin/env python
"""Reproduce Sod (1978) at t = 0.14 with TensorFVM and grade the three criteria.

    python scripts/repro_fvm_sod1978.py --out docs/repro-fvm-sod1978

Writes ``docs/repro-fvm-sod1978/artifacts/sod1978_t0.14.json`` (setup, published
reference values, per-grid tables, criteria with pass/fail, ecosystem friction)
plus four figures, and exits non-zero when a mission criterion is not met.  The
implementation lives in :mod:`tensorfvm.verification.sod1978`; this file is only
the CLI so the reproduction has a single documented command.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.verification.sod1978 import main  # noqa: E402

if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())