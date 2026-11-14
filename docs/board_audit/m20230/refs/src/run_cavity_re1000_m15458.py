"""Back-compat entry for the round-2 (m15458) Re = 1000 cavity runner.

The implementation now lives in :mod:`tensorfvm.case_cavity_re1000` (a real,
importable module with criteria, bundle writer and CLI); this file is kept only
so the round-2 record and the audit report keep pointing at a working command.
It forwards every argument unchanged::

    PYTHONPATH=src python scripts/run_cavity_re1000_m15458.py \
        --out artifacts/CASE-CAVITY-RE1000-FVM --grids 64,128

Equivalent and preferred::

    PYTHONPATH=src python -m tensorfvm.case_cavity_re1000 --grids 64,128 ...
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "src"))

from tensorfvm.case_cavity_re1000 import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())