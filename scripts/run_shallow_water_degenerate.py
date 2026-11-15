#!/usr/bin/env python
"""Driver: degenerate-limit audit of the wet-bed dam-break benchmark.

Writes every artefact consumed by ``docs/case-fvm-dambreak-stoker-degenerate/
REPORT.md`` (which is generated from them by ``make_report.py``).

    PYTHONPATH=src python scripts/run_shallow_water_degenerate.py          # full
    PYTHONPATH=src python scripts/run_shallow_water_degenerate.py --quick  # smoke

The default sweep is 19 depth ratios x 3 grids + 3 deep rungs (~60 1-D runs,
about two minutes on six workers); ``--quick`` keeps four ratios on two grids.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorfvm.shallow_water.degenerate import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())