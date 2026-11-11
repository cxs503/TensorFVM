#!/usr/bin/env python
"""Round-2 convergence diagnosis for the Sod (1978) anchor.

    python scripts/sod1978_convergence_study.py --out docs/repro-fvm-sod1978-convergence
    python scripts/sod1978_convergence_study.py --quick --no-figures   # smoke

Writes ``<out>/study.json`` (all numbers + verdicts) and the figures.  The
implementation lives in :mod:`tensorfvm.verification.sod1978_convergence`; the
round-1 driver (``scripts/repro_fvm_sod1978.py``) is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.verification.sod1978_convergence import (  # noqa: E402
    StudyConfig,
    run_study,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "docs" / "repro-fvm-sod1978-convergence"))
    ap.add_argument("--grids", default="100,200,400,800,1600,3200,6400")
    ap.add_argument("--orders", default="1,2")
    ap.add_argument("--limiter", default="mc")
    ap.add_argument("--cfl", type=float, default=0.4)
    ap.add_argument("--margin-cells", type=int, default=16)
    ap.add_argument("--limiter-grids", default="800,1600,3200")
    ap.add_argument("--limiters", default="mc,vanleer,superbee")
    ap.add_argument("--cfl-values", default="0.2,0.4,0.8")
    ap.add_argument("--smooth-grids", default="100,200,400,800")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--quick", action="store_true",
                    help="tiny sweep used as a smoke test")
    args = ap.parse_args(argv)

    def ints(s: str):
        return tuple(int(x) for x in s.split(",") if x.strip())

    cfg = StudyConfig(
        grids=ints(args.grids), orders=ints(args.orders), limiter=args.limiter,
        cfl=args.cfl, margin_cells=args.margin_cells,
        limiter_grids=ints(args.limiter_grids),
        limiters=tuple(x.strip() for x in args.limiters.split(",") if x.strip()),
        cfl_values=tuple(float(x) for x in args.cfl_values.split(",") if x.strip()),
        smooth_grids=ints(args.smooth_grids),
        figures=not args.no_figures,
        notes={"cli": " ".join(sys.argv[1:])},
    )
    if args.quick:
        cfg.grids = (50, 100, 200)
        cfg.orders = (1, 2)
        cfg.limiter_grids = (100, 200)
        cfg.limiters = ("mc", "superbee")
        cfg.cfl_probe_nx = 200
        cfg.cfl_values = (0.2, 0.4)
        cfg.smooth_grids = (50, 100)

    artifact = run_study(cfg, args.out)
    v = artifact["verdicts"]
    print(json.dumps({k: v[k] for k in v if k != "mechanism"}, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())