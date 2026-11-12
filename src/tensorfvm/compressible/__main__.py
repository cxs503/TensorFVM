"""CLI: ``python -m tensorfvm.compressible --output <dir>``.

Runs the compressible verification legs and writes the JSON artifact plus the
report figures.  ``--quick`` shrinks every grid so the run finishes in about a
minute (the pytest gate uses the same configuration).  Leg ``forward_step`` is
the Mach-3 forward-facing step (Woodward & Colella problem 2); it is not in the
default leg list because its production grid is expensive, so select it
explicitly::

    python -m tensorfvm.compressible --legs forward_step --output runs/forward_step
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .verification import DEFAULT_OUT, RunConfig, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--legs", default="sod,interface,rm,bubble,vortex")
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)
    legs = tuple(x.strip() for x in args.legs.split(",") if x.strip())
    result = run_all(Path(args.output), RunConfig(quick=args.quick), legs=legs,
                     figures=not args.no_figures)
    summary = {
        "all_criteria_met": result["all_criteria_met"],
        "verdicts": result["verdicts"],
        "seconds": result["seconds"],
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if result["all_criteria_met"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
