"""Console entry point of the ``GAP-FVM-TWOPHASE`` verification face.

Examples
--------
Full acceptance face (the reproduction command of the report)::

    python -m tensorfvm.twophase_gap --case all --out docs/GAP-FVM-TWOPHASE/artifacts

Fast subset (used by the pytest gate and by the solver's self-check)::

    python -m tensorfvm.twophase_gap --case zalesak --cells 64 --schemes plic \
        --out /tmp/gap_zalesak64

Hysing bubble benchmark only::

    python -m tensorfvm.twophase_gap --case hysing --hysing-cells 32 64 \
        --out /tmp/gap_hysing

Re-render the report/figures from already committed artefacts (no solve)::

    python -m tensorfvm.twophase_gap --summary-only --out docs/GAP-FVM-TWOPHASE/artifacts
"""

from __future__ import annotations

import argparse
import sys

from .campaign import DEFAULTS, run_campaign, summarise, write_figures


def _grid(text: str) -> tuple[int, int]:
    try:
        nx, ny = (int(part) for part in text.lower().split("x"))
    except ValueError as exc:  # pragma: no cover - argparse plumbing
        raise argparse.ArgumentTypeError(
            f"grid must look like 120x40, got {text!r}") from exc
    return nx, ny


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tensorfvm-twophase-gap",
        description="VOF interface-capture verification (Zalesak + dam break + Hysing bubble) "
                    "with machine-evaluated gates.")
    parser.add_argument("--out", default="docs/GAP-FVM-TWOPHASE/artifacts",
                        help="artefact directory (JSON/NPZ/figures/markdown)")
    parser.add_argument("--case", default="all",
                        choices=("all", "zalesak", "dambreak", "hysing"),
                        help="which legs to run")
    parser.add_argument("--cells", type=int, nargs="+",
                        default=list(DEFAULTS["zalesak_cells"]),
                        help="Zalesak resolutions (cells per direction)")
    parser.add_argument("--schemes", nargs="+",
                        default=list(DEFAULTS["zalesak_schemes"]),
                        help="Zalesak transport schemes")
    parser.add_argument("--dam-grids", type=_grid, nargs="+",
                        default=[tuple(g) for g in DEFAULTS["dambreak_grids"]],
                        help="dam-break grids as NxM (e.g. 120x40 240x80)")
    parser.add_argument("--t-star", type=float,
                        default=DEFAULTS["dambreak_t_star"],
                        help="dimensionless end time of the dam break")
    parser.add_argument("--hysing-cells", type=int, nargs="+",
                        default=list(DEFAULTS["hysing_cells"]),
                        help="Hysing bubble cells per diameter")
    parser.add_argument("--cfl", type=float, default=0.4,
                        help="Zalesak advective Courant number")
    parser.add_argument("--summary-only", action="store_true",
                        help="only re-summarise existing legs")
    parser.add_argument("--no-figures", action="store_true",
                        help="skip figure rendering")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.summary_only:
        run_campaign(
            args.out,
            case=args.case,
            zalesak_cells=tuple(args.cells),
            zalesak_schemes=tuple(args.schemes),
            dambreak_grids=tuple(tuple(g) for g in args.dam_grids),
            dambreak_t_star=args.t_star,
            hysing_cells=tuple(args.hysing_cells),
            cfl=args.cfl,
        )
    summary = summarise(args.out)
    if not args.no_figures:
        try:
            write_figures(args.out)
        except ImportError:  # pragma: no cover - optional dependency
            print("matplotlib unavailable: figures skipped", file=sys.stderr)
    print(f"gates: {sum(1 for g in summary['gates'] if g['passed'])}/"
          f"{len(summary['gates'])} passed; all_passed={summary['all_passed']}")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
