#!/usr/bin/env python3
"""Derive one coarser member of the official TMR 2DBFS grid family.

The Turbulence Modeling Resource publishes five nested backward-facing-step
grids; each coarser grid is exactly every-other-point of the next finer one.
The published coarsest level (33x33/13x33/49x57/17x57 nodes) is the only one
that runs in minutes with a collocated SIMPLE solver on a CPU, so a sixth level
is produced here by applying the same every-other-point rule to the coarsest
published level.  The derived grid is written as a PLOT3D file that keeps the
TMR geometry exactly; ``provenance.json`` records that it is derived, not
downloaded.
"""

import argparse
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO / "src"))
from tensorfvm.tmr_grid import read_plot3d_2d  # noqa: E402


def write_plot3d_2d(path, blocks):
    lines = [f"{len(blocks)}"]
    lines.append(" ".join(f"{x.shape[1]} {x.shape[0]}" for x, _ in blocks))
    for x, y in blocks:
        for array in (x, y):
            for row in array:
                lines.append(" ".join(f"{value:.16e}" for value in row))
    Path(path).write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=REPO / "validation_tmr/reference/tmr/grids/backstep5_4levdn.p2dfmt")
    parser.add_argument("--output", type=Path,
                        default=REPO / "validation_tmr/reference/tmr/grids/backstep5_5levdn_derived.p2dfmt")
    args = parser.parse_args()
    blocks = read_plot3d_2d(args.source)
    coarse = [(x[::2, ::2], y[::2, ::2]) for x, y in blocks]
    write_plot3d_2d(args.output, coarse)
    print("source", [(x.shape[1], x.shape[0]) for x, _ in blocks])
    print("derived", [(x.shape[1], x.shape[0]) for x, _ in coarse],
          "->", args.output)


if __name__ == "__main__":
    main()
