"""Freeze the refinement forecast before the cj3325 rungs land (programmatic, zero hand-copy).

Run: PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/PREDECLARATION-refine.py
Writes /tmp/predecl.json and prints the table quoted in PREDECLARATION-refine.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from tensorfvm.verification.wc_refine import DataMissing, cell_average_l1, load_profile

DATA = Path(__file__).resolve().parents[1] / "woodward-colella-double-blast" / "ladder" / "data"


def main() -> dict:
    ref = load_profile(DATA, 12800, "weno5")
    e = {}
    for nx in (1600, 3200):
        p = load_profile(DATA, nx, "hllc")
        e[nx] = cell_average_l1(p["rho"], ref["rho"])["l1_percent"]
    order = float(__import__("numpy").log(e[1600] / e[3200]) / __import__("numpy").log(2.0))
    # error model e(nx) = e(3200) * (nx/3200)^(-p); the exponent is the
    # log2 of the grid ratio (a float), not an integer bit_length.
    forecast = {
        str(nx): {f"p={pp:.4f}": e[3200] * (nx / 3200) ** (-pp)
                  for pp in (0.79, order, 1.05)}
        for nx in (6400, 12800)
    }
    return {"e1600": e[1600], "e3200": e[3200], "p_measured": order, "forecast": forecast}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
