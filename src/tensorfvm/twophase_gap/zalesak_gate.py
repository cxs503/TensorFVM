"""Zalesak slotted-disk leg: geometric (shape) error of the captured interface.

The transport itself is *not* re-implemented here -- it is the repository's
geometric PLIC (:func:`tensorfvm.vof_plic.advect_plic`) and algebraic CICSAM
(:func:`tensorfvm.twophase_bench.cicsam.advect_cicsam`) through the shared
rotation driver :func:`tensorfvm.twophase_bench.zalesak.run_rotation`.  What
this module adds is the piece the ``GAP-FVM-TWOPHASE`` anchor asks for and the
older campaign deliberately left un-gated: the **thresholded geometric error**

    e_geo = |{alpha > 1/2} XOR {exact > 1/2}| / |{exact > 1/2}|

i.e. the fraction of the slotted-disk area that the captured interface puts on
the wrong side of the 1/2 level after one full rigid revolution (the exact
solution of one revolution is the initial field).  Together with the mass
(volume) error and the perimeter change it separates "kept the liquid" from
"kept the shape", which is the whole point of an interface-capture test.

Declared resolution: the criterion is stated at ``N = 128`` cells per
direction.  ``N = 64`` is reported as the coarse member of the resolution pair
(and is expected to *fail* 5 %: that is the measurement, not a defect of the
gate), which is why :func:`convergence_order` is part of the evidence.
"""

from __future__ import annotations

import math

from ..twophase_bench.zalesak import run_rotation
from ..vof_plic import interface_width_cells

__all__ = ["run_zalesak_leg", "convergence_order", "GEOMETRIC_ERROR_MAX",
           "DECLARED_CELLS", "MASS_DRIFT_MAX", "PERIMETER_ERROR_MAX",
           "BOUND_VIOLATION_MAX"]

#: Thresholded geometric error accepted at the declared resolution.
GEOMETRIC_ERROR_MAX = 0.05
#: Resolution at which the geometric criterion is declared.
DECLARED_CELLS = 128
#: Relative liquid-area drift accepted over one revolution.
MASS_DRIFT_MAX = 1.0e-3
#: Relative contour-length change accepted over one revolution.
PERIMETER_ERROR_MAX = 0.10
#: Largest tolerated excursion of ``alpha`` outside ``[0, 1]``.
BOUND_VIOLATION_MAX = 1.0e-12


def run_zalesak_leg(
    cells: int,
    scheme: str = "plic",
    *,
    cfl: float = 0.4,
    revolutions: float = 1.0,
) -> dict:
    """Rotate the slotted disk once and return the shape-fidelity record.

    Parameters
    ----------
    cells:
        Cells per direction (square, unit domain).
    scheme:
        ``"plic"`` (geometric reconstruction) or one of the algebraic schemes
        accepted by :func:`tensorfvm.twophase_bench.zalesak.run_rotation`.
    cfl:
        Advective Courant number of the sub-steps.
    revolutions:
        Number of full revolutions (1.0 = the standard test).

    Returns
    -------
    dict
        JSON-serialisable record with the measured metrics, the analytic
        reference values and the field itself under ``"field"`` /
        ``"exact"`` (kept separately by the caller when writing artefacts).
    """
    raw = run_rotation(cells, scheme, cfl=cfl, revolutions=revolutions)
    alpha = raw.pop("_field")
    exact = raw.pop("_exact")

    bound_violation = max(0.0, float(alpha.max().item()) - 1.0,
                          -float(alpha.min().item()))
    record = {
        "case": "zalesak",
        "scheme": scheme,
        "cells": int(cells),
        "revolutions": float(revolutions),
        "cfl": float(cfl),
        "geometric_error": float(raw["symmetric_area_error"]),
        "mass_drift_relative": float(raw["area_error_relative"]),
        "perimeter_error": float(raw["perimeter_error"]),
        "perimeter_analytic": float(raw["perimeter_analytic"]),
        "perimeter_initial_discrete": float(raw["perimeter_initial_discrete"]),
        "perimeter_final": float(raw["perimeter_final"]),
        "perimeter_pixelation_error": float(raw["perimeter_pixelation_error"]),
        "area_initial_analytic": float(raw["area_initial_analytic"]),
        "area_initial_discrete": float(raw["area_initial_discrete"]),
        "area_final_discrete": float(raw["area_final_discrete"]),
        "area_pixelation_error": float(raw["area_pixelation_error"]),
        "peak_retention": float(raw["peak_retention"]),
        "relative_l1": float(raw["relative_l1"]),
        "bound_violation": float(bound_violation),
        "interface_width_cells": float(interface_width_cells(alpha)),
        "steps": int(raw["steps"]),
        "dt": float(raw["dt"]),
        "declared": int(cells) == DECLARED_CELLS,
        "reference": {
            "source": "Zalesak (1979) J. Comput. Phys. 31, 335 -- exact "
                      "solution of one rigid revolution is the initial field; "
                      "analytic area/perimeter of the slotted disk",
            "geometric_error_exact": 0.0,
            "area_analytic": float(raw["area_initial_analytic"]),
            "perimeter_analytic": float(raw["perimeter_analytic"]),
        },
        "metrics": raw,
    }
    return {"record": record, "field": alpha.clone(), "exact": exact.clone()}


def convergence_order(coarse: float, fine: float, ratio: float = 2.0) -> float:
    """Observed order ``p`` from two errors and a resolution ratio."""
    coarse = max(float(coarse), 1e-16)
    fine = max(float(fine), 1e-16)
    return math.log(coarse / fine) / math.log(ratio)

