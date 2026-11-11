"""Zalesak slotted-disk rigid rotation: shape fidelity of the interface schemes.

The geometry and the rotation field are **imported** from the repository's
existing verification anchors
(:func:`tensorfvm.verification.vof_rotation.zalesak_disk` /
``rotation_velocity``) so the CICSAM leg of this node cannot silently drift from
the PLIC/FCT legs already registered.  The exact solution of one full rigid
revolution is the initial field, which makes area and perimeter errors
resolution-consistent (no pixelation bias is charged to the scheme).

What is measured per (scheme, resolution):

* ``symmetric_area_error`` — misplaced area at the 1/2 level (criterion C2: <=1 %);
* ``perimeter_error`` — relative change of the captured contour length
  (criterion C2: <=3 %), together with the analytic slotted-disk perimeter so
  the metric's own discretisation bias is visible;
* ``mass_drift_relative`` — global volume change (criterion C3);
* ``peak_retention`` and ``relative_l1`` — sharpness diagnostics that a volume
  criterion cannot see.
"""

from __future__ import annotations

import math

import torch

from ..verification.vof_rotation import rotation_velocity, zalesak_disk
from .cicsam import advect_cicsam
from .metrics2d import (
    contour_area,
    contour_perimeter,
    liquid_area,
    slotted_disk_reference,
    symmetric_area_error,
)

__all__ = ["run_rotation", "SCHEMES"]

SCHEMES = ("upwind", "cicsam", "hric", "plic")


def run_rotation(
    cells: int,
    scheme: str,
    cfl: float = 0.4,
    revolutions: float = 1.0,
    rate: float = 2.0 * math.pi,
    track_every: int = 25,
) -> dict:
    """Rotate the Zalesak disk by ``revolutions`` turns with one scheme."""
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}")
    if cells < 16:
        raise ValueError("need at least 16 cells per direction")
    alpha0 = zalesak_disk(cells)
    u, v, dx = rotation_velocity(cells, rate=rate)
    dy = dx
    u_max = float(max(u.abs().max(), v.abs().max()))
    dt = cfl * dx / u_max
    steps = max(1, int(round(revolutions / dt)))
    dt = revolutions / steps

    if scheme == "plic":
        from ..vof_plic import advect_plic

    alpha = alpha0.clone()
    m0 = float(alpha.sum().item())
    area0 = liquid_area(alpha0, dx, dy)
    drift = []
    worst_bound = 0.0
    for step in range(steps):
        if scheme == "plic":
            alpha = advect_plic(alpha, u, v, dt, dx, dy,
                                order="xy" if step % 2 == 0 else "yx",
                                conservative_bounds=False)
        else:
            alpha = advect_cicsam(alpha, u, v, dt, dx, dy, scheme=scheme,
                                  order="xy" if step % 2 == 0 else "yx",
                                  periodic=True)
        if step % track_every == 0 or step == steps - 1:
            drift.append((step * dt, abs(float(alpha.sum().item()) - m0) / m0))
        worst_bound = max(
            worst_bound,
            float((alpha < -1e-12).sum().item() + (alpha > 1.0 + 1e-12).sum().item()),
        )

    sym = symmetric_area_error(alpha, alpha0)
    p = contour_perimeter(alpha, dx, dy)
    p0 = contour_perimeter(alpha0, dx, dy)
    ref = slotted_disk_reference()
    diff = (alpha - alpha0).abs()
    return {
        "scheme": scheme,
        "cells": cells,
        "dx": dx,
        "dt": dt,
        "steps": steps,
        "cfl": cfl,
        "revolutions": revolutions,
        "area_initial_analytic": ref["area"],
        "area_initial_discrete": area0,
        "area_final_discrete": liquid_area(alpha, dx, dy),
        "area_pixelation_error": abs(area0 - ref["area"]) / ref["area"],
        "area_contour": contour_area(alpha, dx, dy),
        "area_contour_initial": contour_area(alpha0, dx, dy),
        "area_error_relative": abs(liquid_area(alpha, dx, dy) - area0) / area0,
        "area_error_contour": abs(contour_area(alpha, dx, dy) - ref["area"])
        / ref["area"],
        "area_error_contour_vs_initial": abs(
            contour_area(alpha, dx, dy) - contour_area(alpha0, dx, dy))
        / contour_area(alpha0, dx, dy),
        "symmetric_area_error": sym["symmetric_area_error"],
        "perimeter_initial_discrete": p0,
        "perimeter_final": p,
        "perimeter_analytic": ref["perimeter"],
        "perimeter_pixelation_error": abs(p0 - ref["perimeter"]) / ref["perimeter"],
        "perimeter_error": abs(p - p0) / p0,
        "perimeter_error_initial": abs(p - p0) / p0,
        "perimeter_error_analytic": abs(p - ref["perimeter"]) / ref["perimeter"],
        "mass_drift_relative": abs(float(alpha.sum().item()) - m0) / m0,
        "peak_retention": float(alpha.max().item()) / float(alpha0.max().item()),
        "relative_l1": float(diff.sum().item()) / m0,
        "min_alpha": float(alpha.min().item()),
        "max_alpha": float(alpha.max().item()),
        "bounded_violation_cells": worst_bound,
        "drift_series_time": [t for t, _ in drift],
        "drift_series_relative": [d for _, d in drift],
        "_field": alpha,
        "_exact": alpha0,
    }