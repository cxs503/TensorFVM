"""Two-phase VOF benchmark face: CICSAM/HRIC transport, Zalesak shape, Ritter dam break.

This package is the library side of the ``benchmarks/twophase`` node
(P400W2-N01):

* :mod:`.cicsam` — normalised-variable algebraic transport (Hyper-C +
  CBC-limited ULTIMATE-QUICKEST blended by interface orientation), plus the
  HRIC-type Courant blend and the first-order donor reference;
* :mod:`.metrics2d` — area / perimeter / centroid / front metrics with stated
  discretisations, and the closed-form slotted-disk reference;
* :mod:`.zalesak` — slotted-disk rigid rotation (one revolution, exact solution
  = initial field), importing the repository's existing geometry anchors;
* :mod:`.ritter` — Ritter (1892) dam-break analytics, the imposed analytic
  velocity field, the VOF transport driver and the depth-matched front locator.

Reproduction lives in ``benchmarks/twophase`` (``run_benchmark.py``); the
acceptance numbers are machine-written to
``benchmarks/twophase/artifacts/twophase_verification.json``.
"""

from __future__ import annotations

from .cicsam import advect_cicsam, advect_upwind, sweep_1d
from .metrics2d import (
    centroid,
    column_depth,
    contour_perimeter,
    front_position,
    liquid_area,
    mask_area,
    perimeter_error,
    slotted_disk_reference,
    symmetric_area_error,
)
from .ritter import (
    DamBreakGeometry,
    exact_alpha,
    initial_alpha,
    ritter_depth,
    ritter_front,
    ritter_front_at_depth,
    ritter_velocity,
    run_transport,
    velocity_faces,
)
from .zalesak import SCHEMES, run_rotation

__all__ = [
    "advect_cicsam",
    "advect_upwind",
    "sweep_1d",
    "centroid",
    "column_depth",
    "contour_perimeter",
    "front_position",
    "liquid_area",
    "mask_area",
    "perimeter_error",
    "slotted_disk_reference",
    "symmetric_area_error",
    "DamBreakGeometry",
    "exact_alpha",
    "initial_alpha",
    "ritter_depth",
    "ritter_front",
    "ritter_front_at_depth",
    "ritter_velocity",
    "run_transport",
    "velocity_faces",
    "SCHEMES",
    "run_rotation",
]