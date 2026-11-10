"""Two-phase VOF verification surface: analytic anchors + mass balance + GCI.

What it adds over the existing evidence
---------------------------------------
``tensorfvm.verification.vof_rotation`` verified the *algebraic* transports
(``two_phase_vof``, ``vof_fct``) on rigid rotation only, at two grids, with no
GCI and no shear-deformation anchor; ``verification.vof_plic_interface`` covers
the PLIC reconstruction on Zalesak/1-D translation.  This package adds the part
that was missing:

* **Deformation anchors** -- the Rider-Kothe (1995) single vortex and a
  two-counter-rotating-cell double vortex.  Both are exactly reversible
  (``int_0^T cos(pi t/T) dt = 0``), so the analytic solution at ``t = T`` is the
  initial field, and an inexact *or* dissipative transport shows up directly as
  interface error rather than being hidden by a smooth field.
* **Three-grid GCI** on the interface error of every case (Roache), so the
  observed order -- not just a single error number -- is on record.
* **Exact initial geometry**: initial volume fractions are the exact
  cell/disk intersection areas (:mod:`~tensorfvm.vof_verification.geometry`), so
  the reference is closed-form rather than subsampled.
* **Hydrostatic rest-state anchor** with a control: the shipped ``void``
  free-surface model leaves an interior free surface O(h) out of balance
  (measured 1/(2n): 5 % at 20^2, 2.5 % at 40^2, 1.25 % at 80^2); the opt-in
  ``well_balanced_surface`` treatment holds the rest state to round-off
  (``~1e-16``).

Evidence: ``docs/vof-two-phase-verification/report.md`` and
``docs/vof-two-phase-verification/artifacts/campaign.json``.
"""

from __future__ import annotations

from .cases import (
    Case,
    advance,
    double_vortex_case,
    face_divergence,
    sample_faces,
    single_vortex_case,
    zalesak_case,
    Transport,
)
from .geometry import analytic_area, circle_alpha, disk_rect_area, zalesak_alpha
from .hydrostatic import hydrostatic_state, run_hydrostatic
from .metrics import field_metrics, gci, observed_order, triple_summary

__all__ = [
    "Case",
    "Transport",
    "advance",
    "zalesak_case",
    "single_vortex_case",
    "double_vortex_case",
    "sample_faces",
    "face_divergence",
    "disk_rect_area",
    "circle_alpha",
    "zalesak_alpha",
    "analytic_area",
    "field_metrics",
    "observed_order",
    "gci",
    "triple_summary",
    "hydrostatic_state",
    "run_hydrostatic",
]