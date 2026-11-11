"""Three-case two-phase VOF verification face (campaign node P447W2-N01).

Where the sibling package :mod:`tensorfvm.twophase_bench` verifies the
*interface-transport* face (reconstruction + high-resolution advection against
an imposed analytic velocity), this package verifies the face that transport
alone cannot see: the **momentum coupling** between the colour function, the
pressure and the surface-tension force.

Three cases, each with a machine-checkable criterion:

* :mod:`.statics` -- static bubble (Laplace pressure + spurious currents).
  Criterion: ``max|u| / sqrt(sigma/(rho_l R)) <= 0.01`` over a grid sweep and a
  Laplace-number sweep, with the legacy cell-centred CSF as the negative
  control that proves the metric discriminates.
* :mod:`.zalesak` -- Zalesak slotted disk rotated 360 deg *and* 1000 deg.
  Criterion: relative error of the *notch* area <= 5 % at 360 deg and <= 15 %
  at 1000 deg (the slot is the sharpest feature of the disk; a global area
  error cannot see it).
* :mod:`.dambreak` -- dam break of the Martin & Moyce (1952) column
  (width/depth = 1:2) with the coupled variable-density projection solver.
  Criterion: front position within 10 % of the digitised experiment at
  t = 0.1, 0.2, 0.3 s, plus monotone grid convergence over 50x100 / 100x200 /
  200x400 cells.

:mod:`.campaign` assembles the three legs into the node's acceptance JSON.

Everything here is float64 and depends only on the repository's existing
two-phase stack (:mod:`tensorfvm.two_phase_solver`,
:mod:`tensorfvm.two_phase_vof`, :mod:`tensorfvm.two_phase_curvature`,
:mod:`tensorfvm.vof_phase0.anchors`); no reference data is re-typed.
"""

from __future__ import annotations

from .dambreak import (
    DamBreakMMConfig,
    front_position,
    martin_moyce_error,
    reduced_hydrostatic_pressure,
    run_dam_break,
)
from .statics import (
    StaticBubbleConfig,
    laplace_pressure_jump,
    run_static_bubble,
    spurious_current,
)
from .zalesak import NotchConfig, notch_area, run_long_rotation

__all__ = [
    "DamBreakMMConfig",
    "NotchConfig",
    "StaticBubbleConfig",
    "front_position",
    "laplace_pressure_jump",
    "martin_moyce_error",
    "notch_area",
    "reduced_hydrostatic_pressure",
    "run_dam_break",
    "run_long_rotation",
    "run_static_bubble",
    "spurious_current",
]