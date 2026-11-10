"""Phase-0 two-phase capability gate: algebraic VOF + dam-break anchor.

The package bundles three separable pieces:

* :mod:`tensorfvm.vof_phase0.advection` -- conservative, bounded algebraic VOF
  transport (flux-corrected, dimensionally split);
* :mod:`tensorfvm.vof_phase0.anchors` -- Martin & Moyce (1952) digitised surge
  data and the Ritter (1892) analytic dam-break solution;
* :mod:`tensorfvm.vof_phase0.dam_break` -- the coupled dam-break driver used by
  the acceptance campaign (:mod:`tensorfvm.vof_phase0.campaign`) and by
  ``scripts/run_vof_phase0.py``.

Evidence and acceptance numbers: ``docs/vof-phase0/report.md`` and
``docs/vof-phase0/artifacts/phase0_report.json``.
"""

from __future__ import annotations

from .advection import AlgebraicVOF
from .anchors import (
    MARTIN_MOYCE_FRONT_T,
    MARTIN_MOYCE_FRONT_X,
    martin_moyce_front,
    martin_moyce_height,
    ritter_depth,
    ritter_front,
    ritter_velocity,
    to_dimensionless_front,
    to_dimensionless_time,
)
from .dam_break import (
    DamBreakConfig,
    column_height,
    initial_alpha,
    run_dam_break,
    surge_front,
)
from .solver import Phase0TwoPhaseSolver

__all__ = [
    "AlgebraicVOF",
    "MARTIN_MOYCE_FRONT_T",
    "MARTIN_MOYCE_FRONT_X",
    "martin_moyce_front",
    "martin_moyce_height",
    "ritter_front",
    "ritter_depth",
    "ritter_velocity",
    "to_dimensionless_time",
    "to_dimensionless_front",
    "DamBreakConfig",
    "Phase0TwoPhaseSolver",
    "run_dam_break",
    "initial_alpha",
    "surge_front",
    "column_height",
]