"""RANS (k-omega SST / Spalart-Allmaras) ship-resistance verification surface.

Why this package exists
-----------------------
The repository already owns two RANS verification faces that this package
deliberately does **not** duplicate:

* ``tensorfvm.rans_validation`` -- a *1-D equilibrium* channel matrix for
  SST-2003 / Wilcox k-omega / standard k-epsilon (``docs/rans-turbulence-validation-matrix``).
  Equilibrium parallel shear fixes the shear, so the momentum equation is never
  solved; Spalart-Allmaras is absent there.
* ``docs/resistance-chain-form-factor`` -- the double-body 3-D RANS form-factor
  attempt, machine-recorded as **WITHHELD**: the required mesh is ~3e6 cells
  (~63 h of setup on this node) and the native SST path diverged at every
  affordable spacing.

What is missing -- and what this package adds -- is the intermediate rung that is
both *affordable* and *predictive*: a **thin-shear-layer marching RANS** that
solves the streamwise momentum equation together with the full k-omega SST
(Menter 2003) and Spalart-Allmaras (1994) transport equations, for

* a zero-pressure-gradient flat plate (analytic anchors: Blasius, 1/7 power law,
  Schlichting ``0.455/(log Re)^2.58``), and
* the KVLCC2 hull, represented by its axisymmetric equivalent body of revolution
  along with the hull's own wetted-surface distribution,
* and a wake region, so that the drag can be cross-checked by the momentum
  integral (friction + form vs far-wake momentum defect).

Modules
-------
``anchors``          Blasius similarity solution, power-law and friction-line anchors,
                     and the SIMMAN 2008 KVLCC2 reference record with its
                     provenance status.
``closures``         k-omega SST and Spalart-Allmaras closure algebra.
``boundary_layer``   the marching finite-volume solver (2-D, axisymmetric, wake).
``kvlcc2``           KVLCC2 particulars, parametric hull, equivalent body,
                     wetted-surface distribution and Michell wave resistance.
``speed_sweep``      the full-speed range campaign: C_T(Fn) fits, model-to-model
                     comparison, GCI and conservation verdicts.
"""
from __future__ import annotations

from .anchors import (  # noqa: F401
    BLASIUS_FPP, blasius_solution, blasius_cf, blasius_delta, blasius_theta,
    one_seventh_profile, one_seventh_local_cf, one_seventh_mean_cf,
    schlichting_mean_cf, SIMMAN2008_KVLCC2,
)
from .boundary_layer import (  # noqa: F401
    BoundaryLayerResult, march_boundary_layer,
)
from .closures import (  # noqa: F401
    SST_CONSTANTS, SA_CONSTANTS, sst_eddy_viscosity, sa_eddy_viscosity,
)

__all__ = [
    "BLASIUS_FPP", "blasius_solution", "blasius_cf", "blasius_delta", "blasius_theta",
    "one_seventh_profile", "one_seventh_local_cf", "one_seventh_mean_cf",
    "schlichting_mean_cf", "SIMMAN2008_KVLCC2",
    "BoundaryLayerResult", "march_boundary_layer",
    "SST_CONSTANTS", "SA_CONSTANTS", "sst_eddy_viscosity", "sa_eddy_viscosity",
]