"""Manoeuvring / self-propulsion prediction face for KVLCC2 (TensorFVM).

What this package is
--------------------
A self-consistent *manoeuvring* chain built on top of the repository's existing
KVLCC2 hull geometry, resistance chain and body-fitted RANS solver:

======================  =====================================================
module                  role
======================  =====================================================
:mod:`.hull`            KVLCC2 model particulars, geometry gate (Cb/Cwp/LCB),
                        lateral-planform section data, rudder geometry.
:mod:`.resistance`      ITTC-78 model-scale resistance chain (Cf, (1+k), dCf)
                        plus the CFD viscous cross-check already in the repo.
:mod:`.propeller`       blade-element/momentum-theory (BEMT) KP505 propeller:
                        ``K_T(J)``, ``K_Q(J)``, open-water efficiency.
:mod:`.selfpropulsion`  self-propulsion point ``T(n) = R_T(V)/(1-t)`` by root
                        finding, with wake fraction / thrust deduction models.
:mod:`.rudder`          rudder polar (2-D body-fitted RANS, loaded from the
                        campaign artefact, plus an analytic finite-aspect
                        lifting baseline) and the Kose/MMG normal-force law.
:mod:`.hullforces`      hull lateral force model: low-aspect-ratio lifting term
                        (Helmbold) + Hoerner cross-flow drag integrated over the
                        hull's own lateral planform.
:mod:`.dynamics`        3-DOF (surge-sway-yaw) RK4 integrator, turning-circle
                        and zig-zag metric extraction.
:mod:`.reference`       benchmark records **with explicit provenance status**.
======================  =====================================================

Evidence grading (used verbatim in the node report)
---------------------------------------------------
* ``REAL``      -- computed by a solver/analysis in this repository and
                   checkable against an independent anchor (geometry integrals,
                   momentum/blade-element equality, momentum-budget closure).
* ``EMPIRICAL`` -- a published regression/interaction coefficient (wake
                   fraction, thrust deduction, ``a_H``, ``x_H``, ``t_R``).
* ``CFD``       -- a number produced by a RANS solution in this repository.
* ``WITHHELD``  -- a benchmark value whose primary source is not in this
                   workspace; carried as a band, never used as a verdict.

The package deliberately **does not** claim experimental validation of the
predicted manoeuvres: no provenance-verified SIMMAN-2008 EFD record exists in
this workspace (the in-repository ``simman2021_reference`` derivative table was
already audited as internally inconsistent by node IN40-N04).  Every consumer
of :mod:`.reference` must propagate its ``status`` field.
"""
from __future__ import annotations

from .hull import (  # noqa: F401
    KVLCC2_MODEL_LENGTH, Kvlcc2Conditions, RudderGeometry, build_hull,
    geometry_gate, rudder_geometry, section_data,
)
from .propeller import (  # noqa: F401
    PropellerGeometry, OpenWaterPoint, brentq_advance_ratio, open_water_curve,
    open_water_point, propeller_thrust,
)
from .resistance import (  # noqa: F401
    ResistanceComposition, model_scale_resistance, resistance_components,
)
from .selfpropulsion import (  # noqa: F401
    SelfPropulsionPoint, hull_interaction, solve_self_propulsion,
)
from .rudder import (  # noqa: F401
    RudderPolar, analytic_rudder_polar, load_rudder_polar, rudder_normal_force,
)
from .hullforces import (  # noqa: F401
    HullForceModelSet, hull_force, linear_derivatives,
)
from .dynamics import (  # noqa: F401
    ManoeuvreModel, Trajectory, turning_circle_metrics, zigzag_metrics,
    simulate_turning_circle, simulate_zigzag,
)
from . import reference  # noqa: F401

__all__ = [
    "KVLCC2_MODEL_LENGTH", "Kvlcc2Conditions", "RudderGeometry",
    "build_hull", "geometry_gate", "rudder_geometry", "section_data",
    "PropellerGeometry", "OpenWaterPoint", "brentq_advance_ratio",
    "open_water_curve", "open_water_point", "propeller_thrust",
    "ResistanceComposition", "model_scale_resistance", "resistance_components",
    "SelfPropulsionPoint", "hull_interaction", "solve_self_propulsion",
    "RudderPolar", "analytic_rudder_polar", "load_rudder_polar",
    "rudder_normal_force",
    "HullForceModelSet", "hull_force", "linear_derivatives",
    "ManoeuvreModel", "Trajectory", "turning_circle_metrics", "zigzag_metrics",
    "simulate_turning_circle", "simulate_zigzag", "reference",
]