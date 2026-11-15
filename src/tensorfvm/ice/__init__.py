"""Ice-domain parameter validation and physical guardrails.

This module provides **physical-domain guardrails** for ice-ship interaction
parameters, preventing silent extrapolation when inputs fall outside
physically realistic ranges.

Problem statement
-----------------
Prior implementations accepted arbitrary numeric inputs and silently
extrapolated empirical formulas far beyond their calibration domain
(e.g. ice thickness h=50 m → 2450 MN resistance, compressive strength
σ_c=100 MPa → 143.9 MN).  Schema-level validation (Pydantic type checks)
catches only type errors, not physically impossible values.

Guardrail design
----------------
Each ice parameter has a **hard bound** (physically impossible beyond this)
and a **soft bound** (outside the calibration domain of empirical models).
Inputs beyond hard bounds are rejected; inputs beyond soft bounds trigger
a warning but are accepted (with a flag indicating extrapolation risk).

Repair round m22342
-------------------
The m22209 delivery let ``nan`` through as "TYPICAL" (bound comparisons are
False for NaN) and stopped at the schema layer, so nothing between a JSON
payload and the resistance arithmetic asked whether the numbers were inside the
formula's domain.  This package now (a) treats every non-finite value as a
domain violation, (b) guards the full 12-parameter input set (8 ice + 4 hull),
and (c) exposes guarded physics entry points (``guarded_level_ice_resistance``)
plus a benchmark of the guard itself (``python -m tensorfvm.ice``).

References
----------
- ISO 19906:2019 — Petroleum and natural gas industries — Arctic offshore structures
- ITTC Ice Committee procedures (2017, 2021)
- Jordaan, I.J. (2001) "Mechanics of ice–structure interaction", Engineering Fracture Mechanics
- Timco & Frederking (2012) "Ice properties and behaviour", Cold Regions Science and Technology
- Riska, K. et al. (1997) "A model for ice resistance prediction", VTT Publications 319
"""

from .parameters import (
    IceParameter,
    IceParameterBounds,
    ICE_RESISTANCE_INPUTS,
    ICE_ONLY_PARAMETERS,
    SHIP_PARAMETERS,
    SHIP_LENGTH,
    SHIP_BEAM,
    SHIP_DRAFT,
    SHIP_BLOCK_COEFFICIENT,
    ICE_THICKNESS,
    ICE_STRENGTH,
    ICE_FRICTION,
    ICE_SPEED,
    ICE_DENSITY,
    ICE_ELASTIC_MODULUS,
    ICE_POISSON_RATIO,
    ICE_TEMPERATURE,
    ALL_ICE_PARAMETERS,
)
from .guardrails import (
    IceGuardrailResult,
    IceParameterDomainError,
    GuardrailStatus,
    validate_ice_parameters,
    require_ice_parameters,
    clamp_ice_parameters,
    check_extrapolation_risk,
    get_parameter_report,
)
from .load import (
    GuardedResistance,
    IceCondition,
    IceShipGeometry,
    guarded_level_ice_resistance,
    level_ice_resistance_kN,
    lindqvist_terms,
    require_guarded_level_ice_resistance,
)
from .benchmark import run_benchmark, write_artifacts

__all__ = [
    "ICE_RESISTANCE_INPUTS",
    "ICE_ONLY_PARAMETERS",
    "SHIP_PARAMETERS",
    "SHIP_LENGTH",
    "SHIP_BEAM",
    "SHIP_DRAFT",
    "SHIP_BLOCK_COEFFICIENT",
    "IceParameter",
    "IceParameterBounds",
    "ICE_THICKNESS",
    "ICE_STRENGTH",
    "ICE_FRICTION",
    "ICE_SPEED",
    "ICE_DENSITY",
    "ICE_ELASTIC_MODULUS",
    "ICE_POISSON_RATIO",
    "ICE_TEMPERATURE",
    "ALL_ICE_PARAMETERS",
    "IceGuardrailResult",
    "GuardrailStatus",
    "validate_ice_parameters",
    "require_ice_parameters",
    "clamp_ice_parameters",
    "check_extrapolation_risk",
    "get_parameter_report",
    "IceParameterDomainError",
    "GuardedResistance",
    "IceCondition",
    "IceShipGeometry",
    "guarded_level_ice_resistance",
    "level_ice_resistance_kN",
    "lindqvist_terms",
    "require_guarded_level_ice_resistance",
    "run_benchmark",
    "write_artifacts",
]
