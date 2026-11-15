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
    GuardrailStatus,
    validate_ice_parameters,
    clamp_ice_parameters,
    check_extrapolation_risk,
    get_parameter_report,
)

__all__ = [
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
    "clamp_ice_parameters",
    "check_extrapolation_risk",
    "get_parameter_report",
]
