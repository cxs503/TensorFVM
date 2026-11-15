"""Level-ice resistance with the guardrail wired into the physics entry point.

Why this module exists
----------------------
The JOUR-ICE-STRUCT-GUARD anchor failed because the guardrail lived in the
schema layer: the *load* formula happily consumed ``h = 50 m`` (2450 MN-class
resistance) and ``sigma_c = 100 MPa`` (143.9 MN) because nothing between the
JSON payload and the arithmetic asked whether those numbers were in the
formula's domain.  Here the order is inverted: every quantity that reaches the
Lindqvist arithmetic has first passed :func:`tensorfvm.ice.guardrails.validate_ice_parameters`
over the full 12-parameter input set (8 ice properties + 4 hull variables),
and the default mode raises instead of extrapolating.

Model
-----
Lindqvist, G. (1989) "A straightforward method for calculation of ice
resistance of ships", POAC-89:

    R = (R_crushing + R_bending) * (1 + 1.4 V / sqrt(g h))
        + R_submersion + R_drag

    R_crushing   = sigma_c h^2 tan(phi) (1 + mu / tan(phi))
    R_bending    = (sigma_f h^2 / (6 l_c)) (1 + mu pi l_c / (4 h)) * B,
                   l_c = (D / (rho_w g))^(1/4),
                   D   = E h^3 / (12 (1 - nu^2)),  sigma_f = sigma_c / 4
    R_submersion = (rho_w - rho_ice) g h B (1 + mu cos(phi)/sin(phi)) * 0.3 L
    R_drag       = 0.5 rho_w B h V^2 * 0.1

Two caveats carried over from the platform-side review of the same formula
(ship-platform audit m1720, `backend/physics/ice_uq_resistance.py`), recorded
here rather than hidden:

* the ``0.3 L`` length scale in the submersion term and the ``0.1`` factor on
  the drag term are not traceable to the published paper — they are
  engineering constants of that implementation;
* the model is **uncalibrated** here.  Calibration factors fitted per ship are
  circular (they reproduce the trial mean at the nominal point by
  construction), and out of sample the formula misses the 15 % band against the
  three published trial means.  Accuracy against full-scale trials is
  therefore *not* claimed by this module, and no benchmark in this repository
  grades it; the benchmark grades the *guard* (see
  :mod:`tensorfvm.ice.benchmark`).

What is claimed: with the guard in path, an input outside the 12-parameter
validity domain cannot reach the arithmetic at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional

from .guardrails import (
    GuardrailStatus,
    IceGuardrailResult,
    IceParameterDomainError,
    clamp_ice_parameters,
    validate_ice_parameters,
)
from .parameters import (
    ALL_ICE_PARAMETERS,
    ICE_RESISTANCE_INPUTS,
    SHIP_PARAMETERS,
)

GRAVITY = 9.81
SEAWATER_DENSITY = 1025.0
ICE_DENSITY = 917.0
ICE_ELASTIC_MODULUS_PA = 2.5e9
ICE_POISSON_RATIO = 0.33

Mode = Literal["reject", "clamp"]


@dataclass(frozen=True)
class IceShipGeometry:
    """Hull variables used by the level-ice resistance formula."""
    length_m: float
    beam_m: float
    draft_m: float
    block_coefficient: float
    bow_angle_deg: float = 30.0


@dataclass(frozen=True)
class IceCondition:
    """Ice properties used by the level-ice resistance formula."""
    thickness_m: float
    strength_mpa: float
    friction_coeff: float
    speed_ms: float


@dataclass
class GuardedResistance:
    """Result of a guarded level-ice resistance evaluation.

    Attributes
    ----------
    resistance_kN
        Resistance of the *guarded* (validated or clamped) inputs.  ``None``
        when the guard rejected the inputs and ``mode='reject'``.
    unguarded_resistance_kN
        Resistance the same formula would have returned for the raw inputs.
        Present so the magnitude of the silent extrapolation that the guard
        prevented is auditable, not just asserted.  ``None`` if the raw inputs
        cannot be evaluated at all (non-finite).
    guard
        Full :class:`IceGuardrailResult` for the raw inputs.
    """
    resistance_kN: Optional[float]
    unguarded_resistance_kN: Optional[float]
    guard: IceGuardrailResult
    mode: str
    used_params: dict[str, float] = field(default_factory=dict)
    components_kN: dict[str, float] = field(default_factory=dict)

    @property
    def extrapolation_blocked(self) -> bool:
        """True when the guard stopped an input the formula should not see."""
        return (not self.guard.is_valid) or self.guard.extrapolation_risk

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "resistance_kN": self.resistance_kN,
            "unguarded_resistance_kN": self.unguarded_resistance_kN,
            "components_kN": dict(self.components_kN),
            "guard_status": self.guard.status.value,
            "guard_valid": self.guard.is_valid,
            "extrapolation_risk": self.guard.extrapolation_risk,
            "errors": list(self.guard.errors),
            "warnings": list(self.guard.warnings),
            "used_params": dict(self.used_params),
        }


def lindqvist_terms(geometry: IceShipGeometry, ice: IceCondition) -> dict[str, float]:
    """Component terms of the Lindqvist (1989) level-ice resistance, in kN.

    Returns keys ``crushing``, ``bending``, ``submersion``, ``speed`` and the
    dimensionless ``velocity_multiplier``.  The total is
    ``(crushing + bending) * velocity_multiplier + submersion + speed``.
    """
    L, B = geometry.length_m, geometry.beam_m
    phi = math.radians(geometry.bow_angle_deg)
    h = ice.thickness_m
    sigma_c = ice.strength_mpa * 1e6
    mu = ice.friction_coeff
    V = ice.speed_ms

    if phi > 0.01:
        f_crushing = sigma_c * h * h * math.tan(phi) * (1.0 + mu / math.tan(phi))
    else:
        f_crushing = sigma_c * h * h * 0.5

    sigma_f = sigma_c / 4.0
    D = ICE_ELASTIC_MODULUS_PA * h**3 / (12.0 * (1.0 - ICE_POISSON_RATIO**2))
    l_c = (D / (SEAWATER_DENSITY * GRAVITY)) ** 0.25
    f_bend_per_w = sigma_f * h**2 / (6.0 * l_c) * (1.0 + mu * math.pi * l_c / (4.0 * h))
    f_bending = f_bend_per_w * B

    if phi > 0.01:
        f_sub_per_l = ((SEAWATER_DENSITY - ICE_DENSITY) * GRAVITY * h * B
                       * (1.0 + mu * math.cos(phi) / math.sin(phi)))
    else:
        f_sub_per_l = (SEAWATER_DENSITY - ICE_DENSITY) * GRAVITY * h * B
    f_submersion = f_sub_per_l * 0.3 * L

    velocity_multiplier = 1.0 + 1.4 * V / math.sqrt(GRAVITY * h) if h > 0 else 1.0
    f_drag = 0.5 * SEAWATER_DENSITY * B * h * V**2 * 0.1

    return {
        "crushing": f_crushing / 1000.0,
        "bending": f_bending / 1000.0,
        "submersion": f_submersion / 1000.0,
        "speed": f_drag / 1000.0,
        "velocity_multiplier": velocity_multiplier,
    }


def level_ice_resistance_kN(geometry: IceShipGeometry, ice: IceCondition) -> float:
    """Uncalibrated Lindqvist total level-ice resistance in kN.

    This function performs **no** domain checking; it is the arithmetic the
    guard exists to protect.  Public callers should use
    :func:`guarded_level_ice_resistance`.
    """
    t = lindqvist_terms(geometry, ice)
    return ((t["crushing"] + t["bending"]) * t["velocity_multiplier"]
            + t["submersion"] + t["speed"])


def geometry_from_params(params: Mapping[str, float], *,
                        bow_angle_deg: float = 30.0) -> IceShipGeometry:
    """Build hull geometry from a validated parameter mapping."""
    missing = [k for k in SHIP_PARAMETERS if k not in params]
    if missing:
        raise KeyError(f"missing ship parameters: {missing}")
    return IceShipGeometry(
        length_m=float(params["ship_length"]),
        beam_m=float(params["ship_beam"]),
        draft_m=float(params["ship_draft"]),
        block_coefficient=float(params["ship_block_coefficient"]),
        bow_angle_deg=float(bow_angle_deg),
    )


def condition_from_params(params: Mapping[str, float]) -> IceCondition:
    """Build the ice condition from a validated parameter mapping."""
    return IceCondition(
        thickness_m=float(params["ice_thickness"]),
        strength_mpa=float(params["ice_strength"]),
        friction_coeff=float(params["ice_friction"]),
        speed_ms=float(params["ice_speed"]),
    )


def _raw_resistance_if_possible(params: Mapping[str, float], *,
                               bow_angle_deg: float) -> Optional[float]:
    """Evaluate the formula on raw inputs; None when they are non-finite."""
    needed = ("ice_thickness", "ice_strength", "ice_friction", "ice_speed")
    if any(k not in params for k in needed):
        return None
    values = [params[k] for k in needed]
    ship = [params.get(k) for k in SHIP_PARAMETERS]
    for v in list(values) + ship:
        if v is None:
            return None
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if not math.isfinite(float(v)):
            return None
    if any(s < 0 for s in ship[:3]) or params["ice_thickness"] <= 0:
        return None
    try:
        return level_ice_resistance_kN(
            geometry_from_params(params, bow_angle_deg=bow_angle_deg),
            condition_from_params(params),
        )
    except (KeyError, ZeroDivisionError, ValueError):
        return None


def guarded_level_ice_resistance(
    params: Mapping[str, float],
    *,
    mode: Mode = "reject",
    allow_extrapolation: bool = False,
    bow_angle_deg: float = 30.0,
) -> GuardedResistance:
    """Evaluate level-ice resistance **through** the guardrail.

    Parameters
    ----------
    params : mapping
        The 12-parameter input set (see
        :data:`tensorfvm.ice.parameters.ICE_RESISTANCE_INPUTS`).  Extra keys are
        rejected as unknown parameters; missing keys are errors.
    mode : {"reject", "clamp"}
        ``reject`` (default) refuses to compute when any input violates the
        guardrail and returns ``resistance_kN=None``.  ``clamp`` computes on the
        values clamped to the validity domain and records what was clamped —
        the only way an out-of-domain number can reach the formula, and it is
        always flagged by ``extrapolation_risk``.
    allow_extrapolation : bool, default False
        If True, values inside the hard (validity) bounds but outside the
        calibration band are accepted with a warning instead of being rejected.
    bow_angle_deg : float, default 30.0
        Bow angle from horizontal, not part of the guarded 12-parameter set
        (it is a shape descriptor rather than a physical magnitude); defaults to
        Lindqvist's nominal 30 deg.

    Returns
    -------
    GuardedResistance
        Guard verdict, guarded resistance, and the unguarded resistance that
        the raw inputs would have produced (for audit).

    Raises
    ------
    IceParameterDomainError
        Only in ``mode='reject'`` **and** when ``strict=True`` style callers ask
        for it via :func:`require_guarded_level_ice_resistance`.
    """
    raw = dict(params)
    unguarded = _raw_resistance_if_possible(raw, bow_angle_deg=bow_angle_deg)

    if mode == "clamp":
        guard = clamp_ice_parameters(raw)
        used = dict(guard.clamped_params)
        # A clamp result that dropped a NaN cannot compute a load.
        complete = all(k in used for k in ICE_RESISTANCE_INPUTS)
        # Re-check the *clamped* values: clamping is only an honest repair if
        # what lands downstream is itself inside the validity domain.
        domain = validate_ice_parameters(used, allow_extrapolation=True)
        if not guard.is_valid or not complete or not domain.is_valid:
            return GuardedResistance(
                resistance_kN=None, unguarded_resistance_kN=unguarded,
                guard=guard, mode=mode, used_params=used,
            )
        ship = geometry_from_params(used, bow_angle_deg=bow_angle_deg)
        ice = condition_from_params(used)
        return GuardedResistance(
            resistance_kN=level_ice_resistance_kN(ship, ice),
            unguarded_resistance_kN=unguarded,
            guard=guard, mode=mode, used_params=used,
            components_kN=lindqvist_terms(ship, ice),
        )

    if mode != "reject":
        raise ValueError(f"unknown mode {mode!r}; expected 'reject' or 'clamp'")

    guard = validate_ice_parameters(
        raw, required=set(ICE_RESISTANCE_INPUTS),
        allow_extrapolation=allow_extrapolation,
    )
    if not guard.is_valid:
        return GuardedResistance(
            resistance_kN=None, unguarded_resistance_kN=unguarded,
            guard=guard, mode=mode,
        )
    ship = geometry_from_params(guard.validated_params, bow_angle_deg=bow_angle_deg)
    ice = condition_from_params(guard.validated_params)
    return GuardedResistance(
        resistance_kN=level_ice_resistance_kN(ship, ice),
        unguarded_resistance_kN=unguarded,
        guard=guard, mode=mode, used_params=dict(guard.validated_params),
        components_kN=lindqvist_terms(ship, ice),
    )


def require_guarded_level_ice_resistance(
    params: Mapping[str, float], *, bow_angle_deg: float = 30.0
) -> float:
    """Guarded resistance or :class:`IceParameterDomainError` (hard precondition)."""
    result = guarded_level_ice_resistance(params, mode="reject",
                                         bow_angle_deg=bow_angle_deg)
    if result.resistance_kN is None:
        raise IceParameterDomainError(result.guard)
    return result.resistance_kN


def parameter_names() -> tuple[str, ...]:
    """All guarded parameter names (12)."""
    return ICE_RESISTANCE_INPUTS


def known_parameters() -> dict[str, Any]:
    """Registry snapshot for tooling / CLI JSON output."""
    return {
        name: {
            "symbol": p.bounds.symbol,
            "unit": p.bounds.unit,
            "hard": [p.bounds.hard_min, p.bounds.hard_max],
            "soft": [p.bounds.soft_min, p.bounds.soft_max],
            "typical": [p.bounds.typical_min, p.bounds.typical_max],
            "reference": p.bounds.reference,
        }
        for name, p in sorted(ALL_ICE_PARAMETERS.items())
    }


__all__ = [
    "GuardedResistance",
    "IceCondition",
    "IceShipGeometry",
    "geometry_from_params",
    "condition_from_params",
    "guarded_level_ice_resistance",
    "level_ice_resistance_kN",
    "lindqvist_terms",
    "require_guarded_level_ice_resistance",
    "known_parameters",
    "parameter_names",
    "GuardrailStatus",
]