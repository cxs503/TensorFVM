"""Ice-domain guardrail validation engine.

Provides functions to validate ice parameters against physical bounds,
detect extrapolation risk, and optionally clamp values to safe ranges.

Usage
-----
>>> from tensorfvm.ice import validate_ice_parameters, ICE_THICKNESS, ICE_STRENGTH
>>> result = validate_ice_parameters({"ice_thickness": 1.5, "ice_strength": 3.0})
>>> result.is_valid
True
>>> result = validate_ice_parameters({"ice_thickness": 50.0})  # physically impossible
>>> result.is_valid
False
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .parameters import IceParameter, ALL_ICE_PARAMETERS


class GuardrailStatus(str, Enum):
    """Status of a guardrail check."""
    PASS = "pass"                    # Within calibration domain
    WARN = "warn"                    # Outside calibration, within physical bounds
    FAIL = "fail"                    # Outside physical bounds (rejected)


@dataclass
class IceGuardrailResult:
    """Result of validating a set of ice parameters."""
    is_valid: bool
    status: GuardrailStatus
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    extrapolation_risk: bool = False
    validated_params: dict[str, float] = field(default_factory=dict)
    clamped_params: dict[str, float] = field(default_factory=dict)

    @property
    def summary(self) -> str:
        """Human-readable summary of the validation result."""
        if self.is_valid:
            if self.extrapolation_risk:
                return f"WARN: {len(self.warnings)} extrapolation warning(s)"
            return "PASS: All parameters within calibration domain"
        return f"FAIL: {len(self.errors)} violation(s)"


def validate_ice_parameters(
    params: dict[str, float],
    *,
    required: Optional[set[str]] = None,
    allow_extrapolation: bool = False,
) -> IceGuardrailResult:
    """Validate ice parameters against physical and calibration bounds.

    Parameters
    ----------
    params : dict[str, float]
        Parameter name → value mapping. Names must match keys in
        ``ALL_ICE_PARAMETERS`` (e.g. ``"ice_thickness"``, ``"ice_strength"``).
    required : set[str], optional
        Set of parameter names that must be present. If None, only provided
        parameters are validated.
    allow_extrapolation : bool, default False
        If True, parameters outside the calibration domain are accepted
        (with warnings). If False, they cause validation failure.

    Returns
    -------
    IceGuardrailResult
        Validation result with errors, warnings, and status.

    Examples
    --------
    >>> result = validate_ice_parameters({
    ...     "ice_thickness": 1.5,
    ...     "ice_strength": 3.0,
    ...     "ice_friction": 0.1,
    ... })
    >>> result.is_valid
    True
    >>> result.status
    <GuardrailStatus.PASS: 'pass'>
    """
    errors: list[str] = []
    warnings: list[str] = []
    validated: dict[str, float] = {}
    extrapolation_risk = False

    # Check required parameters
    if required:
        missing = required - set(params.keys())
        for name in sorted(missing):
            errors.append(f"Required parameter '{name}' is missing")

    # Validate each provided parameter
    for name, value in params.items():
        param = ALL_ICE_PARAMETERS.get(name)
        if param is None:
            errors.append(f"Unknown ice parameter: '{name}'")
            continue

        # Type check
        if not isinstance(value, (int, float)):
            errors.append(f"Parameter '{name}' must be numeric, got {type(value).__name__}")
            continue

        value = float(value)

        # Hard bound check
        is_valid, err_msg = param.validate(value)
        if not is_valid:
            errors.append(err_msg)
            continue

        # Soft bound check (extrapolation risk)
        needs_extrap, warn_msg = param.check_extrapolation(value)
        if needs_extrap:
            extrapolation_risk = True
            if allow_extrapolation:
                warnings.append(warn_msg)
            else:
                errors.append(warn_msg)
                continue

        validated[name] = value

    # Determine overall status
    if errors:
        status = GuardrailStatus.FAIL
        is_valid = False
    elif extrapolation_risk and not allow_extrapolation:
        status = GuardrailStatus.FAIL
        is_valid = False
    elif extrapolation_risk:
        status = GuardrailStatus.WARN
        is_valid = True
    else:
        status = GuardrailStatus.PASS
        is_valid = True

    return IceGuardrailResult(
        is_valid=is_valid,
        status=status,
        errors=errors,
        warnings=warnings,
        extrapolation_risk=extrapolation_risk,
        validated_params=validated,
    )


def clamp_ice_parameters(
    params: dict[str, float],
    *,
    clamp_to_soft: bool = False,
) -> IceGuardrailResult:
    """Clamp ice parameters to physical bounds (and optionally soft bounds).

    Unlike ``validate_ice_parameters``, this function never rejects inputs.
    Instead, it clamps out-of-bounds values to the nearest valid bound and
    reports what was changed.

    Parameters
    ----------
    params : dict[str, float]
        Parameter name → value mapping.
    clamp_to_soft : bool, default False
        If True, clamp to soft bounds (calibration domain) instead of hard bounds.

    Returns
    -------
    IceGuardrailResult
        Result with ``clamped_params`` showing the adjusted values and
        ``warnings`` listing what was changed.

    Examples
    --------
    >>> result = clamp_ice_parameters({"ice_thickness": 50.0})
    >>> result.clamped_params["ice_thickness"]
    10.0
    """
    errors: list[str] = []
    warnings: list[str] = []
    clamped: dict[str, float] = {}
    extrapolation_risk = False

    for name, value in params.items():
        param = ALL_ICE_PARAMETERS.get(name)
        if param is None:
            errors.append(f"Unknown ice parameter: '{name}'")
            continue

        if not isinstance(value, (int, float)):
            errors.append(f"Parameter '{name}' must be numeric, got {type(value).__name__}")
            continue

        value = float(value)
        original = value

        if clamp_to_soft:
            lo, hi = param.bounds.soft_min, param.bounds.soft_max
        else:
            lo, hi = param.bounds.hard_min, param.bounds.hard_max

        if value < lo:
            warnings.append(
                f"{name}: {original} {param.bounds.unit} → {lo} "
                f"(clamped from below {'soft' if clamp_to_soft else 'hard'} bound)"
            )
            value = lo
            extrapolation_risk = True
        elif value > hi:
            warnings.append(
                f"{name}: {original} {param.bounds.unit} → {hi} "
                f"(clamped from above {'soft' if clamp_to_soft else 'hard'} bound)"
            )
            value = hi
            extrapolation_risk = True

        clamped[name] = value

    return IceGuardrailResult(
        is_valid=len(errors) == 0,
        status=GuardrailStatus.WARN if warnings else GuardrailStatus.PASS,
        errors=errors,
        warnings=warnings,
        extrapolation_risk=extrapolation_risk,
        clamped_params=clamped,
    )


def check_extrapolation_risk(
    params: dict[str, float],
) -> dict[str, tuple[bool, Optional[str]]]:
    """Check which parameters are outside their calibration domain.

    Returns a dict mapping parameter name to (needs_extrapolation, warning_message).
    Parameters within the calibration domain have (False, None).

    Examples
    --------
    >>> risks = check_extrapolation_risk({"ice_thickness": 6.0, "ice_strength": 2.0})
    >>> risks["ice_thickness"][0]
    True
    >>> risks["ice_strength"][0]
    False
    """
    result = {}
    for name, value in params.items():
        param = ALL_ICE_PARAMETERS.get(name)
        if param is None:
            result[name] = (True, f"Unknown parameter: '{name}'")
            continue
        needs_extrap, msg = param.check_extrapolation(float(value))
        result[name] = (needs_extrap, msg)
    return result


def get_parameter_report(params: dict[str, float]) -> str:
    """Generate a human-readable report of parameter status.

    Shows each parameter's value, its bounds, and whether it's within
    the typical range, calibration domain, or physical bounds.
    """
    lines = ["Ice Parameter Status Report", "=" * 40]
    for name, value in sorted(params.items()):
        param = ALL_ICE_PARAMETERS.get(name)
        if param is None:
            lines.append(f"  {name}: UNKNOWN PARAMETER")
            continue

        b = param.bounds
        if param.bounds.is_hard_violation(value):
            status = "OUT OF PHYSICAL BOUNDS"
        elif param.bounds.is_soft_violation(value):
            status = "EXTRAPOLATION RISK"
        elif param.bounds.is_typical(value):
            status = "TYPICAL"
        else:
            status = "WITHIN CALIBRATION"

        lines.append(
            f"  {b.symbol} ({name}) = {value} {b.unit}  [{status}]"
        )
        lines.append(
            f"    physical: [{b.hard_min}, {b.hard_max}]  "
            f"calibration: [{b.soft_min}, {b.soft_max}]  "
            f"typical: [{b.typical_min}, {b.typical_max}]"
        )
    return "\n".join(lines)
