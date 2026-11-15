"""Ice-domain guardrail validation engine.

Provides functions to validate ice parameters against physical bounds,
detect extrapolation risk, and optionally clamp values to safe ranges.

The engine is **total on non-finite input**: ``NaN`` / ``inf`` / ``-inf`` never
reach the caller as an accepted value.  This is the m22342 repair of the
m22209 delivery, where ``isinstance(value, (int, float))`` was the only type
gate, so ``float('nan')`` compared ``False`` against every bound
(``nan < lo`` and ``nan > hi`` are both False) and was reported as "TYPICAL" —
a guardrail that let the most common silent garbage (0/0, uninitialised
arrays, ``numpy.nan`` propagation) straight through to the physics.

Usage
-----
>>> from tensorfvm.ice import validate_ice_parameters
>>> validate_ice_parameters({"ice_thickness": 1.5, "ice_strength": 3.0}).is_valid
True
>>> validate_ice_parameters({"ice_thickness": 50.0}).is_valid  # out of domain
False
>>> validate_ice_parameters({"ice_thickness": float("nan")}).is_valid
False
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .parameters import ALL_ICE_PARAMETERS


class GuardrailStatus(str, Enum):
    """Status of a guardrail check."""
    PASS = "pass"                    # Within calibration domain
    WARN = "warn"                    # Outside calibration, within physical bounds
    FAIL = "fail"                    # Outside physical bounds (rejected)


class IceParameterDomainError(ValueError):
    """Raised by :func:`require_ice_parameters` when inputs leave the domain.

    Carries the full :class:`IceGuardrailResult` so a caller can log exactly
    which parameter failed and by how much without re-parsing strings.
    """

    def __init__(self, result: "IceGuardrailResult") -> None:
        self.result = result
        detail = "; ".join(result.errors) if result.errors else result.summary
        super().__init__(f"ice parameters outside guardrail domain: {detail}")


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
    non_finite_params: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable summary of the validation result."""
        if self.is_valid:
            if self.extrapolation_risk:
                return f"WARN: {len(self.warnings)} extrapolation warning(s)"
            return "PASS: All parameters within calibration domain"
        return f"FAIL: {len(self.errors)} violation(s)"


def _finite_value(name: str, value: Any, errors: list[str],
                  non_finite: list[str]) -> Optional[float]:
    """Coerce ``value`` to a finite float, or record why that is impossible.

    Returns None when the input must not be used downstream.  Booleans are
    rejected explicitly: ``isinstance(True, int)`` is True and a bare ``True``
    silently becomes ``1.0`` of whatever physical unit the parameter carries.
    """
    if isinstance(value, bool):
        errors.append(
            f"Parameter '{name}' must be a real number, got bool "
            f"({value!r}) — booleans are not physical quantities"
        )
        return None
    if not isinstance(value, (int, float)):
        errors.append(
            f"Parameter '{name}' must be numeric, got {type(value).__name__}"
        )
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        # NaN has no ordering, so bound comparisons are silently all-False;
        # +-inf has an ordering but is not a physical measurement.  Both are
        # silent-garbage classes a guardrail exists to stop.
        errors.append(
            f"Parameter '{name}' = {numeric!r} is not finite (NaN/inf) — "
            f"no physical bound can accept a non-finite value"
        )
        non_finite.append(name)
        return None
    return numeric


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

    Notes
    -----
    Non-finite inputs (``NaN``, ``inf``, ``-inf``) are always errors, whatever
    ``allow_extrapolation`` says: extrapolation is a statement about distance
    from a calibration point, not a licence to accept non-numbers.

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
    non_finite: list[str] = []
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

        numeric = _finite_value(name, value, errors, non_finite)
        if numeric is None:
            continue

        # Hard bound check
        is_valid, err_msg = param.validate(numeric)
        if not is_valid:
            errors.append(err_msg)
            continue

        # Soft bound check (extrapolation risk)
        needs_extrap, warn_msg = param.check_extrapolation(numeric)
        if needs_extrap:
            extrapolation_risk = True
            if allow_extrapolation:
                warnings.append(warn_msg)
            else:
                errors.append(warn_msg)
                continue

        validated[name] = numeric

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
        non_finite_params=non_finite,
    )


def require_ice_parameters(
    params: dict[str, float],
    *,
    required: Optional[set[str]] = None,
    allow_extrapolation: bool = False,
) -> dict[str, float]:
    """Validate and return the parameters, or raise :class:`IceParameterDomainError`.

    This is the gate the physics entry points call: it converts a soft
    "please check the result object" convention into a hard precondition, so a
    silent extrapolation cannot reach a load calculation by forgetting an
    ``if``.

    >>> require_ice_parameters({"ice_thickness": 1.2})["ice_thickness"]
    1.2
    """
    result = validate_ice_parameters(
        params, required=required, allow_extrapolation=allow_extrapolation
    )
    if not result.is_valid:
        raise IceParameterDomainError(result)
    return result.validated_params


def clamp_ice_parameters(
    params: dict[str, float],
    *,
    clamp_to_soft: bool = False,
) -> IceGuardrailResult:
    """Clamp ice parameters to physical bounds (and optionally soft bounds).

    Unlike ``validate_ice_parameters``, this function never rejects finite
    inputs.  Instead, it clamps out-of-bounds values to the nearest valid bound
    and reports what was changed.  ``+-inf`` is clamped to the corresponding
    bound (its direction is known); ``NaN`` cannot be clamped, because "which
    side is NaN on?" has no answer, so it is reported as an error and omitted
    from ``clamped_params`` rather than being echoed back as a poisoned value.

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
    >>> clamp_ice_parameters({"ice_thickness": 50.0}).clamped_params["ice_thickness"]
    10.0
    """
    errors: list[str] = []
    warnings: list[str] = []
    clamped: dict[str, float] = {}
    non_finite: list[str] = []
    extrapolation_risk = False

    for name, value in params.items():
        param = ALL_ICE_PARAMETERS.get(name)
        if param is None:
            errors.append(f"Unknown ice parameter: '{name}'")
            continue

        if isinstance(value, bool):
            errors.append(
                f"Parameter '{name}' must be a real number, got bool "
                f"({value!r}) — booleans are not physical quantities"
            )
            continue
        if not isinstance(value, (int, float)):
            errors.append(f"Parameter '{name}' must be numeric, got {type(value).__name__}")
            continue

        numeric = float(value)
        if math.isnan(numeric):
            errors.append(
                f"Parameter '{name}' = nan cannot be clamped (NaN has no "
                f"direction) — rejected instead of echoed back"
            )
            non_finite.append(name)
            continue

        value = numeric
        if math.isinf(value):
            non_finite.append(name)

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
        elif math.isinf(numeric):
            # +-inf always lands outside any finite bound, so this branch is
            # unreachable for a well-formed parameter table; keep the explicit
            # error instead of silently returning a non-finite value.
            errors.append(f"Parameter '{name}' = {original} could not be clamped")
            continue

        clamped[name] = value

    if errors:
        status = GuardrailStatus.FAIL
    elif warnings:
        status = GuardrailStatus.WARN
    else:
        status = GuardrailStatus.PASS

    return IceGuardrailResult(
        is_valid=len(errors) == 0,
        status=status,
        errors=errors,
        warnings=warnings,
        extrapolation_risk=extrapolation_risk,
        clamped_params=clamped,
        non_finite_params=non_finite,
    )


def check_extrapolation_risk(
    params: dict[str, float],
) -> dict[str, tuple[bool, Optional[str]]]:
    """Check which parameters are outside their calibration domain.

    Returns a dict mapping parameter name to (needs_extrapolation, warning_message).
    Parameters within the calibration domain have (False, None).

    Non-finite values are always flagged as needing extrapolation.

    Examples
    --------
    >>> risks = check_extrapolation_risk({"ice_thickness": 6.0, "ice_strength": 2.0})
    >>> risks["ice_thickness"][0]
    True
    >>> risks["ice_strength"][0]
    False
    """
    result: dict[str, tuple[bool, Optional[str]]] = {}
    for name, value in params.items():
        param = ALL_ICE_PARAMETERS.get(name)
        if param is None:
            result[name] = (True, f"Unknown parameter: '{name}'")
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            result[name] = (True, f"Non-numeric parameter: '{name}'")
            continue
        numeric = float(value)
        if not math.isfinite(numeric):
            result[name] = (True, f"{name} = {numeric!r} is not finite")
            continue
        needs_extrap, msg = param.check_extrapolation(numeric)
        result[name] = (needs_extrap, msg)
    return result


def get_parameter_report(params: dict[str, float]) -> str:
    """Generate a human-readable report of parameter status.

    Shows each parameter's value, its bounds, and whether it's within
    the typical range, calibration domain, or physical bounds.  Non-finite
    values are labelled ``NON-FINITE (REJECTED)`` — never ``TYPICAL``.
    """
    lines = ["Ice Parameter Status Report", "=" * 40]
    for name, value in sorted(params.items()):
        param = ALL_ICE_PARAMETERS.get(name)
        if param is None:
            lines.append(f"  {name}: UNKNOWN PARAMETER")
            continue

        b = param.bounds
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            lines.append(f"  {name}: NON-NUMERIC ({type(value).__name__}) — REJECTED")
            continue
        if not math.isfinite(numeric):
            lines.append(f"  {b.symbol} ({name}) = {value!r}  [NON-FINITE — REJECTED]")
            continue
        if b.is_hard_violation(numeric):
            status = "OUT OF PHYSICAL BOUNDS"
        elif b.is_soft_violation(numeric):
            status = "EXTRAPOLATION RISK"
        elif b.is_typical(numeric):
            status = "TYPICAL"
        else:
            status = "WITHIN CALIBRATION"

        lines.append(
            f"  {b.symbol} ({name}) = {numeric} {b.unit}  [{status}]"
        )
        lines.append(
            f"    physical: [{b.hard_min}, {b.hard_max}]  "
            f"calibration: [{b.soft_min}, {b.soft_max}]  "
            f"typical: [{b.typical_min}, {b.typical_max}]"
        )
    return "\n".join(lines)