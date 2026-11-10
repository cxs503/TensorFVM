"""Nusselt-number scaling and the acceptance ledger of the thermal face.

Two jobs:

* :func:`fit_power_law` -- least-squares fit of ``Nu = C Ra^p`` on the measured
  ``(Ra, Nu)`` pairs, returning ``p`` **with its standard error**, the
  coefficient, ``R^2`` and the local (pairwise) exponents.  Quoting a single
  exponent without its error bar is what lets a scaling claim survive on noise;
  the error bar here is the ordinary regression standard error, and the local
  exponents show whether the law is even a single power law over the range.
* :func:`evaluate_criteria` -- every campaign criterion as one record
  ``{id, description, value, reference, error, tolerance, passed, note}``, so
  the report and the machine gate read the same numbers.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np

from .reference import NU_EXPONENT_CLASSICAL

__all__ = ["fit_power_law", "local_exponents", "criterion", "evaluate_criteria"]


def fit_power_law(rayleigh: Sequence[float], nusselt: Sequence[float]
                  ) -> dict[str, float]:
    """Least-squares ``Nu = C Ra^p`` in log-log space, with the error of ``p``.

    The regression is ordinary least squares on ``(ln Ra, ln Nu)``; ``stderr``
    is the standard error of the slope, ``r2`` the coefficient of
    determination, and ``residual_max`` the largest relative deviation of the
    data from the fit (a two-parameter fit through four or more points can look
    excellent and still be the wrong law, so the residuals are reported).
    """
    ra = np.asarray(rayleigh, dtype=float)
    nu = np.asarray(nusselt, dtype=float)
    if ra.shape != nu.shape or ra.ndim != 1:
        raise ValueError("rayleigh and nusselt must be 1-D and of equal length")
    if ra.size < 3:
        raise ValueError("a power law needs at least 3 points to be falsifiable")
    if np.any(ra <= 0.0) or np.any(nu <= 0.0):
        raise ValueError("rayleigh and nusselt must be positive")
    x = np.log(ra)
    y = np.log(nu)
    design = np.vstack([x, np.ones_like(x)]).T
    (slope, intercept), residuals, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if rank < 2:
        raise ValueError("degenerate Rayleigh sweep: cannot fit a power law")
    dof = ra.size - 2
    if dof > 0:
        residual_norm = float(np.sum((y - (slope * x + intercept)) ** 2))
        variance = residual_norm / dof
        stderr = math.sqrt(variance / float(np.sum((x - x.mean()) ** 2)))
    else:
        stderr = float("nan")
    fitted = slope * x + intercept
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "exponent": float(slope),
        "coefficient": float(math.exp(intercept)),
        "stderr": float(stderr),
        "r2": float(1.0 - float(np.sum((y - fitted) ** 2)) / ss_tot) if ss_tot > 0 else 1.0,
        "residual_max": float(np.max(np.abs(np.exp(y - fitted) - 1.0))),
        "points": int(ra.size),
    }


def local_exponents(rayleigh: Sequence[float], nusselt: Sequence[float]
                    ) -> list[dict[str, float]]:
    """Pairwise exponent ``ln(Nu2/Nu1)/ln(Ra2/Ra1)`` between adjacent sweep points."""
    ra = np.asarray(rayleigh, dtype=float)
    nu = np.asarray(nusselt, dtype=float)
    if ra.size < 2:
        raise ValueError("need at least two sweep points")
    out = []
    for i in range(ra.size - 1):
        out.append({
            "ra_from": float(ra[i]),
            "ra_to": float(ra[i + 1]),
            "exponent": float(math.log(nu[i + 1] / nu[i]) / math.log(ra[i + 1] / ra[i])),
        })
    return out


def criterion(identifier: str, description: str, value: float | None,
              reference: float | None, tolerance: float, *,
              relative: bool = True, note: str = "") -> dict[str, object]:
    """One acceptance record; ``passed`` is False when the value is missing."""
    error = None
    passed = False
    if value is not None and reference is not None and math.isfinite(value):
        if relative:
            error = abs(value - reference) / abs(reference)
        else:
            error = abs(value - reference)
        passed = bool(error <= tolerance)
    return {
        "id": identifier,
        "description": description,
        "value": None if value is None else float(value),
        "reference": None if reference is None else float(reference),
        "error": error,
        "tolerance": float(tolerance),
        "relative": relative,
        "passed": passed,
        "note": note,
    }


def evaluate_criteria(records: Iterable[dict[str, object]]) -> dict[str, object]:
    """Aggregate criterion records into ``{criteria, passed_count, all_passed}``."""
    rows = list(records)
    return {
        "criteria": rows,
        "count": len(rows),
        "passed_count": sum(1 for r in rows if r.get("passed")),
        "all_passed": all(bool(r.get("passed")) for r in rows) and bool(rows),
    }


#: The campaign anchor for the scaling exponent, re-exported for the ledger.
CLASSICAL_EXPONENT = NU_EXPONENT_CLASSICAL