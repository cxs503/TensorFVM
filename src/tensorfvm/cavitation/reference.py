"""Reference values and bands, each with an explicit provenance status.

Status vocabulary used throughout the node report
-------------------------------------------------
``EXACT``        closed-form solution of the same equations (conformal mapping,
                 Rayleigh collapse time) -- error is discretisation only.
``ANALYTIC``     a published closed-form correlation evaluated here (Abbott &
                 von Doenhoff thickness correction for the lift slope).
``CRITERION``    a criterion *definition* taken from a procedure (the ITTC
                 minimum-pressure inception criterion); the definition is used,
                 not a tabulated number.
``QUALITATIVE``  a trend/shape/order-of-magnitude taken from the literature with
                 no digitised data in this workspace; never used as a verdict on
                 an absolute number.
``WITHHELD``     a quantity whose primary source is not in this workspace.

There is **no** digitised experimental cavitation-erosion dataset in this
workspace (grep for ``erosion``/``cavitation`` returns nothing outside this new
package).  Absolute erosion validation is therefore ``WITHHELD``; what the node
does verify is (i) the exact single-bubble physics, (ii) the panel-method
``Cp_min`` against the exact conformal solution, and (iii) the *shape* of the
erosion-versus-cavitation-number curve against the documented trend band.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

__all__ = [
    "ITTC_INCEPTION_CRITERION", "TrendBand", "EROSION_PEAK_BAND",
    "EROSION_EXPONENT_BAND", "CAVITY_LENGTH_BAND", "SHEDDING_STROUHAL_BAND",
    "reference_damage_curve", "r_squared", "band_check", "LIFT_SLOPE",
]


#: ITTC cavitation-inception criterion actually used by this node.  The
#: procedure defines model-scale inception by the *minimum pressure* condition
#: ``p_min = p_v``, i.e. ``sigma_i = -Cp_min``; the nuclei/air-content
#: correction is a facility-dependent additive term, which is why the node
#: reports it separately (``sigma_i*``) instead of mixing it into ``sigma_i``.
ITTC_INCEPTION_CRITERION: Dict[str, str] = {
    "statement": "cavitation starts where the local pressure reaches p_v: sigma_i = -Cp_min",
    "status": "CRITERION",
    "procedure": "ITTC recommended procedure, cavitation inception (minimum-pressure criterion)",
    "note": ("the tabulated inception values of that procedure are not available in this "
             "workspace, so the node verifies sigma_i against independent exact/analytic "
             "solutions rather than against a quoted table (WITHHELD for absolute tables)"),
}

#: Lift-slope reference (Abbott & von Doenhoff thickness correction).
LIFT_SLOPE = {"formula": "Cl_alpha = 2 pi (1 + 0.77 t/c)", "status": "ANALYTIC"}


@dataclass(frozen=True)
class TrendBand:
    """A literature *band* (not a point value) with its provenance."""

    name: str
    low: float
    high: float
    status: str
    provenance: str
    used_for: str

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def as_dict(self) -> Dict[str, object]:
        return {"name": self.name, "low": self.low, "high": self.high,
                "status": self.status, "provenance": self.provenance,
                "used_for": self.used_for}


#: ``sigma/sigma_i`` at which hydrofoil erosion damage peaks: damage grows fast
#: just below inception and falls again once the cavity is long and its collapse
#: has moved downstream.  The band is deliberately wide.
EROSION_PEAK_BAND = TrendBand(
    name="peak erosion cavitation number ratio", low=1.05, high=2.6, status="QUALITATIVE",
    provenance="cavitation-erosion literature trend for hydrofoils/rudders (peak damage "
               "at an intermediate sigma below inception); no digitised dataset in workspace",
    used_for="criterion (b) shape check of E(sigma); not an absolute-value verdict",
)

#: Erosion power-law exponent in ``E ~ (p_impact - p_v)^n``.
EROSION_EXPONENT_BAND = TrendBand(
    name="erosion power-law exponent n", low=2.0, high=3.0, status="QUALITATIVE",
    provenance="impact/pitting erosion models use 2 <= n <= 3 (ductile metals)",
    used_for="choice of the default exponent and its sensitivity sweep",
)

#: Cavity-length constant ``C_L`` in ``L/c = C_L sqrt(sigma_i/sigma - 1)``.
CAVITY_LENGTH_BAND = TrendBand(
    name="cavity-length constant C_L", low=0.5, high=1.5, status="QUALITATIVE",
    provenance="square-root cavity closure scaling; the constant is of order one",
    used_for="erosion attenuation sensitivity",
)

#: Shedding Strouhal number of a partial cavity.
SHEDDING_STROUHAL_BAND = TrendBand(
    name="cavity shedding Strouhal number", low=0.2, high=0.4, status="QUALITATIVE",
    provenance="partial-cavity shedding/re-entrant jet Strouhal numbers 0.2-0.4",
    used_for="impact-rate estimate f = St V / L",
)


def reference_damage_curve(ratios: Sequence[float], peak: float = 1.6,
                           tail_fraction: float = 0.10,
                           tail_ratio: float = 4.0) -> np.ndarray:
    """Trend reference ``E/E_max`` versus ``x = sigma_i/sigma`` for the shape check.

    Family ``y = (x-1)^p exp(-q (x-1))`` on ``x >= 1`` (zero above inception),
    which encodes the two documented qualitative features of hydrofoil/rudder
    damage curves: a *steep* rise below inception, and a *slow* decay once the
    cavity is long and its collapse has moved into the wake.

    The two exponents are not free parameters chosen to fit this node's model;
    they follow from two qualitative constraints taken from the literature:

    * the damage peaks at ``x = 1 + p/q`` (band :data:`EROSION_PEAK_BAND`,
      midpoint 1.6 used here);
    * at ``x = tail_ratio`` the damage is still ~``tail_fraction`` of its peak
      (``tail_fraction = 0.1``: long-cavity damage is weak but not zero).

    Solving the pair ``p/q = 0.6`` and ``(tail_ratio-1)^p exp(-q(tail_ratio-1))
    = tail_fraction * (0.6)^p exp(-0.6 q)`` gives ``q = 1.60``, ``p = 0.96``.
    """
    x = np.asarray(ratios, dtype=float)
    ratio = peak - 1.0
    # closed-form solution of the two constraints
    q = math.log(((tail_ratio - 1.0) / ratio) ** 0.0) # placeholder, replaced below
    # solve (tail_ratio-1)^p exp(-q (tail_ratio-1)) = f * ratio^p exp(-q ratio),
    # p = ratio * q  ->  ((tail_ratio-1)/ratio)^(ratio q) = f exp(q (tail_ratio-1-ratio))
    f = tail_fraction
    lo, hi = 1e-6, 50.0
    def residual(qq):
        lhs = math.exp(ratio * qq * math.log((tail_ratio - 1.0) / ratio))
        rhs = f * math.exp(qq * (tail_ratio - 1.0 - ratio))
        return lhs - rhs
    q_sol = lo
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
        q_sol = 0.5 * (lo + hi)
    p_sol = ratio * q_sol
    y = np.where(x > 1.0, np.clip(x - 1.0, 0.0, None) ** p_sol
                 * np.exp(-q_sol * np.clip(x - 1.0, 0.0, None)), 0.0)
    y = y / float(np.max(y)) if np.max(y) > 0 else y
    del q
    return y


def trend_shape_exponents(peak: float = 1.6, tail_fraction: float = 0.10,
                          tail_ratio: float = 4.0):
    """``(p, q)`` of :func:`reference_damage_curve` (see its docstring)."""
    ratio = peak - 1.0
    f = tail_fraction
    lo, hi = 1e-6, 50.0
    def residual(qq):
        return (math.exp(ratio * qq * math.log((tail_ratio - 1.0) / ratio))
                - f * math.exp(qq * (tail_ratio - 1.0 - ratio)))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if residual(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    q_sol = 0.5 * (lo + hi)
    return ratio * q_sol, q_sol


def r_squared(reference: Sequence[float], model: Sequence[float]) -> float:
    """Coefficient of determination of ``model`` against ``reference``."""
    r = np.asarray(reference, dtype=float)
    m = np.asarray(model, dtype=float)
    resid = float(np.sum((r - m) ** 2))
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if resid == 0.0 else 0.0
    return 1.0 - resid / ss_tot


def band_check(name: str, value: float, band: TrendBand) -> Dict[str, object]:
    """Machine-readable band check for the report JSON."""
    return {"band": band.name, "value": float(value), "low": band.low,
            "high": band.high, "passed": bool(band.contains(value)),
            "status": band.status, "label": name}


def scaled(values: Sequence[float]) -> np.ndarray:
    """Normalise a curve to its maximum (used to compare shapes)."""
    v = np.asarray(values, dtype=float)
    m = float(np.max(v)) if v.size else 0.0
    return v / m if m > 0.0 else v