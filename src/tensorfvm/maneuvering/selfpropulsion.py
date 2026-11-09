"""Self-propulsion point: ``T(n) = R_T(V) / (1 - t)`` with the BEMT propeller.

The self-propulsion point is the shaft rate at which the propeller thrust, at the
wake-reduced inflow ``V_A = (1 - w) V``, balances the hull resistance divided by
the thrust-deduction factor.  It is solved by bracketing + bisection on ``n`` and
the **closure residual is returned with the answer**, because criterion (a) of
the node ("``T = R_T`` within 2 %") is a statement about the residual, not about
the model's truth.

Hull-propeller interaction
--------------------------
``w`` and ``t`` are **EMPIRICAL** here and are flagged as such:

* wake fraction -- Harvald/Taylor single-screw regression ``w = 0.5 C_b - 0.05``
  (for KVLCC2 ``C_b = 0.8098`` gives ``w = 0.355``, in the 0.30-0.40 band quoted
  for full single-screw tankers);
* thrust deduction -- Holtrop (1984) regression
  ``t = 0.001979 L/(B - B/T) + 1.0585 B/L - 0.00524 - 0.1418 D^2/(B T)``.

Both are regressions, not solutions of the hull wake; the report states the
sensitivity of the predicted shaft rate to ``w`` instead of pretending the wake
was computed.  (A wake-resolving CFD face is a separate node -- the repository's
thin-shear-layer RANS gives only the boundary-layer state, not a propeller-plane
wake survey.)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from .hull import Kvlcc2Conditions, DEFAULT_CONDITIONS
from .propeller import PropellerGeometry, open_water_point
from .resistance import resistance_components

__all__ = ["SelfPropulsionPoint", "hull_interaction", "solve_self_propulsion"]


def hull_interaction(conditions: Optional[Kvlcc2Conditions] = None,
                     propeller: Optional[PropellerGeometry] = None,
                     wake: Optional[float] = None,
                     thrust_deduction: Optional[float] = None) -> Dict[str, object]:
    """Empirical wake fraction and thrust deduction with their provenance."""
    c = conditions or DEFAULT_CONDITIONS
    g = propeller or PropellerGeometry()
    w = 0.5 * c.block_coefficient - 0.05 if wake is None else float(wake)
    if thrust_deduction is None:
        t = (0.001979 * c.length / (c.beam - c.beam / c.draft)
             + 1.0585 * c.beam / c.length
             - 0.00524
             - 0.1418 * g.diameter ** 2 / (c.beam * c.draft))
    else:
        t = float(thrust_deduction)
    return {
        "wake_fraction": w,
        "thrust_deduction": t,
        "hull_efficiency": (1.0 - t) / (1.0 - w),
        "provenance": {
            "wake_fraction": "EMPIRICAL (Harvald/Taylor 0.5Cb-0.05)",
            "thrust_deduction": "EMPIRICAL (Holtrop 1984 regression)",
        },
    }


@dataclass
class SelfPropulsionPoint:
    """Self-propulsion state at a given ship speed (model scale)."""

    speed: float
    rps: float
    advance_ratio: float
    KT: float
    KQ: float
    eta_open_water: float
    thrust: float
    torque: float
    thrust_required: float
    resistance: float
    wake_fraction: float
    thrust_deduction: float
    throttle: float
    effective_power: float
    delivered_power: float
    quasi_propulsive_efficiency: float
    residual: float
    iterations: int
    provenance: Dict[str, str] = field(default_factory=dict)

    @property
    def rpm(self) -> float:
        return 60.0 * self.rps

    @property
    def closure_pct(self) -> float:
        return 100.0 * self.residual

    def as_dict(self) -> Dict[str, object]:
        out = {"speed_m_s": self.speed, "rps": self.rps, "rpm": self.rpm,
               "J": self.advance_ratio, "KT": self.KT, "KQ": self.KQ,
               "eta_open_water": self.eta_open_water, "thrust_N": self.thrust,
               "torque_Nm": self.torque,
               "thrust_required_N": self.thrust_required,
               "resistance_N": self.resistance,
               "wake_fraction": self.wake_fraction,
               "thrust_deduction": self.thrust_deduction,
               "throttle_tau": self.throttle,
               "effective_power_W": self.effective_power,
               "delivered_power_W": self.delivered_power,
               "quasi_propulsive_efficiency": self.quasi_propulsive_efficiency,
               "closure_residual_rel": self.residual,
               "closure_pct": self.closure_pct,
               "iterations": self.iterations}
        out["provenance"] = self.provenance
        return out


def solve_self_propulsion(conditions: Optional[Kvlcc2Conditions] = None,
                          propeller: Optional[PropellerGeometry] = None,
                          speed: Optional[float] = None,
                          wake: Optional[float] = None,
                          thrust_deduction: Optional[float] = None,
                          resistance: Optional[float] = None,
                          rps_bracket=(0.5, 60.0),
                          tol: float = 1e-12) -> SelfPropulsionPoint:
    """Solve the self-propulsion point (shaft rate) for one speed."""
    c = conditions or DEFAULT_CONDITIONS
    prop = propeller or PropellerGeometry()
    u = c.u_design if speed is None else float(speed)
    inter = hull_interaction(c, prop, wake, thrust_deduction)
    w = float(inter["wake_fraction"])
    t = float(inter["thrust_deduction"])
    r_t = (resistance_components(u, c).r_total if resistance is None
           else float(resistance))
    thrust_required = r_t / (1.0 - t)
    inflow = (1.0 - w) * u

    def thrust_at(n: float) -> float:
        J = inflow / (n * prop.diameter)
        return open_water_point(prop, J, rho=c.rho, n_rps=n).thrust

    lo, hi = rps_bracket
    f_lo = thrust_at(lo) - thrust_required
    f_hi = thrust_at(hi) - thrust_required
    if f_lo * f_hi > 0.0:
        raise ValueError(
            "self-propulsion thrust is not bracketed in the shaft-rate interval "
            f"[{lo}, {hi}]: residual {f_lo:.4g} / {f_hi:.4g} N")
    n = 0.5 * (lo + hi)
    it = 0
    for it in range(1, 400):
        n = 0.5 * (lo + hi)
        f = thrust_at(n) - thrust_required
        if abs(f) <= tol * thrust_required:
            break
        if f * f_lo < 0.0:
            hi = n
        else:
            lo, f_lo = n, f
    point = open_water_point(prop, inflow / (n * prop.diameter), rho=c.rho, n_rps=n)
    residual = (point.thrust - thrust_required) / thrust_required
    p_e = r_t * u
    p_d = 2.0 * math.pi * n * point.torque
    return SelfPropulsionPoint(
        speed=u, rps=n, advance_ratio=point.J, KT=point.KT, KQ=point.KQ,
        eta_open_water=point.eta, thrust=point.thrust, torque=point.torque,
        thrust_required=thrust_required, resistance=r_t,
        wake_fraction=w, thrust_deduction=t,
        throttle=thrust_required / (0.5 * c.rho * u * u * c.beam * c.draft),
        effective_power=p_e, delivered_power=p_d,
        quasi_propulsive_efficiency=p_e / p_d if p_d > 0 else 0.0,
        residual=residual, iterations=it,
        provenance={
            "resistance": "WITHHELD-anchored ITTC-78 chain (see resistance module)",
            "propeller_model": "BEMT on nominal KP505 geometry (see propeller module)",
            "wake_fraction": str(inter["provenance"]["wake_fraction"]),
            "thrust_deduction": str(inter["provenance"]["thrust_deduction"]),
            "closure": "REAL (root-find residual reported)",
        })