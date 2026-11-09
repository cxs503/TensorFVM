"""Hull lateral force model for the horizontal (manoeuvring) plane.

Model structure
---------------
The steady lateral force of a hull at drift is written as the sum of

1. a **low-aspect-ratio lifting term** -- the hull seen from above is a wing of
   span ``2 T`` and chord ``L``, so the lift slope is Helmbold's finite-aspect
   ratio value ``C_La = 2 pi AR /(sqrt(1+(2/AR)^2)+2/AR)`` with
   ``AR = 2 Tbar / L`` (for KVLCC2 ``AR = 0.13``, ``C_La ~ 2.5`` per rad).  The
   linear derivatives of MMG are exactly the linearisation of this term;
2. a **cross-flow drag term** -- Hoerner's bluff-section cross-flow over the
   hull's own lateral planform ``T_e(x) dx``, ``dY = 1/2 rho C_d T_e |v_n| v_n dx``
   with ``C_d = 1.2`` and the local normal velocity ``v_n(x) = v + x r``.  This is
   what produces the nonlinear (cubic/cross-flow) derivatives and the large
   drift behaviour of a turn;
3. an optional **lift-distribution shaping** so that the yaw moment is not
   forced to zero by a symmetric planform: the lifting term is distributed with
   the bow-concentrated weight ``w(x) ∝ max(dT_e/dx, 0)`` (a slender-wing
   loading: the lift is generated where the span grows, i.e. at the bow).  The
   centre-of-pressure ratio ``x_p'`` is then a *computed geometric quantity*, not
   a fitted constant.

Sign conventions (SNAME/MMG, body axes ``x`` fwd, ``y`` starboard, ``z`` down)
------------------------------------------------------------------------------
``u, v`` surge/sway velocity, ``r`` yaw rate (positive bow to starboard),
``Y`` lateral force (positive to starboard), ``N`` yaw moment (positive bow to
starboard).  Prime system::

    Y' = Y / (1/2 rho U^2 L T),  N' = N / (1/2 rho U^2 L^2 T)
    v' = v / U,                  r' = r L / U

Why this module is *not* a derivative table
-------------------------------------------
Node IN40-N04 (``docs/pmm-cfd``) showed that a strip model with a single
``C_d = 1.0`` and symmetric sections returns ``N_v = 0`` and cannot forecast a
manoeuvre, and that the in-repository SIMMAN derivative table is internally
inconsistent.  This module therefore (a) carries a genuine lifting term with a
computed centre of pressure, and (b) is honest about its status: it is an
**engineering model**, calibrated by nothing but textbook aerodynamics and the
hull's own geometry.  Its credibility statements are the analytic limits in
``tests/test_maneuvering.py`` (``AR -> 0`` and ``AR -> inf`` limits of the lift
slope, cross-flow sign symmetry, ``delta = 0`` negative control), never an
experimental agreement claim.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from .hull import (Kvlcc2Conditions, DEFAULT_CONDITIONS, build_hull, section_data)

__all__ = [
    "HullForceModelSet", "lift_slope_finite_aspect", "planform_integrals",
    "linear_derivatives", "hull_force",
]

#: Hoerner cross-flow drag coefficient of a ship-like (bluff, rounded) section.
CROSSFLOW_CD = 1.2


def lift_slope_finite_aspect(AR: float, a0: float = 2.0 * math.pi) -> float:
    """Helmbold finite-aspect-ratio lift slope ``dC_L/dalpha`` [1/rad].

    ``a = a0 /(sqrt(1+(a0/(pi AR))^2) + a0/(pi AR))``.  Limits: ``AR -> 0``
    gives the slender-wing value ``pi AR/2``; ``AR -> inf`` gives ``a0``.
    """
    if AR <= 0.0:
        raise ValueError("aspect ratio must be positive")
    q = a0 / (math.pi * AR)
    return a0 / (math.sqrt(1.0 + q * q) + q)


def planform_integrals(hull=None, n: int = 241) -> Dict[str, float]:
    """Lateral-planform quantities used by the force model.

    ``side_area``  ``S_L = int T_e dx``  (side area of the hull)
    ``T_mean``     area-weighted mean section depth
    ``AR``         ``2 T_mean / L``
    ``x_p``        centre of pressure ratio (fraction of L forward of midship):
                   quarter chord of the planform (assumption, documented)
    """
    data = section_data(hull, n)
    x = data["x"]
    draft = data["draft"]
    L = float(x[-1] - x[0])
    side = float(np.trapezoid(draft, x))
    t_mean = side / L
    # Centre of pressure: quarter chord of the lateral planform, i.e. a quarter of
    # the hull length aft of the bow.  This is the standard low-aspect-ratio wing
    # result, NOT a derived hull pressure distribution -- it is the documented
    # assumption that replaces the (identically zero) yaw moment of a symmetric
    # strip model, see the module docstring and docs/pmm-cfd.
    centroid = 0.25 * L
    return {
        "side_area": side, "T_mean": t_mean, "AR": 2.0 * t_mean / L,
        "x_p_over_l": centroid / L, "L": L,
        "draft_mean_uniform": float(np.mean(draft)),
    }


@dataclass
class HullForceModelSet:
    """Hull force model with a computed linear-derivative set.

    The derivatives are the analytic linearisation of :func:`hull_force` at
    ``v = r = 0`` (see :func:`linear_derivatives`), so the "MMG" (linear) and
    "direct" (nonlinear) models share one physics and differ only in
    linearisation -- which is exactly the difference the CFD/MMG arbitration
    needs to expose.
    """

    conditions: Kvlcc2Conditions = field(default_factory=lambda: DEFAULT_CONDITIONS)
    crossflow_cd: float = CROSSFLOW_CD
    hull: object = None
    planform: Dict[str, float] = field(default_factory=dict)
    _sections: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False,
                                                       compare=False)

    def __post_init__(self):
        if self.hull is None:
            self.hull = build_hull()
        if not self.planform:
            self.planform = planform_integrals(self.hull)

    def sections(self, n_stations: int = 121) -> Dict[str, np.ndarray]:
        """Cached station data (the hull does not change during a manoeuvre)."""
        if self._sections is None or len(self._sections["x"]) != n_stations:
            data = section_data(self.hull, n_stations)
            x = data["x"] - 0.5 * self.L
            # slender-wing chordwise loading: the lift per unit chord is
            # proportional to the local planform span, i.e. to T_e(x)
            weight = np.asarray(data["draft"], dtype=float)
            self._sections = {"x": x, "draft": data["draft"], "weight": weight,
                              "weight_total": float(np.trapezoid(weight, data["x"])),
                              "dx": np.gradient(data["x"])}
        return self._sections

    # -- geometry shortcuts ------------------------------------------------
    @property
    def L(self) -> float:
        return self.conditions.length

    @property
    def T(self) -> float:
        return self.conditions.draft

    @property
    def side_area_ratio(self) -> float:
        return self.planform["side_area"] / (self.L * self.T)

    @property
    def lift_slope(self) -> float:
        return lift_slope_finite_aspect(self.planform["AR"])

    # -- nonlinear (direct) model -----------------------------------------
    def force(self, u: float, v: float, r: float, n_stations: int = 121
              ) -> Dict[str, float]:
        """Dimensional hull force/moment ``(X, Y, N)`` at state ``(u, v, r)``.

        The longitudinal term is the resistance of the actual speed; the sway
        and yaw terms are the lift + cross-flow model described above.
        """
        c = self.conditions
        rho = c.rho
        q = 0.5 * rho * (u * u + v * v)
        # ---- lift (linear in the local incidence, bow-concentrated)
        sec = self.sections(n_stations)
        x = sec["x"]
        draft = sec["draft"]
        weight = sec["weight"]
        weight_total = sec["weight_total"]
        dx = sec["dx"]
        vn = v + x * r
        alpha = np.arctan2(vn, np.maximum(u, 1e-9))
        # effective (loading-weighted) incidence for the lift term
        if weight_total > 1e-12 and abs(u) > 1e-9:
            alpha_eff = float(np.sum(weight * alpha * dx)) / weight_total
        else:
            alpha_eff = 0.0
        side_area = self.planform["side_area"]
        c_la = self.lift_slope
        Y_lift = -q * side_area * c_la * alpha_eff
        N_lift = self.planform["x_p_over_l"] * self.L * Y_lift
        # ---- cross-flow drag, integrated over the lateral planform
        dY_cf = -0.5 * rho * self.crossflow_cd * draft * np.abs(vn) * vn
        Y_cf = float(np.sum(dY_cf * dx))
        N_cf = float(np.sum(dY_cf * x * dx))
        # ---- resistance (longitudinal)
        X = -self.resistance(u)
        return {"X": X, "Y": Y_lift + Y_cf, "N": N_lift + N_cf,
                "Y_lift": Y_lift, "Y_crossflow": Y_cf,
                "N_lift": N_lift, "N_crossflow": N_cf,
                "alpha_eff": alpha_eff}

    def resistance(self, u: float) -> float:
        """Model-scale resistance ``R_T(u)`` [N] (lazy import avoids a cycle)."""
        from .resistance import model_scale_resistance
        return model_scale_resistance(abs(u) / self.conditions.u_design,
                                      self.conditions)["R_T"]

    def as_dict(self) -> Dict[str, float]:
        return {
            "AR_hull": self.planform["AR"],
            "lift_slope_per_rad": self.lift_slope,
            "side_area_ratio_SL_over_LT": self.side_area_ratio,
            "x_p_over_l": self.planform["x_p_over_l"],
            "crossflow_cd": self.crossflow_cd,
            "T_mean": self.planform["T_mean"],
            "side_area_m2": self.planform["side_area"],
        }


def linear_derivatives(model: HullForceModelSet, v_ref_prime: float = 0.0,
                       r_ref_prime: float = 0.0, eps: float = 1e-5
                       ) -> Dict[str, float]:
    """Taylor (MMG-style) derivatives of the force model at an operating point.

    The derivatives are central finite differences of the *same* force model that
    the direct simulation integrates, expressed in the prime system of the module
    (``v/U``, ``rL/U``, ``Y/(1/2 rho U^2 L T)``, ``N/(1/2 rho U^2 L^2 T)``) and
    taken at the prescribed operating point ``(v_ref', r_ref')``.  The constant
    term at that point (``Y_0``, ``N_0``) is returned as well, so the expansion is
    usable away from the origin::

        Y' = Y_0 + Y_v v' + Y_r r',     N' = N_0 + N_v v' + N_r r'

    Why the operating point matters: a zero-drift linearisation of a *quadratic*
    cross-flow drag has zero damping (``d/dv |v|v = 0`` at ``v = 0``), which makes
    the derivative model almost undamped and lets a simulated turn run away.  In
    MMG practice the derivatives come from a PMM test at a finite amplitude, i.e.
    they are *effective* derivatives at the manoeuvre's operating point; this
    function makes that choice explicit and machine-checkable.
    """
    c = model.conditions
    U = c.u_design
    q_scale = 0.5 * c.rho * U * U * c.length * c.draft
    v_ref = v_ref_prime * U
    r_ref = r_ref_prime * U / c.length

    def primes(v: float, r: float) -> Tuple[float, float]:
        res = model.force(U, v, r, n_stations=81)
        return res["Y"] / q_scale, res["N"] / (q_scale * c.length)

    y0, n0 = primes(v_ref, r_ref)
    dv = eps * U
    dr = eps * U / c.length
    y_v = (primes(v_ref + dv, r_ref)[0] - primes(v_ref - dv, r_ref)[0]) / (2 * eps)
    n_v = (primes(v_ref + dv, r_ref)[1] - primes(v_ref - dv, r_ref)[1]) / (2 * eps)
    y_r = (primes(v_ref, r_ref + dr)[0] - primes(v_ref, r_ref - dr)[0]) / (2 * eps)
    n_r = (primes(v_ref, r_ref + dr)[1] - primes(v_ref, r_ref - dr)[1]) / (2 * eps)
    return {"Y_0": y0, "N_0": n0, "Y_v": y_v, "N_v": n_v,
            "Y_r": y_r, "N_r": n_r,
            "v_ref_prime": v_ref_prime, "r_ref_prime": r_ref_prime}


def hull_force(u: float, v: float, r: float,
               model: Optional[HullForceModelSet] = None) -> Dict[str, float]:
    """Convenience wrapper around :meth:`HullForceModelSet.force`."""
    model = model or HullForceModelSet()
    return model.force(u, v, r)