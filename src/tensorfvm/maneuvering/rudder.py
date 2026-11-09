"""Rudder polar and the Kose/MMG rudder normal-force law.

Two sources of the section polar
--------------------------------
1. :func:`analytic_rudder_polar` -- Helmbold finite-aspect-ratio lifting slope
   with a smooth stall saturation and a quadratic drag polar.  Analytic
   baseline; its limits (``AR -> 0``, ``AR -> inf``, ``alpha -> 0``) are asserted
   in the tests.
2. :func:`load_rudder_polar` -- a polar produced by the 2-D body-fitted RANS in
   this repository (``scripts/run_rudder_section_cfd.py``, solver
   :class:`tensorfvm.body_fitted.BodyFittedSolver`, Spalart-Allmaras).  The
   artefact records, per angle, the lift/drag coefficients, the iteration count,
   the steady residuals and the mesh level, so the numbers in the report are
   traceable to a solver run.

The **rudder force derivative** that the node asks for is

    partial Y_R' / partial delta  = -(1 + a_H) (A_R/(L T)) f_alpha (u_R/u)^2
    partial N_R' / partial delta  = -(x_R' + a_H x_H') (1 + a_H)(A_R/(L T)) f_alpha (u_R/u)^2

with ``f_alpha`` the *section* (finite-span) lift slope from the polar,
``u_R`` the effective rudder inflow including propeller acceleration.  So the CFD
enters exactly where it should -- through the section lift slope -- while the
hull-rudder interaction factors stay explicitly empirical:

======================  =========================================  =============
symbol                  value                                      status
======================  =========================================  =============
``A_R/(L T)``           0.0169 (published KVLCC2 rudder area)      GEOMETRY
``a_H``                 0.19                                       EMPIRICAL
``x_H'/L``              -0.42                                      EMPIRICAL
``x_R'/L``              computed from the nominal rudder stock     GEOMETRY
``t_R``                 0.28                                       EMPIRICAL
``w_R``                 0.35 (wake at the rudder)                  EMPIRICAL
``gamma_R``             0.40 (flow straightening)                  EMPIRICAL
``kappa``               0.60 (propeller acceleration at rudder)    EMPIRICAL
======================  =========================================  =============

Those interaction numbers are the classical MMG values for full single-screw
hulls (Kose 1982; Yoshimura 1981; Fossen 2011 ch. 7 tables).  They are *not*
derived here and the report lists them as the main source of uncertainty of the
rudder-force level, while the rudder *derivative* -- the quantity that criterion
(b) addresses -- is dominated by ``f_alpha`` and the area ratio.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from .hull import (
    Kvlcc2Conditions, DEFAULT_CONDITIONS, RudderGeometry, rudder_geometry,
)
from .hullforces import lift_slope_finite_aspect

__all__ = [
    "RudderPolar", "RudderInteraction", "DEFAULT_INTERACTION",
    "analytic_rudder_polar", "load_rudder_polar", "rudder_normal_force",
    "rudder_derivatives",
]


@dataclass
class RudderPolar:
    """Section lift/drag polar of the rudder (2-D section or finite span)."""

    angle_deg: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    source: str
    meta: Dict[str, object] = field(default_factory=dict)

    def cl_at(self, angle_deg: float) -> float:
        return float(np.interp(angle_deg, self.angle_deg, self.cl))

    def cd_at(self, angle_deg: float) -> float:
        return float(np.interp(angle_deg, self.angle_deg, self.cd))

    def lift_slope(self, window_deg: float = 5.0) -> float:
        """``dC_L/dalpha`` [1/rad] from a central fit around zero incidence."""
        mask = np.abs(self.angle_deg) <= window_deg
        if mask.sum() < 2 or not np.any(self.angle_deg == 0.0):
            a = self.angle_deg
            c = self.cl
            slope, _ = np.polyfit(np.radians(a), c, 1)
            return float(slope)
        a = np.radians(self.angle_deg[mask])
        c = self.cl[mask]
        slope, _ = np.polyfit(a, c, 1)
        return float(slope)

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "angle_deg": [float(v) for v in self.angle_deg],
            "cl": [float(v) for v in self.cl],
            "cd": [float(v) for v in self.cd],
            "lift_slope_per_rad": self.lift_slope(),
            "meta": self.meta,
        }


def analytic_rudder_polar(geometry: Optional[RudderGeometry] = None,
                          angles_deg: Sequence[float] = (-35, -20, -10, -5, 0,
                                                         5, 10, 20, 35),
                          cl_max: float = 1.10,
                          alpha_stall_deg: float = 16.0) -> RudderPolar:
    """Finite-aspect lifting-surface polar (Helmbold) with smooth stall."""
    geom = geometry or rudder_geometry()
    a3d = lift_slope_finite_aspect(geom.aspect_ratio)
    alpha = np.radians(np.asarray(angles_deg, dtype=float))
    cl = cl_max * np.tanh(a3d * alpha / cl_max)
    cd0 = 0.008 + 0.020 * geom.thickness_ratio
    cd = cd0 + cl ** 2 / (math.pi * 0.9 * geom.aspect_ratio)
    a_stall = math.radians(alpha_stall_deg)
    beyond = np.abs(alpha) > a_stall
    if np.any(beyond):
        cd = cd.copy()
        cd[beyond] += 1.2 * np.sin(np.abs(alpha[beyond]) - a_stall) ** 2
    return RudderPolar(angle_deg=np.asarray(angles_deg, dtype=float), cl=cl, cd=cd,
                       source="analytic (Helmbold finite-AR + smooth stall)",
                       meta={"aspect_ratio": geom.aspect_ratio,
                             "lift_slope_per_rad": a3d, "cl_max": cl_max,
                             "thickness_ratio": geom.thickness_ratio})


def load_rudder_polar(path: str | Path) -> RudderPolar:
    """Load a polar artefact produced by the 2-D RANS campaign."""
    data = json.loads(Path(path).read_text())
    angles = np.asarray(data["angle_deg"], dtype=float)
    cl = np.asarray(data["cl"], dtype=float)
    cd = np.asarray(data["cd"], dtype=float)
    order = np.argsort(angles)
    return RudderPolar(angle_deg=angles[order], cl=cl[order], cd=cd[order],
                       source=str(data.get("source", "2-D RANS artefact")),
                       meta=data.get("meta", {}))


@dataclass(frozen=True)
class RudderInteraction:
    """Empirical hull-rudder-propeller interaction coefficients (MMG)."""

    a_h: float = 0.19
    x_h_over_l: float = -0.42
    t_r: float = 0.28
    w_r: float = 0.35
    gamma_r: float = 0.40
    kappa: float = 0.60
    rho_ratio: float = 1.0

    def as_dict(self) -> Dict[str, float]:
        return {"a_H": self.a_h, "x_H_over_l": self.x_h_over_l,
                "t_R": self.t_r, "w_R": self.w_r, "gamma_R": self.gamma_r,
                "kappa": self.kappa}


DEFAULT_INTERACTION = RudderInteraction()


def _effective_inflow(u: float, v: float, r: float,
                      geometry: RudderGeometry,
                      interaction: RudderInteraction,
                      KT: float, J: float,
                      conditions: Kvlcc2Conditions) -> Dict[str, float]:
    """Rudder inflow ``(u_R, v_R, alpha_R)`` including propeller acceleration.

    The slipstream is modelled with the standard MMG form

    ``u_R = u (1-w_R) sqrt( eta [1 + kappa (sqrt(1+C_T)-1)]^2 + (1-eta) )``,
    ``eta = (D/h_R)^2``      (fraction of the rudder span inside the slipstream)

    with ``C_T = 8 K_T/(pi J^2)`` the propeller thrust loading.  The contraction
    factor ``eta`` matters: dropping it (an earlier version of this function)
    multiplies the rudder inflow by up to ~3x when the ship slows in a turn and
    produces a spurious rudder drag that stops the ship -- a documented trap.
    """
    thrust_loading = 8.0 * KT / (math.pi * J * J) if J > 1e-9 and KT > 0 else 0.0
    eta = geometry.slipstream_fraction
    accel_sq = (eta * (1.0 + interaction.kappa
                       * (math.sqrt(1.0 + max(thrust_loading, 0.0)) - 1.0)) ** 2
                + (1.0 - eta))
    accel = math.sqrt(accel_sq)
    u_r = max((1.0 - interaction.w_r) * abs(u) * accel, 1e-6)
    x_r = geometry.stock_x_from_midship
    v_r = -interaction.gamma_r * (v + x_r * r)
    return {"u_R": u_r, "v_R": v_r, "accel": accel,
            "alpha_R_flow": math.atan2(v_r, u_r)}


def rudder_normal_force(delta_rad: float, u: float, v: float, r: float,
                        polar: Optional[RudderPolar] = None,
                        geometry: Optional[RudderGeometry] = None,
                        interaction: RudderInteraction = DEFAULT_INTERACTION,
                        KT: float = 0.0, J: float = 0.0,
                        conditions: Optional[Kvlcc2Conditions] = None,
                        linear: bool = False) -> Dict[str, float]:
    """Kose (1982) rudder normal force and its body-axis components.

    ``linear = False`` uses the polar (``C_L(alpha_R)`` from CFD/analytic), i.e.
    the *direct* model; ``linear = True`` uses ``f_alpha * sin(alpha_R)`` with the
    polar's slope, i.e. the *MMG* form.  The difference between the two is one of
    the two effects the MMG/CFD arbitration isolates.
    """
    c = conditions or DEFAULT_CONDITIONS
    geom = geometry or rudder_geometry()
    pol = polar or analytic_rudder_polar(geom)
    inflow = _effective_inflow(u, v, r, geom, interaction, KT, J, c)
    alpha_r = delta_rad - inflow["alpha_R_flow"]
    u_r = inflow["u_R"]
    q_r = 0.5 * c.rho * u_r * u_r * geom.area
    if linear:
        f_alpha = pol.lift_slope()
        normal = q_r * f_alpha * math.sin(alpha_r)
        cl_used = f_alpha * math.sin(alpha_r)
        cd_used = float("nan")
    else:
        cl_used = pol.cl_at(math.degrees(alpha_r))
        cd_used = pol.cd_at(math.degrees(alpha_r))
        normal = q_r * (cl_used * math.cos(math.radians(0.0))
                        * math.sin(alpha_r) + 0.0) if False else q_r * cl_used
        # normal-force form: F_N = 1/2 rho A_R U_R^2 C_L(alpha_R)
        normal = q_r * cl_used
    # chordwise force (rudder drag along the ship axis) -- kept for the surge eq.
    chord = q_r * (cd_used if not linear else 0.0)
    cos_d, sin_d = math.cos(delta_rad), math.sin(delta_rad)
    f_x = -normal * sin_d - chord * cos_d
    f_y = -normal * cos_d + chord * sin_d
    m_z = geometry.stock_x_from_midship * f_y - 0.0 * f_x
    return {"F_N": normal, "F_x": f_x, "F_y": f_y, "M_z": m_z,
            "alpha_R_rad": alpha_r, "alpha_R_deg": math.degrees(alpha_r),
            "u_R": u_r, "v_R": inflow["v_R"], "cl": cl_used, "cd": cd_used,
            "thrust_loading": inflow["accel"]}


def rudder_derivatives(polar: Optional[RudderPolar] = None,
                       geometry: Optional[RudderGeometry] = None,
                       interaction: RudderInteraction = DEFAULT_INTERACTION,
                       conditions: Optional[Kvlcc2Conditions] = None,
                       KT: float = 0.0, J: float = 0.0,
                       delta_rad: float = 0.0) -> Dict[str, float]:
    """Rudder force/moment derivatives in the prime system.

    ``F_y_delta_prime`` is ``partial Y_R'/partial delta`` [per rad] and
    ``M_z_delta_prime`` is ``partial N_R'/partial delta`` [per rad], with
    ``Y' = Y/(1/2 rho U^2 L T)`` and ``N' = N/(1/2 rho U^2 L^2 T)``.
    Also returned per degree, because rudder derivatives are usually quoted that
    way in model-test reports.
    """
    c = conditions or DEFAULT_CONDITIONS
    geom = geometry or rudder_geometry()
    pol = polar or analytic_rudder_polar(geom)
    inflow = _effective_inflow(c.u_design, 0.0, 0.0, geom, interaction, KT, J, c)
    f_alpha = pol.lift_slope()
    speed_ratio = (inflow["u_R"] / c.u_design) ** 2
    area_ratio = geom.area
    f_n_delta = (area_ratio / (c.length * c.draft)) * f_alpha * speed_ratio
    y_delta = -(1.0 + interaction.a_h) * f_n_delta * math.cos(delta_rad)
    n_delta = -((geom.stock_x_from_midship / c.length)
                + interaction.a_h * interaction.x_h_over_l) * (1.0 + interaction.a_h) * f_n_delta * math.cos(delta_rad)
    return {
        "F_N_delta_prime_per_rad": f_n_delta,
        "F_y_delta_prime_per_rad": y_delta,
        "F_y_delta_prime_per_deg": y_delta * math.pi / 180.0,
        "M_z_delta_prime_per_rad": n_delta,
        "M_z_delta_prime_per_deg": n_delta * math.pi / 180.0,
        "section_lift_slope_per_rad": f_alpha,
        "rudder_inflow_speed_ratio_squared": speed_ratio,
        "area_ratio": area_ratio / (c.length * c.draft),
        "polar_source": pol.source,
        "interaction": interaction.as_dict(),
    }