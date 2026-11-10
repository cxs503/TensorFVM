"""Cavitation number, water properties and inception criteria for rudder sections.

The chain implemented here
--------------------------
1. **Cavitation number** (the operating parameter of the flow, not of the section)

       sigma = (p_inf - p_v) / (1/2 rho V^2).                             (1)

2. **Inception cavitation number** of a section at incidence ``alpha``.  A vapour
   cavity can start only where the local pressure reaches the vapour pressure,
   and along a streamline Bernoulli's equation gives
   ``p_local = p_inf + Cp q`` with ``q = 1/2 rho V^2``.  Hence

       p_min = p_v   <=>   sigma = -Cp_min(alpha) =: sigma_i(alpha)       (2)

   which is the *minimum-pressure criterion* used by the ITTC cavitation
   inception procedures.  ``sigma_i`` is a property of the section shape and the
   incidence only; the vapour pressure drops out of (2) because it enters the
   cavitation number (1) and the criterion (2) symmetrically.

3. **Water-quality (nucleus) correction.**  Real water needs a finite tension
   before a nucleus grows explosively.  Using the Rayleigh-Plesset equilibrium of
   a spherical gas nucleus of radius ``R0`` with surface tension ``S``, the
   critical (Blake) pressure is

       P(R) = p_g0 (R0/R)^3 - 2S/R,   dP/dR = 0  =>  R_c = sqrt(3 p_g0 R0^3/(2S))
       P_c  = P(R_c) = -4S/(3 R_c) < 0,   p_g0 = p_0 - p_v + 2S/R0            (3)

   and cavitation starts when ``p_inf - p_v < P_c`` (i.e. when the liquid is
   pulled below the Blake threshold).  The inception index therefore becomes

       sigma_i*(alpha, R0) = -Cp_min(alpha) + |P_c| / q .                 (4)

   ``sigma_i*`` is the tension the flow must supply; ``|P_c| ~ 4S/(3 sqrt(3) R0)``
   diverges as ``R0 -> 0``, so *nuclei-free* water cannot be cavitated at any
   finite cavitation number -- see :func:`sigma_inception_nuclei_free` and the
   negative-control discussion in ``docs/cavitation-rudder-erosion.md``.

Physical properties
-------------------
Vapour pressure from the Antoine correlation for water (valid 1-100 degC),

    log10(p_v / mmHg) = 8.07131 - 1730.63 / (233.426 + T[degC]),

which reproduces 2339 Pa at 20 degC and 1705 Pa at 15 degC to 0.3 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .section import naca4_contour, panel_solve

__all__ = [
    "water_vapour_pressure", "water_density", "WaterState",
    "cavitation_number", "CavitationState",
    "sigma_inception", "sigma_inception_curve", "dsigma_inception_dalpha",
    "blake_critical_pressure", "sigma_inception_with_nuclei",
    "sigma_inception_nuclei_free", "inception_flag", "inception_margin",
]


# --------------------------------------------------------------------------- #
# water properties
# --------------------------------------------------------------------------- #
def water_vapour_pressure(temperature_c: float) -> float:
    """Saturation vapour pressure of water [Pa] (Antoine, 1-100 degC)."""
    if not -1.0 < temperature_c < 100.0:
        raise ValueError("Antoine fit valid for 0 < T[degC] < 100")
    log10_mmhg = 8.07131 - 1730.63 / (233.426 + temperature_c)
    return float(10.0 ** log10_mmhg * 133.322387415)


def water_density(temperature_c: float, salinity: float = 0.0) -> float:
    """Density of water [kg/m^3]; ``salinity`` in g/kg (0 = fresh, 35 = sea)."""
    t = temperature_c
    fresh = 1000.0 * (1.0 - (t + 288.9414) / (508929.2 * (t + 68.12963))
                      * (t - 3.9863) ** 2)
    return float(fresh + 0.8 * salinity)


@dataclass(frozen=True)
class WaterState:
    """Thermophysical state of the working fluid at the rudder inflow."""

    temperature_c: float = 15.0
    salinity_g_per_kg: float = 35.0
    surface_tension: float = 0.0735       # N/m, water/air at 15 degC
    sound_speed: float = 1470.0           # m/s, used for the water-hammer impact

    @property
    def rho(self) -> float:
        return water_density(self.temperature_c, self.salinity_g_per_kg)

    @property
    def p_v(self) -> float:
        return water_vapour_pressure(self.temperature_c)


# --------------------------------------------------------------------------- #
# operating point and inception
# --------------------------------------------------------------------------- #
def cavitation_number(p_inf: float, p_v: float, rho: float, v: float) -> float:
    """``sigma = (p_inf - p_v) / (1/2 rho V^2)``.

    ``p_inf`` is the *local* static pressure at the section (i.e. atmospheric
    plus hydrostatic head plus any propeller-induced suction), not the tank
    pressure: the rudder of a ship sits in a pressure field that the propeller
    modifies by several kPa.
    """
    if rho <= 0.0 or v < 0.0:
        raise ValueError("rho must be positive and v non-negative")
    q = 0.5 * rho * v * v
    if q == 0.0:
        return math.inf
    return float((p_inf - p_v) / q)


@dataclass(frozen=True)
class CavitationState:
    """Operating cavitation number plus the section's inception threshold."""

    sigma: float                 # operating cavitation number
    sigma_inception: float       # sigma_i(alpha) = -Cp_min(alpha)
    cp_min: float
    alpha_deg: float
    q: float                     # dynamic pressure [Pa]
    p_inf: float
    p_v: float

    @property
    def cavitating(self) -> bool:
        return self.sigma < self.sigma_inception

    @property
    def margin(self) -> float:
        """``sigma - sigma_i``: positive = cavitation free."""
        return self.sigma - self.sigma_inception

    @property
    def cavity_strength(self) -> float:
        """``sigma_i/sigma - 1``: the classical cavity-length scaling argument."""
        if self.sigma <= 0.0:
            return math.inf
        return max(0.0, self.sigma_inception / self.sigma - 1.0)


def sigma_inception(alpha_deg: float, thickness_ratio: float = 0.18,
                    n_half: int = 200, camber: float = 0.0,
                    camber_position: float = 0.4) -> float:
    """``sigma_i = -Cp_min`` of a NACA 4-digit section at incidence [deg]."""
    nodes = naca4_contour(t=thickness_ratio, m=camber, p=camber_position,
                          n_half=n_half)
    return float(-panel_solve(nodes, alpha_deg).cp_min)


def sigma_inception_curve(alphas_deg, thickness_ratio: float = 0.18,
                          n_half: int = 200, camber: float = 0.0) -> np.ndarray:
    """``sigma_i(alpha)`` on a grid of incidence angles."""
    return np.array([sigma_inception(a, thickness_ratio=thickness_ratio,
                                     n_half=n_half, camber=camber)
                     for a in np.asarray(alphas_deg, dtype=float)])


def dsigma_inception_dalpha(alpha_deg: float, thickness_ratio: float = 0.18,
                            n_half: int = 200, delta_deg: float = 0.25) -> float:
    """Central difference ``d sigma_i / d alpha`` [1/deg]."""
    plus = sigma_inception(alpha_deg + delta_deg, thickness_ratio, n_half)
    minus = sigma_inception(alpha_deg - delta_deg, thickness_ratio, n_half)
    return float((plus - minus) / (2.0 * delta_deg))


# --------------------------------------------------------------------------- #
# nuclei / tensile-strength correction (Blake threshold)
# --------------------------------------------------------------------------- #
def blake_critical_pressure(r0: float, surface_tension: float, p0: float,
                            p_v: float) -> float:
    """Blake critical pressure ``P_c`` [Pa] of a gas nucleus in equilibrium.

    ``p0`` is the initial liquid pressure at which the nucleus of radius ``r0``
    is in equilibrium.  A negative ``P_c`` is the tension (``p_inf - p_v`` must
    fall below it) needed to make the nucleus grow without bound.
    """
    if r0 <= 0.0:
        raise ValueError("nucleus radius must be positive")
    p_g0 = p0 - p_v + 2.0 * surface_tension / r0     # isothermal gas pressure
    if p_g0 <= 0.0:
        raise ValueError("nucleus would not be in equilibrium: p_g0 <= 0")
    r_c = math.sqrt(3.0 * p_g0 * r0 ** 3 / (2.0 * surface_tension))
    return float(p_g0 * r0 ** 3 / r_c ** 3 - 2.0 * surface_tension / r_c)


def sigma_inception_with_nuclei(alpha_deg: float, r0: float, water: WaterState,
                                p_inf: float, v: float, thickness_ratio: float = 0.18,
                                n_half: int = 200) -> float:
    """Inception index ``sigma_i* = -Cp_min + |P_c|/q`` including nuclei."""
    q = 0.5 * water.rho * v * v
    cp_min = -sigma_inception(alpha_deg, thickness_ratio, n_half)
    p_c = blake_critical_pressure(r0, water.surface_tension, p_inf, water.p_v)
    return float(-cp_min + abs(p_c) / q)


def sigma_inception_nuclei_free(alpha_deg: float, r0: float, water: WaterState,
                                p_inf: float, v: float, thickness_ratio: float = 0.18,
                                n_half: int = 200) -> float:
    """Inception index in the limit of a vanishing nucleus population.

    Same expression as :func:`sigma_inception_with_nuclei`; the point of the
    separate entry point is the *limit*, ``r0 -> 0``: ``|P_c| ~ 4S/(3 sqrt(3) r0)``
    so ``sigma_i* -> inf``, i.e. pure water cannot be cavitated at any finite
    cavitation number.
    """
    return sigma_inception_with_nuclei(alpha_deg, r0, water, p_inf, v,
                                       thickness_ratio, n_half)


def inception_flag(sigma: float, sigma_i: float, tol: float = 0.0) -> bool:
    """``True`` when the operating point is inside the inception threshold."""
    return bool(sigma < sigma_i + tol)


def inception_margin(state: CavitationState) -> float:
    """Relative margin ``(sigma - sigma_i)/sigma_i`` (negative = cavitating)."""
    if state.sigma_inception == 0.0:
        return math.inf
    return float((state.sigma - state.sigma_inception) / state.sigma_inception)


def build_state(alpha_deg: float, p_inf: float, v: float, water: WaterState,
                thickness_ratio: float = 0.18, n_half: int = 200,
                r0: Optional[float] = None) -> CavitationState:
    """Assemble the operating point + inception threshold for one condition."""
    rho, p_v = water.rho, water.p_v
    q = 0.5 * rho * v * v
    sigma = cavitation_number(p_inf, p_v, rho, v)
    sigma_i = sigma_inception(alpha_deg, thickness_ratio=thickness_ratio,
                              n_half=n_half)
    cp_min = -sigma_i
    if r0 is not None:
        sigma_i = sigma_inception_with_nuclei(alpha_deg, r0, water, p_inf, v,
                                              thickness_ratio, n_half)
    return CavitationState(sigma=sigma, sigma_inception=sigma_i, cp_min=cp_min,
                           alpha_deg=alpha_deg, q=q, p_inf=p_inf, p_v=p_v)