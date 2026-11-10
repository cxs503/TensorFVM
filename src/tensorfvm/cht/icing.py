"""Marine-spray icing surface energy balance (W4-N08-consistent closure).

This module is the *coupling surface* between the TensorFVM conjugate-heat-
transfer solver and the icing model of the ship platform (campaign node W4-N08,
``backend/api/marine_icing.py``, API ``calculations/marine-icing-stability``).
It re-implements the five-term surface heat balance independently -- from the
physical correlations, not by importing the platform code -- so that the two
implementations can adjudicate each other:

    q_conv  = h_c (T_s - T_a)                          forced + natural convection
    q_evap  = h_m (rho_v,s - rho_v,a) L_phase           Chilton-Colburn analogy
    q_lw    = eps [sigma T_s^4 - L_down]                Idso-Jackson clear sky
    q_spray = W c_pw (T_sea - T_s)                      spray sensible heat
    q_freeze= f W L_f                                   latent heat of freezing

with ``Q_req(T_s) = q_conv + q_evap + q_lw - q_spray`` the heat the surface must
lose.  The wet/dry growing-regime logic is the W4-N08 one:

* ``Q_req(T_f) <= 0``                 -> no ice (surface warmer than freezing)
* ``0 < Q_req(T_f) < W L_f``          -> **wet**  growth: the surface is pinned at
  the freezing temperature and only a fraction ``f = Q_req/(W L_f)`` freezes
* ``Q_req(T_f) >= W L_f``             -> **dry** growth: everything freezes,
  ``f = 1`` and the surface temperature solves ``Q_req(T_s) = W L_f``

The one place where this module deliberately goes *beyond* W4-N08 is the
substrate: W4-N08 assumes no heat conduction into the structure (a pure surface
balance), while :func:`icing_state_with_conduction` puts the CHT wall resistance
in series and lets that conduction change the regime/freezing fraction.  The
difference between the two is reported as the model discrepancy (it is not a
numerical error).

Reference data for the cross-check lives in
``src/tensorfvm/data/marine_icing_w4n08/reference_panel.json`` and is loaded --
never transcribed -- by :func:`load_w4n08_reference`.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "IcingEnvironment",
    "IcingState",
    "conduction_resistance",
    "convection_coefficient",
    "icing_state_adiabatic",
    "icing_state_with_conduction",
    "load_w4n08_reference",
    "sky_longwave_down",
    "surface_fluxes",
]

SIGMA = 5.670374419e-8
L_F = 3.337e5            # J/kg   freezing latent heat (W4-N08 value)
CP_WATER = 4000.0        # J/(kg K)
CP_AIR = 1005.0
R_DRY = 287.05
R_VAP = 461.5
P_ATM = 101325.0
PR_AIR = 0.71
SC_AIR = 0.60
EMISSIVITY_ICE = 0.97
H_NATURAL = 5.6

DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "marine_icing_w4n08" / \
    "reference_panel.json"


@dataclass(frozen=True)
class IcingEnvironment:
    t_air: float = -10.0          # degC
    t_sea: float = -1.8           # degC
    t_freeze: float = -1.8        # degC (sea-water freezing point)
    wind_speed: float = 10.0      # m/s
    char_length: float = 2.0      # m
    relative_humidity: float = 0.8
    cloud_cover: float = 0.0
    spray_flux: float = 2.5356079670000003e-03   # kg/(m^2 s)
    exposure: float = 6.0 * 3600.0               # s


@dataclass
class IcingState:
    regime: str
    t_surface: float
    freezing_fraction: float
    q_conv: float
    q_evap: float
    q_lw: float
    q_spray: float
    q_req: float
    q_cond: float
    ice_flux: float          # kg/(m^2 s)
    icing_rate_mm_per_h: float
    h_c: float
    terms_balance: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


# --------------------------------------------------------------- correlations
def saturation_vapour_pressure(t_c: float, over_ice: bool = False) -> float:
    if over_ice:
        return 100.0 * 6.112 * math.exp(22.46 * t_c / (272.62 + t_c))
    return 100.0 * 6.112 * math.exp(17.62 * t_c / (243.12 + t_c))


def air_properties(t_c: float):
    t_k = t_c + 273.15
    rho = P_ATM / (R_DRY * t_k)
    mu = 1.458e-6 * t_k ** 1.5 / (t_k + 110.4)
    return rho, mu / rho, 0.02424 * (t_k / 273.15) ** 0.9


def convection_coefficient(env: IcingEnvironment) -> float:
    """Flat-plate forced convection (laminar / turbulent) + natural convection."""
    _, nu, k = air_properties(env.t_air)
    length = max(0.05, env.char_length)
    re = max(0.0, env.wind_speed) * length / nu
    if re < 1.0:
        h_forced = 0.0
    elif re < 5.0e5:
        h_forced = 0.664 * math.sqrt(re) * PR_AIR ** (1.0 / 3.0) * k / length
    else:
        h_forced = max(0.0, (0.037 * re ** 0.8 - 871.0)
                       * PR_AIR ** (1.0 / 3.0)) * k / length
    return (h_forced ** 3 + H_NATURAL ** 3) ** (1.0 / 3.0)


def sky_longwave_down(t_air_c: float, cloud_cover: float = 0.0) -> float:
    t_k = t_air_c + 273.15
    eps_clear = 1.0 - 0.261 * math.exp(-7.77e-4 * t_air_c ** 2)
    eps_sky = eps_clear * (1.0 + 0.26 * max(0.0, min(1.0, cloud_cover)))
    return eps_sky * SIGMA * t_k ** 4


def surface_fluxes(t_s: float, env: IcingEnvironment, wet: bool = True) -> dict:
    """Five surface heat fluxes [W/m^2], positive = heat leaving the surface."""
    rho_a = air_properties(env.t_air)[0]
    h_c = convection_coefficient(env)
    q_conv = h_c * (t_s - env.t_air)

    le = SC_AIR / PR_AIR
    h_m = h_c / (rho_a * CP_AIR) * le ** (-2.0 / 3.0)
    e_sat_s = saturation_vapour_pressure(t_s, over_ice=not wet)
    e_sat_a = saturation_vapour_pressure(env.t_air, over_ice=True)
    rho_v_s = e_sat_s / (R_VAP * (t_s + 273.15))
    rho_v_a = env.relative_humidity * e_sat_a / (R_VAP * (env.t_air + 273.15))
    l_phase = (2.501e6 - 2370.0 * t_s) if wet else (2.501e6 - 2370.0 * t_s + L_F)
    q_evap = h_m * (rho_v_s - rho_v_a) * l_phase

    q_lw = EMISSIVITY_ICE * (SIGMA * (t_s + 273.15) ** 4
                             - sky_longwave_down(env.t_air, env.cloud_cover))
    q_spray = env.spray_flux * CP_WATER * (env.t_sea - t_s)
    return {
        "h_c": h_c,
        "q_conv": q_conv,
        "q_evap": q_evap,
        "q_lw": q_lw,
        "q_spray": q_spray,
        "q_req": q_conv + q_evap + q_lw - q_spray,
    }


def _bisect(f, lo: float, hi: float, tol: float = 1.0e-10, max_iter: int = 200) -> float:
    f_lo = f(lo)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        if abs(hi - lo) < tol:
            return mid
        if (f_mid > 0.0) == (f_lo > 0.0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _mm_per_hour(ice_flux: float, rho_ice: float = 890.0) -> float:
    return ice_flux / rho_ice * 1.0e3 * 3600.0


def icing_state_adiabatic(env: IcingEnvironment, rho_ice: float = 890.0) -> IcingState:
    """W4-N08 wet/dry closure: no conduction into the structure."""
    tf = env.t_freeze
    wet_terms = surface_fluxes(tf, env, wet=True)
    q_req = wet_terms["q_req"]
    capacity = env.spray_flux * L_F
    if q_req <= 0.0:
        state = IcingState("none", tf, 0.0, wet_terms["q_conv"], wet_terms["q_evap"],
                           wet_terms["q_lw"], wet_terms["q_spray"], q_req, 0.0,
                           0.0, 0.0, wet_terms["h_c"], 0.0)
        return state
    if q_req < capacity:
        f = q_req / capacity
        ice_flux = f * env.spray_flux
        return IcingState("wet", tf, f, wet_terms["q_conv"], wet_terms["q_evap"],
                          wet_terms["q_lw"], wet_terms["q_spray"], q_req, 0.0,
                          ice_flux, _mm_per_hour(ice_flux, rho_ice),
                          wet_terms["h_c"], q_req - capacity * f)
    # dry: solve Q_req(T_s) = W L_f for T_s < T_f
    def balance(t_s: float) -> float:
        terms = surface_fluxes(t_s, env, wet=False)
        return terms["q_req"] - capacity

    t_s = _bisect(balance, -80.0, tf)
    terms = surface_fluxes(t_s, env, wet=False)
    ice_flux = env.spray_flux
    return IcingState("dry", t_s, 1.0, terms["q_conv"], terms["q_evap"],
                      terms["q_lw"], terms["q_spray"], terms["q_req"], 0.0,
                      ice_flux, _mm_per_hour(ice_flux, rho_ice), terms["h_c"],
                      balance(t_s))


def conduction_resistance(thicknesses, conductivities, r_contact: float = 0.0) -> float:
    """1-D series resistance [m^2 K/W] of the wall (surface to the back face)."""
    return sum(t / k for t, k in zip(thicknesses, conductivities)) + r_contact


def icing_state_with_conduction(env: IcingEnvironment, r_total: float,
                                t_back: float, rho_ice: float = 890.0) -> IcingState:
    """Wet/dry closure with the CHT wall resistance ``r_total`` in series.

    ``t_back`` is the temperature of the back face of the structure.  Heat flows
    *down* for ``t_s > t_back``; the conducted flux is charged to the surface
    balance, which is what changes ``f`` (and may flip wet -> dry).
    """
    tf = env.t_freeze
    capacity = env.spray_flux * L_F
    wet_terms = surface_fluxes(tf, env, wet=True)
    q_cond_wet = (tf - t_back) / r_total if r_total > 0 else 0.0
    q_req_wet = wet_terms["q_req"] + q_cond_wet
    if q_req_wet <= 0.0:
        return IcingState("none", tf, 0.0, wet_terms["q_conv"], wet_terms["q_evap"],
                          wet_terms["q_lw"], wet_terms["q_spray"], wet_terms["q_req"],
                          q_cond_wet, 0.0, 0.0, wet_terms["h_c"],
                          q_req_wet)
    if q_req_wet < capacity:
        f = q_req_wet / capacity
        ice_flux = f * env.spray_flux
        return IcingState("wet", tf, f, wet_terms["q_conv"], wet_terms["q_evap"],
                          wet_terms["q_lw"], wet_terms["q_spray"], wet_terms["q_req"],
                          q_cond_wet, ice_flux, _mm_per_hour(ice_flux, rho_ice),
                          wet_terms["h_c"], q_req_wet - capacity * f)

    def balance(t_s: float) -> float:
        terms = surface_fluxes(t_s, env, wet=False)
        q_cond = (t_s - t_back) / r_total if r_total > 0 else 0.0
        return terms["q_req"] + q_cond - capacity

    lo = min(t_back, env.t_air) - 40.0
    t_s = _bisect(balance, lo, tf)
    terms = surface_fluxes(t_s, env, wet=False)
    q_cond = (t_s - t_back) / r_total if r_total > 0 else 0.0
    ice_flux = env.spray_flux
    return IcingState("dry", t_s, 1.0, terms["q_conv"], terms["q_evap"],
                      terms["q_lw"], terms["q_spray"], terms["q_req"], q_cond,
                      ice_flux, _mm_per_hour(ice_flux, rho_ice), terms["h_c"],
                      balance(t_s))


def load_w4n08_reference(path: Path | str | None = None) -> dict:
    """Load the machine-readable W4-N08 reference-panel data (with provenance)."""
    path = Path(path) if path is not None else DATA_FILE
    payload = json.loads(path.read_text())
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    payload["_loaded_from"] = str(path)
    payload["_sha256"] = sha
    return payload