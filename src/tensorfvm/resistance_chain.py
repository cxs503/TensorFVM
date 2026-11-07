"""Unified ship viscous-resistance chain: friction lines, form factor and the
two competing correlation-allowance conventions (ITTC-78 roughness vs Holtrop-1984).

Why this module exists
----------------------
A total-resistance prediction is only as trustworthy as the *convention* it is
computed in.  In the ship-performance platform this chain historically contained
three independent defects:

1. two different correlation allowances were added to the same prediction
   (ITTC-78 ``dCf`` *and* Holtrop's ``Ca``), i.e. the roughness/form correction
   was counted twice;
2. the form factor ``(1+k)`` had no independent (CFD) source, so it could not be
   adjudicated against the regression value it was supposed to corroborate;
3. friction was quoted per law without stating which law the rest of the chain
   assumed, so the model/ship extrapolation could silently mix ITTC-57 and
   Schoenherr.

This module turns those three into explicit, machine-checkable interfaces:

* :func:`ittc57_cf`, :func:`schoenherr_cf`, :func:`prandtl_schlichting_cf`
  -- three independent average-friction laws, with the literature spread
  available as an irreducible reference uncertainty (:func:`friction_band`);
* :func:`ittc78_delta_cf` -- the ITTC-78 roughness/correlation allowance
  ``dCf = [105 (ks/L)^(1/3) - 0.64] * 1e-3`` (floored at zero);
* :func:`holtrop_form_factor`, :func:`holtrop_wetted_surface`,
  :func:`holtrop_correlation_allowance` -- the Holtrop-Mennen (1982) /
  Holtrop (1984) regression quantities, ported term-by-term so the CFD form
  factor of :func:`form_factor_from_double_body` can be adjudicated against it;
* :func:`viscous_chain` -- the unified chain ``C_T = (1+k) C_F + allowance``
  where the allowance regime is *chosen*, never accumulated: requesting both
  conventions raises instead of silently double counting;
* :func:`validation_uncertainty_budget` -- the ITTC 7.5-02-02-02 style budget
  whose ``U_V`` is the quantity gated at 5 % in the resistance-chain mission;
* :func:`extrapolate_to_ship` -- the 1978-performance-committee model/ship
  extrapolation, the place where a mixed convention does the most damage.

All correlations here are closed-form and take no solver output, so they can be
used to audit solvers without circularity.  Constants match the platform's
single source (``backend/physics/holtrop/mennen.py``): ``g = 9.80665``,
``rho = 1025 kg/m^3``, ``nu = 1.19e-6 m^2/s`` (15 degC sea water),
``1 kn = 0.5144444 m/s``.  A tank-test freshwater viscosity is exposed
separately as :data:`NU_FRESHWATER_15C`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

__all__ = [
    "G", "RHO_SEAWATER", "NU_SEAWATER_15C", "NU_FRESHWATER_15C", "KNOT",
    "DEFAULT_SHIP_ROUGHNESS_KS", "MODEL_ROUGHNESS_KS", "LAMINAR_RE_LIMIT",
    "ALLOWANCE_REGIMES",
    "HullPrincipal",
    "ittc57_cf", "schoenherr_cf", "prandtl_schlichting_cf", "friction_band",
    "ittc78_delta_cf", "holtrop_lr", "holtrop_form_factor",
    "holtrop_wetted_surface", "holtrop_correlation_allowance",
    "form_factor_from_double_body", "compare_form_factor",
    "viscous_chain", "reynolds_table", "correlation_allowance_table",
    "validation_uncertainty_budget", "extrapolate_to_ship",
]

# --------------------------------------------------------------------------- #
# Physical constants (platform single source of truth)
# --------------------------------------------------------------------------- #
G = 9.80665                      # m/s^2
RHO_SEAWATER = 1025.0            # kg/m^3 at 15 degC
NU_SEAWATER_15C = 1.19e-6        # m^2/s at 15 degC (platform/ITTC-1957 value)
NU_FRESHWATER_15C = 1.139e-6     # m^2/s at 15 degC, towing-tank fresh water
KNOT = 0.5144444                 # m/s

#: ITTC-78 new-ship equivalent sand-grain roughness [m] (blasted + painted steel)
DEFAULT_SHIP_ROUGHNESS_KS = 150e-6
#: Smooth painted model / freshly finished hull [m]; hydraulically smooth at
#: model scale, kept for provenance of the "model" allowance regime.
MODEL_ROUGHNESS_KS = 50e-6
#: Below this Reynolds number the fully turbulent laws have no basis.
LAMINAR_RE_LIMIT = 5.0e5

#: The two correlation-allowance conventions the chain is allowed to use.
ALLOWANCE_REGIMES = ("ittc78_roughness", "holtrop_ca", "none")


# --------------------------------------------------------------------------- #
# Hull principal particulars
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HullPrincipal:
    """Principal particulars of a hull, in SI units, LCB as % of L (fwd +)."""

    name: str
    L: float            # length between perpendiculars [m]
    B: float            # moulded breadth [m]
    T: float            # design draft [m]
    Cb: float           # block coefficient
    Cm: float           # midship section coefficient
    Cwp: float          # waterplane area coefficient
    lcb_pct: float = 0.0
    Cstern: float = 0.0
    #: Transom area immersed [m^2], bulbous-bow transverse area [m^2] and its
    #: centre height [m].  Only used by the Holtrop terms that need them.
    AT: float = 0.0
    ABT: float = 0.0
    hB: float = 0.0

    def __post_init__(self) -> None:
        for name in ("L", "B", "T"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("Cb", "Cm", "Cwp"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.Cb > self.Cm + 1e-12:
            raise ValueError("Cb cannot exceed Cm (Cb = Cp * Cm with Cp <= 1)")

    @property
    def nabla(self) -> float:
        """Displaced volume Cb L B T [m^3]."""
        return self.Cb * self.L * self.B * self.T

    @property
    def Cp(self) -> float:
        """Prismatic coefficient Cb / Cm."""
        return self.Cb / self.Cm

    def offset_consistency(self) -> Dict[str, float]:
        """Dimensionless ratios used to decide whether a hull is in-regression.

        The Holtrop form factor and wetted surface are regressions on exactly
        these ratios; a hull outside them can be *computed* but the numbers are
        extrapolations, which the chain reports instead of hiding.
        """
        return {
            "L_over_B": self.L / self.B,
            "B_over_T": self.B / self.T,
            "T_over_L": self.T / self.L,
            "Cp": self.Cp,
            "nabla_over_L3": self.nabla / self.L ** 3,
            "lcb_pct": self.lcb_pct,
        }


# --------------------------------------------------------------------------- #
# Friction laws
# --------------------------------------------------------------------------- #
def _require_reynolds(re: float, minimum: float = LAMINAR_RE_LIMIT) -> float:
    if not isinstance(re, (int, float)) or isinstance(re, bool):
        raise TypeError("Reynolds number must be a real scalar")
    re = float(re)
    if not math.isfinite(re) or re <= 0.0:
        raise ValueError("Reynolds number must be finite and positive")
    if re < minimum:
        raise ValueError(
            f"fully turbulent laws require Re >= {minimum:g}, got {re:g}; "
            "use a laminar law below that"
        )
    return re


def ittc57_cf(re: float) -> float:
    """ITTC-1957 average friction line ``0.075 / (log10 Re - 2)^2``."""
    re = _require_reynolds(re)
    return 0.075 / (math.log10(re) - 2.0) ** 2


def prandtl_schlichting_cf(re: float) -> float:
    """Prandtl-Schlichting ``0.455 / (log10 Re)^2.58`` (independent law)."""
    re = _require_reynolds(re)
    return 0.455 / math.log10(re) ** 2.58


def schoenherr_cf(re: float, tol: float = 1e-15, max_iter: int = 200) -> float:
    """Schoenherr average friction line, implicit ``0.242/sqrt(Cf) = log10(Re Cf)``.

    Solved by bisection on the monotone residual, so the result is deterministic
    and independent of any starting guess.  This is a genuinely different
    historical correlation from ITTC-57, which is what makes it usable as a
    cross-check rather than a re-derivation.
    """
    re = _require_reynolds(re)

    def residual(cf: float) -> float:
        return 0.242 / math.sqrt(cf) - math.log10(re * cf)

    low, high = 1e-6, 2e-2
    f_low, f_high = residual(low), residual(high)
    if f_low * f_high > 0.0:
        raise ValueError("Schoenherr residual does not bracket a root")
    for _ in range(max_iter):
        mid = 0.5 * (low + high)
        f_mid = residual(mid)
        if abs(f_mid) < tol or 0.5 * (high - low) < tol * mid:
            return mid
        if f_mid * f_low <= 0.0:
            high = mid
        else:
            low, f_low = mid, f_mid
    return 0.5 * (low + high)


def friction_band(re: float) -> Dict[str, float]:
    """The three laws plus their disagreement, as a reference-uncertainty band."""
    values = {
        "ittc57": ittc57_cf(re),
        "schoenherr": schoenherr_cf(re),
        "prandtl_schlichting": prandtl_schlichting_cf(re),
    }
    low, high = min(values.values()), max(values.values())
    mean = sum(values.values()) / len(values)
    return {
        "reynolds": float(re),
        "values": values,
        "low": low,
        "high": high,
        "mean": mean,
        "spread": (high - low) / mean,
        "relative_low": (low - mean) / mean,
        "relative_high": (high - mean) / mean,
    }


# --------------------------------------------------------------------------- #
# Correlation / roughness allowance (ITTC-78)
# --------------------------------------------------------------------------- #
def ittc78_delta_cf(ks: float, L: float) -> float:
    """ITTC-1978 roughness allowance ``dCf = [105 (ks/L)^(1/3) - 0.64] * 1e-3``.

    Floored at zero: for a hydraulically smooth surface the formula goes
    negative, which has no physical meaning as an added resistance.
    ``ks`` is the equivalent sand-grain roughness [m], ``L`` the ship length [m].
    """
    for name, value in (("ks", ks), ("L", L)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if L <= 0.0:
        raise ValueError("L must be positive")
    if ks < 0.0:
        raise ValueError("ks must be non-negative")
    return max((105.0 * (ks / L) ** (1.0 / 3.0) - 0.64) * 1e-3, 0.0)


# --------------------------------------------------------------------------- #
# Holtrop-Mennen (1982) / Holtrop (1984) regression quantities
# --------------------------------------------------------------------------- #
def holtrop_lr(L: float, Cp: float, lcb_pct: float) -> float:
    """Length of run ``LR = L (1 - Cp + 0.06 Cp lcb / (4 Cp - 1))``."""
    denominator = 4.0 * Cp - 1.0
    if abs(denominator) < 1e-12:
        raise ValueError("Cp too close to 0.25 for the Holtrop length-of-run term")
    return L * (1.0 - Cp + 0.06 * Cp * lcb_pct / denominator)


def holtrop_form_factor(principal: HullPrincipal) -> float:
    """Holtrop-Mennen form factor ``(1+k)``.

    ``(1+k) = 0.93 + 0.487118 c14 (B/L)^1.06806 (T/L)^0.46106 (L/LR)^0.121563
    (L^3/nabla)^0.36486 (1 - Cp)^-0.604247``, with ``c14 = 1 + 0.011 Cstern``.
    This is a regression on hull *ratios and coefficients only*: it is the
    empirical branch of the two-convention adjudication, and the quantity a CFD
    double-body form factor is compared against.
    """
    cp = principal.Cp
    nabla = principal.nabla
    lr = holtrop_lr(principal.L, cp, principal.lcb_pct)
    c14 = 1.0 + 0.011 * principal.Cstern
    value = (0.93
             + 0.487118 * c14
             * (principal.B / principal.L) ** 1.06806
             * (principal.T / principal.L) ** 0.46106
             * (principal.L / lr) ** 0.121563
             * (principal.L ** 3 / nabla) ** 0.36486
             * (1.0 - cp) ** -0.604247)
    if not math.isfinite(value) or value <= 1.0:
        raise ValueError(f"Holtrop form factor outside physical range: {value!r}")
    return value


def holtrop_wetted_surface(principal: HullPrincipal) -> float:
    """Holtrop (1984) wetted surface ``S`` [m^2] (appendages excluded)."""
    p = principal
    value = p.L * (2.0 * p.T + p.B) * math.sqrt(p.Cm) * (
        0.453 + 0.4425 * p.Cb - 0.2862 * p.Cm
        - 0.003467 * p.B / p.T + 0.3696 * p.Cwp
    ) + 2.38 * p.ABT / p.Cb
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("wetted surface must be finite and positive")
    return value


def _holtrop_c2(principal: HullPrincipal, TF: float) -> float:
    """Bulbous-bow coefficient ``c2`` (1 when there is no bulb)."""
    if principal.ABT <= 0.0:
        return 1.0
    denominator = principal.B * principal.T * (
        0.31 * math.sqrt(principal.ABT) + TF - principal.hB)
    if denominator <= 0.0:
        raise ValueError("illegal bulbous-bow geometry for the c2 term")
    c3 = 0.56 * principal.ABT ** 1.5 / denominator
    return math.exp(-1.89 * math.sqrt(c3))


def holtrop_correlation_allowance(principal: HullPrincipal,
                                  TF: Optional[float] = None) -> float:
    """Holtrop (1984) model-ship correlation allowance ``Ca`` (dimensionless).

    ``Ca = 0.006 (L+100)^-0.16 - 0.00205
    + 0.003 sqrt(L/7.5) Cb^4 c2 (0.04 - c4)`` with ``c4 = min(TF/L, 0.04)``.

    This is the *shape-driven* convention: it needs no roughness measurement.
    It is an alternative to :func:`ittc78_delta_cf`, not an addition to it.
    """
    tf = principal.T if TF is None else TF
    if tf <= 0.0:
        raise ValueError("TF must be positive")
    c4 = min(tf / principal.L, 0.04)
    c2 = _holtrop_c2(principal, tf)
    return (0.006 * (principal.L + 100.0) ** -0.16 - 0.00205
            + 0.003 * math.sqrt(principal.L / 7.5) * principal.Cb ** 4
            * c2 * (0.04 - c4))


# --------------------------------------------------------------------------- #
# CFD form-factor extraction (double body, no free surface)
# --------------------------------------------------------------------------- #
def form_factor_from_double_body(pressure_force_n: float, shear_force_n: float,
                                 dynamic_pressure_pa: float,
                                 wetted_area_m2: float) -> Dict[str, float]:
    """Split a double-body force into friction and viscous pressure drag.

    ``C_F = R_shear / (q S)``, ``C_Pv = R_pressure / (q S)`` and
    ``(1+k) = (C_F + C_Pv) / C_F``.  Both force components are the *stationary
    double-body* forces referred to the ship's wetted surface ``S`` (the double
    body has twice the wetted area and twice the force, so the coefficient is
    unchanged by the mirroring).

    Sign convention: both components are taken positive downstream.  The total
    double-body resistance is recombining them, which is checked explicitly
    against the sum of the two components by the caller.
    """
    for name, value in (("pressure_force_n", pressure_force_n),
                        ("shear_force_n", shear_force_n),
                        ("dynamic_pressure_pa", dynamic_pressure_pa),
                        ("wetted_area_m2", wetted_area_m2)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if dynamic_pressure_pa <= 0.0 or wetted_area_m2 <= 0.0:
        raise ValueError("dynamic pressure and wetted area must be positive")
    if shear_force_n <= 0.0:
        raise ValueError(
            "friction force must be positive downstream; check the surface "
            "normal orientation used for the integration"
        )
    qS = dynamic_pressure_pa * wetted_area_m2
    cf = shear_force_n / qS
    cpv = pressure_force_n / qS
    one_plus_k = (cf + cpv) / cf
    return {
        "C_F": cf,
        "C_Pv": cpv,
        "C_V": cf + cpv,
        "one_plus_k": one_plus_k,
        "form_factor_k": one_plus_k - 1.0,
        "pressure_fraction": cpv / (cf + cpv),
        "pressure_force_n": pressure_force_n,
        "shear_force_n": shear_force_n,
        "qS_n": qS,
    }


def compare_form_factor(one_plus_k_cfd: float, principal: HullPrincipal,
                        tolerance: float = 0.05) -> Dict[str, float]:
    """Adjudicate a CFD form factor against the Holtrop regression value.

    Returns the relative deviation and whether it is inside ``tolerance``
    (the mission gate is 5 %).
    """
    if not math.isfinite(one_plus_k_cfd) or one_plus_k_cfd <= 1.0:
        raise ValueError("(1+k) must be finite and greater than 1")
    holtrop = holtrop_form_factor(principal)
    deviation = (one_plus_k_cfd - holtrop) / holtrop
    return {
        "one_plus_k_cfd": float(one_plus_k_cfd),
        "one_plus_k_holtrop": holtrop,
        "k_cfd": float(one_plus_k_cfd) - 1.0,
        "k_holtrop": holtrop - 1.0,
        "relative_deviation": deviation,
        "absolute_deviation": abs(deviation),
        "tolerance": tolerance,
        "passes": abs(deviation) <= tolerance,
    }


# --------------------------------------------------------------------------- #
# The unified chain
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ViscousChainResult:
    """One evaluated state of the resistance-in-convention chain."""

    principal_name: str
    speed_ms: float
    reynolds: float
    dynamic_pressure_pa: float
    wetted_area_m2: float
    one_plus_k: float
    one_plus_k_source: str
    cf_law: str
    cf: float
    allowance_regime: str
    allowance: float
    allowance_ks: Optional[float]
    C_F_wetted: float          # (1+k) C_F
    C_T: float                 # (1+k) C_F + allowance
    components: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        payload = {
            "principal": self.principal_name,
            "speed_ms": self.speed_ms,
            "reynolds": self.reynolds,
            "dynamic_pressure_pa": self.dynamic_pressure_pa,
            "wetted_area_m2": self.wetted_area_m2,
            "one_plus_k": self.one_plus_k,
            "one_plus_k_source": self.one_plus_k_source,
            "cf_law": self.cf_law,
            "cf": self.cf,
            "allowance_regime": self.allowance_regime,
            "allowance": self.allowance,
            "allowance_ks": self.allowance_ks,
            "C_F_wetted": self.C_F_wetted,
            "C_T": self.C_T,
            "components": dict(self.components),
        }
        return payload


def viscous_chain(principal: HullPrincipal, speed_ms: float,
                  regime: str = "ittc78_roughness",
                  cf_law: str = "ittc57",
                  one_plus_k: Optional[float] = None,
                  ks: Optional[float] = None,
                  nu: float = NU_SEAWATER_15C,
                  rho: float = RHO_SEAWATER) -> ViscousChainResult:
    """Evaluate ``C_T = (1+k) C_F + allowance`` in exactly one convention.

    Parameters
    ----------
    regime
        ``"ittc78_roughness"`` (needs ``ks``), ``"holtrop_ca"`` (shape-driven),
        or ``"none"`` (double-body / smooth-model comparison, allowance = 0).
    cf_law
        ``"ittc57"``, ``"schoenherr"`` or ``"prandtl_schlichting"``.
    one_plus_k
        Form factor; defaults to the Holtrop regression value and records which
        source was used so the two branches stay distinguishable.
    ks
        Equivalent sand-grain roughness [m] for the ITTC-78 regime.

    The two allowance conventions are mutually exclusive here on purpose: the
    historical chain added both, which is precisely the defect this function
    removes.
    """
    if regime not in ALLOWANCE_REGIMES:
        raise ValueError(f"regime must be one of {ALLOWANCE_REGIMES}")
    if regime == "ittc78_roughness" and ks is None:
        raise ValueError("the ittc78_roughness regime requires an explicit ks")
    if regime == "holtrop_ca" and ks is not None:
        raise ValueError(
            "holtrop_ca is shape-driven: pass ks=None. Supplying a roughness "
            "means you want the ittc78_roughness regime, not both."
        )
    if not math.isfinite(speed_ms) or speed_ms <= 0.0:
        raise ValueError("speed must be finite and positive")
    if not math.isfinite(nu) or nu <= 0.0:
        raise ValueError("nu must be finite and positive")

    re = speed_ms * principal.L / nu
    laws = {
        "ittc57": ittc57_cf,
        "schoenherr": schoenherr_cf,
        "prandtl_schlichting": prandtl_schlichting_cf,
    }
    if cf_law not in laws:
        raise ValueError(f"cf_law must be one of {tuple(laws)}")
    cf = laws[cf_law](re)

    if one_plus_k is None:
        one_plus_k = holtrop_form_factor(principal)
        source = "holtrop_regression"
    else:
        if not math.isfinite(one_plus_k) or one_plus_k <= 1.0:
            raise ValueError("(1+k) must be finite and > 1")
        one_plus_k = float(one_plus_k)
        source = "user_supplied"

    if regime == "ittc78_roughness":
        allowance = ittc78_delta_cf(float(ks), principal.L)
    elif regime == "holtrop_ca":
        allowance = holtrop_correlation_allowance(principal)
    else:
        allowance = 0.0

    q = 0.5 * rho * speed_ms ** 2
    s = holtrop_wetted_surface(principal)
    c_f_wetted = one_plus_k * cf
    return ViscousChainResult(
        principal_name=principal.name,
        speed_ms=float(speed_ms),
        reynolds=re,
        dynamic_pressure_pa=q,
        wetted_area_m2=s,
        one_plus_k=one_plus_k,
        one_plus_k_source=source,
        cf_law=cf_law,
        cf=cf,
        allowance_regime=regime,
        allowance=allowance,
        allowance_ks=None if ks is None else float(ks),
        C_F_wetted=c_f_wetted,
        C_T=c_f_wetted + allowance,
        components={
            "friction": c_f_wetted,
            "form": c_f_wetted - cf,
            "allowance": allowance,
        },
    )


def reynolds_table(re_min: float = 1e6, re_max: float = 1e9, n_points: int = 8,
                   ks: float = DEFAULT_SHIP_ROUGHNESS_KS,
                   L: Optional[float] = None) -> List[Dict[str, float]]:
    """Tabulate the friction laws and the ITTC-78 allowance over ``n_points`` Re.

    ``L`` fixes the ship length used by the allowance formula; when omitted the
    table is *purely* Reynolds-indexed and the allowance column is reported per
    unit: the ITTC-78 ``dCf`` has no Reynolds dependence of its own (it scales
    with ``(ks/L)^(1/3)``), which the table states explicitly through
    ``allowance_reynolds_dependent = 0``.  Supplying ``L`` makes the table a
    build-ready chain table for one ship.
    """
    if not (math.isfinite(re_min) and math.isfinite(re_max)) or re_min <= 0 or re_min >= re_max:
        raise ValueError("require 0 < re_min < re_max")
    if not isinstance(n_points, int) or n_points < 2:
        raise ValueError("n_points must be an integer >= 2")
    rows: List[Dict[str, float]] = []
    step = (math.log10(re_max) - math.log10(re_min)) / (n_points - 1)
    for i in range(n_points):
        re = 10.0 ** (math.log10(re_min) + i * step)
        band = friction_band(re)
        row = {
            "reynolds": re,
            "ittc57_cf": band["values"]["ittc57"],
            "schoenherr_cf": band["values"]["schoenherr"],
            "prandtl_schlichting_cf": band["values"]["prandtl_schlichting"],
            "law_spread": band["spread"],
            "ittc78_delta_cf": ittc78_delta_cf(ks, L) if L else float("nan"),
            "allowance_reynolds_dependent": 0,
        }
        rows.append(row)
    return rows


def correlation_allowance_table(lengths: Sequence[float],
                                ks_values: Sequence[float]) -> List[Dict[str, float]]:
    """``dCf(ks, L)`` on the requested grid, with the smooth-surface floor flagged."""
    if not lengths or not ks_values:
        raise ValueError("lengths and ks_values must be non-empty")
    out: List[Dict[str, float]] = []
    for L in lengths:
        for ks in ks_values:
            raw = (105.0 * (ks / L) ** (1.0 / 3.0) - 0.64) * 1e-3
            out.append({
                "L_m": float(L),
                "ks_m": float(ks),
                "delta_cf": ittc78_delta_cf(float(ks), float(L)),
                "raw_value": raw,
                "floored_to_zero": raw <= 0.0,
                "ks_over_L": float(ks) / float(L),
            })
    return out


# --------------------------------------------------------------------------- #
# Validation uncertainty budget
# --------------------------------------------------------------------------- #
def validation_uncertainty_budget(principal: HullPrincipal, speed_ms: float,
                                  regime: str = "ittc78_roughness",
                                  ks: Optional[float] = None,
                                  one_plus_k_cfd: Optional[float] = None,
                                  u_cfd: Optional[float] = None,
                                  nu: float = NU_SEAWATER_15C,
                                  rho: float = RHO_SEAWATER,
                                  allowance_model_band: float = 0.02,
                                  form_prior_band: float = 0.05,
                                  tolerance: float = 0.05) -> Dict[str, object]:
    """Relative validation-uncertainty budget ``U_V`` of the unified chain.

    Convention (documented so it can be argued with, ITTC 7.5-02-02-02 style):
    each contributions is a *relative standard uncertainty* of ``C_T``, obtained
    from a rectangular (maximum-entropy) bound ``b`` as ``u = b / sqrt(3)``, and
    they are combined in quadrature.

    ``u_friction``
        half-width of the three-law friction band at the operating Re,
        propagated to ``C_T`` (``(1+k) dCf / C_T`` shape, see below);
    ``u_form``
        disagreement between the CFD double-body form factor and the Holtrop
        regression when a CFD value is supplied, else ``form_prior_band``;
    ``u_allowance``
        allowance-model band (default 2 %) *only* when exactly one convention
        is used; if the historical double-counting is re-introduced the
        allowance contribution becomes ``(dCf + Ca)/C_T``, which is the bias
        the gate is meant to catch;
    ``u_cfd``
        discretisation uncertainty of the CFD form factor (grid study); omitted
        from the total when ``None`` and reported as ``withheld``.

    ``U_V`` is the quadrature sum, and ``passes`` compares it to ``tolerance``
    (5 % in the mission).
    """
    chain = viscous_chain(principal, speed_ms, regime=regime, ks=ks, nu=nu, rho=rho)
    band = friction_band(chain.reynolds)
    c_t = chain.C_T
    # friction-law spread enters through the wetted C_F part of C_T
    u_friction = (0.5 * band["spread"] / math.sqrt(3.0)) * chain.C_F_wetted / c_t

    if one_plus_k_cfd is not None:
        dev = abs(one_plus_k_cfd - chain.one_plus_k) / chain.one_plus_k
        u_form = (dev / math.sqrt(3.0)) * chain.C_F_wetted / c_t
        form_source = "cfd_vs_holtrop"
    else:
        dev = form_prior_band
        u_form = (form_prior_band / math.sqrt(3.0)) * chain.C_F_wetted / c_t
        form_source = "prior_band"
    u_allowance = (allowance_model_band / math.sqrt(3.0)) * (chain.allowance / c_t) \
        if chain.allowance > 0 else 0.0
    components = {"friction": u_friction, "form": u_form, "allowance": u_allowance}
    if u_cfd is not None:
        if u_cfd < 0.0:
            raise ValueError("u_cfd must be non-negative")
        components["cfd"] = (u_cfd / math.sqrt(3.0)) * chain.C_F_wetted / c_t
    total = math.sqrt(sum(v * v for v in components.values()))

    # state of the chain *before* unification: both allowances applied, with the
    # roughness this call was given (or the ship default when it was shape-only)
    roughness = DEFAULT_SHIP_ROUGHNESS_KS if ks is None else float(ks)
    double_counted = (ittc78_delta_cf(roughness, principal.L)
                      + holtrop_correlation_allowance(principal))
    u_allowance_unmerged = (double_counted / math.sqrt(3.0)) / c_t
    total_unmerged = math.sqrt(sum(v * v for k, v in components.items() if k != "allowance")
                               + u_allowance_unmerged ** 2)
    return {
        "reynolds": chain.reynolds,
        "C_T": c_t,
        "regime": chain.allowance_regime,
        "allowance": chain.allowance,
        "cfd_form_factor_deviation": dev,
        "form_factor_source": form_source,
        "components": components,
        "U_V": total,
        "U_V_percent": 100.0 * total,
        "tolerance": tolerance,
        "passes": total <= tolerance,
        # the defect the unification removes, quantified
        "unmerged_double_counted_allowance": double_counted,
        "U_V_before_unification": total_unmerged,
        "U_V_before_unification_percent": 100.0 * total_unmerged,
        "cfd_discretisation_withheld": u_cfd is None,
    }


# --------------------------------------------------------------------------- #
# Model-to-ship extrapolation (ITTC-1978 flow)
# --------------------------------------------------------------------------- #
def extrapolate_to_ship(c_t_model: float, model: HullPrincipal, ship: HullPrincipal,
                        speed_kn: float,
                        one_plus_k: Optional[float] = None,
                        ks_ship: float = DEFAULT_SHIP_ROUGHNESS_KS,
                        nu_model: float = NU_FRESHWATER_15C,
                        nu_ship: float = NU_SEAWATER_15C,
                        include_air: bool = False) -> Dict[str, float]:
    """1978-performance-committee model/ship extrapolation in one convention.

    ``C_R = C_TM - (1+k) C_FM`` and
    ``C_TS = (1+k) C_FS + C_R + dCf`` (plus air resistance when requested).
    The allowance is the ITTC-78 roughness term only: adding Holtrop's ``Ca``
    here as well is exactly the double counting this module forbids.
    """
    for name, value in (("c_t_model", c_t_model), ("speed_kn", speed_kn)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if c_t_model <= 0.0 or speed_kn <= 0.0:
        raise ValueError("total resistance coefficient and speed must be positive")
    v = speed_kn * KNOT
    one_plus_k = holtrop_form_factor(ship) if one_plus_k is None else float(one_plus_k)
    re_model = v * model.L / nu_model
    re_ship = v * ship.L / nu_ship
    c_fm = ittc57_cf(re_model)
    c_fs = ittc57_cf(re_ship)
    c_r = c_t_model - one_plus_k * c_fm
    delta = ittc78_delta_cf(ks_ship, ship.L)
    c_aa = 0.0
    if include_air:
        # Holtrop's air-resistance form, evaluated on the ship waterline area
        # 0.5 rho_air V^2 A_TV / (0.5 rho_w V^2 S): reported separately, never
        # merged into the viscous terms.
        a_tv = ship.Cwp * ship.L * ship.B * 0.4
        c_aa = (0.5 * 1.225 * v ** 2 * a_tv) / (0.5 * RHO_SEAWATER * v ** 2
                                               * holtrop_wetted_surface(ship))
    c_ts = one_plus_k * c_fs + c_r + delta + c_aa
    return {
        "reynolds_model": re_model,
        "reynolds_ship": re_ship,
        "one_plus_k": one_plus_k,
        "C_FM": c_fm,
        "C_FS": c_fs,
        "C_R": c_r,
        "delta_cf": delta,
        "C_AA": c_aa,
        "C_TS": c_ts,
        "C_TM": c_t_model,
        "scale_effect": c_ts - c_t_model,
    }