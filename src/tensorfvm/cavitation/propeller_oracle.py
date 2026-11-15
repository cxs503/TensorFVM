"""Independent closed-form oracle for propeller cavitation-criteria quantities.

Why this module exists
----------------------
The ship-performance platform exposes a ``cavitation-detailed`` endpoint whose
criterion quantities (``tau_c``, ``sigma_0.7R``, Keller minimum area ratio,
Burrill verdict, vapour pressure) were declared single-source in
``backend/api/cavitation_criteria.py`` (mission m1938).  On 2026-11-13 five of
those names were found to be referenced but *undefined* in the backend tree
("半接线"): the endpoint was hot-fixed with ``_cav_wiring_pending`` stubs so the
backend could start at all, and every call to the criterion path raises
``NotImplementedError``.

Accepting that hotfix ("五量返回真值") requires an *independent* instrument: a
re-derivation of the same quantities from their published definitions, which can
be (a) hand-checked and (b) compared against whatever the platform returns.  That
is this module.  It is a **verification oracle, never a production path**:

* it is not imported by the platform, and the platform's single source of these
  quantities stays ``backend/api/cavitation_criteria.py``;
* every quantity here is a closed form (no fitted data, no lookup tables that
  cannot be re-derived), so a mismatch against the platform is a defect on one
  side, never a "tuning" question;
* the only criterion *line* used is Burrill's, and the Burrill limit lines
  (5 % / 10 % / 20 % back cavitation) are **WITHHELD**: no digitised chart data
  exists in this workspace, so the default line is the platform's
  ``LEGACY_PLACEHOLDER`` ``sigma >= tau + 0.2`` explicitly tagged as such, and
  :func:`margin_sensitivity_per_unit_sigma` quantifies how sensitive the verdict
  is to that unverified line.

Definitions (all SI: m, s, N, Pa, rad/s)
----------------------------------------
``sigma_0.7R = (p_0 - p_v) / (0.5 rho V_0.7R^2)``,
``V_0.7R = sqrt(Va^2 + (0.7 pi n D)^2)``, ``p_0 = p_atm + rho g h``;
``tau_c = T / (0.5 rho A_p V_0.7R^2)`` with Taylor's projected blade area
``A_p = (Ae/Ao) A_0 (1.067 - 0.229 P/D)``, ``A_0 = pi D^2 / 4``;
Keller (1966) ``(Ae/Ao)_min = (1.3 + 0.3 Z) T / ((p_0 - p_v) D^2) + Delta A_o``.

The definitions are the published ones, unchanged from
``backend/api/cavitation_criteria.py``; the code below is an independent
transcription so that a transcription error in either tree is detectable.
"""
from __future__ import annotations

import json
import math
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

from .inception import water_vapour_pressure

G = 9.80665          # m/s^2
P_ATM = 101325.0     # Pa
RHO_SEA = 1025.0     # kg/m^3

#: Keller (1966) ``Delta A_o`` correction, quoted per ship type (Carlton,
#: *Marine Propellers and Propulsion*, 3rd ed., "Cavitation").
KELLER_DELTA = {
    "fast-twin-screw": 0.0,
    "single-screw": 0.1,
    "slow-twin-screw": 0.2,
}

#: Name of the unverified Burrill limit line (the platform's legacy placeholder).
LEGACY_PLACEHOLDER = "LEGACY_PLACEHOLDER"


# --------------------------------------------------------------------------- #
# 1. kinematics, geometry, statics
# --------------------------------------------------------------------------- #
def advance_ratio(n_rps: float, D: float, Va: float) -> float:
    """Advance ratio ``J = Va / (n D)``."""
    if n_rps <= 0.0 or D <= 0.0:
        raise ValueError("n_rps and D must be > 0")
    return Va / (n_rps * D)


def v_07r(n_rps: float, D: float, Va: float) -> float:
    """Resultant inflow speed at 0.7R, ``sqrt(Va^2 + (0.7 pi n D)^2)`` [m/s]."""
    if n_rps <= 0.0 or D <= 0.0:
        raise ValueError("n_rps and D must be > 0")
    return math.hypot(Va, 0.7 * math.pi * n_rps * D)


def static_pressure(depth: float, rho: float = RHO_SEA,
                    p_atm: float = P_ATM) -> float:
    """Absolute static pressure at the shaft immersion ``h`` [Pa]."""
    if depth < 0.0:
        raise ValueError("depth must be >= 0")
    return p_atm + rho * G * depth


def disc_area(D: float) -> float:
    """Propeller disc area ``A_0 = pi D^2 / 4`` [m^2]."""
    if D <= 0.0:
        raise ValueError("D must be > 0")
    return math.pi * D ** 2 / 4.0


def taylor_projected_area(Ae_Ao: float, D: float, P_D: float) -> float:
    """Taylor (1933) projected blade area ``A_p`` [m^2] (B-series, ~2 %)."""
    if not (0.0 < Ae_Ao <= 1.05):
        raise ValueError("Ae_Ao must be in (0, 1.05]")
    if P_D <= 0.0:
        raise ValueError("P_D must be > 0")
    return Ae_Ao * disc_area(D) * (1.067 - 0.229 * P_D)


def driving_pressure(depth: float, temperature_c: float = 15.0,
                     rho: float = RHO_SEA, p_atm: float = P_ATM,
                     p_v: Optional[float] = None) -> float:
    """``p_0 - p_v`` [Pa]; raises if the section is already vaporous."""
    pv = water_vapour_pressure(temperature_c) if p_v is None else float(p_v)
    dp = static_pressure(depth, rho, p_atm) - pv
    if dp <= 0.0:
        raise ValueError("p_0 - p_v <= 0: vaporous conditions, sigma undefined")
    return dp


# --------------------------------------------------------------------------- #
# 2. criterion quantities (all closed form)
# --------------------------------------------------------------------------- #
def sigma_07r(n_rps: float, D: float, Va: float, depth: float,
              temperature_c: float = 15.0, rho: float = RHO_SEA,
              p_atm: float = P_ATM, p_v: Optional[float] = None) -> float:
    """Cavitation number at 0.7R (the ordinate of the Burrill diagram)."""
    dp = driving_pressure(depth, temperature_c, rho, p_atm, p_v)
    v = v_07r(n_rps, D, Va)
    return dp / (0.5 * rho * v ** 2)


def sigma_n_sigma_07r_ratio(n_rps: float, D: float, Va: float) -> float:
    """Analytic ``sigma_n / sigma_0.7R = J^2 + (0.7 pi)^2``.

    This identity is what makes the pre-m1938 implementation non-conservative:
    substituting ``sigma_n`` (reference speed ``nD``) for ``sigma_0.7R`` inflates
    the cavitation number by this factor, i.e. by ~5x at J = 0.4.
    """
    j = advance_ratio(n_rps, D, Va)
    return j * j + (0.7 * math.pi) ** 2


def tau_c(thrust_N: float, n_rps: float, D: float, Va: float,
          Ae_Ao: float, P_D: float, rho: float = RHO_SEA) -> float:
    """Thrust loading coefficient on the projected blade area (Burrill abscissa)."""
    a_p = taylor_projected_area(Ae_Ao, D, P_D)
    v = v_07r(n_rps, D, Va)
    return thrust_N / (0.5 * rho * a_p * v ** 2)


def keller_min_area_ratio(thrust_N: float, D: float, depth: float, Z: int,
                          ship_type: str = "single-screw",
                          temperature_c: float = 15.0,
                          rho: float = RHO_SEA, p_atm: float = P_ATM,
                          p_v: Optional[float] = None) -> Tuple[float, float]:
    """Keller (1966) minimum expanded area ratio; returns ``(ratio, Delta A_o)``."""
    if ship_type not in KELLER_DELTA:
        raise ValueError(f"ship_type must be one of {sorted(KELLER_DELTA)}")
    dp = driving_pressure(depth, temperature_c, rho, p_atm, p_v)
    delta = KELLER_DELTA[ship_type]
    base = (1.3 + 0.3 * float(Z)) * thrust_N / (dp * D ** 2)
    return base + delta, delta


# --------------------------------------------------------------------------- #
# 3. Burrill verdict on an explicitly-provenanced limit line
# --------------------------------------------------------------------------- #
def legacy_placeholder_curve(tau: float) -> float:
    """The platform's unverified legacy line ``sigma >= tau + 0.2``."""
    return tau + 0.2


CriterionCurve = Callable[[float], float]


@dataclass(frozen=True)
class BurrillPoint:
    """Working point on the Burrill plane plus its verdict provenance."""

    tau_c: float
    sigma_07r: float
    sigma_limit: float
    margin_pct: float
    passes: bool
    curve_provenance: str
    notes: tuple = field(default_factory=tuple)

    @property
    def design_grade(self) -> bool:
        """A verdict is design grade only with a literature-backed limit line."""
        return self.curve_provenance != LEGACY_PLACEHOLDER


def burrill_point(tau: float, sigma: float,
                  curve: Optional[CriterionCurve] = None,
                  curve_provenance: Optional[str] = None) -> BurrillPoint:
    """Margin of a working point against a limit line (WITHHELD by default)."""
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    if curve is None:
        curve = legacy_placeholder_curve
        curve_provenance = LEGACY_PLACEHOLDER
    elif curve_provenance is None:
        raise ValueError("a literature curve must carry its provenance string")
    limit = float(curve(tau))
    margin = (sigma - limit) / sigma * 100.0
    notes = ()
    if curve_provenance == LEGACY_PLACEHOLDER:
        notes = ("limit line has no literature provenance; pass/fail is NOT "
                 "design grade",)
    return BurrillPoint(tau_c=tau, sigma_07r=sigma, sigma_limit=limit,
                        margin_pct=margin, passes=margin > 0.0,
                        curve_provenance=str(curve_provenance), notes=notes)


def margin_sensitivity_per_unit_sigma(sigma: float) -> float:
    """``d(margin_pct)/d(sigma_limit) = -100 / sigma`` [%/unit].

    Quantifies how much an unverified limit line moves the verdict: at
    ``sigma_0.7R`` ~ 0.6 a 10 % error on the line shifts the margin by ~17
    percentage points, i.e. more than a typical margin.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return -100.0 / sigma


def sigma_limit_for_zero_margin(sigma: float) -> float:
    """Limit-line value that would put the working point exactly on the line."""
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return sigma


# --------------------------------------------------------------------------- #
# 4. Definition-only regime screen (no invented cavitation-type bands)
# --------------------------------------------------------------------------- #
def screen_regime(thrust_N: float, D: float, depth: float, Z: int,
                  ship_type: str, n_rps: float, Va: float,
                  Ae_Ao: float, P_D: float, temperature_c: float = 15.0,
                  rho: float = RHO_SEA) -> dict:
    """Screen a design point using *closed-form criteria only*.

    Two independent literature-backed statements are returned, plus an explicit
    ``withheld`` list for what cannot be decided in this workspace:

    * Keller: is the actual ``Ae/Ao`` below the required minimum?
    * Burrill: the margin (provenance-tagged; placeholder line -> not design
      grade).

    No cavitation *type* (sheet / tip vortex / bubble) is asserted: that needs
    chordwise pressure distributions at off-design incidence or digitised
    chart bands, neither of which is available here.  Asserting it would be
    fabrication, so it is reported as withheld instead.
    """
    tau = tau_c(thrust_N, n_rps, D, Va, Ae_Ao, P_D, rho=rho)
    sigma = sigma_07r(n_rps, D, Va, depth, temperature_c, rho=rho)
    ratio_min, delta = keller_min_area_ratio(
        thrust_N, D, depth, Z, ship_type=ship_type,
        temperature_c=temperature_c, rho=rho)
    point = burrill_point(tau, sigma)
    keller_ok = Ae_Ao >= ratio_min
    withheld = ["burrill limit lines (no digitised chart data)",
                "cavitation type bands (needs chordwise Cp or chart bands)"]
    return {
        "tau_c": tau,
        "sigma_0.7R": sigma,
        "J": advance_ratio(n_rps, D, Va),
        "keller_min_area_ratio": ratio_min,
        "keller_delta_ao": delta,
        "keller_ok": keller_ok,
        "keller_deficit": Ae_Ao - ratio_min,
        "burrill_margin_pct": point.margin_pct,
        "burrill_pass_placeholder": point.passes,
        "burrill_curve_provenance": point.curve_provenance,
        "burrill_design_grade": point.design_grade,
        "criteria_disagree": bool(keller_ok != point.passes),
        "withheld": withheld,
    }


# --------------------------------------------------------------------------- #
# 5. probe of the platform endpoint (acceptance instrument)
# --------------------------------------------------------------------------- #
PLATFORM_CHECK_PAYLOAD = {
    "D": 3.0, "P_D": 1.0, "Ae_Ao": 0.65, "Z": 4, "n": 3.0, "Va": 6.17,
    "depth": 4.0, "thrust": 300000.0, "delivered_power": 3500.0,
    "ship_speed": 12.0, "temperature": 20.0, "rho": RHO_SEA,
}


def probe_platform_endpoint(base_url: str,
                            payload: Optional[dict] = None,
                            timeout: float = 5.0) -> dict:
    """POST the criterion endpoint and classify the answer.

    Returns ``{"state": ...}` where state is one of ``"ok"`` (HTTP 200 with a
    JSON body), ``"not_wired"`` (HTTP 5xx: the ``_cav_wiring_pending`` stubs),
    ``"rejected"`` (HTTP 4xx: payload contract changed) or ``"unreachable"``.
    Never raises: this is a diagnostic, not a control path.
    """
    body = json.dumps(payload or PLATFORM_CHECK_PAYLOAD).encode()
    url = base_url.rstrip("/") + "/api/v1/cavitation-detailed/check"
    request = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {"state": "ok", "http_status": response.status,
                    "url": url, "body": response.read().decode()[:2000]}
    except urllib.error.HTTPError as exc:      # 4xx / 5xx answers
        state = "not_wired" if exc.code >= 500 else "rejected"
        return {"state": state, "http_status": exc.code, "url": url,
                "body": exc.read().decode()[:2000]}
    except (urllib.error.URLError, OSError) as exc:
        return {"state": "unreachable", "url": url, "error": str(exc)}


def compare_against_platform(result: dict, *, rtol: float = 0.01) -> dict:
    """Compare a platform answer with the oracle for the same payload.

    Reads whichever of the criterion quantities the platform actually returns
    (nested ``criteria_results`` or flat keys) and reports per-quantity relative
    error.  Quantities absent from the answer are reported as ``"absent"``
    rather than silently passed.
    """
    payload = PLATFORM_CHECK_PAYLOAD
    oracle = {
        "tau_c": tau_c(payload["thrust"], payload["n"], payload["D"],
                       payload["Va"], payload["Ae_Ao"], payload["P_D"],
                       rho=payload["rho"]),
        "sigma_0.7R": sigma_07r(payload["n"], payload["D"], payload["Va"],
                                payload["depth"], payload["temperature"],
                                rho=payload["rho"]),
    }
    flat: dict = {}

    def walk(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    walk(value)
                elif isinstance(value, (int, float)) and not isinstance(value, bool):
                    flat.setdefault(key, value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    body = result.get("body")
    try:
        walk(json.loads(body) if isinstance(body, str) else (body or {}))
    except json.JSONDecodeError:
        return {"state": result.get("state"), "error": "non-JSON body"}
    report: dict = {"state": result.get("state"), "quantities": {}}
    for name, expected in oracle.items():
        found = flat.get(name, flat.get(name.replace("0.7R", "07r")))
        if found is None:
            report["quantities"][name] = {"platform": None, "verdict": "absent"}
            continue
        rel = abs(float(found) - expected) / abs(expected)
        report["quantities"][name] = {
            "platform": float(found), "oracle": expected, "rel_error": rel,
            "verdict": "ok" if rel <= rtol else "mismatch"}
    return report


__all__ = [
    "G", "P_ATM", "RHO_SEA", "KELLER_DELTA", "LEGACY_PLACEHOLDER",
    "advance_ratio", "v_07r", "static_pressure", "disc_area",
    "taylor_projected_area", "driving_pressure", "sigma_07r",
    "sigma_n_sigma_07r_ratio", "tau_c", "keller_min_area_ratio",
    "legacy_placeholder_curve", "BurrillPoint", "burrill_point",
    "margin_sensitivity_per_unit_sigma", "sigma_limit_for_zero_margin",
    "screen_regime", "PLATFORM_CHECK_PAYLOAD", "probe_platform_endpoint",
    "compare_against_platform",
]