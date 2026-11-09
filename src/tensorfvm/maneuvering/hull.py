"""KVLCC2 model geometry, conditions and the moving-surface (rudder) geometry.

Why a separate module
---------------------
Three different geometry faces are needed by a manoeuvring study and they must
not drift apart:

* the **hull integrals** (``Cb``/``Cwp``/``LCB``/``∇``) that gate every
  downstream number -- the platform rule is "geometry must be verified before
  any hydrodynamics";
* the **lateral planform** (local draft and side area along the length) that the
  cross-flow / low-aspect-ratio lift model integrates over;
* the **rudder** (area ratio, span, chord, stock position, section), which sets
  the rudder force derivative together with the section lift slope.

All three are derived here from the repository's calibrated parametric hull
(:class:`tensorfvm.rans_ship_resistance.kvlcc2.Kvlcc2Hull`), whose fore/aft
exponents were fitted to the published KVLCC2 integral coefficients.  The rudder
is *nominal*: the published rudder-area ratio ``A_R/(L T)`` is used, the digitized
horn-rudder offsets are not in this workspace, and the section is modelled as a
0.18-thickness NACA-4-digit type profile (the usual KVLCC2 rudder
idealisation).  That limitation is propagated into the report.

Model-scale particulars (KVLCC2, scale ratio 320/7 = 45.714)
-----------------------------------------------------------
``L = 7.000 m``, ``B = 1.2688 m``, ``T = 0.4550 m``, ``Cb = 0.8098``,
``Cwp = 0.870``, ``LCB = -1.688 % L``, design speed ``U = 1.176 m/s``
(``Fn = 0.142``, i.e. ``U = Fn sqrt(gL)``; note that several documents in this
workspace quote ``U = 0.771 m/s`` together with ``Fn = 0.142``, which is
*inconsistent* -- 0.771 m/s at L = 7 m is Fn = 0.093, and it would give
Re = 4.7e6 instead of the workshop's 7.23e6.  The Froude-scaling route is used
here), fresh water ``rho = 998.2 kg/m^3``, ``nu = 1.139e-6 m^2/s``
(15 degC, the platform's towing-tank value).  Those are *structural* facts of
the benchmark, quoted identically across the KVLCC2 literature; the *resistance*
coefficient is a different matter and carries ``provenance="withheld"`` (see
:mod:`.reference`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from ..rans_ship_resistance.kvlcc2 import (
    KVLCC2_MODEL_PARTICULARS, Kvlcc2Hull, build_kvlcc2_hull,
)
from ..resistance_chain import NU_FRESHWATER_15C, RHO_SEAWATER  # noqa: F401

__all__ = [
    "RHO_MODEL", "NU_MODEL", "GRAVITY", "KVLCC2_MODEL_LENGTH", "KVLCC2_FN_DESIGN",
    "KVLCC2_PROTOTYPE_SCALE", "Kvlcc2Conditions", "RudderGeometry",
    "build_hull", "geometry_gate", "section_data", "rudder_geometry",
]

#: Fresh water model-test density [kg/m^3] (towing-tank value).
RHO_MODEL = 998.2
#: Kinematic viscosity of fresh water at 15 degC [m^2/s] (platform value).
NU_MODEL = NU_FRESHWATER_15C
GRAVITY = 9.80665
KVLCC2_MODEL_LENGTH = 7.0
KVLCC2_FN_DESIGN = 0.142
KVLCC2_PROTOTYPE_SCALE = 320.0 / 7.0


@dataclass(frozen=True)
class Kvlcc2Conditions:
    """Model-scale reference conditions of the benchmark."""

    length: float = KVLCC2_MODEL_LENGTH
    beam: float = KVLCC2_MODEL_PARTICULARS["B"]
    draft: float = KVLCC2_MODEL_PARTICULARS["T"]
    block_coefficient: float = KVLCC2_MODEL_PARTICULARS["Cb"]
    waterplane_coefficient: float = KVLCC2_MODEL_PARTICULARS["Cwp"]
    lcb_pct: float = KVLCC2_MODEL_PARTICULARS["lcb_pct"]
    u_design: float = KVLCC2_FN_DESIGN * math.sqrt(GRAVITY * KVLCC2_MODEL_LENGTH)
    rho: float = RHO_MODEL
    nu: float = NU_MODEL
    g: float = GRAVITY

    @property
    def froude(self) -> float:
        return self.u_design / math.sqrt(self.g * self.length)

    @property
    def reynolds(self) -> float:
        """Hull Reynolds number at the design speed (workshop value 7.23e6)."""
        return self.u_design * self.length / self.nu

    @property
    def displaced_volume(self) -> float:
        return self.block_coefficient * self.length * self.beam * self.draft

    @property
    def mass(self) -> float:
        """Displacement mass [kg] (model scale)."""
        return self.rho * self.displaced_volume


DEFAULT_CONDITIONS = Kvlcc2Conditions()


def build_hull(verify: bool = True) -> Kvlcc2Hull:
    """Build the repository's calibrated KVLCC2 hull."""
    return build_kvlcc2_hull(verify=verify)


# --------------------------------------------------------------------------- #
# geometry gate
# --------------------------------------------------------------------------- #
def geometry_gate(hull: Optional[Kvlcc2Hull] = None, n: int = 4001) -> Dict[str, object]:
    """Integrate the hull and compare with the published coefficients.

    Returns ``{"passed": bool, "checks": {...}}``.  The tolerances are the ones
    the platform uses for every hull: ``Cb`` within 0.5 %, ``Cwp`` within 0.5 %,
    ``LCB`` within 0.05 % L, ``L/B/T`` exact.
    """
    hull = hull or build_hull()
    c = hull.coefficients(n=n)
    checks = {
        "Cb": (c["Cb"], float(hull.Cb), 0.5),
        "Cwp": (c["Cwp"], float(hull.Cwp), 0.5),
        "Cm": (c["Cm"], float(hull.Cm), 0.5),
        "LCB_pct": (c["LCB_pct"], float(hull.lcb_pct), None),
    }
    rows = {}
    passed = True
    for key, (value, target, tol_pct) in checks.items():
        if tol_pct is None:
            deviation = abs(value - target)
            ok = deviation <= 0.05
            rows[key] = {
                "computed": float(value), "published": target,
                "deviation": float(deviation), "tolerance": 0.05,
                "unit": "percent_of_L", "passed": bool(ok),
            }
        else:
            deviation = 100.0 * (value - target) / target
            ok = abs(deviation) <= tol_pct
            rows[key] = {
                "computed": float(value), "published": target,
                "deviation_pct": float(deviation), "tolerance_pct": tol_pct,
                "passed": bool(ok),
            }
        passed = passed and bool(ok)
    rows["L"] = {"computed": float(hull.L), "published": 7.0, "passed": True}
    rows["B"] = {"computed": float(hull.B), "published": float(hull.B), "passed": True}
    rows["T"] = {"computed": float(hull.T), "published": float(hull.T), "passed": True}
    rows["volume"] = {"computed": float(c["volume"]), "unit": "m^3"}
    return {"passed": bool(passed), "checks": rows}


# --------------------------------------------------------------------------- #
# lateral planform: what the cross-flow / lift model integrates over
# --------------------------------------------------------------------------- #
def section_data(hull: Optional[Kvlcc2Hull] = None, n: int = 241) -> Dict[str, np.ndarray]:
    """Lateral-planform section data along the hull length.

    ``x``      stations measured from the aft perpendicular, 0..L
    ``xi``     ``x/L``
    ``b``      waterline half breadth ``b(x)``
    ``draft``  *mean section depth* ``T_e(x) = a(x) / (2 b(x))``; equals ``T``
               over the parallel body and tapers at the fine ends, i.e. the
               effective span of the section in the horizontal-plane strip model
    ``area``   sectional area ``a(x)``
    ``side``   side area of the strip ``T_e dx`` summed to the hull side area
    """
    hull = hull or build_hull()
    xi = np.linspace(0.0, 1.0, n)
    x = xi * hull.L
    b = hull.half_breadth_shape(xi)
    a = hull.area_shape(xi)
    with np.errstate(divide="ignore", invalid="ignore"):
        draft = np.where(b > 1e-12, a / np.maximum(2.0 * b, 1e-12), 0.0)
    draft = np.clip(draft, 0.0, hull.T)
    side_area = float(np.trapezoid(draft, x))
    return {
        "x": x, "xi": xi, "half_breadth": b, "draft": draft,
        "area": a, "side_area": np.asarray(side_area),
        "volume": np.asarray(float(np.trapezoid(a, x))),
    }


# --------------------------------------------------------------------------- #
# rudder
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RudderGeometry:
    """Nominal KVLCC2 rudder geometry (model scale).

    ``area_ratio`` is the published rudder-area ratio ``A_R/(L T) = 0.0169``;
    the span/chord split uses the typical tanker rudder aspect ratio
    ``AR = h/c = 1.85``.  The section is a 0.18-thickness symmetric profile
    (NACA-4-digit parameterisation), which is the usual KVLCC2 rudder
    idealisation -- *not* digitized horn-rudder offsets.
    """

    length: float = KVLCC2_MODEL_LENGTH
    draft: float = KVLCC2_MODEL_PARTICULARS["T"]
    area_ratio: float = 0.0169
    aspect_ratio: float = 1.85
    thickness_ratio: float = 0.18
    stock_from_le_chord: float = 0.55
    leading_edge_from_ap_chord: float = 0.0
    #: propeller diameter -- sets the slipstream fraction seen by the rudder
    propeller_diameter: float = 0.2157
    rho: float = RHO_MODEL
    nu: float = NU_MODEL

    @property
    def area(self) -> float:
        return self.area_ratio * self.length * self.draft

    @property
    def span(self) -> float:
        return math.sqrt(self.area * self.aspect_ratio)

    @property
    def chord(self) -> float:
        return math.sqrt(self.area / self.aspect_ratio)

    @property
    def stock_x_from_midship(self) -> float:
        """Rudder stock position measured from midship (aft negative)."""
        x_le = -0.5 * self.length + self.leading_edge_from_ap_chord * self.chord
        return x_le + self.stock_from_le_chord * self.chord

    @property
    def x_r_over_l(self) -> float:
        return self.stock_x_from_midship / self.length

    @property
    def slipstream_fraction(self) -> float:
        """``eta = (D_prop/h_R)^2`` -- share of the rudder span in the slipstream."""
        return min(1.0, (self.propeller_diameter / self.span) ** 2)

    def reynolds(self, u: float) -> float:
        return u * self.chord / self.nu

    def as_dict(self) -> Dict[str, float]:
        return {
            "area_m2": self.area, "span_m": self.span, "chord_m": self.chord,
            "aspect_ratio": self.aspect_ratio, "area_ratio_AR_over_LT": self.area_ratio,
            "thickness_ratio": self.thickness_ratio,
            "x_r_over_l": self.x_r_over_l,
            "slipstream_fraction": self.slipstream_fraction,
            "reynolds_design": self.reynolds(DEFAULT_CONDITIONS.u_design),
        }


def rudder_geometry(**overrides) -> RudderGeometry:
    return RudderGeometry(**overrides)