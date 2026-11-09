"""Model-scale resistance chain used by the self-propulsion and manoeuvring solves.

Composition (ITTC-78 convention, one convention only -- never both)
-------------------------------------------------------------------
::

    C_T = (1 + k) C_F(Re) + dCf + C_W
    R_T = C_T * 1/2 rho U^2 S

* ``C_F``  ITTC-1957 line ``0.075/(log10 Re - 2)^2``
  (:func:`tensorfvm.resistance_chain.ittc57_cf`);
* ``(1+k)`` Holtrop-Mennen (1982) form factor regression
  (:func:`tensorfvm.resistance_chain.holtrop_form_factor`);
* ``dCf``  ITTC-1978 roughness allowance (zero for a hydraulically smooth model
  surface -- the smooth-model regime is the default here, since the chain is
  evaluated at towing-tank scale);
* ``C_W``  wave/residual term.  It is taken as the positive residual of the KVLCC2
  resistance anchor ``C_T(Fn=0.142)`` over the viscous chain **when the anchor is
  larger than the chain**; when it is not (which is the case here, see below) the
  wave term is set to zero and the shortfall is recorded as a flag.

Audit finding recorded with this module
---------------------------------------
At model scale the chain gives ``(1+k) C_F = 4.29e-3`` for KVLCC2, while the
repository's KVLCC2 resistance anchor is ``C_T = 4.00e-3`` (band ``3.90-4.10e-3``,
``provenance="withheld"`` -- a literature-recalled value, not a machine-verified
SIMMAN 2008 datum).  A wave term can therefore not be positive: either the recalled
anchor is ~8 % low, or the Holtrop form factor ``1+k = 1.35`` is high for this hull,
or the ITTC-57 line is not the law the anchor was reduced with.  The module
**reports the inconsistency instead of hiding it** (``flags`` in the result and the
``c_wave_negative_shortfall`` entry of the band dict), and the self-propulsion solve
is run with the chain-only resistance, with the anchor-implied resistance and the
resulting shaft-rate sensitivity reported alongside.

The chain is used inside the time integration of a manoeuvre, so it must be cheap:
the geometry-dependent pieces (wetted surface, form factor) are computed once and
cached, and ``C_T(u)`` is closed form thereafter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

from ..rans_ship_resistance.anchors import SIMMAN2008_KVLCC2
from ..rans_ship_resistance.kvlcc2 import KVLCC2_MODEL_PARTICULARS
from ..resistance_chain import (
    HullPrincipal, holtrop_form_factor, holtrop_wetted_surface, ittc57_cf,
    ittc78_delta_cf,
)
from .hull import Kvlcc2Conditions, DEFAULT_CONDITIONS, build_hull

__all__ = ["ResistanceComposition", "resistance_components",
           "model_scale_resistance", "cached_geometry"]

_PRINCIPAL = HullPrincipal(**KVLCC2_MODEL_PARTICULARS)
_GEOMETRY_CACHE: Dict[str, float] = {}


def cached_geometry(conditions: Optional[Kvlcc2Conditions] = None) -> Dict[str, float]:
    """Wetted surface and form factor (expensive; computed once per process)."""
    c = conditions or DEFAULT_CONDITIONS
    key = f"{c.length:.6f}|{c.beam:.6f}|{c.draft:.6f}|{c.block_coefficient:.6f}"
    if key not in _GEOMETRY_CACHE:
        hull = build_hull()
        _GEOMETRY_CACHE[key] = float(hull.wetted_surface())
        _GEOMETRY_CACHE[key + "|form"] = float(holtrop_form_factor(_PRINCIPAL))
        _GEOMETRY_CACHE[key + "|holtrop_S"] = float(holtrop_wetted_surface(_PRINCIPAL))
    return {"wetted_surface": _GEOMETRY_CACHE[key],
            "one_plus_k": _GEOMETRY_CACHE[key + "|form"],
            "wetted_surface_holtrop": _GEOMETRY_CACHE[key + "|holtrop_S"]}


@dataclass
class ResistanceComposition:
    """One operating point of the resistance chain (model scale)."""

    speed: float
    froude: float
    reynolds: float
    wetted_surface: float
    wetted_surface_holtrop: float
    cf: float
    one_plus_k: float
    delta_cf: float
    c_wave: float
    c_viscous: float
    c_total: float
    r_friction: float
    r_viscous: float
    r_allowance: float
    r_wave: float
    r_total: float
    roughness: float
    provenance: Dict[str, str] = field(default_factory=dict)
    band: Dict[str, float] = field(default_factory=dict)
    flags: list = field(default_factory=list)

    @property
    def R_T(self) -> float:
        return self.r_total

    def as_dict(self) -> Dict[str, object]:
        out = {k: v for k, v in self.__dict__.items()
               if isinstance(v, (int, float))}
        out["provenance"] = self.provenance
        out["band"] = self.band
        out["flags"] = list(self.flags)
        return out


def resistance_components(speed: Optional[float] = None,
                          conditions: Optional[Kvlcc2Conditions] = None,
                          roughness: Optional[float] = None,
                          hull=None) -> ResistanceComposition:
    """Evaluate the ITTC-78 chain at ``speed`` (default: design speed)."""
    c = conditions or DEFAULT_CONDITIONS
    u = c.u_design if speed is None else float(speed)
    if u <= 0.0:
        raise ValueError("speed must be positive")
    geom = cached_geometry(c)
    surface_geom = geom["wetted_surface"]
    surface_holtrop = geom["wetted_surface_holtrop"]
    one_plus_k = geom["one_plus_k"]
    re = u * c.length / c.nu
    cf = float(ittc57_cf(re))
    ks = 0.0 if roughness is None else float(roughness)
    dcf = float(ittc78_delta_cf(ks, c.length)) if ks > 0.0 else 0.0
    q = 0.5 * c.rho * u * u * surface_geom
    c_viscous = one_plus_k * cf
    # --- wave term from the (withheld) resistance anchor
    anchor = SIMMAN2008_KVLCC2
    re_design = c.u_design * c.length / c.nu
    c_chain_design = one_plus_k * float(ittc57_cf(re_design))
    c_wave = anchor.c_t - c_chain_design
    flags = []
    if c_wave > 0.0:
        pass
    else:
        flags.append(
            "anchor C_T below the viscous chain: wave term clamped to zero "
            f"(anchor {anchor.c_t:.3e} vs chain {c_chain_design:.3e})")
        c_wave = 0.0
    r_friction = cf * q
    r_viscous = c_viscous * q
    r_allowance = dcf * q
    r_wave = c_wave * q
    r_total = r_viscous + r_allowance + r_wave
    c_total = r_total / q
    provenance = {
        "cf": "REAL (ITTC-1957 line)",
        "form_factor": "EMPIRICAL (Holtrop-Mennen 1982 regression)",
        "roughness_allowance": ("zero (hydraulically smooth model surface)"
                                if ks == 0.0 else f"ITTC-1978, ks={ks:.3g} m"),
        "wave_term": ("WITHHELD: residual of the KVLCC2 anchor C_T "
                      f"{anchor.c_t:.3e} (provenance={anchor.provenance}); "
                      "clamped to zero when negative"),
        "wetted_surface": "REAL (hull integral); Holtrop regression shown for scale",
    }
    band = {
        "c_t_anchor": anchor.c_t,
        "c_t_anchor_low": float(anchor.c_t_band[0]),
        "c_t_anchor_high": float(anchor.c_t_band[1]),
        "c_chain_design": c_chain_design,
        "c_wave_negative_shortfall": max(0.0, c_chain_design - anchor.c_t),
        "wetted_surface_spread_pct": 100.0 * (surface_holtrop - surface_geom)
        / surface_geom,
    }
    return ResistanceComposition(
        speed=u, froude=u / math.sqrt(c.g * c.length), reynolds=re,
        wetted_surface=surface_geom, wetted_surface_holtrop=surface_holtrop,
        cf=cf, one_plus_k=one_plus_k, delta_cf=dcf, c_wave=c_wave,
        c_viscous=c_viscous, c_total=c_total,
        r_friction=r_friction, r_viscous=r_viscous, r_allowance=r_allowance,
        r_wave=r_wave, r_total=r_total, roughness=ks,
        provenance=provenance, band=band, flags=flags)


def model_scale_resistance(speed_ratio: float,
                           conditions: Optional[Kvlcc2Conditions] = None,
                           hull=None) -> Dict[str, float]:
    """``R_T`` at ``speed_ratio * U_design`` -- closed form, cached geometry.

    ``C_W`` is held constant over the (small) speed excursion of a manoeuvre, so
    ``R_T(u) = [(1+k) C_F(Re(u)) + dCf + C_W] 1/2 rho u^2 S``.
    """
    c = conditions or DEFAULT_CONDITIONS
    geom = cached_geometry(c)
    u = abs(float(speed_ratio)) * c.u_design
    if u <= 0.0:
        return {"R_T": 0.0, "C_T": 0.0, "speed": 0.0, "Re": 0.0}
    re = u * c.length / c.nu
    # the ITTC-57 law is only defined above Re = 5e5; at the (very low) speeds
    # reached in a hard turn the resistance is negligible, so the law is frozen at
    # its lower limit rather than allowed to raise
    cf = float(ittc57_cf(max(re, 5.5e5)))
    anchor = SIMMAN2008_KVLCC2
    c_chain_design = geom["one_plus_k"] * float(ittc57_cf(
        c.u_design * c.length / c.nu))
    c_wave = max(anchor.c_t - c_chain_design, 0.0)
    c_t = geom["one_plus_k"] * cf + c_wave
    r_t = c_t * 0.5 * c.rho * u * u * geom["wetted_surface"]
    return {"R_T": r_t, "C_T": c_t, "speed": u, "Re": re}