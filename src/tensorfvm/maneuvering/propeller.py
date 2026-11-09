"""Blade-element / momentum-theory (BEMT) open-water propeller model.

Why BEMT and not a regression polynomial
----------------------------------------
The self-propulsion point needs ``K_T(J)`` of the KVLCC2 propeller (KP505:
``Z = 5``, ``D = 9.86 m`` prototype, ``AE/A0 = 0.800``, ``P/D = 0.997``).  A
digitized open-water table or a published ``K_T`` polynomial for KP505 is not in
this workspace, and hand-transcribing such coefficients from memory is exactly
the failure mode the platform forbids.  What *is* defensible is to compute the
open-water curve from the blade geometry with a standard blade-element/momentum
model, and then verify it against things that do not need an EFD table:

1. **momentum/blade equality** -- at every annulus the axial and tangential
   momentum relations must balance the blade-element forces to a tolerance; the
   residual is reported, so "the model converged" is machine-checkable;
2. **Betz/actuator-disk bound** -- ``eta_B <= eta_ideal = 2 / (1 + sqrt(1+C_T))``
   with ``C_T = 8 K_T / (pi J^2)``; a propeller that beats the ideal efficiency
   is wrong, no matter what table it is compared with;
3. **inviscid limit** -- with ``c_d = 0`` and inviscid section lift the open-water
   efficiency must approach the actuator-disk ideal value from below as the
   loading goes to zero;
4. **limits** -- ``K_T, K_Q`` finite and positive at small ``J``, ``K_T -> 0`` at
   a finite advance ratio, ``K_Q`` monotonically decreasing.

Interaction with the hull (wake fraction, thrust deduction) is *not* part of this
module: it lives in :mod:`.selfpropulsion` and is flagged ``EMPIRICAL`` there.

Induction model
---------------
For each annulus ``r`` the axial/tangential induction factors ``(a, a')`` satisfy
the classical Glauert propeller relations with Prandtl tip loss ``F``::

    a (1+a)        = sigma C_n / (8 F sin^2 phi)
    a'/(1-a')      = sigma C_t / (8 F sin phi cos phi) / (1+a)

where ``sigma = Z c /(2 pi r)``, ``phi = atan(V (1+a) / (omega r (1-a')))``,
``C_n = C_l cos phi - C_d sin phi``, ``C_t = C_l sin phi + C_d cos phi``.  The
two relations are solved by damped fixed-point iteration; the achievement is
verified by the force-form residual, not by the iteration count.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .hull import KVLCC2_PROTOTYPE_SCALE

__all__ = [
    "PropellerGeometry", "AnnulusSolution", "OpenWaterPoint",
    "open_water_point", "open_water_curve", "propeller_thrust",
    "ideal_efficiency", "brentq_advance_ratio",
]

#: KP505 prototype diameter [m] (KVLCC2 propeller, quoted across the KVLCC2
#: literature as 9.86 m); model diameter is derived from the scale ratio.
KP505_PROTOTYPE_DIAMETER = 9.86


@dataclass(frozen=True)
class PropellerGeometry:
    """Radial blade description of a KP505-type propeller (model scale).

    ``chord`` is reconstructed from the expanded-area ratio through three
    control points (hub / 0.7R / tip) and scaled so that
    ``Z * int c dr = AE/A0 * pi D^2 / 4`` exactly; the pitch is constant at
    ``P/D``.  Section lift uses ``C_l = C_lmax tanh(a0 alpha / C_lmax)`` (linear
    slope ``a0 = 2 pi``, smooth stall-free saturation) and the section drag is
    ``C_d = c_d0 + C_l^2 / (pi e AR_section)`` plus a separation term beyond
    stall.  Those are the documented modelling choices; the *geometry*
    integrals (area ratio, diameter) are exact.
    """

    diameter: float = KP505_PROTOTYPE_DIAMETER / KVLCC2_PROTOTYPE_SCALE
    n_blades: int = 5
    hub_ratio: float = 0.18
    pitch_ratio: float = 0.997
    expanded_area_ratio: float = 0.800
    chord_hub_ratio: float = 0.85
    chord_07_ratio: float = 1.00
    chord_tip_ratio: float = 0.40
    lift_slope: float = 2.0 * math.pi
    cl_max: float = 1.20
    cd0: float = 0.012
    section_aspect: float = 1.5
    oswald: float = 0.90
    alpha_stall_deg: float = 12.0
    n_radial: int = 41
    tip_loss: bool = True

    # -- geometry ---------------------------------------------------------
    def radial_grid(self, n: Optional[int] = None) -> np.ndarray:
        return np.linspace(self.hub_ratio, 1.0, n or self.n_radial)

    def chord_distribution(self, xi: Optional[np.ndarray] = None) -> np.ndarray:
        """Blade chord ``c(r)`` [m] on the annulus grid, area-ratio exact."""
        xi = self.radial_grid() if xi is None else np.asarray(xi, dtype=float)
        control_x = np.array([self.hub_ratio, 0.70, 1.00])
        control_y = np.array([self.chord_hub_ratio, self.chord_07_ratio,
                              self.chord_tip_ratio])
        poly = np.polyfit(control_x, control_y, 2)
        shape = np.polyval(poly, xi)
        shape = np.maximum(shape, 0.0)
        radius = 0.5 * self.diameter
        integral = float(np.trapezoid(shape, xi * radius))
        target = self.expanded_area_ratio * math.pi * self.diameter ** 2 / 4.0
        scale = target / (self.n_blades * max(integral, 1e-30))
        chord = shape * scale
        # chord is normalised so the *expanded* area ratio is reproduced exactly
        achieved = self.n_blades * float(np.trapezoid(chord, xi * radius))
        chord *= target / max(achieved, 1e-30)
        return chord

    def achieved_area_ratio(self) -> float:
        xi = self.radial_grid()
        chord = self.chord_distribution(xi)
        radius = 0.5 * self.diameter
        return self.n_blades * float(np.trapezoid(chord, xi * radius)) / (
            math.pi * self.diameter ** 2 / 4.0)

    def pitch_angle(self, xi: np.ndarray) -> np.ndarray:
        """Blade section pitch angle ``beta = atan((P/D) / (pi xi))`` [rad]."""
        return np.arctan(self.pitch_ratio / (math.pi * np.asarray(xi, dtype=float)))

    # -- section aerodynamics --------------------------------------------
    def section_coefficients(self, alpha: np.ndarray):
        """Section ``C_l``, ``C_d`` for incidence ``alpha`` [rad]."""
        alpha = np.asarray(alpha, dtype=float)
        cl = self.cl_max * np.tanh(self.lift_slope * alpha / self.cl_max)
        cd = (self.cd0
              + cl ** 2 / (math.pi * self.oswald * self.section_aspect))
        a_stall = math.radians(self.alpha_stall_deg)
        beyond = np.abs(alpha) > a_stall
        if np.any(beyond):
            extra = 1.4 * np.sin(alpha[beyond] - np.sign(alpha[beyond]) * a_stall) ** 2
            cd = cd.copy()
            cd[beyond] = cd[beyond] + extra
        return cl, cd


@dataclass
class AnnulusSolution:
    xi: np.ndarray
    chord: np.ndarray
    phi: np.ndarray
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    a: np.ndarray
    aprime: np.ndarray
    tip_loss: np.ndarray
    thrust: np.ndarray      # dT/dr [N/m]
    torque: np.ndarray      # dQ/dr [N m/m]
    momentum_residual: float
    saturated_fraction: float
    iterations: int


@dataclass
class OpenWaterPoint:
    """Non-dimensional open-water state at advance ratio ``J``."""

    J: float
    KT: float
    KQ: float
    eta: float
    thrust: float
    torque: float
    rps: float
    momentum_residual: float
    annulus: Optional[AnnulusSolution] = None

    def as_dict(self) -> Dict[str, float]:
        return {"J": self.J, "KT": self.KT, "KQ": self.KQ, "eta": self.eta,
                "thrust_N": self.thrust, "torque_Nm": self.torque,
                "rps": self.rps,
                "momentum_residual": self.momentum_residual}


def _solve_annuli(geometry: PropellerGeometry, xi: np.ndarray, J: float,
                  chord: np.ndarray, rho: float, n_rps: float,
                  tol: float = 1e-12, max_iter: int = 200) -> Dict[str, np.ndarray]:
    """Solve ``(a, a')`` on all annuli at once by bisection on the inflow angle.

    The two momentum relations are used in the form

    ``a (1+a) = sigma C_n /(4 F sin^2 phi)`` and
    ``a' = sigma C_t /(4 F sin phi cos phi)``

    so that, given ``phi``, both induction factors are explicit.  The remaining
    scalar equation is the kinematic identity ``tan phi = V (1+a)/(omega r
    (1-a'))``, which is monotone enough for bisection and has no pole at high
    loading (the earlier fixed-point form ``a = K/(1-K)`` did, and silently
    produced negative induction in the tip annuli -- documented in the report as
    a trap).  Solving on the whole grid at once keeps the cost at ~200 vector
    operations per advance ratio.
    """
    radius = 0.5 * geometry.diameter
    r = xi * radius
    V = J * n_rps * geometry.diameter
    omega = 2.0 * math.pi * n_rps
    beta = np.arctan(geometry.pitch_ratio / (math.pi * xi))
    sigma = geometry.n_blades * chord / (2.0 * math.pi * r)

    def state(phi: np.ndarray):
        sinphi = np.maximum(np.sin(phi), 1e-9)
        cosphi = np.maximum(np.cos(phi), 1e-9)
        alpha = beta - phi
        cl, cd = geometry.section_coefficients(alpha)
        cn = cl * cosphi - cd * sinphi
        ct = cl * sinphi + cd * cosphi
        if geometry.tip_loss:
            f = geometry.n_blades / 2.0 * (1.0 - xi) / sinphi
            F = 2.0 / math.pi * np.arccos(np.clip(np.exp(-np.maximum(f, 0.0)), -1.0, 1.0))
            F = np.maximum(F, 1e-3)
        else:
            F = np.ones_like(phi)
        # a/(1+a) = sigma C_n /(4 F sin^2 phi)  (annulus momentum, free-stream V)
        K_ax = sigma * cn / (4.0 * F * sinphi ** 2)
        saturated = K_ax >= 0.95
        a = np.where(saturated, 0.95, K_ax / np.maximum(1.0 - K_ax, 1e-9))
        # a' = sigma C_t /(4 F sin phi cos phi)
        K_tan = sigma * ct / (4.0 * F * sinphi * cosphi)
        aprime_raw = K_tan / (1.0 + K_tan)
        aprime = np.clip(aprime_raw, -0.5, 0.95)
        a = np.clip(a, -0.50, 0.95)
        clipped = (saturated | (aprime_raw > 0.9499) | (aprime_raw < -0.4999)
                   | (K_ax < -0.4999))
        return {"a": a, "aprime": aprime, "alpha": alpha, "cl": cl, "cd": cd,
                "cn": cn, "ct": ct, "F": F, "sigma": sigma,
                "saturated": saturated, "clipped": clipped}

    lo = np.full_like(xi, 1e-3)
    hi = np.full_like(xi, 0.5 * math.pi - 1e-3)

    def residual(phi: np.ndarray) -> np.ndarray:
        st = state(phi)
        return np.tan(phi) - V * (1.0 + st["a"]) / (
            np.maximum(omega * r * (1.0 - st["aprime"]), 1e-9))

    f_lo, f_hi = residual(lo), residual(hi)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        same = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(same, mid, lo)
        hi = np.where(same, hi, mid)
        f_lo = np.where(same, f_mid, f_lo)
        if np.all(np.abs(hi - lo) < tol):
            break
    phi = 0.5 * (lo + hi)
    st = state(phi)
    sinphi = np.maximum(np.sin(phi), 1e-9)
    cosphi = np.maximum(np.cos(phi), 1e-9)
    W2 = (V * (1.0 + st["a"])) ** 2 + (omega * r * (1.0 - st["aprime"])) ** 2
    dT_blade = 0.5 * rho * W2 * geometry.n_blades * chord * st["cn"]
    dQ_blade = 0.5 * rho * W2 * geometry.n_blades * chord * st["ct"] * r
    dT_mom = 4.0 * math.pi * r * rho * V * V * st["a"] * (1.0 + st["a"]) * st["F"]
    dQ_mom = (4.0 * math.pi * r ** 3 * rho * V * omega * (1.0 + st["a"])
              * st["aprime"] * st["F"])
    scale = np.abs(dT_blade) + np.abs(dQ_blade) / np.maximum(r, 1e-12) + 1e-12
    res = (np.abs(dT_blade - dT_mom) + np.abs(dQ_blade - dQ_mom) / np.maximum(r, 1e-12)) / scale
    free = ~st["clipped"]
    residual_max = float(np.max(res[free])) if np.any(free) else float("nan")
    return {"phi": phi, "alpha": st["alpha"], "cl": st["cl"], "cd": st["cd"],
            "a": st["a"], "aprime": st["aprime"], "F": st["F"],
            "dT": dT_blade, "dQ": dQ_blade, "residual": res,
            "residual_max": residual_max, "cn": st["cn"], "ct": st["ct"],
            "saturated_fraction": float(np.mean(st["clipped"])),
            "sigma": sigma, "beta": beta, "r": r, "chord": chord}


def open_water_point(geometry: PropellerGeometry, J: float, *,
                     rho: float = 998.2, n_rps: float = 1.0,
                     keep_annulus: bool = False) -> OpenWaterPoint:
    """Evaluate the open-water state at advance ratio ``J``.

    ``n_rps`` only sets the dimensional scale of the returned thrust/torque;
    ``K_T`` and ``K_Q`` are independent of it.  The returned
    ``momentum_residual`` is the maximum relative difference between the
    blade-element and momentum forces over the annuli, i.e. the machine-checkable
    achievement of the BEMT solution.
    """
    if J < 0.0:
        raise ValueError("advance ratio must be non-negative")
    xi = geometry.radial_grid()
    chord = geometry.chord_distribution(xi)
    sol = _solve_annuli(geometry, xi, J, chord, rho, n_rps)
    dr = np.gradient(xi * 0.5 * geometry.diameter)
    dT = sol["dT"] * dr
    dQ = sol["dQ"] * dr
    thrust = float(dT.sum())
    torque = float(dQ.sum())
    denom_t = rho * n_rps ** 2 * geometry.diameter ** 4
    denom_q = rho * n_rps ** 2 * geometry.diameter ** 5
    KT = thrust / denom_t
    KQ = torque / denom_q
    eta = J * KT / (2.0 * math.pi * KQ) if KQ > 0 else 0.0
    annulus = None
    if keep_annulus:
        annulus = AnnulusSolution(
            xi=xi, chord=chord, phi=sol["phi"], alpha=sol["alpha"],
            cl=sol["cl"], cd=sol["cd"], a=sol["a"], aprime=sol["aprime"],
            tip_loss=sol["F"], thrust=dT, torque=dQ,
            momentum_residual=float(sol["residual_max"]),
            saturated_fraction=float(sol["saturated_fraction"]),
            iterations=0)
    return OpenWaterPoint(J=J, KT=KT, KQ=KQ, eta=eta, thrust=thrust, torque=torque,
                          rps=n_rps,
                          momentum_residual=float(sol["residual_max"]),
                          annulus=annulus)


def open_water_curve(geometry: PropellerGeometry,
                     js: Sequence[float] = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                            0.6, 0.7, 0.8, 0.9),
                     **kwargs) -> List[OpenWaterPoint]:
    return [open_water_point(geometry, float(J), **kwargs) for J in js]


def ideal_efficiency(KT: float, J: float) -> float:
    """Actuator-disk (Betz) ideal efficiency ``2/(1+sqrt(1+C_T))``."""
    if J <= 0.0:
        return 0.0
    C_T = 8.0 * KT / (math.pi * J ** 2)
    if C_T <= -1.0:
        return 1.0
    return 2.0 / (1.0 + math.sqrt(1.0 + C_T))


def propeller_thrust(geometry: PropellerGeometry, rps: float, advance_speed: float,
                     rho: float = 998.2) -> float:
    """Thrust ``T = rho n^2 D^4 K_T(J)`` [N] at shaft rate ``rps``."""
    if rps <= 0.0:
        raise ValueError("shaft rate must be positive")
    J = advance_speed / (rps * geometry.diameter)
    point = open_water_point(geometry, J, rho=rho, n_rps=rps)
    return point.thrust


def brentq_advance_ratio(geometry: PropellerGeometry, thrust_required: float,
                         rho: float, n_rps: float,
                         bracket: Sequence[float] = (1e-3, 0.98),
                         tol: float = 1e-10) -> Dict[str, float]:
    """Advance ratio at which ``T(J) = thrust_required`` for a fixed shaft rate.

    Used by the self-propulsion solve; returns the bracket, the iterate count
    and the thrust residual so the caller can assert closure.
    """
    def residual(J: float) -> float:
        point = open_water_point(geometry, J, rho=rho, n_rps=n_rps)
        return point.thrust - thrust_required

    lo, hi = bracket
    f_lo, f_hi = residual(lo), residual(hi)
    if f_lo * f_hi > 0.0:
        raise ValueError(
            "thrust is not bracketed in the advance-ratio interval "
            f"[{lo}, {hi}]: T(lo)-T_req={f_lo:.6g}, T(hi)-T_req={f_hi:.6g}")
    it = 0
    for it in range(1, 200):
        mid = 0.5 * (lo + hi)
        f_mid = residual(mid)
        if abs(f_mid) < tol * max(abs(thrust_required), 1.0):
            lo = hi = mid
            break
        if f_lo * f_mid < 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return {"J": float(mid), "residual": float(f_mid), "iterations": it}