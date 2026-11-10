"""Cavity-collapse impact pressure and erosion-rate model.

What the model does
-------------------
A vapour cavity that closes on a rudder surface collapses against a liquid that
cannot get out of the way, so the wall sees a short pressure pulse whose
amplitude is set by the *collapse velocity* through the water-hammer relation

    p_impact = p_v + rho * c * |R_dot|                                     (1)

with ``c`` the speed of sound (``rho c`` = 1.5e6 Pa.s/m for water, so a 10 m/s
collapse velocity is ~15 MPa -- the order of magnitude of a cavitation pit).

``R_dot`` comes from the Rayleigh-Plesset equation for a spherical bubble,

    R R_ddot + 3/2 R_dot^2 = (p_v + p_g - p_inf(t)) / rho,                  (2)

which this module (a) integrates for a prescribed fluctuating pressure signal and
(b) cross-checks against the two closed-form results:

* **Rayleigh collapse time** (gas-free, constant pressure): with
  ``Delta p = p_inf - p_v > 0`` the interface follows
  ``R(t) = R_max (1 - t/T_c)^{2/5}`` with ``T_c = 0.915 R_max sqrt(rho/Delta p)``,
  which the RK4 integration reproduces to <0.5 %.
* **Energy integral / collapse velocity**: for the same gas-free collapse

      R_dot^2 = (2 Delta p / 3 rho) [ (R_max/R)^3 - 1 ]                    (3)

  so the velocity at the collapse radius is known exactly and is used to check
  the ODE and to define ``R_dot`` without integrating in production runs.

The erosion rate is then the classical impact-counting product

    E = k_e * f_impact * (p_impact - p_v)^n                                (4)

with the impact count rate ``f_impact = St * V / L_cav`` driven by cavity
shedding (``St ~ 0.25``) and the cavity length scaling
``L_cav = c * C_L sqrt(sigma_i/sigma - 1)``.  The exponent ``n`` is the erosion
"power law" exponent (2..3 in the cavitation-erosion literature); (4) is
*dimensionally consistent* and its log-log slope is ``n`` by construction, which
is asserted analytically in the tests.

Why ``E(sigma)`` peaks at an intermediate cavitation number
----------------------------------------------------------
Two mechanisms compete as ``sigma`` falls below ``sigma_i``:

* the collapse gets more violent -- ``(p_impact - p_v)^n`` grows;
* the cavity lengthens, the closure point moves away from the wall and the pulse
  reaching the surface is attenuated as ``exp(-d/d0)`` with ``d = a L_cav``.

The product is a maximum at ``sigma/sigma_i`` of order 1.2-2, which is the
trend reported for hydrofoil damage tests.  The model is therefore *not*
monotone, and a test pins the peak inside the documented band.

Limitations (repeated in the report)
------------------------------------
* single-bubble dynamics with an isothermal gas content; no bubble clouds,
  no shock-bubble interaction, no material response;
* the attenuation and ``k_e`` are lumped constants, so ``E`` is a **relative
  index** for design comparison (material choice, rudder angle, speed), not a
  predicted depth in mm/h;
* no experimental erosion-depth data exists in this workspace, so the EFD
  validation of ``E`` is ``WITHHELD`` and criterion (b) is evaluated as a
  cross-model/trend check (see :mod:`.reference`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

__all__ = [
    "RayleighCollapse", "rayleigh_collapse_time", "rayleigh_time_to_radius",
    "rayleigh_trajectory", "collapse_velocity",
    "integrate_rayleigh_plesset", "cavity_length", "shedding_frequency",
    "impact_pressure", "wall_impact_pressure", "erosion_rate", "erosion_index_vs_sigma",
    "erosion_exponent_fit", "erosion_depth", "MaterialErosionResistance",
]


# --------------------------------------------------------------------------- #
# single-bubble dynamics
# --------------------------------------------------------------------------- #
def rayleigh_collapse_time(r_max: float, delta_p: float, rho: float) -> float:
    """``T_c = 0.915 R_max sqrt(rho/Delta p)`` (Rayleigh, gas-free collapse)."""
    if delta_p <= 0.0:
        raise ValueError("need Delta p = p_inf - p_v > 0 for a collapse")
    return 0.915 * r_max * math.sqrt(rho / delta_p)


def integrate_rayleigh_plesset(r_max: float, delta_p: float, rho: float,
                               r_end: float = None, n_steps: int = 20000,
                               p_gas0: float = 0.0):
    """Integrate (2) for a gas-free (or ``p_gas0``-bearing) collapse.

    Returns ``(t, r, r_dot)`` arrays.  The collapse is stopped when the radius
    falls below ``r_end`` (default ``r_max/50``) or when the time exceeds 3 T_c.
    """
    if r_end is None:
        r_end = 0.02 * r_max
    t_c = rayleigh_collapse_time(r_max, delta_p, rho)
    t_max = 3.0 * t_c
    dt = t_max / n_steps

    def rhs(r, rdot):
        if r <= 0.0:
            return 0.0
        p_gas = p_gas0 * (r_max / r) ** 3 if p_gas0 else 0.0
        acc = (-delta_p + p_gas) / rho - 1.5 * rdot * rdot
        return acc / r

    ts, rs, vds = [0.0], [r_max], [0.0]
    r, rdot, t = r_max, 0.0, 0.0
    for _ in range(n_steps):
        k1r, k1v = rdot, rhs(r, rdot)
        k2r, k2v = rdot + 0.5 * dt * k1v, rhs(r + 0.5 * dt * k1r, rdot + 0.5 * dt * k1v)
        k3r, k3v = rdot + 0.5 * dt * k2v, rhs(r + 0.5 * dt * k2r, rdot + 0.5 * dt * k2v)
        k4r, k4v = rdot + dt * k3v, rhs(r + dt * k3r, rdot + dt * k3v)
        r = r + dt / 6.0 * (k1r + 2 * k2r + 2 * k3r + k4r)
        rdot = rdot + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt
        ts.append(t), rs.append(r), vds.append(rdot)
        if r <= r_end:
            break
    return np.array(ts), np.array(rs), np.array(vds)


def rayleigh_time_to_radius(r_target: float, r_max: float, delta_p: float,
                           rho: float, n: int = 40001) -> float:
    """Exact time for a gas-free bubble to collapse from ``r_max`` to ``r_target``.

    From the energy integral (3), ``dt = -sqrt(3 rho / (2 Delta p)) R^{3/2}
    (r_max^3 - R^3)^{-1/2} dR``, hence

        t(R) = T_c J(x)/J(1),   x = R/r_max,   J(x) = int_0^x u^{3/2}/sqrt(1-u^3) du

    with ``T_c`` from :func:`rayleigh_collapse_time`.  This is the reference the
    RK4 integration of :func:`integrate_rayleigh_plesset` is checked against --
    *not* the asymptotic ``R = r_max (1-t/Tc)^{2/5}``, which is only the leading
    near-collapse behaviour (it is ~25 % low at ``R/r_max = 0.15``).
    """
    if not 0.0 < r_target <= r_max:
        raise ValueError("need 0 < r_target <= r_max")
    t_c = rayleigh_collapse_time(r_max, delta_p, rho)

    def integral(x: float) -> float:
        if x <= 0.0:
            return 0.0
        u = np.linspace(0.0, x, n)
        f = np.where(u < 1.0, u ** 1.5 / np.sqrt(np.clip(1.0 - u ** 3, 1e-300, None)), 0.0)
        return float(np.trapezoid(f, u))

    return t_c * (1.0 - integral(r_target / r_max) / integral(1.0))


def rayleigh_trajectory(r_max: float, delta_p: float, rho: float,
                        n: int = 400) -> tuple:
    """``(t, R)`` of the exact gas-free collapse (quadrature reference curve)."""
    u = np.linspace(0.0, 1.0 - 1e-9, 40001)
    f = np.where(u < 1.0, u ** 1.5 / np.sqrt(np.clip(1.0 - u ** 3, 1e-300, None)), 0.0)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(u))])
    t_c = rayleigh_collapse_time(r_max, delta_p, rho)
    tt = t_c * (1.0 - cum / cum[-1])
    idx = np.linspace(0, u.size - 1, n).astype(int)
    return tt[idx], r_max * u[idx]


def collapse_velocity(r_max: float, r: float, delta_p: float, rho: float) -> float:
    """``|R_dot|`` from the energy integral (3), zero at ``r = r_max``."""
    if not 0.0 < r <= r_max:
        raise ValueError("collapse radius must satisfy 0 < r <= r_max")
    return math.sqrt(max(0.0, 2.0 * delta_p / (3.0 * rho) * ((r_max / r) ** 3 - 1.0)))


@dataclass(frozen=True)
class RayleighCollapse:
    """A single collapse event: initial radius, driving pressure and output."""

    r_max: float            # m
    delta_p: float          # Pa, p_inf - p_v
    rho: float = 1025.0     # kg/m^3
    sound_speed: float = 1470.0
    p_v: float = 1705.0     # Pa

    @property
    def time(self) -> float:
        return rayleigh_collapse_time(self.r_max, self.delta_p, self.rho)

    def collapse_velocity(self, r_min: float) -> float:
        return collapse_velocity(self.r_max, r_min, self.delta_p, self.rho)

    def impact_pressure(self, r_min: float) -> float:
        """Water-hammer pressure ``p_v + rho c |R_dot|`` at ``r = r_min``."""
        return self.p_v + self.rho * self.sound_speed * self.collapse_velocity(r_min)


# --------------------------------------------------------------------------- #
# cavity scaling and impact counting
# --------------------------------------------------------------------------- #
def cavity_length(sigma: float, sigma_i: float, chord: float,
                  c_length: float = 0.8) -> float:
    """Cavity length ``L = c C_L sqrt(sigma_i/sigma - 1)`` [m].

    The square-root scaling is the classical result for the length of a closing
    cavity behind a body; ``C_L = 0.8`` is a lumped constant of order one
    (``QUALITATIVE`` provenance, see :mod:`.reference`).  Returns 0 when
    ``sigma >= sigma_i`` (no cavity).
    """
    if sigma <= 0.0:
        return float(chord)      # fully supercavitating limit
    x = sigma_i / sigma - 1.0
    if x <= 0.0:
        return 0.0
    return float(c_length * chord * math.sqrt(x))


def shedding_frequency(v: float, length: float, strouhal: float = 0.25) -> float:
    """Cavity shedding / collapse rate ``f = St V / L`` [1/s]."""
    if length <= 0.0:
        raise ValueError("cavity length must be positive")
    return float(strouhal * v / length)


def impact_pressure(delta_p: float, rho: float = 1025.0, sound_speed: float = 1470.0,
                    growth_ratio: float = 12.0, p_v: float = 1705.0) -> float:
    """Wall impact pressure of a collapse driven by ``delta_p`` [Pa].

    The collapse ratio ``R_max/R_min`` is the standard cavitation value
    ``growth_ratio ~ 10-15``; the resulting ``|R_dot|`` from (3) is turned into a
    pressure with the water-hammer relation (1).
    """
    if delta_p <= 0.0:
        return 0.0
    r_dot = math.sqrt(2.0 * delta_p / (3.0 * rho) * (growth_ratio ** 3 - 1.0))
    return float(p_v + rho * sound_speed * r_dot)


def wall_impact_pressure(delta_p: float, cavity_len: float, chord: float,
                         rho: float = 1025.0, p_v: float = 1705.0,
                         sound_speed: float = 1470.0, growth_ratio: float = 12.0,
                         attenuation: float = 1.0) -> float:
    """Pressure pulse reaching the wall at the cavity-closure collapse [Pa].

    Two competing cavity-length effects are lumped here, both of them standard
    arguments of cavitation-erosion modelling:

    * **cloud amplification** -- the pulse comes from the collapse of a vapour
      *cloud* whose mass grows with the cavity, so the pressure scales with the
      non-dimensional cavity length: ``x`` (linear, ``x = L/c``);
    * **wall distance** -- the closure point moves downstream as the cavity
      lengthens, so the pulse is attenuated geometrically: ``exp(-attenuation x)``.

        p_wall - p_v = (p_impact(Delta p) - p_v) * x * exp(-attenuation x)      (5)

    The product ``x exp(-a x)`` has its maximum at ``x = 1/a``, which is what puts
    the erosion maximum at an intermediate cavitation number instead of at
    inception.
    """
    if delta_p <= 0.0 or cavity_len <= 0.0:
        return 0.0
    x = cavity_len / chord
    base = impact_pressure(delta_p, rho=rho, sound_speed=sound_speed,
                           growth_ratio=growth_ratio, p_v=p_v) - p_v
    return float(base * x * math.exp(-attenuation * x))


def erosion_rate(sigma: float, sigma_i: float, chord: float, v: float,
                 exponent: float = 2.5, k_e: float = 1.0,
                 rho: float = 1025.0, p_v: float = 1705.0,
                 sound_speed: float = 1470.0,
                 c_length: float = 0.8, strouhal: float = 0.25,
                 growth_ratio: float = 12.0, attenuation: float = 1.0,
                 threshold_pressure: float = 0.0) -> float:
    """Relative erosion index ``E`` [arbitrary units] at one operating point.

    ``E = k_e f_impact (p_wall - p_thr)^n``: the number of collapse events per
    unit time times the erosion work of one event above the material's threshold
    pressure, i.e. the standard impact-counting form of the erosion-rate law
    with ``p_wall`` from :func:`wall_impact_pressure`.  Returns 0 when the
    section does not cavitate (``sigma >= sigma_i``).
    """
    if sigma >= sigma_i:
        return 0.0
    q = 0.5 * rho * v * v
    delta_p = (sigma_i - sigma) * q
    if delta_p <= 0.0:
        return 0.0
    length = cavity_length(sigma, sigma_i, chord, c_length)
    p_wall = wall_impact_pressure(delta_p, length, chord, rho=rho, p_v=p_v,
                                  sound_speed=sound_speed, growth_ratio=growth_ratio,
                                  attenuation=attenuation)
    excess = p_wall - threshold_pressure
    if excess <= 0.0:
        return 0.0
    # the shedding rate saturates: below ~0.1 c chord the cavity is a thin sheet
    # and the shedding frequency cannot exceed St V / (0.1 c)
    f_imp = shedding_frequency(v, max(length, 0.1 * chord), strouhal)
    return float(k_e * f_imp * excess ** exponent)


def erosion_index_vs_sigma(sigmas: Sequence[float], sigma_i: float, chord: float,
                           v: float, **kwargs) -> np.ndarray:
    """Vectorised :func:`erosion_rate` over a cavitation-number sweep."""
    return np.array([erosion_rate(s, sigma_i, chord, v, **kwargs) for s in sigmas])


def erosion_exponent_fit(sigma: float, sigma_i: float, chord: float, v: float,
                         exponent: float = 2.5, **kwargs):
    """Recover the power-law exponent ``n`` from ``E`` vs ``p_wall - p_thr``.

    Returns ``(n_fit, r_squared)`` from a least-squares fit of
    ``log E = n log(p_wall) + const`` over a sweep of ``sigma``.  It must return
    the ``exponent`` that was passed in: that is the machine check that (4) is
    implemented exactly and that the reported ``n`` really is the model's
    exponent rather than a fitted decoration.
    """
    rho = kwargs.get("rho", 1025.0)
    q = 0.5 * rho * v * v
    sig = np.linspace(0.2 * sigma_i, 0.995 * sigma_i, 400)
    p_vals, e_vals = [], []
    for s in sig:
        length = cavity_length(s, sigma_i, chord, kwargs.get("c_length", 0.8))
        p_wall = wall_impact_pressure((sigma_i - s) * q, length, chord,
                                      rho=rho, p_v=kwargs.get("p_v", 1705.0),
                                      attenuation=kwargs.get("attenuation", 1.0))
        e = erosion_rate(s, sigma_i, chord, v, exponent=exponent, **kwargs)
        if e > 0.0 and p_wall > 0.0:
            # divide out the impact-frequency prefactor: E/f_imp = (p_wall - p_thr)^n
            f_imp = shedding_frequency(v, max(length, 0.1 * chord),
                                       kwargs.get("strouhal", 0.25))
            p_vals.append(p_wall)
            e_vals.append(e / f_imp)
    x, y = np.log(np.array(p_vals)), np.log(np.array(e_vals))
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else 1.0
    return float(slope), float(r2)


# --------------------------------------------------------------------------- #
# material side
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MaterialErosionResistance:
    """Lumped material resistance for the erosion *index* (design comparison).

    ``hardness`` is the Vickers hardness [HV] and ``resistance`` an empirical
    multiplier: erosion resistance of ductile metals scales roughly with
    hardness, so the relative index is ``E * (H_ref/H)``.  The numbers are
    ``EMPIRICAL`` and used only for ranking candidate rudder materials.
    """

    name: str
    hardness_hv: float
    resistance: float = 1.0

    def relative_erosion(self, e_index: float, reference_hv: float = 130.0) -> float:
        """``E * (H_ref/H) / resistance`` -- smaller is better."""
        return float(e_index * (reference_hv / self.hardness_hv) / self.resistance)


def erosion_depth(rate: float, t: float, incubation: float = 0.0) -> float:
    """Cumulative depth proxy ``h = rate (t - t_inc)`` for ``t > t_inc``."""
    if t <= incubation:
        return 0.0
    return float(rate * (t - incubation))


def time_to_threshold(rate: float, threshold: float, incubation: float = 0.0) -> float:
    """Exposure time needed to reach a depth threshold (``inf`` if no erosion)."""
    if rate <= 0.0:
        return math.inf
    return float(incubation + threshold / rate)


def erosion_history(rate: float, times: Sequence[float], incubation: float = 0.0):
    """Vectorised :func:`erosion_depth`."""
    return np.array([erosion_depth(rate, t, incubation) for t in times])


def spectrum_weighted_impact(p_rms: float, f_peak: float, n_harmonics: int = 8):
    """Impact amplitude from a *pressure-spectrum* description (design input).

    The wall-pressure spectrum is modelled as ``n_harmonics`` harmonics whose
    RMS adds up to ``p_rms`` around ``f_peak``; each harmonic drives one
    Rayleigh-Plesset collapse, so the effective driving pressure is

        Delta p_eff = sqrt(2) * p_rms * sqrt(1/n_harmonics) * n_harmonics**(1/2)

    i.e. ``sqrt(2) p_rms`` (the peak of the RMS signal) -- the function exists so
    that a measured or CFD-derived ``p'_rms`` spectrum can be fed into the
    erosion chain instead of the ``(sigma_i - sigma) q`` estimate.
    """
    if p_rms < 0.0:
        raise ValueError("p_rms must be non-negative")
    return float(math.sqrt(2.0) * p_rms), float(f_peak)