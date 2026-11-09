"""Analytic target wave (Airy + Stokes-II bound harmonic) for the sigma-grid NWT.

The quantity of record here is the *velocity field* that the relaxation zone
imposes, because the tank is velocity driven: the elevation is never forced, it
follows from mass conservation.  The kinematics are Stokes second order in
finite depth (Dean & Dalrymple, *Water Wave Mechanics for Engineers and
Scientists*, 2nd ed., eqs. 11.36-11.40)::

    eta   = a cos(th) + (k a^2 / 4) (3 - s^2)/s^3 cos(2 th),      s = tanh(kh)
    phi   = (a g / w) [cosh k(z+h)/cosh kh] sin(th)
          + (3 a^2 w / 8) [cosh 2k(z+h)/sinh^4(kh)] sin(2 th)
    u     = dphi/dx,  w = dphi/dz,  th = k x - w t + phase0

so the first harmonic is Airy and the second harmonic is the bound Stokes wave.
``z`` is measured from the still water level, finite depth ``h``.

The first-order part reuses :func:`tensorfvm.wave_tank.theory.airy_kinematics`
so both tanks measure their waves against exactly the same target; the
second-order velocity terms are added here (``order2=True``).
:func:`stokes2_eta` is checked against
:func:`tensorfvm.wave_tank.theory.stokes2_elevation` in the test suite, which
guards the sign and coefficient conventions of the potential above.
"""
from __future__ import annotations

import math

import numpy as np

from ..wave_tank.theory import GRAVITY, airy_kinematics, dispersion_k

__all__ = ["WaveTarget", "stokes2_eta", "stokes2_potential_velocities",
           "fit_harmonics"]


def stokes2_eta(x, t: float, amp: float, omega: float, k: float, depth: float,
                phase0: float = 0.0, order2: bool = True) -> np.ndarray:
    """Stokes-II surface elevation (identical coefficients to the legacy tank)."""
    th = k * np.asarray(x, dtype=float) - omega * t + phase0
    eta = amp * np.cos(th)
    if order2:
        s = math.tanh(k * depth)
        eta = eta + (k * amp * amp / 4.0) * (3.0 - s * s) / s ** 3 * np.cos(2.0 * th)
    return eta


def stokes2_potential_velocities(x, z, t: float, amp: float, omega: float, k: float,
                                 depth: float, g: float = GRAVITY,
                                 phase0: float = 0.0, order2: bool = True):
    """Stokes-II orbital velocities ``(u, w)`` at ``(x, z)``, ``z`` from SWL.

    Second-order contribution of the potential
    ``(3 a^2 w / 8) cosh 2k(z+h) / sinh^4(kh)`` is the finite-depth Stokes form;
    it vanishes as ``kh -> inf`` (deep water), so the target degenerates to Airy
    exactly where the second-order correction is negligible.
    """
    x = np.asarray(x, dtype=float)
    z = np.asarray(z, dtype=float)
    u1, w1 = _airy_uv(x, z, t, amp, omega, k, depth, phase0)
    if not order2:
        return u1, w1
    th = k * x - omega * t + phase0
    sh = math.sinh(k * depth)
    a2 = 3.0 * amp * amp * omega / 8.0 / sh ** 4
    ch2 = np.cosh(2.0 * k * (z + depth))
    sh2 = np.sinh(2.0 * k * (z + depth))
    u2 = 2.0 * k * a2 * ch2 * np.cos(2.0 * th)
    w2 = 2.0 * k * a2 * sh2 * np.sin(2.0 * th)
    return u1 + u2, w1 + w2


def _airy_uv(x, z, t, amp, omega, k, depth, phase0):
    eta, u, v = airy_kinematics(x, z, t, amp, omega, k, depth, phase0=phase0)
    return u, v


class WaveTarget:
    """Regular Stokes-II target wave of the tank (``x`` along, ``z`` up)."""

    def __init__(self, amp: float, period: float, depth: float,
                 gravity: float = GRAVITY, phase0: float = 0.0,
                 order2: bool = True):
        if amp <= 0.0 or period <= 0.0 or depth <= 0.0:
            raise ValueError("amp, period and depth must be positive")
        self.amp = float(amp)
        self.period = float(period)
        self.omega = 2.0 * math.pi / self.period
        self.depth = float(depth)
        self.gravity = float(gravity)
        self.phase0 = float(phase0)
        self.order2 = bool(order2)
        self.k = dispersion_k(self.omega, self.depth, self.gravity)
        self.wavelength = 2.0 * math.pi / self.k
        self.c = self.omega / self.k

    # -- fields -----------------------------------------------------------
    def eta(self, x, t: float) -> np.ndarray:
        return stokes2_eta(x, t, self.amp, self.omega, self.k, self.depth,
                           self.phase0, self.order2)

    def velocities(self, x, z, t: float):
        return stokes2_potential_velocities(
            x, z, t, self.amp, self.omega, self.k, self.depth, self.gravity,
            self.phase0, self.order2)

    def steepness(self) -> float:
        return self.k * self.amp

    def as_dict(self) -> dict:
        return {
            "amp": self.amp,
            "period": self.period,
            "omega": self.omega,
            "wavenumber": self.k,
            "wavelength": self.wavelength,
            "phase_speed": self.c,
            "depth": self.depth,
            "kh": self.k * self.depth,
            "steepness_ka": self.steepness(),
            "order": "stokes2" if self.order2 else "airy",
        }

def fit_harmonics(t, signal, omega: float, n_harm: int = 2):
    """Least-squares fit of ``mean + sum_k [a_k cos(k w t) + b_k sin(k w t)]``.

    A single-harmonic fit (:func:`tensorfvm.wave_tank.theory.harmonic_fit`) is
    biased whenever the signal also carries the Stokes-II bound harmonic: the
    second harmonic leaks into the residual and its correlation with the first
    harmonic shows up as an amplitude error of a few percent.  Fitting both
    harmonics removes that bias, which matters because the acceptance threshold
    on the wave height is 3 %.

    Returns ``dict(amp=[...], phase=[...], mean, residual_rel, n)`` with
    ``amp[k] = hypot(a_k, b_k)`` in the ``cos(k w t - phase)`` convention.
    """
    t = np.asarray(t, dtype=float)
    s = np.asarray(signal, dtype=float)
    cols = [np.ones_like(t)]
    for k in range(1, n_harm + 1):
        cols.append(np.cos(k * omega * t))
        cols.append(np.sin(k * omega * t))
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, s, rcond=None)
    means = coef[0]
    amps, phases = [], []
    for k in range(1, n_harm + 1):
        ac, as_ = coef[2 * k - 1], coef[2 * k]
        amps.append(float(np.hypot(ac, as_)))
        phases.append(float(np.arctan2(as_, ac)))
    resid = s - A @ coef
    den = float(np.sqrt(np.mean((s - s.mean()) ** 2)))
    return {"amp": amps, "phase": phases, "mean": float(means),
            "residual_rel": float(np.sqrt(np.mean(resid ** 2)) / den if den > 0 else 0.0),
            "n": int(t.size)}
