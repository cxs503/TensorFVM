"""One-dimensional Stefan (solidification) verification: enthalpy method vs Neumann.

The module exists to answer one question with numbers: *does the phase-change
discretisation place the solidification front where the similarity solution says
it must be?*  It provides

``neumann_lambda``            solution of ``St sqrt(pi) L exp(L^2) erf(L) = 1``
``neumann_front``             ``X(t) = 2 L sqrt(alpha_s t)``
``neumann_temperature``       the companion similarity temperature field
``solve_stefan_1d``           enthalpy-method FVM marching in time

Numerical method (single-phase Stefan, cold wall, liquid initially at the fusion
temperature):

* conservative finite volume, cell-centred, backward Euler in time;
* the unknown is the *temperature*; the enthalpy ``h(T)`` is inverted exactly
  through a narrow mushy band ``[T_f - dT, T_f + dT]`` with apparent capacity
  ``c_eff = c_s + L / (2 dT)``.  ``dT`` is a numerical thickness, not a physical
  one: it is refined together with the mesh and its influence is reported.
* the nonlinear system is solved by Picard iteration on ``c_eff`` until the
  temperature increment falls below ``tol``; every iteration is one tridiagonal
  solve, so each time step costs a few microseconds per cell.
* the front position is reconstructed from the *enthalpy* (volume-averaged solid
  fraction), which is the conservative estimate, and cross-checked against the
  location where the liquid fraction passes one half.

The energy ledger sums the wall flux over time and compares it with the change of
sensible plus latent enthalpy in the domain, so a broken latent-heat term shows
up as a non-zero balance rather than as a silent loss of physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.special import erf

__all__ = [
    "ICE_K", "ICE_RHO", "ICE_C", "ICE_LATENT",
    "neumann_lambda", "neumann_front", "neumann_temperature",
    "StefanResult", "solve_stefan_1d",
]

#: thermophysical constants of ice/water at the fusion point (SI)
ICE_K = 2.22          # W/(m K)
ICE_RHO = 917.0       # kg/m^3
ICE_C = 2050.0        # J/(kg K)
ICE_LATENT = 3.337e5  # J/kg


# ------------------------------------------------------------------ similarity
def neumann_lambda(stefan_number: float, *, xtol: float = 1e-15) -> float:
    r"""Similarity constant of the one-phase Stefan problem.

    With the solid occupying ``0 < x < X(t)``, ``T(0,t) = T_w``, ``T(X,t) = T_f``
    and liquid at ``T_f``, the heat balance at the front gives

    .. math::  \sqrt{\pi}\,\Lambda e^{\Lambda^2}\mathrm{erf}(\Lambda) = St,
               \qquad St = \frac{c_s (T_f - T_w)}{L}.

    Derivation: the Stefan condition ``rho L dX/dt = k_s dT_s/dx`` at ``X`` with
    ``X = 2 L sqrt(alpha t)`` and the similarity gradient
    ``(T_f-T_w) exp(-L^2) / (sqrt(pi alpha t) erf L)`` collapses to the
    transcendental relation above.  The left-hand side is strictly increasing on
    ``L > 0`` (from 0 to infinity), so bisection is unconditionally safe.
    """
    st = float(stefan_number)
    if not (st > 0.0):
        raise ValueError("Stefan number must be positive")

    def f(lam: float) -> float:
        return math.sqrt(math.pi) * lam * math.exp(lam * lam) * erf(lam) - st

    lo, hi = 1e-8, 1.0
    while f(hi) < 0.0:
        hi *= 1.6
        if hi > 40.0:                                    # pragma: no cover
            raise RuntimeError("could not bracket the similarity constant")
    lam = 0.5 * (lo + hi)
    for _ in range(400):
        val = f(lam)
        if val > 0.0:
            hi = lam
        else:
            lo = lam
        new = 0.5 * (lo + hi)
        if abs(new - lam) <= xtol * max(1.0, abs(lam)):
            return new
        lam = new
    return lam                                                # pragma: no cover


def neumann_front(t: float, stefan_number: float, alpha_solid: float) -> float:
    """``X(t) = 2 Lambda sqrt(alpha_s t)``."""
    return 2.0 * neumann_lambda(stefan_number) * math.sqrt(alpha_solid * float(t))


def neumann_temperature(x, t: float, stefan_number: float, alpha_solid: float,
                        t_wall: float, t_fusion: float):
    """Similarity temperature in the solid: ``T_w + (T_f-T_w) erf(eta)/erf(L)``."""
    lam = neumann_lambda(stefan_number)
    eta = np.asarray(x, dtype=float) / (2.0 * math.sqrt(alpha_solid * float(t)))
    return t_wall + (t_fusion - t_wall) * erf(eta) / erf(lam)


# ------------------------------------------------------------------- enthalpy
@dataclass
class StefanResult:
    """Fields and diagnostics returned by :func:`solve_stefan_1d`."""

    x: np.ndarray
    times: np.ndarray
    temperatures: np.ndarray       # (n_times, n_cells)
    front_enthalpy: np.ndarray     # volume-averaged solid fraction estimate
    front_half_liquid: np.ndarray  # outermost cell where f_l >= 0.5
    front_exact: np.ndarray
    t_wall: float
    t_fusion: float
    stefan_number: float
    alpha_solid: float
    mushy_thickness: float
    grid: dict = field(default_factory=dict)
    ledger: dict = field(default_factory=dict)

    @property
    def front_rel_error(self) -> np.ndarray:
        return (self.front_enthalpy - self.front_exact) / self.front_exact


def _enthalpy(temperature, t_fusion, dt_mush, c_solid, latent):
    """Enthalpy of the ice/water pair, zero offset at ``T_f - dT``.

    The two-phase band ``[T_f - dT, T_f + dT]`` is a *numerical* device: ``dT`` is
    refined with the mesh and never treated as a physical freezing range.  Inside
    the band the apparent capacity is ``L / (2 dT)``, so a cell that spans the
    front stores the latent heat of the solid fraction it contains while its
    temperature stays within ``dT`` of fusion -- the standard enthalpy method.
    """
    t = np.asarray(temperature, dtype=float)
    h = np.empty_like(t)
    lo = t <= t_fusion - dt_mush
    hi = t >= t_fusion + dt_mush
    mid = ~(lo | hi)
    h[lo] = c_solid * (t[lo] - (t_fusion - dt_mush))
    h[hi] = c_solid * dt_mush + latent + c_solid * (t[hi] - (t_fusion + dt_mush))
    h[mid] = c_solid * dt_mush + latent * (t[mid] - (t_fusion - dt_mush)) / (2.0 * dt_mush)
    return h


def _capacity(temperature, t_fusion, dt_mush, c_solid, latent):
    """Derivative ``dh/dT``: apparent capacity (spike of ``L/2dT`` in the band)."""
    t = np.asarray(temperature, dtype=float)
    c = np.full_like(t, c_solid)
    mid = np.abs(t - t_fusion) < dt_mush
    c[mid] = c_solid + latent / (2.0 * dt_mush)
    return c


def _solid_fraction(temperature, t_fusion, dt_mush, c_solid, latent):
    """Solid fraction reconstructed from the enthalpy (0 = liquid)."""
    h = _enthalpy(temperature, t_fusion, dt_mush, c_solid, latent)
    h_solidus = c_solid * dt_mush
    f_liquid = np.clip((h - h_solidus) / latent, 0.0, 1.0)
    return 1.0 - f_liquid


def _temperature_of_enthalpy(h, t_fusion, dt_mush, c_solid, latent):
    """Exact inverse of :func:`_enthalpy` and its derivative ``dT/dh``.

    Solving the discretised system *in the enthalpy variable* keeps the Newton
    Jacobian well conditioned: the huge apparent capacity of a front cell appears
    as ``dT/dh -> 2 dT / L`` on the diagonal instead of ``L / (2 dT)``, so a
    two-phase cell neither freezes the iteration nor lets it overshoot.
    """
    h = np.asarray(h, dtype=float)
    h_solidus = c_solid * dt_mush
    h_liquidus = h_solidus + latent
    t = np.empty_like(h)
    dt_dh = np.empty_like(h)
    lo = h <= h_solidus
    hi = h >= h_liquidus
    mid = ~(lo | hi)
    t[lo] = (t_fusion - dt_mush) + h[lo] / c_solid
    t[hi] = (t_fusion + dt_mush) + (h[hi] - h_liquidus) / c_solid
    t[mid] = (t_fusion - dt_mush) + 2.0 * dt_mush * (h[mid] - h_solidus) / latent
    dt_dh[lo] = 1.0 / c_solid
    dt_dh[hi] = 1.0 / c_solid
    dt_dh[mid] = 2.0 * dt_mush / latent
    return t, dt_dh


def solve_stefan_1d(*, length: float = 0.15, n_cells: int = 400, t_end: float = 1600.0,
                    dt: float = 0.5, t_wall: float = -20.0, t_fusion: float = 0.0,
                    mushy_thickness: float = 0.05,
                    k_solid: float = ICE_K, rho: float = ICE_RHO,
                    c_solid: float = ICE_C, latent: float = ICE_LATENT,
                    sample_times=(100.0, 500.0, 1600.0), tol: float = 1e-8,
                    max_newton: int = 60) -> StefanResult:
    """March the one-phase Stefan problem and sample the front position.

    The wall (``x = 0``) is held at ``t_wall``, the far end is adiabatic and the
    liquid starts exactly at ``t_fusion`` (single-phase Stefan problem, so the
    Neumann similarity solution applies without a superheat correction).
    """
    nx = int(n_cells)
    dx = length / nx
    alpha = k_solid / (rho * c_solid)
    dt_mush = float(mushy_thickness)
    stefan = c_solid * (t_fusion - t_wall) / latent

    x_center = (np.arange(nx) + 0.5) * dx
    # start at the *liquidus* (T_f + dT) so the initial state is fully liquid: a
    # cell parked at the band centre would already carry half the latent heat and
    # bias the whole front reconstruction.
    temperature = np.full(nx, float(t_fusion) + dt_mush)

    # face conductances (per unit area) divided by the cell width
    k_face = np.full(nx + 1, k_solid)
    coef_int = k_face / (dx * dx)

    sample_times = np.asarray(sorted(sample_times), dtype=float)
    samples = np.zeros(sample_times.size)
    keep = np.zeros((sample_times.size, nx))
    nxt = 0

    flux_integral = 0.0            # integral of the wall heat flux, J/m^2 (from the wall)
    h_initial = _enthalpy(temperature, t_fusion, dt_mush, c_solid, latent)
    n_steps = int(math.ceil(t_end / dt))
    history = []
    newton_exhausted = 0
    for step in range(n_steps):
        t_now = min(t_end, (step + 1) * dt)
        dt_step = t_now - step * dt
        h_old = _enthalpy(temperature, t_fusion, dt_mush, c_solid, latent)
        h_new = h_old.copy()
        delta_t = np.inf
        for it in range(max_newton):
            t_new, dt_dh = _temperature_of_enthalpy(h_new, t_fusion, dt_mush,
                                                    c_solid, latent)
            # residual of  rho dh/dt = d/dx (k dT/dx)   (J m^-3 s^-1)
            flux = coef_int[1:nx] * (t_new[1:] - t_new[:-1])       # face flux / dx
            lap = np.zeros(nx)
            lap[:-1] += flux
            lap[1:] -= flux
            # cold wall: half-cell face, twice the interior conductance
            lap[0] -= 2.0 * coef_int[0] * (t_new[0] - t_wall)
            residual = rho * (h_new - h_old) / dt_step - lap
            # Newton Jacobian (tridiagonal, in the enthalpy variable)
            a = np.zeros(nx)
            b = np.empty(nx)
            c = np.zeros(nx)
            b[:] = rho / dt_step + (coef_int[0:nx] + coef_int[1:nx + 1]) * dt_dh
            b[0] += coef_int[0] * dt_dh[0]          # wall face is twice as strong
            b[nx - 1] -= coef_int[nx] * dt_dh[nx - 1]   # east end is adiabatic
            a[1:] = -coef_int[1:nx] * dt_dh[:-1]
            c[:-1] = -coef_int[1:nx] * dt_dh[1:]
            step_h = _thomas(a, b, c, -residual.copy())
            # bound the update: at most half a latent heat and 4 K per iteration
            step_t = step_h * dt_dh
            scale = max(1.0, float(np.max(np.abs(step_t))) / 4.0,
                        float(np.max(np.abs(step_h))) / (0.5 * latent))
            h_new = h_new + step_h / scale
            delta_t = float(np.max(np.abs(step_t)))
            if delta_t < tol:
                break
        else:
            newton_exhausted += 1
        history.append((it + 1, delta_t))
        temperature = _temperature_of_enthalpy(h_new, t_fusion, dt_mush, c_solid,
                                               latent)[0]
        # heat *leaving* the domain through the cold wall (positive outward)
        q_wall = 2.0 * k_face[0] * (temperature[0] - t_wall) / dx
        flux_integral += q_wall * dt_step
        if nxt < sample_times.size and t_now >= sample_times[nxt] - 1e-12:
            samples[nxt] = t_now
            keep[nxt, :] = temperature
            nxt += 1

    while nxt < len(sample_times):                 # pragma: no cover
        keep[nxt, :] = temperature
        samples[nxt] = t_end
        nxt += 1

    fronts = np.zeros(sample_times.size)
    fronts_half = np.zeros(sample_times.size)
    exact = np.zeros(sample_times.size)
    for i, t in enumerate(samples):
        fs = _solid_fraction(keep[i], t_fusion, dt_mush, c_solid, latent)
        fronts[i] = float(np.sum(fs) * dx)
        fl = 1.0 - fs
        above = np.nonzero(fl >= 0.5)[0]
        if above.size:
            j = int(above[0])
            # linear interpolation of the f_l = 0.5 crossing
            f_hi = fl[j]
            f_lo = fl[j - 1] if j > 0 else 1.0
            w = 0.0 if f_hi == f_lo else (0.5 - f_lo) / (f_hi - f_lo)
            fronts_half[i] = (j + w) * dx
        else:                                        # pragma: no cover
            fronts_half[i] = length
        exact[i] = neumann_front(t, stefan, alpha)

    # energy ledger: store change equals the flux integral
    h_final = _enthalpy(temperature, t_fusion, dt_mush, c_solid, latent)
    stored = float(np.sum(h_final - h_initial)) * dx * rho
    scale = max(abs(latent * rho * fronts[-1]), abs(flux_integral), 1e-30)
    ledger = {
        "wall_heat_removed_J_m2": float(flux_integral),
        "stored_enthalpy_change_J_m2": stored,
        "imbalance_J_m2": float(stored + flux_integral),
        "normalized_imbalance": abs(stored + flux_integral) / scale,
        "newton_iterations_max": int(max(it for it, _ in history)),
        "newton_steps_not_converged": int(newton_exhausted),
        "newton_history_tail": [[int(k), float(v)] for k, v in history[-3:]],
    }
    return StefanResult(
        x=x_center, times=samples, temperatures=keep, front_enthalpy=fronts,
        front_half_liquid=fronts_half, front_exact=exact, t_wall=float(t_wall),
        t_fusion=float(t_fusion), stefan_number=stefan, alpha_solid=alpha,
        mushy_thickness=dt_mush,
        grid={"n_cells": nx, "dx_m": dx, "dt_s": float(dt), "length_m": length},
        ledger=ledger,
    )


def _thomas(a, b, c, d):
    """Solve a tridiagonal system (``a`` sub-, ``b`` main, ``c`` super-diagonal)."""
    n = b.size
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n):
        m = b[i] - a[i] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x