"""1-D finite-volume solver for the shallow-water equations on a flat bed.

The system integrated is

.. math::

    \\frac{\\partial h}{\\partial t} + \\frac{\\partial (h u)}{\\partial x} = 0,
    \\qquad
    \\frac{\\partial (h u)}{\\partial t}
      + \\frac{\\partial}{\\partial x}\\Big(h u^2 + \\tfrac{g h^2}{2}\\Big) = 0,

i.e. the depth-averaged, hydrostatic, inviscid model that owns the Ritter and
Stoker dam-break solutions.  The discretisation is deliberately classical so
that the benchmark measures the *physics* agreement rather than a bespoke
scheme:

* cell-centred finite volume, uniform grid, transmissive (zero-gradient) ends;
* MUSCL reconstruction of the primitive variables ``(h, u)`` with a minmod
  limiter (``order=2``) or plain piecewise constants (``order=1``);
* HLL Riemann solver with Einfeldt/Roe wave-speed estimates, which handles
  wet/dry fronts because the estimates use ``sqrt(g h)`` on both sides;
* two-stage SSP Runge-Kutta (Heun) with an adaptive CFL time step.

Dry cells (``h <= dry_depth``) carry zero velocity; after every stage the depth
is clipped at zero as a positivity safeguard and the momentum is zeroed where
the cell is dry.  Both safeguards are *counted* (``clipped_cells``,
``dry_flips``) so a benchmark can prove they never fired on the accepted grid.

Reference: Toro, *Riemann Solvers and Numerical Methods for Fluid Dynamics*
(3rd ed., 2009), chapters 6-7 and 13; Audusse et al. (2004) for the wet/dry
treatment rationale (flat bed here, so the hydrostatic reconstruction reduces
to the identity).
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["ShallowWater1D"]


class ShallowWater1D:
    """Explicit FV solver for the 1-D shallow-water equations on a flat bed."""

    def __init__(self, n: int, x_min: float, x_max: float, gravity: float = 9.81,
                 cfl: float = 0.4, order: int = 2, dry_depth: float = 1e-12):
        if n < 4:
            raise ValueError("need at least 4 cells")
        if not x_max > x_min:
            raise ValueError("x_max must exceed x_min")
        if order not in (1, 2):
            raise ValueError("order must be 1 or 2")
        if not 0.0 < cfl <= 1.0:
            raise ValueError("cfl must be in (0, 1]")
        self.n = int(n)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.gravity = float(gravity)
        self.cfl = float(cfl)
        self.order = int(order)
        self.dry_depth = float(dry_depth)
        self.dx = (self.x_max - self.x_min) / self.n
        self.x = self.x_min + (np.arange(self.n) + 0.5) * self.dx
        self.x_face = self.x_min + np.arange(self.n + 1) * self.dx
        self.h = np.zeros(self.n)
        self.hu = np.zeros(self.n)
        self.time = 0.0
        self.steps = 0
        self.dt_last = 0.0
        self.clipped_cells = 0
        self.dry_flips = 0

    # ------------------------------------------------------------------ state
    def set_state(self, h, u=None) -> None:
        """Set depth ``h`` (and optional velocity ``u``) on the cell centres."""
        h = np.asarray(h, dtype=float)
        if h.shape != (self.n,):
            raise ValueError(f"h must have shape ({self.n},), got {h.shape}")
        self.h = np.maximum(h, 0.0).copy()
        if u is None:
            self.hu = np.zeros(self.n)
        else:
            u = np.asarray(u, dtype=float)
            self.hu = np.where(self.h > self.dry_depth, self.h * u, 0.0)
        self.time = 0.0
        self.steps = 0
        self.dt_last = 0.0
        self.clipped_cells = 0
        self.dry_flips = 0

    def primitive(self, h=None, hu=None):
        """Return ``(h, u)`` with ``u = 0`` wherever the cell is dry."""
        h = self.h if h is None else h
        hu = self.hu if hu is None else hu
        wet = h > self.dry_depth
        u = np.where(wet, hu / np.where(wet, h, 1.0), 0.0)
        return h, u

    # ---------------------------------------------------------- discretisation
    def _minmod_slope(self, w: np.ndarray) -> np.ndarray:
        """Minmod-limited cell slopes; the two boundary cells get zero slope."""
        slope = np.zeros_like(w)
        if self.order == 1 or w.size < 3:
            return slope
        dw = np.diff(w) / self.dx
        left = dw[:-1]
        right = dw[1:]
        limited = np.where(left * right > 0.0,
                           np.sign(left) * np.minimum(np.abs(left),
                                                      np.abs(right)),
                           0.0)
        slope[1:-1] = limited
        return slope

    def _face_states(self):
        """Reconstructed ``(hL, uL, hR, uR)`` on the ``n + 1`` faces."""
        h, u = self.primitive()
        sh = self._minmod_slope(h)
        su = self._minmod_slope(u)
        n = self.n
        hL = np.empty(n + 1)
        hR = np.empty(n + 1)
        uL = np.empty(n + 1)
        uR = np.empty(n + 1)
        hL[0], hR[0], uL[0], uR[0] = h[0], h[0], u[0], u[0]
        hL[-1], hR[-1], uL[-1], uR[-1] = h[-1], h[-1], u[-1], u[-1]
        half = 0.5 * self.dx
        hL[1:-1] = h[:-1] + half * sh[:-1]
        hR[1:-1] = h[1:] - half * sh[1:]
        uL[1:-1] = u[:-1] + half * su[:-1]
        uR[1:-1] = u[1:] - half * su[1:]
        # positivity of the reconstructed depths (wet/dry fronts)
        hL = np.maximum(hL, 0.0)
        hR = np.maximum(hR, 0.0)
        uL = np.where(hL > self.dry_depth, uL, 0.0)
        uR = np.where(hR > self.dry_depth, uR, 0.0)
        return hL, uL, hR, uR

    def _hll_flux(self, hL, uL, hR, uR):
        """HLL flux of ``(h, hu)`` with Einfeldt wave-speed estimates."""
        g = self.gravity
        hL = np.maximum(hL, 0.0)
        hR = np.maximum(hR, 0.0)
        cL = np.sqrt(g * hL)
        cR = np.sqrt(g * hR)
        fhL = hL * uL
        fhuL = hL * uL * uL + 0.5 * g * hL * hL
        fhR = hR * uR
        fhuR = hR * uR * uR + 0.5 * g * hR * hR
        sqL = np.sqrt(hL)
        sqR = np.sqrt(hR)
        den = sqL + sqR
        safe_den = np.where(den > 0.0, den, 1.0)
        u_roe = np.where(den > 0.0, (sqL * uL + sqR * uR) / safe_den, 0.0)
        c_roe = np.sqrt(g * 0.5 * (hL + hR))
        sL = np.minimum(uL - cL, u_roe - c_roe)
        sR = np.maximum(uR + cR, u_roe + c_roe)
        gap = sR - sL
        safe_gap = np.where(gap > 1e-300, gap, 1.0)
        middle_h = (sR * fhL - sL * fhR + sL * sR * (hR - hL)) / safe_gap
        middle_hu = (sR * fhuL - sL * fhuR
                     + sL * sR * (hR * uR - hL * uL)) / safe_gap
        flux_h = np.where(sL >= 0.0, fhL, np.where(sR <= 0.0, fhR, middle_h))
        flux_hu = np.where(sL >= 0.0, fhuL,
                           np.where(sR <= 0.0, fhuR, middle_hu))
        dry_pair = (hL <= self.dry_depth) & (hR <= self.dry_depth)
        flux_h = np.where(dry_pair, 0.0, flux_h)
        flux_hu = np.where(dry_pair, 0.0, flux_hu)
        return flux_h, flux_hu

    def rhs(self, h: np.ndarray, hu: np.ndarray):
        """Semi-discrete right-hand side ``d/dt (h, hu)``."""
        self_h, self_hu = self.h, self.hu
        self.h, self.hu = h, hu
        try:
            hL, uL, hR, uR = self._face_states()
        finally:
            self.h, self.hu = self_h, self_hu
        flux_h, flux_hu = self._hll_flux(hL, uL, hR, uR)
        dh = -(flux_h[1:] - flux_h[:-1]) / self.dx
        dhu = -(flux_hu[1:] - flux_hu[:-1]) / self.dx
        return dh, dhu

    # ---------------------------------------------------------------- stepping
    def max_wave_speed(self) -> float:
        h, u = self.primitive()
        return float(np.max(np.abs(u) + np.sqrt(self.gravity * h)))

    def _sanitise(self) -> None:
        """Positivity safeguard; every intervention is counted, not silent."""
        neg = self.h < 0.0
        if np.any(neg):
            self.clipped_cells += int(np.count_nonzero(neg))
            self.h = np.maximum(self.h, 0.0)
        dry = self.h <= self.dry_depth
        if np.any(dry & (self.hu != 0.0)):
            self.dry_flips += int(np.count_nonzero(dry & (self.hu != 0.0)))
            self.hu = np.where(dry, 0.0, self.hu)
        if not (np.isfinite(self.h).all() and np.isfinite(self.hu).all()):
            raise FloatingPointError(
                f"non-finite state at t={self.time:.6g}, step {self.steps}")

    def step(self, dt: float) -> None:
        """One SSP-RK3 (Shu-Osher) step of length ``dt``.

        Third order is required, not cosmetic: with the HLL flux and a
        second-order reconstruction the semi-discrete operator is close to
        skew symmetric (the limiter cancels the first-order HLL diffusion in
        smooth regions), and SSP-RK2 has ``|G|^2 = 1 + (omega dt)^4 / 8 > 1``
        on such operators -- a slow unconditional growth that drains a smooth
        test field to zero.  SSP-RK3 is the standard pairing for MUSCL + Riemann
        solvers (Shu & Osher 1988) and is stable there.
        """
        h0, hu0 = self.h, self.hu
        k1h, k1hu = self.rhs(h0, hu0)
        h1 = h0 + dt * k1h
        hu1 = hu0 + dt * k1hu
        self.h, self.hu = h1, hu1
        self._sanitise()
        h1, hu1 = self.h, self.hu
        k2h, k2hu = self.rhs(h1, hu1)
        h2 = 0.75 * h0 + 0.25 * (h1 + dt * k2h)
        hu2 = 0.75 * hu0 + 0.25 * (hu1 + dt * k2hu)
        self.h, self.hu = h2, hu2
        self._sanitise()
        h2, hu2 = self.h, self.hu
        k3h, k3hu = self.rhs(h2, hu2)
        self.h = (1.0 / 3.0) * h0 + (2.0 / 3.0) * (h2 + dt * k3h)
        self.hu = (1.0 / 3.0) * hu0 + (2.0 / 3.0) * (hu2 + dt * k3hu)
        self._sanitise()
        self.time += dt
        self.steps += 1
        self.dt_last = dt

    def advance(self, t_end: float, max_steps: int = 2_000_000,
                callback=None) -> int:
        """Integrate to ``t_end`` with an adaptive CFL step; returns #steps.

        ``callback(solver)`` is called after every accepted step (used by the
        benchmark to record time series without re-running).
        """
        if t_end <= self.time:
            return 0
        start_steps = self.steps
        while self.time < t_end - 1e-15:
            speed = self.max_wave_speed()
            if speed <= 0.0:
                raise RuntimeError("no wave speed: the state is at rest")
            dt = self.cfl * self.dx / speed
            dt = min(dt, t_end - self.time)
            self.step(dt)
            self.dt_last = dt
            if callback is not None:
                callback(self)
            if self.steps - start_steps > max_steps:
                raise RuntimeError(f"step budget exceeded ({max_steps})")
        return self.steps - start_steps

    # -------------------------------------------------------------- diagnostics
    def total_volume(self) -> float:
        """``sum(h) dx``, the conserved water volume."""
        return float(self.h.sum()) * self.dx

    def momentum(self) -> float:
        """``sum(hu) dx``, the conserved momentum."""
        return float(self.hu.sum()) * self.dx

    def cfl_number(self) -> float:
        """Courant number of the last accepted step, ``max(|u| + c) dt / dx``.

        Fixed in m16596: the previous body returned
        ``max_wave_speed() * dx / dx`` -- i.e. the wave speed in m/s, a
        dimensionally wrong quantity that merely *looks* like a Courant number
        and would silently pass any ``cfl < 1`` assertion.  The number is now
        the actual dimensionless Courant number of the last step
        (``dt_last`` is the step the solver took; 0 before the first step).
        """
        return self.max_wave_speed() * self.dt_last / max(self.dx, 1e-300)

    def summary(self) -> dict:
        h, u = self.primitive()
        return {
            "n": self.n,
            "dx": self.dx,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "g": self.gravity,
            "cfl": self.cfl,
            "order": self.order,
            "t": self.time,
            "steps": self.steps,
            "h_min": float(h.min()),
            "h_max": float(h.max()),
            "u_max": float(np.max(np.abs(u))),
            "volume": self.total_volume(),
            "momentum": self.momentum(),
            "clipped_cells": self.clipped_cells,
            "dry_flips": self.dry_flips,
        }


def _as_float(value: float) -> float:
    """Helper kept for readability of error messages (finite check)."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"expected a finite number, got {value!r}")
    return value