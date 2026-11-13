"""Two-dimensional shallow-water equations (SWE-2D) on a bed, well balanced.

Why this module exists
----------------------
``shallow_water.swe1d`` / ``shallow_water.bed`` integrate the depth-averaged
equations in **one** space dimension.  Every genuinely two-dimensional
shallow-water question -- radial run-up on a slope, harbour seiching, the
nonlinear oscillation of a lake in a bowl -- is out of their reach, and so was
any 2-D shallow-water *validation*: the 1-D Ritter/Stoker exact solutions
cannot see an ``x``/``y`` operator inconsistency, a transposed index or a
directional bias in the flux splitting.

This module adds the missing operator.  It is deliberately a direct tensor
generalisation of the 1-D well-balanced scheme already validated in this
package (``bed.ShallowWaterBed1D``), so that a 2-D result can be attributed to
the physics and not to a bespoke discretisation:

* MUSCL reconstruction of the **water surface** ``eta = h + z`` and of both
  velocity components, minmod limiter, piecewise-constant bed;
* Audusse, Bouchut, Bristeau, Klein & Perthame (2004, *SIAM J. Sci. Comput.*
  **25**, 2050) hydrostatic reconstruction at every face:
  ``z_f = max(z_L, z_R)``, ``h*_side = max(0, eta_side - z_f)``;
* **full three-component HLL** flux ``(h, h u_n, h u_t)`` with Einfeldt/Roe
  wave speeds, which keeps the scheme isotropic (an upwind-transported
  transverse velocity would break the ``x <-> y`` symmetry that the Thacker
  benchmark checks);
* the hydrostatic pressure correction ``+g/2 (h_side^2 - h*_side^2)`` applied
  to the momentum flux on the cell's own side of the face -- the price of the
  well-balanced property is that momentum is not globally conservative, the bed
  legitimately exerts a force;
* SSP-RK3 (Shu-Osher) time stepping with an adaptive CFL step, positivity
  clipping and dry-cell momentum zeroing, both **counted** so a benchmark can
  prove they never fired on an accepted grid.

``well_balanced=False`` builds the negative control of the verification driver:
no hydrostatic reconstruction, bed slope as a centred source
``-g h (dz/dx, dz/dy)``.  It is kept in-tree on purpose (a gate that cannot
fail proves nothing) and is never used by a benchmark case.

Boundaries are transmissive (zero-gradient ghost states), matching the 1-D
solver; closed basins that keep the water away from the boundary are the
intended use (the verification driver asserts the boundary stays dry).

References: Audusse et al. (2004) §3; Toro, *Riemann Solvers and Numerical
Methods for Fluid Dynamics* (3rd ed., 2009) ch. 6-7, 13; Xing & Shu (2011,
*J. Comput. Phys.* **230**, 6037) for the well-balanced MUSCL framework;
Shu & Osher (1988) for SSP-RK3.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["ShallowWater2D"]


class ShallowWater2D:
    """Explicit finite-volume solver for the 2-D shallow-water equations.

    Parameters
    ----------
    nx, ny : int
        Number of cells in ``x`` and ``y`` (cell-centred, uniform).
    x_min, x_max, y_min, y_max : float
        Domain extent; ``dx = (x_max - x_min) / nx`` and likewise for ``y``.
    bed : array_like, optional
        Cell-centred bed elevation ``z`` with shape ``(nx, ny)``; default flat.
    gravity : float, optional
        ``g`` in m/s^2.
    cfl : float, optional
        Target Courant number of the adaptive step.
    order : int, optional
        ``2`` (default) = MUSCL/minmod, ``1`` = piecewise constant.
    dry_depth : float, optional
        Depth at or below which a cell is treated as dry.
    well_balanced : bool, optional
        ``True`` (default) = hydrostatic reconstruction; ``False`` = centred
        bed-slope source (negative control).
    """

    def __init__(self, nx: int, ny: int, x_min: float, x_max: float,
                 y_min: float, y_max: float, bed=None, gravity: float = 9.81,
                 cfl: float = 0.4, order: int = 2, dry_depth: float = 1e-8,
                 well_balanced: bool = True):
        if nx < 3 or ny < 3:
            raise ValueError("need at least 3 cells in each direction")
        if not (x_max > x_min and y_max > y_min):
            raise ValueError("domain must have positive extent")
        self.nx = int(nx)
        self.ny = int(ny)
        self.x_min, self.x_max = float(x_min), float(x_max)
        self.y_min, self.y_max = float(y_min), float(y_max)
        self.dx = (self.x_max - self.x_min) / self.nx
        self.dy = (self.y_max - self.y_min) / self.ny
        self.gravity = float(gravity)
        self.cfl = float(cfl)
        self.order = int(order)
        self.dry_depth = float(dry_depth)
        self.well_balanced = bool(well_balanced)

        self.h = np.zeros((self.nx, self.ny))
        self.hu = np.zeros((self.nx, self.ny))
        self.hv = np.zeros((self.nx, self.ny))
        self.bed = np.zeros((self.nx, self.ny))
        if bed is not None:
            self.set_bed(bed)
        self.time = 0.0
        self.steps = 0
        self.dt_last = 0.0
        self.clipped_cells = 0
        self.dry_flips = 0
        self._eta0 = self.water_surface().copy()

    # ------------------------------------------------------------- geometry
    def cell_centers(self):
        """Cell-centre coordinates ``(x, y)`` as ``(nx, ny)`` arrays."""
        x = self.x_min + (np.arange(self.nx) + 0.5) * self.dx
        y = self.y_min + (np.arange(self.ny) + 0.5) * self.dy
        return np.meshgrid(x, y, indexing="ij")

    def set_bed(self, bed) -> None:
        """Install the cell-centred bed elevation ``z``."""
        bed = np.asarray(bed, dtype=float)
        if bed.shape != (self.nx, self.ny):
            raise ValueError(
                "bed must have shape (%d, %d), got %s"
                % (self.nx, self.ny, bed.shape))
        if not np.all(np.isfinite(bed)):
            raise ValueError("bed elevation must be finite")
        self.bed = bed.copy()
        self._eta0 = self.water_surface().copy()

    def set_state(self, h, u=None, v=None) -> None:
        """Set the depth and (optionally) the velocity components."""
        h = np.asarray(h, dtype=float)
        if h.shape != (self.nx, self.ny):
            raise ValueError(
                "h must have shape (%d, %d), got %s"
                % (self.nx, self.ny, h.shape))
        if not np.all(np.isfinite(h)):
            raise ValueError("depth must be finite")
        self.h = np.maximum(h, 0.0)
        u = np.zeros_like(self.h) if u is None else np.asarray(u, dtype=float)
        v = np.zeros_like(self.h) if v is None else np.asarray(v, dtype=float)
        if u.shape != self.h.shape or v.shape != self.h.shape:
            raise ValueError("u and v must have the shape of h")
        wet = self.h > self.dry_depth
        safe = np.where(wet, self.h, 1.0)
        self.hu = np.where(wet, u * safe, 0.0)
        self.hv = np.where(wet, v * safe, 0.0)
        self._eta0 = self.water_surface().copy()

    def water_surface(self) -> np.ndarray:
        """``eta = h + z`` -- the quantity a lake at rest holds constant."""
        return self.h + self.bed

    def lake_at_rest_state(self, surface: float) -> None:
        """Initialise a lake at rest with the given free-surface elevation."""
        surface = float(surface)
        self.set_state(np.maximum(surface - self.bed, 0.0))

    def primitive(self):
        """Return ``(h, u, v)``; dry cells carry exactly zero velocity."""
        wet = self.h > self.dry_depth
        safe = np.where(wet, self.h, 1.0)
        u = np.where(wet, self.hu / safe, 0.0)
        v = np.where(wet, self.hv / safe, 0.0)
        return self.h, u, v

    # ------------------------------------------------------ reconstruction
    @staticmethod
    def _minmod_slope(w: np.ndarray, spacing: float, axis: int,
                      order: int, open_interfaces=None) -> np.ndarray:
        """Minmod slopes along ``axis``; boundary lines get a zero slope.

        ``open_interfaces`` (shape ``n-1`` along ``axis``) is the wet/wet mask:
        reconstruction is switched off across an interface that touches a dry
        cell.  Without that switch the well-balanced property is lost at a
        wet/dry front -- a dry cell carries ``eta = z`` (not the water level),
        so ``eta`` is *discontinuous* there, minmod then extrapolates a
        non-zero ``eta`` into the wet cell and the scheme leaks water onto the
        dry bed (measured: ``max|dh| = 1.3`` for the lake-at-rest test on the
        paraboloid).  This is the standard wet/dry reconstruction limiter
        (Audusse et al. 2004, §3; Xing & Shu 2011).
        """
        slope = np.zeros_like(w)
        if order == 1:
            return slope
        dw = np.diff(w, axis=axis) / spacing
        if open_interfaces is not None:
            dw = np.where(open_interfaces, dw, 0.0)
        if axis == 0:
            left, right = dw[:-1, :], dw[1:, :]
        else:
            left, right = dw[:, :-1], dw[:, 1:]
        limited = np.where(left * right > 0.0,
                           np.sign(left) * np.minimum(np.abs(left),
                                                      np.abs(right)),
                           0.0)
        if axis == 0:
            slope[1:-1, :] = limited
        else:
            slope[:, 1:-1] = limited
        return slope

    @staticmethod
    def _glue_face(first: np.ndarray, last: np.ndarray, inner_l, inner_r,
                   axis: int):
        """Assemble the ``n+1`` face arrays from the ``(left, right)`` states.

        ``first``/``last`` are the boundary-line states (a zero-gradient ghost,
        i.e. transmissive boundary) and ``inner_l``/``inner_r`` hold the
        reconstructed states of the ``n-1`` interior faces along ``axis``.
        """
        shape = list(inner_l.shape)
        shape[axis] += 2
        out_l = np.empty(shape)
        out_r = np.empty(shape)
        if axis == 0:
            out_l[0, :] = first
            out_r[0, :] = first
            out_l[-1, :] = last
            out_r[-1, :] = last
            out_l[1:-1, :] = inner_l
            out_r[1:-1, :] = inner_r
        else:
            out_l[:, 0] = first
            out_r[:, 0] = first
            out_l[:, -1] = last
            out_r[:, -1] = last
            out_l[:, 1:-1] = inner_l
            out_r[:, 1:-1] = inner_r
        return out_l, out_r

    def _face_states(self, axis: int):
        """Hydrostatic-reconstruction face states along ``axis``.

        Returns ``(hL, unL, utL, hR, unR, utR, hsL, hsR)`` where ``un`` is the
        face-normal and ``ut`` the transverse velocity.  ``h*``/``h`` are the
        star/own-side depths used by the pressure correction.
        """
        h, u, v = self.primitive()
        eta = h + self.bed
        spacing = self.dx if axis == 0 else self.dy
        half = 0.5 * spacing
        wet = h > self.dry_depth
        if axis == 0:
            open_face = wet[:-1, :] & wet[1:, :]
        else:
            open_face = wet[:, :-1] & wet[:, 1:]
        se = self._minmod_slope(eta, spacing, axis, self.order, open_face)
        su = self._minmod_slope(u, spacing, axis, self.order, open_face)
        sv = self._minmod_slope(v, spacing, axis, self.order, open_face)

        def split(a):
            if axis == 0:
                return a[:-1, :], a[1:, :]
            return a[:, :-1], a[:, 1:]

        def edge(a):
            idx = 0 if axis == 0 else 1
            if idx == 0:
                return a[0, :], a[-1, :]
            return a[:, 0], a[:, -1]

        eta_l, eta_r = split(eta)
        se_l, se_r = split(se)
        etaL = eta_l + half * se_l
        etaR = eta_r - half * se_r
        head_eta, tail_eta = edge(eta)
        etaLF, etaRF = self._glue_face(head_eta, tail_eta, etaL, etaR, axis)

        u_l, u_r = split(u)
        su_l, su_r = split(su)
        head_u, tail_u = edge(u)
        uLF, uRF = self._glue_face(head_u, tail_u,
                                   u_l + half * su_l, u_r - half * su_r, axis)

        v_l, v_r = split(v)
        sv_l, sv_r = split(sv)
        head_v, tail_v = edge(v)
        vLF, vRF = self._glue_face(head_v, tail_v,
                                   v_l + half * sv_l, v_r - half * sv_r, axis)

        z_l, z_r = split(self.bed)
        head_z, tail_z = edge(self.bed)
        zLF, zRF = self._glue_face(head_z, tail_z, z_l, z_r, axis)

        if self.well_balanced:
            z_face = np.maximum(zLF, zRF)
        else:
            z_face = 0.5 * (zLF + zRF)

        hL = np.maximum(etaLF - zLF, 0.0)
        hR = np.maximum(etaRF - zRF, 0.0)
        if self.well_balanced:
            hsL = np.maximum(etaLF - z_face, 0.0)
            hsR = np.maximum(etaRF - z_face, 0.0)
        else:
            hsL, hsR = hL, hR

        dryL = hsL <= self.dry_depth
        dryR = hsR <= self.dry_depth
        uLF = np.where(dryL, 0.0, uLF)
        vLF = np.where(dryL, 0.0, vLF)
        uRF = np.where(dryR, 0.0, uRF)
        vRF = np.where(dryR, 0.0, vRF)

        if axis == 0:
            return hL, uLF, vLF, hR, uRF, vRF, hsL, hsR
        return hL, vLF, uLF, hR, vRF, uRF, hsL, hsR

    # ---------------------------------------------------------- HLL solver
    def _hll_flux(self, hL, unL, utL, hR, unR, utR):
        """Three-component HLL flux ``(F_h, F_n, F_t)`` of the normal equation."""
        g = self.gravity
        cL = np.sqrt(g * hL)
        cR = np.sqrt(g * hR)
        FhL = hL * unL
        FnL = hL * unL * unL + 0.5 * g * hL * hL
        FtL = hL * unL * utL
        FhR = hR * unR
        FnR = hR * unR * unR + 0.5 * g * hR * hR
        FtR = hR * unR * utR

        sqL = np.sqrt(hL)
        sqR = np.sqrt(hR)
        den = sqL + sqR
        safe_den = np.where(den > 0.0, den, 1.0)
        un_roe = np.where(den > 0.0, (sqL * unL + sqR * unR) / safe_den, 0.0)
        c_roe = np.sqrt(g * 0.5 * (hL + hR))
        sL = np.minimum(unL - cL, un_roe - c_roe)
        sR = np.maximum(unR + cR, un_roe + c_roe)
        gap = sR - sL
        safe_gap = np.where(gap > 1e-300, gap, 1.0)

        def middle(fL, fR, uL_, uR_):
            return (sR * fL - sL * fR + sL * sR * (uR_ - uL_)) / safe_gap

        Fh = np.where(sL >= 0.0, FhL, np.where(sR <= 0.0, FhR,
                                              middle(FhL, FhR, hL, hR)))
        Fn = np.where(sL >= 0.0, FnL, np.where(sR <= 0.0, FnR,
                                              middle(FnL, FnR, hL * unL,
                                                     hR * unR)))
        Ft = np.where(sL >= 0.0, FtL, np.where(sR <= 0.0, FtR,
                                              middle(FtL, FtR, hL * utL,
                                                     hR * utR)))
        dry_pair = (hL <= self.dry_depth) & (hR <= self.dry_depth)
        return (np.where(dry_pair, 0.0, Fh),
                np.where(dry_pair, 0.0, Fn),
                np.where(dry_pair, 0.0, Ft))

    # ------------------------------------------------------------- residual
    def rhs(self, h: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """Semi-discrete right-hand side ``d/dt (h, hu, hv)``."""
        saved = self.h, self.hu, self.hv
        self.h, self.hu, self.hv = h, hu, hv
        try:
            fx = self._face_states(0)
            fy = self._face_states(1)
        finally:
            self.h, self.hu, self.hv = saved

        # The Riemann solver must see the *hydrostatic-reconstruction* star
        # depths (``h* = max(0, eta_side - z_f)``), not the own-side depths.
        # Feeding the own-side depths is exactly the bug that destroys the
        # well-balanced property: on a sloped bed two cells of a lake at rest
        # have different ``h`` but the same ``h*``, so the HLL mass flux is zero
        # only in the starred variables (measured before the fix: a spurious
        # mass flux of 0.15 m^2/s between two still cells).
        hsLx, hsRx = fx[6], fx[7]
        hsLy, hsRy = fy[6], fy[7]
        Fxh, Fxu, Fxv = self._hll_flux(hsLx, fx[1], fx[2], hsRx, fx[4], fx[5])
        Fyh, Fyv, Fyu = self._hll_flux(hsLy, fy[1], fy[2], hsRy, fy[4], fy[5])

        dh = -((Fxh[1:, :] - Fxh[:-1, :]) / self.dx
               + (Fyh[:, 1:] - Fyh[:, :-1]) / self.dy)

        if self.well_balanced:
            g = self.gravity
            # x-faces: cell i-1 sees the left-side correction, cell i the right
            corr_lx = 0.5 * g * (fx[0] * fx[0] - hsLx * hsLx)
            corr_rx = 0.5 * g * (fx[3] * fx[3] - hsRx * hsRx)
            corr_ly = 0.5 * g * (fy[0] * fy[0] - hsLy * hsLy)
            corr_ry = 0.5 * g * (fy[3] * fy[3] - hsRy * hsRy)
            dhu = -(((Fxu + corr_lx)[1:, :] - (Fxu + corr_rx)[:-1, :]) / self.dx
                    + (Fyu[:, 1:] - Fyu[:, :-1]) / self.dy)
            dhv = -((Fxv[1:, :] - Fxv[:-1, :]) / self.dx
                    + ((Fyv + corr_ly)[:, 1:]
                       - (Fyv + corr_ry)[:, :-1]) / self.dy)
        else:
            g = self.gravity
            dzdx = np.gradient(self.bed, self.dx, axis=0)
            dzdy = np.gradient(self.bed, self.dy, axis=1)
            dhu = -((Fxu[1:, :] - Fxu[:-1, :]) / self.dx
                    + (Fyu[:, 1:] - Fyu[:, :-1]) / self.dy) - g * h * dzdx
            dhv = -((Fxv[1:, :] - Fxv[:-1, :]) / self.dx
                    + (Fyv[:, 1:] - Fyv[:, :-1]) / self.dy) - g * h * dzdy
        return dh, dhu, dhv

    # -------------------------------------------------------------- stepping
    def max_wave_speed(self) -> float:
        """``max(|u| + sqrt(g h))`` and the same for ``v``, as one bound."""
        h, u, v = self.primitive()
        c = np.sqrt(self.gravity * h)
        return float(max(np.max(np.abs(u) + c), np.max(np.abs(v) + c)))

    def cfl_number(self) -> float:
        """Dimensionless Courant number of the last accepted step."""
        speed = self.max_wave_speed()
        return speed * self.dt_last / min(self.dx, self.dy)

    def _sanitise(self) -> None:
        """Positivity safeguard; every intervention is counted, not silent."""
        neg = self.h < 0.0
        if np.any(neg):
            self.clipped_cells += int(np.count_nonzero(neg))
            self.h = np.maximum(self.h, 0.0)
        dry = self.h <= self.dry_depth
        if np.any(dry & ((self.hu != 0.0) | (self.hv != 0.0))):
            self.dry_flips += int(np.count_nonzero(
                dry & ((self.hu != 0.0) | (self.hv != 0.0))))
            self.hu = np.where(dry, 0.0, self.hu)
            self.hv = np.where(dry, 0.0, self.hv)
        if not (np.isfinite(self.h).all() and np.isfinite(self.hu).all()
                and np.isfinite(self.hv).all()):
            raise FloatingPointError(
                "non-finite state at t=%.6g, step %d" % (self.time, self.steps))

    def step(self, dt: float) -> None:
        """One SSP-RK3 (Shu-Osher) step of length ``dt``.

        Third order for the same reason as in the 1-D solver: MUSCL + HLL is
        close to skew symmetric in smooth regions, where SSP-RK2 is weakly
        unstable (see ``swe1d.ShallowWater1D.step``).
        """
        h0, hu0, hv0 = self.h, self.hu, self.hv
        k1h, k1u, k1v = self.rhs(h0, hu0, hv0)
        self.h, self.hu, self.hv = h0 + dt * k1h, hu0 + dt * k1u, hv0 + dt * k1v
        self._sanitise()
        h1, hu1, hv1 = self.h, self.hu, self.hv
        k2h, k2u, k2v = self.rhs(h1, hu1, hv1)
        self.h = 0.75 * h0 + 0.25 * (h1 + dt * k2h)
        self.hu = 0.75 * hu0 + 0.25 * (hu1 + dt * k2u)
        self.hv = 0.75 * hv0 + 0.25 * (hv1 + dt * k2v)
        self._sanitise()
        h2, hu2, hv2 = self.h, self.hu, self.hv
        k3h, k3u, k3v = self.rhs(h2, hu2, hv2)
        self.h = (1.0 / 3.0) * h0 + (2.0 / 3.0) * (h2 + dt * k3h)
        self.hu = (1.0 / 3.0) * hu0 + (2.0 / 3.0) * (hu2 + dt * k3u)
        self.hv = (1.0 / 3.0) * hv0 + (2.0 / 3.0) * (hv2 + dt * k3v)
        self._sanitise()
        self.time += dt
        self.steps += 1
        self.dt_last = dt

    def advance(self, t_end: float, max_steps: int = 5_000_000,
                callback=None) -> int:
        """Integrate to ``t_end`` with an adaptive CFL step; returns #steps."""
        if t_end <= self.time:
            return 0
        start_steps = self.steps
        while self.time < t_end - 1e-15:
            speed = self.max_wave_speed()
            if speed <= 0.0:
                raise RuntimeError("no wave speed: the state is at rest")
            dt = self.cfl * min(self.dx, self.dy) / speed
            dt = min(dt, t_end - self.time)
            self.step(dt)
            if callback is not None:
                callback(self)
            if self.steps - start_steps > max_steps:
                raise RuntimeError("step budget exceeded (%d)" % max_steps)
        return self.steps - start_steps

    # ---------------------------------------------------------- diagnostics
    def total_volume(self) -> float:
        """``sum(h) dx dy`` -- the conserved water volume."""
        return float(self.h.sum()) * self.dx * self.dy

    def max_spurious_velocity(self) -> float:
        """``max sqrt(u^2 + v^2)`` -- the lake-at-rest thermometer."""
        _, u, v = self.primitive()
        return float(np.max(np.hypot(u, v)))

    def surface_drift(self) -> float:
        """``max |eta - eta_0|`` against the initial surface."""
        return float(np.max(np.abs(self.water_surface() - self._eta0)))

    def summary(self) -> dict:
        h, u, v = self.primitive()
        return {
            "nx": self.nx,
            "ny": self.ny,
            "dx": self.dx,
            "dy": self.dy,
            "g": self.gravity,
            "cfl": self.cfl,
            "order": self.order,
            "well_balanced": self.well_balanced,
            "t": self.time,
            "steps": self.steps,
            "h_min": float(h.min()),
            "h_max": float(h.max()),
            "speed_max": float(np.max(np.hypot(u, v))),
            "volume": self.total_volume(),
            "clipped_cells": self.clipped_cells,
            "dry_flips": self.dry_flips,
            "bed_min": float(self.bed.min()),
            "bed_max": float(self.bed.max()),
        }


def _finite(value: float, name: str) -> float:
    """Small guard used by callers that build parameters from JSON."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("%s must be finite, got %r" % (name, value))
    return value