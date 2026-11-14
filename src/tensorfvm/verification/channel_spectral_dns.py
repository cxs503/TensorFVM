"""Closure-free forced-channel solver for the KMM180 anchor.

Why this module exists
----------------------
``docs/oss_solutions/CASE-FVM-CHANNEL-KMM180.json`` grades ``u+(y+)`` against a
channel DNS reference with a 5 % RMS criterion.  Every route implemented so far
is a *Reynolds-averaged* one: a 1-D equilibrium model (``rans_validation``) in
which the answer is produced by a turbulence *closure*.  The error budget
(``kmm180_error_budget``) shows that the anchor's FAIL in the physical log layer
is carried by ``nu_t`` of the closure, and its "limits" section states the
consequence explicitly: *"the budget is one-dimensional by construction"*, so
nothing in the repository could decide whether ``u+`` is produced correctly **by
the discretisation, independently of a closure**.

This module supplies that missing route: it integrates the incompressible
Navier--Stokes equations in a plane channel driven by a constant streamwise body
force with **no turbulence model at all**.  Because the driving force and the
friction velocity are related exactly by ``u_tau^2 = G h`` (unit density), the
wall units are fixed by construction: the computed velocity field is already in
wall units, so ``u+`` involves no wall-shear estimator.

Discretisation
--------------
* ``x`` (streamwise) and ``z`` (spanwise): **Fourier** collocation on a uniform
  periodic grid (real FFT);
* ``y`` (wall-normal): staggered **finite volume** -- ``u``, ``w``, ``p`` at cell
  centres, ``v`` at the cell faces, ``v = 0`` on both walls -- with the pressure
  Poisson operator written in the same flux form as the divergence operator, so
  the projection is divergence-free to round-off (measured in the tests and in
  the report);
* time: three-stage SSP-RK3 for convection, wall-normal diffusion and forcing;
  the periodic-plane diffusion is integrated exactly in Fourier space (Strang
  splitting), which removes the ``nu k^2 dt`` restriction and leaves only the
  wall-normal viscous and the convective limits;
* the Nyquist mode of each real transform is truncated (it is its own
  conjugate, so its spectral derivative is not representable and the projection
  could not remove divergence there).

Limits
------
At ``Re_tau = 180`` a wall-resolved DNS needs roughly ``dx+ < 10``,
``dz+ < 5``, ``dy+_1 < 0.5``; a grid below that is an **under-resolved
no-closure** run (numerically dissipative), which is a legitimate but weaker
statement.  :meth:`SpectralChannel.resolution_report` reports the wall-unit
spacings so a report can say which of the two regimes was run.  Only the
wall-normal direction is a finite-volume discretisation (the periodic
directions are spectral), so a verdict from this module is reported as a
*closure-free reference*, never as an FVM grid-convergence claim.

Everything here is deterministic given ``seed``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

import numpy as np
import scipy.fft as sfft

__all__ = [
    'ChannelConfig',
    'SpectralChannel',
    'ChannelStatistics',
    'StatisticsAccumulator',
    'build_wall_normal_grid',
    'fold_half_channel',
    'total_shear_identity',
]


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ChannelConfig:
    """Geometry, resolution and time step of the forced-channel solver.

    All lengths are in units of the channel half width ``h``; velocities are in
    units of the friction velocity ``u_tau`` (fixed exactly by the constant body
    force, ``u_tau = sqrt(G h)``), so code-unit velocities **are** ``u+``.
    """

    re_tau: float = 180.0
    nx: int = 64
    ny: int = 128
    nz: int = 64
    length_x: float = 2.0 * math.pi
    length_z: float = math.pi
    stretching: float = 1.5
    time_step: Optional[float] = None
    fft_workers: int = 1

    def __post_init__(self):
        if not math.isfinite(self.re_tau) or self.re_tau <= 0:
            raise ValueError('re_tau must be positive and finite')
        for name in ('nx', 'ny', 'nz'):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 8:
                raise ValueError(f'{name} must be an int >= 8')
            if name != 'ny' and value % 2:
                raise ValueError(f'{name} must be even for the real FFT')
        if self.ny % 2:
            raise ValueError('ny must be even so the grid is mirror-symmetric')
        for name in ('length_x', 'length_z'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be positive and finite')
        if not math.isfinite(self.stretching) or self.stretching <= 0:
            raise ValueError('stretching must be positive and finite')
        if self.time_step is not None and not (math.isfinite(self.time_step)
                                               and self.time_step > 0):
            raise ValueError('time_step must be positive and finite')
        if not isinstance(self.fft_workers, int) or self.fft_workers < 1:
            raise ValueError('fft_workers must be an int >= 1')

    @property
    def half_width(self) -> float:
        return 1.0

    @property
    def u_tau(self) -> float:
        """Friction velocity implied by the body force (unit by construction)."""
        return 1.0

    @property
    def body_force(self) -> float:
        """Constant streamwise force ``G = u_tau^2 / h`` (unit density)."""
        return self.u_tau ** 2 / self.half_width

    @property
    def nu(self) -> float:
        """Kinematic viscosity for the requested ``Re_tau = u_tau h / nu``."""
        return self.u_tau * self.half_width / self.re_tau


# --------------------------------------------------------------------------- #
# Grid and finite-difference stencils
# --------------------------------------------------------------------------- #

def _tanh_map(xi: np.ndarray, a: float) -> np.ndarray:
    """Map ``[0, 1] -> [0, 1]``, clustering points at both ends."""
    return 0.5 * (1.0 + np.tanh(a * (2.0 * xi - 1.0)) / math.tanh(a))


def build_wall_normal_grid(ny: int, half_width: float, stretching: float):
    """Cell-centre and face coordinates of the stretched wall-normal grid."""
    centres = _tanh_map((np.arange(ny) + 0.5) / ny, stretching)
    faces = _tanh_map(np.arange(ny + 1) / ny, stretching)
    y = -half_width + 2.0 * half_width * centres
    yf = -half_width + 2.0 * half_width * faces
    yf[0], yf[-1] = -half_width, half_width
    return y, yf


def _second_derivative(values: np.ndarray, y: np.ndarray, yf: np.ndarray,
                       lower_value: float = 0.0,
                       upper_value: float = 0.0) -> np.ndarray:
    """Three-point second derivative at centres with prescribed wall values."""
    out = np.empty_like(values)
    pad = (slice(None),) + (None,) * (values.ndim - 1)
    h1 = y[0] - yf[0]
    h2 = y[1] - y[0]
    out[0] = 2.0 / (h1 + h2) * ((values[1] - values[0]) / h2
                                - (values[0] - lower_value) / h1)
    h1i = y[1:-1] - y[0:-2]
    h2i = y[2:] - y[1:-1]
    out[1:-1] = (2.0 / (h1i + h2i))[pad] * (
        (values[2:] - values[1:-1]) / h2i[pad]
        - (values[1:-1] - values[0:-2]) / h1i[pad])
    h1t = y[-1] - y[-2]
    h2t = yf[-1] - y[-1]
    out[-1] = 2.0 / (h1t + h2t) * ((upper_value - values[-1]) / h2t
                                   - (values[-1] - values[-2]) / h1t)
    return out


def _first_derivative(values: np.ndarray, y: np.ndarray, yf: np.ndarray,
                      lower_value: float = 0.0,
                      upper_value: float = 0.0) -> np.ndarray:
    """Three-point (non-uniform) first derivative at centres."""
    out = np.empty_like(values)
    pad = (slice(None),) + (None,) * (values.ndim - 1)
    h1 = y[0] - yf[0]
    h2 = y[1] - y[0]
    out[0] = (-h2 / (h1 * (h1 + h2)) * lower_value
              + (h2 - h1) / (h1 * h2) * values[0]
              + h1 / (h2 * (h1 + h2)) * values[1])
    h1i = y[1:-1] - y[0:-2]
    h2i = y[2:] - y[1:-1]
    denom = h1i * h2i * (h1i + h2i)
    out[1:-1] = (-(h2i ** 2)[pad] * values[0:-2]
                 + (h2i ** 2 - h1i ** 2)[pad] * values[1:-1]
                 + (h1i ** 2)[pad] * values[2:]) / denom[pad]
    h1t = y[-1] - y[-2]
    h2t = yf[-1] - y[-1]
    out[-1] = (-h2t / (h1t * (h1t + h2t)) * values[-2]
               + (h2t - h1t) / (h1t * h2t) * values[-1]
               + h1t / (h2t * (h1t + h2t)) * upper_value)
    return out


def _face_second_derivative(values: np.ndarray, yf: np.ndarray) -> np.ndarray:
    """Second derivative at the interior faces (``v`` is staggered to the faces)."""
    out = np.zeros_like(values)
    pad = (slice(None),) + (None,) * (values.ndim - 1)
    h1 = yf[1:-1] - yf[0:-2]
    h2 = yf[2:] - yf[1:-1]
    out[1:-1] = (2.0 / (h1 + h2))[pad] * (
        (values[2:] - values[1:-1]) / h2[pad]
        - (values[1:-1] - values[0:-2]) / h1[pad])
    return out


def _face_first_derivative(values: np.ndarray, yf: np.ndarray) -> np.ndarray:
    """First derivative at the interior faces."""
    out = np.zeros_like(values)
    pad = (slice(None),) + (None,) * (values.ndim - 1)
    h1 = yf[1:-1] - yf[0:-2]
    h2 = yf[2:] - yf[1:-1]
    denom = h1 * h2 * (h1 + h2)
    out[1:-1] = (-(h2 ** 2)[pad] * values[0:-2]
                 + (h2 ** 2 - h1 ** 2)[pad] * values[1:-1]
                 + (h1 ** 2)[pad] * values[2:]) / denom[pad]
    return out


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #

class SpectralChannel:
    """Forced plane channel: Fourier ``x,z``, finite-volume ``y``, no closure.

    State: ``u``, ``w`` at cell centres ``(ny, nx, nz)``; ``v`` at the faces
    ``(ny + 1, nx, nz)`` with ``v = 0`` on both walls.
    """

    def __init__(self, config: ChannelConfig):
        self.config = config
        self.h = config.half_width
        self.nu = config.nu
        self.u_tau = config.u_tau
        self.force = config.body_force
        self.re_tau = config.re_tau

        self.y, self.yf = build_wall_normal_grid(config.ny, self.h,
                                                 config.stretching)
        self.dy = np.diff(self.yf)
        self.y_plus = self.re_tau * (self.y + self.h) / self.h

        kx = 2.0 * math.pi * np.fft.fftfreq(config.nx, d=config.length_x / config.nx)
        kz = 2.0 * math.pi * np.fft.rfftfreq(config.nz, d=config.length_z / config.nz)
        self.kx = kx[:, None]
        self.kz = kz[None, :]
        self.k2 = self.kx ** 2 + self.kz ** 2
        self.k2_3d = self.k2[None, :, :]
        self.kmag_max = float(np.sqrt(self.k2.max()))

        # The Nyquist mode of a real transform is its own conjugate, so its
        # spectral derivative is not representable and the projection cannot
        # remove divergence there.  Spectral channel codes truncate that mode;
        # this mask applies that rule explicitly to every velocity field.
        mask = np.ones((config.nx, config.nz // 2 + 1))
        mask[config.nx // 2, :] = 0.0
        mask[:, config.nz // 2] = 0.0
        self._nyquist_mask = mask[None, :, :]

        self.u = np.zeros((config.ny, config.nx, config.nz))
        self.v = np.zeros((config.ny + 1, config.nx, config.nz))
        self.w = np.zeros((config.ny, config.nx, config.nz))

        self._dt = config.time_step if config.time_step else self.stable_time_step()
        self.steps = 0
        self.time = 0.0
        self.divergence_max = 0.0

    # -- time step -----------------------------------------------------------
    def stable_time_step(self, safety: float = 0.6,
                         velocity_scale: Optional[float] = None) -> float:
        """Step limit from wall-normal diffusion and from the spectral CFL.

        The wall-normal diffusion is explicit, so the smallest cell sets
        ``dt <= safety * dy_min^2 / (2 nu)``; convection is evaluated with
        spectral derivatives, so the usual ``u_max k_max dt`` bound applies too.
        ``velocity_scale`` defaults to ``20 u_tau``, the level a channel profile
        reaches at ``Re_tau = 180``.
        """
        viscous = safety * 0.5 * float(np.min(self.dy)) ** 2 / self.nu
        scale = 20.0 * self.u_tau if velocity_scale is None else velocity_scale
        courant = 0.6 / (scale * self.kmag_max)
        return float(min(viscous, courant))

    def courant_time_step(self, safety: float = 0.6) -> float:
        """CFL step of the *current* state."""
        umax = max(float(np.abs(self.u).max()), float(np.abs(self.v).max()),
                   float(np.abs(self.w).max()), 1e-12)
        return safety / (umax * self.kmag_max)

    @property
    def time_step(self) -> float:
        return self._dt

    def set_time_step(self, dt: float) -> None:
        if not (math.isfinite(dt) and dt > 0):
            raise ValueError('time step must be positive and finite')
        self._dt = float(dt)

    # -- spectral helpers ----------------------------------------------------
    def _to_spectral(self, field: np.ndarray) -> np.ndarray:
        return sfft.rfft2(field, axes=(-2, -1), workers=self.config.fft_workers)

    def _from_spectral(self, coeff: np.ndarray, shape) -> np.ndarray:
        return sfft.irfft2(coeff, s=shape, axes=(-2, -1),
                           workers=self.config.fft_workers)

    def dx(self, field: np.ndarray) -> np.ndarray:
        return self._from_spectral(1j * self.kx * self._to_spectral(field),
                                   field.shape[-2:])

    def dz(self, field: np.ndarray) -> np.ndarray:
        return self._from_spectral(1j * self.kz * self._to_spectral(field),
                                   field.shape[-2:])

    def divergence(self, u=None, v=None, w=None) -> np.ndarray:
        u = self.u if u is None else u
        v = self.v if v is None else v
        w = self.w if w is None else w
        n = u.ndim - 1
        div = (self._from_spectral(1j * self.kx * self._to_spectral(u), u.shape[-2:])
               + self._from_spectral(1j * self.kz * self._to_spectral(w), w.shape[-2:]))
        div += (v[1:] - v[:-1]) / self.dy[(slice(None),) + (None,) * n]
        return div

    # -- staggered interpolation --------------------------------------------
    def v_at_centres(self, v=None) -> np.ndarray:
        v = self.v if v is None else v
        return 0.5 * (v[:-1] + v[1:])

    def _at_faces(self, field: np.ndarray) -> np.ndarray:
        faces = np.zeros((field.shape[0] + 1,) + field.shape[1:])
        faces[1:-1] = 0.5 * (field[:-1] + field[1:])
        return faces

    def u_at_faces(self, u=None) -> np.ndarray:
        return self._at_faces(self.u if u is None else u)

    def w_at_faces(self, w=None) -> np.ndarray:
        return self._at_faces(self.w if w is None else w)

    # -- differential operators ---------------------------------------------
    def wall_normal_laplacian(self, field: np.ndarray) -> np.ndarray:
        return _second_derivative(field, self.y, self.yf)

    def laplacian(self, field: np.ndarray) -> np.ndarray:
        """Full Laplacian: spectral in ``x, z`` plus finite-volume in ``y``."""
        return (self._from_spectral(-self.k2 * self._to_spectral(field),
                                    field.shape[-2:])
                + self.wall_normal_laplacian(field))

    def wall_normal_derivative(self, field: np.ndarray) -> np.ndarray:
        return _first_derivative(field, self.y, self.yf)

    # -- right-hand side -----------------------------------------------------
    def rhs(self):
        """Convection + wall-normal diffusion + body force.

        The periodic-plane part of the viscous term is integrated exactly in
        :meth:`step`, so it does not appear here.  Each velocity field is
        transformed once and every horizontal derivative is recovered from that
        single set of coefficients.
        """
        shape_u, shape_v = self.u.shape[-2:], self.v.shape[-2:]
        n = self.u.ndim - 1
        pad = (slice(None),) + (None,) * n

        U = self._to_spectral(self.u)
        W = self._to_spectral(self.w)
        V = self._to_spectral(self.v)
        ux = self._from_spectral(1j * self.kx * U, shape_u)
        uz = self._from_spectral(1j * self.kz * U, shape_u)
        wx = self._from_spectral(1j * self.kx * W, shape_u)
        wz = self._from_spectral(1j * self.kz * W, shape_u)
        vx = self._from_spectral(1j * self.kx * V, shape_v)
        vz = self._from_spectral(1j * self.kz * V, shape_v)

        vc = self.v_at_centres(self.v)
        uf, wf = self._at_faces(self.u), self._at_faces(self.w)

        rhs_u = (-(self.u * ux + vc * self.wall_normal_derivative(self.u)
                   + self.w * uz)
                 + self.nu * self.wall_normal_laplacian(self.u)
                 + self.force)
        rhs_u += self._from_spectral(-self.nu * self.k2 * U, shape_u)

        rhs_w = (-(self.u * wx + vc * self.wall_normal_derivative(self.w)
                   + self.w * wz)
                 + self.nu * self.wall_normal_laplacian(self.w))
        rhs_w += self._from_spectral(-self.nu * self.k2 * W, shape_u)

        rhs_v = (-(uf * vx + self.v * _face_first_derivative(self.v, self.yf)
                   + wf * vz))
        rhs_v += self.nu * _face_second_derivative(self.v, self.yf)
        rhs_v += self._from_spectral(-self.nu * self.k2 * V, shape_v)
        rhs_v[0] = 0.0
        rhs_v[-1] = 0.0
        return rhs_u, rhs_v, rhs_w

    # -- exact diffusion in the periodic plane -------------------------------
    def _plane_diffuse(self, decay: np.ndarray) -> None:
        """Multiply every mode by ``exp(-nu (kx^2+kz^2) dt)`` exactly."""
        for name in ('u', 'w'):
            field = getattr(self, name)
            setattr(self, name, self._from_spectral(decay * self._to_spectral(field),
                                                    field.shape[-2:]))
        self.v = self._from_spectral(decay * self._to_spectral(self.v),
                                     self.v.shape[-2:])
        self.v[0] = 0.0
        self.v[-1] = 0.0

    def truncate(self) -> None:
        """Remove the (unrepresentable) Nyquist modes of every velocity field."""
        for name in ('u', 'w'):
            field = getattr(self, name)
            setattr(self, name, self._from_spectral(
                self._nyquist_mask * self._to_spectral(field), field.shape[-2:]))
        self.v = self._from_spectral(self._nyquist_mask * self._to_spectral(self.v),
                                     self.v.shape[-2:])
        self.v[0] = 0.0
        self.v[-1] = 0.0

    # -- pressure projection -------------------------------------------------
    def _poisson_matrix(self):
        """Tridiagonal coefficients of ``(d2/dy2 - k^2)`` with Neumann walls."""
        ny = self.config.ny
        d = np.diff(self.y)
        lower = np.zeros(ny)
        diag = np.zeros(ny)
        upper = np.zeros(ny)
        lower[1:] = 1.0 / (d * self.dy[1:])
        upper[:-1] = 1.0 / (d * self.dy[:-1])
        diag[0] = -1.0 / (d[0] * self.dy[0])
        diag[-1] = -1.0 / (d[-1] * self.dy[-1])
        diag[1:-1] = -(1.0 / d[1:] + 1.0 / d[:-1]) / self.dy[1:-1]
        return lower, diag, upper

    def project(self, dt: Optional[float] = None) -> float:
        """Make ``(u, v, w)`` divergence-free; returns the pre-projection residual."""
        dt = self._dt if dt is None else dt
        self.truncate()
        div = self.divergence()
        residual = float(np.max(np.abs(div)))

        ny = self.config.ny
        nmode = self.config.nx * (self.config.nz // 2 + 1)
        lower1, diag1, upper1 = self._poisson_matrix()
        k2 = self.k2.reshape(-1)
        lower = np.repeat(lower1[:, None], nmode, axis=1)
        upper = np.repeat(upper1[:, None], nmode, axis=1)
        diag = np.repeat(diag1[:, None], nmode, axis=1) - k2[None, :]

        rhs = (self._to_spectral(div) / dt).reshape(ny, nmode)

        # The (kx, kz) = (0, 0) mode is a pure-Neumann 1-D problem whose
        # compatibility condition, sum_j dy_j div_j = v(wall) - v(wall) = 0,
        # holds exactly.  It is solved with the datum phi_0 = 0 instead of being
        # discarded: discarding it would leave the plane-mean divergence behind.
        zero = int(np.argmin(k2))
        diag[0, zero] = 1.0
        lower[0, zero] = 0.0
        upper[0, zero] = 0.0
        rhs[0, zero] = 0.0

        phi = self._solve_tridiagonal(lower, diag, upper, rhs)
        phi = phi.reshape(ny, self.config.nx, self.config.nz // 2 + 1)

        self.u -= dt * self._from_spectral(1j * self.kx * phi, self.u.shape[-2:])
        self.w -= dt * self._from_spectral(1j * self.kz * phi, self.u.shape[-2:])
        phi_real = self._from_spectral(phi, self.u.shape[-2:])
        dphi_dy = np.zeros((ny + 1,) + phi_real.shape[1:])
        dphi_dy[1:-1] = (phi_real[1:] - phi_real[:-1]) / np.diff(self.y)[
            (slice(None),) + (None,) * (phi_real.ndim - 1)]
        self.v -= dt * dphi_dy
        self.v[0] = 0.0
        self.v[-1] = 0.0

        after = float(np.max(np.abs(self.divergence())))
        self.divergence_max = max(self.divergence_max, after)
        return residual

    @staticmethod
    def _solve_tridiagonal(lower, diag, upper, rhs):
        """Vectorised Thomas algorithm over the trailing mode axis."""
        n = diag.shape[0]
        cp = np.empty_like(diag)
        dp = np.empty_like(rhs)
        cp[0] = upper[0] / diag[0]
        dp[0] = rhs[0] / diag[0]
        for j in range(1, n):
            m = diag[j] - lower[j] * cp[j - 1]
            cp[j] = upper[j] / m
            dp[j] = (rhs[j] - lower[j] * dp[j - 1]) / m
        x = np.empty_like(rhs)
        x[-1] = dp[-1]
        for j in range(n - 2, -1, -1):
            x[j] = dp[j] - cp[j] * x[j + 1]
        return x

    # -- reference state -----------------------------------------------------
    def laminar_profile(self) -> np.ndarray:
        """Discrete steady state of the forced laminar channel.

        Solves ``nu d2u/dy2 = -G`` with no-slip walls **with exactly the stencil
        that :meth:`rhs` uses**, so starting the solver from this profile is a
        fixed-point test of the momentum operator (it must not move), while
        comparing it with the analytic parabola is a truncation-error test of the
        stretched grid.
        """
        ny = self.config.ny
        y, yf = self.y, self.yf
        h1 = np.empty(ny)
        h2 = np.empty(ny)
        h1[0] = y[0] - yf[0]
        h2[0] = y[1] - y[0]
        h1[1:-1] = y[1:-1] - y[0:-2]
        h2[1:-1] = y[2:] - y[1:-1]
        h1[-1] = y[-1] - y[-2]
        h2[-1] = yf[-1] - y[-1]
        lower = np.zeros((ny, 1))
        diag = np.zeros((ny, 1))
        upper = np.zeros((ny, 1))
        lower[1:, 0] = 2.0 / ((h1 + h2) * h1)[1:]
        upper[:-1, 0] = 2.0 / ((h1 + h2) * h2)[:-1]
        diag[:, 0] = -(2.0 / (h1 + h2)) * (1.0 / h1 + 1.0 / h2)
        rhs = np.full((ny, 1), -self.force / self.nu)
        return self._solve_tridiagonal(lower, diag, upper, rhs)[:, 0]

    # -- initial condition ---------------------------------------------------
    def initialize(self, *, noise: float = 0.1, seed: int = 1,
                   mean_profile: str = 'blended') -> None:
        """Mean profile plus a divergence-free random perturbation.

        ``blended`` is the composite law ``u+ = y+`` in the sublayer joined to
        the log law ``ln(y+)/0.41 + 5.2`` through a ``tanh`` weight around
        ``y+ = 11``; it is only an initial condition (it relaxes within a few
        flow-through times), not a reference.
        """
        if noise < 0:
            raise ValueError('noise must be non-negative')
        yp = np.maximum(self.y_plus, 1e-6)
        if mean_profile == 'blended':
            log_law = np.log(yp) / 0.41 + 5.2
            weight = 0.5 * (1.0 + np.tanh((yp - 11.0) / 3.0))
            profile = yp * (1.0 - weight) + log_law * weight
        elif mean_profile == 'loglaw':
            profile = np.log1p(0.41 * yp) / 0.41 + 5.2
        elif mean_profile == 'parabolic':
            profile = self.re_tau * (1.0 - ((self.y + self.h) / self.h - 1.0) ** 2)
        else:
            raise ValueError(f'unknown mean profile {mean_profile!r}')
        half = profile[: self.config.ny // 2]
        profile = np.concatenate([half, half[::-1]])
        ny, nx, nz = self.u.shape
        self.u = np.repeat(profile[:, None, None], nx, axis=1)
        self.u = np.repeat(self.u, nz, axis=2).copy()
        rng = np.random.default_rng(seed)
        self.u += noise * rng.standard_normal(self.u.shape)
        self.v = 0.5 * noise * rng.standard_normal(self.v.shape)
        self.w = noise * rng.standard_normal(self.w.shape)
        self.v[0] = 0.0
        self.v[-1] = 0.0
        self.project(dt=min(1e-3, self._dt))
        self.steps = 0
        self.time = 0.0
        self.divergence_max = 0.0

    # -- time integration ----------------------------------------------------
    def _rk3_stage(self, dt: float) -> None:
        u0, v0, w0 = self.u.copy(), self.v.copy(), self.w.copy()

        r_u, r_v, r_w = self.rhs()
        u1, v1, w1 = u0 + dt * r_u, v0 + dt * r_v, w0 + dt * r_w
        v1[0] = v1[-1] = 0.0

        self.u, self.v, self.w = u1, v1, w1
        r_u, r_v, r_w = self.rhs()
        u2 = 0.75 * u0 + 0.25 * (u1 + dt * r_u)
        v2 = 0.75 * v0 + 0.25 * (v1 + dt * r_v)
        w2 = 0.75 * w0 + 0.25 * (w1 + dt * r_w)
        v2[0] = v2[-1] = 0.0

        self.u, self.v, self.w = u2, v2, w2
        r_u, r_v, r_w = self.rhs()
        u3 = (1.0 / 3.0) * u0 + (2.0 / 3.0) * (u2 + dt * r_u)
        v3 = (1.0 / 3.0) * v0 + (2.0 / 3.0) * (v2 + dt * r_v)
        w3 = (1.0 / 3.0) * w0 + (2.0 / 3.0) * (w2 + dt * r_w)
        v3[0] = v3[-1] = 0.0
        self.u, self.v, self.w = u3, v3, w3

    def step(self, dt: Optional[float] = None) -> None:
        """Advance one time step (Strang diffusion + SSP-RK3 + projection)."""
        dt = self._dt if dt is None else dt
        decay_half = np.exp(-self.nu * self.k2_3d * (0.5 * dt))
        self._plane_diffuse(decay_half)
        self._rk3_stage(dt)
        self._plane_diffuse(decay_half)
        self.project(dt)
        self.steps += 1
        self.time += dt
        if not np.isfinite(self.u).all():
            raise FloatingPointError('velocity field became non-finite; '
                                     'reduce the time step')

    def run(self, n_steps: int, dt: Optional[float] = None,
            callback=None) -> None:
        for _ in range(int(n_steps)):
            self.step(dt)
            if callback is not None:
                callback(self)

    # -- diagnostics ---------------------------------------------------------
    def bulk_velocity(self) -> float:
        """Volume-averaged streamwise velocity (already in wall units)."""
        plane = self.u.mean(axis=(1, 2))
        return float(np.sum(plane * self.dy) / np.sum(self.dy))

    def wall_shear(self) -> float:
        """Discrete wall shear ``nu du/dy`` at the lower wall (should be ``u_tau``)."""
        du_dy = _first_derivative(self.u, self.y, self.yf)
        return float(self.nu * du_dy[0].mean())

    def mean_velocity_profile(self) -> np.ndarray:
        return self.u.mean(axis=(1, 2))

    def resolution_report(self) -> Dict[str, float]:
        """Wall-unit spacings and the resolved range of scales."""
        h, re_tau = self.h, self.re_tau
        dy_plus_min = float(np.min(self.dy) * re_tau / h)
        dx_plus = self.config.length_x / self.config.nx * re_tau / h
        dz_plus = self.config.length_z / self.config.nz * re_tau / h
        kmax_plus = self.kmag_max * h / re_tau
        return dict(nx=int(self.config.nx), ny=int(self.config.ny),
                    nz=int(self.config.nz),
                    length_x_plus=float(self.config.length_x * re_tau / h),
                    length_z_plus=float(self.config.length_z * re_tau / h),
                    dy_plus_min=dy_plus_min,
                    dy_plus_centre=float(np.max(self.dy) * re_tau / h),
                    dx_plus=float(dx_plus), dz_plus=float(dz_plus),
                    kmax_plus=float(kmax_plus),
                    kmax_eta_wall=float(kmax_plus),
                    cells=int(self.config.nx * self.config.ny * self.config.nz),
                    time_step=float(self._dt),
                    dt_plus=float(self._dt * re_tau ** 2 / h ** 2),
                    divergence_max=float(self.divergence_max))


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

@dataclass
class ChannelStatistics:
    """Time- and (x, z)-averaged one-point statistics, folded onto one half."""

    re_tau: float
    y_plus: np.ndarray
    u_plus: np.ndarray
    du_plus_dy_plus: np.ndarray
    uu_plus: np.ndarray
    vv_plus: np.ndarray
    ww_plus: np.ndarray
    uv_plus: np.ndarray
    samples: int
    time_span: float

    def as_rows(self) -> Iterator[dict]:
        for i in range(self.y_plus.size):
            yield dict(y_plus=float(self.y_plus[i]),
                       u_plus=float(self.u_plus[i]),
                       du_plus_dy_plus=float(self.du_plus_dy_plus[i]),
                       u_rms_plus=float(math.sqrt(max(self.uu_plus[i], 0.0))),
                       v_rms_plus=float(math.sqrt(max(self.vv_plus[i], 0.0))),
                       w_rms_plus=float(math.sqrt(max(self.ww_plus[i], 0.0))),
                       uv_plus=float(self.uv_plus[i]))


class StatisticsAccumulator:
    """Online ``(x, z, t)`` averages of the moments needed for the stresses."""

    def __init__(self, solver: SpectralChannel):
        self.solver = solver
        ny = solver.config.ny
        self.n = 0
        self.time_span = 0.0
        self._sum_u = np.zeros(ny)
        self._sum_v = np.zeros(ny)
        self._sum_w = np.zeros(ny)
        self._sum_uu = np.zeros(ny)
        self._sum_vv = np.zeros(ny)
        self._sum_ww = np.zeros(ny)
        self._sum_uv = np.zeros(ny)

    def add(self, dt: Optional[float] = None) -> None:
        solver = self.solver
        vc = solver.v_at_centres()
        self._sum_u += solver.u.mean(axis=(1, 2))
        self._sum_v += vc.mean(axis=(1, 2))
        self._sum_w += solver.w.mean(axis=(1, 2))
        self._sum_uu += (solver.u ** 2).mean(axis=(1, 2))
        self._sum_vv += (vc ** 2).mean(axis=(1, 2))
        self._sum_ww += (solver.w ** 2).mean(axis=(1, 2))
        self._sum_uv += (solver.u * vc).mean(axis=(1, 2))
        self.n += 1
        self.time_span += float(solver.time_step if dt is None else dt)

    def profiles(self) -> Dict[str, np.ndarray]:
        """Mean and (co)variances from the running sums.

        ``<u'v'> = <uv> - <u><v>``: the fluctuation product is formed from the
        raw moments, so no assumption ``<v> = 0`` enters.
        """
        if self.n < 4:
            raise RuntimeError('need at least four samples for statistics')
        n = float(self.n)
        mean_u = self._sum_u / n
        mean_v = self._sum_v / n
        mean_w = self._sum_w / n
        return dict(u=mean_u,
                    uu=np.maximum(self._sum_uu / n - mean_u ** 2, 0.0),
                    vv=np.maximum(self._sum_vv / n - mean_v ** 2, 0.0),
                    ww=np.maximum(self._sum_ww / n - mean_w ** 2, 0.0),
                    uv=self._sum_uv / n - mean_u * mean_v)

    def folded(self) -> ChannelStatistics:
        return fold_half_channel(self.solver, self.profiles(),
                                 samples=self.n, time_span=self.time_span)


def fold_half_channel(solver: SpectralChannel, profiles: Dict[str, np.ndarray],
                      *, samples: int, time_span: float) -> ChannelStatistics:
    """Average the two statistically equivalent halves of the channel.

    The mean flow is symmetric about the centreline, so the lower-half value at
    ``Y`` and the mirrored upper-half value at ``2h - Y`` sample the same
    statistics; folding halves the sampling error and removes the residual
    asymmetry of a finite-time average.
    """
    ny = solver.config.ny
    half = ny // 2
    lower = np.arange(half)
    partner = np.arange(ny - 1, ny - 1 - half, -1)
    mean_u = profiles['u']

    def fold(field):
        return 0.5 * (field[lower] + field[partner])

    u_folded = fold(mean_u)
    full = np.concatenate([u_folded, u_folded[::-1]])
    du_dy = _first_derivative(full[:, None, None], solver.y, solver.yf)[:half, 0, 0]

    return ChannelStatistics(
        re_tau=solver.re_tau,
        y_plus=solver.y_plus[lower],
        u_plus=u_folded,
        du_plus_dy_plus=du_dy / solver.re_tau,
        uu_plus=fold(profiles['uu']),
        vv_plus=fold(profiles['vv']),
        ww_plus=fold(profiles['ww']),
        uv_plus=fold(profiles['uv']),
        samples=int(samples),
        time_span=float(time_span),
    )


def total_shear_identity(stats: ChannelStatistics) -> Dict[str, np.ndarray]:
    """Closure-free check ``du+/dy+ - <u'v'>+ = 1 - Y/h``.

    For a statistically stationary forced channel the mean momentum balance is
    exact and independent of any turbulence model, so this residual measures the
    *solver's* momentum bookkeeping rather than a closure error.  Here
    ``Y/h = y+/Re_tau``.
    """
    y_plus = stats.y_plus
    total = stats.du_plus_dy_plus - stats.uv_plus
    target = 1.0 - y_plus / stats.re_tau
    return dict(y_plus=y_plus, total=total, target=target,
                residual=total - target)