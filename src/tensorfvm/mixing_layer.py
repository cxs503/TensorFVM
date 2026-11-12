"""Temporally developing plane mixing layer: solver, diagnostics and stability reference.

The case is the classical temporal mixing layer: the two-stream shear layer
``u0(y) = U tanh(y/delta)`` (free streams ``-U`` and ``+U``, velocity difference
``dU = 2U``) evolves in the frame moving with the layer, on a box that is
periodic in the streamwise direction ``x`` and bounded by free-slip walls in the
transverse direction ``y`` at ``y = +-H/2``.

Discretisation (the plainest second-order scheme):

* staggered MAC grid -- ``u`` at x faces, ``v`` at y faces, ``p`` at cell centres;
* rotational (skew) form central convection plus 2nd-order central diffusion,
  explicit Heun (RK2) stepping;
* exact discrete projection: the pressure Poisson problem is diagonalised by an
  FFT in the periodic ``x`` direction and a DCT-II in the free-slip ``y``
  direction (the same transform pair :mod:`tensorfvm.incompressible_benchmarks`
  uses for its wall-bounded Poisson solve), so no iterative solver enters any
  reported number.

Diagnostics: x-averaged profile, momentum thickness ``theta``, vorticity
thickness ``delta_omega``, fluctuation energy and the amplitude of the
streamwise fundamental.

Linear theory for the same base profile is solved in this module
(:func:`rayleigh_modes`, :func:`orr_sommerfeld_modes`), so a case is compared
against an *independently computed* temporal eigenvalue instead of a number
transcribed from a paper: the inviscid Rayleigh solve and the viscous
Orr--Sommerfeld solve are two different discretisations/limits of the same
eigenproblem, and their agreement (plus the simulation's own linear growth
rate) is what the verification rests on.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import fft as sfft
from scipy.linalg import eig

__all__ = [
    'MixingLayerConfig',
    'TemporalMixingLayer',
    'StabilityMode',
    'rayleigh_modes',
    'orr_sommerfeld_modes',
    'most_unstable',
    'growth_rate_fit',
    'linear_fit',
    'heat_kernel_profile',
    'layer_metrics',
]


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class MixingLayerConfig:
    """Geometry and fluid properties of the temporal mixing layer."""

    nx: int = 64
    ny: int = 256
    wavelength: float = 14.0
    height: float = 16.0
    half_velocity: float = 1.0
    thickness: float = 1.0
    viscosity: float = 2e-3
    phase: float = 0.0

    def __post_init__(self):
        for name in ('nx', 'ny'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 8:
                raise ValueError(f'invalid {name}')
        for name in ('wavelength', 'height', 'half_velocity', 'thickness', 'viscosity'):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'invalid {name}')
        if not math.isfinite(self.phase):
            raise ValueError('invalid phase')
        if self.height <= 4 * self.thickness:
            raise ValueError('transverse extent too short for a free-slip mixing layer')

    @property
    def velocity_difference(self) -> float:
        return 2 * self.half_velocity

    @property
    def wave_number(self) -> float:
        return 2 * math.pi / self.wavelength

    @property
    def reynolds(self) -> float:
        """``U delta / nu`` with ``U`` the half velocity difference."""
        return self.half_velocity * self.thickness / self.viscosity


class TemporalMixingLayer:
    """Time-accurate solver for :class:`MixingLayerConfig`."""

    def __init__(self, config: MixingLayerConfig):
        self.config = config
        nx, ny = config.nx, config.ny
        self.nx, self.ny = nx, ny
        self.dx = config.wavelength / nx
        self.dy = config.height / ny
        self.xf = np.arange(nx) * self.dx
        self.xc = (np.arange(nx) + 0.5) * self.dx
        self.yc = -config.height / 2 + (np.arange(ny) + 0.5) * self.dy
        self.yf = -config.height / 2 + np.arange(ny + 1) * self.dy
        self.u = np.tile(self.base_profile(self.yc), (nx, 1))
        self.v = np.zeros((nx, ny + 1))
        self.p = np.zeros((nx, ny))
        self.time = 0.0
        self.steps = 0
        kx = np.fft.fftfreq(nx) * nx
        self._lam = (-4 * np.sin(math.pi * kx / nx)[:, None] ** 2 / self.dx ** 2
                     - 4 * np.sin(math.pi * np.arange(ny) / (2 * ny))[None, :] ** 2 / self.dy ** 2)
        self._lam_floor = float(np.abs(self._lam).max()) * 1e-13

    # ------------------------------------------------------------------ profiles
    def base_profile(self, y: np.ndarray) -> np.ndarray:
        return self.config.half_velocity * np.tanh(np.asarray(y, float) / self.config.thickness)

    # ------------------------------------------------------------------ seeding
    def seed_mode(self, amplitude: float, alpha: float, phi: np.ndarray,
                  dphi: np.ndarray) -> None:
        """Add ``psi = A phi(y) exp(i alpha x)`` with ``u = dpsi/dy``, ``v = -dpsi/dx``.

        ``phi`` is sampled at the y-faces (``v`` nodes) and ``dphi`` at the cell
        centres (``u`` nodes); both are complex and normalised on
        ``max|dphi|`` so that ``amplitude`` is the peak perturbation velocity in
        units of the half velocity difference.
        """
        if not math.isfinite(amplitude) or amplitude < 0:
            raise ValueError('invalid amplitude')
        if alpha <= 0 or not math.isfinite(alpha):
            raise ValueError('invalid alpha')
        phi = np.asarray(phi, dtype=complex)
        dphi = np.asarray(dphi, dtype=complex)
        if phi.shape != (self.ny + 1,) or dphi.shape != (self.ny,):
            raise ValueError('eigenfunction shapes must be (ny+1,) and (ny,)')
        scale = float(np.abs(dphi).max())
        if scale <= 0:
            raise ValueError('degenerate streamfunction')
        phi = phi / scale
        dphi = dphi / scale
        phase_u = np.exp(1j * alpha * (self.xf + self.config.phase))
        phase_v = np.exp(1j * alpha * (self.xc + self.config.phase))
        self.u = self.u + amplitude * np.real(dphi[None, :] * phase_u[:, None])
        self.v = self.v + amplitude * np.real(-1j * alpha * phi[None, :] * phase_v[:, None])
        self.p = np.zeros((self.nx, self.ny))

    def seed_stability_mode(self, amplitude: float, mode: 'StabilityMode') -> None:
        """Seed the discrete field with an eigenfunction from the stability solve."""
        if mode.y is None or mode.phi is None:
            raise ValueError('stability mode carries no eigenfunction')
        ygrid = mode.y
        phi = mode.phi
        dphi = np.gradient(phi, ygrid[1] - ygrid[0])
        re = np.interp
        phi_f = re(self.yf, ygrid, phi.real) + 1j * re(self.yf, ygrid, phi.imag)
        dphi_c = re(self.yc, ygrid, dphi.real) + 1j * re(self.yc, ygrid, dphi.imag)
        self.seed_mode(amplitude, mode.alpha, phi_f, dphi_c)

    def seed_velocity(self, amplitude: float, profile: str = 'sech2') -> None:
        """Divergence-free single-mode seed with an analytic streamfunction."""
        eta = self.yf / self.config.thickness
        if profile == 'sech2':
            psi = 1.0 / np.cosh(eta) ** 2
        elif profile == 'gauss':
            psi = np.exp(-eta ** 2)
        else:
            raise ValueError('unknown seed profile')
        dpsi = np.interp(self.yc, self.yf, np.gradient(psi, self.yf))
        self.seed_mode(amplitude, self.config.wave_number, psi, dpsi)

    # ------------------------------------------------------------------ operators
    def divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return (np.roll(u, -1, axis=0) - u) / self.dx + (v[:, 1:] - v[:, :-1]) / self.dy

    def solve_pressure(self, rhs: np.ndarray) -> np.ndarray:
        """Exact discrete Poisson solve for the projection (zero-mean pressure)."""
        spec = sfft.fft(rhs, axis=0)
        spec = sfft.dct(spec, type=2, norm='ortho', axis=1)
        out = np.zeros_like(spec)
        mask = np.abs(self._lam) > self._lam_floor
        out[mask] = spec[mask] / self._lam[mask]
        out = sfft.idct(out, type=2, norm='ortho', axis=1)
        return sfft.ifft(out, axis=0).real

    def project(self, u: np.ndarray, v: np.ndarray, dt: float):
        phi = self.solve_pressure(self.divergence(u, v) / dt)
        u = u - dt * (phi - np.roll(phi, 1, axis=0)) / self.dx
        inner = v[:, 1:-1] - dt * (phi[:, 1:] - phi[:, :-1]) / self.dy
        v = np.concatenate((np.zeros((self.nx, 1)), inner, np.zeros((self.nx, 1))), axis=1)
        self.p = phi
        return u, v

    def _tendencies(self, u: np.ndarray, v: np.ndarray):
        dx, dy, nu = self.dx, self.dy, self.config.viscosity
        nx, ny = self.nx, self.ny
        # free-slip walls: du/dy = 0 -> u ghost equals the interior value
        u_ext = np.concatenate((u[:, :1], u, u[:, -1:]), axis=1)
        # --- u momentum at x faces, y centres
        du_dx = (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0)) / (2 * dx)
        du_dy = (u_ext[:, 2:] - u_ext[:, :-2]) / (2 * dy)
        v_left = np.roll(v, 1, axis=0)
        v_at_uface = 0.25 * (v[:, :-1] + v[:, 1:] + v_left[:, :-1] + v_left[:, 1:])
        lap_u = ((np.roll(u, -1, axis=0) - 2 * u + np.roll(u, 1, axis=0)) / dx ** 2
                 + (u_ext[:, 2:] - 2 * u_ext[:, 1:-1] + u_ext[:, :-2]) / dy ** 2)
        tu = -(u * du_dx + v_at_uface * du_dy) + nu * lap_u
        # --- v momentum at y faces, x centres (interior faces only)
        dv_dx = ((np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) / (2 * dx))[:, 1:-1]
        dv_dy = (v[:, 2:] - v[:, :-2]) / (2 * dy)
        u_right = np.roll(u, -1, axis=0)
        u_at_vface = 0.25 * (u[:, :-1] + u_right[:, :-1] + u[:, 1:] + u_right[:, 1:])
        lap_v = ((np.roll(v, -1, axis=0) - 2 * v + np.roll(v, 1, axis=0)) / dx ** 2)[:, 1:-1] \
            + (v[:, 2:] - 2 * v[:, 1:-1] + v[:, :-2]) / dy ** 2
        tv_inner = -(u_at_vface * dv_dx + v[:, 1:-1] * dv_dy) + nu * lap_v
        tv = np.zeros((nx, ny + 1))
        tv[:, 1:-1] = tv_inner
        return tu, tv

    # ------------------------------------------------------------------ stepping
    def stable_time_step(self, cfl: float = 0.4, diffusion_safety: float = 0.25) -> float:
        speed = max(float(np.abs(self.u).max()), float(np.abs(self.v).max()), 1e-8)
        viscous = diffusion_safety * self.dy ** 2 / max(self.config.viscosity, 1e-30)
        return min(cfl * self.dx / speed, cfl * self.dy / speed, viscous)

    def step(self, dt: float) -> float:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('invalid dt')
        tu, tv = self._tendencies(self.u, self.v)
        u1 = self.u + dt * tu
        v1 = self.v + dt * tv
        v1[:, 0] = 0.0
        v1[:, -1] = 0.0
        u1, v1 = self.project(u1, v1, dt)
        tu2, tv2 = self._tendencies(u1, v1)
        u2 = self.u + 0.5 * dt * (tu + tu2)
        v2 = self.v + 0.5 * dt * (tv + tv2)
        v2[:, 0] = 0.0
        v2[:, -1] = 0.0
        self.u, self.v = self.project(u2, v2, dt)
        self.time += dt
        self.steps += 1
        return dt

    def advance(self, t_end: float, dt: float | None = None, samples: int = 0) -> list[dict]:
        """March to ``t_end``; returns ``samples`` evenly spaced diagnostics."""
        dt = self.stable_time_step() if dt is None else float(dt)
        if t_end <= 0:
            raise ValueError('invalid horizon')
        out: list[dict] = []
        interval = t_end / samples if samples else None
        next_sample = 0.0 if samples else None
        while self.time < t_end - 1e-12:
            h = min(dt, t_end - self.time)
            self.step(h)
            if interval is not None and self.time + 1e-12 >= next_sample:
                out.append(self.diagnostics())
                next_sample = next_sample + interval
        return out

    # ------------------------------------------------------------------ diagnostics
    def mean_profile(self) -> np.ndarray:
        return self.u.mean(axis=0)

    def momentum_thickness(self) -> float:
        """``theta = int (1/4 - (ub/dU)^2) dy`` for a two-stream layer."""
        ub = self.mean_profile()
        dU = self.config.velocity_difference
        return float(np.trapezoid(0.25 - (ub / dU) ** 2, self.yc))

    def vorticity_thickness(self) -> float:
        ub = self.mean_profile()
        peak = float(np.abs(np.gradient(ub, self.dy)).max())
        return self.config.velocity_difference / peak if peak > 0 else float('inf')

    def fundamental_amplitude(self) -> float:
        """``sqrt(int |v_hat(k=1)|^2 dy)`` of the transverse velocity."""
        vc = 0.5 * (self.v[:, 1:] + self.v[:, :-1])
        spec = sfft.fft(vc, axis=0)[1] / self.nx
        return float(math.sqrt(float(np.sum(np.abs(spec) ** 2)) * self.dy))

    def fluctuation_energy(self) -> float:
        ub = self.mean_profile()
        vc = 0.5 * (self.v[:, 1:] + self.v[:, :-1])
        up = self.u - ub[None, :]
        return float(0.5 * (np.sum(up ** 2) + np.sum(vc ** 2))
                     * self.dx * self.dy / self.config.wavelength)

    def diagnostics(self) -> dict:
        return {
            'time': self.time,
            'momentum_thickness': self.momentum_thickness(),
            'vorticity_thickness': self.vorticity_thickness(),
            'fundamental_amplitude': self.fundamental_amplitude(),
            'fluctuation_energy': self.fluctuation_energy(),
            'max_divergence': float(np.abs(self.divergence(self.u, self.v)).max()),
            'kinetic_energy': float(0.5 * self.dx * self.dy
                                    * (np.sum(self.u ** 2) + np.sum(self.v ** 2))),
        }

    def vorticity(self) -> np.ndarray:
        du_dy = np.gradient(self.u, self.dy, axis=1)
        vc = 0.5 * (self.v[:, 1:] + self.v[:, :-1])
        dv_dx = (np.roll(vc, -1, axis=0) - np.roll(vc, 1, axis=0)) / (2 * self.dx)
        return dv_dx - du_dy


# --------------------------------------------------------------- linear stability
@dataclass(frozen=True)
class StabilityMode:
    alpha: float
    reynolds: float | None
    c: complex
    y: np.ndarray | None = None
    phi: np.ndarray | None = None

    @property
    def growth(self) -> float:
        """Temporal growth rate in units of ``U / delta``."""
        return float(self.alpha * self.c.imag)

    @property
    def phase_speed(self) -> float:
        return float(self.c.real)


def _uniform_grid(half_width: float, nodes: int) -> np.ndarray:
    if not math.isfinite(half_width) or half_width <= 0:
        raise ValueError('invalid half width')
    if nodes < 41:
        raise ValueError('need at least 41 nodes for the stability solve')
    return np.linspace(-half_width, half_width, nodes)


def _interior_operator(y: np.ndarray, alpha: float) -> np.ndarray:
    """``D2 - alpha^2`` restricted to interior nodes (``phi = 0`` at both walls)."""
    h = y[1] - y[0]
    n = y.size - 2
    m = np.zeros((n, n))
    idx = np.arange(n)
    m[idx, idx] = -2.0 / h ** 2 - alpha ** 2
    m[idx[1:], idx[:-1]] = 1.0 / h ** 2
    m[idx[:-1], idx[1:]] = 1.0 / h ** 2
    return m


def _profile_terms(y: np.ndarray, half_velocity: float, thickness: float):
    eta = y / thickness
    u = half_velocity * np.tanh(eta)
    sech2 = 1.0 / np.cosh(eta) ** 2
    upp = -2.0 * half_velocity * sech2 * np.tanh(eta) / thickness ** 2
    return u, upp


def _solve(y: np.ndarray, a: np.ndarray, b: np.ndarray, alpha: float,
           half_velocity: float, keep: int, reynolds: float | None) -> list[StabilityMode]:
    """Solve the interior generalised eigenproblem ``[U M - U''] phi = c M phi``.

    ``M = D2 - alpha^2`` acts on interior nodes only (``phi = 0`` on both walls),
    so both pencils carry the same ``1/h^2`` scaling and no artificial boundary
    row enters the pencil.
    """
    values, vectors = eig(a, b)
    order: list[tuple[float, int]] = []
    for j, c in enumerate(values):
        if not (math.isfinite(c.real) and math.isfinite(c.imag)):
            continue
        if c.imag <= 1e-3 or not (abs(c.real) < half_velocity):
            continue
        order.append((float(c.imag), j))
    order.sort(reverse=True)
    n = y.size
    out: list[StabilityMode] = []
    for _, j in order[:keep]:
        full = np.zeros(n, dtype=complex)
        full[1:-1] = vectors[:, j]
        full /= max(float(np.abs(full).max()), 1e-300)
        out.append(StabilityMode(alpha=alpha, reynolds=reynolds,
                                 c=complex(values[j]), y=y, phi=full))
    return out


def rayleigh_modes(alpha: float, *, half_velocity: float = 1.0, thickness: float = 1.0,
                   half_width: float = 8.0, nodes: int = 601,
                   keep: int = 2) -> list[StabilityMode]:
    """Inviscid (Rayleigh) temporal modes of ``U tanh(y/delta)``.

    Solves ``(U-c)(phi'' - alpha^2 phi) - U'' phi = 0`` with
    ``phi(+-half_width) = 0`` on a uniform second-order grid.
    """
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError('invalid alpha')
    y = _uniform_grid(half_width, nodes)
    u, upp = _profile_terms(y, half_velocity, thickness)
    m = _interior_operator(y, alpha)
    a = np.diag(u[1:-1]) @ m - np.diag(upp[1:-1])
    return _solve(y, a, m, alpha, half_velocity, keep, None)


def orr_sommerfeld_modes(alpha: float, reynolds: float, *,
                         half_velocity: float = 1.0, thickness: float = 1.0,
                         half_width: float = 8.0, nodes: int = 601,
                         keep: int = 2) -> list[StabilityMode]:
    """Viscous temporal (Orr--Sommerfeld) modes of the same base profile.

    ``[U (D2-a^2) - U'' - (1/(i a Re)) (D2-a^2)^2] phi = c (D2-a^2) phi`` with
    Dirichlet walls; ``half_width >> delta`` so the eigenfunction has decayed to
    ~1e-6 at the walls and the ``Re -> infinity`` limit must reproduce
    :func:`rayleigh_modes`.
    """
    if not math.isfinite(reynolds) or reynolds <= 0:
        raise ValueError('invalid Reynolds number')
    if alpha <= 0 or not math.isfinite(alpha):
        raise ValueError('invalid alpha')
    y = _uniform_grid(half_width, nodes)
    u, upp = _profile_terms(y, half_velocity, thickness)
    m = _interior_operator(y, alpha)
    a = (np.diag(u[1:-1]) @ m - np.diag(upp[1:-1]) - (m @ m) / (1j * alpha * reynolds))
    return _solve(y, a, m, alpha, half_velocity, keep, reynolds)


def most_unstable(viscous: bool, *, reynolds: float | None = None,
                  alphas: np.ndarray | None = None, **kwargs) -> StabilityMode:
    """Mode with the largest temporal growth rate over an alpha sweep."""
    grid = np.linspace(0.1, 1.0, 19) if alphas is None else np.asarray(alphas, float)
    best: StabilityMode | None = None
    for alpha in grid:
        modes = (orr_sommerfeld_modes(float(alpha), float(reynolds), keep=1) if viscous
                 else rayleigh_modes(float(alpha), keep=1, **kwargs))
        if not modes:
            continue
        mode = modes[0]
        if best is None or mode.growth > best.growth:
            best = mode
    if best is None:
        raise RuntimeError('no unstable mode found on the alpha sweep')
    return best


# --------------------------------------------------------------------- fitting
def linear_fit(x, y) -> tuple[float, float]:
    """Least-squares ``y = a + b x`` -> ``(a, b)``."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or x.shape != y.shape:
        raise ValueError('need matching samples for a fit')
    design = np.vstack((np.ones_like(x), x)).T
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coeff[0]), float(coeff[1])


def growth_rate_fit(times, amplitudes, window: tuple[float, float] | None = None
                    ) -> tuple[float, float]:
    """Exponential growth rate ``d ln(amplitude)/dt`` over a time window.

    Returns ``(rate, rms_residual)``; the residual is on ``ln(amplitude)``.
    """
    t = np.asarray(times, float)
    a = np.asarray(amplitudes, float)
    if t.shape != a.shape:
        raise ValueError('time and amplitude arrays must match')
    mask = a > 0
    if window is not None:
        mask &= (t >= window[0]) & (t <= window[1])
    if mask.sum() < 3:
        raise ValueError('not enough positive samples in window')
    offset, rate = linear_fit(t[mask], np.log(a[mask]))
    residual = np.log(a[mask]) - (offset + rate * t[mask])
    return rate, float(math.sqrt(float(np.mean(residual ** 2))))

# ------------------------------------------------------- exact diffusion reference
def heat_kernel_profile(y, t: float, *, viscosity: float = 2e-3,
                        half_velocity: float = 1.0, thickness: float = 1.0,
                        half_width: float = 24.0, spacing: float = 0.005):
    """Exact diffused profile ``u(y,t) = int G(y-y',t) U tanh(y'/delta) dy'``.

    For ``u = u(y, t)``, ``v = 0`` and no forcing the incompressible momentum
    equation reduces to the heat equation with zero pressure, so this is an
    *exact* solution of the Navier--Stokes equations for the initial profile
    ``U tanh(y/delta)``.
    """
    if not math.isfinite(t) or t < 0:
        raise ValueError('invalid time')
    yq = np.asarray(y, float)
    if t == 0:
        return half_velocity * np.tanh(yq / thickness)
    sigma = math.sqrt(2 * viscosity * t)
    yp = np.arange(-half_width, half_width + spacing, spacing)
    f = half_velocity * np.tanh(yp / thickness)
    diff = (yq[:, None] - yp[None, :]) / sigma
    kernel = np.exp(-0.5 * diff ** 2) / (math.sqrt(2 * math.pi) * sigma)
    return spacing * (kernel @ f)


def layer_metrics(profile, y, dy: float, velocity_difference: float = 2.0) -> dict:
    """Momentum and vorticity thickness of a two-stream profile."""
    profile = np.asarray(profile, float)
    y = np.asarray(y, float)
    if profile.shape != y.shape:
        raise ValueError('profile and grid must match')
    theta = float(np.trapezoid(0.25 - (profile / velocity_difference) ** 2, y))
    peak = float(np.abs(np.gradient(profile, dy)).max())
    return {'momentum_thickness': theta,
            'vorticity_thickness': velocity_difference / peak if peak > 0 else float('inf')}
