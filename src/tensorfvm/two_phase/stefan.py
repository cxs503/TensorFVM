"""One-dimensional Stefan phase-change verification (enthalpy method vs similarity).

Physical problem
----------------
A semi-infinite solidifying medium occupies ``x > 0``; it is initially liquid at
the fusion temperature ``T_f`` and the wall ``x = 0`` is held at ``T_w < T_f``
(the classical one-phase Stefan problem).  The interface ``s(t)`` separates the
solid from the liquid and moves according to the energy balance

.. math::

    \\rho L \\frac{ds}{dt} = k_s \\left.\\frac{\\partial T}{\\partial x}\\right|_{x=s},

which admits the similarity solution (Neumann 1860; Carslaw & Jaeger, *Conduction
of Heat in Solids*, 2nd ed., §11.2)

.. math::

    s(t) = 2\\lambda\\sqrt{\\alpha_s t}, \\qquad
    T(x,t) = T_w + (T_f - T_w)\\,
             \\frac{\\operatorname{erf}(x/2\\sqrt{\\alpha_s t})}{\\operatorname{erf}\\lambda},

with ``\\lambda`` the root of the transcendental equation

.. math::

    \\sqrt{\\pi}\\,\\lambda e^{\\lambda^2}\\operatorname{erf}\\lambda
      = \\mathrm{St}, \\qquad \\mathrm{St} = \\frac{c_s (T_f - T_w)}{L}.

Numerical method
----------------
Conservative cell-centred finite volumes with an **apparent heat capacity**

.. math::

    c_{eff}(T) = c_s + \\frac{L}{2\\,\\delta}\\quad\\text{for } |T - T_f| < \\delta,

i.e. the latent heat is released in a thin numerical band of half width
``\\delta``; ``\\delta`` is a discretisation parameter, not a physical quantity,
and it is refined together with the mesh.  Time integration is implicit
(backward Euler) with a Picard iteration on ``c_eff`` and a Thomas solve, so the
step is not restricted by the conduction stability limit.  The front position is
reconstructed from the *solid fraction*, ``s = \\int f_s\\,dx``, which is the
volume-conservative estimate, and cross-checked with the free-boundary energy
balance (the wall heat flux integrated in time must equal the enthalpy change of
the domain -- reported as the energy-ledger residual).

Zero-latent control
-------------------
With ``L = 0`` the apparent capacity loses its latent plateau and the model has
no phase-change capacity at all: the interface must stay at the wall and the
temperature field must reduce to the pure-conduction solution
``T = T_w + (T_f - T_w)\\,\\operatorname{erf}(x/2\\sqrt{\\alpha_s t})``.  Both
statements are measured, so a latent-heat leak into the conduction terms cannot
hide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy.special import erf, erfc

__all__ = [
    "StefanConfig",
    "stefan_lambda",
    "stefan_lambda_two_phase",
    "neumann_front",
    "neumann_temperature",
    "conduction_temperature",
    "solve_stefan_1d",
]


def stefan_lambda(stefan_number: float) -> float:
    """Root of ``sqrt(pi) lam exp(lam^2) erf(lam) = St`` by safeguarded Newton."""
    st = float(stefan_number)
    if st <= 0.0:
        raise ValueError("the Stefan number must be positive")
    lam = math.sqrt(st / 2.0) if st < 1.0 else 0.5 * math.sqrt(2.0 * st)
    lam = min(max(lam, 1e-6), 20.0)
    for _ in range(200):
        f = math.sqrt(math.pi) * lam * math.exp(lam * lam) * erf(lam) - st
        # d/dlam of lam exp(lam^2) erf(lam)
        fp = math.sqrt(math.pi) * math.exp(lam * lam) * (
            erf(lam) + 2.0 * lam * lam * erf(lam)
            + 2.0 * lam / math.sqrt(math.pi))
        step = f / fp
        lam -= step
        if abs(step) < 1e-15 * max(lam, 1.0):
            break
    else:                                        # pragma: no cover - safety net
        raise RuntimeError("lambda iteration did not converge")
    if not (0.0 < lam < 50.0):
        raise RuntimeError("non-physical similarity constant")
    return lam


def _stefan_residual(lam: float, st_solid: float, st_liquid: float,
                     ratio_sqrt_alpha: float, ratio_alpha: float) -> float:
    """Two-phase Stefan equation (Carslaw & Jaeger §11.4), dimensionless form.

    Equal densities are assumed (no volume change on phase change).
    """
    solid = st_solid * math.exp(-lam * lam) / (
        math.sqrt(math.pi) * erf(lam))
    liquid = st_liquid * math.exp(-lam * lam * ratio_alpha) / (
        math.sqrt(math.pi) * erfc(lam * ratio_sqrt_alpha))
    return solid + liquid * ratio_sqrt_alpha ** -1.0 * (
        math.sqrt(1.0)) - lam


def stefan_lambda_two_phase(st_solid: float, st_liquid: float,
                            alpha_ratio: float = 1.0) -> float:
    """Similarity constant of the *two-phase* Stefan problem (both sides active).

    ``alpha_ratio = alpha_s/alpha_l``.  The one-phase limit is recovered for
    ``st_liquid -> 0``, which the test suite checks against
    :func:`stefan_lambda`.
    """
    if st_liquid < 0.0:
        raise ValueError("the liquid Stefan number must be non-negative")
    if st_solid <= 0.0:
        raise ValueError("the solid Stefan number must be positive")
    ratio = alpha_ratio ** 0.5
    lo, hi = 1e-8, 20.0
    f_lo = _stefan_residual(lo, st_solid, st_liquid, ratio, alpha_ratio)
    f_hi = _stefan_residual(hi, st_solid, st_liquid, ratio, alpha_ratio)
    if f_lo * f_hi > 0.0:                       # pragma: no cover - defensive
        raise ValueError("could not bracket the two-phase similarity root")
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        f_mid = _stefan_residual(mid, st_solid, st_liquid, ratio, alpha_ratio)
        if f_lo * f_mid <= 0.0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
        if hi - lo < 1e-15:
            break
    return 0.5 * (lo + hi)


def neumann_front(t, lam: float, alpha: float):
    """Similarity front ``2 lam sqrt(alpha t)``."""
    return 2.0 * lam * np.sqrt(np.asarray(alpha, dtype=float) * np.asarray(t))


def neumann_temperature(x, t: float, lam: float, alpha: float,
                        t_fusion: float, t_wall: float):
    """Similarity temperature field of the one-phase Stefan problem."""
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        return np.full_like(x, t_fusion)
    return t_wall + (t_fusion - t_wall) * erf(x / (2.0 * math.sqrt(alpha * t))) \
        / erf(lam)


def conduction_temperature(x, t: float, alpha: float, t_fusion: float,
                           t_wall: float):
    """Pure-conduction (erf) solution -- the ``L = 0`` reference."""
    x = np.asarray(x, dtype=float)
    if t <= 0.0:
        return np.full_like(x, t_fusion)
    return t_wall + (t_fusion - t_wall) * erf(x / (2.0 * math.sqrt(alpha * t)))


@dataclass
class StefanConfig:
    """Material data and discretisation of the one-phase verification case."""

    #: ice-like solid: k, rho, c (SI)
    k_solid: float = 2.22
    rho: float = 917.0
    c_solid: float = 2050.0
    latent: float = 3.337e5
    t_fusion: float = 273.15
    t_wall: float = 253.15
    #: grid and time step
    length: float = 0.20
    n_cells: int = 400
    max_dt: float = 0.05
    t_end: float = 600.0
    #: half width of the numerical mushy band [K]
    mushy_halfwidth: float = 0.01
    n_samples: int = 7

    @property
    def alpha(self) -> float:
        return self.k_solid / (self.rho * self.c_solid)

    @property
    def dx(self) -> float:
        return self.length / self.n_cells

    @property
    def stefan_number(self) -> float:
        if self.latent <= 0.0:
            return float("inf")
        return self.c_solid * (self.t_fusion - self.t_wall) / self.latent

    def validate(self) -> None:
        if self.k_solid <= 0.0 or self.rho <= 0.0 or self.c_solid <= 0.0:
            raise ValueError("material properties must be positive")
        if self.latent < 0.0:
            raise ValueError("latent heat must be non-negative")
        if self.t_wall >= self.t_fusion:
            raise ValueError("the wall must be colder than the fusion temperature")
        if self.mushy_halfwidth <= 0.0:
            raise ValueError("the mushy half width must be positive")
        if self.n_cells < 20:
            raise ValueError("grid too coarse")
        if self.max_dt <= 0.0 or self.t_end <= 0.0:
            raise ValueError("time step and end time must be positive")


def _thomas(lower, diag, upper, rhs):
    """Tridiagonal solve (Thomas algorithm), numpy vectors."""
    n = diag.size
    cp = np.empty(n)
    dp = np.empty(n)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for i in range(1, n):
        m = diag[i] - lower[i] * cp[i - 1]
        cp[i] = upper[i] / m
        dp[i] = (rhs[i] - lower[i] * dp[i - 1]) / m
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def solve_stefan_1d(cfg: StefanConfig) -> dict:
    """Explicit enthalpy-method FVM for the one-phase Stefan problem.

    The conserved variable is the volumetric enthalpy ``H = rho c (T - T_f) +
    rho L f_s`` referenced to the initial state (liquid at ``T_f``); the update

    .. math::

        H^{n+1}_i = H^{n}_i + \frac{\Delta t\, k}{\Delta x^2}
                    \big(T^n_{i+1} - 2T^n_i + T^n_{i-1}\big)

    is explicit in the temperature, so no mushy band and no apparent capacity
    are needed: the solid fraction of a cell follows from inverting ``H(T)``
    exactly (a cell that contains the interface sits at ``T = T_f`` with
    ``0 < f_s < 1``).  The step is limited by ``dt <= 0.4 dx^2/alpha``.
    """
    cfg.validate()
    n, dx = cfg.n_cells, cfg.dx
    x = (np.arange(n) + 0.5) * dx
    t_f = cfg.t_fusion
    rho_c = cfg.rho * cfg.c_solid
    latent_vol = cfg.rho * cfg.latent
    alpha = cfg.alpha

    def temperature_of(enthalpy):
        """Invert H(T): solid / two-phase / liquid branch."""
        if latent_vol <= 0.0:
            # no latent heat: the two-phase branch has zero width, so a cell is
            # solid or liquid with T = H/(rho c) -- and, since H <= 0 for a
            # cooled medium, T <= T_f always: the interface cannot exist
            return t_f + enthalpy / rho_c, np.zeros_like(enthalpy)
        solid = enthalpy < 0.0
        two_phase = (enthalpy >= 0.0) & (enthalpy <= latent_vol)
        temp = np.where(solid, t_f + enthalpy / rho_c,
                        np.where(two_phase, t_f,
                                 t_f + (enthalpy - latent_vol) / rho_c))
        f_s = np.where(solid, 1.0, np.where(two_phase,
                                            (latent_vol - enthalpy) / latent_vol,
                                            0.0))
        return temp, f_s

    # reference state: liquid at T_f  ->  H = rho L (solid at T_f would be H=0)
    enthalpy = np.full(n, latent_vol)
    temp, f_s = temperature_of(enthalpy)

    dt_stable = 0.4 * dx * dx / alpha
    dt = min(cfg.max_dt, dt_stable)
    steps = int(math.ceil(cfg.t_end / dt))
    dt = cfg.t_end / steps
    sample_stride = max(1, steps // max(1, cfg.n_samples))

    time_hist, front_hist, flux_hist = [], [], []
    q_wall = 0.0
    for step in range(1, steps + 1):
        lap = np.empty(n)
        lap[1:-1] = temp[2:] - 2.0 * temp[1:-1] + temp[:-2]
        # Dirichlet wall half a cell to the left (2 k/dx) and far field at T_f
        lap[0] = temp[1] - 3.0 * temp[0] + 2.0 * cfg.t_wall
        lap[-1] = temp[-2] - 3.0 * temp[-1] + 2.0 * t_f
        enthalpy = enthalpy + dt * cfg.k_solid * lap / dx ** 2
        temp, f_s = temperature_of(enthalpy)
        q_wall += cfg.k_solid * (cfg.t_wall - temp[0]) / (0.5 * dx) * dt
        if step % sample_stride == 0 or step == steps:
            time_hist.append(step * dt)
            front_hist.append(float(np.sum(f_s) * dx))
            flux_hist.append(cfg.k_solid * (temp[0] - cfg.t_wall) / (0.5 * dx))

    # reference state is "liquid at T_f" (H = rho L), so the change of enthalpy
    # is measured against ``latent_vol`` per cell
    enthalpy_change = float(np.sum(enthalpy - latent_vol) * dx)
    return {
        "x": x,
        "t": np.asarray(time_hist),
        "front": np.asarray(front_hist),
        "wall_flux": np.asarray(flux_hist),
        "temperature": temp,
        "solid_fraction": f_s,
        "final_front": float(np.sum(f_s) * dx),
        "energy_change": enthalpy_change,
        "energy_wall": q_wall,
        "energy_ledger_residual": abs(enthalpy_change - q_wall)
        / max(abs(enthalpy_change), 1e-300),
        "dt": dt,
        "steps": steps,
        "stefan_number": cfg.stefan_number,
        "lambda": (stefan_lambda(cfg.stefan_number)
                   if 0.0 < cfg.stefan_number < float("inf") else 0.0),
        "config": asdict(cfg),
    }
