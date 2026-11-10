"""Rayleigh-Plesset bubble dynamics and the cavitation void-transport equation.

Two coupled ingredients of the cavitation model:

**Bubble dynamics (Rayleigh 1917, Plesset 1949).**  The radius of a spherical
vapour bubble in an infinite liquid obeys

.. math::

    R\\ddot R + \\tfrac{3}{2}\\dot R^2
      = \\frac{p_v + p_g(R) - p_\\infty(t)}{\\rho}
        - \\frac{4\\mu\\dot R}{\\rho R} - \\frac{2\\sigma}{\\rho R},

with a polytropic gas content ``p_g(R) = p_g0 (R_0/R)^{3\\gamma}``.  The
equilibrium state used here is ``p_g0 = p_inf0 - p_v + 2 sigma / R_0``, so a
bubble released at rest with ``p_inf = p_inf0`` is in exact equilibrium.

For a small perturbation ``R = R_0 (1 + \\epsilon e^{i\\omega t})`` the equation
linearises to a damped oscillator

.. math::

    \\ddot X + 2\\beta \\dot X + \\omega^2 X = 0,
    \\qquad \\omega^2 = \\frac{3\\gamma p_{g0} - 2\\sigma/R_0}{\\rho R_0^2},
    \\qquad \\beta = \\frac{2\\mu}{\\rho R_0^2},

whose solution ``X(t) = \\epsilon R_0 e^{-\\beta t}\\cos(\\omega_d t)`` with
``\\omega_d = \\sqrt{\\omega^2-\\beta^2}`` is the Plesset (1949) linear reference.
For ``2\\sigma/R_0 \\ll 3\\gamma p_{g0}`` this reduces to the familiar
``\\omega = \\sqrt{3\\gamma p_0/(\\rho R_0^2)}``.  The nonlinear ODE is integrated
with ``scipy`` DOP853 at ``rtol = 1e-12``, i.e. the reference used for the
verification is the *analytic* linear solution, not another discretisation.

**Void transport.**  The gas mass carried by a monodisperse bubble population of
number density ``n_b`` obeys

.. math::

    \\frac{\\partial(\\alpha\\rho_v)}{\\partial t}
      + \\nabla\\cdot(\\alpha\\rho_v \\mathbf{u}) = \\dot m,
    \\qquad \\alpha = \\tfrac{4}{3}\\pi R^3 n_b,
    \\qquad \\dot m = 4\\pi R^2 \\rho_v n_b \\dot R,

i.e. the void fraction is advected conservatively while the bubble radius grows
by the Rayleigh-Plesset dynamics of the local pressure.  The 1-D solver below
advects ``n_b`` with a conservative MUSCL/Hancock scheme and advances ``R`` in
every cell with the bubble ODE; the gas-mass ledger
``d/dt ∫(alpha rho_v) = ∫ m_dot`` is reported as an identity check, which is
what makes the void-transport implementation verifiable independent of any
particular cavitation model.

References
----------
M.S. Plesset, *J. Appl. Mech.* 16 (1949) 277-282.
Lord Rayleigh, *Phil. Mag.* 34 (1917) 94-98.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

import numpy as np
from scipy.integrate import solve_ivp

__all__ = [
    "BubbleConfig",
    "linear_frequency",
    "plesset_solution",
    "integrate_bubble",
    "free_oscillation",
    "CavitationTransportConfig",
    "solve_void_transport",
    "rayleigh_growth",
    "linear_damping",
    "linear_room_frequency",
    "measured_frequency",
]


@dataclass
class BubbleConfig:
    """Single-bubble parameters (SI units)."""

    r0: float = 1.0e-4            # equilibrium radius [m]
    p_inf0: float = 1.0e5         # ambient pressure at equilibrium [Pa]
    p_vapour: float = 2.339e3     # vapour pressure (water, 20 C) [Pa]
    rho: float = 998.2            # liquid density [kg/m^3]
    mu: float = 1.002e-3          # dynamic viscosity [Pa s]
    sigma: float = 0.0728         # surface tension [N/m]
    gamma: float = 1.4            # polytropic exponent of the gas
    p_gas_override: float | None = None   # bypass the equilibrium gas content

    @property
    def p_gas0(self) -> float:
        """Gas pressure at ``R = R0`` (equilibrium value unless overridden)."""
        if self.p_gas_override is not None:
            return float(self.p_gas_override)
        return self.p_inf0 - self.p_vapour + 2.0 * self.sigma / self.r0

    def validate(self) -> None:
        if self.r0 <= 0.0 or self.rho <= 0.0:
            raise ValueError("r0 and rho must be positive")
        if self.p_inf0 <= self.p_vapour:
            raise ValueError("the equilibrium state needs p_inf0 > p_vapour")
        if self.gamma <= 0.0:
            raise ValueError("gamma must be positive")
        if self.p_gas0 <= 0.0:
            raise ValueError("non-physical equilibrium gas pressure")


def linear_frequency(cfg: BubbleConfig) -> float:
    """Linear (Plesset) natural frequency including surface tension [rad/s]."""
    cfg.validate()
    num = 3.0 * cfg.gamma * cfg.p_gas0 - 2.0 * cfg.sigma / cfg.r0
    if num <= 0.0:
        raise ValueError("surface tension dominates: the equilibrium is unstable")
    return math.sqrt(num / (cfg.rho * cfg.r0 ** 2))


def linear_room_frequency(cfg: BubbleConfig) -> float:
    """Task-form frequency ``sqrt(3 gamma p_0 / (rho R_0^2))`` with ``p_0 = p_gas0``.

    This is the surface-tension-free limit of :func:`linear_frequency`; the two
    differ by ``2 sigma/(3 gamma p_g0 R0)`` (reported in the benchmark).
    """
    cfg.validate()
    return math.sqrt(3.0 * cfg.gamma * cfg.p_gas0 / (cfg.rho * cfg.r0 ** 2))


def linear_damping(cfg: BubbleConfig) -> float:
    """Viscous damping rate ``beta = 2 mu/(rho R0^2)`` [1/s]."""
    return 2.0 * cfg.mu / (cfg.rho * cfg.r0 ** 2)


def plesset_solution(cfg: BubbleConfig, t: np.ndarray, eps: float) -> np.ndarray:
    """Plesset (1949) linear solution ``R0 (1 + eps e^{-beta t} cos(w_d t))``."""
    omega = linear_frequency(cfg)
    beta = linear_damping(cfg)
    wd = math.sqrt(max(omega * omega - beta * beta, 0.0))
    return cfg.r0 * (1.0 + eps * np.exp(-beta * t) * np.cos(wd * t))


def _rhs(cfg: BubbleConfig, p_inf: float):
    def rhs(_t, y):
        r, rdot = y
        r = max(r, 1e-12)
        p_gas = cfg.p_gas0 * (cfg.r0 / r) ** (3.0 * cfg.gamma)
        accel = ((cfg.p_vapour + p_gas - p_inf) / cfg.rho
                 - 4.0 * cfg.mu * rdot / (cfg.rho * r)
                 - 2.0 * cfg.sigma / (cfg.rho * r))
        return [rdot, accel / r - 1.5 * rdot * rdot / r]
    return rhs


def integrate_bubble(cfg: BubbleConfig, r_start: float, t_end: float,
                     p_inf: float | float | None = None,
                     rdot0: float = 0.0, rtol: float = 1e-12,
                     atol: float = 1e-18, n_samples: int = 4001,
                     with_velocity: bool = False):
    """Integrate the Rayleigh-Plesset ODE with DOP853 to ``t_end``."""
    cfg.validate()
    p_inf = cfg.p_inf0 if p_inf is None else p_inf
    t_eval = np.linspace(0.0, t_end, n_samples)
    sol = solve_ivp(_rhs(cfg, p_inf), (0.0, t_end), [r_start, rdot0],
                    method="DOP853", rtol=rtol, atol=atol, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(f"Rayleigh-Plesset integration failed: {sol.message}")
    if with_velocity:
        return sol.t, sol.y[0], sol.y[1]
    return sol.t, sol.y[0]


def free_oscillation(cfg: BubbleConfig, eps: float = 1.0e-3,
                     n_periods: float = 6.0, n_samples: int = 8001):
    """Small-amplitude free oscillation ``R(0) = R0(1+eps)``, ``Rdot(0) = 0``."""
    if abs(eps) >= 0.1:
        raise ValueError("the small-amplitude expansion requires |eps| < 0.1")
    omega = linear_frequency(cfg)
    t_end = n_periods * 2.0 * math.pi / omega
    return integrate_bubble(cfg, cfg.r0 * (1.0 + eps), t_end,
                            n_samples=n_samples)


def measured_frequency(t: np.ndarray, r: np.ndarray) -> float:
    """Frequency from the zero crossings of ``R - mean(R)`` (rad/s)."""
    centered = r - np.mean(r)
    idx = np.where(np.diff(np.sign(centered)) != 0)[0]
    if idx.size < 3:
        raise ValueError("not enough zero crossings to measure a frequency")
    crossings = t[idx] - centered[idx] * (t[idx + 1] - t[idx]) / (
        centered[idx + 1] - centered[idx])
    half_period = float(np.mean(np.diff(crossings)))
    return math.pi / half_period


def rayleigh_growth(cfg: BubbleConfig, t: np.ndarray,
                    p_inf: float) -> np.ndarray:
    """Inertia-controlled (Rayleigh) growth ``R = sqrt(2/3 |dp|/rho) t``.

    Valid for ``R >> R_0`` and negligible surface tension/viscosity and gas
    content; used as the late-time asymptotic anchor of the dynamic growth case.
    The drive is ``|p_inf - p_v|`` because a *vapour* bubble grows when the
    ambient pressure falls below the vapour pressure (and collapses in the
    opposite case, which this law does not describe).
    """
    drive = abs(p_inf - cfg.p_vapour)
    return np.sqrt(2.0 * drive / (3.0 * cfg.rho)) * t


# --------------------------------------------------------------------------- #
# void transport
# --------------------------------------------------------------------------- #
@dataclass
class CavitationTransportConfig:
    """1-D void transport in a duct with a prescribed pressure field."""

    length: float = 0.4            # duct length [m]
    n_cells: int = 200
    velocity: float = 2.0          # uniform mixture velocity [m/s]
    t_end: float = 0.05            # integration time [s]
    cfl: float = 0.4
    n_bubbles: float = 1.0e9       # bubble number density in the seeded zone
    pulse_centre: float = 0.05     # pulse centre [m]
    pulse_width: float = 0.02      # pulse half width [m]
    p_base: float = 1.0e5          # duct pressure at z = 0 [Pa]
    p_drop: float = -0.95e5        # pressure change across the duct [Pa]
    rho_vapour: float = 0.0173     # vapour density (saturated, 20 C) [kg/m^3]
    source: str = "rayleigh"       # "rayleigh" (inertia-controlled) or "none"
    bubble: BubbleConfig = field(default_factory=BubbleConfig)

    @property
    def dz(self) -> float:
        return self.length / self.n_cells

    def pressure(self, z: np.ndarray) -> np.ndarray:
        """Linear pressure distribution along the duct."""
        return self.p_base + self.p_drop * np.asarray(z) / self.length

    def validate(self) -> None:
        if self.n_cells < 8 or self.length <= 0.0:
            raise ValueError("invalid grid")
        if self.source not in ("rayleigh", "none"):
            raise ValueError(f"unknown source model {self.source!r}")
        if self.t_end <= 0.0 or self.velocity <= 0.0:
            raise ValueError("t_end and velocity must be positive")


def _minmod(a, b):
    return np.where(a * b > 0.0,
                    np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def _muscl_hancock_step(nb: np.ndarray, u: float, dt: float, dz: float
                        ) -> np.ndarray:
    """One conservative MUSCL-Hancock step of a scalar with uniform velocity.

    Slope-limited (minmod) face values, predictor half step, flux form: the
    update telescopes, so ``sum(nb)`` is conserved up to round-off for any
    boundary treatment that uses the same face value on both sides.
    """
    slope = _minmod(nb - np.concatenate((nb[:1], nb[:-1])),
                    np.concatenate((nb[1:], nb[-1:])) - nb)
    half = 0.5 * dt / dz * u
    face_high = np.where(u >= 0.0,
                         nb + 0.5 * slope * (1.0 - 2.0 * half),
                         nb - 0.5 * slope * (1.0 + 2.0 * half))
    face = np.empty(nb.size + 1)
    face[1:-1] = np.where(u >= 0.0, face_high[:-1], face_high[1:])
    # open (zero-gradient) ends: the boundary face carries the boundary value
    face[0] = nb[0]
    face[-1] = nb[-1]
    out = nb - dt / dz * (face[1:] - face[:-1])
    return np.maximum(out, 0.0)


def solve_void_transport(cfg: CavitationTransportConfig,
                         max_steps: int = 200_000) -> dict:
    """Conservative transport of ``n_b`` coupled to inertia-controlled growth.

    ``n_b`` is advected in flux form (MUSCL-Hancock, minmod limiter) and the
    bubble radius follows the Rayleigh (1917) inertia-controlled growth law

    .. math:: \dot R = \sqrt{\tfrac{2}{3}\,\max(p_\infty-p_v,0)/\rho},

    which is the late-time asymptote of the Rayleigh-Plesset equation (the
    transient oscillation of period ``2 pi/omega`` -- ~30 us here -- is *not*
    resolved by the transport step and is deliberately left to the single-bubble
    verification).  The gas mass ``alpha rho_v = 4/3 pi R^3 n_b rho_v`` and the
    integrated source ``\int \dot m`` are recorded every step so the ledger can
    be audited.
    """
    cfg.validate()
    n, dz = cfg.n_cells, cfg.dz
    z = (np.arange(n) + 0.5) * dz
    u = cfg.velocity
    bubble = cfg.bubble

    nb = np.where(np.abs(z - cfg.pulse_centre) <= cfg.pulse_width,
                  cfg.n_bubbles, 0.0)
    radius = np.full(n, bubble.r0)
    p = cfg.pressure(z)
    # inertia-controlled growth including the (polytropic) gas content: the
    # radius relaxes towards the local equilibrium radius instead of growing
    # without bound
    def growth_rate(r, p_local):
        p_gas = bubble.p_gas0 * (bubble.r0 / np.maximum(r, 1e-12)) ** (
            3.0 * bubble.gamma)
        drive = np.maximum(bubble.p_vapour + p_gas - p_local, 0.0)
        return np.sqrt(2.0 * drive / (3.0 * bubble.rho))

    seeded = nb > 0.0
    rate = np.where(seeded, growth_rate(np.maximum(radius, 0.05 * bubble.r0), p),
                    0.0) if cfg.source == "rayleigh" else np.zeros(n)

    dt = cfg.cfl * dz / u
    steps = int(math.ceil(cfg.t_end / dt))
    dt = cfg.t_end / max(steps, 1)

    nb0_total = float(nb.sum() * dz)
    # conserved gas mass per unit volume: alpha rho_v with alpha = 4/3 pi R^3 n_b
    mass = 4.0 / 3.0 * math.pi * radius ** 3 * nb * cfg.rho_vapour
    mass_initial = float(mass.sum() * dz)
    source_total = 0.0
    ledger = []
    growth_max = float(np.max(rate)) * dt
    n_sub = max(1, int(math.ceil(growth_max / (0.02 * bubble.r0))))
    dt_sub = dt / n_sub
    for _ in range(min(steps, max_steps)):
        nb = _muscl_hancock_step(nb, u, dt, dz)
        mass = _muscl_hancock_step(mass, u, dt, dz)
        r_old = radius.copy()
        if cfg.source == "rayleigh":
            for _sub in range(n_sub):
                safe = np.maximum(radius, 0.05 * bubble.r0)
                radius = np.where(seeded,
                                  radius + growth_rate(safe, p) * dt_sub,
                                  radius)
        # the gas mass created by the radius growth, exact for the cube
        dm = (4.0 * math.pi / 3.0) * (radius ** 3 - r_old ** 3) * nb \
            * cfg.rho_vapour
        mass = mass + dm
        # radius implied by the conserved mass (closes the n_b/mass coupling)
        radius = np.where(seeded,
                          np.cbrt(3.0 * mass / np.maximum(
                              4.0 * math.pi * cfg.rho_vapour * nb, 1e-300)),
                          radius)
        source_total += float(dm.sum())
        ledger.append((float(mass.sum() * dz), source_total * dz))

    alpha = 4.0 / 3.0 * math.pi * radius ** 3 * nb
    return {
        "z": z,
        "n_bubbles": nb,
        "radius": radius,
        "void_fraction": alpha,
        "pressure": p,
        "growth_rate": rate,
        "steps": min(steps, max_steps),
        "dt": dt,
        "nb_total": float(nb.sum() * dz),
        "nb_total_initial": nb0_total,
        "gas_mass": float(mass.sum() * dz),
        "gas_mass_initial": mass_initial,
        "source_total": source_total * dz,
        "ledger": ledger,
        "config": asdict(cfg),
    }
