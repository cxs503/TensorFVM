"""Stefan phase change (ice/water) by the enthalpy-porosity formulation.

The module has two independent halves:

``NeumannSolution``
    the exact similarity solution of the classical one-phase Stefan problem
    (liquid initially at the melting temperature, surface held at ``T_s < T_m``)::

        X(t) = 2 lambda sqrt(alpha t),     alpha = k / (rho c)
        lambda exp(lambda^2) erf(lambda) = St / sqrt(pi),   St = c (T_m - T_s) / L
        T(x, t) = T_s + (T_m - T_s) erf(x / (2 sqrt(alpha t))) / erf(lambda)

``EnthalpySlab1D``
    a conservative cell-centred finite-volume solver of::

        d h / d t = d/dx ( k dT/dx ),   h = c (T - T_sol) + L f_l(T)

    with the *mushy* liquid-fraction relation ``f_l = (T - T_sol)/dT_mush`` between
    ``T_sol = T_m - dT_mush/2`` and ``T_liq = T_m + dT_mush/2`` (the standard
    enthalpy-porosity closure; the Darcy/porosity momentum sink of the original
    Voller formulation is inactive here because the ice layer carries no flow --
    only the enthalpy/temperature relation is used).

The time integration is a Picard-iterated *implicit* apparent-heat-capacity
scheme: the capacity ``c_eff = dh/dT`` is ``c + L/dT_mush`` inside the mushy band
and ``c`` outside it.  An explicit scheme is unusable for hour-long ice accretion
(a Stefan time step limit of ``dx^2/(2 alpha_eff)`` with ``alpha_eff`` ~ 3.6e-9
m^2/s inside the band means ~1e8 steps), while the implicit scheme stays stable
at ``dt`` = seconds.  After the Picard loop the enthalpy is updated from the
*conserved face fluxes* (not from the temperature), so the discrete energy
balance closes to solver tolerance; that residual is reported in the ledger.

The front position is taken from the solid fraction integral

    X(t) = integral (1 - f_l) dx

which is the enthalpy-equivalent (mass) front and does not depend on where the
``f_l = 0.5`` level happens to fall between two cell centres.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

__all__ = [
    "EnthalpySlab1D",
    "NeumannSolution",
    "PhaseChangeMaterial",
    "RadialBoundary",
    "stefan_lambda",
]


def stefan_lambda(stefan_number: float) -> float:
    """Solve ``lambda exp(lambda^2) erf(lambda) = St/sqrt(pi)`` for ``lambda > 0``."""
    st = float(stefan_number)
    if st <= 0.0:
        raise ValueError("the Stefan number must be positive")
    target = st / math.sqrt(math.pi)

    def f(lam: float) -> float:
        return lam * math.exp(lam * lam) * math.erf(lam) - target

    lo, hi = 1.0e-9, 1.0
    while f(hi) < 0.0:
        hi *= 2.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class PhaseChangeMaterial:
    """Thermophysical data for a pure substance with a finite mushy band."""

    k: float = 2.2                 # W/(m K)   (ice)
    rho: float = 917.0             # kg/m^3
    c: float = 2100.0              # J/(kg K)
    latent: float = 334.0e3        # J/kg
    t_melt: float = 273.15         # K
    dT_mush: float = 0.5           # K

    @property
    def t_solidus(self) -> float:
        return self.t_melt - 0.5 * self.dT_mush

    @property
    def t_liquidus(self) -> float:
        return self.t_melt + 0.5 * self.dT_mush

    @property
    def alpha(self) -> float:
        """Thermal diffusivity of the single-phase branches [m^2/s]."""
        return self.k / (self.rho * self.c)

    @property
    def c_mushy(self) -> float:
        return self.c + self.latent / self.dT_mush

    def enthalpy(self, temperature):
        """``h(T)`` with ``h = 0`` at the solidus."""
        t = np.asarray(temperature, dtype=float)
        ts, tl = self.t_solidus, self.t_liquidus
        mushy = self.c * (t - ts) + self.latent * (t - ts) / self.dT_mush
        return np.where(t <= ts, self.c * (t - ts),
                        np.where(t < tl, mushy,
                                 self.c * self.dT_mush + self.latent
                                 + self.c * (t - tl)))

    def temperature(self, enthalpy):
        """Invert ``h -> (T, f_l)``."""
        h = np.asarray(enthalpy, dtype=float)
        ts, tl = self.t_solidus, self.t_liquidus
        h_top = self.c * self.dT_mush + self.latent
        t = np.where(h <= 0.0, ts + h / self.c,
                     np.where(h < h_top, ts + h / self.c_mushy,
                              tl + (h - h_top) / self.c))
        f_l = np.where(h <= 0.0, 0.0,
                       np.where(h < h_top, (t - ts) / self.dT_mush, 1.0))
        return t, f_l

    def capacity(self, temperature):
        """``c_eff = dh/dT`` (the apparent heat capacity)."""
        t = np.asarray(temperature, dtype=float)
        inside = (t > self.t_solidus) & (t < self.t_liquidus)
        return np.where(inside, self.c_mushy, self.c)


@dataclass(frozen=True)
class RadialBoundary:
    """Boundary flux ``q = q0 + (t_ext - T) / resistance`` (positive into the domain).

    ``resistance = 0.5 dx / k`` reproduces a Dirichlet wall value at the face,
    ``resistance = 1/h`` a convective (Robin) condition and ``resistance = inf``
    (with ``q0``) a prescribed heat flux.
    """

    t_ext: float = 0.0
    resistance: float = math.inf
    q0: float = 0.0

    def flux(self, t_face):
        if math.isinf(self.resistance):
            return np.full_like(np.asarray(t_face, dtype=float), self.q0)
        return self.q0 + (self.t_ext - np.asarray(t_face, dtype=float)) / self.resistance

    def conductance(self) -> float:
        return 0.0 if math.isinf(self.resistance) else 1.0 / self.resistance


@dataclass
class NeumannSolution:
    """Exact one-phase Stefan similarity solution (solid growing into liquid at ``T_m``)."""

    material: PhaseChangeMaterial
    t_surface: float

    @property
    def stefan_number(self) -> float:
        return self.material.c * (self.material.t_melt - self.t_surface) / self.material.latent

    @property
    def lam(self) -> float:
        return stefan_lambda(self.stefan_number)

    def front(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        return 2.0 * self.lam * math.sqrt(self.material.alpha * t)

    def temperature(self, x, t: float):
        """Exact temperature at depth ``x`` and time ``t`` (zero inside the liquid)."""
        x = np.asarray(x, dtype=float)
        if t <= 0.0:
            return np.full_like(x, self.material.t_melt)
        eta = x / (2.0 * math.sqrt(self.material.alpha * t))
        solid = self.t_surface + (self.material.t_melt - self.t_surface) \
            * _erf(eta) / math.erf(self.lam)
        return np.where(x <= self.front(t), solid, self.material.t_melt)


def _erf(x: np.ndarray) -> np.ndarray:
    from scipy.special import erf

    return erf(x)


@dataclass
class EnthalpySlab1D:
    """Implicit enthalpy-porosity conduction solver on a uniform 1-D slab."""

    length: float
    nx: int
    material: PhaseChangeMaterial
    t_init: float
    surface: RadialBoundary
    back: RadialBoundary = field(default_factory=lambda: RadialBoundary(q0=0.0))

    def __post_init__(self) -> None:
        if self.nx < 2:
            raise ValueError("need at least two cells")
        self.dx = np.full(self.nx, float(self.length) / self.nx)
        self.xc = (np.arange(self.nx) + 0.5) * self.dx[0]
        self.h = self.material.enthalpy(np.full(self.nx, float(self.t_init)))
        self.t, self.f_l = self.material.temperature(self.h)
        k = self.material.k
        self.cond = 1.0 / (0.5 * self.dx[:-1] / k + 0.5 * self.dx[1:] / k)
        # boundary: the external resistance is placed in series with the half cell,
        # so a Dirichlet wall value is imposed at the *face*, not at the cell centre.
        self.r_half = 0.5 * self.dx[0] / k
        self.r_half_back = 0.5 * self.dx[-1] / k
        self.cb = 0.0 if math.isinf(self.surface.resistance) \
            else 1.0 / (self.r_half + self.surface.resistance)
        self.cb_back = 0.0 if math.isinf(self.back.resistance) \
            else 1.0 / (self.r_half_back + self.back.resistance)
        self.balance_in = 0.0
        self.balance_out = 0.0
        self.time = 0.0

    # ---------------------------------------------------------------- utilities
    def _surface_flux(self, t_cell) -> float:
        """Heat flux entering the first cell through the surface face [W/m^2]."""
        return self.cb * (self.surface.t_ext - float(t_cell)) + self.surface.q0

    def _back_flux(self, t_cell) -> float:
        return self.cb_back * (self.back.t_ext - float(t_cell)) + self.back.q0

    def surface_temperature(self) -> float:
        """Face (surface) temperature reconstructed from the first cell."""
        return float(self.t[0]) - self.r_half * self._surface_flux(self.t[0])

    def front(self) -> float:
        """Enthalpy-equivalent front position ``int (1 - f_l) dx``."""
        return float(np.sum(self.dx * (1.0 - self.f_l)))

    def front_liquid_fraction(self, level: float = 0.5) -> float:
        """Front from the ``f_l = level`` crossing (linear interpolation)."""
        f = self.f_l
        below = np.nonzero(f < level)[0]
        if below.size == 0:
            return 0.0
        j = int(below[-1])
        if j + 1 >= self.nx:
            return self.length
        f0, f1 = f[j], f[j + 1]
        frac = 0.0 if f1 == f0 else (level - f0) / (f1 - f0)
        return float((j + frac + 1.0) * self.dx[0])

    def energy(self) -> float:
        """Total enthalpy per unit area [J/m^2]."""
        return float(np.sum(self.material.rho * self.dx * self.h))

    # ------------------------------------------------------------------- solver
    def _assemble(self, dt: float, c_eff: np.ndarray, h_old: np.ndarray,
                  t_iter: np.ndarray):
        n = self.nx
        rho = self.material.rho
        a = rho * self.dx * c_eff / dt
        lower = np.zeros(n)
        upper = np.zeros(n)
        diag = a.copy()
        diag[:-1] += self.cond
        diag[1:] += self.cond
        upper[:-1] = -self.cond
        lower[1:] = -self.cond
        rhs = rho * self.dx / dt * (c_eff * t_iter - self.material.enthalpy(t_iter) + h_old)

        diag[0] += self.cb
        rhs[0] += self.cb * self.surface.t_ext + self.surface.q0
        diag[-1] += self.cb_back
        rhs[-1] += self.cb_back * self.back.t_ext + self.back.q0
        ab = np.zeros((3, n))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = diag
        ab[2, :-1] = lower[1:]
        return ab, rhs

    def advance(self, dt: float, max_iter: int = 40, tol: float = 1.0e-9) -> float:
        """Advance one implicit time step; returns the Picard update norm [K]."""
        h_old = self.h.copy()
        t_iter = self.t.copy()
        t_new = t_iter
        for _ in range(max_iter):
            c_eff = self.material.capacity(t_iter)
            ab, rhs = self._assemble(dt, c_eff, h_old, t_iter)
            t_new = sla.solve_banded((1, 1), ab, rhs)
            delta = float(np.max(np.abs(t_new - t_iter)))
            t_iter = t_new
            if delta < tol:
                break
        # conservative enthalpy update from the converged face fluxes
        net = np.zeros(self.nx)
        net[:-1] += self.cond * (t_new[1:] - t_new[:-1])
        net[1:] += self.cond * (t_new[:-1] - t_new[1:])
        q_s = self._surface_flux(t_new[0])
        q_b = self._back_flux(t_new[-1])
        net[0] += q_s
        net[-1] += q_b
        self.h = h_old + dt / (self.material.rho * self.dx) * net
        self.t, self.f_l = self.material.temperature(self.h)
        self.balance_in += dt * q_s
        self.balance_out += dt * q_b
        self.time += dt
        return delta

    def run(self, t_end: float, dt: float, max_iter: int = 40,
            tol: float = 1.0e-9, record=None):
        """March to ``t_end``; ``record`` = list of times at which to snapshot."""
        snapshots = []
        targets = sorted(float(t) for t in (record or []))
        next_target = targets.pop(0) if targets else math.inf
        n_steps = int(round(t_end / dt))
        for step in range(n_steps):
            self.advance(dt, max_iter=max_iter, tol=tol)
            while self.time >= next_target - 1e-12:
                snapshots.append(self.snapshot())
                next_target = targets.pop(0) if targets else math.inf
        return snapshots

    def snapshot(self) -> dict:
        return {
            "time": self.time,
            "x": self.xc.copy(),
            "temperature": self.t.copy(),
            "liquid_fraction": self.f_l.copy(),
            "front": self.front(),
            "front_liquid_fraction": self.front_liquid_fraction(),
            "enthalpy_total": self.energy(),
        }

    def ledger(self) -> dict:
        """Discrete energy balance of the whole run."""
        h_change = self.energy() - float(np.sum(
            self.material.rho * self.dx * self.material.enthalpy(
                np.full(self.nx, self.t_init))))
        supplied = self.balance_in + self.balance_out
        scale = max(abs(supplied), abs(h_change), 1e-30)
        return {
            "enthalpy_change": h_change,
            "boundary_supplied": supplied,
            "imbalance": h_change - supplied,
            "imbalance_rel": abs(h_change - supplied) / scale,
            "surface_absorbed": self.balance_in,
            "back_absorbed": self.balance_out,
        }