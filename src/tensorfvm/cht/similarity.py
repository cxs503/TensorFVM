"""Similarity solutions of the laminar flat-plate boundary layer.

Two ordinary differential problems are integrated with a fourth-order Runge-Kutta
march, so the verification references used by :mod:`tensorfvm.cht.flatplate` are
computed here instead of being copied from a table:

Blasius (velocity)::

    f''' + 1/2 f f'' = 0,   f(0)=f'(0)=0,  f'(inf)=1          (eta = y sqrt(U/nu x))

Pohlhausen (thermal, isothermal wall, ``Pr`` = Prandtl number)::

    theta'' + Pr/2 f theta' = 0,  theta(0)=0, theta(inf)=1

With ``Nu_x = -x dT/dy|_w / (T_w - T_inf)`` the similarity reduction gives::

    Nu_x / sqrt(Re_x) = theta'(0)

which is the quantity the FVM energy solve is graded against.  The classical
engineering correlation ``0.332 Pr^(1/3)`` is returned as a secondary anchor; the
ODE value is the primary reference (it is exact up to integration error).

Blasius wall shear is reported as well (``f''(0) = 0.332057...``) so a wrong
shooting root cannot slip through silently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

BLASIUS_WALL_SHEAR = 0.33205733621519630  # f''(0), classical value


@dataclass(frozen=True)
class BlasiusSolution:
    """Tabulated similarity solution on a uniform ``eta`` grid."""

    eta: np.ndarray
    f: np.ndarray
    fp: np.ndarray
    fpp: np.ndarray
    wall_shear: float

    def psi(self, x: np.ndarray, y: np.ndarray, u_inf: float, nu: float) -> np.ndarray:
        """Streamfunction ``psi = sqrt(nu U x) f(eta)`` on arbitrary points."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        eta = y * np.sqrt(u_inf / (nu * np.maximum(x, 1e-300)))
        return np.sqrt(nu * u_inf * np.maximum(x, 0.0)) * np.interp(eta, self.eta, self.f)

    def velocity(self, x, y, u_inf: float, nu: float):
        eta = np.asarray(y, dtype=float) * np.sqrt(u_inf / (nu * np.maximum(x, 1e-300)))
        u = u_inf * np.interp(eta, self.eta, self.fp)
        return u


def _blasius_rhs(state: np.ndarray, pr: float | None) -> np.ndarray:
    """Right-hand side of the coupled Blasius(+Pohlhausen) system."""
    f, fp, fpp = state[0], state[1], state[2]
    d = np.array([fp, fpp, -0.5 * f * fpp])
    if pr is None:
        return d
    th, thp = state[3], state[4]
    return np.concatenate([d, [thp, -0.5 * pr * f * thp]])


def _rk4_march(initial, eta_max: float, h: float, pr: float | None) -> np.ndarray:
    initial = np.asarray(initial, dtype=float)
    n = int(round(eta_max / h))
    out = np.empty((n + 1, initial.size))
    state = np.array(initial, dtype=float)
    out[0] = state
    for k in range(n):
        k1 = _blasius_rhs(state, pr)
        k2 = _blasius_rhs(state + 0.5 * h * k1, pr)
        k3 = _blasius_rhs(state + 0.5 * h * k2, pr)
        k4 = _blasius_rhs(state + h * k3, pr)
        state = state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[k + 1] = state
    return out


def solve_blasius(eta_max: float = 12.0, h: float = 0.002) -> BlasiusSolution:
    """Integrate the Blasius problem by shooting on ``f''(0)``."""
    guess = 0.33
    for _ in range(60):
        traj = _rk4_march([0.0, 0.0, guess], eta_max, h, None)
        err = traj[-1, 1] - 1.0
        if abs(err) < 1e-12:
            break
        d = 1e-6 * max(1.0, abs(guess))
        traj2 = _rk4_march([0.0, 0.0, guess + d], eta_max, h, None)
        slope = (traj2[-1, 1] - traj[-1, 1]) / d
        guess -= err / slope
    eta = np.linspace(0.0, eta_max, traj.shape[0])
    return BlasiusSolution(
        eta=eta,
        f=traj[:, 0],
        fp=traj[:, 1],
        fpp=traj[:, 2],
        wall_shear=float(traj[0, 2]),
    )


def solve_pohlhausen(pr: float, blasius: BlasiusSolution | None = None,
                     eta_max: float = 12.0, h: float = 0.002) -> float:
    """Return ``theta'(0) = Nu_x / sqrt(Re_x)`` for Prandtl number ``pr``.

    The velocity field is the converged Blasius one (wall shear ``f''(0)`` from
    :func:`solve_blasius`), and the linear thermal problem is integrated with the
    arbitrary normalisation ``theta'(0) = 1``, then rescaled by ``theta(inf)``.
    """
    if pr <= 0.0:
        raise ValueError("Prandtl number must be positive")
    if blasius is None:
        blasius = solve_blasius(eta_max=eta_max, h=h)
    traj = _rk4_march([0.0, 0.0, blasius.wall_shear, 0.0, 1.0], eta_max, h, pr)
    theta_inf = float(traj[-1, 3])
    if theta_inf <= 0.0:
        raise RuntimeError("thermal similarity profile did not grow; check eta_max")
    return 1.0 / theta_inf


def pohlhausen_nusselt(pr: float, blasius: BlasiusSolution | None = None) -> float:
    """``Nu_x / sqrt(Re_x)`` from the Pohlhausen similarity solution."""
    return solve_pohlhausen(pr, blasius=blasius)


def classical_correlation(pr: float) -> float:
    """Engineering form ``0.332 Pr^(1/3)`` (valid Pr >= 0.6)."""
    return 0.332 * pr ** (1.0 / 3.0)


def unheated_start_factor(x_over_x0: float) -> float:
    """Pohlhausen correction for an unheated starting length, ``x0`` small.

    ``Nu_x = 0.332 Re_x^{1/2} Pr^{1/3} / (1 - (x0/x)^{3/4})^{1/3}``.
    """
    if x_over_x0 <= 1.0:
        raise ValueError("x/x0 must exceed 1")
    return 1.0 / (1.0 - x_over_x0 ** -0.75) ** (1.0 / 3.0)


@dataclass(frozen=True)
class UniformFluxSolution:
    """Uniform-wall-heat-flux (UHF) thermal similarity solution.

    The temperature is written ``T - T_inf = (q_w / k) sqrt(nu x / U) phi(eta)``,
    which turns the boundary-layer energy equation into

    ``phi'' + Pr/2 (f phi' - f' phi) = 0,  phi'(0) = -1,  phi(inf) = 0``

    and gives ``Nu_x = q_w x / (k (T_w - T_inf)) = sqrt(Re_x) / phi(0)``.  The
    extra ``-f' phi`` term (absent from the isothermal Pohlhausen problem) is what
    makes the uniform-flux coefficient about 36 % larger.
    """

    phi0: float
    coefficient: float
    prandtl: float

    @property
    def nusselt_coefficient(self) -> float:
        """``Nu_x / sqrt(Re_x)`` for a uniform wall heat flux."""
        return self.coefficient


def solve_uniform_flux(pr: float, blasius: BlasiusSolution | None = None,
                       eta_max: float = 12.0, h: float = 0.002) -> UniformFluxSolution:
    """Integrate the uniform-heat-flux thermal problem and return ``phi(0)``.

    The problem is linear, so the wall value is split as ``phi = s A + B`` with
    ``A(0)=1, A'(0)=0`` and ``B(0)=0, B'(0)=-1``; ``phi(inf)=0`` then fixes
    ``s = -B(inf)/A(inf)``.
    """
    if pr <= 0.0:
        raise ValueError("Prandtl number must be positive")
    if blasius is None:
        blasius = solve_blasius(eta_max=eta_max, h=h)
    n = int(round(eta_max / h))
    eta = np.linspace(0.0, eta_max, n + 1)
    f = np.interp(eta, blasius.eta, blasius.f)
    fp = np.interp(eta, blasius.eta, blasius.fp)

    def march(phi0: float, dphi0: float) -> tuple[float, float]:
        state = np.array([phi0, dphi0], dtype=float)
        for k in range(n):
            def rhs(st, i=k):
                return np.array([st[1], -0.5 * pr * (f[i] * st[1] - fp[i] * st[0])])

            k1 = rhs(state)
            k2 = rhs(state + 0.5 * h * k1)
            k3 = rhs(state + 0.5 * h * k2)
            k4 = rhs(state + h * k3)
            state = state + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return float(state[0]), float(state[1])

    a_inf, _ = march(1.0, 0.0)
    b_inf, _ = march(0.0, -1.0)
    if a_inf <= 0.0:
        raise RuntimeError("uniform-flux similarity: A(inf) must stay positive")
    phi0 = -b_inf / a_inf
    if phi0 <= 0.0:
        raise RuntimeError("uniform-flux similarity: phi(0) must be positive")
    return UniformFluxSolution(phi0=float(phi0), coefficient=float(1.0 / phi0),
                               prandtl=float(pr))


def uniform_flux_nusselt(pr: float, blasius: BlasiusSolution | None = None) -> float:
    """``Nu_x / sqrt(Re_x)`` for a uniform wall heat flux."""
    return solve_uniform_flux(pr, blasius=blasius).coefficient


def classical_uniform_flux_correlation(pr: float) -> float:
    """Engineering form ``0.453 Pr^(1/3)`` (laminar, uniform heat flux)."""
    return 0.453 * pr ** (1.0 / 3.0)


def blasius_streamfunction(traj: BlasiusSolution):
    """Convenience closure ``psi(x, y)`` bound to the tabulated solution."""
    def psi(x, y, u_inf: float = 1.0, nu: float = 1.0):
        return traj.psi(x, y, u_inf, nu)
    return psi


def displacement_thickness(traj: BlasiusSolution) -> float:
    """``delta1 sqrt(Re_x)`` for information/debugging (classical 1.7208)."""
    return float(np.trapz(1.0 - traj.fp, traj.eta))


def momentum_thickness(traj: BlasiusSolution) -> float:
    """``delta2 sqrt(Re_x)`` (classical 0.6641)."""
    return float(np.trapz(traj.fp * (1.0 - traj.fp), traj.eta))


def similarity_eta(y: np.ndarray, x: float, u_inf: float, nu: float) -> np.ndarray:
    return np.asarray(y, dtype=float) * math.sqrt(u_inf / (nu * max(x, 1e-300)))