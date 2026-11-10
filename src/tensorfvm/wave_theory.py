"""Linear and second-order Stokes wave theory for numerical wave tank forcing.

Provides analytical wave kinematics (surface elevation, velocity potential,
particle velocities) for regular waves in finite depth, supporting both
Airy (linear) and Stokes-II formulations.

References
----------
- Dean & Dalrymple, "Water Wave Mechanics for Engineers and Scientists"
- Fenton, "The numerical solution of steady water wave problems", 1988
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

__all__ = [
    "WaveParams",
    "dispersion_relation",
    "stokes1_elevation",
    "stokes1_velocity",
    "stokes2_elevation",
    "stokes2_velocity",
    "relaxation_weight",
    "damping_weight",
]


@dataclass
class WaveParams:
    """Regular wave parameters."""
    H: float
    T: float
    h: float
    g: float = 9.81
    order: int = 2

    def __post_init__(self):
        if self.H <= 0:
            raise ValueError("Wave height H must be positive")
        if self.T <= 0:
            raise ValueError("Wave period T must be positive")
        if self.h <= 0:
            raise ValueError("Water depth h must be positive")
        if self.g <= 0:
            raise ValueError("Gravity g must be positive")
        if self.order not in (1, 2):
            raise ValueError("order must be 1 (Airy) or 2 (Stokes-II)")

    @property
    def omega(self) -> float:
        return 2.0 * math.pi / self.T

    @property
    def k(self) -> float:
        return dispersion_relation(self.omega, self.h, self.g)

    @property
    def wavelength(self) -> float:
        return 2.0 * math.pi / self.k

    @property
    def celerity(self) -> float:
        return self.omega / self.k

    @property
    def steepness(self) -> float:
        return self.H / self.wavelength

    @property
    def kh(self) -> float:
        return self.k * self.h


def dispersion_relation(omega: float, h: float, g: float = 9.81,
                        tol: float = 1e-12, max_iter: int = 100) -> float:
    """Solve ω² = gk·tanh(kh) for k using Newton-Raphson."""
    omega2 = omega * omega
    k = omega2 / g
    for _ in range(max_iter):
        kh = k * h
        tanh_kh = math.tanh(kh)
        f = g * k * tanh_kh - omega2
        if abs(f) < tol * omega2:
            break
        sech2 = 1.0 / (math.cosh(kh) ** 2)
        fp = g * (tanh_kh + kh * sech2)
        k -= f / fp
        if k <= 0:
            k = omega2 / g
    return k


def stokes1_elevation(x: torch.Tensor, t: float, wp: WaveParams) -> torch.Tensor:
    """Linear (Airy) free-surface elevation η(x,t)."""
    return (wp.H / 2.0) * torch.cos(wp.k * x - wp.omega * t)


def stokes1_velocity(x: torch.Tensor, z: torch.Tensor, t: float,
                     wp: WaveParams) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear (Airy) particle velocities at (x, z, t)."""
    phase = wp.k * x - wp.omega * t
    cosh_khz = torch.cosh(wp.k * (wp.h + z))
    sinh_khz = torch.sinh(wp.k * (wp.h + z))
    sinh_kh = math.sinh(wp.k * wp.h)

    u = (wp.H / 2.0) * wp.omega * cosh_khz / sinh_kh * torch.cos(phase)
    w = (wp.H / 2.0) * wp.omega * sinh_khz / sinh_kh * torch.sin(phase)
    return u, w


def stokes2_elevation(x: torch.Tensor, t: float, wp: WaveParams) -> torch.Tensor:
    """Second-order Stokes free-surface elevation η(x,t)."""
    a = wp.H / 2.0
    theta = wp.k * x - wp.omega * t
    kh = wp.k * wp.h
    coth_kh = 1.0 / math.tanh(kh)
    sinh_kh = math.sinh(kh)
    B2 = 0.5 * wp.k * a * a * (coth_kh + 1.0 / (2.0 * sinh_kh ** 2))
    return a * torch.cos(theta) + B2 * torch.cos(2.0 * theta)


def stokes2_velocity(x: torch.Tensor, z: torch.Tensor, t: float,
                     wp: WaveParams) -> tuple[torch.Tensor, torch.Tensor]:
    """Second-order Stokes particle velocities at (x, z, t)."""
    a = wp.H / 2.0
    theta = wp.k * x - wp.omega * t
    kh = wp.k * wp.h

    # First-order (Airy) part
    cosh_khz = torch.cosh(wp.k * (wp.h + z))
    sinh_khz = torch.sinh(wp.k * (wp.h + z))
    sinh_kh = math.sinh(kh)

    u1 = a * wp.omega * cosh_khz / sinh_kh * torch.cos(theta)
    w1 = a * wp.omega * sinh_khz / sinh_kh * torch.sin(theta)

    # Second-order correction
    cosh_2khz = torch.cosh(2.0 * wp.k * (wp.h + z))
    sinh_2khz = torch.sinh(2.0 * wp.k * (wp.h + z))

    u2 = (3.0 / 4.0) * a * a * wp.omega * wp.k * (
        cosh_2khz / (sinh_kh ** 2)
    ) * torch.cos(2.0 * theta)
    w2 = (3.0 / 4.0) * a * a * wp.omega * wp.k * (
        sinh_2khz / (sinh_kh ** 2)
    ) * torch.sin(2.0 * theta)

    # Stokes drift (mean Lagrangian velocity, second order)
    u_drift = a * a * wp.omega * wp.k * (
        torch.cosh(2.0 * wp.k * (wp.h + z)) / (2.0 * sinh_kh ** 2)
    )

    return u1 + u2 + u_drift, w1 + w2


def relaxation_weight(x: torch.Tensor, x_start: float, x_end: float,
                      beta: float = 3.0) -> torch.Tensor:
    """Relaxation zone weight function for wave generation/absorption."""
    xi = torch.clamp((x - x_start) / (x_end - x_start), 0.0, 1.0)
    return 1.0 - torch.exp(-beta * xi ** 3.0)


def damping_weight(x: torch.Tensor, x_start: float, x_end: float,
                   gamma: float = 1.0) -> torch.Tensor:
    """Damping zone weight for wave absorption."""
    xi = torch.clamp((x - x_start) / (x_end - x_start), 0.0, 1.0)
    return gamma * xi ** 2.0
