"""Equations of state for the compressible multi-material surface.

The compressible capability of :mod:`tensorfvm` is deliberately minimal and
auditable: every state in the solver is a two-material (``nmat == 2``) mixture
described by mass fraction ``Y`` of material 2, closed with the stiffened-gas
equation of state

.. math::  p = (\\gamma - 1)\\,\\rho e - \\gamma\\, p_\\infty ,

which reduces to the ideal gas for ``p_inf == 0``.  The mixing rules are the
standard Allaire-type algebraic closure for stiffened gases (Allaire, Clerc &
Kokh 2002, *J. Comput. Phys.* 181, 577): with mass fractions ``Y_k``,

.. math::

   \\beta(Y) = \\sum_k \\frac{Y_k}{\\gamma_k - 1}, \\qquad
   \\Pi(Y)   = \\sum_k \\frac{Y_k \\gamma_k p_{\\infty,k}}{\\gamma_k - 1},

and the mixture EOS is ``p = (rho e - Pi)/beta``.  The mixture sound speed is
taken from the ``(p + p_inf)/rho`` form with the mixture parameters

.. math::

   \\gamma(Y) = 1 + \\frac{1}{\\beta}, \\qquad
   p_\\infty(Y) = \\frac{\\Pi}{\\gamma \\beta},
   \\qquad c^2 = \\gamma \\frac{p + p_\\infty}{\\rho}.

All functions are written for ``torch`` tensors and are broadcast safe, so the
same code path serves scalar unit tests and full field solves.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch

__all__ = [
    "StiffenedGasMixture",
    "AIR",
    "HELIUM",
    "SF6",
]

_TINY = torch.finfo(torch.float64).tiny


def _col(vec: Sequence[float], ndim: int) -> torch.Tensor:
    """Reshape a per-material constant vector so it broadcasts over ``ndim``."""
    t = torch.as_tensor(list(vec), dtype=torch.float64)
    return t.view(-1, *([1] * ndim))


class StiffenedGasMixture:
    """Two-material stiffened-gas mixture keyed by the mass fraction ``Y``.

    For exactly two materials both mixture sums are *linear* in ``Y``

    .. math::

        \\beta(Y) = \\frac{1-Y}{\\gamma_1-1} + \\frac{Y}{\\gamma_2-1}, \\qquad
        \\Pi(Y) = \\frac{(1-Y)\\gamma_1 p_{\\infty,1}}{\\gamma_1-1}
                 + \\frac{Y \\gamma_2 p_{\\infty,2}}{\\gamma_2-1},

    so they are evaluated from two precomputed scalars instead of a stacked
    per-material tensor.  That matters: the mixture closure is called a dozen
    times per flux evaluation, and the stacked form dominated the per-step cost
    of the 2-D solver.
    """

    def __init__(
        self,
        gammas: Sequence[float],
        pinfs: Sequence[float] | None = None,
        names: Sequence[str] | None = None,
    ) -> None:
        gammas = tuple(float(g) for g in gammas)
        if len(gammas) != 2:
            raise ValueError("the multi-material closure supports exactly two materials")
        pinfs = (0.0, 0.0) if pinfs is None else tuple(float(p) for p in pinfs)
        if len(pinfs) != 2:
            raise ValueError("gammas and pinfs must have the same length")
        if any(g <= 1.0 for g in gammas):
            raise ValueError("stiffened-gas gamma must exceed one")
        if any(p < 0.0 for p in pinfs):
            raise ValueError("stiffened-gas p_inf must be non-negative")
        names = tuple(names) if names is not None else ("mat1", "mat2")
        if len(names) != 2:
            raise ValueError("names must match the material count")
        self.gammas = gammas
        self.pinfs = pinfs
        self.names = names
        self.nmat = 2
        g1, g2 = gammas
        p1, p2 = pinfs
        b1, b2 = 1.0 / (g1 - 1.0), 1.0 / (g2 - 1.0)
        pi1, pi2 = g1 * p1 / (g1 - 1.0), g2 * p2 / (g2 - 1.0)
        self._b = (b1, b2 - b1)
        self._pi = (pi1, pi2 - pi1)
        self._gamma = gammas

    # -- mixing rules (linear in Y) --------------------------------------
    def beta(self, Y: torch.Tensor) -> torch.Tensor:
        """``1/(gamma-1)`` of the mixture."""
        return self._b[0] + self._b[1] * Y

    def pi(self, Y: torch.Tensor) -> torch.Tensor:
        """``Pi = gamma p_inf/(gamma-1)`` of the mixture."""
        return self._pi[0] + self._pi[1] * Y

    def gamma_eff(self, Y: torch.Tensor) -> torch.Tensor:
        return 1.0 + 1.0 / self.beta(Y)

    def pinf_eff(self, Y: torch.Tensor) -> torch.Tensor:
        beta = self.beta(Y)
        return self.pi(Y) / ((1.0 + 1.0 / beta) * beta)

    # -- thermodynamic closure -------------------------------------------
    def pressure(self, rho: torch.Tensor, e: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Pressure from density, specific internal energy and mass fraction."""
        return (rho * e - self.pi(Y)) / self.beta(Y)

    def internal_energy(self, rho: torch.Tensor, p: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Specific internal energy from ``(rho, p, Y)``."""
        return (p * self.beta(Y) + self.pi(Y)) / rho

    def sound_speed(self, rho: torch.Tensor, p: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Mixture sound speed ``c = sqrt(gamma (p + p_inf)/rho)``."""
        beta = self.beta(Y)
        gamma = 1.0 + 1.0 / beta
        pinf = self.pi(Y) / (gamma * beta)
        return torch.sqrt(torch.clamp(gamma * (p + pinf) / rho, min=_TINY))

    def energy_density(self, rho: torch.Tensor, u: torch.Tensor, v: torch.Tensor,
                       p: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Total energy density ``rho e + rho |u|^2/2`` in one pass."""
        return (p * self.beta(Y) + self.pi(Y)
                + 0.5 * rho * (u * u + v * v))

    def pressure_from_conserved(self, rho, u, v, E, Y) -> torch.Tensor:
        """Pressure from the conserved fields, with the same algebra as above."""
        beta = self.beta(Y)
        return (E - self.pi(Y) - 0.5 * rho * (u * u + v * v)) / beta


AIR = {"gamma": 1.4, "pinfl": 0.0}
HELIUM = {"gamma": 5.0 / 3.0, "pinfl": 0.0}
SF6 = {"gamma": 1.09, "pinfl": 0.0}


def standard_state(gamma: float, rho: float, p: float, vel: float = 0.0) -> dict:
    """Convenience primitive state (used by cases and tests)."""
    e = p / ((gamma - 1.0) * rho)
    c = math.sqrt(gamma * p / rho)
    return {"rho": rho, "u": vel, "p": p, "e": e, "c": c, "gamma": gamma}