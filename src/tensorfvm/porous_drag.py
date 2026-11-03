"""Volume-averaged porous-media resistance (Darcy--Forchheimer) for the 2D SIMPLE solver.

Motivation
----------
``TensorFVM`` had no porous-media capability at all (``grep -ri "darcy\\|forchheimer"
src tests scripts`` was empty) although the volume-averaged momentum equation

    rho (u.grad) u = -grad p + mu lap(u) - (mu/K) u - rho*beta*|u| u

is the standard model for flow through packed beds, filters, grilles, screens and
canopies.  The extra sink is a *momentum* sink only: it must not appear as a mass
source, so mass conservation is unchanged and the discrete continuity equation of
:class:`tensorfvm.solver.SimpleSolver` keeps working untouched.

Discretisation
--------------
``SimpleSolver._momentum`` returns ``(diagonal, coeffs, source, active, width)``
for one staggered velocity component whose face control volume is
``dy*width`` (u-faces) or ``dx*width`` (v-faces), i.e. ``volume = spacing*width``.
Its pressure source is ``-dp/ds*volume``, so a body force ``f`` per unit volume
enters as ``source + volume*f``.

* Darcy term ``-(mu/K) u`` is treated **implicitly** (``diagonal += volume*mu/K``).
  Being a positive sink it only strengthens diagonal dominance, so the SIMPLE
  iteration stays stable for arbitrarily small permeability.
* Forchheimer term ``-rho*beta*|u| u`` is **lagged** one outer iteration
  (``source -= volume*rho*beta*|u_old| u_old``), which is the standard
  Picard linearisation; it also adds to the diagonal.

Because the sink of the u-component is centred on the u-face, the discrete steady
equation is a consistent central discretisation of the Brinkman equation, whose
exact channel solution is available in closed form (see :func:`brinkman_profile`),
which makes the implementation machine-verifiable instead of merely plausible.

Reference forms used for verification
------------------------------------
* Brinkman (1947) / Darcy--Brinkman, half-height ``h``, ``ell = sqrt(K)``:

      u(y) = A [1 - cosh((y-h)/ell)/cosh(h/ell)],   A = u_mean/[1 - ell*tanh(h/ell)/h]

  with the *same* mean velocity as the inlet (incompressibility), and the
  pressure gradient implied by Darcy's law ``dp/dx = -mu*A/K``.
* Forchheimer (1901) bulk law ``-dp/dx = (mu/K) u + rho*beta*u*|u|``, solved for
  the bulk velocity by :func:`forchheimer_bulk_velocity`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .solver import SimpleSolver

__all__ = ["PorousMedia", "PorousChannelSolver", "brinkman_profile",
           "brinkman_pressure_gradient", "forchheimer_bulk_velocity"]


@dataclass(frozen=True)
class PorousMedia:
    """Isotropic porous resistance.

    ``permeability_m2`` is the Darcy permeability ``K`` [m^2]; ``form_drag_1_per_m``
    is the Forchheimer coefficient ``beta`` [1/m] in ``R = (mu/K) u + rho beta |u| u``
    (equivalently ``beta = C_F/sqrt(K)`` with the Ergun/Ward form-drag constant
    ``C_F``).  ``beta = 0`` is the pure Darcy--Brinkman limit.
    """

    permeability_m2: float
    form_drag_1_per_m: float = 0.0

    def __post_init__(self):
        for name in ("permeability_m2", "form_drag_1_per_m"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.permeability_m2 <= 0:
            raise ValueError("permeability_m2 must be strictly positive")


class PorousChannelSolver(SimpleSolver):
    """Production SIMPLE solver plus a volume-averaged porous momentum sink.

    ``porous=None`` reproduces :class:`SimpleSolver` bit for bit (the sink is
    skipped), which is the regression anchor for the extension.
    """

    def __new__(cls, config, porous: PorousMedia | None = None):
        # Bypass the SimpleSolver backend dispatch: this class always wraps the
        # base MAC/SIMPLE implementation that it extends.
        return object.__new__(cls)

    def __init__(self, config, porous: PorousMedia | None = None):
        super().__init__(config)
        self.porous = porous

    def _momentum(self, component):
        diagonal, coeffs, source, active, width = super()._momentum(component)
        media = self.porous
        if media is None:
            return diagonal, coeffs, source, active, width
        c = self.config
        volume = (self.dy if component == "u" else self.dx) * width
        field = self.u if component == "u" else self.v
        zero = torch.zeros_like(diagonal)
        # Darcy: implicit positive sink (u >= 0 everywhere in the discretisation).
        diagonal = diagonal + torch.where(
            active, volume * c.viscosity / media.permeability_m2, zero)
        # Forchheimer: Picard linearisation -rho*beta*|u|u ~ -rho*beta*|u_old| u_new.
        # The magnitude is lagged but the transported velocity is implicit, which
        # keeps the sink on the diagonal (monotone) even for large beta.
        if media.form_drag_1_per_m > 0.0:
            quadratic = volume * c.density * media.form_drag_1_per_m * field.abs()
            diagonal = diagonal + torch.where(active, quadratic, zero)
        return diagonal, coeffs, source, active, width


def brinkman_profile(y, u_mean: float, height: float, permeability_m2: float):
    """Exact Darcy--Brinkman velocity profile with prescribed mean velocity.

    ``y`` may be any array-like of wall-normal coordinates in ``[0, height]``;
    returns a ``torch`` tensor in float64 on the same device as ``y``.
    """
    y = torch.as_tensor(y, dtype=torch.float64)
    h = 0.5 * height
    ell = math.sqrt(permeability_m2)
    shape = 1.0 - ell * math.tanh(h / ell) / h
    amplitude = u_mean / shape
    return amplitude * (1.0 - torch.cosh((y - h) / ell) / math.cosh(h / ell))


def brinkman_pressure_gradient(u_mean: float, height: float,
                               permeability_m2: float, viscosity: float) -> float:
    """``dp/dx`` [Pa/m] implied by Darcy's law for the Brinkman channel profile."""
    h = 0.5 * height
    ell = math.sqrt(permeability_m2)
    shape = 1.0 - ell * math.tanh(h / ell) / h
    return -viscosity * (u_mean / shape) / permeability_m2


def forchheimer_bulk_velocity(pressure_gradient: float, viscosity: float,
                              density: float, permeability_m2: float,
                              form_drag_1_per_m: float) -> float:
    """Positive root of ``(mu/K) u + rho beta u^2 = -dp/dx`` (Forchheimer law)."""
    forcing = -pressure_gradient
    linear = viscosity / permeability_m2
    if form_drag_1_per_m <= 0.0:
        return forcing / linear
    quadratic = density * form_drag_1_per_m
    return (-linear + math.sqrt(linear * linear + 4.0 * quadratic * forcing)) / (
        2.0 * quadratic)