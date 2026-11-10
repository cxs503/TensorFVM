"""Immersed-solid (Brinkman-penalty) incompressible flow with a *masked* projection.

New capability surface: the projection step of an immersed-solid fractional-step
solver is written as a **masked Poisson problem** -- solid cells are excluded from
the pressure-correction system and every face of a solid cell carries a hard zero
normal velocity (homogeneous Neumann for the correction) -- instead of the legacy
``kappa`` penalty relaxation that is applied *after* an unmasked projection and
therefore re-introduces a mass source inside the immersed solid.

Modules
-------
``geometry``    mask-only Cartesian geometries (backward-facing step, symmetric
                sudden expansion) with an exactly realised expansion ratio.
``projection``  masked 5-point Poisson operator, masked Jacobi and geometric
                multigrid preconditioned conjugate gradients.
``solver``      staggered MAC fractional-step solver with ``projection="masked"``
                and ``projection="legacy"`` arms plus a discrete mass ledger.
"""

from .geometry import CartesianGrid, backward_facing_step, symmetric_expansion
from .projection import MaskedPoisson, masked_pcg
from .solver import ImmersedConfig, ImmersedFlow, ProjectionMode

__all__ = [
    "CartesianGrid",
    "backward_facing_step",
    "symmetric_expansion",
    "MaskedPoisson",
    "masked_pcg",
    "ImmersedConfig",
    "ImmersedFlow",
    "ProjectionMode",
]