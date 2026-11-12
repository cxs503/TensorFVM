"""Hydrostatic pressure initialization for arbitrary VOF fields.

The dam-break (and other transient two-phase cases) require the pressure field
to be initialized to the hydrostatic profile *before* the transient simulation
starts.  Without this, the fluid column collapses violently when gravity is
turned on, producing CFL violations and completely wrong surge-front positions.

This module provides a function that computes the hydrostatic pressure for an
arbitrary alpha field by integrating the discrete face densities that the
momentum equation uses.

Reference
---------
The discrete hydrostatic balance requires:
    dp/dy + rho_face * g = 0
where rho_face is the density interpolated to the v-face (the face where the
pressure gradient is evaluated in the MAC staggered grid).

For a column of liquid with a free surface at the top, the pressure at cell
(j, i) is:
    p[j,i] = sum_{k=j}^{ny-1} rho_face[k,i] * g * dy
where rho_face[k,i] = 0.5 * (rho[k,i] + rho[k+1,i]) for k < ny-1,
and rho_face[ny-1,i] = rho_gas (top face is atmospheric).

This ensures that a resting fluid column is an exact solution of the discrete
momentum equation, regardless of the interface shape.
"""

from __future__ import annotations

import torch


def hydrostatic_pressure_from_alpha(
    alpha: torch.Tensor,
    dy: float,
    gravity: float,
    rho_liquid: float,
    rho_gas: float,
) -> torch.Tensor:
    """Compute hydrostatic pressure for an arbitrary VOF field.

    Parameters
    ----------
    alpha : torch.Tensor
        Volume fraction field, shape (ny, nx), values in [0, 1].
    dy : float
        Cell height in y direction.
    gravity : float
        Gravitational acceleration (positive, acts in -y direction).
    rho_liquid : float
        Liquid density.
    rho_gas : float
        Gas density.

    Returns
    -------
    torch.Tensor
        Hydrostatic pressure field, shape (ny, nx), same dtype/device as alpha.
        p = 0 at the top (atmospheric gauge), increasing downward.
    """
    ny, nx = alpha.shape
    dtype = alpha.dtype
    device = alpha.device

    # Cell-centered density
    rho = alpha * rho_liquid + (1.0 - alpha) * rho_gas

    # Face density at v-faces (between cell rows j and j+1)
    # rho_face[j] = 0.5 * (rho[j] + rho[j+1]) for j = 0..ny-2
    # rho_face[ny-1] = rho_gas (top face, atmospheric)
    rho_face = torch.zeros(ny, nx, dtype=dtype, device=device)
    rho_face[:-1, :] = 0.5 * (rho[:-1, :] + rho[1:, :])
    rho_face[-1, :] = rho_gas

    # Integrate downward from the top: p[j] = p[j+1] + rho_face[j] * g * dy
    # p[ny-1] = 0 (atmospheric gauge at top face)
    p = torch.zeros(ny, nx, dtype=dtype, device=device)
    for j in range(ny - 2, -1, -1):
        p[j, :] = p[j + 1, :] + rho_face[j, :] * gravity * dy

    return p
