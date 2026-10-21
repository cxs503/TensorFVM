"""Free-surface / open-boundary boundary conditions for two-phase VOF.

This module provides the boundary-condition handlers needed when a
free surface (air-water interface) can cross domain boundaries:

* :class:`FreeSurfaceBC` — applies an *atmospheric-pressure top* (Dirichlet
  p=0 gauge) and *pressure-outlet* side walls, and advects the VOF marker
  through the top boundary so that fluid leaving through the surface does
  not create spurious mass sources.

* :func:`apply_open_pressure` — sets the gauge pressure to zero on all
  open/free-surface faces, mimicking a pressure outlet.

* :func:`outflow_vof_flux` — computes the volume-fraction flux leaving the
  domain through open boundaries, so that the global volume ledger can be
  closed in the anchor validation.

The design is intentionally mesh-agnostic within the Cartesian staggered
framework of :mod:`tensorfvm.solver`: the BC flags are boolean masks aligned
with the MAC face arrays ``u`` (east, shape (ny, nx+1)) and ``v`` (north,
shape (ny+1, nx)).
"""

from __future__ import annotations

import torch

__all__ = [
    "FreeSurfaceBC",
    "apply_open_pressure",
    "outflow_vof_flux",
]


def apply_open_pressure(
    p: torch.Tensor, open_mask: torch.Tensor
) -> torch.Tensor:
    """Set gauge pressure to zero on open/free-surface cells."""
    return torch.where(open_mask, torch.zeros_like(p), p)


def outflow_vof_flux(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    dt: float,
) -> float:
    """Compute the net liquid volume leaving the domain through boundaries.

    Returns the *outflow* (positive = leaving) of the liquid phase through
    all four domain boundaries, using upwind face values of alpha.
    """
    # East boundary (last east face column): flux out = u[-1]*dy*alpha[-1]
    east_out = max(0.0, float(u[:, -1].sum().item())) * dy
    west_out = max(0.0, float(-u[:, 0].sum().item())) * dy
    north_out = max(0.0, float(v[-1, :].sum().item())) * dx
    south_out = max(0.0, float(-v[0, :].sum().item())) * dx
    # Liquid fraction at boundary cells
    a_e = float(alpha[:, -1].mean().item())
    a_w = float(alpha[:, 0].mean().item())
    a_n = float(alpha[-1, :].mean().item())
    a_s = float(alpha[0, :].mean().item())
    return dt * (east_out * a_e + west_out * a_w
                 + north_out * a_n + south_out * a_s)


class FreeSurfaceBC:
    """Boundary-condition manager for a free-surface / open-top domain.

    Parameters
    ----------
    ny, nx : int
        Grid resolution (cell-centre counts).
    dx, dy : float
        Uniform cell sizes.
    device : str
        Torch device.
    top_atmosphere : bool
        If True, the top (north) boundary is an atmospheric-pressure surface
        (p=0) and zero normal stress; velocity is free to advect out.
    bottom_wall : bool
        If True, the bottom is a no-slip wall (default).
    side_outlet : bool
        If True, left/right boundaries are pressure outlets (p=0, zero
        normal gradient for velocity).
    """

    def __init__(
        self,
        ny: int,
        nx: int,
        dx: float,
        dy: float,
        device: str = "cpu",
        top_atmosphere: bool = True,
        bottom_wall: bool = True,
        side_outlet: bool = True,
    ):
        self.ny = ny
        self.nx = nx
        self.dx = float(dx)
        self.dy = float(dy)
        self.device = device
        self.top_atmosphere = top_atmosphere
        self.bottom_wall = bottom_wall
        self.side_outlet = side_outlet
        # Pressure Dirichlet mask (where p is forced to 0 gauge)
        self.open_pressure = torch.zeros((ny, nx), dtype=torch.bool, device=device)
        if top_atmosphere:
            self.open_pressure[-1, :] = True
        if side_outlet:
            self.open_pressure[:, 0] = True
            self.open_pressure[:, -1] = True
        # Velocity boundary type masks
        # u faces: (ny, nx+1). Active = interior face between two fluid cells.
        # v faces: (ny+1, nx)
        self._init_velocity_masks()

    def _init_velocity_masks(self):
        """Build boundary-type masks for u and v face arrays."""
        # u active mask (True = solve momentum here)
        self.u_active = torch.ones((self.ny, self.nx + 1), dtype=torch.bool,
                                   device=self.device)
        # Inlet Dirichlet (left) is False (prescribed)
        self.u_active[:, 0] = False
        if self.side_outlet:
            # outlet u is unknown (zero gradient)
            self.u_active[:, -1] = True
        else:
            self.u_active[:, -1] = False

        # v active mask
        self.v_active = torch.ones((self.ny + 1, self.nx), dtype=torch.bool,
                                   device=self.device)
        if self.bottom_wall:
            self.v_active[0, :] = False  # no-slip bottom
        if self.top_atmosphere:
            self.v_active[-1, :] = True  # free to advect (open top)
        else:
            self.v_active[-1, :] = False

    def apply_pressure(self, p: torch.Tensor) -> torch.Tensor:
        """Force p=0 on all open / atmospheric boundary cells."""
        return apply_open_pressure(p, self.open_pressure)

    def apply_velocity_bc(self, u: torch.Tensor, v: torch.Tensor):
        """Apply velocity boundary conditions in-place.

        * Bottom wall: v[0,:]=0 (no flux through wall).
        * Top atmosphere: zero-gradient (free) for v[-1,:].
        * Side outlet: u[:,-1] extrapolated (zero gradient).
        * Side inlet: u[:,0] prescribed by caller (not touched here).
        """
        if self.bottom_wall:
            v[0, :] = 0.0
        if self.top_atmosphere:
            # Free outflow: zero gradient (copy from interior)
            v[-1, :] = v[-2, :]
        if self.side_outlet:
            u[:, -1] = u[:, -2]
        return u, v

    def vof_boundary_flux(
        self,
        alpha: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> float:
        """Net liquid volume lost through open boundaries in one step."""
        return outflow_vof_flux(alpha, u, v, self.dx, self.dy, dt)
