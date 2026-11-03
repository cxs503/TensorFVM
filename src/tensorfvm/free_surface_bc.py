"""Free-surface / open-boundary boundary conditions for two-phase VOF.

This module provides the boundary-condition handlers needed when a
free surface (air-water interface) can cross domain boundaries:

* :class:`FreeSurfaceBC` — applies an *atmospheric-pressure top* (Dirichlet
  p=0 gauge) and *pressure-outlet* side walls, and advects the VOF marker
  through the top boundary so that fluid leaving through the surface does
  not create spurious mass sources.

* :func:`apply_open_pressure` — sets the gauge pressure to zero on all
  open/free-surface faces, mimicking a pressure outlet.

* :func:`outflow_vof_flux` — gross liquid volume leaving the domain through
  the open boundaries (the *outflow* side of the ledger).

* :func:`net_vof_flux` — signed net liquid volume leaving the domain
  (outflow − inflow), which is the quantity that must balance the change of
  the total liquid volume ``d(sum(alpha) * dx * dy)/dt``.

The design is intentionally mesh-agnostic within the Cartesian staggered
framework of :mod:`tensorfvm.solver`: the BC flags are boolean masks aligned
with the MAC face arrays ``u`` (east, shape (ny, nx+1)) and ``v`` (north,
shape (ny+1, nx)).

Boundary-flux convention
------------------------
:class:`~tensorfvm.two_phase_vof.VOFTransport` reconstructs face values from
the adjacent cell values and, at the domain boundary, uses a zero-gradient
ghost cell.  The boundary face value is therefore *exactly* the adjacent
interior cell value, so the boundary liquid flux is ``dt * F_face * alpha_face``
with ``F_face`` the volumetric face flux.  Summing the flux-form update over
all cells telescopes, which gives the discrete ledger identity

    (sum(alpha_new) - sum(alpha_old)) * dx * dy  =  -dt * net_vof_flux(...)

to round-off, for **any** velocity field and for **any** VOF face-reconstruction
scheme (:func:`net_vof_flux` reproduces the donor-cell face value, which is the
value used at the boundary by ``scheme="upwind"``).  Both helpers clamp the
in/out split *per face*; clamping the column sum instead silently cancels
co-located inflow against outflow and breaks the ledger.
"""

from __future__ import annotations

import torch

__all__ = [
    "FreeSurfaceBC",
    "apply_open_pressure",
    "outflow_vof_flux",
    "net_vof_flux",
]


def apply_open_pressure(
    p: torch.Tensor, open_mask: torch.Tensor
) -> torch.Tensor:
    """Set gauge pressure to zero on open/free-surface cells."""
    return torch.where(open_mask, torch.zeros_like(p), p)


def _check_mac_shapes(alpha: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
    """Validate the staggered MAC shape contract and return ``(ny, nx)``."""
    if alpha.dim() != 2:
        raise ValueError(f"alpha must be a 2-D cell field, got shape {tuple(alpha.shape)}")
    ny, nx = alpha.shape
    if tuple(u.shape) != (ny, nx + 1):
        raise ValueError(
            f"u must be the east-face field of shape {(ny, nx + 1)}, got {tuple(u.shape)}"
        )
    if tuple(v.shape) != (ny + 1, nx):
        raise ValueError(
            f"v must be the north-face field of shape {(ny + 1, nx)}, got {tuple(v.shape)}"
        )
    return ny, nx


def _boundary_liquid_flux(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    dt: float,
):
    """Per-face boundary liquid volume fluxes.

    Returns ``(gross_out, net_out, terms)`` where ``gross_out`` is the liquid
    volume leaving, ``net_out`` is ``outflow - inflow``, and ``terms`` maps each
    boundary to ``(outflow, inflow)`` in m^3 for one step.  All four boundary
    faces use the donor-cell (upwind) face value, which equals the adjacent
    interior cell value because the transport operator closes the boundary with
    a zero-gradient ghost cell.
    """
    ny, nx = _check_mac_shapes(alpha, u, v)
    if not torch.isfinite(alpha).all():
        raise ValueError("alpha must be finite")
    if not torch.isfinite(u).all() or not torch.isfinite(v).all():
        raise ValueError("face velocities must be finite")
    for name, value in (("dx", dx), ("dy", dy), ("dt", dt)):
        if not float(value) >= 0.0 or not (float(value) < float("inf")):
            raise ValueError(f"{name} must be finite and non-negative")

    # Liquid volume flux through each boundary face (m^3/s, upwind alpha),
    # signed positive *out of the domain*: east/north faces leave with positive
    # velocity, west/south faces leave with negative velocity.
    leave_east = u[:, -1] * dy * alpha[:, -1]
    leave_west = -u[:, 0] * dy * alpha[:, 0]
    leave_north = v[-1, :] * dx * alpha[-1, :]
    leave_south = -v[0, :] * dx * alpha[0, :]

    zero = torch.zeros((), dtype=alpha.dtype, device=alpha.device)

    def split(flux):  # per-face in/out split; never across a column
        return torch.clamp(flux, min=0.0), torch.clamp(-flux, min=0.0)

    terms = {
        "east": split(leave_east),
        "west": split(leave_west),
        "north": split(leave_north),
        "south": split(leave_south),
    }
    gross = sum((out.sum() for out, _ in terms.values()), zero)
    net = sum((out.sum() - inc.sum() for out, inc in terms.values()), zero)
    return dt * gross, dt * net, {
        key: (float(o.sum()), float(i.sum())) for key, (o, i) in terms.items()
    }


def outflow_vof_flux(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    dt: float,
) -> float:
    """Gross liquid volume leaving the domain through all four boundaries.

    Positive means liquid leaves the domain.  The in/out split is applied
    *per boundary face* and every face uses its own upwind volume fraction, so
    an inflow segment next to an outflow segment cannot cancel outflow.

    Returns the outflow only; use :func:`net_vof_flux` for the signed balance
    that closes the global liquid-volume ledger.
    """
    gross, _net, _terms = _boundary_liquid_flux(alpha, u, v, dx, dy, dt)
    return float(gross.item())


def net_vof_flux(
    alpha: torch.Tensor,
    u: torch.Tensor,
    v: torch.Tensor,
    dx: float,
    dy: float,
    dt: float,
) -> float:
    """Signed net liquid volume leaving the domain through all four boundaries.

    ``net > 0`` means the domain loses liquid, so a consistent transport step
    must satisfy ``(sum(alpha_new) - sum(alpha_ref)) * dx * dy == -net``.
    """
    _gross, net, _terms = _boundary_liquid_flux(alpha, u, v, dx, dy, dt)
    return float(net.item())


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
        """Gross liquid volume lost through open boundaries in one step."""
        return outflow_vof_flux(alpha, u, v, self.dx, self.dy, dt)

    def vof_net_flux(
        self,
        alpha: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> float:
        """Signed net liquid volume lost through open boundaries in one step."""
        return net_vof_flux(alpha, u, v, self.dx, self.dy, dt)

    def ledger_residual(
        self,
        alpha_ref: torch.Tensor,
        alpha_new: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> float:
        """Global liquid-volume ledger residual for one transport step.

        Returns ``(sum(alpha_new) - sum(alpha_ref)) * dx * dy + net_vof_flux``,
        which is zero (to round-off) whenever the advection step and the
        boundary-flux witness are consistent, and equals the unaccounted liquid
        volume otherwise.
        """
        if tuple(alpha_ref.shape) != (self.ny, self.nx):
            raise ValueError(
                f"alpha_ref must have shape {(self.ny, self.nx)}, got {tuple(alpha_ref.shape)}"
            )
        if tuple(alpha_new.shape) != (self.ny, self.nx):
            raise ValueError(
                f"alpha_new must have shape {(self.ny, self.nx)}, got {tuple(alpha_new.shape)}"
            )
        delta_volume = float((alpha_new.sum() - alpha_ref.sum()).item()) * self.dx * self.dy
        return delta_volume + self.vof_net_flux(alpha_ref, u, v, dt)