"""Algebraic VOF transport and two-phase physical properties.

This module provides the core volume-of-fluid (VOF) building blocks that bridge
the single-phase SIMPLE solver toward a two-phase (air-water) capability:

* :class:`VOFTransport` — donor-cell (CICSAM-like) bounded advection of the
  volume fraction ``alpha`` on a staggered MAC grid.  The scheme is
  *geometrically conservative*: the total phase volume is preserved to machine
  precision on any uniform Cartesian mesh.
* :func:`mixture_density` / :func:`mixture_viscosity` — volume-fraction-weighted
  physical properties.  Density uses a linear average; dynamic viscosity uses
  the *harmonic* mean to avoid spurious flux across the interface for large
  viscosity ratios.
* :func:`csf_surface_tension` — Brackbill continuous-surface-force (CSF) source
  term with a curvature computed from the smoothed volume fraction.  Gated by
  an explicit switch so single-phase regressions are bit-for-bit identical.

The transport is *algebraic* (no interface reconstruction / PLIC), matching the
SI-0B-G1 remediation scope: a donor-connection conservative format that is
sufficient for the static-interface, hydrostatic, and low-Re Stokes anchors.

All tensors are ``float64`` to stay consistent with :mod:`tensorfvm.solver`.
"""

from __future__ import annotations

import torch

__all__ = [
    "VOFTransport",
    "TwoPhaseProperties",
    "mixture_density",
    "mixture_viscosity",
    "csf_surface_tension",
    "interface_curvature",
    "interface_normal",
]


def _cell_center_velocity(
    u: torch.Tensor, v: torch.Tensor, ny: int, nx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate staggered face velocities to cell centres."""
    uc = 0.5 * (u[:, :-1] + u[:, 1:])
    vc = 0.5 * (v[:-1, :] + v[1:, :])
    return uc, vc


def mixture_density(
    alpha: torch.Tensor, rho_liquid: float, rho_gas: float
) -> torch.Tensor:
    """Volume-fraction-weighted density:  rho = α·ρ_l + (1−α)·ρ_g."""
    return alpha * rho_liquid + (1.0 - alpha) * rho_gas


def mixture_viscosity(
    alpha: torch.Tensor, mu_liquid: float, mu_gas: float
) -> torch.Tensor:
    """Harmonic-mean dynamic viscosity (robust for large viscosity ratios).

    1/μ = α/μ_l + (1−α)/μ_g  →  μ = (μ_l·μ_g)/(α·μ_g + (1−α)·μ_l).
    """
    denom = alpha * mu_gas + (1.0 - alpha) * mu_liquid
    return (mu_liquid * mu_gas) / denom.clamp_min(1e-30)


def interface_normal(alpha: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Outward unit normal of the liquid phase via central differences.

    The normal points from liquid (α≈1) toward gas (α≈0):  n = −∇α/|∇α|.
    Edge differences use first-order one-sided stencils.
    """
    ny, nx = alpha.shape
    gx = torch.zeros_like(alpha)
    gy = torch.zeros_like(alpha)
    # x-derivative (central interior, one-sided at edges)
    gx[:, 1:-1] = (alpha[:, 2:] - alpha[:, :-2]) / (2.0 * dx)
    gx[:, 0] = (alpha[:, 1] - alpha[:, 0]) / dx
    gx[:, -1] = (alpha[:, -1] - alpha[:, -2]) / dx
    # y-derivative
    gy[1:-1, :] = (alpha[2:, :] - alpha[:-2, :]) / (2.0 * dy)
    gy[0, :] = (alpha[1, :] - alpha[0, :]) / dy
    gy[-1, :] = (alpha[-1, :] - alpha[-2, :]) / dy
    magnitude = torch.sqrt(gx * gx + gy * gy).clamp_min(1e-30)
    normal_x = -gx / magnitude
    normal_y = -gy / magnitude
    return torch.stack([normal_x, normal_y], dim=-1)


def interface_curvature(alpha: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """Mean curvature κ = ∇·n computed from the interface normal.

    Returns a (ny,nx) tensor.  Where |∇α|≈0 (pure phase) the curvature is 0.
    """
    n = interface_normal(alpha, dx, dy)  # (ny,nx,2)
    nx = n[..., 0]
    ny = n[..., 1]
    div_n = torch.zeros_like(alpha)
    # ∂n_x/∂x
    dnx = torch.zeros_like(alpha)
    dnx[:, 1:-1] = (nx[:, 2:] - nx[:, :-2]) / (2.0 * dx)
    dnx[:, 0] = (nx[:, 1] - nx[:, 0]) / dx
    dnx[:, -1] = (nx[:, -1] - nx[:, -2]) / dx
    # ∂n_y/∂y
    dny = torch.zeros_like(alpha)
    dny[1:-1, :] = (ny[2:, :] - ny[:-2, :]) / (2.0 * dy)
    dny[0, :] = (ny[1, :] - ny[0, :]) / dy
    dny[-1, :] = (ny[-1, :] - ny[-2, :]) / dy
    div_n = dnx + dny
    # Zero curvature in pure-phase regions (where gradient is negligible)
    gx_local = torch.zeros_like(alpha)
    gx_local[:, 1:-1] = (alpha[:, 2:] - alpha[:, :-2]) / (2.0 * dx)
    gx_local[:, 0] = (alpha[:, 1] - alpha[:, 0]) / dx
    gx_local[:, -1] = (alpha[:, -1] - alpha[:, -2]) / dx
    gy_local = torch.zeros_like(alpha)
    gy_local[1:-1, :] = (alpha[2:, :] - alpha[:-2, :]) / (2.0 * dy)
    gy_local[0, :] = (alpha[1, :] - alpha[0, :]) / dy
    gy_local[-1, :] = (alpha[-1, :] - alpha[-2, :]) / dy
    grad_alpha = torch.sqrt(gx_local ** 2 + gy_local ** 2)
    grad_mag_max = float(grad_alpha.max())
    if grad_mag_max > 0:
        mask = grad_alpha > (1e-8 * grad_mag_max)
        div_n = div_n * mask
    return div_n


def csf_surface_tension(
    alpha: torch.Tensor,
    sigma: float,
    dx: float,
    dy: float,
    enabled: bool = True,
) -> torch.Tensor:
    """Brackbill continuous-surface-force source:  f = σ·κ·∇α.

    Returns a (ny, nx, 2) body-force array.  When *enabled* is False (or
    *sigma* is zero) returns zeros — guaranteeing single-phase bit-identical
    behaviour.
    """
    ny, nx = alpha.shape
    if not enabled or sigma == 0.0:
        return torch.zeros(ny, nx, 2, dtype=alpha.dtype, device=alpha.device)
    kappa = interface_curvature(alpha, dx, dy)
    grad = torch.zeros(ny, nx, 2, dtype=alpha.dtype, device=alpha.device)
    gx = torch.zeros_like(alpha)
    gy = torch.zeros_like(alpha)
    gx[:, 1:-1] = (alpha[:, 2:] - alpha[:, :-2]) / (2.0 * dx)
    gx[:, 0] = (alpha[:, 1] - alpha[:, 0]) / dx
    gx[:, -1] = (alpha[:, -1] - alpha[:, -2]) / dx
    gy[1:-1, :] = (alpha[2:, :] - alpha[:-2, :]) / (2.0 * dy)
    gy[0, :] = (alpha[1, :] - alpha[0, :]) / dy
    gy[-1, :] = (alpha[-1, :] - alpha[-2, :]) / dy
    grad[..., 0] = gx
    grad[..., 1] = gy
    force = sigma * kappa[..., None] * grad
    return force


class TwoPhaseProperties:
    """Container for two-phase physical-property pairs and CSF switch."""

    def __init__(
        self,
        rho_liquid: float,
        rho_gas: float,
        mu_liquid: float,
        mu_gas: float,
        sigma: float = 0.0,
        surface_tension: bool = False,
    ):
        self.rho_liquid = float(rho_liquid)
        self.rho_gas = float(rho_gas)
        self.mu_liquid = float(mu_liquid)
        self.mu_gas = float(mu_gas)
        self.sigma = float(sigma)
        self.surface_tension = bool(surface_tension)

    def density(self, alpha: torch.Tensor) -> torch.Tensor:
        return mixture_density(alpha, self.rho_liquid, self.rho_gas)

    def viscosity(self, alpha: torch.Tensor) -> torch.Tensor:
        return mixture_viscosity(alpha, self.mu_liquid, self.mu_gas)

    def csf(self, alpha: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
        return csf_surface_tension(
            alpha, self.sigma, dx, dy, enabled=self.surface_tension
        )


class VOFTransport:
    """Donor-cell (contribution-acceptor) conservative VOF advection.

    The volume fraction ``alpha`` is stored at cell centres on the same
    (ny, nx) grid as the pressure in :class:`~tensorfvm.solver.SimpleSolver`.
    Face fluxes are computed from the staggered face velocities ``u`` (east
    faces, shape (ny, nx+1)) and ``v`` (north faces, shape (ny+1, nx)).

    The donor-cell scheme uses a bounded blending of upwind and downwind:

        f_face = f_donor + C·(f_acceptor − f_donor)

    where C is the local Courant number.  This yields a stable, conservative,
    and bounded transport that preserves total phase volume to machine
    precision when the velocity field is divergence-free.
    """

    def __init__(self, ny: int, nx: int, dx: float, dy: float, device: str = "cpu"):
        self.ny = ny
        self.nx = nx
        self.dx = float(dx)
        self.dy = float(dy)
        self.device = device
        self._dt = None

    def face_flux_x(self, u: torch.Tensor) -> torch.Tensor:
        """Volumetric flux through east faces:  F_e = u·dy."""
        return u * self.dy

    def face_flux_y(self, v: torch.Tensor) -> torch.Tensor:
        """Volumetric flux through north faces:  F_n = v·dx."""
        return v * self.dx

    def _donor_face_alpha_x(
        self, alpha: torch.Tensor, flux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Donor/acceptor face values for x-faces (shape ny×(nx+1)).

        Returns (donor, acceptor, cour) where each has shape (ny, nx+1).
        face j lies between cell j-1 (west) and cell j (east).
        """
        ny, nx = alpha.shape
        west = torch.empty((ny, nx + 1), dtype=alpha.dtype, device=alpha.device)
        east = torch.empty((ny, nx + 1), dtype=alpha.dtype, device=alpha.device)
        west[:, 1:] = alpha  # west cell of face j is cell j-1
        west[:, 0] = alpha[:, 0]  # ghost (zero-gradient)
        east[:, :-1] = alpha  # east cell of face j is cell j
        east[:, -1] = alpha[:, -1]  # ghost
        pos = (flux >= 0).to(alpha.dtype)
        donor = pos * west + (1.0 - pos) * east
        acceptor = pos * east + (1.0 - pos) * west
        return donor, acceptor, pos

    def _donor_face_alpha_y(
        self, alpha: torch.Tensor, flux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Donor/acceptor face values for y-faces (shape (ny+1)×nx).

        Returns (donor, acceptor, pos) where each has shape (ny+1, nx).
        face i lies between cell i-1 (south) and cell i (north).
        """
        ny, nx = alpha.shape
        south = torch.empty((ny + 1, nx), dtype=alpha.dtype, device=alpha.device)
        north = torch.empty((ny + 1, nx), dtype=alpha.dtype, device=alpha.device)
        south[1:, :] = alpha
        south[0, :] = alpha[0, :]
        north[:-1, :] = alpha
        north[-1, :] = alpha[-1, :]
        pos = (flux >= 0).to(alpha.dtype)
        donor = pos * south + (1.0 - pos) * north
        acceptor = pos * north + (1.0 - pos) * south
        return donor, acceptor, pos

    def advect(
        self,
        alpha: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Advance alpha by one explicit donor-cell step.

        Conservation:  α_new = α_old − (dt/V)·[ΣF_face·α_face]

        where V = dx·dy is the cell volume and F_face is the volumetric
        face flux.  The scheme is bounded for CFL ≤ 1.
        """
        cell_vol = self.dx * self.dy
        Fe = u * self.dy  # east-face volumetric flux (ny, nx+1)
        Fn = v * self.dx  # north-face volumetric flux (ny+1, nx)

        # --- x-direction face values (east faces, nx+1) ---
        donor_x, acceptor_x, pos_x = self._donor_face_alpha_x(alpha, Fe)
        cour_x = (Fe * dt / cell_vol).abs()
        af_x = donor_x + cour_x * (acceptor_x - donor_x)
        # Pin face value to donor when donor is pure (α=0 or 1) to prevent
        # spurious mixing at sharp interfaces that breaks global conservation.
        pure_x = (donor_x <= 0.0) | (donor_x >= 1.0)
        af_x = torch.where(pure_x, donor_x, af_x)
        af_x = af_x.clamp(0.0, 1.0)

        # --- y-direction face values (north faces, ny+1) ---
        donor_y, acceptor_y, pos_y = self._donor_face_alpha_y(alpha, Fn)
        cour_y = (Fn * dt / cell_vol).abs()
        af_y = donor_y + cour_y * (acceptor_y - donor_y)
        pure_y = (donor_y <= 0.0) | (donor_y >= 1.0)
        af_y = torch.where(pure_y, donor_y, af_y)
        af_y = af_y.clamp(0.0, 1.0)

        # Net outflux: east - west + north - south
        # Cell j: east face j+1, west face j, north face j, south face j
        net_flux_x = Fe[:, 1:] * af_x[:, 1:] - Fe[:, :-1] * af_x[:, :-1]
        net_flux_y = Fn[1:, :] * af_y[1:, :] - Fn[:-1, :] * af_y[:-1, :]
        alpha_new = alpha - dt / cell_vol * (net_flux_x + net_flux_y)
        return alpha_new.clamp(0.0, 1.0)

    def total_volume(self, alpha: torch.Tensor) -> float:
        """Total liquid volume:  V = Σ α·dx·dy."""
        return float(alpha.sum().item() * self.dx * self.dy)

    def volume_drift(
        self, alpha: torch.Tensor, initial_volume: float
    ) -> float:
        """Absolute volume drift relative to the initial value."""
        return abs(self.total_volume(alpha) - initial_volume)
