"""Structured 3-D geometry used by the external-cylinder projection solver."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CartesianCylinderMesh3D:
    """Uniform 3-D Cartesian mesh with an extruded circular solid mask.

    It is intentionally separate from the 2-D O-grid: this gives the first 3-D
    solver an unambiguous structured topology and supplies a canonical place to
    add cut-cell or body-fitted 3-D mesh implementations later.
    """

    nx: int
    ny: int
    nz: int
    length: float
    height: float
    span: float
    cylinder_x: float
    cylinder_y: float
    cylinder_radius: float
    device: torch.device
    x: torch.Tensor
    y: torch.Tensor
    z: torch.Tensor
    fluid: torch.Tensor

    @classmethod
    def build(cls, config, device: torch.device) -> "CartesianCylinderMesh3D":
        opts = {"dtype": torch.float64, "device": device}
        dx, dy, dz = config.length / config.nx, config.height / config.ny, config.span / config.nz
        x = (torch.arange(config.nx, **opts) + 0.5) * dx
        y = (torch.arange(config.ny, **opts) + 0.5) * dy
        z = (torch.arange(config.nz, **opts) + 0.5) * dz
        radial = ((x[None, :] - config.cylinder_x).square()
                  + (y[:, None] - config.cylinder_y).square())
        fluid_2d = radial > config.cylinder_radius ** 2
        fluid = fluid_2d.expand(config.nz, -1, -1).clone()
        if not bool((~fluid).any()):
            raise ValueError("cylinder is not represented on this 3-D grid; refine the mesh")
        if not bool(fluid[:, :, 0].all()) or not bool(fluid[:, :, -1].all()):
            raise ValueError("cylinder must not intersect the inlet or outlet plane")
        return cls(config.nx, config.ny, config.nz, config.length, config.height,
                   config.span, config.cylinder_x, config.cylinder_y,
                   config.cylinder_radius, device, x, y, z, fluid)

    @property
    def dx(self) -> float:
        return self.length / self.nx

    @property
    def dy(self) -> float:
        return self.height / self.ny

    @property
    def dz(self) -> float:
        return self.span / self.nz

    @property
    def cell_volume(self) -> float:
        return self.dx * self.dy * self.dz

    def centers(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Broadcast physical cell-centre coordinates in z, y, x order."""
        z, y, x = torch.meshgrid(self.z, self.y, self.x, indexing="ij")
        return x, y, z
