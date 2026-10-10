"""Structured 3-D geometry used by the external-cylinder projection solver."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CartesianCylinderMesh3D:
    """Uniform local z-slab of a 3-D Cartesian mesh with an extruded cylinder.

    ``nz`` always denotes the global spanwise cell count.  ``z_start`` and
    ``z_stop`` identify the locally owned half-open slab, so mesh geometry stays
    independent of the distribution implementation while all field arrays have
    only local z extent.
    """

    nx: int
    ny: int
    nz: int
    z_start: int
    z_stop: int
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
    def build(cls, config, device: torch.device, z_start: int = 0,
              z_stop: int | None = None) -> "CartesianCylinderMesh3D":
        """Build a local z slab while preserving the global physical coordinates."""
        if z_stop is None:
            z_stop = config.nz
        if not (0 <= z_start < z_stop <= config.nz):
            raise ValueError("local z slab must be a nonempty interval inside the global span")
        opts = {"dtype": torch.float64, "device": device}
        dx, dy, dz = config.length / config.nx, config.height / config.ny, config.span / config.nz
        x = (torch.arange(config.nx, **opts) + 0.5) * dx
        y = (torch.arange(config.ny, **opts) + 0.5) * dy
        z = (torch.arange(z_start, z_stop, **opts) + 0.5) * dz
        radial = ((x[None, :] - config.cylinder_x).square()
                  + (y[:, None] - config.cylinder_y).square())
        fluid_2d = radial > config.cylinder_radius ** 2
        fluid = fluid_2d.expand(z_stop - z_start, -1, -1).clone()
        if not bool((~fluid).any()):
            raise ValueError("cylinder is not represented on this 3-D grid; refine the mesh")
        if not bool(fluid[:, :, 0].all()) or not bool(fluid[:, :, -1].all()):
            raise ValueError("cylinder must not intersect the inlet or outlet plane")
        return cls(config.nx, config.ny, config.nz, z_start, z_stop, config.length,
                   config.height, config.span, config.cylinder_x, config.cylinder_y,
                   config.cylinder_radius, device, x, y, z, fluid)

    @property
    def local_nz(self) -> int:
        """Number of z planes stored on this rank."""
        return self.z_stop - self.z_start

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
        """Broadcast local physical cell-centre coordinates in z, y, x order."""
        z, y, x = torch.meshgrid(self.z, self.y, self.x, indexing="ij")
        return x, y, z
