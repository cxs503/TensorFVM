"""Distributed-ready extruded O-grid geometry for a three-dimensional cylinder.

This module deliberately owns geometry and topology only.  The existing
``Cylinder3DSolver`` uses Cartesian stencils and therefore must not silently
consume a curved O-grid.  Keeping the mesh independent makes its metric and
connectivity contracts testable before a conservative curvilinear 3-D finite-
volume kernel is introduced.
"""

from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch

from .body_fitted import BodyFittedMesh


@dataclass(frozen=True)
class BodyFittedCylinderMesh3D:
    """One local z slab of an extruded, body-fitted cylinder O-grid.

    ``nx`` is circumferential, ``ny`` is radial, and ``nz`` is the global
    spanwise cell count.  The two-dimensional O-grid has a polygonal cylinder
    surface and a rectangular outer boundary; it is extruded linearly in z.
    Vertices on the theta seam are intentionally duplicated so each local cell
    has ordinary hexahedral connectivity, while cells remain periodic across
    the seam in the two-dimensional face topology.
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
    vertices: torch.Tensor
    centers: torch.Tensor
    cell_volumes: torch.Tensor
    connectivity: torch.Tensor
    fluid: torch.Tensor
    lateral_face_vertices: torch.Tensor
    lateral_face_centers: torch.Tensor
    lateral_face_area_vectors: torch.Tensor
    lateral_owner: torch.Tensor
    lateral_neighbor: torch.Tensor
    lateral_boundary: torch.Tensor
    lateral_labels: tuple[str, ...]
    spanwise_face_centers: torch.Tensor
    spanwise_face_areas: torch.Tensor
    cylinder_face_vertices: torch.Tensor
    cylinder_face_centers: torch.Tensor
    cylinder_face_area_vectors: torch.Tensor
    cross_section_area: torch.Tensor

    @classmethod
    def build(cls, config, device: torch.device, z_start: int = 0,
              z_stop: int | None = None) -> "BodyFittedCylinderMesh3D":
        """Build the local part of a periodic extruded O-grid.

        The supplied config follows :class:`Cylinder3DConfig`; only body-fitted
        geometry fields are consumed.  No distributed collectives occur here,
        so callers can create it safely after choosing a z-slab partition.
        """
        if z_stop is None:
            z_stop = config.nz
        if not (0 <= z_start < z_stop <= config.nz):
            raise ValueError("local z slab must be a nonempty interval inside the global span")
        cross_config = SimpleNamespace(
            nx=config.nx,
            ny=config.ny,
            length=config.length,
            height=config.height,
            cylinder_x=config.cylinder_x,
            cylinder_y=config.cylinder_y,
            cylinder_radius=config.cylinder_radius,
            mesh_type="body-fitted",
            body_fitted_stretching=config.body_fitted_stretching,
            outer_boundary="far-field",
            device=device,
        )
        cross = BodyFittedMesh(cross_config)
        opts = {"dtype": torch.float64, "device": device}
        local_nz = z_stop - z_start
        dz = config.span / config.nz
        z_nodes = torch.arange(z_start, z_stop + 1, **opts) * dz
        z_centers = (torch.arange(z_start, z_stop, **opts) + 0.5) * dz

        vertices = torch.empty((local_nz + 1, config.ny + 1, config.nx + 1, 3), **opts)
        vertices[..., :2] = cross.vertices[None]
        vertices[..., 2] = z_nodes[:, None, None]
        centers = torch.empty((local_nz, config.ny, config.nx, 3), **opts)
        centers[..., :2] = cross.centers[None]
        centers[..., 2] = z_centers[:, None, None]
        cell_volumes = cross.volumes[None].expand(local_nz, -1, -1).clone() * dz
        if not bool(torch.all(cell_volumes > 0)):
            raise ValueError("invalid nonpositive hexahedral cell volume")
        fluid = torch.ones_like(cell_volumes, dtype=torch.bool)

        connectivity = cls._connectivity(config.nx, config.ny, local_nz, device)
        lateral = cls._lateral_faces(cross, z_nodes, dz, config.nx, config.ny, device)
        spanwise_face_centers = torch.empty((local_nz + 1, config.ny, config.nx, 3), **opts)
        spanwise_face_centers[..., :2] = cross.centers[None]
        spanwise_face_centers[..., 2] = z_nodes[:, None, None]
        spanwise_face_areas = cross.volumes[None].expand(local_nz + 1, -1, -1).clone()

        cylinder_mask = cross.masks["cylinder"]
        cylinder_vertices = lateral[0][:, cylinder_mask]
        cylinder_centers = lateral[1][:, cylinder_mask]
        cylinder_area_vectors = lateral[2][:, cylinder_mask]
        return cls(
            config.nx, config.ny, config.nz, z_start, z_stop, config.length,
            config.height, config.span, config.cylinder_x, config.cylinder_y,
            config.cylinder_radius, device, vertices, centers, cell_volumes,
            connectivity, fluid, lateral[0], lateral[1], lateral[2], lateral[3],
            lateral[4], lateral[5], tuple(cross.boundary_labels),
            spanwise_face_centers, spanwise_face_areas, cylinder_vertices,
            cylinder_centers, cylinder_area_vectors, cross.volumes.sum(),
        )

    @staticmethod
    def _connectivity(nx: int, ny: int, local_nz: int,
                      device: torch.device) -> torch.Tensor:
        """Return local hexahedral connectivity with standard lower/upper rings."""
        stride_theta = nx + 1
        stride_z = (ny + 1) * stride_theta
        connectivity = torch.empty((local_nz, ny, nx, 8), dtype=torch.long, device=device)
        for k in range(local_nz):
            for j in range(ny):
                base = k * stride_z + j * stride_theta + torch.arange(nx, device=device)
                connectivity[k, j, :, 0] = base
                connectivity[k, j, :, 1] = base + 1
                connectivity[k, j, :, 2] = base + stride_theta + 1
                connectivity[k, j, :, 3] = base + stride_theta
                connectivity[k, j, :, 4] = base + stride_z
                connectivity[k, j, :, 5] = base + stride_z + 1
                connectivity[k, j, :, 6] = base + stride_z + stride_theta + 1
                connectivity[k, j, :, 7] = base + stride_z + stride_theta
        return connectivity

    @staticmethod
    def _lateral_faces(cross: BodyFittedMesh, z_nodes: torch.Tensor, dz: float,
                       nx: int, ny: int, device: torch.device):
        """Extrude every 2-D face into one oriented quad per local z layer."""
        local_nz = len(z_nodes) - 1
        face_count = len(cross.owner)
        opts = {"dtype": torch.float64, "device": device}
        endpoints = cross.face_vertices
        lower = torch.empty((local_nz, face_count, 2, 3), **opts)
        upper = torch.empty_like(lower)
        lower[..., :2] = endpoints[None]
        upper[..., :2] = endpoints[None]
        lower[..., 2] = z_nodes[:-1, None, None]
        upper[..., 2] = z_nodes[1:, None, None]
        vertices = torch.stack((lower[:, :, 0], lower[:, :, 1],
                                upper[:, :, 1], upper[:, :, 0]), dim=2)
        centers = vertices.mean(dim=2)
        area_vectors = torch.zeros((local_nz, face_count, 3), **opts)
        area_vectors[..., :2] = cross.face_area_vectors[None] * dz
        cell_count = nx * ny
        offsets = torch.arange(local_nz, device=device)[:, None] * cell_count
        owner = cross.owner[None] + offsets
        neighbor = torch.full((local_nz, face_count), -1, dtype=torch.long, device=device)
        interior = cross.neighbor >= 0
        neighbor[:, interior] = cross.neighbor[None, interior] + offsets
        boundary = cross.boundary[None].expand(local_nz, -1).clone()
        return vertices, centers, area_vectors, owner, neighbor, boundary

    @property
    def local_nz(self) -> int:
        """Number of z cell layers owned by this mesh instance."""
        return self.z_stop - self.z_start

    @property
    def dz(self) -> float:
        return self.span / self.nz

    @property
    def local_span(self) -> float:
        return self.local_nz * self.dz

    @property
    def node_count(self) -> int:
        return (self.local_nz + 1) * (self.ny + 1) * (self.nx + 1)

    @property
    def cell_count(self) -> int:
        return self.local_nz * self.ny * self.nx

    @property
    def cylinder_surface_area(self) -> torch.Tensor:
        """Polygonal cylinder-wall area for this local z slab."""
        return torch.linalg.vector_norm(self.cylinder_face_area_vectors, dim=-1).sum()

    @property
    def analytic_cylinder_surface_area(self) -> float:
        """Smooth-cylinder area over the local span, for mesh-quality reporting."""
        return 2 * math.pi * self.cylinder_radius * self.local_span

    def centers_xyz(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return broadcast local cell-centre coordinates in z, radial, theta order."""
        return self.centers[..., 0], self.centers[..., 1], self.centers[..., 2]
