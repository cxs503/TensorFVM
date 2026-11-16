"""Dimension-agnostic finite-volume mesh contract (``MeshData``).

Seeded from ``tensorfvm.mesh_api.Mesh2D`` (the repository's only ``Protocol``
mesh interface, 2-D only) and extended to 3-D polyhedral cells.  The
owner-oriented face convention is inherited unchanged: ``neighbor == -1``
marks a boundary face and the area vector points outward from its owner.

``MeshData`` is a plain dataclass (structural, like ``Mesh2D``): existing mesh
classes do not inherit it, they are *adapted* to it by
:mod:`tensorfvm.core.mesh.adapters`.  Geometry stays in NumPy because the 3-D
mesh classes (``native_mesh3d``, ``openfoam_polymesh``) are NumPy-based; the
2-D torch-based ``Mesh2D`` is converted at the adapter boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

__all__ = ["MeshData", "validate_mesh_data"]


@dataclass(frozen=True)
class MeshData:
    """2-D/3-D finite-volume geometry/topology contract.

    Fields
    ------
    dim
        Spatial dimension, 2 or 3.
    vertices
        ``(nvert, dim)`` float vertex coordinates.
    cell_centers
        ``(ncell, dim)`` float cell-centre coordinates.
    cell_volumes
        ``(ncell,)`` float, strictly positive.
    owner
        ``(nface,)`` int, owner cell of each face.
    neighbor
        ``(nface,)`` int; ``-1`` on boundary faces, else the neighbour cell.
    face_centers
        ``(nface, dim)`` float.
    area_vectors
        ``(nface, dim)`` float, owner-oriented (outward on boundary faces).
    patch_labels
        ``(nface,)`` string labels; ``'internal'`` on interior faces, one
        physical patch name per boundary face.
    masks
        Mapping of boundary patch name -> boolean face vector.
    field_shape
        Tuple of positive ints whose product equals ``ncell``.
    name
        Free-form mesh identifier (grid name, case directory, ...).
    """

    dim: int
    vertices: np.ndarray
    cell_centers: np.ndarray
    cell_volumes: np.ndarray
    owner: np.ndarray
    neighbor: np.ndarray
    face_centers: np.ndarray
    area_vectors: np.ndarray
    patch_labels: np.ndarray
    masks: Mapping[str, np.ndarray] = field(default_factory=dict)
    field_shape: tuple[int, ...] = ()
    name: str = ""

    # -- derived conveniences (mirroring Mesh2D / PolyMesh) ---------------- #
    @property
    def n_cells(self) -> int:
        return int(self.cell_volumes.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.owner.shape[0])

    @property
    def interior(self) -> np.ndarray:
        return self.neighbor >= 0

    @property
    def boundary(self) -> np.ndarray:
        return self.neighbor < 0

    @property
    def face_lengths(self) -> np.ndarray:
        """Face measure: edge length in 2-D, area in 3-D."""
        return np.linalg.norm(self.area_vectors, axis=1)


def validate_mesh_data(
    mesh: MeshData,
    *,
    required_masks: tuple[str, ...] = (),
    check_finite_geometry: bool = True,
    check_closure: bool = False,
    closure_tolerance: float = 1e-10,
) -> MeshData:
    """Validate a mesh against the solver-facing finite-volume contract.

    Raises ``TypeError`` for missing or wrongly typed fields and ``ValueError``
    for inconsistent topology, nonpositive cells/faces, malformed masks, or
    non-finite geometry.  Returns ``mesh`` to support validation at a factory
    boundary.

    ``check_closure`` additionally verifies the discrete divergence identity
    (the sum of owner-oriented area vectors around each cell is zero) within
    ``closure_tolerance`` of the cell's total face measure -- the same
    identity ``NativeMesh3D.audit`` / ``PolyMesh.audit`` report.
    """
    if not isinstance(mesh, MeshData):
        raise TypeError(f"expected MeshData, got {type(mesh).__name__}")
    dim = mesh.dim
    if dim not in (2, 3):
        raise ValueError("MeshData dim must be 2 or 3")

    for name in ("vertices", "cell_centers", "cell_volumes", "face_centers",
                 "area_vectors"):
        value = getattr(mesh, name)
        if not isinstance(value, np.ndarray):
            raise TypeError(f"MeshData field {name!r} must be a numpy array")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"MeshData field {name!r} must be floating point")
    for name in ("owner", "neighbor"):
        value = getattr(mesh, name)
        if not isinstance(value, np.ndarray) or value.dtype != np.int64:
            raise TypeError(f"MeshData field {name!r} must be an int64 array")
    if not isinstance(mesh.patch_labels, np.ndarray):
        raise TypeError("MeshData field 'patch_labels' must be a numpy array")
    if not isinstance(mesh.masks, Mapping):
        raise TypeError("MeshData field 'masks' must be a mapping of face masks")

    ncell = int(mesh.cell_volumes.shape[0])
    if ncell < 1:
        raise ValueError("MeshData must contain at least one cell")
    if mesh.cell_centers.shape != (ncell, dim):
        raise ValueError("MeshData cell_centers must have shape (ncell, dim)")
    if mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != dim:
        raise ValueError("MeshData vertices must have shape (nvert, dim)")
    if mesh.field_shape and int(np.prod(mesh.field_shape)) != ncell:
        raise ValueError("MeshData field_shape product must equal the cell count")

    nface = int(mesh.owner.shape[0])
    if nface < 1:
        raise ValueError("MeshData must contain at least one face")
    if mesh.owner.shape != (nface,) or mesh.neighbor.shape != (nface,):
        raise ValueError("MeshData owner and neighbor must be one-dimensional and equal-sized")
    if mesh.face_centers.shape != (nface, dim) or mesh.area_vectors.shape != (nface, dim):
        raise ValueError("MeshData face arrays must have shape (nface, dim)")
    if mesh.patch_labels.shape != (nface,):
        raise ValueError("MeshData patch_labels must have shape (nface,)")

    if (mesh.owner < 0).any() or (mesh.owner >= ncell).any():
        raise ValueError("MeshData owner indices are outside the cell range")
    if (mesh.neighbor < -1).any() or (mesh.neighbor >= ncell).any():
        raise ValueError("MeshData neighbor indices must be -1 or inside the cell range")
    interior = mesh.interior
    if not np.array_equal(interior, mesh.neighbor >= 0):
        raise ValueError("MeshData interior/boundary flags disagree with neighbor indices")
    labels = mesh.patch_labels
    if not np.array_equal(labels[interior] == "internal", np.ones(int(interior.sum()), dtype=bool)):
        raise ValueError("MeshData interior faces must carry the 'internal' patch label")
    if (labels[~interior] == "internal").any():
        raise ValueError("MeshData boundary faces must carry a physical patch label")

    if not np.isfinite(mesh.cell_volumes).all() or not (mesh.cell_volumes > 0).all():
        raise ValueError("MeshData cell volumes must be finite and strictly positive")
    lengths = mesh.face_lengths
    if not np.isfinite(lengths).all() or not (lengths > 0).all():
        raise ValueError("MeshData face measures must be finite and strictly positive")

    for name, mask in mesh.masks.items():
        if not isinstance(name, str):
            raise TypeError("MeshData masks must map string names to boolean arrays")
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != (nface,):
            raise ValueError(f"MeshData mask {name!r} must be a boolean face vector")
        if (mask & interior).any():
            raise ValueError(f"MeshData mask {name!r} may only mark boundary faces")
    missing = set(required_masks) - set(mesh.masks)
    if missing:
        raise ValueError("MeshData is missing required boundary masks: "
                         + ", ".join(sorted(missing)))

    if check_finite_geometry:
        for name, tensor in (
            ("vertices", mesh.vertices), ("cell_centers", mesh.cell_centers),
            ("face_centers", mesh.face_centers), ("area_vectors", mesh.area_vectors),
        ):
            if not np.isfinite(tensor).all():
                raise ValueError(f"MeshData {name} must contain only finite values")

    if check_closure:
        closure = np.zeros((ncell, dim))
        np.add.at(closure, mesh.owner, mesh.area_vectors)
        np.add.at(closure, mesh.neighbor[interior], -mesh.area_vectors[interior])
        scale = np.zeros(ncell)
        face_measure = np.linalg.norm(mesh.area_vectors, axis=1)
        np.add.at(scale, mesh.owner, face_measure)
        np.add.at(scale, mesh.neighbor[interior], face_measure[interior])
        relative = np.linalg.norm(closure, axis=1) / np.maximum(scale, np.finfo(float).tiny)
        if relative.max() > closure_tolerance:
            raise ValueError(
                f"MeshData area-vector closure violated: max relative closure "
                f"{relative.max():.3e} > {closure_tolerance:.1e}")
    return mesh
