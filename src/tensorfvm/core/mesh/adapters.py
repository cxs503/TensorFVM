"""Adapters from the existing mesh classes to the ``MeshData`` contract.

Each adapter is a pure mapping over the source object's public fields -- no
geometry is recomputed except where the source stores it implicitly (the
2-D overset grid keeps faces only as shared polygon edges, and the body-fitted
3-D slab keeps its spanwise faces as centres plus scalar areas).  Nothing in
the source modules is modified.
"""

from __future__ import annotations

import numpy as np

from ...registry import capability
from .contract import MeshData, validate_mesh_data

__all__ = [
    "from_mesh2d",
    "from_native_mesh3d",
    "from_polymesh",
    "from_overset_grid",
    "from_body_fitted3d",
    "adapt_mesh",
    "MESH_ADAPTERS",
]


def _torch_to_numpy(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _masks_from_labels(labels: np.ndarray) -> dict[str, np.ndarray]:
    """One boolean face vector per physical patch label (boundary faces only)."""
    masks: dict[str, np.ndarray] = {}
    for label in np.unique(labels):
        if label == "internal":
            continue
        masks[str(label)] = labels == label
    return masks


@capability('fvm.core.mesh_contract',
            kind='mesh',
            maturity='developing',
            test_path='tests/test_core_mesh.py',
            wired=True,
            tested=True,
            verified=False,
            evidence={
                'grade': 'MECHANICS',
                'note': ('MeshData generalizes mesh_api.Mesh2D to 2-D/3-D; adapters '
                         'delegate to NativeMesh3D / PolyMesh / OversetGrid / '
                         'BodyFittedCylinderMesh3D / Mesh2D public fields only'),
                'docs': 1,
            })
def from_native_mesh3d(mesh) -> MeshData:
    """Adapt ``tensorfvm.native_mesh3d.NativeMesh3D`` (3-D, NumPy)."""
    labels = np.asarray(mesh.patch_labels, dtype=object)
    return MeshData(
        dim=3,
        vertices=np.asarray(mesh.vertices, dtype=float),
        cell_centers=np.asarray(mesh.cell_centers, dtype=float),
        cell_volumes=np.asarray(mesh.cell_volumes, dtype=float),
        owner=np.asarray(mesh.owner, dtype=np.int64),
        neighbor=np.asarray(mesh.neighbor, dtype=np.int64),
        face_centers=np.asarray(mesh.face_centers, dtype=float),
        area_vectors=np.asarray(mesh.area_vectors, dtype=float),
        patch_labels=labels,
        masks=_masks_from_labels(labels),
        field_shape=(len(mesh.cells),),
        name="native_mesh3d",
    )


def from_polymesh(mesh) -> MeshData:
    """Adapt ``tensorfvm.openfoam_polymesh.PolyMesh`` (3-D, NumPy).

    ``PolyMesh.neighbour`` covers internal faces only; the full per-face
    neighbour array is taken from ``neighbour_full()``.
    """
    labels = np.asarray(mesh.patch_labels, dtype=object)
    return MeshData(
        dim=3,
        vertices=np.asarray(mesh.points, dtype=float),
        cell_centers=np.asarray(mesh.cell_centers, dtype=float),
        cell_volumes=np.asarray(mesh.cell_volumes, dtype=float),
        owner=np.asarray(mesh.owner, dtype=np.int64),
        neighbor=np.asarray(mesh.neighbour_full(), dtype=np.int64),
        face_centers=np.asarray(mesh.face_centers, dtype=float),
        area_vectors=np.asarray(mesh.face_area_vectors, dtype=float),
        patch_labels=labels,
        masks=_masks_from_labels(labels),
        field_shape=(mesh.n_cells,),
        name=f"polymesh:{mesh.case_dir}" if mesh.case_dir else "polymesh",
    )


def from_overset_grid(grid) -> MeshData:
    """Adapt ``tensorfvm.overset.OversetGrid`` (2-D, NumPy).

    The grid stores cells as CCW polygons and cell adjacency; faces are
    recovered from the polygon edges (one face per unique edge).  The area
    vector follows the *owner's* edge direction: for a CCW polygon the
    outward normal of edge ``a -> b`` is ``(e_y, -e_x)``.
    """
    polygons = [np.asarray(p, dtype=float) for p in grid.polygons]
    owner_list, neighbor_list, centers_list, areas_list = [], [], [], []
    seen: dict[tuple, tuple[int, tuple, tuple]] = {}
    for i, poly in enumerate(polygons):
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            key = tuple(sorted((tuple(a), tuple(b))))
            if key in seen:
                j, aj, bj = seen[key]
                owner_list.append(j)
                neighbor_list.append(i)
                edge = np.asarray(bj) - np.asarray(aj)
            else:
                seen[key] = (i, tuple(a), tuple(b))
                owner_list.append(i)
                neighbor_list.append(-1)
                edge = b - a
            centers_list.append((a + b) * 0.5)
            areas_list.append(np.array([edge[1], -edge[0]]))
    owner = np.asarray(owner_list, dtype=np.int64)
    neighbor = np.asarray(neighbor_list, dtype=np.int64)
    labels = np.full(len(owner), "boundary", dtype=object)
    labels[neighbor >= 0] = "internal"
    face_masks = {}
    for name, mask in grid.boundaries.items():
        cell_mask = np.asarray(mask, dtype=bool)
        face_masks[name] = cell_mask[owner] & (neighbor < 0)
        labels[face_masks[name]] = name
    vertices = np.unique(np.vstack(polygons), axis=0)
    return MeshData(
        dim=2,
        vertices=vertices,
        cell_centers=np.asarray(grid.centers, dtype=float),
        cell_volumes=np.asarray(grid.volumes, dtype=float),
        owner=owner,
        neighbor=neighbor,
        face_centers=np.asarray(centers_list, dtype=float),
        area_vectors=np.asarray(areas_list, dtype=float),
        patch_labels=labels,
        masks=face_masks,
        field_shape=tuple(grid.shape),
        name=grid.name,
    )


def from_mesh2d(mesh) -> MeshData:
    """Adapt the seed contract ``tensorfvm.mesh_api.Mesh2D`` (2-D, torch)."""
    owner = _torch_to_numpy(mesh.owner).astype(np.int64)
    neighbor = _torch_to_numpy(mesh.neighbor).astype(np.int64)
    labels = np.full(len(owner), "boundary", dtype=object)
    labels[neighbor >= 0] = "internal"
    for name, mask in mesh.masks.items():
        labels[_torch_to_numpy(mask)] = name
    return MeshData(
        dim=2,
        vertices=_torch_to_numpy(mesh.vertices).astype(float).reshape(-1, 2),
        cell_centers=_torch_to_numpy(mesh.centers).reshape(-1, 2),
        cell_volumes=_torch_to_numpy(mesh.volumes).reshape(-1),
        owner=owner,
        neighbor=neighbor,
        face_centers=_torch_to_numpy(mesh.face_centers),
        area_vectors=_torch_to_numpy(mesh.face_area_vectors),
        patch_labels=labels,
        masks={k: _torch_to_numpy(v) for k, v in mesh.masks.items()},
        field_shape=tuple(int(s) for s in mesh.field_shape),
        name="mesh2d",
    )


def from_body_fitted3d(mesh) -> MeshData:
    """Adapt ``tensorfvm.body_fitted3d.BodyFittedCylinderMesh3D`` (3-D, torch).

    The slab mesh carries two face families: extruded lateral faces (full
    owner/neighbor/area-vector data) and spanwise faces (centres plus scalar
    areas; the normal is the slab direction, ``-z`` on the lower slab face and
    ``+z`` on the upper one, interior spanwise faces owned by the lower cell).
    """
    local_nz = mesh.z_stop - mesh.z_start
    ny, nx = mesh.ny, mesh.nx
    ncell = local_nz * ny * nx
    cell_index = np.arange(ncell, dtype=np.int64).reshape(local_nz, ny, nx)

    lat_owner = _torch_to_numpy(mesh.lateral_owner).reshape(-1)
    lat_neighbor = _torch_to_numpy(mesh.lateral_neighbor).reshape(-1)
    lat_centers = _torch_to_numpy(mesh.lateral_face_centers).reshape(-1, 3)
    lat_areas = _torch_to_numpy(mesh.lateral_face_area_vectors).reshape(-1, 3)
    lat_labels = np.repeat(np.asarray(mesh.lateral_labels, dtype=object), local_nz)

    sw_centers = _torch_to_numpy(mesh.spanwise_face_centers)  # (local_nz+1, ny, nx, 3)
    sw_areas = _torch_to_numpy(mesh.spanwise_face_areas)      # (local_nz+1, ny, nx)
    sw_owner = np.empty((local_nz + 1, ny, nx), dtype=np.int64)
    sw_owner[:local_nz] = cell_index
    sw_owner[local_nz] = cell_index[-1]
    sw_neighbor = np.full((local_nz + 1, ny, nx), -1, dtype=np.int64)
    sw_neighbor[1:local_nz] = cell_index[1:local_nz]
    sw_normal = np.zeros((local_nz + 1, ny, nx, 3))
    sw_normal[0, :, :, 2] = -1.0
    sw_normal[1:, :, :, 2] = 1.0
    sw_areas_vec = sw_areas[:, :, :, None] * sw_normal

    owner = np.concatenate([lat_owner, sw_owner.reshape(-1)])
    neighbor = np.concatenate([lat_neighbor, sw_neighbor.reshape(-1)])
    face_centers = np.concatenate([lat_centers, sw_centers.reshape(-1, 3)])
    area_vectors = np.concatenate([lat_areas, sw_areas_vec.reshape(-1, 3)])
    sw_labels = (["spanwise_lower"] * (ny * nx)
                 + ["internal"] * ((local_nz - 1) * ny * nx)
                 + ["spanwise_upper"] * (ny * nx))
    labels = np.concatenate([lat_labels, np.array(sw_labels, dtype=object)])
    return MeshData(
        dim=3,
        vertices=_torch_to_numpy(mesh.vertices).reshape(-1, 3),
        cell_centers=_torch_to_numpy(mesh.centers).reshape(-1, 3),
        cell_volumes=_torch_to_numpy(mesh.cell_volumes).reshape(-1),
        owner=owner,
        neighbor=neighbor,
        face_centers=face_centers,
        area_vectors=area_vectors,
        patch_labels=labels,
        masks=_masks_from_labels(labels),
        field_shape=(local_nz, ny, nx),
        name=f"body_fitted3d:z[{mesh.z_start}:{mesh.z_stop}]",
    )


#: Adapter registry: source class name -> adapter function.
MESH_ADAPTERS = {
    "NativeMesh3D": from_native_mesh3d,
    "PolyMesh": from_polymesh,
    "OversetGrid": from_overset_grid,
    "Mesh2D": from_mesh2d,
    "BodyFittedCylinderMesh3D": from_body_fitted3d,
}


def adapt_mesh(source) -> MeshData:
    """Dispatch on the source class name and validate the result."""
    key = type(source).__name__
    if key not in MESH_ADAPTERS:
        raise TypeError(f"no MeshData adapter for {key!r}; known: {sorted(MESH_ADAPTERS)}")
    return validate_mesh_data(MESH_ADAPTERS[key](source))
