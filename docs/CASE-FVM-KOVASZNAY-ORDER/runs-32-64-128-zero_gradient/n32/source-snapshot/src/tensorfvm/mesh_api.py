"""Structural interface and runtime validation for two-dimensional FV meshes.

Mesh implementations expose geometry/topology tensors; discretization and
boundary-condition policy remain the solver's responsibility.  The protocol is
structural, so third-party mesh classes do not need to inherit a TensorFVM base
class.
"""

from typing import Mapping, Protocol, Sequence, runtime_checkable

import torch


@runtime_checkable
class Mesh2D(Protocol):
    """Minimum geometry/topology contract for collocated 2-D FV solvers.

    Cell tensors may be structured (for example ``(ny, nx, 2)`` centers and
    ``(ny, nx)`` volumes) or unstructured (for example ``(1, ncell, 2)`` and
    ``(1, ncell)``). Faces are flat and owner-oriented: ``neighbor == -1`` marks
    a boundary face, and the area vector points outward from its owner.
    """

    vertices: torch.Tensor
    centers: torch.Tensor
    volumes: torch.Tensor
    field_shape: tuple[int, int]
    owner: torch.Tensor
    neighbor: torch.Tensor
    face_vertices: torch.Tensor
    face_centers: torch.Tensor
    face_area_vectors: torch.Tensor
    face_lengths: torch.Tensor
    face_normals: torch.Tensor
    interior: torch.Tensor
    boundary: torch.Tensor
    masks: Mapping[str, torch.Tensor]


def validate_mesh2d(
    mesh: Mesh2D,
    *,
    required_masks: Sequence[str] = (),
    check_finite_geometry: bool = True,
) -> Mesh2D:
    """Validate a mesh against the solver-facing 2-D finite-volume contract.

    Raises ``TypeError`` for missing or wrongly typed fields and ``ValueError``
    for inconsistent topology, nonpositive cells/faces, malformed masks, or
    non-finite geometry.  Returns ``mesh`` to support validation at a factory
    boundary.
    """
    tensor_fields = (
        "vertices", "centers", "volumes", "owner", "neighbor", "face_vertices",
        "face_centers", "face_area_vectors", "face_lengths", "face_normals",
        "interior", "boundary",
    )
    for name in tensor_fields:
        if not hasattr(mesh, name):
            raise TypeError(f"Mesh2D is missing required field {name!r}")
        if not isinstance(getattr(mesh, name), torch.Tensor):
            raise TypeError(f"Mesh2D field {name!r} must be a torch.Tensor")
    if not hasattr(mesh, "masks") or not isinstance(mesh.masks, Mapping):
        raise TypeError("Mesh2D field 'masks' must be a mapping of face masks")

    if not hasattr(mesh, "field_shape"):
        raise TypeError("Mesh2D is missing required field 'field_shape'")
    field_shape = mesh.field_shape
    if (not isinstance(field_shape, tuple) or len(field_shape) != 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 1
                   for size in field_shape)):
        raise ValueError("Mesh2D field_shape must be a pair of positive integers")

    vertices = mesh.vertices
    centers = mesh.centers
    volumes = mesh.volumes
    owners = mesh.owner
    neighbors = mesh.neighbor
    face_vertices = mesh.face_vertices
    face_centers = mesh.face_centers
    area_vectors = mesh.face_area_vectors
    lengths = mesh.face_lengths
    normals = mesh.face_normals
    interior = mesh.interior
    boundary = mesh.boundary

    if vertices.ndim < 2 or vertices.shape[-1] != 2:
        raise ValueError("Mesh2D vertices must have a coordinate axis of length 2")
    cell_count = int(volumes.numel())
    if (cell_count < 1 or centers.numel() != 2 * cell_count
            or field_shape[0] * field_shape[1] != cell_count):
        raise ValueError(
            "Mesh2D centers, volumes, and field_shape must describe the same nonempty cells"
        )
    if owners.ndim != 1 or neighbors.ndim != 1 or owners.shape != neighbors.shape:
        raise ValueError("Mesh2D owner and neighbor arrays must be one-dimensional and equal-sized")
    face_count = int(owners.numel())
    if face_count < 1:
        raise ValueError("Mesh2D must contain at least one face")
    for name, tensor, shape in (
        ("face_vertices", face_vertices, (face_count, 2, 2)),
        ("face_centers", face_centers, (face_count, 2)),
        ("face_area_vectors", area_vectors, (face_count, 2)),
        ("face_normals", normals, (face_count, 2)),
        ("face_lengths", lengths, (face_count,)),
        ("interior", interior, (face_count,)),
        ("boundary", boundary, (face_count,)),
    ):
        if tuple(tensor.shape) != shape:
            raise ValueError(f"Mesh2D {name} must have shape {shape}")
    if owners.dtype != torch.long or neighbors.dtype != torch.long:
        raise ValueError("Mesh2D owner and neighbor indices must use torch.long")
    if interior.dtype != torch.bool or boundary.dtype != torch.bool:
        raise ValueError("Mesh2D interior and boundary flags must use torch.bool")
    geometry = (
        vertices, centers, volumes, face_vertices, face_centers,
        area_vectors, lengths, normals,
    )
    if any(not torch.is_floating_point(tensor) for tensor in geometry):
        raise ValueError("Mesh2D geometry tensors must use a floating-point dtype")
    devices = {tensor.device for tensor in geometry}
    if len(devices) != 1 or any(tensor.device not in devices for tensor in
                                (owners, neighbors, interior, boundary)):
        raise ValueError("Mesh2D tensors must be on a common device")
    if owners.numel() and (bool((owners < 0).any()) or bool((owners >= cell_count).any())):
        raise ValueError("Mesh2D owner indices are outside the cell range")
    if bool((neighbors < -1).any()) or bool((neighbors >= cell_count).any()):
        raise ValueError("Mesh2D neighbor indices must be -1 or inside the cell range")
    expected_interior = neighbors >= 0
    if not torch.equal(interior, expected_interior) or not torch.equal(boundary, ~expected_interior):
        raise ValueError("Mesh2D interior/boundary flags disagree with neighbor indices")
    if not bool(torch.isfinite(volumes).all()) or not bool((volumes > 0).all()):
        raise ValueError("Mesh2D cell volumes must be finite and strictly positive")
    if not bool(torch.isfinite(lengths).all()) or not bool((lengths > 0).all()):
        raise ValueError("Mesh2D face lengths must be finite and strictly positive")

    for name, mask in mesh.masks.items():
        if not isinstance(name, str) or not isinstance(mask, torch.Tensor):
            raise TypeError("Mesh2D masks must map string names to torch.Tensor values")
        if mask.dtype != torch.bool or tuple(mask.shape) != (face_count,):
            raise ValueError(f"Mesh2D mask {name!r} must be a boolean face vector")
        if mask.device != interior.device:
            raise ValueError(f"Mesh2D mask {name!r} must be on the mesh device")
        if bool((mask & interior).any()):
            raise ValueError(f"Mesh2D mask {name!r} may only mark boundary faces")
    missing_masks = set(required_masks) - set(mesh.masks)
    if missing_masks:
        raise ValueError("Mesh2D is missing required boundary masks: "
                         + ", ".join(sorted(missing_masks)))

    if check_finite_geometry:
        for name, tensor in (
            ("vertices", vertices), ("centers", centers),
            ("face_vertices", face_vertices), ("face_centers", face_centers),
            ("face_area_vectors", area_vectors),
            ("face_normals", normals),
        ):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"Mesh2D {name} must contain only finite values")
    midpoint = face_vertices.mean(1)
    if not torch.allclose(face_centers, midpoint, rtol=1e-10, atol=1e-12):
        raise ValueError("Mesh2D face centers must match their endpoint midpoints")
    edge = face_vertices[:, 1] - face_vertices[:, 0]
    endpoint_area_vectors = torch.stack((edge[:, 1], -edge[:, 0]), -1)
    if not torch.allclose(area_vectors, endpoint_area_vectors, rtol=1e-10, atol=1e-12):
        raise ValueError("Mesh2D area vectors must be derived from oriented face endpoints")
    if not torch.allclose(torch.linalg.vector_norm(area_vectors, dim=-1), lengths,
                          rtol=1e-10, atol=1e-12):
        raise ValueError("Mesh2D face lengths disagree with face area vectors")
    if not torch.allclose(torch.linalg.vector_norm(normals, dim=-1),
                          torch.ones_like(lengths), rtol=1e-8, atol=1e-10):
        raise ValueError("Mesh2D face normals must be unit vectors")
    if not torch.allclose(normals, area_vectors / lengths[:, None],
                          rtol=1e-8, atol=1e-10):
        raise ValueError("Mesh2D face normals must follow owner-oriented area vectors")
    return mesh
