"""Analytic audit of the 2-D face/cell quality metrics and their acceptance gate.

``tensorfvm.mesh_quality.mesh_quality`` decides whether a generated case mesh is
admissible (``quality_gate_passed``: positive cell areas, positive projected
centre--face distances and ``nonorthogonality_max_deg < 70``).  Until now the
only tests touching it needed ``gmsh``, so the metric definitions -- and the
70-degree gate itself -- had no executed, mesh-generator-independent check.

Every metric here has a closed form on a *parallelogram* (uniformly sheared)
grid:

* cell centres and face centres form an affine lattice, so the owner--neighbour
  connector passes exactly through the face centre: ``internal_skewness = 0``;
* the face normal deviates from that connector by exactly the shear angle, so
  ``nonorthogonality = arctan(gamma)``;
* projected face distances stay ``dy``/``dx`` regardless of the shear.

The definitions follow the standard finite-volume reading used by
OpenFOAM ``checkMesh`` (angle between the face normal and the vector joining
cell centres; skewness = distance from the face centre to the intersection of
that connector with the face, normalised by the connector length).  Boundary
faces use the cell-centre-to-face-centre vector, which is normal to the face on
a conforming mesh, so they contribute a zero angle.

Nothing here is a physics claim: it verifies a geometric pre-condition.
"""
import math
import numpy as np
import torch
from ..mesh_quality import mesh_quality

#: documented acceptance threshold, see ``mesh_quality.quality_gate_passed``
GATE_DEGREES = 70.

#: arccos near one amplifies double rounding to ~sqrt(eps) ~ 1e-6 deg; any
#: physically meaningful angle is orders of magnitude larger.
NOISE_DEG = 1e-5


def _straight_channel(nx=8, ny=4, gamma=0., *, length=.12, height=.02):
    """Conforming grid whose vertices are mapped by ``(x, y) -> (x + g y, y)``.

    The topology, face ownership and outward orientation are produced by the
    shipped ``CurvedChannelMesh`` construction; only its bottom-wall profile is
    replaced by the linear shear ``g = gamma x`` (the regular channel family is
    the ``gamma = 0`` member).  Re-using the production constructor keeps the
    audit from testing a re-implementation of the mesh.
    """
    import tensorfvm.curved_channel as curved_channel
    from ..solver import SolverConfig
    original = curved_channel.shape
    try:
        curved_channel.shape = lambda x, len_, amp: (
            gamma * x, torch.full_like(x, float(gamma)), torch.zeros_like(x), torch.zeros_like(x))
        config = SolverConfig(nx=nx, ny=ny, length=length, height=height, inlet_velocity=5e-4,
                              density=1000., reynolds=1., cylinder_radius=None,
                              tolerance=1e-6, max_iterations=1)
        return curved_channel.CurvedChannelMesh(config)
    finally:
        curved_channel.shape = original


def curved_channel(nx=8, ny=4):
    """The shipped sinusoidal channel: a genuinely non-orthogonal conforming mesh."""
    from tensorfvm.curved_channel import CurvedChannelMesh
    from ..solver import SolverConfig
    return CurvedChannelMesh(SolverConfig(nx=nx, ny=ny, length=.12, height=.02,
        inlet_velocity=5e-4, density=1000., reynolds=1., cylinder_radius=None,
        tolerance=1e-6, max_iterations=1))


def _array(t):
    return t.detach().cpu().numpy()


def independent_metrics(mesh):
    """Recompute every reported scalar from raw tensors, without the module."""
    owner, neighbor = _array(mesh.owner), _array(mesh.neighbor)
    interior = neighbor >= 0
    centers = _array(mesh.centers).reshape(-1, 2)
    face_centers = _array(mesh.face_centers)
    area_vectors = _array(mesh.face_area_vectors)
    volumes = _array(mesh.volumes).ravel()
    connector = face_centers - centers[owner]
    connector[interior] = centers[neighbor[interior]] - centers[owner[interior]]
    projected = np.einsum('ij,ij->i', area_vectors, connector)
    cosine = projected / np.linalg.norm(area_vectors, axis=1) / np.linalg.norm(connector, axis=1)
    angle = np.degrees(np.arccos(np.clip(cosine, -1., 1.)))
    fraction = np.einsum('ij,ij->i', area_vectors, face_centers - centers[owner]) / projected
    crossing = centers[owner] + fraction[:, None] * connector
    skewness = np.linalg.norm(face_centers - crossing, axis=1) / np.linalg.norm(connector, axis=1)
    return dict(
        cells=int(len(volumes)), faces=int(len(owner)),
        minimum_cell_area_m2=float(volumes.min()),
        minimum_face_projected_distance_m=float(np.min(projected / np.linalg.norm(area_vectors, axis=1))),
        nonorthogonality_max_deg=float(angle.max()),
        nonorthogonality_p95_deg=float(np.percentile(angle, 95)),
        internal_skewness_max=float(skewness[interior].max()),
        positive_cell_areas=bool(np.all(volumes > 0)),
        positive_projected_distances=bool(np.all(projected > 0)))


def rotate(mesh, angle_deg):
    """Rigid rotation of every geometric tensor; topology and volumes are unchanged."""
    theta = math.radians(angle_deg)
    rotation = torch.tensor([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]], dtype=torch.float64)
    rotated = object.__new__(type(mesh))
    rotated.__dict__.update(mesh.__dict__)
    rotated.vertices = mesh.vertices @ rotation.T
    rotated.centers = mesh.centers @ rotation.T
    rotated.face_centers = mesh.face_centers @ rotation.T
    rotated.face_area_vectors = mesh.face_area_vectors @ rotation.T
    rotated.face_normals = mesh.face_normals @ rotation.T
    return rotated


def shear_closed_form(angles_deg, nx=8, ny=4):
    """Measured non-orthogonality of a parallelogram grid versus the shear angle."""
    rows = []
    for angle in angles_deg:
        gamma = math.tan(math.radians(angle))
        mesh = _straight_channel(nx, ny, gamma)
        quality = mesh_quality(mesh)
        rows.append(dict(shear_deg=float(angle), gamma=float(gamma),
                         measured_nonorthogonality_deg=quality['nonorthogonality_max_deg'],
                         measured_skewness=quality['internal_skewness_max'],
                         measured_min_projected_distance_m=quality['minimum_face_projected_distance_m'],
                         gate_passed=quality['quality_gate_passed'],
                         absolute_error_deg=abs(quality['nonorthogonality_max_deg'] - angle)))
    return rows


def gate_flip_angle(nx=8, ny=4, tolerance=1e-9):
    """Bisect the shear angle at which the shipped gate flips, then compare to 70 deg."""
    low, high = 0., 89.
    assert mesh_quality(_straight_channel(nx, ny, math.tan(math.radians(low))))['quality_gate_passed']
    assert not mesh_quality(_straight_channel(nx, ny, math.tan(math.radians(high))))['quality_gate_passed']
    while high - low > tolerance:
        middle = .5 * (low + high)
        passed = mesh_quality(_straight_channel(nx, ny, math.tan(math.radians(middle))))['quality_gate_passed']
        low, high = (middle, high) if passed else (low, middle)
    return dict(flip_angle_deg=float(low), documented_gate_deg=GATE_DEGREES,
                absolute_error_deg=abs(low - GATE_DEGREES))


def inverted_face_control(nx=8, ny=4):
    """A single flipped face orientation must be rejected, not averaged away."""
    mesh = _straight_channel(nx, ny, 0.)
    flipped = object.__new__(type(mesh))
    flipped.__dict__.update(mesh.__dict__)
    index = int(np.flatnonzero(_array(mesh.neighbor) >= 0)[len(_array(mesh.neighbor)) // 8])
    area_vectors = mesh.face_area_vectors.clone()
    area_vectors[index] *= -1
    flipped.face_area_vectors = area_vectors
    flipped.face_normals = area_vectors / torch.linalg.vector_norm(area_vectors, dim=1)[:, None]
    quality = mesh_quality(flipped)
    return dict(flipped_face=int(index), positive_projected_distances=quality['positive_projected_distances'],
                nonorthogonality_max_deg=quality['nonorthogonality_max_deg'],
                gate_passed=quality['quality_gate_passed'])


def inverted_cell_control(nx=8, ny=4):
    """A negative cell area must fail the gate even though faces look fine."""
    mesh = _straight_channel(nx, ny, 0.)
    broken = object.__new__(type(mesh))
    broken.__dict__.update(mesh.__dict__)
    volumes = mesh.volumes.clone()
    row, column = volumes.shape[0] // 2, volumes.shape[1] // 2
    volumes[row, column] *= -2.
    broken.volumes = volumes
    quality = mesh_quality(broken)
    return dict(inverted_cell_area_m2=float(volumes[row, column]),
                positive_cell_areas=quality['positive_cell_areas'],
                gate_passed=quality['quality_gate_passed'])


def audit(nx=8, ny=4):
    """Run every control and return the machine-readable evidence dictionary."""
    orthogonal = mesh_quality(_straight_channel(nx, ny, 0.))
    reference = dict(nonorthogonality_max_deg=orthogonal['nonorthogonality_max_deg'],
                     internal_skewness_max=orthogonal['internal_skewness_max'],
                     minimum_face_projected_distance_m=orthogonal['minimum_face_projected_distance_m'],
                     positive_cell_areas=orthogonal['positive_cell_areas'],
                     positive_projected_distances=orthogonal['positive_projected_distances'],
                     gate_passed=orthogonal['quality_gate_passed'])
    curved = curved_channel(nx, ny)
    shipped = mesh_quality(curved)
    independent = independent_metrics(curved)
    differences = {key: abs(float(shipped[key]) - float(value)) for key, value in independent.items()
                   if key in shipped and isinstance(value, (bool, int, float))}
    rotation = []
    for angle in (0., 17., 43., 90.):
        quality = mesh_quality(rotate(curved, angle))
        rotation.append(dict(angle_deg=angle,
                             nonorthogonality_max_deg=quality['nonorthogonality_max_deg'],
                             internal_skewness_max=quality['internal_skewness_max'],
                             minimum_face_projected_distance_m=quality['minimum_face_projected_distance_m']))
    evidence = dict(
        grid=dict(nx=nx, ny=ny), gate_degrees=GATE_DEGREES,
        orthogonal_reference=reference,
        shear_closed_form=shear_closed_form([0., 5., 10., 20., 40., 60., 69., 71.], nx, ny),
        gate_flip=gate_flip_angle(nx, ny),
        curved_channel=dict(shipped=dict(shipped), independent=independent,
                            max_absolute_difference=max(differences.values()) if differences else 0.,
                            differences=differences),
        rotation_invariance=rotation,
        inverted_face_control=inverted_face_control(nx, ny),
        inverted_cell_control=inverted_cell_control(nx, ny))
    evidence['verdict'] = bool(
        reference['nonorthogonality_max_deg'] < NOISE_DEG and reference['internal_skewness_max'] < 1e-12
        and reference['gate_passed']
        and evidence['gate_flip']['absolute_error_deg'] < 1e-6
        and all(row['absolute_error_deg'] < NOISE_DEG and row['measured_skewness'] < 1e-10
                for row in evidence['shear_closed_form'])
        and all(row['gate_passed'] for row in evidence['shear_closed_form'] if row['shear_deg'] < GATE_DEGREES)
        and all(not row['gate_passed'] for row in evidence['shear_closed_form'] if row['shear_deg'] > GATE_DEGREES)
        and evidence['curved_channel']['max_absolute_difference'] < 1e-12
        and max(abs(row['nonorthogonality_max_deg'] - rotation[0]['nonorthogonality_max_deg'])
                for row in rotation) < 1e-10
        and max(abs(row['internal_skewness_max'] - rotation[0]['internal_skewness_max'])
                for row in rotation) < 1e-12
        and not evidence['inverted_face_control']['gate_passed']
        and not evidence['inverted_face_control']['positive_projected_distances']
        and not evidence['inverted_cell_control']['gate_passed']
        and not evidence['inverted_cell_control']['positive_cell_areas'])
    return evidence