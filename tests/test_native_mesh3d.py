import numpy as np
import pytest
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_operators3d import NativeOperators3D


@pytest.mark.parametrize(
    "kind,points,volume",
    [
        ("tetra", [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], 1 / 6),
        (
            "hexahedron",
            [
                [0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 0, 1],
                [1, 1, 1],
                [0, 1, 1],
            ],
            1,
        ),
        (
            "wedge",
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1]],
            0.5,
        ),
        ("pyramid", [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]], 1 / 3),
    ],
)
def test_linear_cells_have_real_volume_and_affine_response(kind, points, volume):
    # Shearing coordinates prevents Cartesian-only gradient formulas passing.
    transform = np.array([[1, 0.21, 0.17], [0, 1, 0.32], [0, 0, 1]])
    m = NativeMesh3D.from_cells(
        np.asarray(points) @ transform.T, [range(len(points))], [kind]
    )
    assert m.cell_volumes[0] == pytest.approx(volume)
    assert m.audit()["maximum_relative_closure"] < 1e-14
    a = NativeOperators3D(m).audit_affine()
    assert max(a.values()) < 1e-12


def test_shared_faces_cancel_independent_of_field():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]]
    m = NativeMesh3D.from_cells(vertices, [[0, 1, 2, 3], [0, 2, 1, 4]], ["tetra"] * 2)
    op = NativeOperators3D(m)
    assert op.internal.sum() == 1
    values = np.zeros((len(m.owner), 3))
    values[op.internal] = [4, -7, 2]
    result = op.integrate(values)
    np.testing.assert_allclose(result.sum(axis=0), 0, atol=0)
    np.testing.assert_allclose(result[0], -result[1], atol=0)


def test_exact_boundary_tags_required_and_unsupported_rejected():
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    with pytest.raises(ValueError, match="untagged"):
        NativeMesh3D.from_cells(vertices, [[0, 1, 2, 3]], ["tetra"], {})
    with pytest.raises(ValueError, match="unsupported"):
        NativeMesh3D.from_cells(vertices, [[0, 1, 2, 3]], ["tetra10"])
    with pytest.raises(ValueError, match="nonconvex|degenerate"):
        NativeMesh3D.from_cells(
            [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], [[0, 1, 2, 3]], ["tetra"]
        )


def test_nonplanar_hex_is_explicitly_rejected():
    vertices = [
        [0, 0, 0],
        [1, 0, 0],
        [1, 1, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1],
        [1, 1, 1.2],
        [0, 1, 1],
    ]
    with pytest.raises(ValueError, match="nonplanar"):
        NativeMesh3D.from_cells(vertices, [range(8)], ["hexahedron"])


def test_gmsh_volume_and_physical_patches(tmp_path):
    gmsh = pytest.importorskip("gmsh")
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    try:
        box = gmsh.model.occ.addBox(0, 0, 0, 1, 1, 1)
        gmsh.model.occ.synchronize()
        surfaces = [
            tag for dim, tag in gmsh.model.getBoundary([(3, box)], oriented=False)
        ]
        patch = gmsh.model.addPhysicalGroup(2, surfaces)
        gmsh.model.setPhysicalName(2, patch, "outer")
        gmsh.model.addPhysicalGroup(3, [box])
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.7)
        gmsh.model.mesh.generate(3)
        path = tmp_path / "cube.msh"
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    m = NativeMesh3D.from_gmsh(path)
    assert set(m.patch_labels) == {"internal", "outer"}
    assert m.cell_volumes.sum() == pytest.approx(1)
    assert max(NativeOperators3D(m).audit_affine().values()) < 1e-10


def test_face_reconstruction_reduces_to_geometric_projection_on_nonuniform_pair():
    """Zero-gradient reconstruction must reproduce the linear field at the
    geometric projection point of the face centroid.

    The pre-existing affine checks use single cells, i.e. no internal face, so
    the owner/neighbour interpolation weight of ``reconstruct_faces`` was never
    exercised (measured: mutant OP3-RECON-SWAP survived).  Here the second
    tetrahedron is four times longer than the first, so the owner->neighbour
    line meets the shared face well away from its midpoint: a swapped weight
    changes the result by 0.6 of the jump, while the criterion below is
    satisfied identically (to round-off) for any correct weight because an
    affine field is reproduced exactly at the plane intersection.
    """
    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -4]]
    m = NativeMesh3D.from_cells(vertices, [[0, 1, 2, 3], [0, 2, 1, 4]], ["tetra"] * 2)
    op = NativeOperators3D(m)
    (face,) = np.nonzero(op.internal)
    face = int(face[0])
    o = int(m.owner[face])
    n = int(m.neighbor[face])
    d = m.cell_centers[n] - m.cell_centers[o]
    s = m.area_vectors[face]
    # Plane-intersection parameter of the centre line with the face plane,
    # derived from geometry alone (independent of the operator's blend form).
    t = float(((m.face_centers[face] - m.cell_centers[o]) @ s) / (d @ s))
    assert abs(t - 0.5) > 0.1, "mesh must make the criterion sensitive to a weight swap"
    gradient = np.array([0.7, -1.3, 2.1])
    values = 1.7 + m.cell_centers @ gradient
    faces = op.reconstruct_faces(
        values, np.zeros_like(m.cell_centers), np.zeros(op.boundary.sum())
    )
    exact = 1.7 + (m.cell_centers[o] + t * d) @ gradient
    assert faces[face] == pytest.approx(exact, abs=1e-12)
    # Sensitivity: the arithmetic midpoint value is measurably different, so the
    # assertion above cannot be satisfied by an incorrect blend weight.
    midpoint = 1.7 + 0.5 * (m.cell_centers[o] + m.cell_centers[n]) @ gradient
    assert abs(midpoint - exact) > 1e-2
