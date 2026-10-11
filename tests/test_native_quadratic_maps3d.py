import numpy as np
import pytest
from scipy import sparse
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_quadratic3d import NativeQuadratic3D
from tensorfvm.native_quadratic_maps3d import face_mean_matrix, face_gradient_matrices


def actual_mesh(mixed=False):
    n = 3
    points = np.array([[x, y, z] for z in np.linspace(0, 1, n+1)
                       for y in np.linspace(0, 1, n+1)
                       for x in np.linspace(0, 1, n+1)])
    x, y, z = points.T.copy()
    points += np.column_stack((.06*y*z, .04*x*z, .08*x*y))
    points = points@np.array([[1, .18, .12], [0, 1, .2], [0, 0, 1]]).T
    def index(x, y, z):
        return z*(n+1)**2+y*(n+1)+x
    cells = []
    for z in range(n):
        for y in range(n):
            for x in range(n):
                cells.append([index(x, y, z), index(x+1, y, z),
                              index(x+1, y+1, z), index(x, y+1, z),
                              index(x, y, z+1), index(x+1, y, z+1),
                              index(x+1, y+1, z+1), index(x, y+1, z+1)])
    mesh = NativeMesh3D.from_cells(points, cells, ['hexahedron']*len(cells),
                                 triangulate_faces=True)
    if mixed:
        # Split one actual native hex into face-fan tetrahedra. Its interface
        # triangles are exactly those shared by the remaining native hexes.
        center = len(points)
        points = np.vstack((points, mesh.cell_centers[0]))
        faces = np.flatnonzero((mesh.owner == 0)|(mesh.neighbor == 0))
        cells = cells[1:]+[list(mesh.face_vertices[f])+[center] for f in faces]
        types = ['hexahedron']*26+['tetra']*len(faces)
        mesh = NativeMesh3D.from_cells(points, cells, types, triangulate_faces=True)
        assert np.count_nonzero(mesh.neighbor >= 0) > 0
        assert set(mesh.cell_types) == {'hexahedron', 'tetra'}
    return mesh


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('boundary', ['dirichlet', 'mixed', 'free'])
def test_sparse_face_maps_match_original_random_scalar_and_vector(mixed, boundary):
    mesh = actual_mesh(mixed)
    mask = mesh.neighbor < 0
    if boundary == 'mixed':
        mask[np.flatnonzero(mask)[::2]] = False
    elif boundary == 'free':
        mask[:] = False
    op = NativeQuadratic3D(mesh, mask)
    rng = np.random.default_rng(174)
    for values in (rng.normal(size=op.N), rng.normal(size=(op.N, 3))):
        for fixed_mean in (False, True):
            matrix = face_mean_matrix(op, fixed_mean)
            target = op.face_mean(values, 0., boundary_face_means=0. if fixed_mean else None)
            np.testing.assert_allclose(matrix@values, target, rtol=2e-13, atol=3e-13)
            assert matrix.format == 'csr'
            if fixed_mean:
                assert matrix[mask].nnz == 0
        for compact in (False, True):
            maps = face_gradient_matrices(op, compact)
            target = op.face_gradient(values, 0., compact=compact)
            np.testing.assert_allclose(np.stack([g@values for g in maps], axis=-1),
                                       target, rtol=3e-13, atol=2e-12)
            assert all(g.format == 'csr' for g in maps)


@pytest.mark.parametrize('mixed', [False, True])
@pytest.mark.parametrize('boundary', ['dirichlet', 'mixed', 'free'])
def test_provided_conservative_gauss_gradient_maps_and_free_boundaries(mixed, boundary):
    mesh = actual_mesh(mixed)
    mask = mesh.neighbor < 0
    if boundary == 'mixed':
        mask[np.flatnonzero(mask)[::2]] = False
    elif boundary == 'free':
        mask[:] = False
    op = NativeQuadratic3D(mesh, mask)
    faces = face_mean_matrix(op)
    rows = np.r_[mesh.owner, mesh.neighbor[op.internal]]
    columns = np.r_[np.arange(op.F), np.flatnonzero(op.internal)]
    values = np.r_[np.ones(op.F), -np.ones(op.internal.sum())]
    divergence = sparse.coo_matrix((values, (rows, columns)), shape=(op.N, op.F)).tocsr()
    gauss = tuple((sparse.diags(1/mesh.cell_volumes)@divergence
                   @sparse.diags(mesh.area_vectors[:, a])@faces).tocsr() for a in range(3))
    unknown = np.random.default_rng(37).normal(size=(op.N, 2))
    cell_gradient = np.stack([g@unknown for g in gauss], axis=-1)
    for compact in (False, True):
        maps = face_gradient_matrices(op, compact, gauss)
        actual = np.stack([g@unknown for g in maps], axis=-1)
        expected = op.face_gradient(unknown, 0., compact=compact,
                                    cell_gradients=cell_gradient)
        np.testing.assert_allclose(actual, expected, rtol=3e-13, atol=3e-12)
        # Free boundaries keep their true extrapolated gradient even when
        # compact internal gradients are requested.
        raw = op.face_gradient(unknown, 0., compact=False, cell_gradients=cell_gradient)
        free = op.boundary & ~op.dirichlet
        np.testing.assert_allclose(actual[free], raw[free],
                                   rtol=3e-13, atol=3e-12)


def test_reject_incompatible_or_nonfinite_gradient_maps():
    op = NativeQuadratic3D(actual_mesh())
    with pytest.raises(ValueError, match='Three finite'):
        face_gradient_matrices(op, cell_gradient_matrices=[sparse.eye(op.N)]*2)
    with pytest.raises(ValueError, match='Three finite'):
        face_gradient_matrices(op, cell_gradient_matrices=[np.eye(op.N)]*3)
    broken = sparse.eye(op.N, format='csr')
    broken.data[0] = np.nan
    with pytest.raises(ValueError, match='Three finite'):
        face_gradient_matrices(op, cell_gradient_matrices=[broken]*3)
