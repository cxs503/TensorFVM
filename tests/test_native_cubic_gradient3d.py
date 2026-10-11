import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from tensorfvm.native_cubic_gradient3d import NativeCubicFaceGradient3D
from tests.test_native_quadratic3d import mesh


POWERS = [(a, b, c) for a in range(4) for b in range(4-a) for c in range(4-a-b)]


def basis(x):
    return np.prod(x[:, None, :]**np.asarray(POWERS)[None, :, :], axis=-1)


def reference(m):
    t, w = leggauss(4); t = (t+1)/2; w = w/2
    bary = np.array([[1-r, r*(1-s), r*s] for r in t for s in t])
    weights = np.array([2*r*wr*ws for r, wr in zip(t, w) for s, ws in zip(t, w)])
    points = np.einsum('qj,fjk->fqk', bary, m.vertices[np.asarray(m.face_vertices)])
    result = np.zeros((len(m.owner), len(POWERS), 3))
    for col, powers in enumerate(POWERS):
        for axis in range(3):
            if powers[axis]:
                e = np.array(powers); e[axis] -= 1
                result[:, col, axis] = powers[axis]*np.prod(points**e, axis=-1)@weights
    return result


@pytest.mark.parametrize('kind', ['fixed', 'mixed', 'free'])
def test_every_cubic_monomial_shared_area_gradient_and_boundary_sources(kind):
    m = mesh(n=4); fixed = m.neighbor < 0
    if kind == 'mixed': fixed[np.flatnonzero(fixed)[::2]] = False
    if kind == 'free': fixed[:] = False
    op = NativeCubicFaceGradient3D(m, fixed)
    actual = op.face_gradient(basis(m.cell_centers), basis(m.face_centers))
    np.testing.assert_allclose(actual, reference(m), atol=2e-11)
    np.testing.assert_allclose(op.face_gradient(np.ones(len(m.cells)), np.ones(len(m.owner))), 0, atol=3e-12)
    if kind == 'free': assert sum(r.nnz for r in op.boundary_response) == 0


def test_rotated_thin_cells_and_invalid_stencils():
    m = mesh(n=4, thin=True); op = NativeCubicFaceGradient3D(m, m.neighbor < 0)
    np.testing.assert_allclose(op.face_gradient(basis(m.cell_centers), basis(m.face_centers)), reference(m), atol=2e-7)
    with pytest.raises(ValueError, match='full-rank'):
        NativeCubicFaceGradient3D(mesh(n=1))
    with pytest.raises(ValueError, match='>=19'):
        NativeCubicFaceGradient3D(m, min_samples=18)
