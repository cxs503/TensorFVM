"""Comprehensive verification of NativeOperators3D core operators.

Covers: integrate (conservative cancellation), least_squares_gradient
(affine exactness on arbitrary polyhedra), reconstruct_faces (skew-corrected
interpolation), gauss_gradient (Gauss theorem), and audit_affine (end-to-end).

Reference: the operators implement the standard FVM gradient reconstruction
on unstructured polyhedral meshes (Jasak 1996, PhD thesis, Ch. 3).
Affine exactness is the minimum requirement for second-order consistency.
"""
import numpy as np
import pytest

from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.overset3d import CartesianGrid3D
from tensorfvm.native_operators3d import NativeOperators3D


def _skewed_hex(n=4):
    """Unit cube with shear transform to produce nonorthogonal faces."""
    base = CartesianGrid3D.from_bounds([0, 0, 0], [1, 1, 1], [n, n, n])
    shear = np.array([[1, 0.3, 0.1], [0, 1, 0.2], [0, 0, 1]])
    points = base.vertices @ shear.T
    return NativeMesh3D.from_cells(points, base.cells, base.cell_types,
                                   triangulate_faces=True)


def _single_hex():
    """Single hexahedron — minimal stencil, exercises boundary-only paths."""
    base = CartesianGrid3D.from_bounds([0, 0, 0], [1, 1, 1], [1, 1, 1])
    return NativeMesh3D.from_cells(base.vertices, base.cells, base.cell_types,
                                   triangulate_faces=True)


def _mixed_mesh(n=3):
    """Hex+tetra mixed mesh via boundary subdivision."""
    base = CartesianGrid3D.from_bounds([0, 0, 0], [1, 1, 1], [n, n, n])
    points = base.vertices @ np.array([[1, .18, .12], [0, 1, .2], [0, 0, 1]]).T
    original = NativeMesh3D.from_cells(points, base.cells, base.cell_types,
                                       triangulate_faces=True)
    center = len(points)
    points = np.vstack((points, original.cell_centers[0]))
    faces = np.flatnonzero((original.owner == 0) | (original.neighbor == 0))
    cells = list(base.cells[1:]) + [list(original.face_vertices[f]) + [center]
                                     for f in faces]
    return NativeMesh3D.from_cells(points, cells,
                                   ['hexahedron'] * (n**3 - 1) + ['tetra'] * len(faces),
                                   triangulate_faces=True)


# ---------------------------------------------------------------------------
# integrate: conservative cancellation
# ---------------------------------------------------------------------------

class TestIntegrate:
    """Conservative face integration: internal faces cancel, boundary sums."""

    def test_global_sum_equals_boundary_sum(self):
        """The global sum of integrate(f) over all cells equals the sum of
        f over boundary faces only (internal faces cancel globally)."""
        m = _skewed_hex(4)
        op = NativeOperators3D(m)
        field = np.random.default_rng(42).normal(size=(len(m.owner), 3))
        result = op.integrate(field)
        global_sum = result.sum(axis=0)
        boundary_sum = field[~op.internal].sum(axis=0)
        np.testing.assert_allclose(global_sum, boundary_sum, atol=1e-13)

    def test_single_cell_all_boundary(self):
        m = _single_hex()
        op = NativeOperators3D(m)
        assert op.internal.sum() == 0
        assert op.boundary.sum() == len(m.owner)
        # All faces are boundary; integrate with signed=True returns
        # the sum of all face values for the single cell.
        field = np.arange(len(m.owner), dtype=float)
        result = op.integrate(field)
        assert result.shape == (1,)
        np.testing.assert_allclose(result[0], field.sum(), atol=1e-14)

    def test_wrong_shape_rejected(self):
        m = _skewed_hex(3)
        op = NativeOperators3D(m)
        with pytest.raises(ValueError, match='one quantity per native face'):
            op.integrate(np.ones(len(m.owner) - 1))

    def test_unsigned_adds_internal(self):
        """With signed=False, internal faces are added (not subtracted)."""
        m = _skewed_hex(3)
        op = NativeOperators3D(m)
        field = np.ones(len(m.owner))
        signed_result = op.integrate(field, signed=True)
        unsigned_result = op.integrate(field, signed=False)
        # Unsigned should be larger (internal faces add instead of cancel)
        assert unsigned_result.sum() > signed_result.sum()


# ---------------------------------------------------------------------------
# least_squares_gradient: affine exactness
# ---------------------------------------------------------------------------

class TestLeastSquaresGradient:
    """Weighted LS gradient must reproduce the exact gradient of any affine field."""

    @pytest.mark.parametrize('mesh_factory', [_skewed_hex, _mixed_mesh])
    def test_affine_scalar_gradient_exact(self, mesh_factory):
        m = mesh_factory(4)
        op = NativeOperators3D(m)
        exact_grad = np.array([0.7, -1.3, 2.1])
        offset = 1.7
        cell_values = offset + m.cell_centers @ exact_grad
        face_values = offset + m.face_centers @ exact_grad
        grad = op.least_squares_gradient(cell_values, face_values[op.boundary])
        np.testing.assert_allclose(grad, np.tile(exact_grad, (len(m.cells), 1)),
                                   atol=1e-12)

    def test_constant_field_zero_gradient(self):
        m = _skewed_hex(4)
        op = NativeOperators3D(m)
        cell_values = np.full(len(m.cells), 3.14)
        face_values = np.full(op.boundary.sum(), 3.14)
        grad = op.least_squares_gradient(cell_values, face_values)
        np.testing.assert_allclose(grad, 0, atol=1e-14)

    def test_boundary_data_shape_checked(self):
        m = _skewed_hex(3)
        op = NativeOperators3D(m)
        cell_values = np.zeros(len(m.cells))
        with pytest.raises(ValueError, match='scalar cell values'):
            op.least_squares_gradient(cell_values, np.ones(op.boundary.sum() - 1))

    def test_single_cell_gradient_from_boundary(self):
        """On a single hex, the LS gradient with exact boundary data must
        recover the affine gradient."""
        m = _single_hex()
        op = NativeOperators3D(m)
        grad_exact = np.array([2.0, -0.5, 1.0])
        cell_values = m.cell_centers @ grad_exact
        face_values = m.face_centers @ grad_exact
        grad = op.least_squares_gradient(cell_values, face_values)
        np.testing.assert_allclose(grad, grad_exact[None, :], atol=1e-12)


# ---------------------------------------------------------------------------
# reconstruct_faces: skew-corrected interpolation
# ---------------------------------------------------------------------------

class TestReconstructFaces:
    """Face reconstruction must be exact for affine fields."""

    @pytest.mark.parametrize('mesh_factory', [_skewed_hex, _mixed_mesh])
    def test_affine_face_reconstruction_exact(self, mesh_factory):
        m = mesh_factory(4)
        op = NativeOperators3D(m)
        exact_grad = np.array([0.7, -1.3, 2.1])
        offset = 1.7
        cell_values = offset + m.cell_centers @ exact_grad
        face_exact = offset + m.face_centers @ exact_grad
        grad = op.least_squares_gradient(cell_values, face_exact[op.boundary])
        faces = op.reconstruct_faces(cell_values, grad, face_exact[op.boundary])
        np.testing.assert_allclose(faces, face_exact, atol=1e-12)

    def test_boundary_values_passed_through(self):
        m = _skewed_hex(3)
        op = NativeOperators3D(m)
        cell_values = np.zeros(len(m.cells))
        gradients = np.zeros((len(m.cells), 3))
        boundary_values = np.arange(op.boundary.sum(), dtype=float)
        faces = op.reconstruct_faces(cell_values, gradients, boundary_values)
        np.testing.assert_allclose(faces[op.boundary], boundary_values, atol=1e-15)


# ---------------------------------------------------------------------------
# gauss_gradient: Gauss theorem
# ---------------------------------------------------------------------------

class TestGaussGradient:
    """Gauss gradient = (1/V) * sum_f (phi_f * S_f)."""

    def test_constant_field_zero_gradient(self):
        m = _skewed_hex(4)
        op = NativeOperators3D(m)
        face_values = np.full(len(m.owner), 2.5)
        grad = op.gauss_gradient(face_values)
        np.testing.assert_allclose(grad, 0, atol=1e-14)

    def test_affine_field_gradient_exact_on_uniform_grid(self):
        """On a uniform Cartesian grid, the Gauss gradient of an affine
        face field recovers the exact gradient."""
        base = CartesianGrid3D.from_bounds([0, 0, 0], [1, 1, 1], [4, 4, 4])
        m = NativeMesh3D.from_cells(base.vertices, base.cells, base.cell_types,
                                    triangulate_faces=True)
        op = NativeOperators3D(m)
        exact_grad = np.array([1.0, 2.0, 3.0])
        face_values = m.face_centers @ exact_grad
        grad = op.gauss_gradient(face_values)
        np.testing.assert_allclose(grad, np.tile(exact_grad, (len(m.cells), 1)),
                                   atol=1e-12)

    def test_gauss_gradient_on_skew_grid(self):
        """On a skewed grid, the Gauss gradient of affine face values
        recovers the exact gradient (Gauss theorem is exact for linear
        fields on any polyhedral mesh)."""
        m = _skewed_hex(4)
        op = NativeOperators3D(m)
        exact_grad = np.array([0.7, -1.3, 2.1])
        face_values = m.face_centers @ exact_grad
        grad = op.gauss_gradient(face_values)
        np.testing.assert_allclose(grad, np.tile(exact_grad, (len(m.cells), 1)),
                                   atol=1e-11)


# ---------------------------------------------------------------------------
# audit_affine: end-to-end verification
# ---------------------------------------------------------------------------

class TestAuditAffine:
    """audit_affine runs a battery of affine-exactness checks."""

    @pytest.mark.parametrize('mesh_factory', [_skewed_hex, _mixed_mesh])
    def test_all_audit_metrics_near_zero(self, mesh_factory):
        m = mesh_factory(4)
        op = NativeOperators3D(m)
        audit = op.audit_affine()
        for key, value in audit.items():
            assert value < 1e-10, f"{key} = {value}"

    def test_single_hex_audit(self):
        m = _single_hex()
        op = NativeOperators3D(m)
        audit = op.audit_affine()
        for key, value in audit.items():
            assert value < 1e-10, f"{key} = {value}"

    def test_audit_keys_present(self):
        m = _skewed_hex(3)
        op = NativeOperators3D(m)
        audit = op.audit_affine()
        expected_keys = {
            'least_squares_gradient_max_error',
            'face_affine_max_error',
            'gauss_gradient_max_error',
            'constant_vector_divergence_max_error',
            'affine_vector_divergence_max_error',
            'global_flux_balance_error',
        }
        assert set(audit.keys()) == expected_keys

    def test_affine_divergence_is_six(self):
        """The vector field u=(x, 2y, 3z) has div(u)=6 everywhere."""
        m = _skewed_hex(4)
        op = NativeOperators3D(m)
        audit = op.audit_affine()
        assert audit['affine_vector_divergence_max_error'] < 1e-10

    def test_global_flux_balance(self):
        """Total flux through all faces equals boundary flux sum."""
        m = _skewed_hex(4)
        op = NativeOperators3D(m)
        audit = op.audit_affine()
        assert audit['global_flux_balance_error'] < 1e-14
