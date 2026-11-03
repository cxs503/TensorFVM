"""Verification of the energy-compatible (Galerkin) viscous operator.

Scope: this module certifies *dissipativity* only.  The absolute scaling
(consistency with ``-div(2 mu S)``) is inherited from the native gradient
stencil and is deliberately not claimed here; see the audit report for the
reserved follow-up.

Covered:
* symmetry and positive semidefiniteness across a non-orthogonality sweep;
* dissipation on the open-defect geometry of ``test_native_energy_small``;
* the negative control: the legacy defect is *not* produced by the
  non-orthogonal correction term (damping sweep stays energy producing);
* that the new assembly changes no existing default path.
"""
import numpy as np
import pytest

from tensorfvm.native_compatible_viscous3d import (
    compatible_boundary_source,
    compatible_energy_rate,
    compatible_momentum_operators,
)
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra


def distorted_cube(n=2, shear=.18, tilt=.12):
    """Affine-sheared hexahedral cube with non-orthogonal faces."""
    points = np.array([[x, y, z] for z in np.linspace(0, 1, n + 1)
                       for y in np.linspace(0, 1, n + 1) for x in np.linspace(0, 1, n + 1)])
    points = points @ np.array([[1, shear, tilt], [0, 1, .2], [0, 0, 1]]).T

    def index(x, y, z):
        return z * (n + 1) ** 2 + y * (n + 1) + x

    cells = []
    for z in range(n):
        for y in range(n):
            for x in range(n):
                cells.append([index(x, y, z), index(x + 1, y, z), index(x + 1, y + 1, z),
                              index(x, y + 1, z), index(x, y, z + 1), index(x + 1, y, z + 1),
                              index(x + 1, y + 1, z + 1), index(x, y + 1, z + 1)])
    return NativeMesh3D.from_cells(points, cells, ['hexahedron'] * len(cells),
                                   triangulate_faces=True)


def prescribed_open_defect_field(mesh):
    velocity = np.zeros((len(mesh.cells), 3))
    velocity[0, 0] = 1.
    return velocity


def legacy_form(mesh, mask, velocity, mu=1., **kwargs):
    """``u^T (D A G) u``: positive values mean discrete kinetic-energy production."""
    viscous = NativeViscous3D(mesh, mask, mu, **kwargs)
    flat = velocity.ravel()
    return float(flat @ (viscous.momentum_diffusion_matrix @ flat))


def test_compatible_operator_is_symmetric_positive_semidefinite_across_skew_family():
    for skew in np.linspace(0, 1, 9):
        mesh = two_tetrahedra(skew)
        operators = compatible_momentum_operators(mesh, mesh.neighbor < 0)
        dense = operators['matrix'].toarray()
        scale = np.abs(dense).max()
        np.testing.assert_allclose(dense, dense.T, atol=1e-12 * scale)
        eigenvalue = np.linalg.eigvalsh(dense)
        assert eigenvalue[0] > -1e-10 * scale, (skew, eigenvalue[0])


def test_compatible_operator_dissipates_where_the_legacy_assembly_produces_energy():
    rng = np.random.default_rng(20261103)
    for skew in (0.5, 0.75, 0.9, 0.975, 1.):
        mesh = two_tetrahedra(skew)
        mask = mesh.neighbor < 0
        operators = compatible_momentum_operators(mesh, mask)
        fields = [prescribed_open_defect_field(mesh)] + [
            rng.normal(size=(len(mesh.cells), 3)) for _ in range(10)]
        for velocity in fields:
            assert compatible_energy_rate(operators, velocity) <= 1e-12
        if skew >= 0.975:
            assert legacy_form(mesh, mask, prescribed_open_defect_field(mesh)) > 0


def test_legacy_defect_survives_nonorthogonal_correction_damping():
    """Negative control: the correction term is not the root cause of the defect."""
    mesh = two_tetrahedra(1.)
    mask = mesh.neighbor < 0
    velocity = prescribed_open_defect_field(mesh)
    forms = [legacy_form(mesh, mask, velocity, correction_damping=damping)
             for damping in (1., .5, .1, 0.)]
    assert all(form > 0 for form in forms), forms
    assert forms[0] > forms[1] > forms[2] > forms[3], forms


def test_correction_damping_default_keeps_legacy_matrices_bitwise():
    mesh = distorted_cube(2)
    mask = mesh.neighbor < 0
    default = NativeViscous3D(mesh, mask, 1.3)
    explicit = NativeViscous3D(mesh, mask, 1.3, correction_damping=1.)
    np.testing.assert_array_equal(default.momentum_diffusion_matrix.toarray(),
                                  explicit.momentum_diffusion_matrix.toarray())
    np.testing.assert_array_equal(default.scalar_diffusion_matrix.toarray(),
                                  explicit.scalar_diffusion_matrix.toarray())


def test_invalid_arguments_are_rejected():
    mesh = distorted_cube(2)
    mask = mesh.neighbor < 0
    with pytest.raises(ValueError):
        compatible_momentum_operators(mesh, mask, -1.)
    with pytest.raises(ValueError):
        NativeViscous3D(mesh, mask, 1., correction_damping=1.5)
    with pytest.raises(ValueError):
        NativeViscous3D(mesh, mask, 1., correction_damping=np.zeros(3))
    operators = compatible_momentum_operators(mesh, mask)
    with pytest.raises(ValueError):
        compatible_boundary_source(operators, 0., np.full((len(mesh.owner), 3), np.nan))
    with pytest.raises(ValueError):
        compatible_energy_rate(operators, np.zeros((len(mesh.cells), 2)))