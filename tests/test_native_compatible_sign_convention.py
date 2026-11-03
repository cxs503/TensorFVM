"""Sign convention, absolute scaling and defect boundary of the viscous assemblies.

Complements ``tests/test_native_compatible_viscous3d.py``, which certifies
dissipativity only.  Here the three questions that audit left open are pinned:

1. sign convention: the compatible operator ``K`` is PSD while the production
   slot ``D A G`` is negative (semi)definite on well-shaped meshes, so ``K`` is
   not a drop-in replacement and ``-K`` is;
2. absolute scaling: ``u^T K u`` equals the volume-weighted discrete strain
   energy exactly, so no artificial viscosity hides inside the operator;
3. defect boundary: the open legacy defect needs ``lambda_max(sym M) > 0``,
   which the two-tetrahedron family reaches but the sheared hexahedral family
   does not.
"""
import numpy as np
import pytest

from tensorfvm.native_compatible_viscous3d import (
    compatible_momentum_matrix_in_legacy_sign,
    compatible_momentum_operators,
)
from tensorfvm.native_mesh3d import NativeMesh3D
from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification.native_energy import two_tetrahedra


def sheared_cube(n=2, shear=.18, tilt=.12):
    """Affine-sheared hexahedral cube with non-orthogonal faces."""
    points = np.array([[x, y, z] for z in np.linspace(0, 1, n + 1)
                       for y in np.linspace(0, 1, n + 1) for x in np.linspace(0, 1, n + 1)])
    points = points @ np.array([[1, shear, tilt], [0, 1, .2], [0, 0, 1]]).T

    def index(x, y, z):
        return z * (n + 1) ** 2 + y * (n + 1) + x

    cells = [[index(x, y, z), index(x + 1, y, z), index(x + 1, y + 1, z), index(x, y + 1, z),
              index(x, y, z + 1), index(x + 1, y, z + 1), index(x + 1, y + 1, z + 1),
              index(x, y + 1, z + 1)]
             for z in range(n) for y in range(n) for x in range(n)]
    return NativeMesh3D.from_cells(points, cells, ['hexahedron'] * len(cells),
                                   triangulate_faces=True)


def assemblies(mesh, dynamic_viscosity=1.):
    mask = mesh.neighbor < 0
    legacy = NativeViscous3D(mesh, mask, dynamic_viscosity)
    operators = compatible_momentum_operators(mesh, mask, dynamic_viscosity)
    return legacy, operators


def eigenvalue_bounds(matrix):
    dense = np.asarray(matrix, dtype=float)
    values = np.linalg.eigvalsh(.5 * (dense + dense.T))
    return float(values[0]), float(values[-1])


def discretised_strain_energy(velocity, base, deviatoric=True, dynamic_viscosity=1.):
    gradient = np.stack([operator @ velocity for operator in base.gradient_matrices], axis=2)
    strain = .5 * (gradient + gradient.transpose(0, 2, 1))
    if deviatoric:
        strain = strain - np.trace(strain, axis1=1, axis2=2)[:, None, None] * np.eye(3) / 3
    return 2 * dynamic_viscosity * float(
        np.sum(base.mesh.cell_volumes * np.einsum('cij,cij->c', strain, strain)))


@pytest.mark.parametrize('mesh_name', ('sheared_cube', 'sheared_cube_strong'))
def test_compatible_matrix_is_ppsd_where_the_production_slot_is_negative(mesh_name):
    """The two assemblies use opposite sign conventions for the same physics."""
    mesh = sheared_cube(2) if mesh_name == 'sheared_cube' else sheared_cube(2, shear=.5, tilt=.3)
    legacy, operators = assemblies(mesh)
    low_legacy, high_legacy = eigenvalue_bounds(legacy.momentum_diffusion_matrix.toarray())
    low_compatible, high_compatible = eigenvalue_bounds(operators['matrix'].toarray())
    scale = max(abs(low_legacy), abs(high_compatible))
    assert high_legacy <= 1e-12 * scale, (high_legacy, scale)
    assert low_compatible >= -1e-12 * scale, (low_compatible, scale)
    rng = np.random.default_rng(4)
    for _ in range(32):
        probe = rng.normal(size=(len(mesh.cells), 3)).ravel()
        legacy_form = float(probe @ (legacy.momentum_diffusion_matrix @ probe))
        compatible_form = float(probe @ (operators['matrix'] @ probe))
        assert legacy_form <= 1e-12 * scale ** 2 * len(probe)
        assert compatible_form >= -1e-12 * scale ** 2 * len(probe)
        assert np.sign(legacy_form) != np.sign(compatible_form)


def test_legacy_sign_helper_makes_the_compatible_operator_a_dissipative_drop_in():
    mesh = sheared_cube(2)
    legacy, operators = assemblies(mesh)
    flipped = compatible_momentum_matrix_in_legacy_sign(operators)
    np.testing.assert_allclose(flipped.toarray(), -operators['matrix'].toarray(), atol=0)
    low, high = eigenvalue_bounds(flipped.toarray())
    scale = abs(low)
    assert low <= 0 <= 1e-12 * scale
    rng = np.random.default_rng(99)
    for _ in range(32):
        probe = rng.normal(size=(len(mesh.cells), 3)).ravel()
        assert float(probe @ (flipped @ probe)) <= 1e-12 * scale * len(probe) + 1e-12
    # Same dissipative sign as the production slot it replaces.
    assert high <= 1e-12 * scale


def test_absolute_scaling_is_the_volume_weighted_discrete_strain_norm():
    """No artificial viscosity: the matrix form reproduces 2 mu sum V dev(S):dev(S)."""
    rng = np.random.default_rng(20261103)
    meshes = [('two_tetrahedra_skew_0', two_tetrahedra(0.)),
              ('two_tetrahedra_skew_1', two_tetrahedra(1.)),
              ('sheared_cube_n2', sheared_cube(2)),
              ('sheared_cube_n3_shear_0.35', sheared_cube(3, shear=.35))]
    for label, mesh in meshes:
        for mu in (1., 3.5):
            velocity = rng.normal(size=(len(mesh.cells), 3))
            legacy, operators = assemblies(mesh, mu)
            quadratic = float(velocity.ravel() @ (operators['matrix'] @ velocity.ravel()))
            reference = discretised_strain_energy(velocity, legacy, operators['deviatoric'], mu)
            assert reference > 0, label
            assert abs(quadratic - reference) <= 1e-12 * reference, (label, mu, quadratic, reference)


def test_scaling_is_linear_in_viscosity():
    mesh = sheared_cube(2)
    velocity_seed = np.random.default_rng(5).normal(size=(len(mesh.cells), 3))
    forms = []
    for mu in (1., 2.5, 7.):
        _, operators = assemblies(mesh, mu)
        forms.append(float(velocity_seed.ravel() @ (operators['matrix'] @ velocity_seed.ravel())))
    np.testing.assert_allclose(forms / np.array([1., 2.5, 7.]), forms[0], rtol=1e-12)


def test_legacy_defect_requires_an_indefinite_symmetric_part():
    """lambda_max(sym M) <= 0 on sheared hexahedra; the skewed tetrahedra exceed it."""
    for n, shear in ((2, .18), (2, .5), (4, .35)):
        legacy, _ = assemblies(sheared_cube(n, shear=shear))
        _, high = eigenvalue_bounds(legacy.momentum_diffusion_matrix.toarray())
        assert high <= 1e-10 * max(1., abs(high)), (n, shear, high)
    legacy_defect, _ = assemblies(two_tetrahedra(1.))
    _, high = eigenvalue_bounds(legacy_defect.momentum_diffusion_matrix.toarray())
    assert high > 1., high


def test_momentum_residual_of_a_smooth_field_decays_on_refinement():
    """L2 residual of the production operator against -div(2 mu S) must shrink."""
    pi = np.pi
    field = lambda x: np.stack([np.sin(pi * x[:, 0]) * np.cos(pi * x[:, 1]),  # noqa: E731
                                -np.cos(pi * x[:, 0]) * np.sin(pi * x[:, 1]),
                                np.zeros(len(x))], axis=1)
    errors = []
    for n in (2, 4, 8):
        mesh = sheared_cube(n)
        legacy = NativeViscous3D(mesh, mesh.neighbor < 0, 1.)
        velocity = field(mesh.cell_centers)
        source = legacy.vector_divergence_matrix @ legacy.momentum_flux_source(
            np.zeros((len(mesh.owner), 3)), field(mesh.face_centers)).ravel()
        residual = ((legacy.momentum_diffusion_matrix @ velocity.ravel() + source)
                    .reshape(-1, 3)) / mesh.cell_volumes[:, None]
        exact = -2 * pi ** 2 * field(mesh.cell_centers)
        errors.append(float(np.linalg.norm(residual - exact) / np.linalg.norm(exact)))
    assert errors[0] > errors[-1] / 2, errors
    assert errors[-1] < .2, errors