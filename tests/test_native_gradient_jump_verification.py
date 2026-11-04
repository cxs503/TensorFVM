"""Verification of the tangential gradient-jump stabilisation against exact algebra.

``NativeGradientJump3D`` (``src/tensorfvm/native_gradient_jump3d.py``) is the
polyhedral face-jump (Korn) term required by Hansbo & Larson (2003,
doi:10.1051/m2an:2003020, eq. 26) for hybrid face velocities.  It carries no
reference verification of its own, so this module pins the three properties the
literature demands of such a term plus the exact scaling of its weights:

1. nonnegative energy (``M`` symmetric positive semidefinite);
2. it vanishes on the approximation space (energy exactly zero for every
   affine velocity field, i.e. all 12 rigid/linear modes are in the kernel);
3. its absolute scaling is ``mu*area/height`` per face with the analytic face
   inertia tensor of the triangulated polygonal face, so no hidden constant;
4. its per-cell stress replay closes on ``M @ u`` exactly, which keeps the
   interface-flux audit free of a defect the assembled matrix does not contain.

The companion audit script is ``scripts/verify_native_gradient_jump.py``.
"""
import numpy as np
import pytest

from tensorfvm.native_hybrid_viscous3d import NativeHybridViscous3D
from tensorfvm.native_mesh3d import NativeMesh3D


def hex_cube(n=2, shear=0.0):
    """Structured hexahedral unit cube, optionally sheared by an affine map."""
    points = np.array([[x, y, z] for z in np.linspace(0, 1, n + 1)
                       for y in np.linspace(0, 1, n + 1) for x in np.linspace(0, 1, n + 1)])
    if shear:
        points = points @ np.array([[1, shear, .05], [0, 1, .2], [0, 0, 1]]).T

    def index(x, y, z):
        return z * (n + 1) ** 2 + y * (n + 1) + x

    cells = [[index(x, y, z), index(x + 1, y, z), index(x + 1, y + 1, z), index(x, y + 1, z),
              index(x, y, z + 1), index(x + 1, y, z + 1), index(x + 1, y + 1, z + 1),
              index(x, y + 1, z + 1)]
             for z in range(n) for y in range(n) for x in range(n)]
    return NativeMesh3D.from_cells(points, cells, ['hexahedron'] * len(cells),
                                   triangulate_faces=True)


def tetra_pair():
    """Two tetrahedra sharing one face: the smallest mesh with one internal face."""
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [.2, .2, 1.], [.2, .2, -1.]])
    return NativeMesh3D.from_cells(points, [[0, 1, 2, 3], [0, 2, 1, 4]], ['tetra'] * 2)


FAMILIES = {
    'hex_cube_n2': lambda: hex_cube(2),
    'hex_sheared_n2': lambda: hex_cube(2, shear=.18),
    'hex_sheared_n3': lambda: hex_cube(3, shear=.1),
    'tetra_pair': tetra_pair,
}


def jump_operator(mesh, viscosity=1., strength=1.):
    operator = NativeHybridViscous3D(mesh, viscosity,
                                    gradient_jump_stabilization=strength).gradient_jump
    assert operator is not None
    return operator


def nodal_coordinates(mesh):
    return np.vstack([mesh.cell_centers, mesh.face_centers])


def stress_replay(operator, state):
    """Per-cell reaction rebuilt from the face stresses, signs included.

    ``gradient_reactions`` returns the per-face response; distributing it to the
    owner (+) and neighbour (-) and pulling it back through the cell gradient
    operators must reproduce the assembled operator action ``M @ u``.
    """
    mesh = operator.mesh
    nodes = len(mesh.cells) + len(mesh.owner)
    ids = operator.internal
    gradient = np.stack([g @ state.reshape(nodes, 3) for g in operator.gradient], axis=2)
    jump = gradient[mesh.owner[ids]] - gradient[mesh.neighbor[ids]]
    response = operator.weight[:, None, None] * np.einsum('fij,fjk->fik',
                                                          jump, operator.covariance)
    out = np.zeros((nodes, 3))
    for spatial in range(3):
        cell = np.zeros((len(mesh.cells), 3))
        np.add.at(cell, mesh.owner[ids], response[:, :, spatial])
        np.add.at(cell, mesh.neighbor[ids], -response[:, :, spatial])
        for component in range(3):
            out[:, component] += operator.gradient[spatial].T @ cell[:, component]
    return out.ravel()


@pytest.mark.parametrize('name', list(FAMILIES))
def test_matrix_is_symmetric_positive_semidefinite(name):
    mesh = FAMILIES[name]()
    matrix = jump_operator(mesh).matrix.toarray()
    assert np.array_equal(matrix, matrix.T)
    eigenvalues = np.linalg.eigvalsh(.5 * (matrix + matrix.T))
    assert eigenvalues[0] >= -1e-13
    assert eigenvalues[-1] > 0


@pytest.mark.parametrize('name', list(FAMILIES))
def test_affine_velocity_fields_have_zero_energy(name):
    """The stabilisation must not penalise rigid or linear velocity fields."""
    mesh = FAMILIES[name]()
    operator = jump_operator(mesh)
    coordinates = nodal_coordinates(mesh)
    generator = np.random.default_rng(20261103)
    for _ in range(3):
        affine = coordinates @ generator.normal(size=(3, 3)).T + generator.normal(size=3)
        stress, energy = operator.gradient_reactions(affine.ravel())
        assert abs(energy) < 1e-24
        assert np.abs(stress).max() < 1e-12
    assert operator.matrix.shape[0] - np.linalg.matrix_rank(operator.matrix.toarray()) >= 12


@pytest.mark.parametrize('name', list(FAMILIES))
def test_stress_replay_closes_on_assembled_operator(name):
    mesh = FAMILIES[name]()
    operator = jump_operator(mesh)
    nodes = len(mesh.cells) + len(mesh.owner)
    state = np.random.default_rng(7).normal(size=(nodes, 3))
    action = operator.matrix @ state.ravel()
    replayed = stress_replay(operator, state.ravel())
    assert np.abs(replayed - action).max() <= 1e-12 * max(np.abs(action).max(), 1e-30)
    _, energy = operator.gradient_reactions(state.ravel())
    assert energy == pytest.approx(float(state.ravel() @ operator.matrix @ state.ravel()),
                                   rel=1e-12, abs=1e-15)


def test_booked_reaction_is_the_energy_gradient_up_to_the_full_quadratic_form_convention():
    """The codebase books energies as ``u^T M u`` (no 1/2), so ``dE/du = 2 M u``.

    Central differences of the replayed energy therefore return twice the
    booked reaction ``M @ u``; both are separated here so that a future change
    of either convention is caught instead of averaging out.
    """
    mesh = hex_cube(2)
    operator = jump_operator(mesh)
    nodes = len(mesh.cells) + len(mesh.owner)
    state = np.random.default_rng(11).normal(size=(nodes, 3)).ravel()
    step = 1e-6
    for index in (0, 5, 17, 41):
        plus = state.copy()
        plus[index] += step
        minus = state.copy()
        minus[index] -= step
        finite_difference = (operator.gradient_reactions(plus)[1]
                             - operator.gradient_reactions(minus)[1]) / (2 * step)
        assert finite_difference == pytest.approx(2 * (operator.matrix @ state)[index],
                                                 rel=1e-6, abs=1e-9)


def test_absolute_scaling_matches_hand_derived_face_geometry():
    """Uniform 2x2x2 cube: internal faces are triangles of leg h = 1/2.

    ``weight = mu*area/height = (h^2/2)/h = h/2`` and a right triangle of leg h
    has centroid inertia eigenvalues ``h^2/12``, ``h^2/36`` in its plane and zero
    normal to it (trace ``h^2/9``).
    """
    h = .5
    operator = jump_operator(hex_cube(2))
    assert len(operator.internal) == 24
    assert np.allclose(operator.weight, h / 2., rtol=1e-12, atol=0.)
    eigenvalues = np.sort(np.linalg.eigvalsh(operator.covariance), axis=1)
    assert np.allclose(eigenvalues[:, 2], h ** 2 / 12., rtol=1e-12)
    assert np.allclose(eigenvalues[:, 1], h ** 2 / 36., rtol=1e-12)
    assert np.allclose(eigenvalues[:, 0], 0., atol=1e-15)
    assert np.allclose(eigenvalues.sum(axis=1), h ** 2 / 9., rtol=1e-12)


def test_strength_and_viscosity_scale_the_energy_linearly():
    mesh = hex_cube(2)
    nodes = len(mesh.cells) + len(mesh.owner)
    state = np.random.default_rng(3).normal(size=(nodes, 3))
    reference = float(state.ravel() @ jump_operator(mesh).matrix @ state.ravel())
    for strength in (.5, 2.5):
        energy = float(state.ravel() @ jump_operator(mesh, strength=strength).matrix
                       @ state.ravel())
        assert energy == pytest.approx(strength * reference, rel=1e-12)
    doubled = float(state.ravel() @ jump_operator(mesh, viscosity=2.).matrix @ state.ravel())
    assert doubled == pytest.approx(2. * reference, rel=1e-12)


def test_disabled_strength_builds_no_term_and_guards_reject_invalid_values():
    mesh = hex_cube(2)
    assert NativeHybridViscous3D(mesh, 1., gradient_jump_stabilization=0.).gradient_jump is None
    for invalid in (-1., float('nan'), float('inf')):
        with pytest.raises(ValueError):
            NativeHybridViscous3D(mesh, 1., gradient_jump_stabilization=invalid)


def test_defect_energy_vanishes_under_refinement():
    """A smooth field's energy must fall with h; no h-independent floor."""
    energies = []
    for n in (2, 3, 4):
        mesh = hex_cube(n)
        operator = jump_operator(mesh)
        coordinates = nodal_coordinates(mesh)
        scalar = coordinates[:, 0] ** 2 + coordinates[:, 1] * coordinates[:, 2]
        _, energy = operator.gradient_reactions(np.stack([scalar] * 3, axis=1).ravel())
        energies.append(energy)
    assert all(later < earlier for earlier, later in zip(energies, energies[1:]))
    order = np.log(energies[0] / energies[1]) / np.log(.5 / (1 / 3))
    assert 1.0 < order < 2.2


def test_term_stiffens_the_reduced_operator_without_touching_affine_modes():
    """The added stiffness must not change the signature of the velocity operator."""
    mesh = hex_cube(2, shear=.18)
    bare = NativeHybridViscous3D(mesh, 1., gradient_jump_stabilization=0.).matrix.toarray()
    stabilised = NativeHybridViscous3D(mesh, 1., gradient_jump_stabilization=1.).matrix.toarray()
    low_bare = np.linalg.eigvalsh(.5 * (bare + bare.T))[0]
    low_stabilised = np.linalg.eigvalsh(.5 * (stabilised + stabilised.T))[0]
    assert low_bare > 0 and low_stabilised > low_bare