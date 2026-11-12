"""Solver-level zero-normal-derivative (symmetry / free-slip) face closure.

``CASE-FVM-KOVASZNAY-ORDER`` was blocked by a first-order spurious spanwise
velocity on extruded meshes.  m16524 attributed it to the hybrid face closure
on the symmetry planes: with the *tangential* components left as free momentum
unknowns the discrete zero-traction row lets the plane face value drift away
from the owner cell value, which feeds ``tau_xz`` and drives an O(h) ``w``.

The physical symmetry condition is ``d u_t / d n = 0``, which the solver could
not impose (it offered Dirichlet, traction/Neumann and Robin only).  This test
locks the added capability -- ``zero_gradient_faces``: the face component is
*tied to its owner cell value*, its equation is replaced by the constraint, and
the tied dofs leave the unknown set.  The Kovasznay extrusion then reproduces
``w == 0`` exactly without being told the exact tangential data.
"""

import numpy as np
import pytest
from scipy import sparse

from tensorfvm.native_velocity_space3d import NativeVelocitySpace3D
from tensorfvm.verification.kovasznay_extrusion import error_budgets, extrusion_solver
from tensorfvm.verification.native_hybrid_flow import channel_mesh

N = 8


def _mesh(divisions=(4, 4, 1)):
    return channel_mesh(divisions, triangulate_faces=True)


def _planes(mesh):
    z = mesh.face_centers[:, 2]
    return (mesh.neighbor < 0) & ((z < 1e-10) | (z > 1 - 1e-10))


def _zero_gradient_mask(mesh):
    mask = np.zeros((len(mesh.owner), 3), bool)
    mask[_planes(mesh), :2] = True
    return mask


def test_zero_gradient_mask_requires_boundary_faces():
    mesh = _mesh()
    internal = np.zeros((len(mesh.owner), 3), bool)
    internal[int(np.flatnonzero(mesh.neighbor >= 0)[0]), 0] = True
    with pytest.raises(ValueError):
        NativeVelocitySpace3D(mesh, 3, np.zeros((len(mesh.owner), 3), bool), None,
                              zero_gradient_components=internal)


def test_zero_gradient_and_dirichlet_components_cannot_overlap():
    mesh = _mesh()
    planes = _planes(mesh)
    dirichlet = np.zeros((len(mesh.owner), 3), bool)
    dirichlet[planes, :] = True
    with pytest.raises(ValueError):
        NativeVelocitySpace3D(mesh, 3, dirichlet, None,
                              zero_gradient_components=_zero_gradient_mask(mesh))


def test_tied_dofs_leave_the_unknown_set_and_copy_their_owner_cell():
    mesh = _mesh()
    zero_gradient = _zero_gradient_mask(mesh)
    plain = NativeVelocitySpace3D(mesh, 3, np.zeros((len(mesh.owner), 3), bool))
    tied = NativeVelocitySpace3D(mesh, 3, np.zeros((len(mesh.owner), 3), bool), None,
                                 zero_gradient_components=zero_gradient)
    assert tied.tied_any and not plain.tied_any
    assert plain.free.size - tied.free.size == int(zero_gradient.sum())
    # the constraint manifold: face value == owner cell value on every tied dof
    state = tied.expand(np.random.default_rng(0).standard_normal(tied.free.size))
    faces, components = np.nonzero(zero_gradient)
    face = state[3 * tied.N:].reshape(tied.F, 3)
    cell = state[:3 * tied.N].reshape(tied.N, 3)
    assert np.array_equal(face[zero_gradient], cell[mesh.owner[faces], components])
    # and restrict/expand round-trips on the manifold
    assert np.array_equal(tied.expand(tied.restrict(state)), state)


def test_reduced_matrix_is_row_restricted_substitution():
    """Tied rows are replaced by the constraint, not added to the cell balance.

    ``(A P)[free]`` -- the Galerkin projection ``P^T A P`` would double-count
    the tied face flux in the owner cell row; the two differ, so this pins the
    convention.
    """
    mesh = _mesh()
    zero_gradient = _zero_gradient_mask(mesh)
    space = NativeVelocitySpace3D(mesh, 3, np.zeros((len(mesh.owner), 3), bool), None,
                                  zero_gradient_components=zero_gradient)
    rng = np.random.default_rng(1)
    full_size = 3 * (space.N + space.F)
    full = sparse.random(full_size, full_size, density=0.02, random_state=1,
                         format='csr').toarray()
    full = sparse.csr_matrix((full + full.T) * .5)     # the viscous operator is symmetric
    reduced = space.matrix(full)
    assert reduced.shape == (space.free.size, space.free.size)
    y = rng.standard_normal(space.free.size)
    state = space.prolongation @ y
    assert np.allclose(reduced @ y, (full @ state)[space.free])
    assert not np.allclose(reduced @ y, (space.prolongation.T @ full @ space.prolongation) @ y)


@pytest.fixture(scope='module')
def tied_run():
    mesh, solver, exact = extrusion_solver(N, 40., tangential='zero_gradient')
    solver.solve(max_outer_iterations=80, tolerance=1e-8, relaxation=.7)
    return mesh, solver, exact


def test_zero_gradient_closure_uses_no_exact_tangential_data(tied_run):
    """The closure is a boundary *condition*, not a manufactured-data crutch."""
    mesh, solver, exact = tied_run
    planes = _planes(mesh)
    prescribed = solver.prescribed.reshape(-1, 3)[solver.N:]
    assert np.all(prescribed[planes, :2] == 0.)      # no analytic u, v injected
    assert np.all(prescribed[planes, 2] == 0.)       # w = 0 is the symmetry value
    assert solver.viscous.velocity_space.zero_gradient_components[planes, :2].all()


def test_zero_gradient_planes_remove_the_spanwise_artefact(tied_run):
    """w is analytically zero and the tie closure keeps it at zero exactly."""
    mesh, solver, exact = tied_run
    budgets = error_budgets(mesh, solver, exact)
    assert budgets['spanwise_l2'] == 0.
    assert budgets['max_abs_w'] == 0.
    # the constraint is satisfied by the accepted state (face value = cell value)
    difference = solver.state - solver.prescribed
    assert solver.viscous.velocity_space.constraint_error(difference) <= 1e-12
    # in-plane accuracy is not degraded by the closure (pinned n = 8 value,
    # the delivered free-traction closure gives 4.0092e-02)
    assert budgets['inplane_l2'] == pytest.approx(4.0054e-02, rel=0.02)
    assert budgets['total_l2'] == pytest.approx(budgets['inplane_l2'], rel=1e-12)