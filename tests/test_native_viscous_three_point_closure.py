"""Verification gates for ``NativeViscous3D(wall_closure='three_point')``.

The option was merged (m2849) with no test and no measurement.  These gates pin
what the *fixed* option actually does, and they are written so that the two
defects found in the audit fail loudly if they ever come back:

* the option used to upgrade **zero** Dirichlet faces on every mesh (the
  distances were measured with the outward face normal while the algorithm
  needs the inward direction, so the ``delta0 <= 0`` guard rejected all faces);
* forcing it to engage crashed with ``all index and data arrays must have the
  same length`` (a COO row/column block was built with ``3`` columns instead of
  ``N``); that crash is why "the path was never executed" was provable;
* after fixing both, the stencil was **not** affine exact (4.7e-2 relative)
  because the Dirichlet datum lives at the laterally offset face centroid, not
  at the foot of the owner-centre normal; the tangential shift ``grad_t.t``
  rectifies it.

Recorded honest limits (see ``docs/native-viscous-three-point-closure/report.md``):
the wall ``du/dn`` error of the three-point stencil is *larger* in absolute
value than the two-point one at the resolutions tested (its error constant is
about twice as large) even though its observed order is closer to two, so the
option is not a drop-in accuracy win; the observed rates are pinned, the claim
"second order" is not.
"""

import numpy as np
import pytest

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import native_order3d as harmonic
from tensorfvm.verification import native_viscous_order3d as scalar

CLOSURES = ("two_point", "three_point")


def _affine_error(divisions, closure, viscosity=1.3):
    mesh = scalar.cube_mesh(divisions)
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, viscosity, wall_closure=closure)
    gradient = np.array([0.7, -0.3, 0.2])
    cell_values = 1.2 + mesh.cell_centers @ gradient
    boundary_values = 1.2 + mesh.face_centers @ gradient
    exact = viscosity * (mesh.area_vectors @ gradient)
    computed = op.scalar_flux(cell_values, 0.0, boundary_values)
    return float(np.abs(computed[boundary] - exact[boundary]).max()
                 / np.linalg.norm(exact[boundary]))


def _wall_flux(divisions, closure, mode):
    mesh = scalar.cube_mesh(divisions)
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, 1.3, wall_closure=closure)
    cell_values, _ = scalar.scalar_field(mesh.cell_centers, mode)
    boundary_values, boundary_gradient = scalar.scalar_field(mesh.face_centers, mode)
    computed = op.scalar_flux(cell_values, 0.0, boundary_values) / (1.3 * op.area)
    exact = np.einsum("ij,ij->i", boundary_gradient, op.normal)
    return float(np.linalg.norm(computed[boundary] - exact[boundary])
                 / np.linalg.norm(exact[boundary])), op


def _harmonic(n, closure):
    from scipy.sparse.linalg import spsolve

    mesh = harmonic.hex_box(n)
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, 1.3, wall_closure=closure)
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = harmonic.harmonic_velocity(mesh.face_centers[op.dirichlet])
    rhs = -(op.vector_divergence_matrix
            @ op.momentum_flux_source(dirichlet_values=data).ravel())
    solved = spsolve(op.momentum_diffusion_matrix.tocsc(), rhs)
    error = np.linalg.norm(solved.reshape(len(mesh.cells), 3)
                           - harmonic.harmonic_velocity(mesh.cell_centers), axis=1)
    return float(np.sqrt(np.mean(error ** 2))), op


def test_default_closure_is_two_point():
    mesh = scalar.cube_mesh((4, 4, 4))
    op = NativeViscous3D(mesh, mesh.neighbor < 0, 1.3)
    assert op.wall_closure == "two_point"
    assert op.wall_second_order_faces == 0


def test_unknown_closure_is_rejected():
    mesh = scalar.cube_mesh((4, 4, 4))
    with pytest.raises(ValueError):
        NativeViscous3D(mesh, mesh.neighbor < 0, 1.3, wall_closure="four_point")


@pytest.mark.parametrize("divisions", [(4, 4, 4), (6, 6, 6), (8, 8, 8)])
def test_three_point_engages_on_every_dirichlet_face(divisions):
    """Regression: this used to upgrade 0 faces and crash when forced to."""
    mesh = scalar.cube_mesh(divisions)
    boundary = mesh.neighbor < 0
    op = NativeViscous3D(mesh, boundary, 1.3, wall_closure="three_point")
    assert op.wall_second_order_faces == int(boundary.sum())
    assert np.all(op.wall_second_cell[boundary] >= 0)


@pytest.mark.parametrize("divisions", [(6, 6, 6), (4, 4, 4)])
def test_affine_fields_are_exact_for_both_closures(divisions):
    """A wrong one-sided sign or a missing lateral shift shows up as O(1e-2)."""
    for closure in CLOSURES:
        assert _affine_error(divisions, closure) < 1e-12


def test_two_point_wall_flux_matches_the_published_audit_values():
    """The default path must not move; values from docs/native_viscous_dirichlet_order.md."""
    value, _ = _wall_flux((8, 8, 8), "two_point", "trig3d")
    assert value == pytest.approx(2.4128e-02, rel=2e-2)


def test_three_point_halves_the_harmonic_solution_error():
    """The physically relevant quantity: error of a solve whose exact solution is known."""
    coarse_two, _ = _harmonic(4, "two_point")
    coarse_three, _ = _harmonic(4, "three_point")
    fine_two, _ = _harmonic(6, "two_point")
    fine_three, _ = _harmonic(6, "three_point")
    assert coarse_three < 0.7 * coarse_two
    assert fine_three < 0.7 * fine_two


def test_three_point_solution_converges_at_least_first_order():
    errors = [_harmonic(n, "three_point")[0] for n in (4, 6, 8)]
    order = float(np.log(errors[0] / errors[2]) / np.log(8.0 / 4.0))
    assert order > 1.5, order


def test_recorded_wall_normal_derivative_rate():
    """Recorded limit, not a claim of second order: rate climbs, absolute error does not fall.

    The two-point closure benefits from the corrected face-gradient estimate and
    keeps a smaller absolute error at these resolutions; the one-sided stencil
    reaches a rate near two but with roughly twice the error constant.  Both
    facts are pinned so neither is silently reworded later.
    """
    two = [_wall_flux(d, "two_point", "trig3d")[0] for d in ((4, 4, 4), (6, 6, 6), (8, 8, 8))]
    three = [_wall_flux(d, "three_point", "trig3d")[0] for d in ((4, 4, 4), (6, 6, 6), (8, 8, 8))]
    orders_two = float(np.log(two[1] / two[2]) / np.log(8.0 / 6.0))
    orders_three = float(np.log(three[1] / three[2]) / np.log(8.0 / 6.0))
    assert orders_three > orders_two
    assert 1.3 < orders_three < 1.8
    assert three[2] > two[2]


def test_one_dimensional_control_keeps_second_order_for_both_closures():
    """In 1-D the face centroid sits on the normal line, so the stencil is at its best."""
    for closure in CLOSURES:
        coarse = _wall_flux((8, 4, 4), closure, "trig1d")[0]
        fine = _wall_flux((16, 4, 4), closure, "trig1d")[0]
        order = float(np.log(coarse / fine) / np.log(2.0))
        assert order > 1.6, (closure, order)