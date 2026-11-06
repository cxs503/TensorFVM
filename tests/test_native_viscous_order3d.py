"""Order-of-accuracy facts for the native 3-D viscous operator.

These tests do not state what the order *should* be; they pin what the merged
operator actually delivers, measured against analytic fields, so that the
recorded limitation cannot silently change in either direction.

Two protocols are used:

``manufactured``
    :func:`tensorfvm.verification.native_order3d.manufactured`.  This field has
    ``div(tau(u)) = mu*lap u != 0``, so it solves no homogeneous problem and its
    Dirichlet solve error *saturates*; that saturation is the measured fact, and
    the field is used for truncation / face-traction diagnostics only.
``harmonic``
    :func:`tensorfvm.verification.native_order3d.harmonic_velocity`, built as
    ``curl`` of a component-wise harmonic potential, hence ``lap u = 0`` and
    ``div u = 0``, hence ``div(tau(u)) = 0`` identically.  This one *is* an exact
    solution of the Dirichlet problem, so its error slope is the solution order.
"""
from functools import lru_cache

import numpy as np
import pytest

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import native_order3d as study
from tensorfvm.verification import native_viscous_order3d as nv    # m2572 wall-closure module (name kept per m2530 ruling)

VISCOSITY = 1.3


# Mesh assembly dominates the runtime; one study per (n, warp) is enough.
@lru_cache(maxsize=None)
def _truncation(n, warp):
    return study.truncation_study(n, warp, VISCOSITY)


@lru_cache(maxsize=None)
def _faces(n, warp):
    return study.face_traction_study(n, warp, VISCOSITY)


@lru_cache(maxsize=None)
def _solution(n, warp):
    return study.solution_study(n, warp, VISCOSITY)


def test_analytic_stress_divergence_matches_finite_differences():
    points = np.array([[.13, .27, .41], [.71, .12, .88], [.5, .5, .5]])
    step = 1e-6
    gradient = study.manufactured_gradient(points)
    np.testing.assert_allclose(
        gradient, np.stack([(study.manufactured(points + step*np.eye(3)[b])
                            - study.manufactured(points - step*np.eye(3)[b]))/(2*step)
                           for b in range(3)], axis=2), atol=1e-7)
    numeric = np.zeros((len(points), 3))
    for j in range(3):
        unit = step*np.eye(3)[j]
        numeric += (study.stress(points + unit, VISCOSITY)
                    - study.stress(points - unit, VISCOSITY))[:, :, j]/(2*step)
    np.testing.assert_allclose(numeric, study.divergence_of_stress(points, VISCOSITY),
                               rtol=1e-7, atol=1e-6)


@pytest.mark.parametrize('warp', [0.0, 2.0])
def test_affine_velocity_gives_machine_precision_traction(warp):
    mesh = study.hex_box(4, warp)
    op = NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY)
    gradient = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
    base = np.array([1.2, .8, -.4])
    data = np.zeros((op.F, 3))
    data[op.dirichlet] = base + mesh.face_centers[op.dirichlet] @ gradient.T
    flux = op.momentum_flux(base + mesh.cell_centers @ gradient.T, dirichlet_values=data)
    symmetric = gradient + gradient.T - 2/3*np.trace(gradient)*np.eye(3)
    expected = VISCOSITY*mesh.area_vectors @ symmetric.T
    np.testing.assert_allclose(flux, expected, atol=1e-12)
    # Piecewise-linear fields carry no cell-integrated viscous divergence.
    assert np.abs(op.op.integrate(flux)/mesh.cell_volumes[:, None]).max() < 1e-12


def test_interior_cell_divergence_is_second_order():
    rows = [_truncation(n, 0.0) for n in (4, 8)]
    order = study.orders(rows, 'l2_interior')[0]
    assert order > 1.5, order


def test_wall_cell_divergence_does_not_refine():
    """Recorded defect: wall-adjacent cells stay at O(1) truncation error."""
    rows = [_truncation(n, 0.0) for n in (4, 8)]
    assert study.orders(rows, 'l2_wall')[0] < 0.3
    assert rows[1]['linf_wall'] > 0.8*rows[0]['linf_wall']


def test_wall_normal_derivative_is_first_order():
    rows = [_faces(n, 0.0) for n in (4, 8)]
    order = study.orders(rows, 'rel_wall_normal_derivative')[0]
    assert 0.6 < order < 1.6, order


def test_homogeneous_comparison_shows_no_measurable_convergence():
    """Recorded plateau: comparing a homogeneous solve with a non-solution field."""
    rows = [_solution(n, 0.0) for n in (4, 8)]
    ratio = rows[1]['l2_global']/rows[0]['l2_global']
    assert ratio > 0.85, (ratio, study.orders(rows, 'l2_global')[0])
    assert rows[1]['linf_global'] > 0.9*rows[0]['linf_global']
    for row in rows:  # the linear solve itself is exact
        assert row['residual'] < 1e-12


@pytest.mark.xfail(strict=True, reason='Open defect: wall-adjacent cell divergence is O(1), not second order')
def test_wall_cell_divergence_must_be_second_order():
    rows = [_truncation(n, 0.0) for n in (4, 8)]
    assert study.orders(rows, 'l2_wall')[0] > 1.5


def test_dirichlet_diffusion_solution_converges_once_the_source_is_restored():
    """The former strict xfail here measured a missing source term, not the operator.

    ``_solution`` solves ``div(tau(u)) = 0`` and compares the result with the
    manufactured field, which has O(1) ``div(tau)``; the plateau above is that
    mismatch.  Restoring the analytic stress divergence as a cell source makes
    the manufactured field the exact solution of the discrete problem, and the
    same assembly converges at second order (see
    ``tests/test_native_viscous_manufactured_solve.py``).
    """
    rows = [study.manufactured_solution_study(n, 0.0, VISCOSITY) for n in (4, 8)]
    assert study.orders(rows, 'l2_interior')[0] > 1.9
    assert study.orders(rows, 'l2_global')[0] > 1.7


# --------------------------------------------------------------------------- #
# m2572 additions rebased on the merged test file (the two m2530-era
# test_dirichlet_diffusion_* anchors are superseded by the merged
# test_homogeneous_comparison_* / ..._converges_once_the_source_is_restored)
# --------------------------------------------------------------------------- #

def test_affine_exactness_is_preserved_on_every_family_mesh():
    for divisions in [(8, 8, 8), (16, 4, 4), (6, 6, 6)]:
        mesh = nv.cube_mesh(divisions)
        assert nv.affine_exactness(mesh) < 1e-12
    warped = nv.cube_mesh((6, 6, 6), warp=0.35)
    assert nv.affine_exactness(warped) < 1e-12




def test_one_dimensional_control_is_second_order():
    rows = nv.family([(16, 4, 4), (32, 4, 4)], mode="trig1d")
    orders = nv.orders(rows, "wall_flux")
    assert orders[0] > 1.8, orders
    assert rows[-1]["wall_flux"] < 6e-4




def test_three_dimensional_wall_flux_is_only_first_order():
    """Regression pin for the known limitation: m2530's FAIL is reproduced."""
    rows = nv.family([(4, 4, 4), (8, 8, 8)], mode="trig3d")
    orders = nv.orders(rows, "wall_flux")
    assert 0.8 < orders[0] < 1.6, orders
    assert rows[-1]["wall_flux"] > 1e-2




def test_metrics_disagree_in_magnitude_but_share_the_rate_direction():
    rows = nv.family([(4, 4, 4), (8, 8, 8)], mode="trig3d")
    assert rows[-1]["face"] > rows[-1]["wall_flux"]
    assert rows[-1]["face"] < rows[0]["face"]
    assert rows[-1]["face_floored"] < rows[-1]["face"]




def test_fixed_value_matrix_constant_mode_vanishes_with_refinement():
    coarse = nv.wall_closure_metrics(nv.cube_mesh((4, 4, 4)))
    fine = nv.wall_closure_metrics(nv.cube_mesh((8, 8, 8)))
    assert fine["constant_mode"] < coarse["constant_mode"]
    assert fine["constant_mode"] < 0.3
    ratio = coarse["constant_mode"] / fine["constant_mode"]
    assert 1.2 < ratio < 2.5, ratio




def test_manufactured_fields_match_finite_differences():
    points = np.array([[0.31, 0.42, 0.27], [0.68, 0.13, 0.77]])
    for mode in ("trig1d", "trig3d"):
        values, gradient = nv.scalar_field(points, mode)
        assert values.shape == (2,)
        assert gradient.shape == (2, 3)
        step = 1e-6
        numeric = np.zeros((2, 3))
        for axis in range(3):
            shift = np.zeros(3)
            shift[axis] = step
            numeric[:, axis] = (
                nv.scalar_field(points + shift, mode)[0] - nv.scalar_field(points - shift, mode)[0]
            ) / (2 * step)
        np.testing.assert_allclose(gradient, numeric, rtol=0, atol=2e-6)




def test_unknown_field_mode_and_bad_divisions_are_rejected():
    with pytest.raises(ValueError, match="unknown"):
        nv.scalar_field(np.zeros((1, 3)), "cubic")
    with pytest.raises(ValueError, match="subdivisions"):
        nv.cube_mesh((4, 4))


@lru_cache(maxsize=None)
def _harmonic(n, warp):
    return study.harmonic_solution_study(n, warp, VISCOSITY)


def test_harmonic_field_really_solves_the_homogeneous_problem():
    """The exact-solution protocol is only as good as this identity."""
    points = np.array([[.13, .27, .41], [.71, .12, .88], [.5, .5, .5], [.93, .07, .65]])
    step = 1e-6
    gradient = study.harmonic_gradient(points)
    np.testing.assert_allclose(
        gradient, np.stack([(study.harmonic_velocity(points + step*np.eye(3)[b])
                            - study.harmonic_velocity(points - step*np.eye(3)[b]))/(2*step)
                           for b in range(3)], axis=2), atol=1e-7)
    # div u = 0 and lap u = 0, hence div tau = mu*(lap u + grad(div u)/3) = 0.
    assert np.abs(np.trace(gradient, axis1=1, axis2=2)).max() < 1e-12
    laplacian = sum((study.harmonic_gradient(points + step*np.eye(3)[b])
                     - study.harmonic_gradient(points - step*np.eye(3)[b]))[:, :, b]/(2*step)
                    for b in range(3))
    scale = max(np.abs(study.harmonic_velocity(points)).max(), 1.0)
    assert np.abs(laplacian).max() < 1e-6*scale


def test_manufactured_field_is_not_a_solution_of_the_homogeneous_problem():
    """Why the manufactured Dirichlet solve cannot converge (documented for years).

    ``div(tau(u_manufactured)) = mu*lap u`` is O(10) on the unit box while the
    solution error is O(1): the residual never vanishes with ``h``, so the
    discrete solution converges to the *solution* of the homogeneous problem and
    its distance to the manufactured field saturates at a mesh-independent value.
    """
    points = np.array([[.2, .3, .4], [.6, .1, .8], [.5, .5, .5]])
    residual = study.divergence_of_stress(points, VISCOSITY)
    assert np.linalg.norm(residual, axis=1).max() > 5.0
    for row in (_solution(4, 0.0), _solution(8, 0.0)):
        assert row['residual'] < 1e-12          # the linear solve itself is exact
    ratio = _solution(8, 0.0)['l2_global']/_solution(4, 0.0)['l2_global']
    assert ratio > 0.85, ratio                   # and the error saturates


def test_exact_solution_dirichlet_solve_converges():
    """The operator converges once the target really is a solution.

    Measured L2 orders on the harmonic protocol for n = 4, 8, 12, 16 are
    1.43 / 1.03 / 0.78 (interior 0.88 / 0.87 / 1.07): first order, with the
    spread of a pre-asymptotic family.  The gate below is a floor, not a claim.
    """
    rows = [_harmonic(n, 0.0) for n in (4, 8)]
    order = study.orders(rows, 'l2_global')[0]
    assert order > 0.6, order
    assert rows[0]['l2_global'] < 0.05*rows[0]['field_scale']
    for row in rows:
        assert row['residual'] < 1e-10


def test_wall_cell_divergence_is_carried_by_the_dirichlet_face_flux():
    """Mechanism behind the O(1) wall truncation: it is a wall-flux deficit.

    Summing only the Dirichlet-face contributions of the traction error
    reproduces >= 85 % of the total wall-cell divergence error (92 % at n=4,
    98 % at n=8), and Dirichlet faces are never owned by an interior cell, so the
    interior share is exactly zero.  An interior-operator defect would not look
    like this.
    """
    for n in (4, 8):
        row = _truncation(n, 0.0)
        share = row['l2_wall_wall_faces_only']/row['l2_wall']
        assert share > 0.85, (n, share)
        assert row['l2_interior_wall_faces_only'] == 0.0


@lru_cache(maxsize=None)
def _msolution(n, warp):
    return study.manufactured_solution_study(n, warp, VISCOSITY)


@lru_cache(maxsize=None)
def _hextension(n, warp):
    return study.harmonic_extension_study(n, warp, VISCOSITY)


def test_manufactured_source_is_the_operator_source():
    """The integrated source must be the one this operator actually sees.

    A mis-signed or mis-scaled source would make the convergence ratio blow up
    (or add a fixed offset); the measured ratio for this family is ~0.29.
    """
    rows = [_msolution(n, 0.0) for n in (4, 8)]
    ratio = rows[1]['l2_global']/rows[0]['l2_global']
    assert 0.15 < ratio < 0.6, (ratio, [r['l2_global'] for r in rows])


def test_manufactured_dirichlet_solution_converges():
    """Well-posed Dirichlet solve: the discrete field must refine toward ``u_man``.

    The analytic source ``div(tau(u_man))`` is integrated, so the manufactured
    field solves the boundary-value problem exactly.  Measured L2 order is 1.80
    on this family; the gate is a first-order floor because the wall flux is
    one-sided (see ``test_wall_cell_divergence_does_not_refine``).
    """
    rows = [_msolution(n, 0.0) for n in (4, 8)]
    order = study.orders(rows, 'l2_global')[0]
    assert order > 1.5, (order, [r['l2_global'] for r in rows])
    for row in rows:
        assert row['residual'] < 1e-12
        assert row['l2_global'] < 0.05*row['field_scale']


def test_harmonic_extension_distance_has_no_convergence_rate():
    """A source-free solve measures the harmonic extension, so it cannot refine.

    ``div(tau(u)) = 0`` with the manufactured field on the boundary has the
    harmonic extension as its exact solution; the distance to ``manufactured``
    therefore converges to a nonzero constant (measured ratio ~0.94 per
    refinement).  Pinned so nobody re-reads this as a solver defect.
    """
    rows = [_hextension(n, 0.0) for n in (4, 8)]
    ratio = rows[1]['l2_global']/rows[0]['l2_global']
    assert ratio > 0.85, (ratio, study.orders(rows, 'l2_global')[0])
    for row in rows:  # the linear solve itself is exact
        assert row['residual'] < 1e-12
