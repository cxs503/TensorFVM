"""Qualification of the ``wall_closure='three_point'`` Dirichlet wall stencil.

History: the option shipped with three defects that made it *silently identical*
to ``two_point`` -- the alignment search compared the candidate internal-face
normal with the outward wall normal using the same sign (so it selected
nothing on a Cartesian box), the one-sided distances were taken along the
outward normal (so both cell centres looked like they sat behind the face), and
the sparse row replacement built ``drop`` with three entries per face while the
normal-derivative row has ``N`` columns (a ``ValueError`` that no test ever
reached).  ``wall_second_order_faces`` was therefore ``0`` on every mesh, which
these tests now pin.

What the revived closure actually buys is measured, not asserted: it is exactly
affine (round-off) and it cuts the *quadratic* wall normal-derivative error by
about 2.4x at matched grids, but it does **not** restore second order in the
manufactured-field study and it does **not** repay the documented O(1)
wall-adjacent cell divergence (``docs/native_viscous_dirichlet_order.md``).
The last two facts are pinned as recorded limitations so neither direction can
change silently.
"""
from functools import lru_cache

import numpy as np
import pytest

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import native_order3d as study

VISCOSITY = 1.3
GRADIENT = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
BASE = np.array([1.2, .8, -.4])


@lru_cache(maxsize=None)
def _operator(n, warp, closure):
    mesh = study.hex_box(n, warp)
    return mesh, NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY, wall_closure=closure)


@lru_cache(maxsize=None)
def _study(n, warp, closure, kind):
    """``truncation_study`` / ``face_traction_study`` for one closure."""
    original = study._operator
    study._operator = (lambda mesh, mu: NativeViscous3D(mesh, mesh.neighbor < 0, mu,
                                                       wall_closure=closure))
    try:
        if kind == "truncation":
            return study.truncation_study(n, warp, VISCOSITY)
        return study.face_traction_study(n, warp, VISCOSITY)
    finally:
        study._operator = original


def _affine_flux_error(n, warp, closure):
    mesh, operator = _operator(n, warp, closure)
    data = np.zeros((operator.F, 3))
    data[operator.dirichlet] = BASE + mesh.face_centers[operator.dirichlet] @ GRADIENT.T
    flux = operator.momentum_flux(BASE + mesh.cell_centers @ GRADIENT.T, dirichlet_values=data)
    symmetric = GRADIENT + GRADIENT.T - 2/3*np.trace(GRADIENT)*np.eye(3)
    return float(np.abs(flux - VISCOSITY*mesh.area_vectors @ symmetric.T).max())


def _quadratic_normal_derivative(n, warp, closure):
    """``u_x = x^2``: the analytic wall ``du/dn`` is ``2 x n_x``."""
    mesh, operator = _operator(n, warp, closure)
    field = mesh.cell_centers[:, 0]**2
    datum = mesh.face_centers[:, 0]**2
    gradient = np.stack([operator.face_gradient_matrices[a] @ field
                         + operator.face_gradient_source(0.0, datum)[:, a] for a in range(3)], axis=1)
    numeric = np.einsum('fa,fa->f', gradient, operator.normal)
    exact = 2*mesh.face_centers[:, 0]*operator.normal[:, 0]
    return float(np.abs(numeric - exact)[operator.dirichlet].max())


@pytest.mark.parametrize('warp', [0.0, 2.0])
def test_three_point_upgrades_every_dirichlet_face(warp):
    """Regression pin: the closure used to upgrade exactly zero faces."""
    mesh, operator = _operator(4, warp, 'three_point')
    assert int(operator.dirichlet.sum()) > 0
    assert operator.wall_second_order_faces == int(operator.dirichlet.sum())
    assert np.all(operator.wall_second_cell[operator.dirichlet] >= 0)
    # The stencil weights are a derivative of a constant: they sum to zero.
    assert np.abs(operator.wall_coeff[operator.dirichlet].sum(axis=1)).max() < 1e-12


def test_two_point_default_stays_untouched():
    """The historical default must not acquire any upgraded face."""
    _, operator = _operator(4, 0.0, 'two_point')
    assert operator.wall_second_order_faces == 0
    assert np.all(operator.wall_second_cell < 0)


def test_unknown_closure_is_rejected():
    mesh = study.hex_box(4)
    with pytest.raises(ValueError, match="wall_closure"):
        NativeViscous3D(mesh, mesh.neighbor < 0, VISCOSITY, wall_closure='four_point')


@pytest.mark.parametrize('closure', ['two_point', 'three_point'])
@pytest.mark.parametrize('warp', [0.0, 2.0])
def test_affine_traction_stays_exact(closure, warp):
    """Matrix and boundary-data parts of the stencil must stay consistent.

    Breaking the offset term (the tangential part of the radius vectors of a
    triangulated facet) leaves this at ~4e-1 instead of round-off.
    """
    assert _affine_flux_error(4, warp, closure) < 1e-12


@pytest.mark.parametrize('n', [4, 8])
def test_three_point_reduces_the_quadratic_wall_normal_derivative(n):
    ratio = _quadratic_normal_derivative(n, 0.0, 'three_point')/_quadratic_normal_derivative(n, 0.0, 'two_point')
    assert ratio < 0.45, ratio


def test_interior_operator_is_unchanged():
    """The wall stencil may not perturb the interior divergence operator."""
    for warp in (0.0, 2.0):
        for key in ('l2_interior', 'linf_interior'):
            two = _study(8, warp, 'two_point', 'truncation')[key]
            three = _study(8, warp, 'three_point', 'truncation')[key]
            assert three == pytest.approx(two, rel=0, abs=1e-12), (warp, key)


def test_recorded_limit_three_point_does_not_reach_second_order():
    """Recorded fact: the manufactured-field wall-normal derivative stays ~first order.

    The offset term is evaluated with the *two-point* tangential gradient, whose
    wall error is O(h); the correction weight is O(1), so the residual is O(h).
    Measured orders are 1.48 (warp 0) and 1.18 (warp 2) between n = 4 and 8.
    """
    for warp, expected in ((0.0, 1.48), (2.0, 1.18)):
        rows = [_study(n, warp, 'three_point', 'faces') for n in (4, 8)]
        order = study.orders(rows, 'rel_wall_normal_derivative')[0]
        assert order == pytest.approx(expected, abs=0.25), (warp, order)
        assert order < 1.8


def test_recorded_limit_wall_cell_divergence_is_not_repaid():
    """Recorded fact: the O(1) wall-cell divergence of the two-point closure survives.

    Matches the independent attribution in
    ``docs/native_viscous_dirichlet_order.md``: an exact wall normal derivative
    is not the driver, so upgrading it cannot restore refinement.  Measured
    L2 errors are 3.693 -> 3.379 (warp 0) and 3.918 -> 3.333 (warp 2).
    """
    for warp, bounds in ((0.0, (3.60, 3.30)), (2.0, (3.85, 3.25))):
        coarse, fine = (_study(n, warp, 'three_point', 'truncation')['l2_wall'] for n in (4, 8))
        assert coarse > bounds[0], (warp, coarse)
        assert fine > bounds[1], (warp, fine)
        assert fine/coarse > 0.85, (warp, fine/coarse)