"""Attribution of the O(1) wall-adjacent cell divergence (pinned facts).

The audit script ``scripts/audit_wall_cell_gradient_attribution.py`` replaces
individual components of the assembled face-gradient tensor by oracle values and
measures the cell divergence.  The facts pinned here are the ones that decide
which repair route is worth investing in:

* the analytic wall-normal derivative alone changes nothing -> the recorded
  ``native_order3d.wall_error_model`` (a half-cell normal difference) is *not*
  the driver of the ``O(1)`` wall-cell error;
* an exact wall-face gradient still only refines at order ~1.3-1.5, while an
  exact face gradient on *every* face refines at second order -> the wall face
  cannot be fixed in isolation;
* exact cell-centre gradients averaged to the faces do not help either -> the
  defect is the evaluation point / stencil of the face gradient, not the
  accuracy of the cell gradient;
* a second-order Taylor least-squares reconstruction of every face gradient
  (:mod:`tensorfvm.verification.face_lsq3d`) does not lift the wall-cell error
  either, because the divergence amplifies a face-gradient error ``dg`` by
  ``1/h`` -- an ``O(h**2)`` gradient is not enough.
"""
import importlib.util
from functools import lru_cache
from pathlib import Path

import numpy as np

from tensorfvm.native_viscous3d import NativeViscous3D
from tensorfvm.verification import face_lsq3d as fl
from tensorfvm.verification import native_order3d as order

_AUDIT = Path(__file__).resolve().parents[1]/'scripts'/'audit_wall_cell_gradient_attribution.py'
_SPEC = importlib.util.spec_from_file_location('wall_cell_attribution_audit', _AUDIT)
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


@lru_cache(maxsize=None)
def _study(n):
    return audit.study(n)


def _order(rows_a, rows_b, h_a, h_b, key, field=0):
    return float(np.log(rows_a[key][field]/rows_b[key][field])/np.log(h_a/h_b))


def test_taylor_reconstruction_reproduces_an_affine_gradient():
    mesh = order.hex_box(3)
    gradient = np.array([[.7, .2, -.1], [-.3, .4, .6], [.15, -.25, .1]])
    values = mesh.cell_centers @ gradient.T
    wall = mesh.neighbor < 0
    prescribed = mesh.face_centers[wall] @ gradient.T
    for degree in (1, 2):
        reconstructed, info = fl.reconstruct(mesh, values, degree=degree, rings=2,
                                            dirichlet=wall, dirichlet_values=prescribed)
        np.testing.assert_allclose(
            reconstructed, np.broadcast_to(gradient, reconstructed.shape), atol=1e-11)
        assert np.isfinite(info['condition']).all() and info['condition'].max() < 1e3


def test_oracle_substitution_reproduces_the_assembled_flux():
    rows, diagnostics = _study(4)
    assert diagnostics['consistency'] < 1e-12, diagnostics['consistency']
    # The substitution framework must not perturb the untouched operator.
    mesh = order.hex_box(4)
    wall = mesh.neighbor < 0
    velocity = order.manufactured(mesh.cell_centers)
    data = np.zeros((len(mesh.face_centers), 3))
    data[wall] = order.manufactured(mesh.face_centers[wall])
    operator = NativeViscous3D(mesh, wall, audit.VISCOSITY)
    flux, gradients = fl.operator_face_gradients(operator, velocity, data)
    np.testing.assert_allclose(
        flux, operator.momentum_flux(velocity, dirichlet_values=data), atol=1e-12)
    np.testing.assert_allclose(fl.divergence(mesh, flux),
                               operator.op.integrate(flux)/mesh.cell_volumes[:, None],
                               atol=1e-12)


def test_wall_cell_error_does_not_refine():
    rows4, _ = _study(4)
    rows6, _ = _study(6)
    order_46 = _order(rows4, rows6, 1/4, 1/6, 'as_built')
    assert abs(order_46) < 0.5, order_46
    assert rows6['as_built'][0] > 0.8*rows4['as_built'][0]


def test_exact_wall_normal_derivative_changes_nothing():
    """Kills the half-cell normal-difference explanation of the defect."""
    rows4, _ = _study(4)
    rows6, _ = _study(6)
    assert rows4['wall_normal_exact'][0] > 0.95*rows4['as_built'][0]
    assert rows6['wall_normal_exact'][0] > 0.95*rows6['as_built'][0]
    assert abs(_order(rows4, rows6, 1/4, 1/6, 'wall_normal_exact')) < 0.5


def test_exact_wall_face_gradient_is_not_enough():
    """The wall face in isolation tops out well below second order."""
    rows4, _ = _study(4)
    rows6, _ = _study(6)
    order_46 = _order(rows4, rows6, 1/4, 1/6, 'wall_full_exact')
    assert 1.0 < order_46 < 1.8, order_46
    assert rows6['wall_full_exact'][0] > 3*rows6['all_faces_exact'][0]


def test_exact_cell_gradients_averaged_to_faces_do_not_help():
    """The global ceiling is second order only when every face gradient is exact."""
    rows4, _ = _study(4)
    rows6, _ = _study(6)
    order_46 = _order(rows4, rows6, 1/4, 1/6, 'all_faces_exact')
    assert order_46 > 1.8, order_46
    stalled = _order(rows4, rows6, 1/4, 1/6, 'interp_exact_cell_gradients')
    assert stalled < 0.9, stalled
    assert rows6['interp_exact_cell_gradients'][0] > 5*rows6['all_faces_exact'][0]


def test_second_order_face_lsq_does_not_fix_the_wall_cells():
    """An O(h^2) face gradient is not enough: the divergence amplifies it by 1/h."""
    rows4, diagnostics4 = _study(4)
    rows6, diagnostics6 = _study(6)
    order_46 = _order(rows4, rows6, 1/4, 1/6, 'lsq_degree2')
    assert order_46 < 1.2, order_46
    assert rows6['lsq_degree2'][0] > rows6['as_built'][0]
    # The reconstructed wall-face gradient is genuinely more accurate than the
    # as-built one -- the failure is in the amplification, not the fit.
    assert diagnostics6['lsq_degree2_wall_gradient_error'] < \
        diagnostics6['as_built_wall_gradient_error']


def test_divergence_amplification_law_at_the_wall():
    """``err_div ~ mu*|dg_face|/h`` holds to a factor of two across the family."""
    for n in (4, 6):
        rows, diagnostics = _study(n)
        predicted = diagnostics['predicted_wall_error_from_wall_gradient']
        ratio = rows['as_built'][0]/predicted
        assert 0.3 < ratio < 1.5, (n, ratio)
        predicted = diagnostics['predicted_wall_error_from_lsq2']
        ratio = rows['lsq_degree2'][0]/predicted
        assert 0.3 < ratio < 1.6, (n, ratio)