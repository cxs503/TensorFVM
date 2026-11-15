"""Poisson grid-convergence gate on tetrahedral meshes (FEAT-UNSTR3D-MESH).

The gate is a measured order, not an assertion of intent: the linear
least-squares correction is measured in the same test and must stay *below*
second order, which proves the gate can fail.
"""
import numpy as np
import pytest

from tensorfvm.unstructured_fvm3d import (UnstructuredFVM3D, discrete_l2_error,
                                          observed_order)
from tensorfvm.unstructured_mesh3d import cube_tet_msh, tet_mesh_from_msh

PI = np.pi
LADDER = (2, 4, 8)


def exact(point):
    return (np.sin(PI * point[:, 0]) * np.sin(PI * point[:, 1])
            * np.sin(PI * point[:, 2]))


def source(point):
    return -3.0 * PI ** 2 * exact(point)


def solve(n, *, warp=0.0, face_gradient_method="quadratic", gradient_method="linear"):
    mesh = tet_mesh_from_msh(text=cube_tet_msh(n, warp=warp))
    operator = UnstructuredFVM3D(mesh)
    boundary = exact(operator.face_centers[operator.boundary])
    result = operator.poisson(source(np.asarray(mesh.cell_centers)), boundary,
                              gradient_method=gradient_method,
                              face_gradient_method=face_gradient_method,
                              tolerance=1e-13)
    return discrete_l2_error(mesh, result["values"], exact), result


def measured_orders(warp):
    errors = [solve(n, warp=warp)[0] for n in LADDER]
    return errors, [observed_order(errors[i], errors[i + 1], 2.0)
                    for i in range(len(errors) - 1)]


def test_kexact_correction_reaches_second_order():
    """Gate: observed order >= 2 on the refined pair of the tet ladder."""
    errors, orders = measured_orders(0.0)
    assert orders[-1] >= 2.0, f"order on the refined pair is {orders[-1]:.2f}: {orders}"


def test_warped_ladder_order_is_reported_honestly():
    """Secondary case: the warped mesh is pre-asymptotic (1.91 < 2.0 measured)."""
    errors, orders = measured_orders(0.25)
    assert 1.7 <= orders[-1] <= 2.6, (orders, errors)
    assert len(errors) == len(LADDER)


def test_linear_least_squares_correction_is_below_second_order():
    errors = [solve(n, face_gradient_method="interpolate")[0] for n in LADDER]
    order = observed_order(errors[-2], errors[-1], 2.0)
    assert order < 2.0, f"gate is not discriminating: linear correction reached {order:.2f}"


def test_affine_field_is_reproduced_exactly():
    mesh = tet_mesh_from_msh(text=cube_tet_msh(3, warp=0.25))
    operator = UnstructuredFVM3D(mesh)
    gradient = np.array([1.0, -2.0, 0.5])
    affine = lambda point: 0.3 + point @ gradient
    result = operator.poisson(np.zeros(len(mesh.cells)),
                              affine(operator.face_centers[operator.boundary]),
                              face_gradient_method="quadratic")
    values = result["values"]
    assert np.max(np.abs(values - affine(np.asarray(mesh.cell_centers)))) < 1e-9
