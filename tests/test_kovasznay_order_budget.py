"""Error-budget invariants behind the CASE-FVM-KOVASZNAY-ORDER attribution (m16396).

The delivered round scored the extruded Kovasznay case with a single velocity L2
norm and got order 1.684 (FAIL).  ``scripts/audit_kovasznay_order_components.py``
splits that norm into the in-plane (u, v) budget and the spanwise w budget, whose
exact value is identically zero.  These tests pin the invariants that make the
split legitimate, so the attribution cannot silently rot:

* the two budgets add to the total in quadrature (they are an orthogonal
  decomposition of the same volume-weighted norm, not two different metrics),
* the analytic spanwise component is exactly zero (so the w budget is pure error),
* the extruded path does produce a non-zero spanwise velocity -- the artefact
  being attributed -- and it is small compared with the in-plane field.

A coarse grid (n = 8) is used: the test is about the decomposition, not accuracy.
"""

import numpy as np

from tensorfvm.verification.native_hybrid_ns import kovasznay_solver
from tensorfvm.native_manufactured3d import volume_weighted_error


def _solve(n=8):
    mesh, solver, exact = kovasznay_solver(n, 40., symmetry_extrusion=True,
                                           convection_order=2, coupled_linear_method='reused_lu')
    result = solver.solve(algorithm='coupled', outer_relaxation=.7, tolerance=1e-8,
                          max_outer_iterations=200)
    return (np.asarray(result['velocity']), np.asarray(exact['velocity']),
            np.asarray(mesh.cell_volumes))


def test_analytic_spanwise_component_is_zero():
    _, exact, _ = _solve()
    assert np.max(np.abs(exact[:, 2])) == 0.0


def _rel(error, reference, volumes):
    """Relative L2 with *one* common denominator, so budgets are orthogonal."""
    w = volumes.reshape((len(volumes),) + (1,) * (error.ndim - 1))
    return float(np.sqrt(np.sum(w * np.square(error)) / np.sum(w * np.square(reference))))


def test_budgets_close_in_quadrature():
    velocity, exact, volumes = _solve()
    total = _rel(velocity - exact, exact, volumes)
    inplane = _rel(velocity[:, :2] - exact[:, :2], exact, volumes)
    spanwise = _rel(velocity[:, 2:] - exact[:, 2:], exact, volumes)
    # the library metric agrees once the reference is the same full field
    assert abs(volume_weighted_error(velocity, exact, volumes) - total) < 1e-15
    assert abs(np.hypot(inplane, spanwise) - total) / total < 1e-12
    assert 0.0 < spanwise < total


def test_extruded_spanwise_artefact_is_present_but_subdominant():
    velocity, exact, _ = _solve()
    max_span = float(np.max(np.abs(velocity[:, 2])))
    assert max_span > 0.0, 'the artefact this audit attributes has vanished: re-run the attribution'
    assert max_span < 0.05
    assert max_span < np.max(np.abs(velocity[:, :2]))
