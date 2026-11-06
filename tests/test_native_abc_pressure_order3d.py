"""Attribution of the sub-second-order ABC pressure convergence (verification debt).

Pins the finding that the ~1.5 pressure order of the collocated 3D ABC case is a
property of the pressure-continuity coupling, not of the convection
discretisation: the linear Stokes limit and the full nonlinear NS case show the
same pressure order, the velocity stays second order, and the discrete
continuity residual is machine zero in every run.
"""
import numpy as np
import pytest

from tensorfvm.verification.native_abc_pressure_order import (observed_order,
                                                              pressure_order_study,
                                                              run_case)


def test_observed_order_rejects_nonpositive_errors():
    with pytest.raises(ValueError):
        observed_order(0.0, 1.0)
    with pytest.raises(ValueError):
        observed_order(1.0, -1.0)
    np.testing.assert_allclose(observed_order(4.0, 1.0), 2.0)
    np.testing.assert_allclose(observed_order(8.0, 1.0, ratio=2.0), 3.0)


def test_stokes_limit_pressure_order_matches_full_ns():
    """Convection is not the cause: Stokes and NS pressure orders agree."""
    study = pressure_order_study(grids=(3, 6, 12))
    s, f = study['stokes'], study['full_ns']
    # Velocity is second order in both limits (finest refinement).
    assert s[-1]['velocity_order'] > 1.7
    assert f[-1]['velocity_order'] > 1.7
    # Pressure error decreases monotonically in both limits.
    for rows in (s, f):
        errors = [row['pressure'] for row in rows]
        assert all(a > b > 0 for a, b in zip(errors, errors[1:]))
    # Finest-grid pressure passes the 3% gate in both limits.
    assert s[-1]['pressure'] < 3.0
    assert f[-1]['pressure'] < 3.0
    # The attribution evidence: the two limits agree on the pressure order.
    assert study['order_agreement'] < 0.25, study
    # The order reduction is distributed, not a boundary layer: the interior
    # (non boundary-adjacent) cells converge at the same reduced order.
    assert s[-1]['interior_pressure_order'] > 1.3
    # Discrete continuity is enforced to machine precision.
    for rows in (s, f):
        for row in rows:
            assert row['continuity_residual'] < 1e-10


def test_single_case_contract():
    row = run_case(3, convection=False)
    assert row['converged'] and row['cells'] == 27
    assert row['velocity'] < 1.0 and row['pressure'] > 0.0
    assert row['continuity_residual'] < 1e-10
