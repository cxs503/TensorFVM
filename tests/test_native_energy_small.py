import numpy as np
import pytest

from tensorfvm.verification.native_energy import two_tetrahedra, energy_diagnostic


def test_two_cell_control_dissipates_and_stress_balance_agrees():
    result = energy_diagnostic(two_tetrahedra(0.))
    assert result['passed']
    assert result['symmetric_minimum_eigenvalue'] > 0


def test_two_cell_skew_geometry_and_independent_stress_balance():
    result = energy_diagnostic(two_tetrahedra())
    np.testing.assert_allclose(result['cell_volumes'], [1/24, 1/3], atol=1e-14)
    assert result['maximum_nonorthogonality_degrees'] < 90
    assert result['energy_balance_error'] < 1e-12


@pytest.mark.xfail(strict=True, reason='Open defect: skew two-cell viscous operator creates kinetic energy')
def test_two_cell_skew_viscosity_must_dissipate():
    # Keep the physical requirement red until the operator is corrected.
    # XPASS must fail CI so this marker is removed when the defect is fixed.
    assert energy_diagnostic(two_tetrahedra())['kinetic_energy_rate'] <= 1e-12
