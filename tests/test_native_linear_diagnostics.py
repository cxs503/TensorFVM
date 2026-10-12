import numpy as np
from scipy import sparse
from tensorfvm.native_linear_diagnostics import linear_system_diagnostics


def test_scaled_diagonal_condition_and_original_residual():
    matrix = sparse.diags([1e-6, 1., 1e6]).tocsc()
    rhs = np.ones(3)
    value = np.array([1e6, 1., 1e-6])
    result = linear_system_diagnostics(matrix, rhs, value, estimate_condition=True)
    assert result['condition_one_norm_estimate'] == 1e12
    assert result['normwise_backward_error'] == 0.
    assert result['relative_rhs_residual'] == 0.
    assert result['row_maximum_span']['minimum_positive'] == 1e-6
    wrong = value.copy(); wrong[1] += 1.
    result = linear_system_diagnostics(matrix, rhs, wrong)
    assert result['relative_rhs_residual'] > .5
    assert result['normwise_backward_error'] < 1e-11
    # Small backward error alone must not be substituted for the rhs gate.
    np.testing.assert_array_equal(value, [1e6, 1., 1e-6])


def test_nonfinite_diagnostic_is_json_serializable_and_preserves_values():
    import json
    value = np.array([np.nan, 1.])
    result = linear_system_diagnostics(sparse.eye(2), np.ones(2), value)
    assert not result['finite_solution']
    assert result['relative_rhs_residual'] is None
    json.dumps(result, allow_nan=False)
    assert np.isnan(value[0])
