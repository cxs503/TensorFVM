import json
from pathlib import Path
import numpy as np
import pytest
from tensorfvm.suboff_validation import drag_comparison, cp_comparison, require_reynolds, require_physical_iterate


def cases():
    return json.loads((Path(__file__).resolve().parents[1] / 'benchmarks/suboff_aff1_validation.json').read_text())['cases']


def test_dimensional_total_force_and_frontal_area_agree_with_reference():
    case = cases()[0]; properties = case['operating_conditions']
    scale = properties['rho_kg_m3'] * properties['U_m_s']**2 * case['geometry']['L_m']**2
    result = drag_comparison([87.4 / scale, 0, 0], case)
    assert result['relative_error'] < 1e-14
    assert result['Cd_calculated'] == pytest.approx(.09312081297064881)
    assert not drag_comparison([1.03 * 87.4 / scale + 1e-12, 0, 0], case)['nominal_error_below_3_percent']


def test_cp_metric_retains_unmodified_secondary_reference_and_uncertainty():
    points = cases()[1]['reference']['corrected_candidate_points']
    calculated = [row['Cp_corrected'] for row in points]
    result = cp_comparison(calculated, points)
    assert result['nominal_relative_L2_error'] == 0
    assert result['correction_sensitivity_relative_L2_error'] > 0
    assert not result['full_primary_reference_verified']
    assert result['experimental_Cp_95_absolute_uncertainty'] == .015
    with pytest.raises(ValueError):cp_comparison(calculated[:-1], points)


def test_separate_drag_and_cp_conditions_cannot_be_mixed():
    drag, cp = cases()[:2]
    require_reynolds(drag['operating_conditions']['Re_L'], drag['operating_conditions']['Re_L'])
    with pytest.raises(ValueError):require_reynolds(drag['operating_conditions']['Re_L'], cp['operating_conditions']['Re_L'])


def test_failed_initialization_is_not_a_physical_benchmark_result():
    for result in ({}, {'solution_state':'initialization_only', 'completed_momentum_iterations':0},
                   {'solution_state':'coupled_iterate', 'completed_momentum_iterations':0}):
        with pytest.raises(ValueError,match='initialization cannot be compared'):
            require_physical_iterate(result)
    require_physical_iterate({'solution_state':'momentum_only', 'completed_momentum_iterations':1})
