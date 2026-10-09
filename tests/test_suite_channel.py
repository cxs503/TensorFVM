"""Cross-solver export preserves physical scaling and computed face fields."""
import json

import pytest

from tensorfvm.benchmark_suite_channel import run


def test_si_channel_exports_real_flux_and_shear(tmp_path):
    result = run(tmp_path, grids=(12,))
    case = result['cases'][0]
    fields = json.loads((tmp_path/'36x12/si-fields.json').read_text())
    assert result['moving_ice_backend_qualified'] is False
    assert fields['config']['density'] == 1000
    assert len(fields['u_faces_m_s'][0]) == 37
    assert case['mass_inlet_kg_s_per_m_depth'] == pytest.approx(.01)
    assert case['pressure_gradient'] == pytest.approx(-.15, rel=.03)
    assert case['developed_wall_force_on_fluid_n_per_m_depth'] < 0
    assert case['wall_shear_relative_linf_error'] < .03
    assert '36x12/si-fields.json' in result['artifact_sha256']


def test_unconverged_result_is_retained(tmp_path):
    result = run(tmp_path, grids=(12,), max_iterations=1)
    assert not result['passed']
    assert not result['cases'][0]['converged']
    assert (tmp_path/'36x12/si-fields.json').exists()
