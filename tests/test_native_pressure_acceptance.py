import pytest
from tensorfvm.native_pressure_acceptance import regional_pressure_acceptance


def measured(error):
    return dict(cells=16, pressure_relative_L2_global_gauge_aligned=error,
                maximum_error_over_rho_U2=.001)


def test_global_pass_and_small_fixed_scale_do_not_hide_local_failure():
    regions={'all':measured(.008617079249215), 'wall':measured(.20661605772724)}
    result=regional_pressure_acceptance(regions,['all','wall'],equation_audit_passed=True)
    assert result['checks'][0]['passed']
    assert not result['checks'][1]['passed']
    assert not result['manufactured_regional_pressure_qualified']
    assert result['equation_and_global_audit_passed']


@pytest.mark.parametrize('error',[.03, float('nan'), float('inf'), None, -.01, False])
def test_threshold_and_invalid_errors_fail_closed(error):
    result=regional_pressure_acceptance({'wall':measured(error)},['wall'],equation_audit_passed=True)
    assert not result['manufactured_regional_pressure_qualified']


def test_missing_regions_and_failed_equation_audit_cannot_qualify():
    assert not regional_pressure_acceptance({},['wall'],equation_audit_passed=True)['regional_pressure_passed']
    result=regional_pressure_acceptance({'wall':measured(.01)},['wall'],equation_audit_passed=False)
    assert result['regional_pressure_passed'] and not result['manufactured_regional_pressure_qualified']
    good=regional_pressure_acceptance({'wall':measured(.01)},['wall'],equation_audit_passed=True)
    assert good['manufactured_regional_pressure_qualified'] and not good['experimental_accuracy_qualified']
