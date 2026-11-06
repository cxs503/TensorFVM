"""Passivity verdict for the native 3D viscous operator.

The strict xfail in ``test_native_energy_small.py`` stays red: no member of the
deferred-correction family satisfies the discrete dissipation inequality on the
two-cell counterexample.  These tests pin the quantified verdict instead of the
wish: where passivity holds (angle boundary), that it cannot be restored by
re-weighting, and what the strictly passive alternative costs in consistency.
"""

from tensorfvm.verification.native_energy import (affine_consistency, correction_family_scan,
                                                 passivity_audit, passivity_boundary,
                                                 random_shear_survey, two_point_control,
                                                 two_tetrahedra)

QUALITY_GATE_DEGREES = 70.  # tensorfvm.mesh_quality.quality_gate_passed


def test_default_operator_is_passive_inside_the_quality_gate():
    # Nonorthogonality up to ~61 deg, well inside the 70 deg gate.
    for skew in (0., .25, .375, .5):
        mesh = two_tetrahedra(skew)
        audit = passivity_audit(mesh, 'over-relaxed')
        assert audit['maximum_nonorthogonality_degrees'] < QUALITY_GATE_DEGREES
        assert audit['passive'], (skew, audit['vector'])
        assert audit['vector']['creation_rate'] == 0.
        assert audit['vector']['lambda_min'] > 0.  # strictly dissipative
        assert audit['scalar']['creation_rate'] == 0.


def test_passivity_boundary_is_far_outside_the_quality_gate():
    for correction in ('over-relaxed', 'minimum'):
        boundary = passivity_boundary(correction)
        assert boundary['creation_skew'] > boundary['passive_skew']
        # The gate is 70 deg; the failure starts ~14-16 deg beyond it.
        assert boundary['passive_degrees'] > QUALITY_GATE_DEGREES + 10
        assert 80. < boundary['passive_degrees'] < 90.
        assert abs(boundary['creation_degrees']-boundary['passive_degrees']) < 1e-3
    assert (passivity_boundary('over-relaxed')['passive_degrees']
            > passivity_boundary('minimum')['passive_degrees'])


def test_no_correction_scale_restores_passivity_on_the_counterexample():
    scan = correction_family_scan(two_tetrahedra())
    assert not scan['any_passive']
    # gamma -> 0 (no correction) is already creating energy: the interpolated
    # cell gradient, not the correction weight, breaks the adjointness.
    assert scan['rows'][0]['gamma'] == 0.
    assert min(row['vector_lambda_max'] for row in scan['rows']) > 1.
    assert all(row['vector_lambda_max'] > 0 for row in scan['rows'])


def test_strictly_passive_two_point_operator_is_not_affine_consistent():
    mesh = two_tetrahedra(1.)
    control = two_point_control(mesh)
    assert control['passive']
    assert control['spectrum']['lambda_min'] > 0
    # ... but the face flux projects the normal derivative on d instead of s.
    assert control['affine_flux_max_relative_error'] > 1.
    for correction in ('over-relaxed', 'minimum', 'none', 0., 3.):
        assert affine_consistency(mesh, correction)['maximum_relative_error'] < 1e-12


def test_random_sheared_pairs_stay_passive_for_the_shipped_default():
    survey = random_shear_survey(levels=(.2, .4, .6), samples=10)
    assert survey['samples'] >= 24
    assert survey['over_relaxed_failures'] == 0
    assert survey['over_relaxed_max_creation_rate'] == 0.
    assert survey['max_nonorthogonality_degrees'] < passivity_boundary('over-relaxed')['creation_degrees']


def test_creation_grows_with_nonorthogonality_past_the_boundary():
    creation = [passivity_audit(two_tetrahedra(skew), 'over-relaxed')['vector']['creation_rate']
                for skew in (1., 1.125, 1.25)]
    assert all(value > 0 for value in creation)
    assert creation[0] < creation[1] < creation[2]
    # Over-relaxation amplifies the defect far more than the minimum correction.
    minimum = [passivity_audit(two_tetrahedra(skew), 'minimum')['vector']['creation_rate']
               for skew in (1., 1.125, 1.25)]
    assert creation[-1]/minimum[-1] > 50.


def test_energy_identity_survives_each_correction_choice():
    from tensorfvm.verification.native_energy import energy_diagnostic
    mesh = two_tetrahedra()
    for correction in ('over-relaxed', 'minimum', 'none'):
        result = energy_diagnostic(mesh, correction=correction)
        assert result['energy_balance_error'] < 1e-12
        assert result['passed'] == (result['kinetic_energy_rate'] <= 1e-12)
    assert not energy_diagnostic(mesh)['passed']