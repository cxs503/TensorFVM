"""Self-verification of the Richardson/GCI tooling (verification of the verifier).

The order estimator, the extrapolation and the uncertainty band used to qualify
discretisations are themselves tested against synthetic series whose order and
limit are known in closed form, against the legacy uniform-ratio helper in the
regime where that helper is valid, and against negative controls where a
confident answer would be wrong.
"""
import json
import math

import numpy as np
import pytest
import torch

from tensorfvm.annular import AnnularConfig, analytic_flow, solve
from tensorfvm.verification.core import observed_orders
from tensorfvm.verification.richardson import (MIN_RATIO, RefinementStudy,
                                               observed_order, power_law_series,
                                               reference_agreement,
                                               richardson_gci)

UNIFORM = (1/32, 1/16, 1/8)
NONUNIFORM = (1/32, 1/20, 1/8)


def _metrics(spacings, values):
    return [dict(spacing_m=h, metrics=[dict(name='e', value=v)])
            for h, v in zip(spacings, values)]


@pytest.mark.parametrize('spacings', [UNIFORM, NONUNIFORM])
@pytest.mark.parametrize('order', [1., 1.5, 2., 3.])
def test_synthetic_series_recovered_to_roundoff(spacings, order):
    """Pure C h^p data: the order and the extrapolated limit are exact."""
    values = power_law_series(1., .7, order, spacings)
    study = richardson_gci(*values, spacings, theory_order=order)
    assert isinstance(study, RefinementStudy)
    assert study.convergence == 'monotone' and study.asymptotic
    assert abs(study.order-order) < 1e-9
    assert abs(study.extrapolated-1.) < 1e-12
    assert study.gci > 0 and math.isfinite(study.gci)
    agreement = reference_agreement(study, 1.)
    assert agreement['band_covered']
    # With an exact two-term series the extrapolation is exact, so the GCI is a
    # strictly conservative band: the efficiency index collapses to round-off.
    assert agreement['efficiency_index'] < 1e-6


def test_uniform_ratio_agrees_with_legacy_helper_in_its_valid_regime():
    """On a zero-limit error metric the legacy helper and this tool coincide."""
    order = 2.
    values = power_law_series(1., .7, order, UNIFORM)
    errors = [v-1. for v in values]
    legacy = observed_orders(_metrics(UNIFORM, errors), 'e')[0]
    study = richardson_gci(*values, UNIFORM, theory_order=order)
    assert abs(study.order-legacy) < 1e-9
    assert observed_order(UNIFORM, errors) == pytest.approx(legacy, abs=1e-9)


def test_legacy_helper_cannot_read_a_raw_metric():
    """The legacy helper assumes a zero limit; a raw integral quantity breaks it."""
    order = 2.
    values = power_law_series(1., .7, order, UNIFORM)
    legacy = observed_orders(_metrics(UNIFORM, values), 'e')
    assert all(abs(p-order) > .05 for p in legacy)
    study = richardson_gci(*values, UNIFORM, theory_order=order)
    assert abs(study.order-order) < 1e-9


def test_inconsistent_series_is_flagged_while_legacy_stays_silent():
    """A sign-alternating series must not produce a confident order or band."""
    values = (.1, .08, .09)
    study = richardson_gci(*values, UNIFORM, theory_order=2.)
    assert study.convergence == 'inconsistent'
    assert not study.asymptotic
    assert math.isnan(study.order) and study.order_used == 2.
    assert study.safety_factor == 3.
    assert len(observed_orders(_metrics(UNIFORM, values), 'e')) == 2  # silent


def test_pre_asymptotic_study_is_detected_and_then_becomes_asymptotic():
    """A visible higher-order term biases the order; refinement restores it."""
    coarse = richardson_gci(*power_law_series(1., .5, 2., (.1, .2, .4),
                                              secondary=4., secondary_order=4.),
                            (.1, .2, .4), theory_order=2.)
    assert coarse.convergence == 'monotone' and not coarse.asymptotic
    assert abs(coarse.order-2.) > .5
    fine = richardson_gci(*power_law_series(1., .5, 2., (.00625, .0125, .025),
                                            secondary=4., secondary_order=4.),
                          (.00625, .0125, .025), theory_order=2.)
    assert fine.asymptotic and abs(fine.order-2.) < .05
    assert abs(fine.extrapolated-1.) < abs(coarse.extrapolated-1.)
    assert abs(fine.extrapolated-1.) < 1e-6
    assert reference_agreement(fine, 1.)['band_covered']


@pytest.mark.parametrize('spacings,values,message', [
    ((.1, .05, .025), (1., 2., 3.), 'strictly increasing'),
    ((.025, .025, .1), (1., 2., 3.), 'strictly increasing'),
    ((.025, .0252, .1), (1., 2., 3.), 'refinement ratios'),
    ((.025, .05, .1), (1., float('nan'), 3.), 'non-finite'),
    ((.025, .05, .1), (1., 1., 3.), 'no order signal'),
    ((.025, .05), (1., 2., 3.), 'three grids'),
    ((.025, .05, .1), (0., 1., 2.), 'non-zero fine-grid metric'),
])
def test_invalid_inputs_rejected(spacings, values, message):
    with pytest.raises(ValueError, match=message):
        richardson_gci(*values, spacings, theory_order=2.)


def test_order_bounds_and_safety_factor_contract():
    assert MIN_RATIO < 1.1
    values = power_law_series(1., .5, 2., UNIFORM)
    with pytest.raises(ValueError, match='safety factor'):
        richardson_gci(*values, UNIFORM, safety_factor=.5)
    with pytest.raises(ValueError, match='theory order'):
        richardson_gci(*values, UNIFORM, theory_order=0.)
    forced = richardson_gci(*values, UNIFORM, safety_factor=2.)
    observed = richardson_gci(*values, UNIFORM, theory_order=2.)
    assert forced.gci == pytest.approx(2.*observed.gci/1.25)


def test_reference_agreement_requires_nonzero_reference():
    study = richardson_gci(*power_law_series(1., .5, 2., UNIFORM), UNIFORM,
                           theory_order=2.)
    with pytest.raises(ValueError, match='zero reference'):
        reference_agreement(study, 0.)
    agreement = reference_agreement(study, 1.004)
    assert agreement['relative_error'] == pytest.approx(
        abs(study.extrapolated-1.004)/1.004)
    assert agreement['efficiency_index'] == pytest.approx(
        agreement['relative_error']/study.gci)
    assert json.dumps(study.as_dict(), allow_nan=False)


def test_annular_flow_rate_recovers_the_formal_second_order():
    """Application instance: production annular diffusion backend vs closed form."""
    torch.set_num_threads(1)
    spacings, flow = [], []
    reference = None
    for ntheta, nr in ((128, 32), (64, 16), (32, 8)):
        config = AnnularConfig(ntheta=ntheta, nr=nr)
        result = solve(config)
        fields = result['fields']
        velocity = np.asarray(fields['axial_velocity_m_s'], dtype=float)
        volume = np.asarray(fields['cell_volumes_m2'], dtype=float)
        reference = float(analytic_flow(type('C', (), dict(
            inner_radius_m=config.inner_radius_m,
            outer_radius_m=config.outer_radius_m,
            pressure_gradient_pa_m=config.pressure_gradient_pa_m,
            viscosity_pa_s=config.viscosity_pa_s))()))
        spacings.append(1./ntheta)
        flow.append(float((velocity.ravel()*volume.ravel()).sum()))
    study = richardson_gci(*flow, spacings, theory_order=2.)
    agreement = reference_agreement(study, reference)
    assert study.asymptotic and abs(study.order-2.) < .1
    assert agreement['band_covered']
    assert agreement['relative_error'] < abs(flow[0]-reference)/reference/10.
    # The coarsest grid already satisfies the 3% single-grid gate: a single grid
    # cannot distinguish it from the 4x refined solution, so the gate alone is
    # not a verification statement.
    assert abs(flow[-1]-reference)/reference < .03
    assert agreement['relative_error'] < 1e-4