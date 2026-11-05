"""Self-verification of the N-level (N >= 3) grid-convergence estimator.

Same philosophy as ``tests/test_richardson_gci.py``: the estimator is checked
against synthetic series whose order and limit are known in closed form, and
against negative controls where a confident answer would be wrong.  The
backwards-compatibility test is the important one: for N == 3 the new estimate
must reproduce :func:`richardson_gci` to solver tolerance, so adopting it cannot
move a number that is already published.
"""
import math

import numpy as np
import pytest

from tensorfvm.verification.multi_grid import (MultiGridStudy, multi_grid_order,
                                               per_level_residuals)
from tensorfvm.verification.richardson import (RefinementStudy, observed_order,
                                               power_law_series,
                                               reference_agreement,
                                               richardson_gci)

UNIFORM = (1/32, 1/16, 1/8)
NONUNIFORM = (1/32, 1/20, 1/8)


def _ladder(n, coarse=1/16):
    """``h = coarse/2**k`` for k = 0..n-1, ordered fine -> coarse."""
    return tuple(sorted(coarse/2.**k for k in range(n)))


@pytest.mark.parametrize('spacings', [UNIFORM, NONUNIFORM])
@pytest.mark.parametrize('order', [1., 1.5, 2., 3.])
def test_three_levels_reproduce_the_three_grid_routine(spacings, order):
    """N == 3: the least-squares fit is an interpolation and must agree exactly."""
    values = power_law_series(1., .7, order, spacings)
    ref = richardson_gci(*values, spacings, theory_order=order)
    study = multi_grid_order(spacings, values, theory_order=order)
    assert isinstance(ref, RefinementStudy) and isinstance(study, MultiGridStudy)
    assert study.levels == 3
    assert abs(study.order-ref.order) < 1e-6
    assert abs(study.extrapolated-ref.extrapolated) <= 1e-9*abs(ref.extrapolated)
    assert abs(study.extrapolated_triple-ref.extrapolated) <= 1e-12*abs(ref.extrapolated)
    assert abs(study.gci-ref.gci) <= 1e-10*ref.gci
    assert study.convergence == ref.convergence == 'monotone'
    assert study.asymptotic is ref.asymptotic is True


@pytest.mark.parametrize('levels', [4, 5, 6, 7])
@pytest.mark.parametrize('order', [1., 2., 3.])
def test_extra_levels_recover_a_pure_power_law(levels, order):
    """A pure ``L + C h^p`` series is recovered from all N levels."""
    spacings = _ladder(levels, coarse=1/32)
    values = power_law_series(2.5, 1.3, order, spacings)
    study = multi_grid_order(spacings, values, theory_order=order)
    assert study.levels == levels
    assert study.order_dof == levels-3
    assert abs(study.order-order) < 1e-8
    assert abs(study.extrapolated-2.5) <= 1e-9*2.5
    assert study.rms_residual < 1e-12
    assert study.convergence == 'monotone' and study.asymptotic
    # The classical triple over the finest three levels agrees as well.
    assert abs(study.extrapolated_triple-2.5) <= 1e-9*2.5


def test_order_uncertainty_is_withheld_at_three_levels():
    """Three points and three unknowns: the order standard error is not identifiable."""
    spacings = _ladder(3)
    values = power_law_series(1., .5, 2., spacings)
    three = multi_grid_order(spacings, values, theory_order=2.)
    assert three.order_dof == 0
    assert math.isnan(three.order_stderr)
    assert not math.isnan(multi_grid_order(_ladder(4), power_law_series(
        1., .5, 2., _ladder(4)), theory_order=2.).order_stderr)


def test_two_term_bias_is_visible_in_the_order_and_the_residual():
    """Documented trap: an active second term inflates the N-level order.

    The fit weights all levels equally, so a ``C2 h^(p+s)`` term that only
    matters on the coarse levels pulls the single-term model away from ``p``.
    The bias is not silent: it shows up in the order and in the residual, which
    is exactly why the estimator also reports them.
    """
    spacings = _ladder(5, coarse=1/16)
    p, s, alpha, primary = 2., 2, .3, .5
    secondary = alpha*primary*spacings[0]**p/spacings[0]**(p+s)
    values = power_law_series(1., primary, p, spacings,
                              secondary=secondary, secondary_order=p+s)
    study = multi_grid_order(spacings, values, theory_order=p)
    assert study.order > p+1.                       # biased high, not hidden
    assert study.residual_relative > 1e-6            # misfit is measurable
    assert abs(study.extrapolated-1.) < 1e-2         # the limit is still useful
    res = per_level_residuals(study)
    assert res[0] < 0. < res[-1]                      # structured, not random misfit


def test_residuals_of_an_inconsistent_study_are_not_available():
    study = multi_grid_order(_ladder(5), (1., 2., 3., 2.5, 1.), theory_order=2.)
    with pytest.raises(ValueError):
        per_level_residuals(study)


@pytest.mark.parametrize('values', [
    (1., 2., 3., 2.5, 1.),        # rises then falls
    (1., 1., 1.5, 2.),            # duplicated levels: no order signal
])
def test_non_monotone_series_falls_back_conservatively(values):
    spacings = _ladder(len(values))
    study = multi_grid_order(spacings, values, theory_order=2.)
    assert study.convergence == 'inconsistent'
    assert math.isnan(study.order) and math.isnan(study.order_stderr)
    assert study.safety_factor == 3.0
    assert study.asymptotic is False


def test_residuals_are_orthogonal_to_the_fitted_basis():
    """A least-squares fit leaves residuals orthogonal to span{1, h^p}."""
    spacings = _ladder(5, coarse=1/4)
    rng = np.random.default_rng(3)
    values = np.array(power_law_series(1., 5., 1., spacings))
    values = values + 2e-3*rng.standard_normal(values.size)
    study = multi_grid_order(spacings, tuple(values), theory_order=1.)
    assert study.convergence == 'monotone'
    res = np.asarray(per_level_residuals(study))
    basis = np.power(np.asarray(spacings), study.order)
    assert abs(res.sum()) <= 1e-10*abs(values).max()*values.size
    assert abs(res @ basis) <= 1e-10*abs(values).max()*(basis @ basis)


def test_reference_agreement_accepts_a_multi_grid_study():
    """The band bookkeeping of the three-grid routine applies unchanged."""
    spacings = _ladder(4)
    values = power_law_series(1., .5, 2., spacings)
    study = multi_grid_order(spacings, values, theory_order=2.)
    inside = study.extrapolated + .5*study.gci*abs(study.extrapolated)
    agree = reference_agreement(study, inside)
    assert agree['band_covered'] is True
    assert agree['efficiency_index'] <= 1.
    assert reference_agreement(study, inside + 10*study.gci)['band_covered'] is False


def test_theory_order_mismatch_flags_non_asymptotic():
    spacings = _ladder(4)
    values = power_law_series(1., .5, 2., spacings)
    assert multi_grid_order(spacings, values, theory_order=2.).asymptotic is True
    assert multi_grid_order(spacings, values, theory_order=1.).asymptotic is False


@pytest.mark.parametrize('spacings,values,message', [
    ((1/32, 1/16), (1., .99), 'at least three'),
    ((1/8, 1/16, 1/32), (1., .9, .8), 'increasing'),
    ((1/32, 1/32*(1+1e-6), 1/8), (1., .9, .8), 'refinement ratios'),
    ((1/32, 1/16, 1/8), (0., .1, .7), 'non-zero fine-grid'),
    ((1/32, 1/16, 1/8), (1., .9, .8), 'safety factor'),
    ((1/32, 1/16, 1/8), (1., .9, .8), 'theory order'),
])
def test_invalid_input_is_rejected(spacings, values, message):
    kwargs = {}
    if message == 'safety factor':
        kwargs['safety_factor'] = .5
    if message == 'theory order':
        kwargs['theory_order'] = -1.
    with pytest.raises(ValueError) as exc:
        multi_grid_order(spacings, values, **kwargs)
    assert message in str(exc.value)


def test_observed_order_agrees_on_the_finest_triple():
    """Sanity cross-check against the standalone two-difference solver."""
    spacings = _ladder(5, coarse=1/8)
    values = power_law_series(1., .5, 2., spacings)
    triple = observed_order(spacings[:3], values[:3])
    assert abs(triple-2.) < 1e-9
    study = multi_grid_order(spacings, values, theory_order=2.)
    assert abs(study.order-triple) < 1e-6

def test_fallback_reason_names_the_rule_that_fired():
    """The qualifier is derived from the fall-back path, in the RefinementStudy vocabulary."""
    spacings = _ladder(5)
    converged = multi_grid_order(spacings, power_law_series(1., .5, 2., spacings), theory_order=2.)
    assert converged.fallback_reason == 'none'
    assert reference_agreement(converged, converged.extrapolated)['fallback_reason'] == 'none'

    broken = multi_grid_order(spacings, (1., 2., 3., 2.5, 1.), theory_order=2.)
    assert broken.convergence == 'inconsistent' and broken.monotone is False
    assert broken.fallback_reason == 'non-monotone'
    agree = reference_agreement(broken, broken.extrapolated)
    assert agree['fallback_reason'] == 'non-monotone'
    assert agree['band_covered'] is True          # the band, not the order, is what is claimed


def test_infeasible_fit_of_a_monotone_ladder_is_an_implausible_order(monkeypatch):
    """Defensive branch: a fall-back must always name its rule.

    No real monotone ladder reached this branch in a 400-ladder fuzz (the order
    minimiser always returns a finite fit when the values are monotone), so the
    invariant ``fallback => fallback_reason != 'none'`` is pinned white-box.
    """
    from tensorfvm.verification import multi_grid as mg

    def unusable(h, v, p):
        return float('nan'), float('nan'), np.full(np.shape(h), np.nan), math.inf

    monkeypatch.setattr(mg, '_linear_fit', unusable)
    spacings = _ladder(4)
    study = mg.multi_grid_order(spacings, power_law_series(1., .5, 2., spacings), theory_order=2.)
    assert study.monotone is True and study.convergence == 'inconsistent'
    assert study.fallback_reason == 'implausible-order'
    assert study.safety_factor == 3.0 and study.order_used == 2.
    assert reference_agreement(study, study.extrapolated)['fallback_reason'] == 'implausible-order'
