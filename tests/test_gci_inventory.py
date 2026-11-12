"""Gate tests for the cross-implementation GCI audit (``gci_inventory``).

The audit grades every grid-convergence estimator in the tree by one question:
does the band it reports cover the actual fine-grid error of a series whose
limit is known, and does it refuse to look confident where the series carries no
order signal?  These tests freeze the two states of the tree:

* the two public estimators that carried no order guard
  (``immersed.study.richardson``, ``rans_ship_resistance...grid_convergence_index``)
  now cover the near-cancellation regime and never publish a negative exponent;
* ``verification.ladder.gci_three_grid`` and
  ``verification.annular_convergence.observed_order`` are still overconfident in
  that regime -- a recorded, non-blocking limitation.  The assertions below fail
  if either behaviour changes, so a fix cannot land silently and a regression
  cannot hide.
"""
import math

import pytest

from tensorfvm.verification import gci_inventory as audit

CANONICAL = 'verification.richardson.richardson_gci'
IMMERSED = 'immersed.study.richardson'
SHIP = 'rans_ship_resistance.speed_sweep.grid_convergence_index'
LADDER = 'verification.ladder.gci_three_grid'
ANNULAR = 'verification.annular_convergence.observed_order'


def status(name, regime):
    rows = [r for r in audit.run()['rows']
            if r['implementation'] == name and r['regime'] == regime]
    assert len(rows) == 1
    return rows[0]


@pytest.mark.parametrize('name', [CANONICAL, IMMERSED, SHIP, LADDER, ANNULAR])
@pytest.mark.parametrize('regime', ['asymptotic_p1', 'asymptotic_p2'])
def test_asymptotic_series_are_covered_by_every_estimator(name, regime):
    row = status(name, regime)
    assert row['covered'], row
    assert math.isfinite(row['gci']) and row['gci'] > 0.0


def test_canonical_recovers_known_order_and_limit():
    for regime in audit.regimes()[:2]:
        order = float(regime.name.rsplit('p', 1)[1])
        adapter = audit.IMPLEMENTATIONS[CANONICAL]
        out = adapter(*regime.values)
        assert out['order'] == pytest.approx(order, rel=1e-9)
        assert out['extrapolated'] == pytest.approx(regime.exact, rel=1e-9)


@pytest.mark.parametrize('name', [CANONICAL, IMMERSED, SHIP])
def test_guarded_estimators_do_not_publish_a_confident_band_under_near_cancellation(name):
    """``(v3-v2)/(v2-v1)=1e3`` fakes an order of 10; the band must not shrink to it."""
    row = status(name, 'near_cancellation')
    assert row['status'] != 'overconfident', row
    assert row['covered'] or math.isinf(row['gci']) or row['gci'] > 1e-6


def test_guarded_estimators_never_return_a_negative_observed_order():
    """Non-monotone increment signs used to give ``p = log|q|/log r < 0``."""
    from tensorfvm.immersed.study import richardson
    from tensorfvm.rans_ship_resistance.speed_sweep import grid_convergence_index

    out = richardson(1.0, 0.9, 1.05)          # alternating increments
    assert not (out['observed_order'] < 0.0)
    gci = grid_convergence_index(1.0, 0.9, 1.05)
    assert not (gci['observed_order'] < 0.0)
    assert gci['monotonic'] is False
    assert gci['convergence'] in ('inconsistent', 'order-implausible', 'monotone')


def test_ship_resistance_gci_matches_the_canonical_rule_on_a_clean_series():
    from tensorfvm.rans_ship_resistance.speed_sweep import grid_convergence_index
    from tensorfvm.verification.richardson import richardson_gci

    exact, amplitude = 0.003, 1e-5
    values = tuple(exact + amplitude * h ** 2 for h in (0.25, 0.5, 1.0))
    fine, medium, coarse = values
    got = grid_convergence_index(fine, medium, coarse, ratio=2.0)
    reference = richardson_gci(fine, medium, coarse, (0.25, 0.5, 1.0))
    assert got['extrapolated'] == pytest.approx(reference.extrapolated, rel=1e-12)
    assert got['gci_fine'] == pytest.approx(reference.gci, rel=1e-12)
    assert got['observed_order'] == pytest.approx(2.0, abs=1e-6)


def test_immersed_richardson_keeps_its_legacy_keys():
    from tensorfvm.immersed.study import richardson

    out = richardson(1.1, 1.2, 1.5)
    for key in ('value_fine', 'value_medium', 'value_coarse', 'observed_order',
                'gci_fine_pct', 'extrapolated', 'ratio_eps32_eps21', 'convergence'):
        assert key in out
    out = richardson(1.0, 1.0, 1.0)                     # no order signal at all
    assert math.isnan(out['observed_order'])
    assert out['convergence'] == 'not-observable'


@pytest.mark.parametrize('name', [LADDER, ANNULAR])
def test_recorded_non_blocking_limitation_of_the_unguarded_estimators(name):
    """Documented limitation: these two still shrink the band under cancellation."""
    row = status(name, 'near_cancellation')
    assert row['status'] == 'overconfident', row
    assert row['gci'] < 1e-6 < abs(row['detail'].get('x', 0.0)) or row['gci'] < 1e-6


def test_audit_reports_every_implementation_for_every_regime():
    payload = audit.run()
    assert set(payload['summary']) == set(audit.IMPLEMENTATIONS)
    assert len(payload['rows']) == len(audit.IMPLEMENTATIONS) * len(audit.regimes())
    text = audit.markdown(payload)
    assert text.count('|') > 6 * len(audit.regimes())