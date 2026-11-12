"""Machine checks for the resumable Kovasznay order ladder (``CASE-FVM-KOVASZNAY-ORDER``).

The ladder itself runs on the compute fleet one grid per process; these tests
lock (a) the arithmetic of the ladder (two-level orders, least-squares slope,
the pre-declared gate logic) on synthetic data where the answer is known, and
(b) the schema/closure properties of a real level record at the cheapest grid
(n = 8), where the tied symmetry closure must give ``w == 0`` exactly.
"""

import json

import pytest

from tensorfvm.verification import kovasznay_order_asymptotic as ladder


def _record(n, factor, closure='zero_gradient', method='reused_lu',
            spanwise=0., max_w=0., tie=0., converged=True):
    """Synthetic ladder record whose error decays as ``factor / n**2``."""
    return dict(
        n=n, h=1. / n, closure=closure, linear_method=method,
        total_l2=factor / n ** 2, inplane_l2=factor / n ** 2,
        pressure_mean_free_l2=factor / n ** 2, pressure_l2=factor / n ** 2,
        spanwise_l2=spanwise, max_abs_w=max_w, tie_residual=tie,
        converged=converged,
    )


def test_pair_orders_recover_second_order():
    rows = ladder.pair_orders([_record(n, 1.) for n in (16, 32, 64)])
    assert [r['pair'] for r in rows] == ['16->32', '32->64']
    for row in rows:
        assert row['total'] == pytest.approx(2., abs=1e-12)
        assert row['inplane'] == pytest.approx(2., abs=1e-12)


def test_pair_orders_respect_unequal_grid_ratio():
    """A 3:1 grid ratio must not be scored as if it were 2:1."""
    coarse, fine = _record(8, 1.), _record(24, 1.)
    order = ladder.observed_order(coarse['total_l2'], fine['total_l2'],
                                  coarse['h'] / fine['h'])
    assert order == pytest.approx(2., abs=1e-12)


def test_fit_order_slope_and_unusable_levels():
    fit = ladder.fit_order([_record(n, 3.) for n in (16, 32, 64)], 'total_l2')
    assert fit['slope'] == pytest.approx(2., abs=1e-10)
    assert fit['levels'] == 3 and fit['max_log_residual'] < 1e-12
    # a zero error (exact) cannot enter a log fit and must be dropped, not zeroed
    records = [_record(16, 1.), dict(_record(32, 1.), total_l2=0.)]
    assert ladder.fit_order(records, 'total_l2') is None


def test_gate_flips_with_the_measured_order():
    good = ladder.evaluate([_record(n, 1.) for n in (16, 32)])
    assert good['anchor_closed'] is True
    assert [c['verdict'] for c in good['checks'] if c['id'] == 'C3'] == ['PASS']
    # first order plus a first-order spurious spanwise component -> honest FAIL
    first = [_record(16, 1.), _record(32, 1.), ]
    first[0]['total_l2'] = 1. / 16          # first order: each refinement halves
    first[1]['total_l2'] = 0.5 / 16
    bad = ladder.evaluate(first)
    assert bad['anchor_closed'] is False
    assert [c['verdict'] for c in bad['checks'] if c['id'] == 'C3'] == ['FAIL']
    assert [c['verdict'] for c in bad['checks'] if c['id'] == 'C5'] == ['FAIL']


def test_nonzero_spanwise_fails_closure_criterion():
    records = [_record(n, 1., spanwise=1e-3) for n in (16, 32)]
    verdict = ladder.evaluate(records)
    assert [c['verdict'] for c in verdict['checks'] if c['id'] == 'C2'] == ['FAIL']
    assert verdict['anchor_closed'] is False


def test_non_converged_level_is_not_a_pass():
    records = [_record(16, 1.), _record(32, 1., converged=False)]
    verdict = ladder.evaluate(records)
    assert [c['verdict'] for c in verdict['checks'] if c['id'] == 'C6'] == ['FAIL']
    assert verdict['anchor_closed'] is False


def test_linear_configuration_independence_is_reported():
    base = _record(32, 1.)
    other = dict(_record(32, 1., method='direct'), total_l2=base['total_l2'] * (1 + 1e-3))
    verdict = ladder.evaluate([_record(16, 1.), base], linear_records=[other])
    check = [c for c in verdict['checks'] if c['id'] == 'C8'][0]
    assert check['value'] == pytest.approx(1e-3, rel=1e-2)
    assert verdict['anchor_closed'] is True  # informational rows never gate
    assert check['verdict'] == 'FAIL'  # informational row, but the value is visible


def test_level_record_roundtrip_and_family_filter(tmp_path):
    ladder.write_level(tmp_path, _record(16, 1.))
    ladder.write_level(tmp_path, _record(32, 1.))
    ladder.write_level(tmp_path, _record(32, 1., method='direct'))
    ladder.write_level(tmp_path, _record(32, 1., closure='symmetry'))
    reused = ladder.load_records(tmp_path)
    assert [r['n'] for r in reused] == [16, 32]
    assert {r['linear_method'] for r in reused} == {'reused_lu'}
    payload = ladder.merge(tmp_path)
    assert payload['levels'] == [16, 32]
    summary = json.loads((tmp_path / 'summary.json').read_text())
    assert summary['anchor_closed'] is payload['anchor_closed'] is True
    # the symmetry control record must not contaminate the graded ladder
    assert all(r['closure'] == 'zero_gradient' for r in summary['records'])


def test_real_level_record_closure_properties():
    """n = 8: the tied symmetry closure must give w == 0 with no injected data."""
    record = ladder.run_level(8)
    assert record['closure'] == 'zero_gradient'
    assert record['tied_components'] > 0
    assert record['tie_residual'] <= 1e-12
    assert record['spanwise_l2'] == 0.
    assert record['max_abs_w'] == 0.
    assert record['converged'] is True
    assert record['inplane_l2'] == pytest.approx(4.0054e-02, rel=1e-3)
    assert record['total_l2'] == pytest.approx(record['inplane_l2'], rel=1e-12)