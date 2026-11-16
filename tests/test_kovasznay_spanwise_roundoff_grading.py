"""A round-off series must not be graded as a convergence order.

The exact spanwise velocity of Kovasznay flow is identically zero, so on a
sequence solved with the physical symmetry closure (``du/dz = dv/dz = 0``, id
``zero_gradient``) the measured ``velocity_spanwise`` budget is
``6.8e-18 / 1.1e-18 / 9.0e-19`` on 32/64/128 -- three floating-point noise
levels.  A log-log slope through them (1.46, R^2 0.83) is a fit to that noise.

Grading such a series against the ledger threshold ``>= 1.8`` therefore reports
a FAIL for the *best* correctness evidence in the artifact set, and the failed
number cannot be told apart from a genuine O(h) spurious mode
(``free`` closure: ``4.5e-03`` on the same grid).  ``summarize_kovasznay_32_64_128``
now classifies a quantity as ``machine zero`` when **every** level sits inside
the round-off band, reports ``n/a`` instead of an order, and grades it on its
absolute size.  These tests pin that rule down in both directions: the noise
series must be excused, and a series that leaves the band on any level must
still be graded the ordinary way (the negative control -- otherwise the rule is
just "spanwise never fails").
"""
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'summarize_kovasznay_32_64_128.py'
RUNS = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'runs-32-64-128-zero_gradient'
GRIDS = (32, 64, 128)
THRESHOLD = 1.8


def _load_module():
    spec = importlib.util.spec_from_file_location('summarize_kovasznay_32_64_128', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _series(values):
    return [dict(n=n, h=1. / n, velocity_spanwise=float(value))
            for n, value in zip(GRIDS, values)]


def test_zero_gradient_noise_series_is_machine_zero_and_not_graded():
    module = _load_module()
    audit = module.roundoff_audit(_series([6.779241e-18, 1.119141e-18, 9.003311e-19]),
                                  'velocity_spanwise')
    assert audit['machine_zero'] is True
    assert audit['graded'] is False
    assert audit['max_share'] == pytest.approx(6.779241e-18)
    assert audit['limit'] == module.SPANWISE_ROUNDOFF
    assert audit['note'].startswith('n/a (machine zero')


def test_free_closure_series_is_still_graded():
    """Negative control: the rule must be able to say "this one is real"."""
    module = _load_module()
    # ``free`` closure, O(h) spurious mode: 4.53e-03 -> 2.27e-03 -> 1.13e-03.
    audit = module.roundoff_audit(_series([4.53e-03, 2.27e-03, 1.13e-03]),
                                  'velocity_spanwise')
    assert audit['machine_zero'] is False
    assert audit['graded'] is True
    assert audit['note'] is None
    fit = module.fit_slope(_series([4.53e-03, 2.27e-03, 1.13e-03]), 'velocity_spanwise')
    # O(h) spurious mode: a *measurable* first order that genuinely misses the
    # ledger threshold.  Excusing it as round-off would hide a real defect.
    assert fit['slope'] == pytest.approx(1.0, abs=0.01)
    assert fit['slope'] < THRESHOLD


def test_one_level_outside_the_band_keeps_the_series_graded():
    """Machine zero needs *every* level in the band, not just the finest."""
    module = _load_module()
    audit = module.roundoff_audit(_series([6.8e-18, 2.0e-12, 9.0e-19]),
                                  'velocity_spanwise')
    assert audit['machine_zero'] is False
    assert audit['graded'] is True


def test_committed_three_grid_artifact_records_the_rule():
    summary = json.loads((RUNS / 'summary.json').read_text())
    audit = summary['roundoff_grading']['velocity_spanwise']
    verdict = summary['verdict']
    assert audit['machine_zero'] is True and audit['graded'] is False
    assert verdict['spanwise_machine_zero'] is True
    assert verdict['spanwise_criterion_pass'] is True
    assert verdict['spanwise_order_graded'] is False
    assert verdict['spanwise_max_share'] <= verdict['spanwise_share_limit']
    fit = summary['fits']['velocity_spanwise']
    assert fit['graded'] is False and fit['max_share'] == pytest.approx(audit['max_share'])
    consistency = summary['pair_consistency']['velocity_spanwise']
    assert consistency['graded'] is False and consistency['machine_zero'] is True


def test_committed_two_grid_artifact_records_the_rule():
    summary = json.loads((RUNS / 'summary-32-64' / 'summary.json').read_text())
    assert summary['roundoff_grading']['velocity_spanwise']['machine_zero'] is True
    assert summary['fits']['velocity_spanwise']['graded'] is False


@pytest.mark.parametrize('markdown', ['RESULTS.md', 'summary-32-64/RESULTS.md'])
def test_committed_markdown_does_not_grade_a_roundoff_series(markdown):
    text = (RUNS / markdown).read_text()
    spanwise_rows = [line for line in text.splitlines()
                     if line.startswith('| spanwise w RMS')]
    assert spanwise_rows, 'the spanwise budget must stay visible in the error budget'
    for row in spanwise_rows:
        assert 'machine zero' in row, row
        assert 'FAIL' not in row, row
        # no bare fitted exponent and no fabricated pair orders
        assert '1.4563' not in row and '2.5987' not in row
    assert 'Round-off series are *not* graded on an order' in text
    # the ledger criterion itself must still be graded normally
    velocity_row = [line for line in text.splitlines()
                    if line.startswith('| velocity L2 (full 3D')][0]
    assert 'PASS' in velocity_row