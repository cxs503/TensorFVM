"""Round-3 evidence integrity for CASE-FVM-KOVASZNAY-ORDER (m16838).

The delivered ladder lives in ``docs/verification-kovasznay-order-closure/runs/
ladder_round3.json`` and was produced by ``round3_make.py`` from two raw
per-level payloads.  These tests re-derive every graded number from that
payload with the instrument's own functions, so a silent edit of the archived
evidence (or a regression in ``observed_orders`` / ``three_level_fit``) fails
the gate instead of being discovered by a reader.
"""
import json
from pathlib import Path

import pytest

from tensorfvm.verification.kovasznay_order_closure import (
    evaluate, merge_tables, observed_orders, three_level_fit)

RAW = (Path(__file__).resolve().parents[1]
       / 'docs/verification-kovasznay-order-closure/runs/ladder_round3.json')


@pytest.fixture(scope='module')
def payload():
    if not RAW.is_file():
        pytest.skip('round-3 ladder payload not present')
    return json.loads(RAW.read_text())


def test_ladder_levels_and_monotone_error(payload):
    tied = sorted(payload['table']['zero_gradient'], key=lambda r: r['n'])
    assert [r['n'] for r in tied] == [8, 16, 32]
    errors = [r['total_l2'] for r in tied]
    assert errors == sorted(errors, reverse=True), 'L2 error must decay monotonically'
    # every level converged and the closure is exact at every level
    for record in tied:
        assert record['converged'] is True
        assert record['tie_residual'] <= 1e-12
        assert record['spanwise_l2'] == 0.0 and record['max_abs_w'] == 0.0


def test_observed_orders_recompute_from_raw_payload(payload):
    tied = sorted(payload['table']['zero_gradient'], key=lambda r: r['n'])
    recomputed = {o['pair']: o for o in observed_orders(tied)}
    archived = {o['pair']: o for o in payload['orders']}
    assert set(recomputed) == {'8->16', '16->32'}
    for pair, order in recomputed.items():
        assert order['total'] == pytest.approx(archived[pair]['total'], rel=1e-12)
    # the anchor is graded on the finest available pair
    assert recomputed['16->32']['total'] == pytest.approx(1.80833, abs=1e-4)
    assert recomputed['16->32']['total'] >= 1.8
    # the coarsest pair is pre-asymptotic: this is why the 3-level fit is biased low
    assert recomputed['8->16']['total'] < 1.8


def test_graded_verdict_reproduces(payload):
    verdict = evaluate(payload['table'])
    archived = {c['id']: c['verdict'] for c in payload['verdict']['checks']}
    assert {c['id']: c['verdict'] for c in verdict['checks']} == archived
    assert archived['C4'] == 'PASS' and archived['C5'] == 'PASS' and archived['C7'] == 'PASS'
    # honest FAIL kept: the three-level L+C h^p fit is not asymptotic on 8/16/32
    assert archived['C8'] == 'FAIL'
    fit = verdict['three_level_fit']
    assert fit['asymptotic'] is False and fit['monotone'] is True
    assert fit['order'] == pytest.approx(1.4889, abs=1e-3)
    assert verdict['anchor_closed'] is False


def test_three_level_fit_is_biased_low_while_coarsest_level_is_pre_asymptotic():
    """Regression on the interpretation the report states.

    A ladder whose finest pair is second order but whose coarsest level has not
    entered the asymptotic range yields a *low* three-level fit; the fit's own
    ``asymptotic`` flag is the quantity that says so.
    """
    records = [dict(n=8, total_l2=4.0e-2), dict(n=16, total_l2=1.332788e-2),
               dict(n=32, total_l2=3.805383e-3)]
    fit = three_level_fit(records, 'total_l2')
    assert fit is not None
    assert fit['order'] < 1.8              # biased low by the pre-asymptotic n=8
    assert fit['asymptotic'] is False      # instrument flags it, so C8 cannot be read as a loss of order
    # a genuinely h^2 ladder fits h^2
    square = [dict(n=n, total_l2=1.0e-2 / n ** 2) for n in (8, 16, 32)]
    assert three_level_fit(square, 'total_l2')['order'] == pytest.approx(2.0, abs=1e-9)


def test_merge_does_not_resolve_anything(payload):
    table = payload['table']
    merged = merge_tables(table, table)
    tied = sorted(merged['zero_gradient'], key=lambda r: r['n'])
    assert len(tied) == 3, '(closure, n) de-duplication: merging a table with itself adds no level'
