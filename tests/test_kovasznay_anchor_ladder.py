"""Tests for the Kovasznay order-ladder instrument (per-level runs, merge, C1-C8).

These tests exercise the *bookkeeping* of ``verification.kovasznay_order_closure``
against synthetic ladders whose answer is known by construction, so they are
fast and do not depend on a converged Navier-Stokes solve.  The physical ladder
itself is graded in ``docs/verification-kovasznay-order-closure/ROUND2.md``.
"""
import json
import math

import pytest

from tensorfvm.verification.kovasznay_order_closure import (
    ASYMPTOTIC_MIN_N, GATE, evaluate, merge_tables, merge_payloads, observed_orders,
    three_level_fit)


def record(n, total, spanwise=0.0, closure='zero_gradient', **extra):
    """A level record with the fields the evaluator reads."""
    base = dict(n=n, closure=closure, cells=n * n, faces=6 * n * n, dofs=1,
                converged=True, outer_iterations=1, seconds=0.0, peak_rss_mb=0.0,
                total_l2=total, inplane_l2=total, spanwise_l2=spanwise,
                max_abs_w=spanwise, tie_residual=0.0, tied_components=1,
                prescribed_tangential_on_planes=0.0,
                pressure_l2=total, pressure_mean_free_l2=total)
    base.update(extra)
    return base


def second_order_ladder(levels=(8, 16, 32)):
    return [record(n, 2.0 / n ** 2) for n in levels]


def check(verdict, identifier):
    return [c for c in verdict['checks'] if c['id'] == identifier][0]


def test_observed_pairs_carry_the_coarse_level_for_grading():
    orders = observed_orders(second_order_ladder())
    assert [o['pair'] for o in orders] == ['8->16', '16->32']
    assert [o['coarse_n'] for o in orders] == [8, 16]
    # a pure h^2 ladder is second order on every pair
    assert all(abs(o['total'] - 2.0) < 1e-9 for o in orders)


def test_second_order_ladder_passes_the_anchor_and_the_three_level_fit():
    verdict = evaluate(dict(zero_gradient=second_order_ladder()))
    for identifier in ('C1', 'C2', 'C4', 'C5', 'C6', 'C7', 'C8'):
        assert check(verdict, identifier)['verdict'] == 'PASS', identifier
    assert verdict['anchor_closed'] is True
    fit = verdict['three_level_fit']
    assert fit['levels'] == [32, 16, 8]          # fine -> coarse
    assert abs(fit['order'] - 2.0) < 1e-3
    assert fit['convergence'] == 'monotone'


def test_one_lucky_pair_cannot_pass_the_asymptotic_criterion():
    """C4 can pass on the finest pair while C7 (every asymptotic pair) fails."""
    # 16->32 is first-and-a-half order (below the gate); 32->64 drops a clean 8x.
    # C4 grades the finest pair only, C7 grades every pair with n_coarse >= 16.
    ladder = [record(8, 4e-2), record(16, 1e-2), record(32, 1e-2 / 2 ** 1.5), record(64, 0.)]
    ladder[-1]['total_l2'] = ladder[-2]['total_l2'] / 8.
    verdict = evaluate(dict(zero_gradient=ladder))
    assert check(verdict, 'C4')['verdict'] == 'PASS'
    assert check(verdict, 'C7')['verdict'] == 'FAIL'
    assert verdict['anchor_closed'] is False
    assert ASYMPTOTIC_MIN_N == 16 and GATE == 1.8


def test_first_order_spanwise_artefact_fails_C2():
    ladder = [record(8, 4e-2, spanwise=9e-3), record(16, 1.2e-2, spanwise=4.1e-3)]
    verdict = evaluate(dict(zero_gradient=ladder))
    assert check(verdict, 'C2')['verdict'] == 'FAIL'
    assert verdict['anchor_closed'] is False


def test_three_level_fit_is_not_run_below_three_levels():
    verdict = evaluate(dict(zero_gradient=second_order_ladder((8, 16))))
    assert check(verdict, 'C8')['verdict'] == 'NOT_RUN'
    assert 'C8' in verdict['not_run']
    assert verdict['three_level_fit'] is None


def test_missing_control_level_is_not_run_not_pass():
    verdict = evaluate(dict(zero_gradient=second_order_ladder()))
    assert check(verdict, 'C3')['verdict'] == 'NOT_RUN'


def test_merge_takes_the_first_record_per_closure_and_level():
    a = dict(zero_gradient=[record(8, 4e-2)], symmetry=[record(8, 4.01e-2, closure='symmetry')])
    b = dict(zero_gradient=[record(8, 9.9e-9), record(16, 1.3e-2)])
    merged = merge_tables(a, b)
    assert [r['n'] for r in merged['zero_gradient']] == [8, 16]
    assert merged['zero_gradient'][0]['total_l2'] == pytest.approx(4e-2)
    assert [r['n'] for r in merged['symmetry']] == [8]


def test_merge_payloads_reproduces_the_verdict_of_a_single_process(tmp_path):
    ladder = second_order_ladder()
    whole = dict(levels=[8, 16, 32], closures=['zero_gradient'],
                 table=dict(zero_gradient=ladder))
    halves = [dict(levels=[8, 16], closures=['zero_gradient'],
                   table=dict(zero_gradient=ladder[:2])),
              dict(levels=[32], closures=['zero_gradient'],
                   table=dict(zero_gradient=ladder[2:]))]
    paths = []
    for index, payload in enumerate(halves):
        path = tmp_path / f'p{index}.json'
        path.write_text(json.dumps(payload))
        paths.append(path)
    merged = merge_payloads(paths)
    assert merged['levels'] == [8, 16, 32]
    assert merged['verdict']['three_level_fit']['order'] == pytest.approx(
        three_level_fit(whole['table']['zero_gradient'])['order'])
    assert merged['verdict']['anchor_closed'] is True


def test_inconsistent_ladder_is_reported_not_averaged_away():
    """A non-decaying ladder must not produce a confident order (C8 refuses)."""
    ladder = [record(8, 1e-3), record(16, 1.2e-3), record(32, 9e-4)]
    fit = three_level_fit(ladder)
    assert fit is None or (not (math.isfinite(fit['order']) and fit['order'] >= GATE)) \
        or fit['convergence'] != 'monotone'
    verdict = evaluate(dict(zero_gradient=ladder))
    assert check(verdict, 'C8')['verdict'] == 'FAIL'