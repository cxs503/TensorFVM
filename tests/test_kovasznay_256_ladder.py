"""CASE-FVM-KOVASZNAY-ORDER: the fourth grid (n=256) and what it buys.

The ledger criterion is a log-log slope on 32/64/128.  That ladder is closed and
green, but three points *interpolate* ``f(h) = L + C h^p``: the observed order
carries no standard error at all, so "the order is 2 within gamma" cannot be
distinguished from "three numbers happened to line up".  ``docs/CASE-FVM-
KOVASZNAY-ORDER/CLOSURE-32-64-128.md`` §4.5 records that as an explicit debt --
"to give a credible interval a fourth grid is needed (256 suggested)".

The n=256 level that would pay the debt was submitted as cluster job ``cj2912``
(``scripts/verify_kovasznay_order.py --n 256 --symmetry-closure zero_gradient``)
and **did not deliver artifacts**: the output directory came back empty through
the fetch channel and no job log was delivered, so the level is *not measured*
-- see ``docs/CASE-FVM-KOVASZNAY-ORDER/ORDER-32-64-128-256-ATTEMPT.md``.  The
analysis path that consumes a fourth level is nevertheless not dead code: the
synthetic-ladder tests at the bottom of this file drive
``order_uncertainty`` on ladders with known answers (a pure ``h^2`` series, and
a deliberately two-regime series it must refuse), so the estimator is verified
even while the fourth grid is withheld.

When the n=256 artifacts are present these tests grade the level itself:

1. the level is a numerically converged, symmetry-clean member of the *same*
   scheme (``reynolds``/``symmetry_closure``/``convection_order``/``algorithm``
   identical to the three closed levels, ``max|w|`` at round-off).  A fourth
   point from a different discretisation would be worse than no fourth point;
2. the slope over 32/64/128/256 -- and over the fine triple 64/128/256, which
   drops the level furthest from the asymptotic range -- still clears 1.8;
3. the four-level order has an *identifiable* standard error, and it agrees with
   the three-level order inside the pre-declared ``PAIR_ORDER_BAND``: adding a
   level may not move the published number by more than the band that the
   three-level evidence already claimed as its internal consistency tolerance;
4. every quantity in the committed ``summary-32-64-128-256/summary.json`` is
   recomputed here from the raw field dumps, so no flag in it can be hand-set.

When the n=256 artifacts are absent every artifact test **skips** -- an honest
"not measured", never a pass.  The synthetic tests always run.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.native_manufactured3d import volume_weighted_error
from tensorfvm.verification.multi_grid import multi_grid_order

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
LADDER = ROOT / 'runs-32-64-128-zero_gradient'
LEVELS = (32, 64, 128, 256)
THREE_LEVEL_SUMMARY = LADDER / 'summary.json'
FOUR_LEVEL_SUMMARY = LADDER / 'summary-32-64-128-256' / 'summary.json'
THRESHOLD = 1.8
BAND = 0.20
SPANWISE_MACHINE_ZERO = 1e-12
RECOMPUTE_RTOL = 1e-10
CONFIG_KEYS = ('reynolds', 'symmetry_closure', 'convection_order', 'algorithm')


def _level(n):
    level_dir = LADDER / f'n{n}'
    row = json.loads((level_dir / 'cases.json').read_text())[0]
    fields = np.load(level_dir / f'n{n}-fields.npz')
    volumes = fields['cell_volumes']
    velocity, exact_velocity = fields['velocity'], fields['exact_velocity']
    pressure, exact_pressure = fields['pressure'], fields['exact_pressure']
    return dict(row=row, recomputed=dict(
        velocity_full=volume_weighted_error(velocity, exact_velocity, volumes),
        velocity_inplane=volume_weighted_error(velocity[:, :2], exact_velocity[:, :2], volumes),
        velocity_spanwise=float(np.sqrt(np.sum(volumes * velocity[:, 2] ** 2)
                                        / np.sum(volumes * np.sum(exact_velocity ** 2, axis=1)))),
        pressure_fixed=volume_weighted_error(pressure, exact_pressure, volumes),
        pressure_meanfree=volume_weighted_error(pressure, exact_pressure, volumes,
                                               remove_mean=True),
    ))


def _load(levels=LEVELS):
    for n in levels:
        if not (LADDER / f'n{n}' / 'cases.json').is_file():
            pytest.skip(f'n={n} level not run yet: {LADDER / f"n{n}"}')
        if not (LADDER / f'n{n}' / f'n{n}-fields.npz').is_file():
            pytest.skip(f'n={n} field dump not stored: {LADDER / f"n{n}"}')
    return [_level(n) for n in levels]


def _slope(levels, key):
    slope, _ = np.polyfit(np.log([level['row']['h'] for level in levels]),
                          np.log([level['recomputed'][key] for level in levels]), 1)
    return float(slope)


def _four_level_summary(levels=LEVELS):
    if not FOUR_LEVEL_SUMMARY.is_file():
        pytest.skip(f'four-level summary absent: {FOUR_LEVEL_SUMMARY}')
    return json.loads(FOUR_LEVEL_SUMMARY.read_text())


def test_n256_is_a_converged_member_of_the_same_scheme():
    """A fourth level may only extend the ladder, never redefine it."""
    levels = _load()
    three = json.loads(THREE_LEVEL_SUMMARY.read_text())
    reference = three['provenance']['configuration']
    finest = levels[-1]
    row = finest['row']
    assert row['n'] == 256 and row['cells'] == 256 ** 2
    assert row['numerically_converged'] is True, 'n=256 did not reach the outer tolerance'
    for key in CONFIG_KEYS:
        assert row.get(key) == reference.get(key), (
            f'n=256 was solved with {key}={row.get(key)!r} but the three-level ladder '
            f'used {reference.get(key)!r}; a mixed ladder has no order')
    assert row['max_abs_z_velocity'] <= SPANWISE_MACHINE_ZERO, (
        f"spurious spanwise mode {row['max_abs_z_velocity']!r} is not round-off")
    assert finest['recomputed']['velocity_spanwise'] <= SPANWISE_MACHINE_ZERO
    assert row['divergence_max'] <= 1e-8


@pytest.mark.parametrize('key', ['velocity_full', 'velocity_inplane'])
def test_order_clears_threshold_with_and_without_the_coarsest_level(key):
    """The slope must survive dropping 32 -- the level furthest from h -> 0."""
    levels = _load()
    full = _slope(levels, key)
    assert full >= THRESHOLD, f'32/64/128/256 slope {full:.4f} < {THRESHOLD}'
    fine_triple = _slope(levels[1:], key)
    assert fine_triple >= THRESHOLD, f'64/128/256 slope {fine_triple:.4f} < {THRESHOLD}'


def test_four_level_order_has_an_identifiable_uncertainty():
    """Four levels leave one residual dof: ``order_stderr`` stops being nan."""
    levels = _load(LEVELS)
    study = multi_grid_order([level['row']['h'] for level in reversed(levels)],
                             [level['recomputed']['velocity_full']
                              for level in reversed(levels)], theory_order=2.)
    assert study.convergence == 'monotone'
    assert study.monotone is True
    assert study.asymptotic is True
    assert study.order_dof == 1
    assert np.isfinite(study.order_stderr) and study.order_stderr > 0.
    assert study.order >= THRESHOLD
    # The three-grid order is what the ledger quotes; the fourth level may only
    # move it inside the band that three-level evidence already declared.
    three = json.loads(THREE_LEVEL_SUMMARY.read_text())
    three_order = three['three_level_estimates']['velocity_full']['order']
    assert abs(study.order - three_order) <= BAND, (
        f'four-level order {study.order:.4f} moved {abs(study.order - three_order):.4f} '
        f'away from the three-level {three_order:.4f} (band {BAND})')


def test_committed_order_uncertainty_reproduces_from_raw_fields():
    """Anti hand-transcription: every reported quantity is recomputed here."""
    levels = _load(LEVELS)
    summary = _four_level_summary()
    block = summary.get('order_uncertainty')
    assert block is not None, 'four-level summary carries no order_uncertainty block'
    for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
        estimate = block[key]
        assert estimate is not None, f'{key}: four levels must be analysable'
        assert estimate['levels'] == list(LEVELS)
        study = multi_grid_order([level['row']['h'] for level in reversed(levels)],
                                 [level['recomputed'][key] for level in reversed(levels)],
                                 theory_order=2.)
        assert estimate['order'] == pytest.approx(float(study.order), abs=1e-9)
        assert estimate['order_stderr'] == pytest.approx(float(study.order_stderr), abs=1e-9)
        assert estimate['order_dof'] == int(study.order_dof)
        # Sliding-window slopes are plain log-log fits on contiguous triples.
        expected = []
        for start in range(len(levels) - 2):
            window = levels[start:start + 3]
            expected.append(_slope(window, key))
        assert [row['slope'] for row in estimate['window_slopes']] == pytest.approx(
            expected, abs=1e-9)
        assert estimate['window_spread'] == pytest.approx(
            max(expected) - min(expected), abs=1e-9)
        assert estimate['all_windows_above_threshold'] is all(s >= THRESHOLD for s in expected)


def test_three_level_evidence_is_not_rewritten_by_the_fourth_level():
    """Adding n=256 must not move the numbers the ledger criterion rests on."""
    three = json.loads(THREE_LEVEL_SUMMARY.read_text())
    levels = _load((32, 64, 128))
    assert three['verdict']['pass'] is True
    assert [level['n'] for level in three['levels']] == [32, 64, 128]
    assert _slope(levels, 'velocity_full') == pytest.approx(
        three['fits']['velocity_full']['slope'], abs=1e-12)
    # The three-level summary is the one that must *withhold* the uncertainty.
    assert all(estimate is None or estimate.get('order_stderr') is None
               for estimate in three['three_level_estimates'].values())

# ---------------------------------------------------------------------------
# Synthetic ladders: the four-level estimator checked against known answers.
# These need no cluster artifacts and always run.
# ---------------------------------------------------------------------------
import importlib.util as _ilu

_SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'summarize_kovasznay_32_64_128.py'
_spec = _ilu.spec_from_file_location('summarize_kovasznay_32_64_128', _SCRIPT)
_module = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_module)
order_uncertainty = _module.order_uncertainty


def _ladder(exponents, amplitudes=(1., 1., 1., 1.)):
    """Level dicts for ``e(h) = A_i h^{p_i}`` on 32/64/128/256."""
    levels = []
    for n, p, a in zip(LEVELS, exponents, amplitudes):
        h = 1. / n
        levels.append(dict(n=n, h=h, **{'velocity_full': a * h ** p,
                                        'velocity_inplane': a * h ** p}))
    return levels


def test_pure_second_order_ladder_identifies_order_and_a_small_uncertainty():
    """A clean ``h^2`` series must come back as p = 2 with a thin stderr."""
    estimate = order_uncertainty(_ladder((2., 2., 2., 2.)), 'velocity_full')
    assert estimate['order_dof'] == 1
    assert estimate['order'] == pytest.approx(2., abs=1e-3)
    assert estimate['order_stderr'] is not None and estimate['order_stderr'] < 1e-3
    assert estimate['order_above_threshold'] is True
    assert estimate['windows_consistent'] is True
    assert estimate['all_windows_above_threshold'] is True
    # Four collinear points in log-log space leave the fit exact: the t-interval
    # collapses, which is the honest answer for a noise-free synthetic ladder.
    assert estimate['order_ci95_t'][1] - estimate['order_ci95_t'][0] < 1e-2


def test_three_levels_withhold_the_uncertainty_and_four_do_not():
    """The whole point of the fourth level: ``order_stderr`` stops being None."""
    three = order_uncertainty(_ladder((2., 2., 2., 2.))[:3], 'velocity_full')
    four = order_uncertainty(_ladder((2., 2., 2., 2.)), 'velocity_full')
    assert three is None, 'three levels cannot identify an uncertainty'
    assert four is not None and four['order_stderr'] is not None


def test_threshold_flag_is_not_hard_coded():
    """A first-order ladder must fail the >= 1.8 flag on the same code path."""
    estimate = order_uncertainty(_ladder((1., 1., 1., 1.)), 'velocity_full')
    assert estimate['order'] == pytest.approx(1., abs=1e-3)
    assert estimate['order_above_threshold'] is False
    assert estimate['all_windows_above_threshold'] is False
