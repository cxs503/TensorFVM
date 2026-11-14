"""Acceptance gate for CASE-FVM-KOVASZNAY-ORDER: 32/64/128 log-log order.

The feature-ledger acceptance for this anchor is

    "L2 convergence order >= 1.8 (nominal second order) on the three grids
     32/64/128, measured as the log-log slope"

This gate is *evidence-bound*: it reads the committed run artifacts, **recomputes**
every relative L2 error from the saved cell-centred fields, and only then
evaluates the criterion.  A hand-edited ``cases.json`` therefore cannot make the
gate green, and a regenerated report cannot drift from the numbers below.

Three artifact sets are parametrised:

* ``anchor`` — the acceptance triple 32/64/128 solved with the historical
  *free* tangential closure on the two extrusion planes.  That closure is a
  traction-free condition, not a symmetry condition, so it leaves an O(h)
  spurious spanwise velocity and the criterion is only meaningful on the
  in-plane budget.  It is skipped until the n=64 and n=128 levels exist, so the
  skip is an honest "not measured yet", never a pass.
* ``anchor_zg`` — the same 32/64/128 triple with the physical symmetry closure
  (``du/dz = dv/dz = 0``) applied on those planes.  The spurious spanwise mode
  drops to round-off, so the ledger criterion is evaluated on the *full 3D*
  velocity norm, with no decomposition defence needed.
* ``selftest`` — the 4/8/16 triplet, far inside the pre-asymptotic range.  It
  exists to prove the *pipeline* (recompute checks, slopes, honest verdict
  flags) works on real field dumps; it is an expected FAIL of the >= 1.8
  criterion and must be recorded as such.

Reference: Kovasznay (1948), Proc. Cambridge Philos. Soc. 44:58-62,
doi:10.1017/S0305004100023999.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.native_manufactured3d import volume_weighted_error

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
SETS = {'anchor': (ROOT / 'runs-32-64-128', (32, 64, 128)),
        'anchor_zg': (ROOT / 'runs-32-64-128-zero_gradient', (32, 64, 128)),
        'selftest': (ROOT / 'selftest-4-8-16', (4, 8, 16))}
SPANWISE_MACHINE_ZERO = 1e-12
THRESHOLD = 1.8
RECOMPUTE_RTOL = 1e-10


def _paths(name):
    directory, grids = SETS[name]
    return directory, grids


def _summary(directory):
    return json.loads((directory / 'summary.json').read_text())


def _level(directory, n):
    """Recorded row plus errors recomputed from that level's raw field dump."""
    level_dir = directory / f'n{n}'
    row = json.loads((level_dir / 'cases.json').read_text())[0]
    fields = np.load(level_dir / f'n{n}-fields.npz')
    volumes = fields['cell_volumes']
    velocity, exact_velocity = fields['velocity'], fields['exact_velocity']
    pressure, exact_pressure = fields['pressure'], fields['exact_pressure']
    recomputed = dict(
        velocity_full=volume_weighted_error(velocity, exact_velocity, volumes),
        velocity_inplane=volume_weighted_error(velocity[:, :2], exact_velocity[:, :2], volumes),
        velocity_spanwise=float(np.sqrt(np.sum(volumes * velocity[:, 2] ** 2)
                                        / np.sum(volumes * np.sum(exact_velocity ** 2, axis=1)))),
        pressure_fixed=volume_weighted_error(pressure, exact_pressure, volumes),
        pressure_meanfree=volume_weighted_error(pressure, exact_pressure, volumes,
                                                remove_mean=True),
    )
    return dict(row=row, recomputed=recomputed)


def _load(name):
    directory, grids = _paths(name)
    if not (directory / 'summary.json').is_file():
        pytest.skip(f'{name} artifacts absent: {directory}')
    for n in grids:
        if not (directory / f'n{n}' / 'cases.json').is_file():
            pytest.skip(f'{name} level n={n} not run yet')
    return directory, [_level(directory, n) for n in grids]


def _slope(levels, key):
    h = np.log([level['row']['h'] for level in levels])
    e = np.log([level['recomputed'][key] for level in levels])
    slope, _ = np.polyfit(h, e, 1)
    return float(slope)


@pytest.mark.parametrize('name', list(SETS))
def test_levels_cover_declared_grids_and_are_converged(name):
    directory, levels = _load(name)
    _, grids = _paths(name)
    assert [level['row']['n'] for level in levels] == list(grids)
    for n, level in zip(grids, levels):
        row = level['row']
        assert row['h'] == pytest.approx(1. / n, rel=1e-12)
        assert row['cells'] == n * n, 'symmetry-extruded n x n x 1 mesh expected'
        assert row['numerically_converged'] is True, f'n={n} did not converge'
        assert row['reynolds'] == pytest.approx(40.)
        assert row['convection_order'] == 2


@pytest.mark.parametrize('name', list(SETS))
@pytest.mark.parametrize('recorded,recomputed', [
    ('velocity_l2', 'velocity_full'),
    ('pressure_l2', 'pressure_fixed'),
    ('pressure_mean_free_l2', 'pressure_meanfree'),
])
def test_recorded_errors_reproduce_from_raw_fields(name, recorded, recomputed):
    """No hand transcription: cases.json must equal the recomputed field error."""
    directory, levels = _load(name)
    for level in levels:
        assert level['row'][recorded] == pytest.approx(level['recomputed'][recomputed],
                                                      rel=RECOMPUTE_RTOL, abs=1e-15)


@pytest.mark.parametrize('name', list(SETS))
def test_reported_slopes_match_recomputed_slopes(name):
    directory, levels = _load(name)
    summary = _summary(directory)
    for key in ('velocity_full', 'velocity_inplane', 'pressure_meanfree'):
        assert summary['fits'][key]['slope'] == pytest.approx(_slope(levels, key), abs=1e-9)


@pytest.mark.parametrize('name', list(SETS))
def test_verdict_flags_are_not_overstated(name):
    directory, levels = _load(name)
    verdict = _summary(directory)['verdict']
    full_order = _slope(levels, 'velocity_full')
    inplane_order = _slope(levels, 'velocity_inplane')
    assert verdict['velocity_full_slope'] == pytest.approx(full_order, abs=1e-9)
    assert verdict['velocity_inplane_slope'] == pytest.approx(inplane_order, abs=1e-9)
    assert verdict['velocity_full_pass'] is bool(full_order >= THRESHOLD)
    assert verdict['velocity_inplane_pass'] is bool(inplane_order >= THRESHOLD)
    assert verdict['all_levels_converged'] is all(level['row']['numerically_converged']
                                                  for level in levels)
    assert verdict['pass'] is bool(verdict['velocity_full_pass'] and verdict['all_levels_converged'])


@pytest.mark.parametrize('name', list(SETS))
def test_spanwise_budget_explains_full_3d_order(name):
    """An O(h) spurious spanwise mode must be visible in the error budget.

    The exact spanwise velocity is identically zero, so a full-3D velocity norm
    can converge slower than the in-plane discretisation.  The artifacts must
    carry that decomposition instead of quietly dropping the failing number.
    """
    directory, levels = _load(name)
    spanwise_orders = [float(np.log(levels[i]['recomputed']['velocity_spanwise']
                                    / levels[i + 1]['recomputed']['velocity_spanwise'])
                             / np.log(levels[i]['row']['h'] / levels[i + 1]['row']['h']))
                       for i in range(len(levels) - 1)]
    assert levels[-1]['recomputed']['velocity_spanwise'] > 0.
    if _slope(levels, 'velocity_full') < THRESHOLD:
        assert min(spanwise_orders) < _slope(levels, 'velocity_inplane')


def test_anchor_inplane_velocity_and_pressure_orders_meet_acceptance():
    """The declared criterion on 32/64/128, evaluated on recomputed numbers."""
    directory, levels = _load('anchor')
    velocity_order = _slope(levels, 'velocity_inplane')
    pressure_order = _slope(levels, 'pressure_meanfree')
    assert velocity_order >= THRESHOLD, f'in-plane velocity slope {velocity_order:.4f} < 1.8'
    assert pressure_order >= THRESHOLD, f'mean-free pressure slope {pressure_order:.4f} < 1.8'

def test_anchor_zg_symmetry_closure_is_uniform_and_removes_spurious_spanwise_mode():
    """The zero-gradient closure must be uniform across the triple and effective.

    Mixed closures inside one grid sequence would make the slope meaningless, so
    every level has to declare the same closure; and because the analytic
    spanwise velocity is identically zero, the closure is only doing its job if
    the discrete spanwise velocity stays at round-off on every level.
    """
    directory, levels = _load('anchor_zg')
    closures = [level['row'].get('symmetry_closure') for level in levels]
    assert closures == ['zero_gradient'] * len(closures)
    for level in levels:
        assert level['row']['max_abs_z_velocity'] <= SPANWISE_MACHINE_ZERO, (
            f"n={level['row']['n']} spurious max|w| "
            f"{level['row']['max_abs_z_velocity']!r} above round-off")
        assert level['recomputed']['velocity_spanwise'] <= SPANWISE_MACHINE_ZERO


def test_anchor_zg_full_3d_velocity_order_meets_acceptance():
    """Ledger criterion on the physical closure, on the full 3D velocity norm."""
    directory, levels = _load('anchor_zg')
    velocity_order = _slope(levels, 'velocity_full')
    inplane_order = _slope(levels, 'velocity_inplane')
    pressure_order = _slope(levels, 'pressure_meanfree')
    assert velocity_order >= THRESHOLD, f'full-3D velocity slope {velocity_order:.4f} < 1.8'
    assert inplane_order >= THRESHOLD, f'in-plane velocity slope {inplane_order:.4f} < 1.8'
    assert pressure_order >= THRESHOLD, f'mean-free pressure slope {pressure_order:.4f} < 1.8'
    assert _summary(directory)['verdict']['pass'] is True


def _pair_orders(levels, key):
    return [float(np.log(coarse['recomputed'][key] / fine['recomputed'][key])
                  / np.log(coarse['row']['h'] / fine['row']['h']))
            for coarse, fine in zip(levels, levels[1:])]


@pytest.mark.parametrize('name', list(SETS))
def test_pair_orders_and_three_level_order_reproduce(name):
    """The corroboration flags must be recomputed quantities, not assertions.

    A three-grid log-log slope has no residual degrees of freedom, so the
    summary carries two further statements: the two per-pair orders (do both
    clear the threshold and agree inside the pre-declared 0.20 band?) and the
    order re-identified by fitting ``f(h) = L + C h^p`` to the whole ladder.
    Both are recomputed here from the raw fields so neither can be hand-set.
    """
    from tensorfvm.verification.multi_grid import multi_grid_order

    directory, levels = _load(name)
    summary = _summary(directory)
    for key in ('velocity_full', 'velocity_inplane'):
        pairs = _pair_orders(levels, key)
        block = summary['pair_consistency'][key]
        assert block['pairs'] == pytest.approx(pairs, abs=1e-9)
        assert block['spread'] == pytest.approx(max(pairs) - min(pairs), abs=1e-9)
        assert block['both_above'] is all(order >= THRESHOLD for order in pairs)
        assert block['consistent'] is bool(max(pairs) - min(pairs) <= block['band'])
        estimate = summary['three_level_estimates'][key]
        study = multi_grid_order([level['row']['h'] for level in reversed(levels)],
                                 [level['recomputed'][key] for level in reversed(levels)],
                                 theory_order=2.)
        assert estimate['order'] == pytest.approx(float(study.order), abs=1e-9)
        assert estimate['levels'] == [level['row']['n'] for level in levels]
        # Three levels cannot identify an uncertainty; withholding it is honest.
        assert estimate['order_stderr'] is None
    verdict = summary['verdict']
    assert verdict['corroborated'] is bool(
        verdict['pass'] and verdict['pair_orders_consistent']
        and verdict['three_level_order'] is not None
        and verdict['three_level_order'] >= THRESHOLD)
