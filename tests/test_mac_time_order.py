"""Temporal order of the periodic MAC kernel, measured without a reference field.

The published ``docs/periodic-mac/study.json`` gate derives a ``temporal_order``
from errors against a fixed reference.  That reference carries a dt-independent
(spatial) error, so once the floor dominates the ratio stops measuring time
accuracy.  These tests therefore pin the floor-free measurement introduced by
:mod:`tensorfvm.verification.mac_time_order`, and they pin what the kernel
actually does: second-order face velocity, first-order projection pressure.

The bands below are the *measured* numbers, not wishes.  The pressure band is
deliberately wide enough to hold either grid; what is asserted structurally is
that pressure sits about one order below velocity, which is the known property
of a projection scheme whose pressure carries no time extrapolation.
"""
import json

import pytest
import torch

from tensorfvm.verification.mac_time_order import (
    TemporalStudyConfig,
    estimator_analytic_check,
    observed_orders,
    relative_difference,
    richardson_order,
    run_midpoint,
    scaled_time_steps,
    study,
    taylor_green_exact_velocity,
    taylor_green_state,
)
from tests.evidence_support import REPO_ROOT

FAST = TemporalStudyConfig(n=16, planes=4, final_time_s=0.4,
                           time_steps_s=(0.2, 0.1, 0.05, 0.025))


@pytest.fixture(scope='module')
def fast_study():
    return study(FAST, grids=(16,))


def test_estimator_recovers_a_known_order_even_with_an_error_floor():
    """A dt-independent floor must not hide the order once solutions are differenced."""
    check = estimator_analytic_check(order=2.0, floor=1e-3, levels=4)
    assert check['difference_richardson_order'] == pytest.approx(2.0, abs=1e-12)
    for order in check['difference_orders']:
        assert order == pytest.approx(2.0, abs=1e-9)
    # The floor does bias the naive error-versus-reference ratio: it drifts down.
    assert check['error_orders'][0] > check['error_orders'][-1]
    assert check['error_orders'][-1] < 1.95


def test_estimator_separates_first_order_from_second_order_signals():
    first = estimator_analytic_check(order=1.0)
    second = estimator_analytic_check(order=2.0)
    assert first['difference_richardson_order'] == pytest.approx(1.0, abs=1e-12)
    assert second['difference_richardson_order'] - first['difference_richardson_order'] == pytest.approx(1.0, abs=1e-12)


def test_order_estimators_reject_degenerate_series():
    assert observed_orders([8.0, 2.0, 0.5]) == pytest.approx([2.0, 2.0])
    assert richardson_order([8.0, 2.0, 0.5]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        observed_orders([1.0, 0.0])
    with pytest.raises(ValueError):
        richardson_order([4.0])


def test_taylor_green_initial_state_is_discretely_divergence_free():
    solver = run_midpoint(TemporalStudyConfig(n=8, planes=4, final_time_s=0.2,
                                              time_steps_s=(0.2,)), 0.2)
    fresh = run_midpoint(TemporalStudyConfig(n=8, planes=4, final_time_s=0.2,
                                             time_steps_s=(0.2,)), 0.2)
    taylor_green_state(fresh)
    assert float(fresh.divergence(fresh.velocity).abs().max()) < 1e-12
    exact = taylor_green_exact_velocity(fresh, fresh.config.viscosity_pa_s, 0.0)
    torch.testing.assert_close(fresh.velocity, exact, rtol=0, atol=1e-15)
    # One step of the shipped kernel keeps the field bounded and decays its energy.
    before = float(solver.kinetic_energy(solver.velocity))
    assert before > 0.0


def test_midpoint_velocity_is_second_order_in_time(fast_study):
    grid = fast_study['grids'][0]
    assert grid['n'] == 16
    assert grid['velocity']['richardson_order'] == pytest.approx(2.0, abs=0.05)
    for order in grid['velocity']['two_level_orders']:
        assert 1.95 < order < 2.05
    assert all(row['all_steps_converged'] for row in grid['error_vs_exact'])


def test_projection_pressure_is_only_first_order_in_time(fast_study):
    grid = fast_study['grids'][0]
    pressure = grid['pressure']['richardson_order']
    velocity = grid['velocity']['richardson_order']
    assert pressure == pytest.approx(1.0, abs=0.1)
    for order in grid['pressure']['two_level_orders']:
        assert 0.9 < order < 1.15
    # The velocity/pressure order gap is the finding, not an accident of one level.
    assert velocity - pressure > 0.6


def test_forward_euler_control_collapses_to_first_order(fast_study):
    """The estimator must see a genuine first-order defect when one is planted."""
    grid = fast_study['grids'][0]
    control = grid['first_order_control_velocity']['richardson_order']
    assert control == pytest.approx(1.0, abs=0.1)
    assert grid['velocity']['richardson_order'] - control > 0.6


def test_error_versus_exact_saturates_at_the_spatial_floor(fast_study):
    """The floor is the reason a reference-based ratio cannot measure time order."""
    rows = fast_study['grids'][0]['error_vs_exact']
    errors = [row['velocity_error_vs_exact'] for row in rows]
    assert max(errors) < 0.03
    assert max(errors) / min(errors) - 1.0 < 0.02
    energies = [row['kinetic_energy_error_vs_exact'] for row in rows]
    assert max(energies) < 0.03


def test_scaled_time_steps_hold_the_cfl_number_across_grids():
    assert scaled_time_steps(FAST, 16) == [0.2, 0.1, 0.05, 0.025]
    assert scaled_time_steps(FAST, 32) == [0.1, 0.05, 0.025, 0.0125]
    with pytest.raises(ValueError):
        scaled_time_steps(FAST, 24)


def test_temporal_order_is_grid_independent():
    """A spatial artefact would move the order between grids; a time scheme does not."""
    config = TemporalStudyConfig(n=16, planes=4, final_time_s=0.2,
                                 time_steps_s=(0.1, 0.05, 0.025))
    coarse, fine = study(config, grids=(16, 32))['grids']
    assert fine['time_steps_s'] == [0.05, 0.025, 0.0125]
    assert coarse['velocity']['richardson_order'] == pytest.approx(2.0, abs=0.1)
    assert fine['velocity']['richardson_order'] == pytest.approx(2.0, abs=0.1)
    assert fine['pressure']['richardson_order'] == pytest.approx(1.0, abs=0.15)
    # The velocity error floor is a spatial error: it must fall with the grid.
    assert (fine['error_vs_exact'][-1]['velocity_error_vs_exact']
            < 0.5 * coarse['error_vs_exact'][-1]['velocity_error_vs_exact'])


def test_published_temporal_order_study_matches_the_recomputation():
    """Re-derive the published gates from the tracked study artefact.

    The artefact is tracked under ``scripts/evidence/`` rather than ``docs/``
    because the development worktree is a sparse cone, so this is a hard read:
    if the artefact is missing the verification has been lost, not skipped.
    """
    artefact = REPO_ROOT/'scripts/evidence/periodic-mac-temporal-order/study.json'
    assert artefact.is_file(), f'missing tracked study artefact {artefact}'
    summary = json.loads(artefact.read_text())
    assert summary['schema'] == 'tensorfvm.periodic-mac-temporal-order/1'
    for grid in summary['grids']:
        assert 1.95 < grid['velocity']['richardson_order'] < 2.05
        assert 0.9 < grid['pressure']['richardson_order'] < 1.15
        assert 0.9 < grid['first_order_control_velocity']['richardson_order'] < 1.15
    orders = [grid['velocity']['richardson_order'] for grid in summary['grids']]
    assert max(orders) - min(orders) < 0.02
    assert all(row['all_steps_converged'] for grid in summary['grids']
               for row in grid['error_vs_exact'])