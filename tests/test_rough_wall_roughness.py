"""Tests for the equivalent sand-grain roughness closure and its verification.

Three layers:

* **synthetic** -- the Nikuradse/Wilcox relations and the log-law fit are checked
  against data whose answer is known by construction, so a sign slip or a
  swapped all-rough / transitional branch fails here rather than silently
  re-basing the graded sweep;
* **contract** -- the ``k_s`` hook of :func:`solve_channel` may not leak: the
  smooth default must be bit-identical to the pre-existing call, the wall
  function model must refuse roughness, and a mesh too coarse for the condition
  to act must raise instead of returning a smooth answer;
* **registration** -- a reduced but real sweep runs the closure and pins the
  numbers the report quotes, so a silent regression (or a quietly relaxed
  criterion) fails the suite.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.rans_validation.channel import solve_channel
from tensorfvm.rans_validation.models import implied_kappa, wall_values, constants_for
from tensorfvm.rans_validation.roughness import (B_ROUGH, B_SMOOTH, KAPPA,
                                                 S_R_SWITCH,
                                                 crossover_k_s_plus, fit_rough_log_law,
                                                 nikuradse_shift, nikuradse_u_plus,
                                                 rough_wall_value, wilcox_s_r)
from tensorfvm.verification import rough_wall_channel as driver
from tensorfvm.verification.rough_wall_channel import (LADDER, finish_criteria, grade,
                                                       main, run)

RE_TAU = 20000.
CELLS = 64
FIRST_CELL_Y_PLUS = 0.4


# --------------------------------------------------------------------------- #
# synthetic layer
# --------------------------------------------------------------------------- #

def test_s_r_is_continuous_at_the_switch_and_its_branches_meet():
    below, at, above = wilcox_s_r(np.array([S_R_SWITCH - 1e-9, S_R_SWITCH, S_R_SWITCH + 1e-9]))
    # the two branches are (50/k_s+)^2 and 100/k_s+: both give 4 at k_s+ = 25
    assert below == pytest.approx(4.0, rel=1e-8)
    assert at == pytest.approx(4.0, rel=1e-15)
    assert above == pytest.approx(4.0, rel=1e-8)


def test_s_r_decreases_with_roughness():
    k_s_plus = np.array([2., 5., 10., 25., 50., 100., 400.])
    s_r = wilcox_s_r(k_s_plus)
    assert np.all(np.diff(s_r) < 0)


def test_nikuradse_shift_matches_the_two_log_laws():
    # du+ = (B_rough - B_smooth) - ln(k_s+)/kappa, and it must equal the
    # difference of the two laws at any y+ (the shift is y-independent).
    for k_s_plus in (30., 100., 1000.):
        shift = float(nikuradse_shift(k_s_plus))
        y_plus = 10. * k_s_plus
        smooth = np.log(y_plus) / KAPPA + B_SMOOTH
        rough = float(nikuradse_u_plus(y_plus, k_s_plus))
        assert rough - smooth == pytest.approx(shift, rel=1e-12)
    assert float(nikuradse_shift(100.)) == pytest.approx(3.3 - np.log(100.) / KAPPA, rel=1e-13)


def test_shift_rejects_nonphysical_roughness():
    with pytest.raises(ValueError):
        nikuradse_shift(0.)
    with pytest.raises(ValueError):
        nikuradse_u_plus(100., -1.)


def test_fit_rough_log_law_recovers_a_manufactured_law():
    k_s_plus, kappa, b = 120., 0.41, 8.5
    y_plus = np.geomspace(3 * k_s_plus, 4000., 200)
    u_plus = nikuradse_u_plus(y_plus, k_s_plus, kappa=kappa, b_rough=b)
    fit = fit_rough_log_law(y_plus, u_plus, low=3 * k_s_plus, high=4000., k_s_plus=k_s_plus)
    assert fit['kappa_fit'] == pytest.approx(kappa, rel=1e-10)
    assert fit['b_fit'] == pytest.approx(b, abs=1e-10)
    assert fit['rms_u_plus'] < 1e-10
    assert fit['points'] == 200


def test_fit_reports_none_when_the_band_is_empty():
    y_plus = np.array([1., 2., 3.])
    assert fit_rough_log_law(y_plus, y_plus, low=10., high=20., k_s_plus=1.) is None


# --------------------------------------------------------------------------- #
# contract layer
# --------------------------------------------------------------------------- #

def test_rough_wall_value_caps_the_smooth_value_and_records_the_branch():
    smooth_value = 1e6
    value, info = rough_wall_value('komega', k_s_plus=100., smooth_value=smooth_value,
                                   u_tau_over_nu=1e4)
    assert info['branch'] == 'rough'
    assert value == pytest.approx(1e4 * float(wilcox_s_r(100.)), rel=1e-15)
    assert value < smooth_value
    # a vanishing roughness cannot make the wall rougher than the smooth value
    value_small, info_small = rough_wall_value('komega', k_s_plus=1.0,
                                               smooth_value=smooth_value, u_tau_over_nu=1e4)
    assert info_small['branch'] == 'smooth'
    assert value_small == pytest.approx(smooth_value, rel=0)


def test_rough_wall_value_refuses_the_wall_function_model():
    with pytest.raises(NotImplementedError):
        rough_wall_value('kepsilon', k_s_plus=100., smooth_value=1e6, u_tau_over_nu=1e4)


def test_rough_wall_value_refuses_invalid_inputs():
    with pytest.raises(ValueError):
        rough_wall_value('komega', k_s_plus=0., smooth_value=1e6, u_tau_over_nu=1e4)
    with pytest.raises(ValueError):
        rough_wall_value('komega', k_s_plus=100., smooth_value=1e6, u_tau_over_nu=0.)


def test_coarse_mesh_raises_instead_of_dropping_the_roughness():
    # y_1^+ = 10 puts the smooth omega below the rough one for k_s^+ = 100:
    # returning the smooth value would silently delete the roughness.
    with pytest.raises(ValueError, match='first cell'):
        solve_channel('komega', 32, re_tau=RE_TAU, first_cell_y_plus=10., k_s=100. / RE_TAU)


def test_wall_function_channel_refuses_roughness():
    with pytest.raises(NotImplementedError):
        solve_channel('kepsilon', 32, re_tau=5000., k_s=1e-3)


def test_smooth_default_is_unchanged_and_reports_no_roughness():
    plain = solve_channel('komega', 64, re_tau=2000.)
    explicit = solve_channel('komega', 64, re_tau=2000., k_s=None)
    assert plain['k_s'] is None and plain['k_s_plus'] is None and plain['roughness'] is None
    assert np.array_equal(plain['u_plus'], explicit['u_plus'])
    assert plain['bulk_u_plus'] == explicit['bulk_u_plus']


def test_k_s_plus_is_the_requested_wall_unit_height():
    result = solve_channel('komega', 64, re_tau=RE_TAU, first_cell_y_plus=FIRST_CELL_Y_PLUS,
                           k_s=100. / RE_TAU)
    assert result['k_s_plus'] == pytest.approx(100., rel=1e-12)
    assert result['roughness']['branch'] == 'rough'


def test_crossover_marks_where_roughness_begins_to_act():
    re_tau, first_cell = 20000., 0.4
    cross = crossover_k_s_plus('komega', re_tau=re_tau, first_cell_y_plus=first_cell)
    assert 0.5 < cross < 10.
    below = solve_channel('komega', 64, re_tau=re_tau, first_cell_y_plus=first_cell,
                          k_s=0.5 * cross / re_tau)
    above = solve_channel('komega', 64, re_tau=re_tau, first_cell_y_plus=first_cell,
                          k_s=2. * cross / re_tau)
    assert below['roughness']['branch'] == 'smooth'
    assert above['roughness']['branch'] == 'rough'


def test_smooth_limit_is_exact_at_the_same_grid():
    re_tau, cells, first_cell = 4000., 64, 0.4
    cross = crossover_k_s_plus('komega', re_tau=re_tau, first_cell_y_plus=first_cell)
    baseline = solve_channel('komega', cells, re_tau=re_tau, first_cell_y_plus=first_cell)
    capped = solve_channel('komega', cells, re_tau=re_tau, first_cell_y_plus=first_cell,
                           k_s=0.5 * cross / re_tau)
    assert capped['roughness']['branch'] == 'smooth'
    assert np.array_equal(baseline['u_plus'], capped['u_plus'])


# --------------------------------------------------------------------------- #
# registration layer: the closure really produces Nikuradse's roughness function
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def light():
    return run(quick=False, include_reynolds=False, re_tau=RE_TAU, cells=CELLS,
               first_cell_y_plus=FIRST_CELL_Y_PLUS, ladder=(30., 100.))


def test_watertight_smooth_limit(light):
    assert light['criteria']['R1']['passed'] is True
    assert light['smooth_limit']['komega']['max_profile_difference'] == 0.0


def test_shift_matches_nikuradse_within_half_a_wall_unit(light):
    rows = {round(row['k_s_plus']): row for row in light['komega']['rows']}
    # one third of a wall unit over u+ ~ 15 is ~2 % -- the tolerance is a
    # physical one (sand-grain data scatter), not a regression band
    for k_s_plus, row in rows.items():
        if not np.isfinite(row['error']):
            continue
        # 0.50 w.u. is the registration bound of this reduced (64-cell) grid;
        # the graded 512-cell sweep of the report reaches 0.24 w.u. at k_s^+=100,
        # so the resolver bound does not stand in for the graded criterion
        assert abs(row['error']) <= 0.50, (k_s_plus, row['error'])
    # the criterion itself is graded against the 512-cell sweep, so on this
    # reduced grid it must be *decided* (not skipped) but may be either verdict
    assert light['criteria']['R2']['passed'] in (True, False)


def test_shift_grows_with_roughness(light):
    rows = light['komega']['rows']
    assert all(a['bulk_shift'] > b['bulk_shift'] for a, b in zip(rows, rows[1:]))


def test_kappa_of_the_rough_log_layer_is_the_models_own(light):
    for row in light['komega']['rows']:
        if not np.isfinite(row['kappa_fit']):
            continue
        # the roughness must not change the log-layer slope of the closure
        assert row['kappa_fit'] == pytest.approx(row['kappa_model'], rel=0.10)


def test_intercept_jump_matches_the_sand_grain_constant(light):
    for row in light['komega']['rows']:
        if not np.isfinite(row['intercept_jump']):
            continue
        # reduced-grid bound (the graded run gives 3.06 at k_s^+=100)
        assert row['intercept_jump'] == pytest.approx(B_ROUGH - B_SMOOTH, abs=0.40)


def test_verification_driver_is_gradable_and_honest(light):
    criteria = light['criteria']
    assert criteria['R9']['passed'] is True          # wall-function scope is explicit
    assert criteria['R7']['passed'] is True          # coarse mesh raises, y1+=3 is registered
    assert criteria['R8']['passed'] is True
    # criteria that need the full ladder are not_run here, never "passed"
    assert criteria['R4']['passed'] is None
    assert criteria['R5']['passed'] is None


def test_grade_does_not_pass_unmeasured_criteria():
    summary = dict(criteria={}, komega=dict(rows=[], slope=dict(slope=float('nan'), target=-2.4,
                                                             relative=float('nan'))),
                   sst=dict(rows=[], slope=dict(slope=float('nan'), target=-2.4,
                                                relative=float('nan'))),
                   smooth_limit={'komega': dict(k_s_plus=1., branch='rough',
                                                max_profile_difference=0.1)},
                   mesh_control=dict(error=0.0, branch='rough'),
                   coarse_control=dict(raised=False),
                   wall_function_scope=dict(raised=False, message='leak'))
    checks = grade(summary)
    assert checks['R1']['passed'] is False
    assert checks['R3']['passed'] is None
    assert checks['R9']['passed'] is False


def test_kappa_used_by_the_grades_is_the_model_implied_value():
    # guards the analytic reference against drifting to a hard-coded 0.41
    constants = constants_for('komega')
    _, smooth = wall_values('komega', 0.1, 1e-4, constants)
    assert smooth > 0
    assert implied_kappa('komega') == pytest.approx(0.4336, abs=5e-4)


# --------------------------------------------------------------------------- #
# artifact layer: the graded record in docs/ is re-derivable, never hand-written
# --------------------------------------------------------------------------- #

ARTIFACT = (Path(__file__).resolve().parents[1] / 'docs' / 'verification-rough-wall-rans'
            / 'summary.json')


def test_committed_artifact_regrades_to_itself():
    """Re-grading ``docs/.../summary.json`` from its own numbers must reproduce
    every verdict and detail string, so a hand edit of a number or of a pass flag
    fails here instead of quietly changing what the report claims."""
    summary = json.loads(ARTIFACT.read_text())
    stored = copy.deepcopy(summary['criteria'])
    summary['criteria'] = grade(summary)
    finish_criteria(summary, summary['reynolds'])
    assert sorted(summary['criteria']) == sorted(stored)
    for key, want in stored.items():
        got = summary['criteria'][key]
        assert got['passed'] is want['passed'], (key, got['detail'], want['detail'])
        assert got['detail'] == want['detail'], key
    # the honest FAILs are part of the record: a later re-run that turned them
    # green without changing the closure must trip the drift gate, not the report
    assert stored['R1']['passed'] is True
    assert stored['R2']['passed'] is False
    assert stored['R3']['passed'] is False
    assert stored['R5']['passed'] is False
    assert stored['R7']['passed'] is True


def test_artifact_holds_the_graded_ladder_for_both_models():
    summary = json.loads(ARTIFACT.read_text())
    assert [float(v) for v in summary['ladder']] == [float(v) for v in LADDER]
    for model in ('komega', 'sst'):
        rows = {round(float(r['k_s_plus'])): r for r in summary[model]['rows']}
        assert sorted(rows) == [round(v) for v in LADDER]
        for k_s_plus, row in rows.items():
            assert row['branch'] == 'rough'
            assert row['analytic'] == pytest.approx(
                float(nikuradse_shift(k_s_plus, kappa=row['kappa_model'])), rel=1e-9)
            # the graded record's split: k-omega holds Nikuradse's shift up to
            # k_s^+ = 100, SST only in the transitional point (deeper k_s^+ is the
            # registered honest FAIL of R10, so it is pinned here, not tolerated)
            bound = 0.30 if model == 'komega' else 0.10
            if model == 'komega' and k_s_plus <= 100:
                assert abs(row['error']) <= bound, (model, k_s_plus, row['error'])
            elif model == 'sst' and k_s_plus == 30:
                assert abs(row['error']) <= bound, (model, k_s_plus, row['error'])
            elif model == 'sst':
                assert row['error'] > 0.30, (model, k_s_plus, row['error'])


def test_main_writes_the_artifact_and_a_quick_run_never_overwrites_it(tmp_path, monkeypatch,
                                                                      capsys):
    stub = dict(criteria={'R1': dict(passed=True, detail='stub')}, re_tau=1., cells=1)
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        return copy.deepcopy(stub)

    monkeypatch.setattr(driver, 'run', fake_run)
    before = ARTIFACT.read_bytes()
    assert main(['--directory', str(tmp_path)]) == 0
    assert calls[-1]['quick'] is False
    assert json.loads((tmp_path / 'summary.json').read_text())['criteria']['R1']['passed'] is True
    # a smoke run prints its summary and leaves the graded record alone
    assert main(['--quick']) == 0
    assert calls[-1]['quick'] is True
    assert 'artifact NOT written' in capsys.readouterr().out
    assert ARTIFACT.read_bytes() == before
