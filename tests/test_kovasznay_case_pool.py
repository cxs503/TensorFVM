"""Regression gate for the ``CASE-FVM-KOVASZNAY-ORDER`` case-pool entry.

The entry is only worth anything if it can *reject*.  These tests therefore come
in three groups:

* **admission** — the committed 32/64/128 ladder is admitted, every graded
  number is reproduced from the raw field dumps, and the record agrees with the
  artefacts the acceptance gate already reads;
* **falsifiability** — the verdict flips to FAIL when the ladder is tampered
  with (error scaled at the finest level), the configuration is mixed, a closure
  label is contradicted by its own field dump, or a level is not converged;
* **reference provenance** — the graded reference really is the Kovasznay (1948)
  closed form: it satisfies the 1948 exponent identity, satisfies the unforced
  steady Navier--Stokes equations identically, and matches the committed
  ``exact_*`` arrays to round-off through an independent algebraic path.
"""
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification import kovasznay_case_pool as pool

ROOT = pool.case_root()
ADMITTED = ROOT / 'runs-32-64-128-zero_gradient'


@pytest.fixture(scope='module')
def payload():
    return pool.evaluate()


def _copy_ladder(tmp_path, grids=pool.GRIDS):
    """Copy the admitted ladder into a scratch docs tree (never mutate evidence)."""
    scratch = tmp_path / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
    target = scratch / ADMITTED.name
    target.mkdir(parents=True)
    for n in grids:
        shutil.copytree(ADMITTED / f'n{n}', target / f'n{n}')
    shutil.copy2(ADMITTED / 'summary.json', target / 'summary.json')
    for name, _, _ in pool.CONTROLS:
        shutil.copytree(ROOT / name, scratch / name)
    return scratch


def _rewrite(level, mutate_fields=None, **recorded):
    """Rewrite one level's record (and optionally its dump) inside a scratch tree."""
    cases_path = Path(level['paths']['cases'])
    row = json.loads(cases_path.read_text())[0]
    row.update({key: value for key, value in recorded.items() if value is not None})
    if mutate_fields is not None:
        fields_path = Path(level['paths']['fields'])
        data = dict(np.load(fields_path))
        mutate_fields(data)
        np.savez(fields_path, **data)
        volumes = data['cell_volumes']
        from tensorfvm.native_manufactured3d import volume_weighted_error
        row['velocity_l2'] = volume_weighted_error(data['velocity'], data['exact_velocity'], volumes)
        row['pressure_l2'] = volume_weighted_error(data['pressure'], data['exact_pressure'], volumes)
        row['pressure_mean_free_l2'] = volume_weighted_error(
            data['pressure'], data['exact_pressure'], volumes, remove_mean=True)
    cases_path.write_text(json.dumps([row]) + '\n')


# ---------------------------------------------------------------- admission --
def test_admission_verdict_is_pass(payload):
    assert payload['verdict']['status'] == 'PASS'
    assert payload['verdict']['failures'] == []
    assert payload['verdict']['graded_quantity'] == 'velocity_full'


def test_graded_order_clears_the_ledger_threshold(payload):
    slope = payload['fits']['velocity_full']
    assert slope['slope'] >= pool.ORDER_MIN
    assert slope['r_squared'] > 0.9999
    # The ledger contract is "L2 收敛阶 >= 1.8 on 32/64/128"; the record must
    # grade exactly that triple and that threshold.
    assert payload['grids'] == [32, 64, 128]
    assert payload['verdict']['threshold'] == 1.8


def test_both_pair_orders_clear_the_threshold(payload):
    orders = payload['pair_orders']
    assert len(orders) == 2
    assert all(order >= pool.ORDER_MIN for order in orders)
    assert max(orders) - min(orders) <= pool.PAIR_ORDER_BAND


def test_graded_errors_are_reproduced_from_the_dumps(payload):
    """Every graded number in the record equals a fresh recomputation."""
    recorded = {n: json.loads((ADMITTED / f'n{n}' / 'cases.json').read_text())[0]['velocity_l2']
                for n in pool.GRIDS}
    for level in payload['levels']:
        assert level['errors']['velocity_full'] == pytest.approx(recorded[level['n']], rel=1e-12)


def test_record_agrees_with_the_committed_summary(payload):
    summary = json.loads((ADMITTED / 'summary.json').read_text())
    assert payload['fits']['velocity_full']['slope'] == pytest.approx(
        summary['fits']['velocity_full']['slope'], rel=1e-9)
    assert payload['fits']['pressure_mean_free']['slope'] == pytest.approx(
        summary['fits']['pressure_meanfree']['slope'], rel=1e-9)


def test_primary_criteria_are_all_graded_and_diagnostic_ones_are_labelled(payload):
    primary = {row['id'] for row in payload['criteria'] if row['primary']}
    assert primary == {'order_velocity_full', 'pair_order_consistency',
                       'reference_is_kovasznay1948', 'configuration_homogeneous',
                       'closure_label_self_consistent', 'first_order_control_fails',
                       'pre_asymptotic_control_fails'}
    assert all(row['passed'] for row in payload['criteria'] if row['primary'])
    fixed = [row for row in payload['criteria'] if row['id'] == 'pressure_fixed_gauge'][0]
    assert fixed['primary'] is False and fixed['passed'] is None


def test_threshold_does_not_drift_from_the_existing_gate_and_summariser():
    """One tolerance, three code paths: pool / summariser / acceptance gate."""
    script = Path(pool.__file__).resolve().parents[3] / 'scripts' / 'summarize_kovasznay_32_64_128.py'
    spec = importlib.util.spec_from_file_location('summarize_kovasznay_32_64_128', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.THRESHOLD == pool.ORDER_MIN
    assert module.PAIR_ORDER_BAND == pool.PAIR_ORDER_BAND
    gate = Path(pool.__file__).resolve().parents[3] / 'tests' / 'test_kovasznay_32_64_128_acceptance.py'
    assert 'THRESHOLD = 1.8' in gate.read_text().replace('  ', ' ')


def test_record_is_json_serialisable_and_digest_bound(payload, tmp_path):
    text = json.dumps(payload, allow_nan=False)
    assert json.loads(text)['anchor'] == 'CASE-FVM-KOVASZNAY-ORDER'
    digests = payload['evidence']['digests']
    assert len(digests) == len(pool.GRIDS) + 1
    assert all(len(value) == 64 and set(value) <= set('0123456789abcdef')
               for value in digests.values())
    assert payload['evidence']['digests']['runs-32-64-128-zero_gradient/n128/cases.json'] == \
        pool._sha256(ADMITTED / 'n128' / 'cases.json')
    out = tmp_path / 'entry.json'
    written = pool.write_admission(out)
    assert json.loads(out.read_text())['verdict'] == written['verdict']


# ------------------------------------------------------------- falsifiability --
def test_verdict_flips_to_fail_when_finest_error_is_scaled(payload, tmp_path):
    """Anti-hardcoding: a ladder that is no longer second order must be refused."""
    scratch = _copy_ladder(tmp_path)
    finest = pool.load_level(scratch / ADMITTED.name, 128)

    def inflate(data):
        # Keep the dump self-consistent but double the error at n=128.
        data['velocity'] = data['exact_velocity'] + 2.0 * (data['velocity'] - data['exact_velocity'])

    _rewrite(finest, mutate_fields=inflate)
    tampered = pool.evaluate(root=scratch)
    assert tampered['verdict']['status'] == 'FAIL'
    assert 'order_velocity_full' in tampered['verdict']['failures']
    assert tampered['fits']['velocity_full']['slope'] < pool.ORDER_MIN


def test_hand_edited_record_is_refused(tmp_path):
    """A cases.json that the dump cannot reproduce is a hard error, not a PASS."""
    scratch = _copy_ladder(tmp_path)
    level = pool.load_level(scratch / ADMITTED.name, 64)
    _rewrite(level, velocity_l2=level['recorded']['velocity_l2'] * 0.5)
    with pytest.raises(pool.CasePoolError, match='not reproducible'):
        pool.load_level(scratch / ADMITTED.name, 64)


def test_mixed_configuration_is_refused():
    levels = pool.load_sequence(ADMITTED)
    levels[1]['config']['convection_order'] = 1
    with pytest.raises(pool.CasePoolError, match='convection_order: mixed sequence'):
        pool.validate_sequence(levels)


def test_unadmitted_configuration_is_refused():
    levels = pool.load_sequence(ADMITTED)
    for level in levels:
        level['config']['reynolds'] = 41.0
    with pytest.raises(pool.CasePoolError, match='admitted configuration'):
        pool.validate_sequence(levels)


def test_closure_label_contradicted_by_the_dump_is_refused():
    levels = pool.load_sequence(ADMITTED)
    levels[0]['max_abs_z_velocity'] = 1e-3          # what the `free` closure leaves
    with pytest.raises(pool.CasePoolError, match='labelled zero_gradient'):
        pool.validate_sequence(levels)


def test_unconverged_level_is_refused():
    levels = pool.load_sequence(ADMITTED)
    levels[-1]['numerically_converged'] = False
    with pytest.raises(pool.CasePoolError, match='did not converge'):
        pool.validate_sequence(levels)


def test_both_adversarial_controls_fail_the_criterion(payload):
    """If a first-order or a pre-asymptotic ladder passed, the gate would be vacuous."""
    control = payload['controls']['runs-32-64-128-firstorder']
    assert control['configuration']['convection_order'] == 1
    assert control['velocity_full_slope'] < pool.ORDER_MIN and control['passed']
    preasymptotic = payload['controls']['selftest-4-8-16']
    assert preasymptotic['levels'] == [4, 8, 16]
    assert preasymptotic['velocity_full_slope'] < pool.ORDER_MIN and preasymptotic['passed']


def test_slope_estimator_recovers_a_known_order():
    """Unit check on the estimator itself: h^2 errors must be reported as 2."""
    levels = [{'n': n, 'h': 1.0 / n, 'errors': {'velocity_full': 4.0 / n ** 2}}
              for n in (32, 64, 128)]
    fit = pool.lsq_slope(levels)
    assert fit['slope'] == pytest.approx(2.0, abs=1e-12)
    assert pool.pair_orders(levels) == pytest.approx([2.0, 2.0], abs=1e-12)


# --------------------------------------------------------- reference provenance --
def test_reference_satisfies_the_1948_exponent_identity():
    identity = pool.lambda_identity(40.0)
    assert identity['identity_residual'] <= 1e-12
    assert identity['form_gap'] <= 1e-12
    assert identity['lambda'] < 0                       # decay downstream
    assert pool.lam_1948(40.0) != pool.lam_1948(41.0)


def test_reference_satisfies_unforced_steady_navier_stokes():
    """The reference needs no body force; a look-alike field would not pass this."""
    points = np.array([[0.13, 0.21, 0.0], [0.5, 0.44, 0.5], [0.87, 0.70, 1.0]])
    residual = pool.momentum_residual(points, 40.0)
    assert np.max(np.abs(residual)) <= 1e-14


def test_committed_reference_field_is_the_closed_form(payload):
    audit = payload['reference_audit']
    assert audit['velocity_max_abs_gap'] <= 1e-12
    assert audit['pressure_max_abs_gap'] <= 1e-12
    assert audit['momentum_residual_max_scaled'] <= 1e-12
    assert audit['points_checked'] == 16384
    assert audit['source_doi'] == '10.1017/s0305004100023999'


def test_closed_form_is_independent_of_the_solver_side_implementation():
    """Two algebraic paths to the same 1948 solution must agree to round-off."""
    from tensorfvm.verification.native_hybrid_ns import kovasznay_fields
    points = np.column_stack((np.linspace(0.01, 0.99, 40), np.linspace(0.02, 0.97, 40),
                              np.zeros(40)))
    velocity, pressure = pool.exact_fields(points, 40.0)
    reference = kovasznay_fields(points, 40.0)
    assert np.max(np.abs(velocity - reference['velocity'])) <= 1e-14
    assert np.max(np.abs(pressure - reference['pressure'])) <= 1e-14

# ------------------------------------------------------------ committed entry --
def test_committed_entry_is_fresh(payload):
    """The published record must equal a fresh adjudication of the evidence."""
    entry_path = (Path(pool.__file__).resolve().parents[3] / 'docs'
                  / 'case-fvm-kovasznay-order-pool' / 'case_pool_entry.json')
    committed = json.loads(entry_path.read_text())
    assert committed['verdict']['status'] == payload['verdict']['status']
    assert committed['verdict']['observed'] == pytest.approx(
        payload['verdict']['observed'], rel=1e-12)
    for key in ('velocity_full', 'pressure_mean_free', 'pressure_fixed', 'velocity_inplane'):
        assert committed['fits'][key]['slope'] == pytest.approx(payload['fits'][key]['slope'],
                                                                rel=1e-12)
    assert committed['evidence']['digests'] == payload['evidence']['digests']
    assert committed['criteria'] == payload['criteria']
