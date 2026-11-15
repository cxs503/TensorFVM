"""Regression gate for the ``CASE-FVM-CHANNEL-KMM180`` case-pool entry.

What these tests defend
-----------------------
A pool record is only worth something if it can be *wrong*.  So the assertions
are of four kinds:

1. **Contract fidelity** -- the tolerance and window in the record are the
   ledger's, inherited from the grading module rather than re-typed.
2. **Recomputation** -- the graded numbers are reproduced by an independent code
   path (raw CSV + PCHIP in the test) and agree with the committed
   ``kmm180_direct_anchor`` summary to ``1e-9``.
3. **The gate can fail** -- a degraded profile must be rejected; every closure
   admitted would mean the criteria no longer discriminate.
4. **Missing data is not zero** -- the log-layer window is ``NOT_GRADED`` at
   ``n = 64`` and must not enter any worst-case number.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from tensorfvm.verification import kmm180_case_pool as pool
from tensorfvm.verification.kmm180_direct_anchor import reference_windows
from tensorfvm.verification.kmm1987 import TOLERANCE_LOG_REGION


@pytest.fixture(scope='module')
def entry():
    record = pool.case_pool_entry()
    record['audit'] = pool.audit_direct_anchor_summary(record)
    return record


# --------------------------------------------------------------------------- #
# 1. Contract fidelity
# --------------------------------------------------------------------------- #

def test_tolerance_is_the_ledger_tolerance():
    """The pool tolerance is the grading constant, not a number typed here."""
    assert pool.POOL_TOLERANCE == pytest.approx(0.05, abs=0.0)
    assert pool.POOL_TOLERANCE is TOLERANCE_LOG_REGION
    assert pool.LEDGER_CONTRACT['tolerance_percent'] == pytest.approx(
        100.0 * pool.POOL_TOLERANCE)
    assert pool.LEDGER_CONTRACT['slug'] == 'CASE-FVM-CHANNEL-KMM180'


def test_criteria_table_is_well_formed():
    names = [c['name'] for c in pool.CRITERIA]
    assert len(names) == len(set(names)), 'duplicate criterion name'
    assert set(pool.INDICATOR_ORDER) == {c['indicator'] for c in pool.CRITERIA}
    for criterion in pool.CRITERIA:
        assert criterion['comparator'] in pool._COMPARATORS
        assert criterion['threshold'] is not None
        if criterion['primary']:
            assert 'excluded_reason' not in criterion, (
                f"{criterion['name']} is a gate and needs no exclusion reason")
        else:
            assert criterion.get('excluded_reason'), (
                f"{criterion['name']} is a diagnostic and must justify it")
    # The gate must be graded on the worst grid, and a diagnostic must exist for
    # the physically meaningful log layer: both are load-bearing design choices.
    assert any(c['name'] == 'u_plus_rms_full' and c['primary']
               for c in pool.CRITERIA)
    assert any(c['name'] == 'u_plus_rms_log_layer' and not c['primary']
               for c in pool.CRITERIA)


# --------------------------------------------------------------------------- #
# 2. Recomputation
# --------------------------------------------------------------------------- #

def test_ladder_is_complete_and_monotone():
    profiles = pool.load_profiles()
    assert set(profiles) >= {(m, n) for m in pool.MODELS for n in pool.LADDER}
    for (model, n), data in profiles.items():
        assert np.all(np.diff(data['y_plus']) > 0), f'{model} n={n}: y+ not increasing'
        # Each profile holds exactly its own cell count: no silent truncation.
        assert data['y_plus'].size == n, f'{model} n={n}: {data["y_plus"].size} points'
    # The closures are solved on *different meshes by design*: SST and k-omega
    # resolve the sublayer (first cell y+ ~ 0.2, halving with refinement), while
    # k-epsilon runs its wall-function mesh with the first cell at y+ ~ 30
    # (benchmark_channel_kmm1987 sets first_cell_y_plus = 30 for that model).
    # That single fact explains why k-epsilon has 12 gradable log-layer cells at
    # n=64 while SST has 2 -- it is not a solver error and not an artifact defect.
    for n in pool.LADDER:
        assert profiles[('sst', n)]['y_plus'][0] < 1.0
        assert profiles[('komega', n)]['y_plus'][0] < 1.0
        assert 25.0 < profiles[('kepsilon', n)]['y_plus'][0] < 40.0
    assert profiles[('sst', 64)]['y_plus'][0] > profiles[('sst', 256)]['y_plus'][0]


def test_numbers_recomputed_by_an_independent_path(entry):
    """Recompute the SST full-window deviation in the test, from raw CSV + PCHIP."""
    dns = pool.anchor_reference()
    profiles = pool.load_profiles()
    for n in pool.LADDER:
        y_plus = profiles[('sst', n)]['y_plus']
        u_plus = profiles[('sst', n)]['u_plus']
        upper = float(np.max(dns['y_plus']))
        mask = (y_plus >= 30.0) & (y_plus <= upper)
        reference = PchipInterpolator(dns['y_plus'], dns['u_plus'])(y_plus[mask])
        relative = (u_plus[mask] - reference) / reference
        independent = float(np.sqrt(np.mean(relative ** 2)))
        committed = entry['model_summaries']['sst']['rms_full_by_grid'][n]
        assert independent == pytest.approx(committed, abs=1e-12), (
            f'n={n}: record {committed} != independent recomputation {independent}')


def test_recomputation_matches_the_direct_anchor_summary(entry):
    """Cross-artifact audit against the module that first graded the anchor ref."""
    audit = entry['audit']
    assert audit['available'], audit
    assert audit['compared'] >= 20, audit
    assert audit['max_abs_difference'] < 1e-9, audit


def test_criteria_values_agree_with_model_summaries(entry):
    """No criterion may grade a number that the matrix table shows differently."""
    designated = entry['model_summaries'][entry['designated_model']]
    by_name = {c['name']: c for c in entry['criteria']}
    assert by_name['u_plus_rms_full']['value'] == pytest.approx(
        designated['rms_full_worst'], abs=1e-15)
    assert by_name['u_plus_rms_log_layer']['value'] == pytest.approx(
        designated['rms_log_worst_gradable'], abs=1e-15)
    assert by_name['cf']['value'] == pytest.approx(designated['cf_rel_worst'], abs=1e-15)
    # The worst case must actually be the maximum over the ladder.
    assert designated['rms_full_worst'] == pytest.approx(
        max(designated['rms_full_by_grid'].values()), abs=1e-15)


# --------------------------------------------------------------------------- #
# 3. The gate can fail
# --------------------------------------------------------------------------- #

def test_designated_closure_admitted_controls_rejected(entry):
    """The matrix must split: SST inside 5 %, k-omega / k-epsilon outside."""
    assert entry['admitted_models'] == ['sst']
    assert set(entry['rejected_models']) == {'komega', 'kepsilon'}
    for model in entry['rejected_models']:
        summary = entry['model_summaries'][model]
        assert summary['rms_full_worst'] > entry['tolerance'], model
        assert all(v == 'FAIL' for v in summary['full_verdicts'].values()), model
    sst = entry['model_summaries']['sst']
    assert sst['rms_full_worst'] < entry['tolerance']
    assert sst['verdict_margin_to_threshold'] > 0.0
    pool.check_ladder_entries(entry)


def test_degraded_profile_is_rejected():
    """Inflate the SST profile by 20 %: the pool entry must flip to REJECTED."""
    profiles = pool.load_profiles()
    corrupted = {key: dict(value) for key, value in profiles.items()}
    corrupted[('sst', 256)] = dict(
        y_plus=profiles[('sst', 256)]['y_plus'],
        u_plus=profiles[('sst', 256)]['u_plus'] * 1.20)
    degraded = pool.case_pool_entry(profiles=corrupted)
    assert degraded['pool_status'] == 'REJECTED'
    assert 'u_plus_rms_full' in degraded['failed_primary']
    assert degraded['model_summaries']['sst']['rms_full_worst'] > 0.10


def test_disposition_semantics():
    """WITHHELD (missing data) must not be reported as REJECTED (measured failure)."""
    base = dict(quantity='q', indicator='v', comparator='lt', threshold=0.05,
                unit='-', source='test')
    primary_ok = dict(base, name='p', primary=True, excluded_reason=None)
    diag_fail = dict(base, name='d', primary=False, excluded_reason='diagnostic')

    def rows(value_primary, value_diag):
        out = []
        for criterion, value in ((primary_ok, value_primary), (diag_fail, value_diag)):
            row = dict(criterion)
            row['value'] = value
            row['pass'] = None if value is None else bool(value < 0.05)
            out.append(row)
        return out

    assert pool._dispositions(rows(0.01, 0.01))['pool_status'] == 'ADMITTED'
    assert pool._dispositions(rows(0.01, 0.09))['pool_status'] == 'ADMITTED_WITH_CAVEAT'
    assert pool._dispositions(rows(0.09, 0.01))['pool_status'] == 'REJECTED'
    withheld = pool._dispositions(rows(None, 0.01))
    assert withheld['pool_status'] == 'WITHHELD'
    assert withheld['withheld_primary'] == ['p']


def test_every_closure_admitted_is_treated_as_degenerate(entry):
    """A record where nothing fails cannot demonstrate a working gate."""
    degenerate = json.loads(json.dumps(entry))
    for summary in degenerate['model_summaries'].values():
        summary['admitted'] = True
    degenerate['rejected_models'] = []
    with pytest.raises(ValueError):
        pool.check_ladder_entries(degenerate)


# --------------------------------------------------------------------------- #
# 4. Missing data is not zero
# --------------------------------------------------------------------------- #

def test_not_graded_log_layer_is_excluded_from_the_worst_case(entry):
    sst = entry['model_summaries']['sst']
    assert sst['log_verdicts'][64] == 'NOT_GRADED'
    assert sst['rms_log_by_grid'][64] is None
    assert sst['log_layer_gradable_grids'] == [128, 256]
    assert sst['rms_log_worst_gradable'] == pytest.approx(
        max(sst['rms_log_by_grid'][n] for n in (128, 256)), abs=1e-15)
    # ... and the guard that produced NOT_GRADED is the cell count, not a cheat.
    assert sst['cells_log_by_grid'][64] < pool.GRADABILITY_MIN_CELLS
    windows = reference_windows(pool.anchor_reference())
    assert windows['log_layer']['high'] < 0.25 * float(
        pool.anchor_reference()['re_tau'])


def test_reference_identity_is_the_kmm1987_lineage(entry):
    reference = entry['reference']
    assert reference['digest_matches'], reference
    assert reference['re_tau'] == pytest.approx(178.12, abs=1e-6)
    assert abs(reference['re_tau'] - 180.0) < 5.0
    # The LM2015 proxy is a documented substitute, not an equivalent: the
    # substitution must be visible as a non-zero Cf difference.
    assert 0.0 < reference['delta_Cf_relative'] < 0.01


def test_proxy_grading_changes_the_numbers():
    """Grading against the proxy instead of the anchor reference must move the value."""
    dns_anchor = pool.anchor_reference()
    dns_proxy = pool.proxy_reference()
    profiles = pool.load_profiles()
    metrics = pool.load_solver_metrics()
    anchor_row = pool.model_ladder('sst', dns_anchor, profiles, metrics)[-1]
    proxy_row = pool.model_ladder('sst', dns_proxy, profiles, metrics)[-1]
    delta = abs(anchor_row['full']['rms_relative'] - proxy_row['full']['rms_relative'])
    assert delta > 1e-4, (
        'the reference substitution has no measurable effect, which contradicts '
        'docs/verification-channel-kmm180-direct-anchor')


# --------------------------------------------------------------------------- #
# 5. CLI contract
# --------------------------------------------------------------------------- #

def test_cli_writes_entry_and_reports_status(tmp_path):
    assert pool.main(['--out', str(tmp_path), '--quiet']) == 0
    json_path = tmp_path / 'case_pool_entry.json'
    md_path = tmp_path / 'REPORT.md'
    assert json_path.exists() and md_path.exists()

    payload = json.loads(json_path.read_text())
    assert payload['schema'] == pool.SCHEMA
    assert payload['slug'] == 'CASE-FVM-CHANNEL-KMM180'
    assert payload['pool_status'] in ('ADMITTED', 'ADMITTED_WITH_CAVEAT')
    assert payload['tolerance'] == pytest.approx(0.05)
    assert payload['caveats'], 'an admitted-with-caveat record must state its caveats'

    text = md_path.read_text()
    assert payload['pool_status'] in text
    assert 'u_plus_rms_full' in text
    assert 'Re_tau' in text


def test_cli_exit_code_is_one_when_rejected(tmp_path, monkeypatch):
    profiles = pool.load_profiles()
    corrupted = {key: dict(value) for key, value in profiles.items()}
    for n in pool.LADDER:
        corrupted[('sst', n)] = dict(y_plus=profiles[('sst', n)]['y_plus'],
                                     u_plus=profiles[('sst', n)]['u_plus'] * 1.30)
    monkeypatch.setattr(pool, 'load_profiles', lambda path=None: corrupted)
    assert pool.main(['--quiet']) == 1

def test_json_round_trip_restores_grid_keys(tmp_path, entry):
    """A consumer reading the committed JSON must get the same numbers back."""
    pool.write_entry(entry, tmp_path)
    reloaded = pool.load_entry(tmp_path / 'case_pool_entry.json')
    for model, summary in entry['model_summaries'].items():
        restored = reloaded['model_summaries'][model]
        for field in pool.GRID_KEYED_FIELDS:
            assert list(restored[field]) == list(summary[field]), (model, field)
            for n, value in summary[field].items():
                assert restored[field][n] == value, (model, field, n)
    # ... and the restored record still passes the record guard.
    pool.check_ladder_entries(reloaded)
