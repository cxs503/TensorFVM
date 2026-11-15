"""The acceptance gate must actually catch corrupted evidence.

``scripts/audit_kovasznay_acceptance_gate_mutations.py`` corrupts a *copy* of the
32/64/128 ``zero_gradient`` artifact set (hand-edited ``cases.json``, a field dump
replaced by the analytic solution, a hand-edited slope, an overstated verdict, a
relaxed threshold) and reruns the real gate checks against each copy.  The
expected failure set of every mutation is declared in the script before it runs,
so a corruption that slips through shows up as a gate defect.

These tests run that audit and pin its outcome: without them "the gate is
evidence-bound" would be an assertion in a docstring rather than a measured
property.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts' / 'audit_kovasznay_acceptance_gate_mutations.py'
ANCHOR_ZG = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'runs-32-64-128-zero_gradient'


def _module():
    spec = importlib.util.spec_from_file_location('kovasznay_gate_mutations', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def audit(tmp_path_factory):
    if not (ANCHOR_ZG / 'summary.json').is_file():
        pytest.skip('zero_gradient 32/64/128 artifacts absent')
    module = _module()
    payload = module.run(ANCHOR_ZG, tmp_path_factory.mktemp('mutation') / 'audit.json')
    return module, payload


def test_control_copy_passes_the_gate(audit):
    _, payload = audit
    control = payload['mutations'][0]
    assert control['mutation'].startswith('none')
    assert control['observed_failures'] == []


def test_every_declared_mutation_is_detected(audit):
    _, payload = audit
    assert payload['undetected'] == []
    assert payload['verdict'] == 'PASS'
    rows = {row['mutation']: row for row in payload['mutations']}
    assert set(rows) == set(_module().MUTATIONS) | {'none (control)'}
    for name, row in rows.items():
        if name.startswith('none'):
            continue
        assert row['detected'] is True, f'{name} slipped past the gate: {row}'
        assert set(row['expected_failures']) <= set(row['observed_failures'])


def test_hand_edited_numbers_are_caught_by_the_recompute_check(audit):
    _, payload = audit
    rows = {row['mutation']: row for row in payload['mutations']}
    assert 'recorded_errors_match_fields' in rows['hand_edited_error']['observed_failures']
    assert 'reported_slopes_match_fields' in rows['hand_edited_slope']['observed_failures']


def test_tampering_with_the_published_threshold_changes_nothing(audit):
    """The criterion is fixed in code, so relaxing it in summary.json is inert."""
    _, payload = audit
    rows = {row['mutation']: row for row in payload['mutations']}
    relaxed = rows['relaxed_threshold_in_summary']
    assert relaxed['observed_failures'] == []
    assert relaxed['detected'] is True