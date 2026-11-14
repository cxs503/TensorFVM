"""Tests for the KMM180 number-provenance gate.

The gate's job is to fail, so every test here is an attempt to make it fail for
the right reason:

* the published bundle must pass;
* a decimal moved by one unit of its last quoted digit must be caught -- this is
  the mutation that matters, because "we only changed the last digit" is exactly
  how a superseded vintage survives in a catalog;
* a value that only the ``Re_tau = 182.088`` solve reproduces must be reported
  as ``SNAPSHOT_ONLY`` rather than accepted;
* the canonical artifact must say which Re_tau it was solved at.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from tensorfvm.verification import kmm180_number_provenance as prov

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def catalog_copy(tmp_path):
    target = tmp_path / 'CATALOG.json'
    shutil.copy(prov.CATALOG, target)
    return target


def _patch(path, dotted, value):
    data = json.loads(path.read_text())
    node = data
    *parents, leaf = dotted.split('.')
    for part in parents:
        node = node[part]
    node[leaf] = value
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


# --------------------------------------------------------------------------- #
# 1. The published bundle
# --------------------------------------------------------------------------- #

def test_gate_passes_on_the_published_bundle():
    result = prov.audit()
    assert result['verdict'] == 'PASS', prov.summarise(result)
    assert result['blocking'] == []
    # every graded decimal is claimed, not silently skipped
    assert len(result['claims']) >= 30


def test_canonical_vintage_is_recorded_by_the_artifact():
    """An artifact that omits its Re_tau makes every (model, n) label ambiguous."""
    data = json.loads((prov.CASE_DIR / 'benchmark.json').read_text())
    assert 'nominal_re_tau' in data
    assert float(data['nominal_re_tau']) == prov.canonical_re_tau()
    assert prov.audit()['canonical_re_tau_recorded'] is True


def test_every_record_carries_a_vintage():
    records, present = prov.load_records()
    assert records and present
    missing = [r for r in records if r['re_tau'] is None]
    assert missing == [], f'{len(missing)} records without a Re_tau'
    # the bundle genuinely spans more than one solve, which is why the key matters
    assert len({r['re_tau'] for r in records}) >= 2


# --------------------------------------------------------------------------- #
# 2. Mutations the gate must catch
# --------------------------------------------------------------------------- #

def test_last_digit_change_that_reintroduces_the_superseded_vintage(catalog_copy):
    """0.03439 -> 0.03456 is the value the Re_tau=182.088 solve produces."""
    _patch(catalog_copy, 'grading_windows.legacy_full.sst_rms_relative', 0.03456)
    result = prov.audit(catalog_path=catalog_copy)
    assert result['verdict'] == 'FAIL'
    blocking = [b for b in result['blocking'] if b['kind'] == 'number']
    assert [b['field'] for b in blocking] == [
        'grading_windows.legacy_full.sst_rms_relative']
    assert blocking[0]['status'] == 'SNAPSHOT_ONLY'
    assert blocking[0]['matching_vintages'] == [182.088]


def test_grid_count_with_no_record_at_all_is_reported_as_mismatch(catalog_copy):
    """96 appears in the bundle's history but in no artifact any more."""
    _patch(catalog_copy, 'grading_windows.legacy_full.cells_n256', 96)
    result = prov.audit(catalog_path=catalog_copy)
    blocking = [b for b in result['blocking'] if b['kind'] == 'number']
    assert len(blocking) == 1
    assert blocking[0]['status'] == 'MISMATCH'
    assert blocking[0]['resolved'] is None


def test_superseded_vintage_in_prose_is_flagged(catalog_copy):
    data = json.loads(catalog_copy.read_text())
    data['summary'] = data['summary'] + ' (unsourceable restatement: 12.34 %)'
    catalog_copy.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')
    result = prov.audit(catalog_path=catalog_copy)
    prose = [b for b in result['blocking'] if b['kind'] == 'text']
    assert prose and prose[0]['status'] in ('SNAPSHOT_ONLY', 'UNTRACEABLE')
    assert '12.34' in prose[0]['token'] or prose[0]['status'] == 'SNAPSHOT_ONLY'


def test_verdict_flip_claim_must_match_an_artifact(catalog_copy):
    _patch(catalog_copy, 'grading_windows.log_layer.sst_verdict', 'PASS')
    result = prov.audit(catalog_path=catalog_copy)
    blocking = [b for b in result['blocking'] if b['kind'] == 'number']
    assert [b['field'] for b in blocking] == ['grading_windows.log_layer.sst_verdict']
    assert blocking[0]['status'] == 'MISMATCH'


# --------------------------------------------------------------------------- #
# 3. Repair is a copy, not a retyping
# --------------------------------------------------------------------------- #

def test_repair_copies_the_artifact_value_to_the_quoted_precision(catalog_copy):
    _patch(catalog_copy, 'grading_windows.legacy_full.sst_rms_relative', 0.03456)
    catalog, changes = prov.repair(catalog_path=catalog_copy)
    assert changes and changes[0]['field'] == 'grading_windows.legacy_full.sst_rms_relative'
    # 5 decimals in, 5 decimals out, and the value is the canonical record's
    repaired = catalog['grading_windows']['legacy_full']['sst_rms_relative']
    assert str(repaired) == '0.03439'
    records, _ = prov.load_records()
    canonical = [r for r in records if r['quantity'] == 'rms_relative'
                 and r['model'] == 'sst' and r['n'] == 256
                 and r['window'] == prov.LEGACY
                 and r['re_tau'] == prov.canonical_re_tau()]
    assert float(repaired) == float(f'{canonical[0]["value"]:.5f}')