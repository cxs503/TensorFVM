"""Gates for the evidence-resolvability audit (``scripts/audit_verification_evidence_resolvability.py``).

The audit answers one machine-answerable question per citation in ``docs/``: can a
reader of *this commit* open the artifact the claim rests on?  These tests pin the
three things that make such a ledger trustworthy:

* classification -- each reference lands in exactly one bucket, and the buckets are
  decided by the filesystem, not by the report's own prose (positive + negative
  controls on a synthetic tree, including a planted bogus reference that must be
  reported as ``missing``);
* ratchet -- ``--check`` fails on a reference the committed baseline does not
  record, so debt stays visible instead of new debt being absorbed silently;
* no drift -- the committed ledger (``docs/evidence-resolvability/audit.json``)
  still matches the tree it was generated from.
"""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / 'scripts/audit_verification_evidence_resolvability.py'


def _audit():
    spec = importlib.util.spec_from_file_location('evidence_resolvability', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = _audit()


def _tree(tmp_path, body, allowlist=None):
    docs = tmp_path / 'docs'
    (docs / 'bundle').mkdir(parents=True)
    (docs / 'bundle' / 'report.md').write_text(body)
    (docs / 'bundle' / 'profile.png').write_bytes(b'\x89PNG')
    patterns = (allowlist or {}).get('patterns', [])
    reports, totals, stale = audit.scan(docs, tmp_path, patterns)
    return audit.summarise(reports, totals, stale), totals, stale


def test_buckets_are_decided_by_the_filesystem(tmp_path):
    summary, _, _ = _tree(tmp_path, '\n'.join([
        '![profile](profile.png)',
        '[raw](fields.npz)',
        '[gen](outputs/ladder/summary.json)',
        'Artefacts: `outputs/abc/result.json`.',
    ]))
    buckets = summary['buckets']
    assert buckets['resolved'] == 1
    # every other citation names a file that does not exist
    assert buckets['missing'] == 1
    # outputs/ is never versioned, so a citation there is reproducible but not evidence
    assert buckets['regenerable'] == 2


def test_planted_bogus_reference_is_reported_missing(tmp_path):
    """Negative control: a citation that cannot be opened must not be ignored."""
    summary, _, _ = _tree(tmp_path, '[raw fields](n-99/fields.npz)\n')
    assert summary['buckets']['missing'] == 1
    entry = summary['missing'][0]
    assert entry['path'] == 'docs/bundle/n-99/fields.npz'
    assert entry['report'] == 'docs/bundle/report.md'
    assert summary['reports_with_missing'] == 1


def test_allowlist_is_reported_and_stale_entries_are_flagged(tmp_path):
    allowlist = {'patterns': [
        {'glob': 'docs/bundle/*/fields.npz', 'reason': 'gate-filtered binary, regenerable by the documented run'},
        {'glob': 'docs/nowhere/*.png', 'reason': 'no longer cited'},
    ]}
    summary, _, stale = _tree(tmp_path, '[raw](n-9/fields.npz)\n', allowlist)
    assert summary['buckets']['allowlisted'] == 1
    assert summary['buckets']['missing'] == 0
    assert summary['stale_allowlist'] == ['docs/nowhere/*.png']


def test_regressions_only_returns_what_the_baseline_does_not_record():
    baseline = {'missing': [{'report': 'docs/a/report.md', 'path': 'docs/a/known.npz'}]}
    summary = {'missing': [
        {'report': 'docs/a/report.md', 'path': 'docs/a/known.npz'},
        {'report': 'docs/a/report.md', 'path': 'docs/a/new.npz'},
    ]}
    new = audit.regressions(summary, baseline)
    assert [(entry['report'], entry['path']) for entry in new] == [('docs/a/report.md', 'docs/a/new.npz')]


def test_committed_ledger_still_matches_the_tree():
    """The committed baseline must cover today's citations, and must have been produced here."""
    baseline_path = REPO / 'docs/evidence-resolvability/audit.json'
    if not baseline_path.exists():
        pytest.skip('evidence-resolvability ledger not committed yet')
    baseline = json.loads(baseline_path.read_text())
    patterns = []
    allowlist_path = REPO / 'docs/evidence-resolvability-allowlist.json'
    if allowlist_path.exists():
        patterns = json.loads(allowlist_path.read_text())['patterns']
    reports, totals, stale = audit.scan(REPO / 'docs', REPO, patterns)
    summary = audit.summarise(reports, totals, stale)
    assert audit.regressions(summary, baseline) == []
    assert stale == []
    assert summary['buckets'] == baseline['buckets']