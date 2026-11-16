"""The Kovasznay 32/64/128 close-out must be *machine* regenerable, end to end.

Two drifts actually happened in this case and are gated here:

1. **Doc drifted from the data it quotes.**  ``summary.json`` / ``RESULTS.md`` /
   ``closeout-32-64-128.md`` were committed before the summarizer grew the
   ``pair_consistency`` / ``three_level_estimates`` / ``provenance.metadata``
   blocks.  The committed doc therefore printed a *shortened* verdict json, and
   ``tests/test_kovasznay_committed_artifacts_fresh.py`` was red on the merged
   tree while the docs described a green gate.  The drift was invisible unless
   someone thought to re-run the generator.

2. **Part of the doc was hand written.**  §8 (the first-order negative control)
   had been appended by hand, so ``docs/CASE-FVM-KOVASZNAY-ORDER/report_payload.py``
   could not reproduce its own deliverable: regenerating the doc silently
   deleted the paragraph that shows the criterion discriminates.  A generator
   that drops evidence on re-run cannot be used to keep evidence fresh.

Both artefacts are regenerated here into a scratch directory and compared byte
for byte with what is committed; the payload's contract is asserted separately.
Nothing here solves anything — it is pure recomputation from committed npz/json.
"""
import base64
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
RUNS = CASE / 'runs-32-64-128-zero_gradient'
GENERATOR = CASE / 'report_payload.py'
SUMMARIZER = ROOT / 'scripts' / 'summarize_kovasznay_32_64_128.py'
GRIDS = (32, 64, 128)
MISSION_ID = 20033
SECTIONS = ('## 一、案例介绍', '## 二、怎么做的', '## 三、计算结果', '## 四、结果图',
            '## 五、注意点', '## 六、误差判定')


def _run(args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    completed = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                               env=env, capture_output=True, text=True)
    assert completed.returncode == 0, (
        f'{args[0]} failed rc={completed.returncode}\n'
        f'stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}')
    return completed


def _rel(path):
    return Path(path).resolve().relative_to(ROOT)


def test_closeout_doc_reproduces_from_committed_artifacts(tmp_path):
    """``closeout-32-64-128.md`` must be exactly what the generator emits."""
    committed_path = CASE / 'closeout-32-64-128.md'
    committed = committed_path.read_text()
    _run([GENERATOR, '--out-doc', tmp_path / 'closeout.md',
          '--out-payload', tmp_path / 'report.json'])
    fresh = (tmp_path / 'closeout.md').read_text()
    assert committed == fresh, (
        'closeout-32-64-128.md is not reproducible from the run artifacts: '
        'regenerate it with `python docs/CASE-FVM-KOVASZNAY-ORDER/report_payload.py`')
    # Regeneration must not have written the committed copy behind our back.
    assert committed_path.read_text() == committed


def test_results_markdown_reproduces_from_committed_fields(tmp_path):
    """``RESULTS.md`` in the run directory must match the summarizer's output."""
    _run([SUMMARIZER, '--runs', _rel(RUNS), '--output', tmp_path,
          '--markdown', tmp_path / 'RESULTS.md', '--require-grids', '32,64,128'])
    assert (RUNS / 'RESULTS.md').read_text() == (tmp_path / 'RESULTS.md').read_text()


def test_payload_contract(tmp_path):
    """Mission id, six mandatory sections, and a real embedded figure."""
    _run([GENERATOR, '--out-doc', tmp_path / 'closeout.md',
          '--out-payload', tmp_path / 'report.json'])
    payload = json.loads((tmp_path / 'report.json').read_text())
    assert payload['mission_id'] == MISSION_ID
    assert payload['scenario_key'] == 'tensorfvm.case.kovasznay-order'
    assert 'CASE-FVM-KOVASZNAY-ORDER' in payload['tags']
    assert payload['title'] and payload['summary']
    for section in SECTIONS:
        assert section in payload['markdown'], f'payload lost section {section!r}'
    figures = payload['figures']
    assert figures, 'report payload carries no figure'
    for figure in figures:
        assert figure['name'] and figure['caption']
        blob = base64.b64decode(figure['png_base64'])
        assert blob[:8] == b'\x89PNG\r\n\x1a\n', f"{figure['name']} is not a PNG"


def test_payload_headline_matches_committed_summary(tmp_path):
    """The slope quoted to the platform is the one in ``summary.json``."""
    _run([GENERATOR, '--out-doc', tmp_path / 'closeout.md',
          '--out-payload', tmp_path / 'report.json'])
    payload = json.loads((tmp_path / 'report.json').read_text())
    summary = json.loads((RUNS / 'summary.json').read_text())
    slope = summary['fits']['velocity_full']['slope']
    assert f'{slope:.4f}' in payload['summary']
    assert f'{slope:.4f}' in payload['markdown']
    assert summary['verdict']['pass'] is True
    assert [level['n'] for level in summary['levels']] == list(GRIDS)


def test_summary_provenance_metadata_matches_cases_json():
    """The closure recorded in provenance is the closure the levels declare.

    ``provenance.metadata`` is the block whose absence made the committed
    summary stale; a present-but-wrong block would be worse than a missing one,
    so the declared closure is re-read from each level's own ``cases.json``.
    """
    summary = json.loads((RUNS / 'summary.json').read_text())
    metadata = summary['provenance']['metadata']
    assert metadata['metadata_complete'] is True
    assert metadata['reconstructed'] == []
    for n in GRIDS:
        declared = json.loads((RUNS / f'n{n}' / 'cases.json').read_text())[0]['symmetry_closure']
        assert metadata['declared'][str(n)] == declared, f'n={n} provenance/metadata disagree'
        assert metadata['from_fields'][str(n)] == declared, f'n={n} field dump disagrees'
    assert metadata['closure'] == 'zero_gradient'
    for level in summary['levels']:
        assert level['configuration']['symmetry_closure'] == metadata['closure']