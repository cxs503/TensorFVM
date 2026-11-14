"""Committed Kovasznay artifacts must stay in sync with the raw fields.

Two failure modes have actually happened in this case and are gated here:

1. **Stale summary.**  ``scripts/summarize_kovasznay_32_64_128.py`` grew the
   ``pair_consistency`` / ``three_level_estimates`` corroboration blocks (a
   three-grid log-log slope has no residual degrees of freedom, so the verdict
   needs independent corroboration).  The committed ``summary.json`` predated
   that change, so ``tests/test_kovasznay_32_64_128_acceptance.py`` was red on
   the merged tree while the docs still described a green gate: the evidence
   bundle, not the solver, was out of date.  A green acceptance test must never
   be the *only* thing keeping an artifact honest, so the artifact itself is
   regenerated here and compared byte for byte.

2. **Series claiming a grid it does not have.**  The negative control
   (``runs-32-64-128-firstorder``) was merged into the main line with n=32 and
   n=128 only; ``n=64`` was dropped in the merge.  Because the control test
   skips when artifacts are absent, the loss showed up as an honest "not
   measured" skip rather than as a defect, while ``NEGATIVE_CONTROL.md`` still
   quoted a 32/64/128 slope.  Here the claim is bound to the evidence: an
   artifact that reports levels 32/64/128 must have the raw field dump for
   each of those levels.

Both checks are cheap (no solving, only recomputation from committed npz).
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
REFERENCE = CASE / 'runs-32-64-128-zero_gradient'
CONTROL = CASE / 'runs-32-64-128-firstorder'
GRIDS = (32, 64, 128)
SUMMARIZER = ROOT / 'scripts' / 'summarize_kovasznay_32_64_128.py'
SENSITIVITY = ROOT / 'scripts' / 'kovasznay_order_sensitivity.py'


def _rel(path):
    """Artifacts record the run directory as given, so drive the scripts with
    repo-relative paths -- that is how the committed files were produced."""
    return Path(path).resolve().relative_to(ROOT)


def _run(args):
    env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
    completed = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                               env=env, capture_output=True, text=True)
    assert completed.returncode == 0, (
        f'{args[0]} failed rc={completed.returncode}\n'
        f'stdout:\n{completed.stdout[-4000:]}\nstderr:\n{completed.stderr[-4000:]}')
    return completed


def _requirements(directory, grids):
    return [name for name in ('cases.json', 'environment.json')
            if not (directory / name).is_file()]


def test_reference_series_has_every_level_it_reports():
    """A 32/64/128 claim requires a raw field dump for each level."""
    missing = [str(path.relative_to(ROOT))
               for n in GRIDS
               for path in (REFERENCE / f'n{n}' / f'n{n}-fields.npz',
                            REFERENCE / f'n{n}' / 'cases.json')
               if not path.is_file()]
    assert not missing, f'reference series is missing levels: {missing}'


def test_control_series_has_every_level_it_reports():
    """The negative control lost its n=64 level in a merge once; never again."""
    missing = [str(path.relative_to(ROOT))
               for n in GRIDS
               for path in (CONTROL / f'n{n}' / f'n{n}-fields.npz',
                            CONTROL / f'n{n}' / 'cases.json')
               if not path.is_file()]
    assert not missing, (
        'negative-control series is missing levels: '
        f'{missing} — NEGATIVE_CONTROL.md quotes a 32/64/128 slope, so every '
        'level it quotes must be reproducible from the committed fields')


def test_committed_summary_reproduces_byte_for_byte(tmp_path):
    """``summary.json`` must be exactly what the current summarizer emits."""
    _run([SUMMARIZER, '--runs', _rel(REFERENCE), '--output', tmp_path,
          '--markdown', tmp_path / 'RESULTS.md', '--require-grids', '32,64,128'])
    fresh = json.loads((tmp_path / 'summary.json').read_text())
    committed = json.loads((REFERENCE / 'summary.json').read_text())
    assert committed == fresh, (
        'committed summary.json is stale: regenerate it with '
        'scripts/summarize_kovasznay_32_64_128.py')
    # The corroboration blocks the acceptance gate reads must be present.
    for key in ('pair_consistency', 'three_level_estimates', 'verdict'):
        assert key in committed, f'summary.json lacks the {key} block'
    for key in ('velocity_full', 'velocity_inplane'):
        assert key in committed['three_level_estimates']
    assert committed['verdict']['corroborated'] is True
    assert [level['n'] for level in committed['levels']] == list(GRIDS)


def test_committed_negative_control_reproduces_byte_for_byte(tmp_path):
    """``order-sensitivity.json`` must be what the gate script emits today."""
    _run([SENSITIVITY, '--reference', _rel(REFERENCE), '--control', _rel(CONTROL),
          '--grids', '32,64,128', '--output', tmp_path / 'order-sensitivity.json'])
    fresh = json.loads((tmp_path / 'order-sensitivity.json').read_text())
    committed = json.loads((CASE / 'order-sensitivity.json').read_text())
    assert committed == fresh, (
        'committed order-sensitivity.json is stale: regenerate it with '
        'scripts/kovasznay_order_sensitivity.py')
    verdict = committed['verdict']
    assert verdict['control_premise_ok'] is True, (
        'the control differs from the reference in more than the convection order, '
        'so the margin would prove nothing')
    assert verdict['reference_slope'] >= 1.8
    assert verdict['control_slope'] < 1.8
    assert verdict['margin'] >= 0.5