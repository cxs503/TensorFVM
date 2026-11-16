"""The 32/64/128 ladder's scalars must survive loss of the binary dumps.

``CASE-FVM-KOVASZNAY-ORDER`` keeps its evidence in ``n<N>-fields.npz`` only.  Twice
a cluster submission came back ``rc=0`` with the output directory missing
(``cj2912``, ``cj3258``), and the fourth level was recorded as honest WITHHELD
because of it: a binary-only evidence bundle is one delivery filter away from
having no evidence at all.  ``docs/CASE-FVM-KOVASZNAY-ORDER/ORDER-32-64-128-256-
ATTEMPT.md`` §5.1 therefore asks for the scalars to travel as text.

The committed ``n<N>/n<N>-evidence.md`` packs make that claim falsifiable, and
these tests are the falsifiers:

* the pack's embedded ``cases.json`` row must equal the file on disk (no drift),
  and its ``binary_dump`` block must be bound to the committed ``npz`` by sha256
  and byte count -- text and binary are provably the *same* run;
* the L2 indicators **recomputed from the pack's CSV alone** must reproduce the
  recorded full-grid values inside a pre-declared band, and the 32/64/128 slope
  rebuilt from those text numbers must still clear the ``1.8`` acceptance
  threshold and agree with the committed ``summary.json``;
* the pack must be a *function* of the committed dump: re-rendering it from the
  npz reproduces the same scalars and the same CSV body byte for byte, so the
  text channel cannot be hand-edited into a nicer ladder.

The declared bands are structural, not fitted: the CSV is a lattice subsample
(stride 1 for n=32/64, stride 2 for n=128 -- 4096 cells in both cases), so its
norm estimator is expected to be close but not identical to the full-grid value.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.verification.evidence_pack import (l2_from_table, read_text_evidence,
                                                 sha256_file, write_text_evidence)

ROOT = Path(__file__).resolve().parents[1]
LADDER = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'runs-32-64-128-zero_gradient'
LEVELS = (32, 64, 128)
THRESHOLD = 1.8
#: Pre-declared band for "text-only" reproduction of a full-grid L2 indicator.
TEXT_CHANNEL_RTOL = 0.03
#: Pre-declared band for the slope rebuilt from text against the committed fit.
SLOPE_RTOL = 0.01


def _pack(n):
    path = LADDER / f'n{n}' / f'n{n}-evidence.md'
    if not path.is_file():
        pytest.skip(f'n={n}: text evidence pack not committed: {path}')
    return path


def _load(n):
    path = _pack(n)
    block, table = read_text_evidence(path)
    return path, block, table, l2_from_table(table)


def _row(n):
    return json.loads((LADDER / f'n{n}' / 'cases.json').read_text())[0]


def _summary():
    return json.loads((LADDER / 'summary.json').read_text())


@pytest.mark.parametrize('n', LEVELS)
def test_committed_pack_binds_text_to_the_committed_dump(n):
    """Same run, two channels: the row matches the JSON and the sha256 the npz."""
    path, block, _, _ = _load(n)
    assert block['row'] == _row(n), f'n={n}: embedded row drifted from cases.json'
    dump = block['binary_dump']
    npz = ROOT / dump['path']
    assert npz.is_file(), f'n={n}: pack is bound to a missing dump {dump["path"]}'
    assert dump['sha256'] == sha256_file(npz), f'n={n}: pack is bound to another run'
    assert dump['bytes'] == npz.stat().st_size
    assert block['provenance']['cases_json'] == (
        f'docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n{n}/cases.json')
    assert path.stat().st_size > 0


@pytest.mark.parametrize('n', LEVELS)
@pytest.mark.parametrize('recorded,recomputed', [
    ('velocity_l2', 'velocity_l2'),
    ('pressure_l2', 'pressure_l2'),
    ('pressure_mean_free_l2', 'pressure_mean_free_l2'),
])
def test_text_channel_reproduces_full_grid_indicator_inside_declared_band(
        n, recorded, recomputed):
    """Recompute from the markdown alone and compare to the recorded scalar."""
    _, block, _, values = _load(n)
    full = float(block['row'][recorded])
    text_only = float(values[recomputed])
    deviation = abs(text_only - full) / abs(full)
    assert deviation <= TEXT_CHANNEL_RTOL, (
        f'n={n} {recorded}: text channel gives {text_only!r} vs recorded {full!r} '
        f'({deviation:.3%} > {TEXT_CHANNEL_RTOL:.0%})')


def test_slope_from_text_alone_clears_acceptance_and_matches_the_fit():
    """The ledger criterion must be re-derivable without ever opening an npz."""
    errors = {n: _load(n)[3]['velocity_l2'] for n in LEVELS}
    h = [1. / n for n in LEVELS]
    text_slope = float(np.polyfit(np.log(h), np.log([errors[n] for n in LEVELS]), 1)[0])
    committed = _summary()['fits']['velocity_full']['slope']
    assert text_slope >= THRESHOLD, f'text-only slope {text_slope:.4f} < {THRESHOLD}'
    assert abs(text_slope - committed) / committed <= SLOPE_RTOL, (
        f'text-only slope {text_slope:.4f} drifted from the committed {committed:.4f} '
        f'by more than {SLOPE_RTOL:.0%}')
    assert _summary()['verdict']['pass'] is True


@pytest.mark.parametrize('n', LEVELS)
def test_pack_is_a_function_of_the_committed_dump(n, tmp_path, monkeypatch):
    """Re-rendering from the npz must not change a single scalar or CSV row.

    ``written_utc`` is the only field allowed to move; everything the numbers
    depend on has to be reproducible from the committed dump, so the text
    channel cannot be edited into a nicer ladder.
    """
    _, block, table, _ = _load(n)
    monkeypatch.chdir(ROOT)          # the pack records repo-relative paths
    fields = np.load(LADDER / f'n{n}' / f'n{n}-fields.npz')
    npz = (LADDER / f'n{n}' / f'n{n}-fields.npz').relative_to(ROOT).as_posix()
    out = tmp_path / f'n{n}-evidence.md'
    write_text_evidence(
        out, row=_row(n), cell_centres=fields['cell_centers'],
        cell_volumes=fields['cell_volumes'], velocity=fields['velocity'],
        pressure=fields['pressure'],
        exact={'velocity': fields['exact_velocity'], 'pressure': fields['exact_pressure']},
        npz_path=npz, max_cells=4096,
        argv=['scripts/write_kovasznay_evidence_packs.py', '--grids', str(n)],
        generator='scripts/write_kovasznay_evidence_packs.py (text rendering of the '
                  'committed dump)',
        provenance=block['provenance'])
    fresh_block, fresh_table = read_text_evidence(out)
    for key in ('row', 'downsample', 'subset_recomputation', 'binary_dump', 'provenance'):
        assert fresh_block[key] == block[key], f'n={n}: block field {key} moved'
    assert [key for key in fresh_block if key != 'environment'] == [
        key for key in block if key != 'environment']
    for key in table:
        assert np.array_equal(fresh_table[key], table[key]), f'n={n}: CSV column {key} moved'
