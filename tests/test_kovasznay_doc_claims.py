"""Bind the numbers printed in the CASE-FVM-KOVASZNAY-ORDER docs to the artifacts.

The repeated failure mode of this anchor is not numerical: the closure document
kept a set of numbers while the artifacts underneath it moved one generation
ahead (``docs/CASE-FVM-KOVASZNAY-ORDER/m20571-artifact-schema-drift-diagnosis.md``
-- "报告 PASS != 树 PASS").  A reader therefore could not tell, from the tree
alone, whether a quoted slope still described the committed field dumps.

This gate removes that ambiguity by *parsing the document* and comparing every
quoted quantity against ``runs-32-64-128-zero_gradient/summary.json`` (which the
summariser recomputes from the npz field dumps, under the same ``1e-10``
recompute tolerance enforced by
``tests/test_kovasznay_32_64_128_acceptance.py``).  Two independent checks:

1. the three-grid closure document (``CLOSURE-32-64-128.md``) -- per-level L2
   errors, the ledger criterion slopes, the per-pair observed orders and the
   three-level re-identified order must all match the artifacts to the number of
   digits the document prints;
2. the four-level document (``ORDER-32-64-128-256.md``) must be *reproducible*:
   regenerating it from the committed four-level summary in a scratch directory
   must yield byte-identical text, so a hand-edited number is a failure rather
   than a silent claim.  A non-vacuity control asserts that the two documents
   differ, i.e. the comparison has discriminating power.

Sub/cell counts are compared exactly, real-valued quantities at the precision
they are printed with, so the gate is about "which solution produced this
number", not about float formatting.
"""
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
LADDER = CASE / 'runs-32-64-128-zero_gradient'
THREE = LADDER / 'summary.json'
FOUR = LADDER / 'summary-32-64-128-256' / 'summary.json'
CLOSURE = CASE / 'CLOSURE-32-64-128.md'
FOUR_LEVEL_DOC = CASE / 'ORDER-32-64-128-256.md'
FOUR_LEVEL_GENERATOR = CASE / 'make_order_uncertainty_report.py'
#: The pre-n=256 attempt note: handwritten, and it must *not* pass the
#: reproducibility comparison -- otherwise the comparison proves nothing.
HANDWRITTEN = CASE / 'ORDER-32-64-128-256-ATTEMPT.md'

MINUS = '\u2212'


def _number(text):
    """Parse a printed number, tolerating the U+2212 minus and `**bold**` marks."""
    cleaned = text.replace(MINUS, '-').replace('*', '').strip()
    return float(cleaned)


def _rows(text, first_cell_pattern):
    """Yield the cell lists of markdown table rows whose first cell matches."""
    pattern = re.compile(first_cell_pattern)
    for line in text.splitlines():
        if not line.startswith('|'):
            continue
        cells = [cell.strip() for cell in line.strip().strip('|').split('|')]
        if cells and pattern.search(cells[0]):
            yield cells


def _summary():
    if not THREE.is_file():
        pytest.skip(f'three-grid summary absent: {THREE}')
    return json.loads(THREE.read_text())


def _close(doc_value, actual, printed_digits):
    """Compare a printed value against the artifact at the printed precision.

    ``printed_digits`` is the number of significant digits the document kept, so
    the tolerance is one unit in the last printed place -- a document that rounds
    differently is still fine, a document that quotes a different solution is not.
    """
    exponent = math.floor(math.log10(abs(actual))) if actual else 0
    unit = 10. ** (exponent - printed_digits + 1)
    assert abs(doc_value - actual) <= unit, (doc_value, actual, unit)


def test_closure_document_level_table_matches_committed_summary():
    summary = _summary()
    levels = {level['n']: level for level in summary['levels']}
    text = CLOSURE.read_text()
    seen = set()
    for cells in _rows(text, r'^\d+$'):
        # | n | h | cells | vel L2 full | vel L2 in-plane | max|w| | p L2 mean-free | converged |
        if len(cells) != 8:
            continue
        n = int(cells[0])
        if n not in levels:
            continue
        level = levels[n]
        seen.add(n)
        assert int(cells[2].replace(',', '')) == level['cells'], (n, cells[2])
        _close(_number(cells[1]), level['h'], 4)
        _close(_number(cells[3]), level['velocity_full'], 7)
        _close(_number(cells[4]), level['velocity_inplane'], 7)
        _close(_number(cells[6]), level['pressure_meanfree'], 7)
        # max|w| is a round-off witness quoted with three significant digits, so
        # the same tolerance applies: it must name *this* solution's machine zero.
        _close(_number(cells[5]), level['max_abs_z_velocity'], 3)
    assert seen == {32, 64, 128}, f'level table no longer covers the ladder: {sorted(seen)}'


def test_closure_document_quotes_the_criterion_slopes_it_claims():
    summary = _summary()
    verdict = summary['verdict']
    fits = summary['fits']
    text = CLOSURE.read_text()
    # | 速度 L2（全 3D u,v,w） | **1.9623** | 0.999944 | **PASS** |
    quoted = {}
    for cells in _rows(text, r'^(速度|压力|展向)'):
        if len(cells) != 4:
            continue
        quoted[cells[0]] = cells[1]
    full = next((value for label, value in quoted.items() if '3D' in label), None)
    meanfree = next((value for label, value in quoted.items() if '去均值' in label), None)
    assert full is not None and meanfree is not None, f'slope table not found: {quoted}'
    _close(_number(full), fits['velocity_full']['slope'], 5)
    _close(_number(meanfree), fits['pressure_meanfree']['slope'], 5)
    # The PASS/FAIL word must follow the artifact, not be transcribed independently.
    assert verdict['velocity_full_pass'] is (fits['velocity_full']['slope'] >= 1.8)
    assert f"{fits['velocity_full']['slope']:.4f}"[:6] in full.replace('*', '').replace(MINUS, '-')


def test_closure_document_pair_and_three_level_orders_match_artifacts():
    summary = _summary()
    text = CLOSURE.read_text()
    corroboration = {}
    for cells in _rows(text, r'^(速度|压力)'):
        if len(cells) == 6:            # quantity | pairs | spread | both | consistent | 3-level
            corroboration[cells[0]] = cells
    assert corroboration, 'corroboration table not found'
    velocity = next(cells for label, cells in corroboration.items() if '3D' in label)
    pairs = [float(part) for part in velocity[1].replace(MINUS, '-').split('/')]
    recorded = summary['pair_consistency']['velocity_full']
    assert len(pairs) == len(recorded['pairs'])
    for quoted, actual in zip(pairs, recorded['pairs']):
        _close(quoted, actual, 5)
    _close(float(velocity[2]), recorded['spread'], 3)
    three_level = _number(velocity[5].split('（')[0])
    _close(three_level, summary['three_level_estimates']['velocity_full']['order'], 5)


def test_four_level_document_reproduces_from_committed_summary(tmp_path):
    if not FOUR.is_file():
        pytest.skip(f'four-level summary absent (n=256 not measured): {FOUR}')
    if not FOUR_LEVEL_DOC.is_file():
        pytest.fail(f'four-level artifacts exist but the document is missing: {FOUR_LEVEL_DOC}')
    scratch = tmp_path / 'case'
    scratch.mkdir()
    (scratch / FOUR_LEVEL_GENERATOR.name).write_text(FOUR_LEVEL_GENERATOR.read_text())
    (scratch / LADDER.name).symlink_to(LADDER, target_is_directory=True)
    env = {'PYTHONPATH': str(ROOT / 'src'), 'PATH': '/usr/bin:/bin:/usr/local/bin'}
    subprocess.run([sys.executable, str(scratch / FOUR_LEVEL_GENERATOR.name)],
                   check=True, cwd=str(scratch), env=env, capture_output=True)
    regenerated = (scratch / FOUR_LEVEL_DOC.name).read_text()
    assert regenerated == FOUR_LEVEL_DOC.read_text(), (
        'ORDER-32-64-128-256.md is not what the generator produces from the committed '
        'four-level summary: a number in the document has drifted from the artifacts')


def test_reproducibility_gate_is_not_vacuous():
    """Two documents that a human wrote differ -- so byte equality discriminates."""
    if not FOUR_LEVEL_DOC.is_file():
        pytest.skip(f'four-level document absent: {FOUR_LEVEL_DOC}')
    assert HANDWRITTEN.is_file()
    assert HANDWRITTEN.read_text() != FOUR_LEVEL_DOC.read_text()

FREE_LADDER = CASE / 'runs-32-64-128'
FREE_LEVELS = (16, 32, 64)
#: Order that several case documents quote for the historical ``free`` closure
#: (``docs/CASE-FVM-KOVASZNAY-ORDER/CLOSURE-32-64-128.md`` section 4.1 and
#: ``m20571-artifact-schema-drift-diagnosis.md`` section 4: "free leaves an O(h)
#: spurious spanwise velocity that caps the full-3D order at ~1.68").
FREE_ORDER_QUOTED = 1.68


def _free_level(n):
    if not (FREE_LADDER / f'n{n}' / 'cases.json').is_file():
        pytest.skip(f'free-closure level n={n} absent: {FREE_LADDER / f"n{n}"}')
    return json.loads((FREE_LADDER / f'n{n}' / 'cases.json').read_text())[0]


def test_free_closure_control_group_is_measured_and_stays_below_threshold():
    """The quoted "~1.68" is a *measurement*, not a memory.

    The acceptance gate skips its ``free`` arm at n=64/128, so that control group
    never shows a number by itself.  It is nevertheless measurable where the
    artifacts exist: the committed 16/32/64 ``free`` ladder.  Two things are
    checked, and both are the reason the ledger criterion is evaluated on the
    ``zero_gradient`` ladder instead:

    * the slope of that ladder reproduces the quoted ~1.68 and stays below 1.8;
    * its spanwise velocity is *not* a machine zero (O(h), 9.4e-3 -> 2.2e-3),
      which is the physical effect the closure removes.

    n=128 of this ladder is genuinely not measured; that is why the 9 ``anchor``
    skips in ``tests/test_kovasznay_32_64_128_acceptance.py`` remain skips.
    """
    rows = [_free_level(n) for n in FREE_LEVELS]
    closure = [row.get('symmetry_closure') for row in rows]
    assert closure[0] == 'free' and closure[2] == 'free', closure
    # n=32 predates the recorded key; reconstruct it from the level's own fields
    # instead of trusting a label (same rule as the summariser's guard).
    assert closure[1] is None, f'n=32 now records a closure: {closure[1]!r}'
    spanwise = [row['max_abs_z_velocity'] for row in rows]
    assert all(value > 1e-4 for value in spanwise), spanwise
    assert spanwise[1] < spanwise[0] and spanwise[2] < spanwise[1], spanwise

    errors = [row['velocity_l2'] for row in rows]
    slope = float(__import__('numpy').polyfit(
        __import__('numpy').log([row['h'] for row in rows]),
        __import__('numpy').log(errors), 1)[0])
    assert slope < 1.8, f'free ladder no longer caps the order: {slope}'
    assert abs(slope - FREE_ORDER_QUOTED) < 0.02, (
        f'documents quote ~{FREE_ORDER_QUOTED} for the free ladder, artifacts give {slope:.4f}')
