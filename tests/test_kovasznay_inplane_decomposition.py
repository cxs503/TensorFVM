"""Evidence guards for the Kovasznay in-plane / spanwise error-budget split.

Background: the committed ``RESULTS_INPLANE.md`` table for the 16/32/64 ``free``
sequence used to carry an "in-plane (u, v)" column that had never been computed
from a field -- it was the full-3D error times a constant 0.97 -- and a pressure
column that was shifted by one level with a rescaled last row.  The figure
``convergence_inplane.png`` was produced by a script that hard-coded those
columns and the slopes as literals, so it could not disagree with them.

These tests pin the invariants that make that class of pseudo-evidence fail
loudly instead of being cited:

1. every number in the committed decomposition artifact is recomputable from
   the stored cell-centred fields (``summary.json`` vs. fresh recompute);
2. the committed generated table agrees with the artifact (no drift between the
   document and the machine-readable evidence);
3. a missing level is a refusal, never a fallback to a documented sequence;
4. a sequence with mixed discrete configuration is a refusal;
5. a field that does not reproduce its own ``cases.json`` is a refusal;
6. every PNG cited by the Kovasznay documents has a producing script that loads
   ``npz`` data -- a figure whose generator holds its own numbers is not
   evidence.
"""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from tensorfvm.native_manufactured3d import volume_weighted_error

REPO = Path(__file__).resolve().parents[1]
BUILD = REPO / 'scripts' / 'kovasznay_inplane_decomposition.py'
RUNS = REPO / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'runs-32-64-128'
ARTIFACT = RUNS / 'inplane-decomposition' / 'summary.json'
TABLE = REPO / 'docs' / 'case-kovasznay-order-order-verification' / 'RESULTS_INPLANE_TABLE.md'
#: Documents that present the in-plane case as evidence.  The guard is scoped to
#: them: figures cited from unit-level case directories follow other conventions.
DOCS = list((REPO / 'docs' / 'case-kovasznay-order-order-verification').glob('*.md')) + \
       [REPO / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER' / 'VERIFICATION_REPORT.md']

LEVELS = (16, 32, 64)


def _load_builder():
    import importlib.util
    spec = importlib.util.spec_from_file_location('kovasznay_inplane_decomposition', BUILD)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _artifact():
    if not ARTIFACT.exists():
        pytest.skip(f'{ARTIFACT} not generated')
    return json.loads(ARTIFACT.read_text())


def test_committed_summary_matches_recompute_of_stored_fields():
    """The artifact is a function of the npz files, not of anyone's notes."""
    module = _load_builder()
    committed = {level['n']: level for level in _artifact()['levels']}
    for n in LEVELS:
        fresh = module.measure_level(n, RUNS / f'n{n}' / f'n{n}-fields.npz')
        stored = committed[n]
        for key in ('velocity_l2', 'velocity_l2_inplane', 'pressure_l2',
                    'pressure_mean_free_l2', 'spanwise_share', 'max_abs_w'):
            assert fresh[key] == pytest.approx(stored[key], rel=1e-12, abs=0.0), (
                f'n={n} {key}: artifact says {stored[key]!r}, the stored field '
                f'gives {fresh[key]!r}')


def test_inplane_error_is_not_a_constant_fraction_of_the_full_error():
    """The retracted column was ``full * 0.97``; a real split is not constant."""
    levels = _artifact()['levels']
    ratios = [level['velocity_l2_inplane'] / level['velocity_l2'] for level in levels]
    # The artifacts have ratios ~0.9556, ~0.8937, ~0.7365 -- a strongly varying
    # share.  A constant ratio is the signature of a rescaled column.
    assert max(ratios) - min(ratios) > 0.1, ratios
    assert all(ratio < 0.99 for ratio in ratios), ratios


def test_inplane_slope_is_the_least_squares_slope_not_a_pair_order():
    """The retracted report quoted the 16->32 pair order as the 3-grid slope."""
    orders = _artifact()['orders']['velocity_l2_inplane']
    least_squares = orders['least_squares']['slope']
    pairs = {f'{pair["from"]}->{pair["to"]}': pair['order'] for pair in orders['pair_orders']}
    assert least_squares == pytest.approx(1.8722, abs=5e-4)
    assert pairs['16->32'] == pytest.approx(1.8080, abs=5e-4)
    assert abs(least_squares - pairs['16->32']) > 0.05
    assert orders['least_squares']['residual_degrees_of_freedom'] == len(LEVELS) - 2


def test_committed_table_agrees_with_the_artifact():
    """A generated table may not drift from the JSON it was generated from."""
    text = TABLE.read_text()
    committed = {level['n']: level for level in _artifact()['levels']}
    for n in LEVELS:
        row = next(line for line in text.splitlines()
                   if re.match(rf'^\|\s*{n}\s*\|', line))
        cells = [cell.strip() for cell in row.strip('|').split('|')]
        assert float(cells[3]) == pytest.approx(committed[n]['velocity_l2'], rel=1e-5)
        assert float(cells[4]) == pytest.approx(committed[n]['velocity_l2_inplane'], rel=1e-5)
        assert float(cells[7]) == pytest.approx(committed[n]['pressure_mean_free_l2'], rel=1e-5)
    slope = _artifact()['orders']['velocity_l2_inplane']['least_squares']['slope']
    assert f'{slope:.4f}' in text


def test_results_document_table_also_matches_the_artifact():
    """The narrative document may not drift from the generated evidence either."""
    doc = TABLE.parent / 'RESULTS_INPLANE.md'
    text = doc.read_text()
    committed = {level['n']: level for level in _artifact()['levels']}
    for n in LEVELS:
        row = next(line for line in text.splitlines()
                   if re.match(rf'^\|\s*{n}\s*\|', line))
        cells = [cell.strip() for cell in row.strip('|').split('|')]
        assert float(cells[3]) == pytest.approx(committed[n]['velocity_l2'], rel=1e-5)
        assert float(cells[4]) == pytest.approx(committed[n]['velocity_l2_inplane'], rel=1e-5)
        assert float(cells[7]) == pytest.approx(committed[n]['pressure_mean_free_l2'], rel=1e-5)


def test_spanwise_share_is_the_quadrature_of_the_full_and_inplane_errors():
    """Budget identity: full^2 == inplane^2 + spanwise^2 for a common denominator."""
    for level in _artifact()['levels']:
        quadrature = np.hypot(level['velocity_l2_inplane'], level['spanwise_share'])
        assert quadrature == pytest.approx(level['velocity_l2'], rel=1e-12)


def test_inplane_error_dominance_falls_with_refinement():
    """The spurious mode overtakes the physical error as h -> 0 (see the report)."""
    shares = [(level['velocity_l2_inplane'] / level['velocity_l2']) ** 2
              for level in _artifact()['levels']]
    assert shares == sorted(shares, reverse=True)
    assert shares[0] > 0.9 and shares[-1] < 0.6


def test_missing_level_is_refused_rather_than_fallen_back():
    module = _load_builder()
    with pytest.raises(SystemExit) as excinfo:
        module.main(['--level', '256=/nonexistent/n256-fields.npz'])
    assert 'does not exist' in str(excinfo.value)
    assert 'fall back' in str(excinfo.value)


def test_mixed_discrete_configuration_is_refused(tmp_path):
    module = _load_builder()
    levels = [_artifact()['levels'][0] | {'symmetry_closure': 'free', 'reynolds': 40.0,
                                          'convection_order': 2, 'algorithm': 'coupled'}]
    mixed = [dict(levels[0]), dict(levels[0], n=32, h=1 / 32, symmetry_closure='zero_gradient')]
    result = module.analyse(levels + mixed[1:])
    with pytest.raises(SystemExit) as excinfo:
        module.check_same_scheme(result)
    assert 'symmetry_closure' in str(excinfo.value)


def test_field_that_does_not_reproduce_its_cases_json_is_refused(tmp_path):
    module = _load_builder()
    row = module.measure_level(32, RUNS / 'n32' / 'n32-fields.npz')
    tampered = tmp_path / 'cases.json'
    tampered.write_text(json.dumps([{'velocity_l2': row['velocity_l2'] * 1.01,
                                     'pressure_l2': row['pressure_l2'],
                                     'pressure_mean_free_l2': row['pressure_mean_free_l2'],
                                     'max_abs_z_velocity': row['max_abs_w']}]))
    with pytest.raises(SystemExit) as excinfo:
        module.cross_check(row, json.loads(tampered.read_text())[0], tampered)
    assert 'velocity_l2' in str(excinfo.value)


def test_every_cited_kovasznay_figure_has_an_npz_loading_generator():
    """No committed figure may be drawn from numbers baked into its generator."""
    cited = set()
    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text()
        cited.update(re.findall(r'!\[[^\]]*\]\(([^)]+\.png)\)', text))
        cited.update(re.findall(r'`([^`]*\.png)`', text))
    assert cited, 'no Kovasznay figure is cited; the guard would be vacuous'

    scripts = list((REPO / 'scripts').glob('*.py'))
    sources = {script: script.read_text() for script in scripts}
    for figure in sorted(cited):
        name = Path(figure).name
        producers = [script for script, text in sources.items()
                     if name in text and 'savefig' in text]
        assert producers, f'{figure} is cited but no script writes it'
        assert any('np.load' in sources[script] for script in producers), (
            f'{figure} is written by {[str(p) for p in producers]} without loading '
            'any stored field: the figure cannot disagree with its generator')


def test_fabricating_generator_is_gone():
    """The deleted script scaled the in-plane column by a literal ``0.97``."""
    assert not (REPO / 'scripts' / 'make_kovasznay_order_figure.py').exists()
    for script in (REPO / 'scripts').glob('*.py'):
        text = script.read_text()
        assert 'velocity_l2_inplane=1.25748e-02 * 0.97' not in text, script