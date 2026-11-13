"""Tests for the KMM180 grid-refinement audit.

Three layers, in increasing cost:

* **synthetic** -- a manufactured series ``f(h) = f0 + a h^p`` must be recovered
  exactly (order *and* limit), and a flat series must be refused rather than
  extrapolated, so a defect in the audit cannot be mistaken for a physics
  finding;
* **decision rule** -- every branch of the pre-declared rule is exercised on
  synthetic verdict/value pairs, including the one that matters here: a coarse
  grid that passes while the refined ones fail is ``grid_decided``;
* **committed artifacts** -- the case's own grid record must agree with a
  recomputation from ``benchmark.json`` and the audit's headline finding (the
  SST log-layer verdict is grid-decided) is pinned, so a solver change cannot
  quietly turn "grid-decided" into "robust".
"""
from __future__ import annotations

import importlib.util
import json
import math
import pathlib

import pytest

from tensorfvm.verification import kmm180_grid as grid

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / 'docs' / 'case-fvm-channel-kmm180'
SCRIPT = REPO_ROOT / 'scripts' / 'audit_case_fvm_channel_kmm180_grid.py'
BENCHMARK = CASE_DIR / 'benchmark.json'
TOL = 0.05


def _load_script():
    spec = importlib.util.spec_from_file_location('audit_kmm180_grid', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _manufactured(order, limit, amplitude, cells=(64, 128, 256)):
    return [limit + amplitude * (1.0 / n) ** order for n in cells]


# --------------------------------------------------------------------------- #
# Synthetic: the extrapolation must be exact on a known series
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize('order,limit,amplitude', [(1.0, 0.03, 0.5), (1.5, 0.04, 0.9),
                                                   (2.0, 0.052, -0.7)])
def test_richardson_recovers_a_manufactured_series(order, limit, amplitude):
    values = _manufactured(order, limit, amplitude)
    info = grid.observed_order(values)
    assert info['valid'] is True
    assert info['order'] == pytest.approx(order, rel=1e-9)
    assert grid.richardson_limit(values, info['order']) == pytest.approx(limit, rel=1e-9)
    band = grid.gci(values, info['order'])
    expected = grid.SAFETY_FACTOR * abs(values[-1] - values[-2]) / (2 ** order - 1)
    assert band['absolute'] == pytest.approx(expected, rel=1e-9)
    assert band['relative'] == pytest.approx(expected / abs(values[-1]), rel=1e-9)
    assert band['safety'] == pytest.approx(grid.SAFETY_FACTOR)


def test_richardson_band_encloses_the_true_limit_for_a_third_order_series():
    values = _manufactured(3.0, 0.061, 2.0)
    order = grid.observed_order(values)['order']
    limit = grid.richardson_limit(values, order)
    band = grid.gci(values, order)['absolute']
    assert limit - band < 0.061 < limit + band


def test_refinement_ratio_is_validated():
    assert grid.refinement_ratio([64, 128, 256]) == pytest.approx(2.0)
    with pytest.raises(ValueError):
        grid.refinement_ratio([64, 256])
    with pytest.raises(ValueError):
        grid.refinement_ratio([64, 128, 512])            # not uniform
    with pytest.raises(ValueError):
        grid.refinement_ratio([256, 128, 64])            # wrong order


def test_flat_series_is_refused_not_extrapolated():
    """A plateau carries no order: quoting a limit for it is a defect."""
    values = [0.1062, 0.1061, 0.1059]
    study = grid.station_study([64, 128, 256], values, ['FAIL'] * 3, TOL)
    assert study['limit'] is None
    assert study['gci'] is None
    assert study['order_valid'] is False
    assert 'flat' in study['order_reason']
    assert study['classification'] == 'plateau_robust'


def test_non_monotone_series_has_no_asymptotic_range():
    values = [0.02, 0.035, 0.03]                          # sign change between differences
    info = grid.observed_order(values)
    assert info['valid'] is False
    assert 'sign' in info['reason']
    growing = [0.02, 0.021, 0.030]                        # differences grow with refinement
    assert grid.observed_order(growing)['valid'] is False


def test_order_outside_the_predeclared_window_is_rejected():
    values = _manufactured(4.0, 0.04, 0.5)                # p = 4 is above ORDER_BOUNDS
    info = grid.observed_order(values)
    assert info['valid'] is False
    assert 'outside' in info['reason']


# --------------------------------------------------------------------------- #
# Decision rule: every branch is reachable and does what it says
# --------------------------------------------------------------------------- #

def test_coarse_grid_pass_with_refined_fails_is_grid_decided():
    values = [0.0461, 0.0566, 0.0608]
    study = grid.station_study([64, 128, 256], values, ['PASS', 'FAIL', 'FAIL'], TOL)
    assert study['verdict_consistent'] is False
    assert study['classification'] == 'grid_decided'
    assert 'changes with the grid' in study['reason']
    # the extrapolated limit is on the FAIL side of the tolerance
    assert study['limit'] > TOL
    assert study['limit_band'][0] > TOL


def test_robust_verdict_needs_the_limit_band_on_the_same_side():
    convergent = _manufactured(1.0, 0.03, 0.4)
    study = grid.station_study([64, 128, 256], convergent, ['PASS'] * 3, TOL)
    assert study['classification'] == 'grid_robust'
    assert study['limit_band'][1] < TOL

    # same three verdicts, but the limit crosses the tolerance -> grid decided
    crossing = _manufactured(1.0, 0.0495, 0.4)
    study = grid.station_study([64, 128, 256], crossing, ['PASS'] * 3, TOL)
    assert study['classification'] == 'grid_decided'
    assert 'crosses the tolerance' in study['reason']

    failing = _manufactured(1.0, 0.09, 0.4)
    study = grid.station_study([64, 128, 256], failing, ['FAIL'] * 3, TOL)
    assert study['classification'] == 'grid_robust'
    assert study['limit_band'][0] >= TOL


def test_verdict_flip_inside_the_plateau_scatter_is_grid_decided():
    """A verdict that hinges on a difference below the plateau width is grid-decided."""
    values = [0.0499, 0.0501, 0.0500]
    study = grid.station_study([64, 128, 256], values, ['PASS', 'FAIL', 'FAIL'], TOL)
    assert study['classification'] == 'grid_decided'
    assert 'changes with the grid' in study['reason']


def test_inconclusive_when_the_series_is_neither_flat_nor_ordered():
    values = [0.02, 0.06, 0.05]
    study = grid.station_study([64, 128, 256], values, ['PASS', 'FAIL', 'FAIL'], TOL)
    assert study['classification'] == 'grid_decided'      # verdicts disagree first

    flat_but_split = [0.02, 0.061, 0.021]
    study = grid.station_study([64, 128, 256], flat_but_split, ['PASS'] * 3, TOL)
    assert study['classification'] in ('inconclusive', 'grid_decided')
    assert study['limit'] is None


def test_study_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        grid.station_study([64, 128, 256], [0.1, 0.2], ['FAIL', 'FAIL'], TOL)


# --------------------------------------------------------------------------- #
# Committed artifacts: the audit record must agree with its own inputs
# --------------------------------------------------------------------------- #

@pytest.fixture(scope='module')
def record():
    return _load_script().build(BENCHMARK)


def test_committed_audit_passes_its_gate():
    assert _load_script().check(_load_script().build(BENCHMARK)) == []


def test_record_is_derived_from_the_benchmark_it_names(record):
    module = _load_script()
    benchmark = json.loads(BENCHMARK.read_text())
    assert record['source']['file'] == str(BENCHMARK.relative_to(REPO_ROOT))
    assert record['source']['sha256'] == module._sha256(BENCHMARK)
    assert record['tolerance'] == benchmark['tolerance']
    # every studied series is exactly the benchmark's own numbers
    for key, study in record['studies'].items():
        model, _, station = key.partition('/')
        rows = sorted((c for c in benchmark['cases'] if c['model'] == model),
                      key=lambda c: c['n'])
        assert study['cells'] == [row['n'] for row in rows]
        assert study['values'] == [row[station]['rms_relative'] for row in rows]
        assert study['verdicts'] == [row[station]['verdict'] for row in rows]


def test_grid_studies_cover_every_model_and_window(record):
    benchmark = json.loads(BENCHMARK.read_text())
    models = {case['model'] for case in benchmark['cases']}
    expected = {'%s/%s' % (model, station) for model in models
                for station in grid.STATIONS}
    assert set(record['studies']) == expected


def test_sst_log_layer_verdict_is_pinned_as_grid_decided(record):
    """The headline finding: the n = 64 PASS does not survive refinement."""
    study = record['studies']['sst/log_layer']
    assert study['verdicts'] == ['PASS', 'FAIL', 'FAIL']
    assert study['classification'] == 'grid_decided'
    assert study['observed_order'] == pytest.approx(1.31, abs=0.05)
    assert study['limit'] == pytest.approx(0.0636, abs=5e-4)
    assert study['limit'] > record['tolerance']
    assert study['limit_band'][0] > record['tolerance']       # band entirely above


def test_sst_legacy_window_is_robust_and_the_closure_plateaus_are_marked(record):
    study = record['studies']['sst/legacy_full']
    assert study['classification'] == 'grid_robust'
    assert study['limit'] < record['tolerance']
    for key in ('komega/legacy_full', 'komega/log_layer',
                'kepsilon/legacy_full', 'kepsilon/log_layer'):
        assert record['studies'][key]['classification'] == 'plateau_robust', key
        assert record['studies'][key]['limit'] is None, key
    assert record['grid_decided'] == ['sst/log_layer']


def test_committed_markdown_and_figure_are_generated_not_typed(record):
    module = _load_script()
    assert (CASE_DIR / 'grid_convergence.md').read_text() == module.render_markdown(record)
    text = (CASE_DIR / 'grid_convergence.md').read_text()
    for needle in ('Decision and alternatives', 'Pre-declared thresholds',
                   'Interpretation limits', 'sst/log_layer', 'fig_grid_convergence.png'):
        assert needle in text, needle
    png = (CASE_DIR / 'fig_grid_convergence.png').read_bytes()
    assert png.startswith(b'\x89PNG') and len(png) > 1024


def test_headline_states_both_the_count_and_the_limit(record):
    headline = record['headline']
    assert 'grid-decided' in headline
    assert 'sst/log_layer' in headline
    assert '0.06' in headline or '6.3' in headline or '6.36' in headline
    assert math.isfinite(record['studies']['sst/log_layer']['limit'])

# --------------------------------------------------------------------------- #
# Coupling: the case gate refuses a stale grid audit
# --------------------------------------------------------------------------- #

def test_case_gate_flags_a_grid_audit_left_behind(tmp_path):
    """Regenerating benchmark.json must invalidate the audit, mechanically."""
    import shutil

    module = _load_script()
    spec = importlib.util.spec_from_file_location(
        'run_case_kmm180', REPO_ROOT / 'scripts' / 'run_case_fvm_channel_kmm180.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)  # type: ignore[union-attr]
    catalog = REPO_ROOT / 'docs' / 'oss_solutions' / 'CASE-FVM-CHANNEL-KMM180.json'

    case_copy = tmp_path / 'case-fvm-channel-kmm180'
    shutil.copytree(CASE_DIR, case_copy)
    # an out-of-tree copy audits cleanly once its provenance line (the recorded
    # source path) is regenerated against the copy itself
    copied = module.build(case_copy / 'benchmark.json')
    assert copied['source']['file'] == str(case_copy / 'benchmark.json')
    module.write(copied, out_json=case_copy / 'grid_convergence.json',
                 out_md=case_copy / 'grid_convergence.md',
                 out_png=case_copy / 'fig_grid_convergence.png')
    assert module.check(module.build(case_copy / 'benchmark.json')) == []  # own dir
    assert runner.check_artifacts(case_copy, catalog) == []

    # a benchmark regenerated under the audit's feet must be a gate failure
    audit = json.loads((case_copy / 'grid_convergence.json').read_text())
    audit['source']['sha256'] = '0' * 64
    (case_copy / 'grid_convergence.json').write_text(json.dumps(audit))
    problems = runner.check_artifacts(case_copy, catalog)
    assert any('source sha256 mismatch' in problem for problem in problems), problems

    # and a missing companion artifact must be caught too
    (case_copy / 'grid_convergence.json').unlink()
    problems = runner.check_artifacts(case_copy, catalog)
    assert any('grid_convergence.json is missing' in problem for problem in problems), problems
