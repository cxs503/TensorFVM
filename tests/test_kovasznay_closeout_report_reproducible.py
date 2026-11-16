"""Drift gate for the CASE-FVM-KOVASZNAY-ORDER close-out document.

``docs/CASE-FVM-KOVASZNAY-ORDER/report_payload.py`` claims that every number in
``closeout-32-64-128.md`` is injected programmatically.  These tests hold it to
that claim on the committed evidence:

* the committed markdown is *exactly* what the generator renders from the
  committed ``summary.json`` / ``n128/cases.json`` / ``n128/environment.json`` /
  ``order-sensitivity.json`` — so regenerating the document can no longer drop a
  section (the §8 negative-control block used to be appended by hand and was
  silently deleted by the next regeneration);
* the generator actually follows the evidence: perturb a slope or a wall time and
  the rendered text moves with it (a hand-typed number would not);
* a failing verdict is rendered as an honest FAIL, not a canned PASS;
* the report payload's figure is the committed ``order.png`` and the registered
  ``tensor_benchmark_cases`` id in §8 matches ``NEGATIVE_CONTROL.md``.
"""
import copy
import importlib.util
import json
import re
from pathlib import Path

import pytest

CASE_DIR = Path(__file__).resolve().parents[1] / 'docs' / 'CASE-FVM-KOVASZNAY-ORDER'
RUNS = CASE_DIR / 'runs-32-64-128-zero_gradient'
DOC_PATH = CASE_DIR / 'closeout-32-64-128.md'
MODULE_PATH = CASE_DIR / 'report_payload.py'


def _load_module():
    spec = importlib.util.spec_from_file_location('kovasznay_report_payload', MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


payload_mod = _load_module()


def _render(summary=None, sensitivity=None, finest=None, environment=None):
    s, sens, fine, env = payload_mod.load_evidence()
    return payload_mod.build_document(
        summary if summary is not None else s,
        sensitivity if sensitivity is not None else sens,
        finest if finest is not None else fine,
        environment if environment is not None else env)


def test_committed_doc_is_what_the_generator_renders():
    """Regeneration is non-destructive: the committed doc is reproduced byte for byte."""
    committed = DOC_PATH.read_text()
    rendered = _render()
    assert rendered == committed, (
        'closeout-32-64-128.md drifts from report_payload.py; '
        'regenerate it instead of hand-editing')


def test_section8_is_generated_from_the_negative_control_evidence():
    """§8 must survive regeneration because it is rendered, not appended."""
    sensitivity = json.loads((CASE_DIR / 'order-sensitivity.json').read_text())
    verdict = sensitivity['verdict']
    doc = _render()
    section = doc.split('## 八、', 1)
    assert len(section) == 2, 'close-out doc lost its §8 negative-control block'
    body = section[1]
    assert f"id={payload_mod.BENCHMARK_CASE_ID}" in body
    assert f"{verdict['control_slope']:.4f}" in body
    assert f"{verdict['margin']:.4f}" in body
    assert f"{verdict['threshold']:g}" in body
    expected = 'PASS' if verdict['pass'] else 'FAIL'
    assert f'本节闸门自评：**{expected}**' in body


def test_registered_benchmark_case_id_matches_the_negative_control_doc():
    """The id in §8 is the registered criterion row, not a free-floating constant."""
    text = (CASE_DIR / 'NEGATIVE_CONTROL.md').read_text()
    assert re.search(rf'id={payload_mod.BENCHMARK_CASE_ID}\b', text)


def test_wall_times_come_from_the_n128_cases_file():
    """Build/solve seconds are read from evidence, not transcribed."""
    summary, sensitivity, finest, environment = payload_mod.load_evidence()
    doc = _render(summary, sensitivity, finest, environment)
    assert finest['build_seconds'] > 0
    assert f"build {finest['build_seconds']:.1f} s" in doc
    assert f"solve {finest['solve_seconds']:.1f} s" in doc
    # perturbing the recorded wall clock must move the rendered text
    altered = copy.deepcopy(finest)
    altered['build_seconds'] = finest['build_seconds'] + 1234.5
    altered_doc = _render(summary, sensitivity, altered, environment)
    assert f"build {altered['build_seconds']:.1f} s" in altered_doc
    assert altered_doc != doc


def test_perturbed_slope_propagates_to_the_headline_number():
    """A hand-typed headline would not follow the evidence; a generated one does."""
    summary, sensitivity, finest, environment = payload_mod.load_evidence()
    altered = copy.deepcopy(summary)
    altered['fits']['velocity_full']['slope'] = 1.2300
    altered['fits']['velocity_full']['r_squared'] = 0.900000
    doc = _render(altered, sensitivity, finest, environment)
    assert '**1.2300**' in doc
    assert '**1.9623**' not in doc


def test_failing_verdict_is_reported_as_fail():
    """Honest FAIL: the conclusion word and the <3% column follow the verdict, not prose."""
    summary, sensitivity, finest, environment = payload_mod.load_evidence()
    altered = copy.deepcopy(summary)
    altered['verdict']['pass'] = False
    altered['verdict']['velocity_full_pass'] = False
    altered['verdict']['pressure_meanfree_pass'] = False
    altered['verdict']['all_levels_converged'] = False
    doc = _render(altered, sensitivity, finest, environment)
    conclusion = doc.split('## 结论', 1)[1].split('## 一、', 1)[0]
    assert '**FAIL**' in conclusion
    assert '**PASS**' not in conclusion
    assert '并非所有网格都数值收敛' in conclusion
    assert '判据"斜率 ≥ 1.8"' in doc and '→\n**FAIL**' in doc


def test_failing_negative_control_is_reported_as_fail():
    """§8 self-evaluates its own gates instead of asserting a fixed PASS."""
    summary, sensitivity, finest, environment = payload_mod.load_evidence()
    altered = copy.deepcopy(sensitivity)
    altered['verdict']['pass'] = False
    altered['verdict']['margin_reached'] = False
    doc = _render(summary, altered, finest, environment)
    body = doc.split('## 八、', 1)[1]
    assert '本节闸门自评：**FAIL**' in body
    assert '`margin_reached`=False' in body


def test_payload_figure_is_the_committed_order_png():
    import base64

    summary, sensitivity, finest, environment = payload_mod.load_evidence()
    doc = _render(summary, sensitivity, finest, environment)
    png = (RUNS / 'order.png').read_bytes()
    payload = payload_mod.build_payload(doc, png, summary)
    figure = payload['figures'][0]
    decoded = base64.b64decode(figure['png_base64'])
    assert decoded == png
    assert decoded.startswith(b'\x89PNG\r\n\x1a\n')
    assert payload['markdown'] == doc
    assert f"{summary['fits']['velocity_full']['slope']:.4f}" in payload['title']
    assert payload['scenario_key'] == 'tensorfvm.case.kovasznay-order'


def test_generator_has_no_import_time_side_effects(tmp_path):
    """Importing the generator must not rewrite the committed document."""
    before = DOC_PATH.read_text()
    _load_module()
    assert DOC_PATH.read_text() == before


@pytest.mark.parametrize('path', ['summary.json', 'n128/cases.json', 'n128/environment.json'])
def test_required_evidence_is_committed(path):
    assert (RUNS / path).is_file(), f'missing evidence {path}'