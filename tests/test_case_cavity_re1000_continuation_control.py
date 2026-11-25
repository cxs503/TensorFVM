"""Provenance gate for the continuation counter-check of ``CASE-FVM-CAVITY-RE1000``.

The published bundle claims "continuation buys cost only" on the strength of
``runs/summary.json -> continuation_check``: n = 64 reached by bilinear
prolongation from n = 32 versus the same grid started cold (59750 vs 58750 steps,
``max|du| = 3.34e-08``).  That claim has two ways of silently evaporating:

* the driver's control grid is pointed at a grid the campaign itself started from
  rest.  "Cold" and "prolonged" are then the same computation, agreement is a
  tautology (``max|du| = 0``, ``speedup`` = wall-clock noise) and it still prints
  as ``same_steady_state: true``.  ``repro.sh`` shipped exactly that
  (``--cold-control 32``) while ``runs/summary.json`` recorded
  ``--cold-control 64``, so the only documented full-campaign command could not
  reproduce the provenance of the committed bundle;
* the report renders the degenerate comparison as if it were evidence.

These tests pin the *invariant*, not the current numbers: the control grid is
selected from the grids the campaign actually warm-started, a from-rest control
withholds the verdict, the published check reproduces from the stored artefacts,
and the shipped ``repro.sh`` arguments equal the recorded provenance.

    PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000_continuation_control.py -q
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib
import shlex
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # repo checkout, not an installed copy

CASE_DIR = ROOT / "docs" / "CASE-FVM-CAVITY-RE1000"
RUNS = CASE_DIR / "runs"
GRIDS = (32, 64, 128)


def _module(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def run_case():
    return _module("case_cavity_re1000_run_case", CASE_DIR / "run_case.py")


@pytest.fixture(scope="module")
def make_report():
    return _module("case_cavity_re1000_make_report", CASE_DIR / "make_report.py")


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((RUNS / "summary.json").read_text())


@pytest.fixture(scope="module")
def bench() -> dict:
    return json.loads((RUNS / "benchmark.json").read_text())


def _record(n: int) -> dict:
    return json.loads((RUNS / f"lid_cavity_Re1000_n{n}.json").read_text())


def _control(n: int) -> dict:
    return json.loads((RUNS / "control" / f"lid_cavity_Re1000_n{n}.json").read_text())


@pytest.fixture(scope="module")
def artifacts() -> dict:
    return {n: _record(n) for n in GRIDS}


# --------------------------------------------------------------------------- #
# 1. selection: the control must land on a grid the campaign warm-started
# --------------------------------------------------------------------------- #
def test_warm_started_grids_reads_the_initial_condition(run_case, artifacts):
    assert run_case.warm_started_grids(artifacts) == [64, 128]
    rest_only = {32: {"initial_condition": "rest"}, 64: {"initial_condition": "rest"}}
    assert run_case.warm_started_grids(rest_only) == []
    missing_key = {48: {"steps": 10}}  # legacy artefact without the key => from rest
    assert run_case.warm_started_grids(missing_key) == []


def test_default_control_grid_is_the_coarsest_warm_started_grid(run_case, artifacts):
    # the coarsest grid of the campaign (32) is from rest, so it must NOT be chosen
    assert run_case.select_control_grid(list(GRIDS), artifacts) == (64, True)


def test_default_control_falls_back_and_reports_from_rest(run_case):
    grids = [32, 64]
    rest_only = {n: {"initial_condition": "rest"} for n in grids}
    assert run_case.select_control_grid(grids, rest_only) == (32, False)


def test_explicit_control_grid_is_honoured_but_flagged(run_case, artifacts):
    assert run_case.select_control_grid(list(GRIDS), artifacts, 32) == (32, False)
    assert run_case.select_control_grid(list(GRIDS), artifacts, 64) == (64, True)
    assert run_case.select_control_grid([], {}) == (None, False)


# --------------------------------------------------------------------------- #
# 2. a from-rest control may not claim "same steady state"
# --------------------------------------------------------------------------- #
def test_from_rest_control_withholds_the_verdict(run_case):
    recorded = _record(32)
    check = run_case.build_continuation_check(
        recorded, recorded, grid=32, warm_started=False)
    assert check["kind"] == "repeat_run"
    assert check["warm_started"] is False
    assert check["same_steady_state"] is None, "a self-comparison must not be reported as agreement"
    assert check["max_abs_du_centreline"] == 0.0 and check["max_abs_dv_centreline"] == 0.0
    assert "note" in check and "not" in check["note"].lower()


def test_prolonged_control_reports_the_verdict(run_case):
    check = run_case.build_continuation_check(
        _record(64), _control(64), grid=64, warm_started=True)
    assert check["kind"] == "continuation"
    assert check["same_steady_state"] is True
    assert check["max_abs_du_centreline"] < 1e-6
    assert "note" not in check


# --------------------------------------------------------------------------- #
# 3. the published bundle really is the non-degenerate case
# --------------------------------------------------------------------------- #
def test_published_control_grid_was_warm_started(summary, artifacts):
    cont = summary["continuation_check"]
    assert cont["grid"] in artifacts, "control grid must be one of the campaign's grids"
    assert artifacts[cont["grid"]]["initial_condition"] != "rest", (
        "the published counter-check would be a self-comparison")
    assert cont["kind"] == "continuation"
    assert cont["warm_started"] is True


def test_published_continuation_numbers_reproduce_from_artefacts(run_case, summary):
    cont = summary["continuation_check"]
    n = int(cont["grid"])
    recomputed = run_case.build_continuation_check(
        _record(n), _control(n), grid=n, warm_started=True)

    assert recomputed["kind"] == "continuation"
    assert recomputed["same_steady_state"] is cont["same_steady_state"] is True
    assert recomputed["max_abs_du_centreline"] == pytest.approx(
        cont["max_abs_du_centreline"], rel=1e-9, abs=1e-15)
    assert recomputed["max_abs_dv_centreline"] == pytest.approx(
        cont["max_abs_dv_centreline"], rel=1e-9, abs=1e-15)
    assert recomputed["steps_continued"] == cont["steps_continued"]
    assert recomputed["steps_cold"] == cont["steps_cold"]
    assert recomputed["wall_seconds_continued"] == pytest.approx(
        cont["wall_seconds_continued"], abs=1e-6)
    assert recomputed["wall_seconds_cold"] == pytest.approx(cont["wall_seconds_cold"], abs=1e-6)
    assert recomputed["speedup"] == pytest.approx(cont["speedup"], rel=1e-9)


def test_control_artefact_is_the_one_quoted(summary):
    cont = summary["continuation_check"]
    n = int(cont["grid"])
    control = _control(n)
    assert control["initial_condition"] == "rest"
    assert control["steps"] == cont["steps_cold"]
    assert control["wall_seconds"] == pytest.approx(cont["wall_seconds_cold"], abs=1e-6)
    assert control["converged"] is True


# --------------------------------------------------------------------------- #
# 4. the shipped reproduction command equals the recorded provenance
# --------------------------------------------------------------------------- #
def _campaign_args(script: pathlib.Path) -> list[list[str]]:
    """Every ``run_case.py`` invocation in ``script``, as argv fragments."""
    text = script.read_text().replace("\\\n", " ")  # join continued lines
    calls = []
    for chunk in text.split("python ")[1:]:
        if "run_case.py" not in chunk:
            continue
        body = chunk.split("run_case.py", 1)[1].split("\n\n")[0]
        calls.append([a for a in shlex.split(body.replace("\\", " ")) if a])
    return calls


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_repro_campaign_matches_recorded_command(summary):
    calls = _campaign_args(CASE_DIR / "repro.sh")
    recorded = shlex.split(summary["command"].split("run_case.py", 1)[1])
    campaign = [c for c in calls if _flag(c, "--grids") == _flag(recorded, "--grids")]
    assert len(campaign) == 1, f"expected one full-campaign call, got {calls}"
    got = campaign[0]
    for flag in ("--grids", "--cold-control", "--tolerance", "--max-steps"):
        assert _flag(got, flag) == _flag(recorded, flag), (
            f"repro.sh {flag}={_flag(got, flag)} but runs/summary.json records "
            f"{_flag(recorded, flag)}")
    assert int(_flag(got, "--cold-control")) == int(summary["continuation_check"]["grid"])


def test_repro_smoke_run_does_not_touch_the_published_bundle():
    text = (CASE_DIR / "repro.sh").read_text()
    smoke = [ln for ln in text.splitlines() if "--max-steps 2000" in ln]
    assert smoke, "smoke invocation missing"
    joined = text.replace("\\\n", " ")
    smoke_block = joined.split("--fast", 1)[1].split("fi")[0]
    assert "--out-dir /tmp" in smoke_block, "smoke run must not overwrite runs/"


# --------------------------------------------------------------------------- #
# 5. the report may not launder a degenerate check as evidence
# --------------------------------------------------------------------------- #
def test_report_withholds_a_repeat_run(make_report, summary, bench):
    degenerate = copy.deepcopy(summary)
    cont = degenerate["continuation_check"]
    cont["kind"] = "repeat_run"
    cont["warm_started"] = False
    cont["same_steady_state"] = None
    cont["max_abs_du_centreline"] = 0.0
    cont["max_abs_dv_centreline"] = 0.0

    text = make_report.render(degenerate, bench, None)
    assert "未评估" in text
    assert "不足以" in text
    assert "同解判定 = True" not in text


def test_report_states_the_continuation_verdict(make_report, summary, bench):
    text = make_report.render(summary, bench, None)
    assert "同解判定 = True" in text
    assert "不足以" not in text

# --------------------------------------------------------------------------- #
# 6. the report itself must be reproducible, and may not lose published prose
# --------------------------------------------------------------------------- #
def test_reference_floor_caveat_is_injected_not_dropped(make_report, summary, bench):
    text = make_report.render(summary, bench, None)
    dec = make_report._decomposition()
    assert dec is not None
    assert "「吻合」不等于「精度」" in text
    # numbers come from the artefact, not from prose typed into the generator
    truth_u = dec["u"]["by_grid"]["128"]["judge_truth_max_abs"] * 1e3
    band = dec["reference_floor_fraction_of_band"]["v"]["registered"]
    assert f"{truth_u:.2f}e-3" in text
    assert f"{100 * band:.0f} %" in text


def test_rendering_reproduces_the_committed_report(make_report, summary):
    import numpy as np

    fields = np.load(RUNS / "fields.npz")
    rendered = make_report.render(summary, json.loads((RUNS / "benchmark.json").read_text()),
                                  fields)
    committed = (CASE_DIR / "REPORT.md").read_text()
    dropped = make_report.dropped_lines(committed, rendered)
    assert dropped == [], f"make_report.py would delete published prose: {dropped[:3]}"


def test_dropped_lines_detects_a_hand_folded_passage(make_report):
    committed = "# t\n\n* bullet A\n* **caveat**：the 3 % band is 61 % reference\n"
    rendered = "# t\n\n* bullet A\n"
    dropped = make_report.dropped_lines(committed, rendered)
    assert dropped == ["* **caveat**：the 3 % band is 61 % reference"]


def test_dropped_lines_tolerates_rewording_and_the_timestamp(make_report):
    committed = ("> 本文件与全部数字由 `make_report.py` 注入；生成时间戳：old\n"
                 "* **caveat**：n = 128 的 `max|Δu| = 4.31e-3`\n")
    rendered = ("> 本文件与全部数字由 `make_report.py` 注入；生成时间戳：new\n"
                "* **caveat**：n = 128 的 `max|Δu| = 4.31e-3`（由工件注入）\n")
    assert make_report.dropped_lines(committed, rendered) == []
