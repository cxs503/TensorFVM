"""Gate for the Re = 400 v-centreline *proposal* (m3240 reissue on current main).

``docs/GHIA_RE400_V_ATTRIBUTION.md`` registers the one printed Ghia entry the solver
cannot reproduce and exonerates the extraction; ``tests/test_ghia_reference_consistency.py``
keeps that attribution pinned.  What those leave open is the counterfactual: *what would
the ladder read if the entry were corrected?*  ``scripts/render_ghia_re400_v_proposal.py``
answers it and writes ``docs/verification-ghia-re400-v-centreline/``; this file keeps the
answer honest:

* the transcription is untouched and the opt-in path is the only way to see a corrected
  table, with nothing applied by default (:func:`self_check` proves both);
* the proposal is read out of the dispute register, so it cannot drift from its evidence;
* applying it touches exactly the registered entry and moves no extremum;
* rescored from the committed bundle, Re = 400 v goes from 14 % (growing with refinement)
  to under 3 % at n >= 64 -- the numbers the report quotes;
* the committed bundle itself still carries the printed reference value, and the pointwise
  gate still fails by default at Re = 400 (a pass cannot be bought by editing the data);
* the figures, ``metrics.json`` and the prose numbers are regenerated from the bundle and
  agree with what this test recomputes (no hand-copied numbers).
"""
from __future__ import annotations

import importlib.util
import json
import os

import pytest

from tensorfvm.lid_cavity_reference import (
    DISPUTES, SUPPORTED_REYNOLDS, V_CENTRELINE, corrected_table, proposed_corrections,
    self_check, table,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUNDLE = os.path.join(REPO, "docs", "verification-lid-cavity-ladder", "summary.json")
SCRIPT = os.path.join(REPO, "scripts", "render_ghia_re400_v_proposal.py")
OUT = os.path.join(REPO, "docs", "verification-ghia-re400-v-centreline")
DISPUTED = (400.0, "v", 0.9063)
GATE = 0.03


def _render_module():
    spec = importlib.util.spec_from_file_location("ghia_re400_v_proposal", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _bundle() -> dict:
    if not os.path.exists(BUNDLE):  # pragma: no cover - evidence checkout policy
        pytest.skip("evidence bundle not present in this checkout")
    with open(BUNDLE) as handle:
        return json.load(handle)


def _case(bundle: dict, reynolds: float, n: int) -> dict:
    for case in bundle["cases"]:
        if float(case["reynolds"]) == reynolds and int(case["n"]) == n:
            return case
    pytest.skip(f"Re={reynolds:g} n={n} not in this bundle")


# ----------------------------------------------------------- transcription safety
def test_the_printed_entry_is_still_the_published_number():
    printed = dict(table(*DISPUTED[:2]))
    assert printed[DISPUTED[2]] == -0.23827
    assert printed[DISPUTED[2]] == dict(V_CENTRELINE[DISPUTED[0]])[DISPUTED[2]]
    assert table(*DISPUTED[:2]) is V_CENTRELINE[DISPUTED[0]], (
        "the default path must hand back the transcription, not a copy of a corrected one")


def test_no_proposal_is_applied_by_default():
    records = proposed_corrections()
    assert records, "the register must keep the Re=400 entry visible"
    assert all(record["applied"] is False for record in records)
    assert all(record["status"] == "proposed" for record in records)
    check = self_check()
    assert check["ok"], check
    assert check["proposals"]["nothing_applied_by_default"] is True
    assert check["proposals"]["applied"] == 0
    for reynolds in SUPPORTED_REYNOLDS:
        for which in ("u", "v"):
            assert table(reynolds, which, corrected=False) == table(reynolds, which)
            assert corrected_table(reynolds, which) == table(reynolds, which, corrected=True)


# ------------------------------------------------------------ proposal integrity
def test_one_proposal_per_dispute_and_none_invented():
    assert len(proposed_corrections()) == len(DISPUTES)
    for record in proposed_corrections():
        key = (record["reynolds"], record["which"], record["x"])
        assert key in DISPUTES
        assert record["published"] == DISPUTES[key]["published"]
        assert record["proposed"] == DISPUTES[key]["richardson_limit"], (
            "the proposed value must be read from the register, not retyped")


def test_the_proposal_touches_exactly_the_registered_entry():
    for record in proposed_corrections():
        reynolds, which, x = record["reynolds"], record["which"], record["x"]
        printed = dict(table(reynolds, which))
        corrected = dict(corrected_table(reynolds, which))
        touched = [coord for coord in printed if printed[coord] != corrected[coord]]
        assert touched == [x], touched
        assert corrected[x] == record["proposed"]


def test_the_proposal_moves_no_extremum():
    for record in proposed_corrections():
        reynolds, which = record["reynolds"], record["which"]
        for points in (table(reynolds, which), corrected_table(reynolds, which)):
            assert min(points, key=lambda p: p[1])[1] == pytest.approx(-0.44993)
            assert max(points, key=lambda p: p[1])[1] == pytest.approx(0.30203)


def test_tables_without_a_proposal_are_identical_under_the_flag():
    for reynolds in SUPPORTED_REYNOLDS:
        for which in ("u", "v"):
            if (reynolds, which) in {(r["reynolds"], r["which"]) for r in proposed_corrections()}:
                continue
            assert corrected_table(reynolds, which) == table(reynolds, which)


# ------------------------------------------------------------- rescoring from data
def test_the_committed_bundle_still_holds_the_printed_reference_value():
    case = _case(_bundle(), 400.0, 64)
    coord = [round(c, 4) for c in case["v"]["profiles"]["coord"]]
    index = coord.index(DISPUTED[2])
    assert case["v"]["profiles"]["reference"][index] == -0.23827, (
        "the evidence bundle must not be regenerated against a corrected table")


def test_rescoring_flips_re400_v_and_only_re400_v():
    rows = {(row["reynolds"], row["n"]): row for row in _render_module().rescore(_bundle())}
    for n in (64, 128):
        row = rows[(400.0, n)]
        assert row["v"]["printed_max_abs_error"] > 0.10          # the open defect
        assert row["v"]["proposed_max_abs_error"] < GATE         # resolved under the proposal
        assert not row["v"]["pass_printed"]
        assert row["v"]["pass_proposed"]
        assert row["u"]["printed_max_abs_error"] == row["u"]["proposed_max_abs_error"]
    for key, row in rows.items():
        if key[0] == 400.0:
            continue                                             # the disputed resolution only
        assert row["v"]["printed_max_abs_error"] == row["v"]["proposed_max_abs_error"]


def test_the_coarsest_re400_row_still_misses_for_a_resolution_reason():
    rows = {(row["reynolds"], row["n"]): row for row in _render_module().rescore(_bundle())}
    row = rows[(400.0, 32)]
    assert row["v"]["proposed_max_abs_error"] > GATE, (
        "n=32 must not be talked into a pass: 32 cells do not resolve the lid shear layer")
    assert row["u"]["printed_max_abs_error"] > GATE       # and it fails on u as well
    summary = _render_module().summarise(list(rows.values()), _bundle())
    assert summary["failed_rows_proposed"] == ["400/n32"]


def test_the_pointwise_gate_still_fails_by_default_on_the_committed_row():
    case = _case(_bundle(), 400.0, 64)
    assert case["acceptance"]["pass"] is False, (
        "the committed acceptance record for Re=400 is red; the proposal must not green it")


# --------------------------------------------------------------- artefacts on disk
def test_artefacts_exist_and_match_a_fresh_recomputation():
    from tensorfvm.lid_cavity_reference import table as _table  # noqa: F401  (documents origin)

    metrics_path = os.path.join(OUT, "metrics.json")
    if not os.path.exists(metrics_path):  # pragma: no cover - evidence checkout policy
        pytest.skip("rendered bundle not present in this checkout")
    with open(metrics_path) as handle:
        committed = json.load(handle)
    module = _render_module()
    fresh = module.summarise(module.rescore(_bundle()), _bundle())
    assert committed["cases"] == fresh["cases"]
    assert committed["disputed"] == fresh["disputed"]
    assert committed["proposals"] == fresh["proposals"]
    for name in ("re400_v_disputed_point.png", "v_error_vs_resolution.png",
                 "all_centrelines_n128.png"):
        path = os.path.join(OUT, name)
        assert os.path.getsize(path) > 5000, name
        with open(path, "rb") as handle:
            assert handle.read(8) == b"\x89PNG\r\n\x1a\n", name


def test_the_report_quotes_the_numbers_it_computed():
    report_path = os.path.join(OUT, "report.md")
    if not os.path.exists(report_path):  # pragma: no cover - evidence checkout policy
        pytest.skip("rendered bundle not present in this checkout")
    with open(report_path) as handle:
        report = handle.read()
    module = _render_module()
    metrics = module.summarise(module.rescore(_bundle()), _bundle())
    printed = max(row["printed"] for row in metrics["disputed_rows"])
    proposed = max(row["proposed"] for row in metrics["disputed_rows"] if row["n"] >= 64)
    assert "Re = 400 v-centreline: the proposal, scored both ways" in report
    assert f"{printed:.4f}" in report and f"{proposed:.4f}" in report
    assert "-0.23827" in report and "-0.38989" in report
    assert "NOT inside 3 %" in report, "the honest answer must be stated, not implied"