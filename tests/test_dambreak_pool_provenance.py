"""Tests for the dam-break case-pool provenance auditor.

The auditor's own power is the point: a green provenance audit is worthless if
the checks cannot fail.  Every test here either pins an invariant of the
committed evidence chain, or proves that the corresponding check *does* fire on
corrupted data (negative controls).

Deliberately no solver re-runs in this file (those live in
``scripts/audit_dambreak_pool_provenance.py`` and take seconds); the tests stay
sub-second so they can run in the normal gate.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "audit_dambreak_pool_provenance.py"

spec = importlib.util.spec_from_file_location("audit_dambreak_pool_provenance", SCRIPT)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture(scope="module")
def entry():
    return json.loads(audit.ENTRY_PATH.read_text())


@pytest.fixture(scope="module")
def artifact():
    return json.loads((audit.ARTIFACT_DIR / "metrics.json").read_text())


def test_committed_entry_is_the_derived_payload(entry):
    """No hand edits: the committed entry is byte-identical to the generator."""
    assert audit.builder_identity(entry)["identical"] is True


def test_every_graded_criterion_is_re_derivable_from_the_artifact(entry, artifact):
    report = audit.entry_derivation(entry, artifact)
    assert report["unmapped"] == []
    assert report["drifted"] == []


def test_graded_criteria_are_graded_by_the_ledger_contract(entry):
    """Every criterion in the pool grades one of the three contracted quantities."""
    contract = entry["ledger_contract"]["graded_quantities"]
    assert len(contract) == 3
    for case in entry["cases"]:
        names = [c["criterion"] for c in case["criteria"]]
        assert "front_series<5%" in names  # x_f(t)
        assert any(n in names for n in ("front_fan<5%", "shock_position<5%"))  # x_f(t*)
        assert "monotone(no_extrema)" in names
    wet = [c for c in entry["cases"] if c["kind"] == "wet"][0]
    assert {"step_ratio<5%", "plateau_h2<5%"} <= {c["criterion"] for c in wet["criteria"]}


def test_verdicts_follow_from_the_grid_rows(entry):
    report = audit.verdict_algebra(entry)
    assert report["consistent"] is True
    assert report["problems"] == []


def test_accepted_grid_is_not_set_by_a_diagnostic_criterion(entry):
    """The pool's accepted grid must survive dropping non-primary criteria."""
    report = audit.verdict_algebra(entry)
    assert report["accepted_grid_min"] == report["accepted_grid_min_primary_only"]


def test_domain_and_dam_location_are_recorded(entry):
    """x_min/x_max/x_dam move the graded numbers, so they must be in the entry."""
    for case in entry["cases"]:
        assert case["x_min"] < case["x_dam"] < case["x_max"]
    report = audit.config_completeness(entry)
    # `dry_depth` is the one input the archived artifact predates (see D31).
    assert set(report["missing_anywhere"]) <= {"dry_depth"}


def test_negative_control_hand_edit_is_caught(entry, artifact):
    bad = copy.deepcopy(entry)
    bad["cases"][0]["criteria"][0]["value"] *= 1.10
    assert audit.entry_derivation(bad, artifact)["clean"] is False


def test_negative_control_flipped_pass_flag_is_caught(entry):
    bad = copy.deepcopy(entry)
    bad["cases"][1]["criteria"][0]["pass"] = not bad["cases"][1]["criteria"][0]["pass"]
    assert audit.verdict_algebra(bad)["consistent"] is False


def test_negative_control_shifted_accepted_grid_is_caught(entry):
    bad = copy.deepcopy(entry)
    bad["accepted_grid_min"] = bad["accepted_grid_min"] * 2
    assert audit.verdict_algebra(bad)["consistent"] is False


def test_negative_control_missing_input_is_caught(entry):
    bad = copy.deepcopy(entry)
    bad["cases"][0]["x_dam"] = None
    assert audit.config_completeness(bad)["complete"] is False


def test_negative_control_corrupted_artifact_breaks_reproduction(artifact):
    """A +10 % perturbation of a graded reading must not reproduce."""
    bad = copy.deepcopy(artifact)
    bad["cases"]["ritter-dry"]["metrics"]["front_fan_error"] *= 1.10
    report = audit.reproduction(bad, cases={"ritter-dry": audit.CASES["ritter-dry"]})
    assert report["all_reproduced"] is False


def test_derivation_scales_are_the_documented_conversions(entry, artifact):
    """tv_no_growth and the mode criteria are ratio-style; check the sign convention."""
    metrics = artifact["cases"]["stoker-wet"]["metrics"]
    assert audit.derived_criterion_value("stoker-wet", "tv_no_growth<0.5%", metrics) >= 0.0
    for name in ("bore_position_modes<5%", "step_ratio_modes<5%"):
        assert audit.derived_criterion_value("stoker-wet", name, metrics) >= 0.0