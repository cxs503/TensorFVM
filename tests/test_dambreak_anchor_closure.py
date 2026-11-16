"""Regression gate for the anchor-closure dossier of CASE-FVM-DAMBREAK-STOKER.

These tests do two different jobs, and it is worth keeping them apart:

* the **static** tests recompute the dossier from the committed artifacts and
  pin the audit's findings, so a change to the grading semantics, to the
  acceptance mapping or to a pool amendment cannot silently invalidate the
  published audit;
* one **dynamic** test re-runs the solver at the declared evidence floor
  (``n = 800``, ~2 s) and requires bit-identical agreement with the committed
  ``convergence.json``.  ``tests/test_case_fvm_dambreak_stoker_pool.py`` only
  re-derives typed numbers *from* the artifact; nothing re-runs the solver, so a
  drift between code and artifact was previously invisible.

The pinned finding is deliberate, not cosmetic: ``front_wetted<5%`` feeds
``_verdict``'s aggregate ``all_pass`` while the same artifact bundle publishes
the quantity as ``front_wetted_diagnostic`` with a ladder that grows under
refinement (8.75 % -> 1.56 % -> 2.89 %).  If someone either promotes the reading
properly or demotes it to diagnostic-only, this test fails and the audit report
must be updated with it -- which is the point.
"""

from __future__ import annotations

import json

import pytest

from tensorfvm.verification import dambreak_anchor_closure as closure


@pytest.fixture(scope="module")
def static_dossier():
    return closure.dossier(reproduce=False)


def test_acceptance_sentence_is_fully_graded_and_passing(static_dossier):
    """Every element of the ledger's acceptance sentence has a live, passing gate."""
    assert static_dossier["acceptance_fully_covered"] is True
    elements = static_dossier["acceptance_elements"]
    assert set(elements) == {"front_position", "still_water_step", "monotone"}
    for name, status in elements.items():
        assert status["applicable_cases"], f"{name} has no applicable case"
        assert status["pass"], f"{name} is not graded green"
    # the still-water step is only defined on the wet bed
    assert elements["still_water_step"]["applicable_cases"] == ["stoker-wet"]


def test_graded_set_is_pinned(static_dossier):
    """Pin which keys feed ``all_pass``.

    A new gate (or a silently dropped one) must be reviewed against the audit,
    not absorbed into the aggregate ``PASS``.
    """
    expected = {
        "ritter-dry": {
            "front_fan<5%", "front_wetted<5%", "depth_l1<5%",
            "monotone(no_extrema)", "tv_no_growth<0.5%", "front_series<5%",
        },
        "stoker-wet": {
            "shock_position<5%", "step_ratio<5%", "plateau_h2<5%",
            "depth_l1<5%", "monotone(no_extrema)", "tv_no_growth<0.5%",
            "bore_position_modes<5%", "step_ratio_modes<5%", "front_series<5%",
        },
    }
    for case, keys in expected.items():
        assert set(static_dossier["cases"][case]["graded_keys"]) == keys


def test_diagnostic_promoted_to_gate_is_front_wetted(static_dossier):
    """The one graded-but-diagnostic quantity is ``front_wetted<5%``."""
    flagged = {
        case: {finding["gate"] for finding in case_dossier["diagnostic_graded"]}
        for case, case_dossier in static_dossier["cases"].items()
    }
    assert flagged["ritter-dry"] == {"front_wetted<5%"}
    assert flagged["stoker-wet"] == set()

    finding = static_dossier["cases"]["ritter-dry"]["diagnostic_graded"][0]
    assert finding["diagnostic_name"] == "front_wetted_diagnostic"
    # the ladder is what disqualifies it as a yardstick: not monotone decreasing
    assert finding["ladder_monotone_decreasing"] is False
    values = [value for _, value in finding["ladder"]]
    assert values[-1] > values[2], "ladder must be non-convergent for the finding to hold"
    assert finding["exceeds_tolerance_at"] == [100]


def test_orphaned_amendments_are_recorded(static_dossier):
    """Un-applied extension-leg amendments stay visible instead of vanishing."""
    orphans = {entry["file"]: entry for entry in static_dossier["orphaned_evidence"]}
    momentum = orphans["docs/CASE-FVM-DAMBREAK-STOKER-MOMENTUM/"
                        "proposed_pool_amendment.json"]
    assert set(momentum["unapplied_ids"]) == {"G7_momentum_identity",
                                              "G8_velocity_l1_mass"}
    galilean = orphans["docs/CASE-FVM-DAMBREAK-STOKER-GALILEAN/case_pool_entry.json"]
    assert galilean["unapplied_ids"], "a pool entry not in docs/oss_solutions is orphaned"


def test_pool_entry_registers_the_acceptance_gates(static_dossier):
    gate_ids = set(static_dossier["pool_entry"]["gate_ids"])
    assert {"G1_ritter_front_fan", "G2_stoker_shock", "G3_step_ratio",
            "G5_front_series", "G6_monotone"} <= gate_ids
    assert static_dossier["pool_entry"]["status"] == "pass"


def test_dossier_is_json_serialisable(static_dossier):
    text = json.dumps(static_dossier, allow_nan=False)
    assert closure.ANCHOR_SLUG in text


def test_solver_still_reproduces_the_committed_metrics():
    """Drift gate: the solver at this tree must reproduce the artifact exactly."""
    repro = closure.reproducibility_check(n=800)
    assert repro["all_pass"] is True
    for row in repro["rows"]:
        assert row["relative_difference"] < 1e-12, row
    assert repro["all_bit_identical"] is True