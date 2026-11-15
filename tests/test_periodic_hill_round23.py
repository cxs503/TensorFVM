"""Round 23 contract tests for ``CASE-FVM-PERIODIC-HILLS``.

Three things are pinned here, all of them defects or claims that round 22 left
in a state where a reader could silently get the wrong answer:

1. **Artefact stem resolution** -- round 22's ladder driver pre-registered
   ``round22_laminar_*`` stems while the producer writes ``round20_laminar_*``;
   three of five rungs were therefore skipped and the ladder was reported as
   "fewer than two levels".  ``resolve_stem`` must find every rung.
2. **Usability audit** -- the three refined rungs exist but diverged inside the
   startup transient (``samples = 0``, zeroed mean shear).  They must be flagged
   *unusable*, not counted as levels.
3. **Honest verdict** -- the committed round-23 table must not claim a
   discretisation attribution, and the surviving 96x48 pair (``dt`` 0.02 / 0.01)
   must show the metric is time-step insensitive there (0.42 %), which is what
   makes the probe's "halve dt and the fine rungs survive" claim meaningful.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CASEDIR = REPO / "docs" / "case-periodic-hills"
sys.path.insert(0, str(REPO / "src"))


def _load_script(name: str):
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ladder_driver():
    return _load_script("round22_grid_ladder")


@pytest.fixture(scope="module")
def analysis():
    return _load_script("round23_analysis")


def test_resolve_stem_finds_every_rung(ladder_driver):
    """Every pre-registered rung must resolve to the producer's artefact name."""
    for level, stem, *_ in ladder_driver.LEVELS:
        resolved = ladder_driver.resolve_stem(stem)
        assert resolved is not None, f"{level}: stem {stem} does not resolve"
        assert (CASEDIR / f"{resolved}.json").exists()
        assert (CASEDIR / f"{resolved}.npz").exists()


def test_resolve_stem_rejects_unknown_suffix(ladder_driver):
    assert ladder_driver.resolve_stem("round22_laminar_Re2800_999x999") is None


def test_resolve_stem_accepts_both_prefixes(ladder_driver):
    """The producer prefix may change without dropping levels again."""
    a = ladder_driver.resolve_stem("round20_laminar_Re2800_144x72")
    b = ladder_driver.resolve_stem("round22_laminar_Re2800_144x72")
    assert a == b == "round20_laminar_Re2800_144x72"


def test_refined_rungs_are_unusable(analysis):
    """B/C/D diverged inside the startup transient: zero samples, no level."""
    audit = {a["level"]: a for a in
             (analysis.audit_rung(lv, stem) for lv, stem in analysis.RUNGS)}
    for level in ("B", "C", "D"):
        row = audit[level]
        assert row["samples"] == 0
        assert row["diverged_at_step"] is not None
        assert row["mean_tau_identically_zero"] is True
        assert row["usable"] is False
    for level in ("A", "Ap"):
        assert audit[level]["usable"] is True
        assert audit[level]["diverged_at_step"] is None


def test_divergence_times_are_in_the_startup_transient(analysis):
    audit = {a["level"]: a for a in
             (analysis.audit_rung(lv, stem) for lv, stem in analysis.RUNGS)}
    for level in ("B", "C", "D"):
        t_div = audit[level]["t_diverged"]
        assert 4.0 < t_div < 8.0, f"{level}: t_diverged = {t_div}"


def test_committed_table_claims_no_attribution():
    table = json.loads((CASEDIR / "round23_table.json").read_text())
    verdict = table["verdict"]
    assert verdict["ladder_usable"] is False
    assert verdict["n_usable"] == 2
    assert verdict["attribution_status"] == "not_measurable_with_stored_artefacts"
    assert verdict["usable_levels"] == ["A", "Ap"]
    assert "not to be reported as" in verdict["reason"].lower()


def test_committed_probe_reproduces_the_cluster_divergence():
    """The cheap local probe must reproduce the cluster blow-up step."""
    probe = json.loads((CASEDIR / "round23_probe.json").read_text())
    cases = {(c["nx"], c["ny"], round(c["dt"], 5)): c for c in probe["cases"]}
    b = cases[(144, 72, 0.0133)]
    c = cases[(192, 96, 0.01)]
    assert b["diverged_at_step"] is not None
    assert c["diverged_at_step"] is not None
    # rung B diverged at step 508 on the cluster, rung C at 538
    assert abs(b["diverged_at_step"] - 508) / 508.0 < 0.05
    assert abs(c["diverged_at_step"] - 538) / 538.0 < 0.05
    # halving / quartering dt at the same mesh keeps the run bounded
    for key in ((144, 72, 0.00665), (144, 72, 0.00333), (192, 96, 0.0025),
                (96, 48, 0.02)):
        assert cases[key]["diverged_at_step"] is None, key


def test_surviving_mesh_is_dt_insensitive():
    """96x48 at dt 0.02 vs 0.01: 0.42 % movement -> dt is not what limits it."""
    table = json.loads((CASEDIR / "round23_table.json").read_text())
    half = table["temporal_dt_half"]["dt_half"]
    assert half["resolved"] is True
    assert abs(half["relative"]) < 0.01


def test_one_mesh_cannot_make_a_ladder():
    """Two runs on the *same* mesh are not two refinement levels.

    The surviving artefacts of round 23 are 96x48 at dt 0.02 and dt 0.01: a
    time-step pair, not a spatial ladder.  Feeding them to the pre-registered
    rule as if they were a spatial ladder must not produce a verdict (the
    module refuses a non-refined pair outright).
    """
    from tensorfvm.verification.periodic_hill_grid_ladder import attribute

    with pytest.raises(ValueError):
        attribute([(9.0 / 96, 8.264277859446059), (9.0 / 96, 8.22962080445255)],
                  5.41, tolerance=0.10)