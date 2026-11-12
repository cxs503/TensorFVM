"""Drift gate for the machine-generated catalog sections (§17, §18).

The catalog entries for the rotating-frame (MRF) and buoyant-cavity faces quote
numbers that come from committed verification artifacts.  Typed numbers go
stale; these two tests make the catalog fail loudly instead:

1. regenerating the sections from the artifacts must reproduce the committed
   catalog byte for byte (``scripts/generate_catalog_verification_entries.py
   --check``), so a driver re-run with changed metrics, or a hand edit of the
   table, is a test failure rather than a silent inconsistency;
2. the artifacts those sections quote must still say what the entries claim
   (all criteria passing for the rotating frame; the honest `Ra = 1e6` miss for
   the cavity), so the section cannot be regenerated from a stale artifact that
   nobody re-ran.
"""

from __future__ import annotations

import importlib.util
import json
import os

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "generate_catalog_verification_entries.py")


def _load_script():
    spec = importlib.util.spec_from_file_location("gen_catalog_entries", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


def _load_json(rel: str) -> dict:
    with open(os.path.join(REPO_ROOT, rel), encoding="utf-8") as handle:
        return json.load(handle)


def test_catalog_sections_match_their_artifacts():
    script = _load_script()
    assert script.main(["--check"]) == 0, "catalog sections are stale; run the generator --write"


def test_generator_is_idempotent(tmp_path):
    script = _load_script()
    catalog = open(os.path.join(REPO_ROOT, "docs", "verification-catalog.md"), encoding="utf-8").read()
    once = script.splice(catalog, script.render_sections())
    twice = script.splice(once, script.render_sections())
    assert once == twice


def test_rotating_frame_artifact_still_supports_the_closed_claim():
    data = _load_json("docs/verification-rotating-frame/summary.json")
    assert data["all_pass"] is True
    criteria = data["criteria"]
    assert len(criteria) == 18
    assert all(c["pass"] for c in criteria.values())
    # discriminating power: the anchors must be able to reject a wrong solver
    for control in (
        "C6_anchor_rejects_no_coriolis_solver",
        "C7c_controls_reject_wrong_coriolis",
        "C10b_controls_reject_missing_frame_terms",
    ):
        assert criteria[control]["value"] >= 0.05
    # the physics anchors themselves, not just the bookkeeping
    assert criteria["C5_rotating_channel_vs_closed_form"]["value"] <= 0.03
    assert criteria["C8b_mrf_frame_invariance"]["value"] <= 1e-12


def test_buoyant_cavity_artifact_supports_the_partial_claim():
    """The entry says ``partially-closed-by``: one Ra passes, one does not.

    If the honest miss silently turned into a pass (or the pass into a miss),
    this test fails before the catalog can be quietly regenerated around it.
    """

    gci = _load_json("docs/natural-convection-gci/summary.json")
    gate = float(gci["gate"])
    ra1e5 = gci["families"]["Ra1e+05"]["verdict"]
    ra1e6 = gci["families"]["Ra1e+06"]["verdict"]
    assert ra1e5["finest_error"] <= gate
    assert ra1e6["finest_error"] > gate
    assert ra1e6["extrapolated"] is None  # withheld, not hidden
    assert gci["families"]["Ra1e+06"]["verdict"]["grids_passing_gate"] == []


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))