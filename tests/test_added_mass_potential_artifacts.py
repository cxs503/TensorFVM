"""Drift gate for the added-mass potential verification artifacts (catalog 14.1).

The published state of this case had drifted away from the code: ``main`` carried
a ``REPORT.md``/``summary.json`` with the *8* criteria of the historical midpoint
quadrature while the driver already emitted the *9* criteria of the consistent
Gauss rule, so the catalog quoted numbers the artifact did not contain.  Nothing
in the suite noticed, because the existing tests only exercise the code.

These tests make that class of drift a failure instead of a review question:

1. the committed ``summary.json`` criteria are recomputed from the committed
   ``cases`` with the driver's own ``_criteria`` -- a code change that adds, drops
   or re-limits a criterion fails until the artifact is regenerated;
2. every figure the summary lists exists and matches its recorded ``bytes`` and
   ``sha256`` -- a figure left over from an older run cannot survive a refresh of
   the numbers;
3. ``REPORT.md`` quotes every criterion id, the summary timestamp and each figure
   hash, and catalog 14.1 quotes the same verdict count and the same finest-level
   percentages as the artifact (no hand-typed drift).

The study itself (1200-panel ladders) is *not* re-run here; that is the driver's
job.  These assertions are cheap (~0.2 s) and run on every test invocation.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from tensorfvm.verification import added_mass_potential as amp

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DOC_DIR = os.path.join(REPO_ROOT, "docs", "verification-added-mass-potential")
SUMMARY = os.path.join(DOC_DIR, "summary.json")
REPORT = os.path.join(DOC_DIR, "REPORT.md")
CATALOG = os.path.join(REPO_ROOT, "docs", "verification-catalog.md")

EXPECTED_IDS = ["C1", "C2", "C3", "C6a", "C6b", "C4", "C5", "C7", "C8"]


def _summary() -> dict:
    with open(SUMMARY, encoding="utf-8") as handle:
        return json.load(handle)


def _catalog_section_14_1() -> str:
    with open(CATALOG, encoding="utf-8") as handle:
        catalog = handle.read()
    head = "### 14.1 Potential-Flow Added Mass of Spheroids"
    assert head in catalog, "catalog section 14.1 disappeared"
    section = catalog.split(head, 1)[1]
    for stop in ("\n### ", "\n## ", "\n---"):
        section = section.split(stop, 1)[0]
    return section


def _finest(name: str) -> dict:
    summary = _summary()
    case = next(c for c in summary["cases"] if c["label"] == name)
    return case["levels"][-1]


def test_artifact_criteria_match_the_driver():
    """The stored criteria must be exactly what the current code would emit."""
    summary = _summary()
    stored = summary["criteria"]
    recomputed = amp._criteria(summary["cases"])

    assert [c["id"] for c in stored] == EXPECTED_IDS
    assert [c["id"] for c in recomputed] == [c["id"] for c in stored]
    for old, new in zip(stored, recomputed):
        assert old["pass"] == new["pass"], old["id"]
        for key, value in new.items():
            if isinstance(value, float):
                assert old[key] == pytest.approx(value, rel=1e-12), (old["id"], key)
            elif key in ("statement", "pass"):
                continue
            else:
                assert old[key] == value, (old["id"], key)
    assert summary["all_pass"] is True


def test_artifact_quotes_the_consistent_quadrature():
    summary = _summary()
    assert summary["quadrature"]["primary"] == amp.PRIMARY_QUADRATURE == "gauss"
    check = {c["id"]: c for c in summary["criteria"]}
    # the accuracy verdict is the Gauss rule, not the midpoint baseline
    finest_sphere = _finest("sphere")
    finest_spheroid = _finest("spheroid_ar2")
    assert check["C1"]["measured_pct"] == pytest.approx(
        abs(finest_sphere["k_axial_rel_error"]) * 100.0, rel=1e-12)
    assert check["C3"]["measured_pct"] == pytest.approx(
        abs(finest_spheroid["k_transverse_rel_error"]) * 100.0, rel=1e-12)
    # every accuracy criterion is inside the 3 % acceptance standard of the platform
    for cid in ("C1", "C2", "C3"):
        assert check[cid]["measured_pct"] < 3.0, cid


def test_figures_match_their_recorded_hash():
    summary = _summary()
    figures = summary.get("figures")
    assert figures, "summary.json lost its figure provenance (written before make_figures?)"
    assert [f["name"] for f in figures] == ["fig1_added_mass_convergence.png",
                                            "fig2_error_ladder.png"]
    for fig in figures:
        path = os.path.join(DOC_DIR, fig["name"])
        assert os.path.isfile(path), fig["name"]
        data = open(path, "rb").read()
        assert len(data) == fig["bytes"], fig["name"]
        assert hashlib.sha256(data).hexdigest() == fig["sha256"], fig["name"]
        assert data[:8] == b"\x89PNG\r\n\x1a\n", fig["name"]


def test_report_is_rendered_from_the_artifact():
    summary = _summary()
    report = open(REPORT, encoding="utf-8").read()
    assert "on %s" % summary["generated"] in report
    for criterion in summary["criteria"]:
        assert "| %s |" % criterion["id"] in report, criterion["id"]
    for fig in summary["figures"]:
        assert fig["name"] in report
        assert fig["sha256"] in report
    assert "PASS" in report


def test_catalog_entry_matches_the_artifact():
    summary = _summary()
    section = _catalog_section_14_1()
    n = len(summary["criteria"])
    assert "PASS (%d/%d criteria)" % (n, n) in section
    assert "8/8" not in section, "catalog still quotes the stale criterion count"

    check = {c["id"]: c for c in summary["criteria"]}
    # the three headline percentages are injected from the artifact, not typed
    for cid in ("C1", "C2", "C3"):
        assert "%.3f" % check[cid]["measured_pct"] in section, cid
    # the quadrature comparison evidence quoted in the catalog
    assert "%.3f" % check["C7"]["measured_ratio"] in section
    assert "x%.2f" % check["C8"]["measured_reduction"] in section

    for fig in summary["figures"]:
        assert fig["name"] in section
        assert fig["sha256"][:16] in section