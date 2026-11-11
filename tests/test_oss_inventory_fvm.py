"""Gates for the open-source FVM inventory wave (FEAT-OSS-INVENTORY-FVM).

These tests do not re-run the survey or the benchmark; they verify that the
committed artifacts are *internally consistent and complete*, so the inventory
cannot silently degrade into a hand-written wish list:

* the registry covers >=20 projects, tiers are valid and counts agree;
* every ``GAP-FVM-*`` anchor has a real in-repo evidence path, >=2 reference
  implementations from the registry, a machine-checkable acceptance criterion
  and an external anchor that is not the repository itself;
* the committed Sod artifact really contains ``ref vs got`` numbers whose
  errors recompute, and the README table quotes exactly those numbers
  (anti hand-copy gate);
* the independent exact Riemann solver used as the reference reproduces the
  published Sod (1978) star state;
* the markdown views are reproducible from the JSON (no hand editing).
"""

from __future__ import annotations

import importlib.util
import json
import math
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs" / "oss-inventory-fvm"
REGISTRY = DOCS / "registry.json"
ARTIFACT = DOCS / "artifacts" / "sod_literature_benchmark.json"
README = DOCS / "README.md"


def _load_module(name: str, relpath: str):
    path = ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_registry_covers_at_least_twenty_projects(registry):
    repos = registry["repos"]
    assert len(repos) >= 20, f"inventory only covers {len(repos)} projects"
    ids = [r["id"] for r in repos]
    assert len(ids) == len(set(ids)), "duplicate repo ids"
    assert registry["counts"]["repos"] == len(repos)


def test_registry_tiers_and_counts_are_consistent(registry):
    valid = set(registry["tiers"])
    assert valid == {"T1", "T2", "T3"}
    repos = registry["repos"]
    for repo in repos:
        assert repo["tier"] in valid
        assert repo["family"] and repo["discretisation"]
        assert repo["tier_rationale"] and repo["license_risk"]
    computed = {t: sum(1 for r in repos if r["tier"] == t) for t in valid}
    assert computed == registry["counts"]["by_tier"]
    measured = sum(1 for r in repos
                   if r["metadata"].get("source") == "github_api")
    assert measured == registry["counts"]["measured_metadata"]


def test_metadata_evidence_is_declared_per_entry(registry):
    allowed = {"github_api", "upstream_not_github", "commercial", "unresolved",
               "not_attempted", "offline"}
    for repo in registry["repos"]:
        meta = repo["metadata"]
        assert meta.get("source") in allowed, (repo["id"], meta)
        if meta["source"] == "github_api":
            assert meta.get("slug") and meta.get("retrieved_at")
            assert isinstance(meta.get("stars"), int)
        else:
            # unreachable metadata must not be silently filled in
            assert meta.get("stars") is None or "stars" not in meta
        assert repo["curation"]["capability_evidence"] == "curated_knowledge"


def test_tier_one_entries_point_at_gaps(registry):
    for repo in registry["repos"]:
        if repo["tier"] == "T1":
            assert repo["gap_targets"], f"T1 {repo['id']} does not target any gap"


# ---------------------------------------------------------------------------
# gap anchors
# ---------------------------------------------------------------------------
def test_at_least_six_gap_anchors_with_full_contract(registry):
    gaps = registry["gaps"]
    assert len(gaps) >= 6, f"only {len(gaps)} gap anchors"
    ids = {r["id"] for r in registry["repos"]}
    seen = set()
    for gap in gaps:
        assert gap["id"].startswith("GAP-FVM-"), gap["id"]
        assert gap["id"] not in seen, f"duplicate gap {gap['id']}"
        seen.add(gap["id"])
        assert gap["priority"] in {"P0", "P1", "P2"}
        assert len(gap["reference_repos"]) >= 2
        assert set(gap["reference_repos"]) <= ids
        assert gap["current_state"] and gap["evidence_note"]
        assert gap["acceptance"].strip()
        assert gap["external_anchor"].strip()
        # the anchor must cite an external datum (a year-bearing literature or
        # analytic reference); a bare "our own run" is not an external anchor
        assert re.search(r"(19|20)\d\d", gap["external_anchor"]), gap["id"]


def test_gap_evidence_paths_exist_in_this_repository(registry):
    for gap in registry["gaps"]:
        assert gap["evidence_in_tensorfvm"], gap["id"]
        for rel in gap["evidence_in_tensorfvm"]:
            assert (ROOT / rel).exists(), f"{gap['id']}: missing evidence path {rel}"


def test_gap_priorities_cover_the_two_compressible_p0_items(registry):
    by_priority = {g["id"]: g["priority"] for g in registry["gaps"]}
    assert by_priority["GAP-FVM-IMPLICIT-DUAL-TIME"] == "P0"
    assert by_priority["GAP-FVM-LOWMACH-PRECOND"] == "P0"
    assert sum(1 for p in by_priority.values() if p == "P0") >= 2


# ---------------------------------------------------------------------------
# benchmark artifact
# ---------------------------------------------------------------------------
def test_artifact_is_a_real_run_with_repro_and_anchor(artifact):
    assert artifact["feature_slug"] == "FEAT-OSS-INVENTORY-FVM"
    assert artifact["tensorfvm_commit"]
    assert artifact["repro_cmd"]
    script = artifact["repro_cmd"].split()[1]
    assert (ROOT / script).exists(), f"repro script missing: {script}"
    assert artifact["reference"]["source"]
    assert "1978" in artifact["reference"]["source"]
    for name in artifact["figures"]:
        assert (ARTIFACT.parent / name).exists()


def test_artifact_errors_recompute_from_ref_and_got(artifact):
    for row in artifact["rows"]:
        scaling = 0.5 if row["kind"] != "plateau" else abs(row["ref"])
        recomputed = abs(row["got"] - row["ref"]) / scaling * 100.0
        assert math.isclose(recomputed, row["rel_err_pct"], rel_tol=1e-9,
                            abs_tol=1e-9), row


def test_artifact_acceptance_criteria_hold(artifact):
    plateau = [r for r in artifact["rows"] if r["kind"] == "plateau"]
    assert len(plateau) == 4
    assert max(r["rel_err_pct"] for r in plateau) < 3.0
    assert artifact["verdict"]["plateau_all_within_3pct"] is True
    positions = [r for r in artifact["rows"] if r["kind"] != "plateau"]
    assert len(positions) == 4
    for row in positions:
        assert row["err_cells"] <= row["criterion_cells"], row
    assert artifact["verdict"]["positions_within_tolerance"] is True
    orders = artifact["verdict"]["observed_order_rho"]
    assert len(orders) == 2 and min(orders) >= 0.7


def test_independent_exact_solver_reproduces_published_star_state():
    """The reference used to grade TensorFVM must itself match the literature."""
    module = _load_module("sod_bench",
                          "scripts/oss_inventory_fvm/sod_literature_benchmark.py")
    star = module.exact_star_state()
    # 5e-3 tolerance: the derived tail/contact speeds in the published table are
    # rounded, so they carry ~0.2 % scatter even for a correct solver
    for key, published in module.PUBLISHED.items():
        assert math.isclose(star[key], published, rel_tol=5e-3), (key, star[key], published)


def test_readme_quotes_the_artifact_numbers_not_hand_copies(artifact):
    text = README.read_text(encoding="utf-8")
    assert "sod_literature_benchmark.json" in text
    expected = {r["quantity"]: r for r in artifact["rows"]}
    matched = 0
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 3 or not line.strip().startswith("|"):
            continue
        key = cells[0].replace("\\", "")
        alias = {
            "p*": "p_star", "u*": "u_star", "ρ*_L": "rho_star_left",
            "ρ*_R": "rho_star_right", "稀疏波头 x": "rarefaction_head_x",
            "稀疏波尾 x": "rarefaction_tail_x", "接触间断 x": "contact_x",
            "激波 x": "shock_x",
        }
        if key not in alias or alias[key] not in expected:
            continue
        row = expected[alias[key]]
        try:
            ref, got = float(cells[1]), float(cells[2])
        except ValueError:
            continue  # prose table (e.g. the summary sheet) is not a number row
        assert math.isclose(ref, row["ref"], rel_tol=1e-4), (key, ref, row["ref"])
        assert math.isclose(got, row["got"], rel_tol=1e-4), (key, got, row["got"])
        matched += 1
    assert matched == len(expected), f"README table matched {matched}/{len(expected)} rows"


# ---------------------------------------------------------------------------
# rendered views must be reproducible from the JSON
# ---------------------------------------------------------------------------
def test_markdown_views_are_reproducible_from_the_registry(registry, tmp_path):
    module = _load_module("render_report",
                          "scripts/oss_inventory_fvm/render_report.py")
    assert module.render_registry(registry) + "\n" == (DOCS / "registry.md").read_text(encoding="utf-8")
    assert module.render_gaps(registry) + "\n" == (DOCS / "gap-proposals.md").read_text(encoding="utf-8")