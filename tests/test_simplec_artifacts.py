"""Guard the shipped SIMPLEC mobility artifacts against PR-gate and drift regressions.

The evidence for this change is a per-iteration audit whose full JSON is ~320 KB,
which trips the repository artifact cap (200 KB per pull request).  The shipped
layout therefore keeps only a digest plus downscaled figures in the git tree and
stores the full JSON outside it.  The figures additionally sit under ``docs/``
because the push gate only carries ``*.png`` out of ``docs/``.  These tests pin
that contract, and pin that the numbers printed in ``report.md`` still come from
``summary.json``:

1. the in-git artifact directory stays under the cap;
2. the full-evidence files are *not* re-added to the tree;
3. the digest drops per-iteration histories but keeps every scalar the report cites;
4. the report quotes the digest, not hand-copied numbers;
5. the digest records the SHA-256 of the sources that produced the numbers, so the
   evidence cannot survive a change to the solver it describes;
6. every cited figure ships at a path the push gate does not filter.
"""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "scripts" / "artifacts" / "simplec-mobility"
#: The pull-request push gate filters ``*.png`` outside ``docs/`` (binary figures
#: count as documentation, not as run records).  Figures therefore ship here, and
#: the guards below fail if one is moved back under :data:`ARTIFACTS`, where the
#: gate would silently drop it from the pushed commit.
FIGURES = ROOT / "docs" / "simplec-mobility" / "evidence"
SCRIPT = ROOT / "scripts" / "verify_simplec_mobility.py"

#: The pull-request artifact cap enforced by the merge gate.
ARTIFACT_CAP = 200 * 1024
#: No single shipped file may dominate the budget.
SINGLE_FILE_CAP = 80 * 1024


def _load_script():
    sys.path.insert(0, str(ROOT / "src"))
    spec = importlib.util.spec_from_file_location("verify_simplec_mobility", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _files(directory=ARTIFACTS):
    return {path.relative_to(directory): path.stat().st_size
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _digest():
    return json.loads((ARTIFACTS / "summary.json").read_text())


def test_artifacts_exist():
    files = _files()
    for required in ("summary.json", "report.md", "evidence/simplec-audit.md"):
        assert Path(required) in files, f"{required} missing from the shipped artifacts"
    # The three figures cited by the report must be in the tree, at the location
    # the push gate lets through (see FIGURES).
    cited = _digest()["audit"]["figures"]
    assert len(cited) >= 3, f"the audit must ship >= 3 figures, got {cited}"
    shipped = _files(FIGURES)
    for figure in cited:
        assert Path(figure) in shipped, f"figure {figure} missing from {FIGURES}"
        assert shipped[Path(figure)] > 0, f"figure {figure} is empty"


def test_figures_are_shipped_where_the_push_gate_allows_them():
    """A figure under an artifact directory is dropped on push -> CI goes red.

    The gate that publishes this branch strips ``*.png`` everywhere except under
    ``docs/``; that is exactly how the first CI run of this change lost its
    figures.  Pin the location so the failure cannot come back silently.
    """
    for name in _files(ARTIFACTS):
        assert name.suffix not in (".png", ".db", ".log"), (
            f"{ARTIFACTS / name} would be filtered out of the pushed commit")
    assert FIGURES.is_dir(), f"{FIGURES} missing: the figures would not ship"
    report = (ARTIFACTS / "report.md").read_text()
    link = Path(os.path.relpath(FIGURES, ARTIFACTS)).as_posix()
    for figure in _digest()["audit"]["figures"]:
        assert f"({link}/{figure})" in report, (
            f"report must link {figure} via {link}/ when rendered")


def test_total_artifact_size_under_pr_cap():
    files = _files()
    figures = _files(FIGURES)
    total = sum(files.values()) + sum(figures.values())
    assert total < ARTIFACT_CAP, (
        f"shipped artifacts+figures total {total} B >= cap {ARTIFACT_CAP} B")
    oversized = {str(name): size for name, size in {**files, **figures}.items()
                 if size >= SINGLE_FILE_CAP}
    assert not oversized, f"single artifact too large: {oversized}"


def test_full_evidence_stays_out_of_the_tree():
    files = {str(name) for name in _files()}
    # The full audit JSON and the raw audit table are regenerated into the evidence
    # store by the script; only the digest and the markdown summary ship in git.
    assert "verification.json" not in files
    assert "evidence/simplec-audit.json" not in files


def test_digest_drops_histories_but_keeps_cited_metrics():
    module = _load_script()
    payload = dict(
        audit=dict(case="c", error_limit=0.03, summary_resolution=12, resolutions=[12],
                   runs=[dict(configuration=dict(label="simplec-a08-p10"), resolution=12,
                              iterations=43, converged=True, passed=True, velocity_l2=0.01,
                              pressure_gradient=0.002, relative_mass_imbalance=1e-12,
                              scaled_divergence=1e-9,
                              history=[dict(iteration=i, continuity=1.0 / (i + 1),
                                            momentum=1.0 / (i + 2)) for i in range(50)])],
                   failed_simple_history={"SIMPLE a_p=1": dict(
                       failure="coupling breakdown", converged=False,
                       history=[dict(iteration=i, continuity=1.0) for i in range(7)])},
                   figures=["a.png"]),
        guards=dict(summary="11 passed", returncode=0, counts={"passed": 11}, failed=[]),
        mutation_control=dict(sensitive=True, mutated=dict(failed=["x"], summary="1 failed")),
    )
    slim = module.digest(payload)
    row = slim["runs"][0]
    assert "history" not in row
    assert row["history_length"] == 50
    assert row["final_residuals"]["iteration"] == 49  # converged terminal state survives
    for key in ("velocity_l2", "pressure_gradient", "relative_mass_imbalance",
                "scaled_divergence", "iterations", "converged", "passed"):
        assert key in row, f"digest dropped a metric the report cites: {key}"
    failure = next(iter(slim["failed_simple_history"].values()))
    assert failure["failure"] == "coupling breakdown" and failure["history_length"] == 7
    assert slim["audit"]["figures"] == ["a.png"]  # names only, no image bytes
    assert slim["mutation_control"]["sensitive"] is True


def test_shipped_digest_matches_digest_contract():
    digest = _digest()
    assert "histories omitted" in digest["note"]
    assert digest["guards"]["returncode"] == 0
    assert digest["guards"]["counts"]["passed"] == 11
    assert digest["mutation_control"]["sensitive"] is True, (
        "the mutation control must still show the guards turning red")
    assert digest["audit"]["passed"] is True
    for row in digest["runs"]:
        assert "history" not in row and row["history_length"] > 0
    assert digest["failed_simple_history"], "the SIMPLE a_p=1 failure must stay in the digest"


def test_report_numbers_are_rendered_from_the_digest():
    digest = _digest()
    report = (ARTIFACTS / "report.md").read_text()
    resolution = digest["audit"]["summary_resolution"]
    rows = {row["configuration"]["label"]: row for row in digest["runs"]
            if row["resolution"] == resolution}
    simplec = rows["simplec-a08-p10"]
    # Each of these strings is produced by render() from summary.json; if someone
    # hand-edits the report the values diverge from the digest and this fails.
    assert f"{100 * simplec['velocity_l2']:.3f}%" in report
    assert f"{simplec['iterations']} 次" in report
    assert f"{100 * simplec['pressure_gradient']:.3f}%" in report
    assert digest["guards"]["summary"] in report
    assert digest["mutation_control"]["restored_sha256"][:16] in report
    for figure in digest["audit"].get("figures", []):
        assert f"({Path(os.path.relpath(FIGURES, ARTIFACTS)).as_posix()}/{figure})" in report


def test_provenance_pins_the_sources_that_produced_the_numbers():
    """A changed solver (or changed audit) must invalidate the shipped evidence.

    The evidence is a finite measurement of one specific mobility block; without a
    source digest in the digest file the numbers would keep looking valid after
    ``_predict`` changed.  This pins the contract: rerun the script, not the prose.
    """
    module = _load_script()
    provenance = _digest()["provenance"]
    assert set(provenance) == set(module.SOURCES), "provenance must cover every source"
    for name, recorded in provenance.items():
        current = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert recorded == current, (
            f"{name} changed after the evidence was generated; rerun "
            "python scripts/verify_simplec_mobility.py --output "
            "scripts/artifacts/simplec-mobility")
    report = (ARTIFACTS / "report.md").read_text()
    cited = provenance[module.SOURCES[0]][:16]
    assert cited in report, "the report must name the solver digest it was measured on"
