"""Gate on whether archived verification evidence is backed by the current tree.

Four statements are made executable:

* classification -- a binding whose file still has the recorded digest is
  ``fresh``; an edited file is ``changed``; a binding that resolves nowhere is
  ``missing`` (unverifiable, which is worse than stale);
* resolution order -- a frozen ``source_snapshot_root`` copy wins over the live
  tree, so an archived case that froze its source stays ``fresh`` even after the
  live module was rewritten, while a handler written against the live file
  correctly reports ``changed``;
* ledger -- the repository's public ledger (``docs/benchmarks-index.md``) marks
  which stale cases are cited claims rather than internal notes;
* ratchet -- the debt recorded in ``staleness.json`` may not grow silently: a
  case that becomes stale without being accepted turns this gate red, and
  ``--update-ledger`` is the deliberate way to accept it.

Negative controls build fixtures in which the digest, the file, or the frozen
snapshot is tampered with, so each rule is shown to fail when it should instead
of passing vacuously.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from tensorfvm.verification import source_staleness as ss

CASE = pathlib.Path(__file__).resolve().parents[1] / (
    "docs/verification-archived-source-staleness"
)


def make_case(root, name="verification-demo", snapshot=None, entries=(), ledger=""):
    """Build ``docs/<name>/manifest.json`` plus the files its bindings point at."""
    case = root / "docs" / name
    case.mkdir(parents=True, exist_ok=True)
    manifest = {"schema": "fixture/1", "source_sha256": dict(entries)}
    if snapshot:
        manifest["source_snapshot_root"] = snapshot
    (case / "manifest.json").write_text(json.dumps(manifest))
    (root / "docs" / "benchmarks-index.md").write_text(ledger)
    return case


def write(path, text="x = 1\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return ss.sha256(path)


def test_fresh_changed_missing_are_distinguished(tmp_path):
    """One case exercising all three statuses at once."""
    fresh = write(tmp_path / "src" / "fvm" / "ok.py")
    stale = write(tmp_path / "src" / "fvm" / "edited.py", "x = 1\n")
    write(tmp_path / "src" / "fvm" / "edited.py", "x = 2\n")
    case = make_case(
        tmp_path,
        entries={
            "src/fvm/ok.py": fresh,
            "src/fvm/edited.py": stale,
            "src/fvm/gone.py": "0" * 64,
        },
    )
    classified = ss.classify_case(case, root=tmp_path)
    status = {e["path"]: e["status"] for e in classified["entries"]}
    assert status == {
        "src/fvm/ok.py": "fresh",
        "src/fvm/edited.py": "changed",
        "src/fvm/gone.py": "missing",
    }
    assert classified["counts"] == {"fresh": 1, "changed": 1, "missing": 1}
    assert classified["stale"] is True
    # A binding that resolves nowhere must not be reported as "fresh" anywhere.
    assert {e["provenance"] for e in classified["entries"]} == {
        "root",
        "missing",
    }


def test_frozen_snapshot_wins_over_rewritten_live_module(tmp_path):
    """Provenance order: frozen copy first, then the tree.

    The live module is rewritten (its digest no longer matches) while the frozen
    copy still does, which is exactly the case of a frozen-source mission; the
    same binding *without* a snapshot is stale.  Both branches are asserted, so
    the test fails if resolution is hard-wired to either side.
    """
    write(tmp_path / "src" / "fvm" / "core.py", "x = 1\n")
    recorded = ss.sha256(tmp_path / "src" / "fvm" / "core.py")
    frozen = tmp_path / "docs" / "verification-frozen" / "source-snapshot" / "src" / "fvm" / "core.py"
    write(frozen, "x = 1\n")
    write(tmp_path / "src" / "fvm" / "core.py", "x = 999\n")
    frozen_case = make_case(
        tmp_path, "verification-frozen", snapshot="source-snapshot", entries={"src/fvm/core.py": recorded}
    )
    live_case = make_case(tmp_path, "verification-live", entries={"src/fvm/core.py": recorded})
    frozen_report = ss.classify_case(frozen_case, root=tmp_path)
    live_report = ss.classify_case(live_case, root=tmp_path)
    assert frozen_report["entries"][0]["provenance"] == "snapshot"
    assert frozen_report["entries"][0]["status"] == "fresh"
    assert live_report["entries"][0]["provenance"] == "root"
    assert live_report["entries"][0]["status"] == "changed"


def test_ledger_citation_separates_headline_debt_from_internal_notes(tmp_path):
    """Citing the case directory is what makes a stale case a headline claim."""
    write(tmp_path / "src" / "fvm" / "a.py")
    digest = ss.sha256(tmp_path / "src" / "fvm" / "a.py")
    write(tmp_path / "src" / "fvm" / "a.py", "changed\n")
    make_case(tmp_path, "verification-cited", entries={"src/fvm/a.py": digest})
    make_case(tmp_path, "verification-internal", entries={"src/fvm/a.py": digest})
    (tmp_path / "docs" / "benchmarks-index.md").write_text("see verification-cited/report.md\n")
    report = ss.audit(tmp_path)
    by_name = {c["case"]: c for c in report["cases"]}
    assert by_name["verification-cited"]["cited_by_ledger"] is True
    assert by_name["verification-internal"]["cited_by_ledger"] is False
    assert report["summary"]["stale_cases"] == 2
    assert report["summary"]["headline_stale_cases"] == 1


def test_ratchet_flags_only_unregistered_stale_cases(tmp_path):
    """Known debt stays quiet; a newly stale case is loud."""
    write(tmp_path / "src" / "fvm" / "a.py")
    digest = ss.sha256(tmp_path / "src" / "fvm" / "a.py")
    write(tmp_path / "src" / "fvm" / "a.py", "changed\n")
    make_case(tmp_path, "verification-known", entries={"src/fvm/a.py": digest})
    report = ss.audit(tmp_path)
    assert ss.unregistered_stale(report, ["verification-known"]) == []
    assert ss.unregistered_stale(report, []) == ["verification-known"]


def test_fixture_vacuity_control_no_bindings_no_debt(tmp_path):
    """A manifest without source bindings carries no claims to audit."""
    case = tmp_path / "docs" / "verification-empty"
    case.mkdir(parents=True)
    (case / "manifest.json").write_text(json.dumps({"schema": "fixture/1"}))
    (tmp_path / "docs" / "benchmarks-index.md").write_text("")
    report = ss.audit(tmp_path)
    assert report["summary"] == dict(
        cases=0, entries=0, fresh=0, changed=0, missing=0, stale_cases=0, headline_stale_cases=0
    )
    assert ss.stale_cases(report) == []


@pytest.mark.parametrize("key", ss.SOURCE_KEYS)
def test_every_source_key_is_audited(tmp_path, key):
    """No producer binding is skipped because it lives under a newer key name."""
    write(tmp_path / "src" / "fvm" / "a.py")
    digest = ss.sha256(tmp_path / "src" / "fvm" / "a.py")
    case = tmp_path / "docs" / "verification-key"
    case.mkdir(parents=True)
    (case / "manifest.json").write_text(json.dumps({"schema": "fixture/1", key: {"src/fvm/a.py": digest}}))
    (tmp_path / "docs" / "benchmarks-index.md").write_text("")
    report = ss.audit(tmp_path)
    assert report["summary"]["entries"] == 1
    assert report["cases"][0]["counts"]["fresh"] == 1


def test_live_tree_ledger_reproduces_and_ratchet_is_respected():
    """The committed ledger must describe the current tree, and debt may not grow."""
    ledger_path = CASE / "staleness.json"
    assert ledger_path.is_file(), "run scripts/audit_archived_source_staleness.py"
    ledger = json.loads(ledger_path.read_text())
    root = pathlib.Path(__file__).resolve().parents[1]
    report = ss.audit(root)
    known = ledger["known_stale_cases"]
    assert ss.unregistered_stale(report, known) == []
    # Independent re-derivation of the pinned census: the ledger cannot be edited
    # into a story the tree does not tell.
    census = {
        "cases": report["summary"]["cases"],
        "entries": report["summary"]["entries"],
        "fresh": report["summary"]["fresh"],
        "changed": report["summary"]["changed"],
        "missing": report["summary"]["missing"],
        "stale_cases": report["summary"]["stale_cases"],
        "headline_stale_cases": report["summary"]["headline_stale_cases"],
    }
    assert census == ledger["summary"]
    # Every known-stale case must still exist and still be stale, otherwise the
    # ledger is stale-in-the-other-direction and the acceptance hides a fix.
    by_name = {c["case"]: c for c in report["cases"]}
    for name in known:
        assert name in by_name
        assert by_name[name]["stale"] is True
    # The ledger records resolved paths for every entry, so a reader can re-check
    # the audit by hand.
    for case in report["cases"]:
        for entry in case["entries"]:
            if entry["status"] == "missing":
                assert entry["resolved"] is None
            else:
                assert pathlib.Path(entry["resolved"]).is_file()