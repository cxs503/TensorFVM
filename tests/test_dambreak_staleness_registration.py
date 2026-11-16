"""The dam-break case is inside the archived-source staleness net (D31 FAIL-1).

Before this file existed, ``docs/shallow-water-dam-break/`` had no ``manifest.json``:
the case that the feature ledger anchors on ``CASE-FVM-DAMBREAK-STOKER`` was the one
verification case whose producer source no gate watched, so an edit to
``shallow_water/dam_break.py`` could leave the archived numbers unverifiable with the
suite still green (``docs/VERIFICATION_DEBT_LEDGER.md`` D31).

What is asserted here, in the order the questions matter:

* the case is *registered* (``source_staleness.audit`` counts it, ``frozen=True``,
  ``stale=False``) and the registration adds **no** ratchet debt;
* the committed manifest is *derived*, not hand-typed: it is byte-identical to the
  manifest the frozen snapshot renders, and every frozen copy matches its digest;
* the frozen source really is the source the artefacts were produced from -- the
  snapshot copy of each bound producer file is byte-identical to the live file at
  registration time (reported, not enforced, so a later edit of the solver shows up as
  a readable drift entry instead of a red suite: drift is the provenance audit's job,
  ``scripts/audit_dambreak_pool_provenance.py`` P1);
* the machinery has power over *these* bindings: tampering, deleting, or live-mode
  editing flips this case to ``stale`` (a vacuous registration would stay green).

Run: ``PYTHONPATH=src pytest tests/test_dambreak_staleness_registration.py -q``
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import shutil

import pytest

from tensorfvm.verification import source_staleness as ss

ROOT = pathlib.Path(__file__).resolve().parents[1]
CASE_DIR = ROOT / "docs" / "shallow-water-dam-break"
MANIFEST = CASE_DIR / "manifest.json"
SNAPSHOT = CASE_DIR / "source-snapshot"
LEDGER = ROOT / "docs" / "verification-archived-source-staleness" / "staleness.json"
POOL_ENTRY = ROOT / "docs" / "case-fvm-dambreak-stoker" / "case_pool_entry.json"
REGISTER = ROOT / "scripts" / "register_dam_break_staleness.py"
AUDIT_SCRIPT = ROOT / "scripts" / "audit_dambreak_pool_provenance.py"


def _load_register():
    """Import ``scripts/register_dam_break_staleness.py`` as a module."""

    spec = importlib.util.spec_from_file_location("register_dam_break_staleness", REGISTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_auditor():
    """Import the provenance auditor to reuse its ``REQUIRED_INPUTS`` contract."""

    spec = importlib.util.spec_from_file_location("audit_dambreak_pool_provenance", AUDIT_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def register():
    return _load_register()


@pytest.fixture(scope="module")
def manifest():
    assert MANIFEST.is_file(), "run scripts/register_dam_break_staleness.py --write"
    return json.loads(MANIFEST.read_text())


@pytest.fixture(scope="module")
def report():
    return ss.audit(str(ROOT))


# ---------------------------------------------------------------------------
# 1. registration: the case is in the net, and no new debt was taken on
# ---------------------------------------------------------------------------
def test_case_is_registered_frozen_and_fresh(report, manifest):
    by_name = {case["case"]: case for case in report["cases"]}
    assert "shallow-water-dam-break" in by_name, "dam-break case is still outside the net"
    case = by_name["shallow-water-dam-break"]
    assert case["frozen"] is True
    assert case["stale"] is False
    assert case["counts"] == {"fresh": len(manifest["source_sha256"]), "changed": 0, "missing": 0}
    assert case["producer_bindings"] == len(manifest["source_sha256"])


def test_registration_does_not_grow_the_ratchet(report):
    ledger = json.loads(LEDGER.read_text())
    assert ss.unregistered_stale(report, ledger["known_stale_cases"]) == []
    assert ss.unregistered_unfrozen(report, ledger["known_unfrozen_cases"]) == []
    assert "shallow-water-dam-break" not in ledger["known_unfrozen_cases"]
    assert "shallow-water-dam-break" not in ledger["known_stale_cases"]


def test_ledger_census_matches_the_live_tree(report):
    """The committed census must describe a tree that contains the new case."""

    ledger = json.loads(LEDGER.read_text())
    assert {key: report["summary"][key] for key in ledger["summary"]} == ledger["summary"]
    # The registration is what moved the count: 53 -> 54 audited cases.
    assert ledger["summary"]["cases"] == len(report["cases"])
    assert ledger["summary"]["cases"] >= 54


# ---------------------------------------------------------------------------
# 2. the manifest is derived, and the frozen bytes ARE the recorded bytes
# ---------------------------------------------------------------------------
def test_committed_manifest_is_derived_from_the_snapshot(register, manifest):
    rebuilt = register.build_manifest(use_snapshot=True)
    assert register.render(rebuilt) == MANIFEST.read_text()
    assert rebuilt["source_sha256"] == manifest["source_sha256"]
    assert rebuilt["artifacts_sha256"] == manifest["artifacts_sha256"]


def test_every_frozen_copy_matches_its_digest(register, manifest):
    assert register.snapshot_integrity(manifest) == {}


def test_frozen_copy_of_each_producer_file_exists():
    manifest = json.loads(MANIFEST.read_text())
    for relative in manifest["source_sha256"]:
        assert (SNAPSHOT / relative).is_file(), relative


def test_drift_between_live_and_frozen_is_reported_not_hidden(register, manifest):
    """Live drift is *allowed* here but must be visible (it belongs to the P1 audit).

    A registration that silently track the live tree would make the archived numbers
    unverifiable; one that silently ignores drift would hide a stale archive.  So the
    invariant is: anything reported as drift has live bytes that differ from the frozen
    digest, and nothing else does.
    """

    drifted = register.drift(manifest)
    for relative, pair in drifted.items():
        live = ROOT / relative
        assert not live.is_file() or ss.sha256(live) == pair["live"]
        assert pair["live"] != pair["frozen"]
    assert set(drifted) <= set(manifest["source_sha256"])


def test_manifest_declares_what_the_gate_needs(manifest):
    assert manifest["schema"].startswith("tensorfvm.case-")
    assert manifest["case"] == "CASE-FVM-DAMBREAK-STOKER"
    assert manifest["tolerance"] == pytest.approx(0.05)
    assert manifest["t_star"] == pytest.approx(2.0)
    assert "Ritter" in manifest["reference"]["dry"] and "1892" in manifest["reference"]["dry"]
    assert "Stoker" in manifest["reference"]["wet"] and "1957" in manifest["reference"]["wet"]
    impl = set(manifest["impl_files"])
    assert {
        "src/tensorfvm/shallow_water/dam_break.py",
        "src/tensorfvm/shallow_water/swe1d.py",
        "src/tensorfvm/shallow_water/exact.py",
        "scripts/run_shallow_water_dam_break.py",
    } <= impl
    assert manifest["source_snapshot_root"] == "source-snapshot"
    assert manifest["repro_cmd"].startswith("PYTHONPATH=src python scripts/run_shallow_water_dam_break.py")


# ---------------------------------------------------------------------------
# 3. power: the registration can fail, in the three ways that matter
# ---------------------------------------------------------------------------
def test_power_tampered_snapshot_copy_is_stale(tmp_path, manifest):
    """A +1 byte edit of the frozen solver must be classified ``changed``."""

    case = tmp_path / "docs" / "shallow-water-dam-break"
    shutil.copytree(CASE_DIR, case)
    victim = case / "source-snapshot" / "src/tensorfvm/shallow_water/swe1d.py"
    victim.write_text(victim.read_text() + "\n# tampered\n")

    result = ss.classify_case(case, root=tmp_path)
    assert result["stale"] is True
    assert result["counts"]["changed"] == 1
    assert [e["path"] for e in result["entries"] if e["status"] == "changed"] == [
        "src/tensorfvm/shallow_water/swe1d.py"
    ]


def test_power_missing_binding_is_stale_not_fresh(tmp_path, manifest):
    """A binding that resolves nowhere is ``missing`` (unverifiable), hence stale."""

    case = tmp_path / "docs" / "shallow-water-dam-break"
    (case / "source-snapshot" / "src/tensorfvm/shallow_water").mkdir(parents=True)
    keep = "src/tensorfvm/shallow_water/exact.py"
    shutil.copyfile(CASE_DIR / "source-snapshot" / keep, case / "source-snapshot" / keep)
    shutil.copyfile(MANIFEST, case / "manifest.json")

    result = ss.classify_case(case, root=tmp_path)
    assert result["stale"] is True
    statuses = {e["path"]: e["status"] for e in result["entries"]}
    assert statuses[keep] == "fresh"
    assert statuses["src/tensorfvm/shallow_water/swe1d.py"] == "missing"


def test_power_live_mode_flags_a_source_edit(tmp_path, manifest):
    """The unfrozen variant -- the mode D31 asked for -- catches an edit of the solver.

    This is the behaviour the frozen registration trades away for durability, so it is
    pinned explicitly: if the snapshot root is dropped, editing ``dam_break.py`` makes
    the case stale.  The production registration keeps the snapshot (reproducibility),
    therefore *source drift* is the provenance audit's responsibility, not this gate's.
    """

    unfrozen = dict(manifest)
    unfrozen.pop("source_snapshot_root")
    live = tmp_path / "src" / "tensorfvm" / "shallow_water"
    live.mkdir(parents=True)
    for relative in manifest["source_sha256"]:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)
    case = tmp_path / "docs" / "shallow-water-dam-break"
    case.mkdir(parents=True)
    (case / "manifest.json").write_text(json.dumps(unfrozen))

    assert ss.classify_case(case, root=tmp_path)["stale"] is False
    victim = tmp_path / "src/tensorfvm/shallow_water/dam_break.py"
    victim.write_text(victim.read_text() + "\n# tampered\n")
    result = ss.classify_case(case, root=tmp_path)
    assert result["stale"] is True
    assert [e["path"] for e in result["entries"] if e["status"] == "changed"] == [
        "src/tensorfvm/shallow_water/dam_break.py"
    ]


# ---------------------------------------------------------------------------
# 4. the companion defect (D31 FAIL-2): the pool entry records all 12 inputs
# ---------------------------------------------------------------------------
def test_pool_entry_records_every_required_input():
    required = _load_auditor().REQUIRED_INPUTS
    entry = json.loads(POOL_ENTRY.read_text())
    assert len(required) == 12
    for case in entry["cases"]:
        missing = [name for name in required if case.get(name) is None]
        assert missing == [], f"{case['key']} cannot be re-run from the entry: {missing}"


def test_dry_depth_is_the_solver_default_and_is_recorded():
    """The key that used to be missing must match the solver's declared default."""

    from tensorfvm.shallow_water.swe1d import ShallowWater1D  # noqa: F401  (documented default)
    import inspect

    from tensorfvm.shallow_water.dam_break import run_case

    default = inspect.signature(run_case).parameters["dry_depth"].default
    entry = json.loads(POOL_ENTRY.read_text())
    for case in entry["cases"]:
        assert case["dry_depth"] == default

    metrics = json.loads(
        (
            ROOT
            / "src/tensorfvm/verification/artifacts/shallow-water-dam-break/metrics.json"
        ).read_text()
    )
    for key, blob in metrics["cases"].items():
        assert blob["discretisation"]["dry_depth"] == default, key