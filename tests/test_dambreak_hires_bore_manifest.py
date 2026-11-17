"""Drift gate for hires.json and bore_subcell.json (D32 closure, m24426).

These two artifacts were **not** in the original manifest (D31/32), so they
could drift from the producer source without any gate noticing.  This test
verifies that the updated manifest (register_dam_break_staleness.py) now
binds them, and that the committed bytes match the live tree.

Run: ``PYTHONPATH=src pytest tests/test_dambreak_hires_bore_manifest.py -q``
"""

from __future__ import annotations

import importlib.util
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "docs/shallow-water-dam-break/manifest.json"
ARTIFACT_DIR = ROOT / "src/tensorfvm/verification/artifacts/shallow-water-dam-break"
REGISTER = ROOT / "scripts/register_dam_break_staleness.py"


def _load_register():
    spec = importlib.util.spec_from_file_location("register_dam_break_staleness", REGISTER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def register():
    return _load_register()


@pytest.fixture(scope="module")
def manifest():
    return json.loads(MANIFEST.read_text())


def test_hires_json_is_in_manifest(manifest):
    """hires.json must be tracked by the manifest (D32 closure)."""
    assert "hires.json" in manifest["artifacts_sha256"], (
        "hires.json not in manifest -- D32 not closed")


def test_bore_subcell_json_is_in_manifest(manifest):
    """bore_subcell.json must be tracked by the manifest (D32 closure)."""
    assert "bore_subcell.json" in manifest["artifacts_sha256"], (
        "bore_subcell.json not in manifest -- D32 not closed")


def test_hires_producer_is_bound(manifest):
    """run_shallow_water_dam_break_hires.py must be a bound producer."""
    assert "scripts/run_shallow_water_dam_break_hires.py" in manifest["source_sha256"], (
        "hires producer not bound -- D32 not closed")


def test_bore_subcell_producer_is_bound(manifest):
    """verify_dambreak_bore_subcell.py must be a bound producer."""
    assert "scripts/verify_dambreak_bore_subcell.py" in manifest["source_sha256"], (
        "bore_subcell producer not bound -- D32 not closed")


def test_hires_artifact_digest_matches_live(manifest):
    """The committed hires.json digest must match the live file."""
    import hashlib
    live = hashlib.sha256((ARTIFACT_DIR / "hires.json").read_bytes()).hexdigest()
    assert live == manifest["artifacts_sha256"]["hires.json"], (
        "hires.json is stale -- re-run: scripts/run_shallow_water_dam_break_hires.py")


def test_bore_subcell_artifact_digest_matches_live(manifest):
    """The committed bore_subcell.json digest must match the live file."""
    import hashlib
    live = hashlib.sha256((ARTIFACT_DIR / "bore_subcell.json").read_bytes()).hexdigest()
    assert live == manifest["artifacts_sha256"]["bore_subcell.json"], (
        "bore_subcell.json is stale -- re-run: scripts/verify_dambreak_bore_subcell.py")


def test_hires_snapshot_integrity(register, manifest):
    """The frozen hires producer must match the snapshot."""
    bad = register.snapshot_integrity(manifest)
    assert "scripts/run_shallow_water_dam_break_hires.py" not in bad, (
        f"hires producer snapshot integrity failed: {bad}")


def test_bore_subcell_snapshot_integrity(register, manifest):
    """The frozen bore_subcell producer must match the snapshot."""
    bad = register.snapshot_integrity(manifest)
    assert "scripts/verify_dambreak_bore_subcell.py" not in bad, (
        f"bore_subcell producer snapshot integrity failed: {bad}")


def test_manifest_is_derived(register, manifest):
    """The committed manifest must be byte-identical to the snapshot render."""
    rebuilt = register.build_manifest(use_snapshot=True)
    assert register.render(rebuilt) == MANIFEST.read_text(), (
        "manifest is not derived from snapshot -- re-run: scripts/register_dam_break_staleness.py --write")


def test_hires_payload_has_required_fields():
    """hires.json must have the fields the REPORT.md data block relies on."""
    payload = json.loads((ARTIFACT_DIR / "hires.json").read_text())
    # New structure: top-level 'cases' dict with case keys
    cases = payload.get("cases", {})
    for key in ("ritter-dry", "stoker-wet"):
        assert key in cases, f"hires.json missing case {key!r}"
        for row in cases[key]["rows"]:
            assert "depth_l1" in row, f"hires.json {key} row missing depth_l1"
            assert "headline" in row, f"hires.json {key} row missing headline"
            assert "front_series" in row, f"hires.json {key} row missing front_series"


def test_bore_subcell_payload_has_required_fields():
    """bore_subcell.json must have the fields the REPORT.md relies on."""
    payload = json.loads((ARTIFACT_DIR / "bore_subcell.json").read_text())
    assert "refinement" in payload, "bore_subcell.json missing refinement"
    for row in payload["refinement"]["rows"]:
        # Field names: bore_x_error -> error, bore_x_cell_error -> error_cell_reading
        assert "error" in row, f"bore_subcell.json row missing error"
        assert "error_cells" in row, f"bore_subcell.json row missing error_cells"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
