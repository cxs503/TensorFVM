"""Tests for the verification-catalog gate (``tensorfvm.verification.catalog_gate``).

The gate is a *verification of the verifier's index*: it must fail on the
defects the historical catalog actually carried (dead test pointer, PASS entry
with no live test, un-annotated gap) and pass on the repaired catalog that is
committed next to it.
"""

from __future__ import annotations

import json
import os

import pytest

from tensorfvm.verification import catalog_gate as gate

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CATALOG = "docs/verification-catalog.md"


def _write(tmp_path, text):
    path = tmp_path / "catalog.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_committed_catalog_passes_the_gate():
    """The catalog shipped in this repo must be self-consistent."""

    audit = gate.audit_catalog(CATALOG, root=REPO_ROOT)
    data = audit.to_dict()
    assert audit.passed, json.dumps(data, indent=1)
    assert data["counts"]["entries"] >= 30
    assert data["counts"]["gaps"] >= 5


def test_gate_detects_dead_pointer(tmp_path):
    text = """# Catalog

### 1.1 Cavity

**Implementation**: `src/tensorfvm/does_not_exist.py`
**Tests**: `tests/test_does_not_exist.py`

**Verification Status**: \u2705 PASS
"""
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert not audit.passed
    assert {row["pointer"] for row in audit.dangling_pointers} == {
        "src/tensorfvm/does_not_exist.py",
        "tests/test_does_not_exist.py",
    }


def test_gate_detects_pass_entry_without_live_test(tmp_path):
    text = """# Catalog

### 1.1 Cavity

**Implementation**: `src/tensorfvm/solver.py`
**Tests**: across the suite

**Verification Status**: \u2705 PASS
"""
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert audit.unsupported_entries == ["1.1 Cavity"]


def test_gate_detects_unannotated_and_dangling_gap(tmp_path):
    text = """# Catalog

### 1.1 Cavity

**Implementation**: `src/tensorfvm/solver.py`
**Tests**: `tests/test_lid_cavity.py`

**Verification Status**: \u2705 PASS

## 10. Verification Gaps & Future Work

### 10.1 Current Gaps

1. **Un-annotated**: nothing says whether this is still open
2. **Dangling closure**: gone [[gap-status: closed-by tests/test_gone.py]]
3. **Open by design**: declared open [[gap-status: open]]

### 10.2 Planned Additions

1. something planned
"""
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert audit.unannotated_gaps == ["Un-annotated"]
    assert [row["pointer"] for row in audit.dangling_gap_closures] == ["tests/test_gone.py"]
    assert [g.status for g in audit.gaps] == ["unannotated", "closed-by", "open"]
    assert not audit.passed


def test_gap_status_annotation_parsing():
    text = open(os.path.join(REPO_ROOT, CATALOG), encoding="utf-8").read()
    entries, gaps = gate.parse_catalog(text, root=REPO_ROOT)
    statuses = {gap.text: gap.status for gap in gaps}
    assert statuses["3D Lid-Driven Cavity"] == "open"
    assert statuses["Moving Mesh"] == "closed-by"
    assert statuses["Compressible Flow"] == "closed-by"
    assert len(entries) >= 30


def test_cli_exit_code(tmp_path, capsys):
    ok = gate.main(["--catalog", CATALOG, "--root", REPO_ROOT])
    out = capsys.readouterr().out
    assert ok == 0
    assert "verdict: **PASS**" in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))