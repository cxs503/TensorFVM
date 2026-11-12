"""Tests for the verification-catalog gate (``tensorfvm.verification.catalog_gate``).

The gate is a *verification of the verifier's index*: it must fail on the
defects the historical catalog actually carried (dead test pointer, PASS entry
with no live test, un-annotated gap, and -- added at m16836 -- un-annotated
*planned* item) and pass on the repaired catalog committed next to it.
"""

from __future__ import annotations

import json
import os

import pytest

from tensorfvm.verification import catalog_gate as gate

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CATALOG = "docs/verification-catalog.md"

_ENTRY = """# Catalog

### 1.1 Cavity

**Implementation**: `src/tensorfvm/solver.py`
**Tests**: `tests/test_lid_cavity.py`

**Verification Status**: \u2705 PASS
"""


def _write(tmp_path, text):
    path = tmp_path / "catalog.md"
    path.write_text(text, encoding="utf-8")
    return str(path)


def _with_gaps(current: str, planned: str) -> str:
    return (
        _ENTRY
        + "\n## 10. Verification Gaps & Future Work\n\n### 10.1 Current Gaps\n\n"
        + current
        + "\n### 10.2 Planned Additions\n\n"
        + planned
    )


def test_committed_catalog_passes_the_gate():
    """The catalog shipped in this repo must be self-consistent."""

    audit = gate.audit_catalog(CATALOG, root=REPO_ROOT)
    data = audit.to_dict()
    assert audit.passed, json.dumps(data, indent=1)
    assert data["counts"]["entries"] >= 30
    assert data["counts"]["gaps"] >= 5
    # §10.2 is audited since m16836 -- it must not be empty and must be annotated
    assert data["counts"]["planned"] >= 5
    assert data["counts"]["planned_open"] == 0
    assert data["counts"]["planned_closed"] + data["counts"]["planned_partial"] == data[
        "counts"
    ]["planned"]


def test_gate_detects_dead_pointer(tmp_path):
    text = _ENTRY.replace("solver.py", "does_not_exist.py").replace(
        "test_lid_cavity.py", "test_does_not_exist.py"
    )
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert not audit.passed
    assert {row["pointer"] for row in audit.dangling_pointers} == {
        "src/tensorfvm/does_not_exist.py",
        "tests/test_does_not_exist.py",
    }


def test_gate_detects_pass_entry_without_live_test(tmp_path):
    text = _ENTRY.replace("`tests/test_lid_cavity.py`", "across the suite")
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert audit.unsupported_entries == ["1.1 Cavity"]


def test_gate_detects_unannotated_and_dangling_gap(tmp_path):
    text = _with_gaps(
        "1. **Un-annotated**: nothing says whether this is still open\n"
        "2. **Dangling closure**: gone [[gap-status: closed-by tests/test_gone.py]]\n"
        "3. **Open by design**: declared open [[gap-status: open]]\n",
        "1. **Planned, annotated**: still listed [[gap-status: open]]\n",
    )
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert audit.unannotated_gaps == ["Un-annotated"]
    assert [row["pointer"] for row in audit.dangling_gap_closures] == ["tests/test_gone.py"]
    assert [g.status for g in audit.gaps] == ["unannotated", "closed-by", "open"]
    assert not audit.passed


def test_gate_detects_unannotated_planned_item(tmp_path):
    """A *Planned Additions* line with no status is an unauditable claim."""

    text = _with_gaps(
        "1. **Gap**: declared open [[gap-status: open]]\n",
        "1. **Delivered but still listed as planned**\n"
        "2. **Still planned**: no delivery yet [[gap-status: open]]\n",
    )
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert audit.unannotated_planned == ["Delivered but still listed as planned"]
    assert audit.passed is False
    assert audit.to_dict()["counts"]["planned"] == 2
    assert audit.to_dict()["counts"]["planned_open"] == 1


def test_gate_detects_dangling_planned_closure(tmp_path):
    text = _with_gaps(
        "1. **Gap**: declared open [[gap-status: open]]\n",
        "1. **Closure pointing at a deleted artifact**: "
        "[[gap-status: closed-by tests/test_lid_cavity.py, docs/gone/summary.json]]\n",
    )
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert [row["pointer"] for row in audit.dangling_planned_closures] == [
        "docs/gone/summary.json"
    ]
    assert audit.passed is False


def test_gate_accepts_bounded_partial_closure(tmp_path):
    """``partially-closed-by`` is a first-class status: honest partial delivery.

    Its pointers are checked exactly like ``closed-by`` (a partial claim still
    needs evidence), and the binary statuses are not confused by the prefix.
    """

    text = _with_gaps(
        "1. **Gap**: declared open [[gap-status: open]]\n",
        "1. **Partial**: one Ra inside the band, one outside "
        "[[gap-status: partially-closed-by tests/test_lid_cavity.py]]\n"
        "2. **Partial with a dead pointer**: "
        "[[gap-status: partially-closed-by docs/gone/report.md]]\n",
    )
    path = _write(tmp_path, text)
    audit = gate.audit_catalog(path, root=REPO_ROOT)
    assert [g.status for g in audit.planned] == [
        "partially-closed-by",
        "partially-closed-by",
    ]
    assert [row["pointer"] for row in audit.dangling_planned_closures] == [
        "docs/gone/report.md"
    ]

    good = _with_gaps(
        "1. **Gap**: declared open [[gap-status: open]]\n",
        "1. **Partial**: one Ra inside the band, one outside "
        "[[gap-status: partially-closed-by tests/test_lid_cavity.py]]\n",
    )
    audit = gate.audit_catalog(_write(tmp_path, good), root=REPO_ROOT)
    assert audit.passed, json.dumps(audit.to_dict(), indent=1)
    assert audit.to_dict()["counts"]["planned_partial"] == 1


def test_gap_status_annotation_parsing():
    text = open(os.path.join(REPO_ROOT, CATALOG), encoding="utf-8").read()
    entries, gaps = gate.parse_catalog(text, root=REPO_ROOT)
    statuses = {gap.text: gap.status for gap in gaps}
    # §10.1: delivered gaps are closed-by once their artifacts exist
    assert statuses["3D Lid-Driven Cavity"] == "closed-by"
    assert statuses["Moving Mesh"] == "closed-by"
    assert statuses["Compressible Flow"] == "closed-by"
    # §10.2: the delivery decision is recorded, not left implied (m16836)
    assert statuses["Rotating Machinery"] == "closed-by"
    assert statuses["Heat Transfer"] == "partially-closed-by"
    assert statuses["Multiphase"] == "partially-closed-by"
    assert len(entries) >= 30


def test_cli_exit_code(tmp_path, capsys):
    ok = gate.main(["--catalog", CATALOG, "--root", REPO_ROOT])
    out = capsys.readouterr().out
    assert ok == 0
    assert "verdict: **PASS**" in out


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))