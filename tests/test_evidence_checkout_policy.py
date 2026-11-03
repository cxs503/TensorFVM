"""The evidence-checkout policy itself must be tested, not trusted.

Three properties are pinned here, and each one is a way the guard could rot
into a silent pass or a spurious failure:

1. absent artefact -> ``pytest.skip`` (never a plain ``return`` / ``pass``);
2. present artefact -> the real path, so the owning test runs unchanged;
3. ``TENSORFVM_EVIDENCE_ROOT`` redirects resolution, which is what lets the
   guarded tests be exercised from a sparse worktree against a materialised
   ``docs/`` tree.
"""

from pathlib import Path

import pytest

from tests.evidence_support import REPO_ROOT, evidence_root, published


def test_missing_evidence_skips_with_an_actionable_reason(tmp_path, monkeypatch):
    monkeypatch.setenv("TENSORFVM_EVIDENCE_ROOT", str(tmp_path))
    with pytest.raises(pytest.skip.Exception) as excinfo:
        published("docs/suite-channel/does-not-exist.json")
    message = str(excinfo.value)
    assert "does-not-exist.json" in message
    assert "TENSORFVM_EVIDENCE_ROOT" in message


def test_present_evidence_resolves_to_the_artefact(tmp_path, monkeypatch):
    artefact = tmp_path / "docs" / "case" / "summary.json"
    artefact.parent.mkdir(parents=True)
    artefact.write_text("{}")
    monkeypatch.setenv("TENSORFVM_EVIDENCE_ROOT", str(tmp_path))
    assert published("docs/case/summary.json") == artefact


def test_evidence_root_defaults_to_the_repository_root(monkeypatch):
    monkeypatch.delenv("TENSORFVM_EVIDENCE_ROOT", raising=False)
    assert evidence_root() == REPO_ROOT
    assert isinstance(evidence_root(), Path)