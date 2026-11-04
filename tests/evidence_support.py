"""Resolve the published evidence archives that tests re-derive numbers from.

Several tests recompute published quantities from tracked artefacts under
``docs/`` (audit manifests, per-grid field archives, parameter studies, the
archived CAD source of a benchmark hull).  Those artefacts are tracked in git
but they are not shipped by every checkout:

* the working tree used for solver development is a sparse cone
  (``src/ tests/ scripts/`` plus two top-level files), so ``docs/`` is absent;
* a few evidence directories are intentionally "produced, not published" (for
  example ``docs/verification-external-*/summary.json``).

A missing artefact is a *checkout* condition, not a violated contract, so the
owning test must **skip with an explicit reason** instead of failing, and must
run unchanged whenever the artefacts are present.  ``published`` implements
exactly that rule; ``TENSORFVM_EVIDENCE_ROOT`` points it at an out-of-tree
materialisation of ``docs/`` so the guarded tests stay executable from a sparse
worktree (see ``scripts/verification-evidence-checkout/run_evidence_tests.sh``).

``tracked`` is the same rule for git-tracked artefacts that live *outside*
``docs/`` -- most importantly ``benchmarks/*.json`` validation specs.  Those
specs are tracked in the repository root, so they are absent from the sparse
cone but still recoverable from the object store; reading them by path with a
plain ``Path.read_text`` made three SUBOFF validation tests red on every clean
worktree while their contract was intact.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_BLOB_CACHE_DIR: Path | None = None


def evidence_root() -> Path:
    """Root that ``relative`` evidence paths are resolved against."""
    return Path(os.environ.get("TENSORFVM_EVIDENCE_ROOT", REPO_ROOT))


def published(relative: str) -> Path:
    """Return ``relative`` under the evidence root, or skip with the reason.

    ``relative`` always starts with ``docs/`` so that a materialised tree and a
    plain checkout resolve identically.
    """
    path = evidence_root() / relative
    if not path.exists():
        pytest.skip(
            "tracked evidence absent from this checkout: "
            f"{relative} (sparse cone excludes docs/; set "
            "TENSORFVM_EVIDENCE_ROOT to a materialised docs/ tree to run it)"
        )
    return path


def _worktree(relative: str) -> Path:
    """Worktree location of a repo-relative path (may not exist)."""
    return REPO_ROOT / relative


def _git_blob(relative: str) -> bytes | None:
    """``HEAD:<relative>`` blob bytes, or ``None`` when git cannot supply them."""
    try:
        completed = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "show", f"HEAD:{relative}"],
            capture_output=True,
            check=False,
        )
    except OSError:  # no git executable in the image
        return None
    if completed.returncode != 0 or not completed.stdout:
        return None
    return completed.stdout


def _blob_cache_dir() -> Path:
    global _BLOB_CACHE_DIR
    if _BLOB_CACHE_DIR is None or not _BLOB_CACHE_DIR.is_dir():
        _BLOB_CACHE_DIR = Path(tempfile.mkdtemp(prefix="tensorfvm-tracked-"))
    return _BLOB_CACHE_DIR


def tracked(relative: str) -> Path:
    """Resolve a git-tracked repo-relative artefact, or skip with the reason.

    Resolution order: the worktree, a ``TENSORFVM_EVIDENCE_ROOT``
    materialisation, the ``HEAD`` blob extracted into a temporary cache, then
    ``pytest.skip``.  The blob step is what keeps ``benchmarks/*.json`` specs
    executable from a sparse cone without weakening the assertion: the bytes
    are the tracked ones, not a substitute.
    """
    parts = Path(relative).parts
    if Path(relative).is_absolute() or ".." in parts:
        raise ValueError("repository-relative artefact path required")
    candidate = _worktree(relative)
    if candidate.is_file():
        return candidate
    materialised = evidence_root() / relative
    if materialised.is_file():
        return materialised
    cached = _blob_cache_dir() / relative
    if cached.is_file():
        return cached
    payload = _git_blob(relative)
    if payload is None:
        pytest.skip(
            "git-tracked artefact absent from this checkout and not resolvable "
            f"from the object store: {relative} (sparse cone excludes it and no "
            "HEAD blob is available; set TENSORFVM_EVIDENCE_ROOT to a "
            "materialised tree to run it)"
        )
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(payload)
    return cached