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
"""

import os
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


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