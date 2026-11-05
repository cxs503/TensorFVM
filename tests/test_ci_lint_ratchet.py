"""Contract tests for the scoped ruff gate in CI.

The lint job used to sweep the whole tree, which carries a multi-thousand-finding
historic ruff backlog: it failed even for a docs-only pull request (PR #103),
so it reported nothing about the change under test.  ``scripts/ci/lint_changed.sh``
now resolves the Python files a change set touches and
``scripts/ci/ruff_ratchet.py`` judges their *delta* against the base revision.

These tests pin the properties the gate relies on:

1. the workflow really calls the scoped gate and no longer sweeps the tree;
2. a change set that touches no Python file is green;
3. new files must be clean (an unused import in a new file is red);
4. editing a file that was already dirty is allowed as long as the change adds
   no finding -- otherwise the gate would freeze every pull request that
   touches existing code, i.e. repeat the original failure one step later;
5. adding a finding to such a file is red.

Cases 2-5 build throw-away git repositories, so they are fast and independent
of this repository's history.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "ci" / "lint_changed.sh"
HELPER = REPO_ROOT / "scripts" / "ci" / "ruff_ratchet.py"
WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

needs_ruff = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="ruff is not installed in this environment"
)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _new_repo(tmp_path: Path, files: dict[str, str] | None = None) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.email", "ci@example.invalid")
    _git(repo, "config", "user.name", "ci")
    (repo / "README.rst").write_text("baseline\n")
    for rel, text in (files or {}).items():
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    return repo, _commit(repo, "baseline")


def _run_gate(repo: Path, base: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, BASE_SHA=base)
    env.pop("PUSH_BEFORE", None)
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=repo, capture_output=True, text=True, env=env
    )


def test_workflow_uses_the_scoped_gate():
    workflow = WORKFLOW.read_text()
    # Only executable lines matter: comments may name the old whole-tree
    # commands when they explain why the gate moved.
    commands = "\n".join(
        line for line in workflow.splitlines() if not line.strip().startswith("#")
    )
    assert "scripts/ci/lint_changed.sh" in commands
    assert "ruff check src tests scripts" not in commands
    assert "ruff format --check src tests scripts" not in commands
    # The scoped gate diffs against the base revision, so history is required.
    assert "fetch-depth: 0" in commands


def test_scripts_are_present_and_parse():
    assert SCRIPT.is_file() and os.access(SCRIPT, os.X_OK)
    assert HELPER.is_file()
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)


@needs_ruff
def test_docs_only_change_set_is_green(tmp_path):
    repo, base = _new_repo(tmp_path)
    (repo / "notes").mkdir()
    (repo / "notes" / "results.md").write_text("# results\n")
    (repo / "notes" / "audit.json").write_text("{}\n")
    _commit(repo, "docs only")

    result = _run_gate(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "touches no Python file" in result.stdout


@needs_ruff
def test_new_unclean_file_is_red(tmp_path):
    repo, base = _new_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_dirty.py").write_text("import os\n\n\nVALUE = 1\n")
    _commit(repo, "add unclean test module")

    result = _run_gate(repo, base)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "F401" in (result.stdout + result.stderr)


@needs_ruff
def test_new_clean_file_is_green(tmp_path):
    repo, base = _new_repo(tmp_path)
    (repo / "tests").mkdir()
    (repo / "tests" / "test_clean.py").write_text(
        '"""A clean module."""\n\n\ndef test_add():\n    assert 1 + 1 == 2\n'
    )
    _commit(repo, "add clean test module")

    result = _run_gate(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "no new lint debt" in result.stdout


LEGACY = "import os\n\nVALUE=1\n"


@needs_ruff
def test_editing_a_legacy_dirty_file_is_not_blocked(tmp_path):
    """The legacy backlog must not turn unrelated edits red."""
    repo, base = _new_repo(tmp_path, {"tests/test_legacy.py": LEGACY})
    (repo / "tests" / "test_legacy.py").write_text(
        LEGACY + "\n\ndef test_extra():\n    assert True\n"
    )
    _commit(repo, "extend the legacy module without new debt")

    result = _run_gate(repo, base)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pre-existing finding(s), unchanged" in result.stdout


@needs_ruff
def test_adding_a_finding_to_a_legacy_file_is_red(tmp_path):
    repo, base = _new_repo(tmp_path, {"tests/test_legacy.py": LEGACY})
    (repo / "tests" / "test_legacy.py").write_text(LEGACY + "\nimport sys\n")
    _commit(repo, "add a second unused import")

    result = _run_gate(repo, base)
    assert result.returncode != 0, result.stdout + result.stderr
    assert "new ruff findings" in result.stderr
