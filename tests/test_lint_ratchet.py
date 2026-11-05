"""Contract tests for the split lint gate (pyproject config + changed-file ratchet).

The CI lint job used to be red by construction (no ruff configuration -> the
default rule set over a legacy tree).  These tests pin the two halves of the
replacement gate so a later edit cannot quietly turn it back into a no-signal or
never-green check.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
RATCHET = REPO / "scripts" / "lint_ratchet.py"

pytestmark = pytest.mark.skipif(
    shutil.which("ruff") is None, reason="ruff is not installed"
)


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, cwd=REPO, capture_output=True, text=True, check=False)


def test_declared_rule_set_is_the_correctness_gate() -> None:
    """[tool.ruff.lint].select must stay the satisfiable correctness set."""
    if sys.version_info >= (3, 11):
        import tomllib
    else:  # pragma: no cover - CI runs 3.11/3.12
        pytest.skip("tomllib requires Python 3.11+")
    config = tomllib.loads((REPO / "pyproject.toml").read_text())
    select = config["tool"]["ruff"]["lint"]["select"]
    assert set(select) == {"E9", "F63", "F7", "F82"}
    # The tree-wide gate is only honest if it actually passes.
    result = _run("ruff", "check", "src", "tests", "scripts")
    assert result.returncode == 0, result.stdout + result.stderr


def test_ratchet_passes_clean_file(tmp_path: Path) -> None:
    clean = tmp_path / "clean_module.py"
    clean.write_text("import math\n\n\nprint(math.pi)\n")
    assert _run(sys.executable, str(RATCHET), "--files", str(clean)).returncode == 0


def test_ratchet_flags_backlog_tax(tmp_path: Path) -> None:
    """A touched file with unused imports/unused locals must fail the ratchet."""
    dirty = tmp_path / "dirty_module.py"
    dirty.write_text("import math\nimport json  # unused\n\n\ndef f():\n    dead = 1\n")
    result = _run(sys.executable, str(RATCHET), "--files", str(dirty))
    assert result.returncode == 1
    assert "F401" in result.stdout and "F841" in result.stdout


def test_backlog_report_never_fails() -> None:
    result = _run(sys.executable, str(RATCHET), "--report")
    assert result.returncode == 0
    assert "informational" in result.stdout


def test_unresolvable_base_skips_instead_of_failing() -> None:
    """A fork/shallow clone must degrade to a skip, not to a red job."""
    result = _run(sys.executable, str(RATCHET), "--base", "refs/heads/does-not-exist")
    assert result.returncode == 0
    assert "skipping" in result.stdout or "no touched" in result.stdout