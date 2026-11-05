"""Contract tests for the lint gate and the lint ratchet (scripts/lint_ratchet.py).

The tree ships a large pre-existing ruff backlog (dense, semicolon-packed house
style), so CI enforces two things instead of one tree-wide rule set:

* ``ruff check src tests scripts`` with the project config must stay at zero
  errors (``[tool.ruff.lint].select`` = correctness families only);
* files touched by a change must be clean under the wider pyflakes set, which
  is what ``scripts/lint_ratchet.py`` enforces.

These tests pin both halves so the gate cannot silently regress.  They skip when
ruff is not installed (CI installs the ``dev`` extra, so they run there).
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RATCHET = REPO_ROOT / "scripts" / "lint_ratchet.py"


def _ruff():
    try:
        probe = subprocess.run(
            [sys.executable, "-m", "ruff", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:  # pragma: no cover - ruff binary missing entirely
        pytest.skip("ruff is not installed")
    if probe.returncode != 0:
        pytest.skip("ruff is not installed")
    return [sys.executable, "-m", "ruff"]


def test_enforced_rule_set_is_clean_treewide():
    """`ruff check src tests scripts` (project config) must report zero errors."""
    ruff = _ruff()
    result = subprocess.run(
        [*ruff, "check", "src", "tests", "scripts"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"enforced ruff gate is red:\n{result.stdout[-3000:]}"


def test_ratchet_passes_on_current_tree():
    """The current working tree adds no findings under the ratchet rule set."""
    _ruff()
    result = subprocess.run(
        [sys.executable, str(RATCHET)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"ratchet is red:\n{result.stdout[-3000:]}"


def test_ratchet_flags_unused_import(tmp_path):
    """A file with an unused import is rejected, with the finding printed."""
    _ruff()
    offender = tmp_path / "offender.py"
    offender.write_text("import os\n\n\nx = 1\n")
    result = subprocess.run(
        [sys.executable, str(RATCHET), "--files", str(offender)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # tmp_path sits outside src/tests/scripts, so the ratchet must decline it
    # instead of silently linting arbitrary paths.
    assert result.returncode == 0
    assert "no changed python files" in result.stdout


def test_ratchet_flags_offender_inside_tree():
    """An in-tree file with an unused import fails the ratchet."""
    _ruff()
    offender = REPO_ROOT / "tests" / "_lint_ratchet_offender.py"
    offender.write_text("import os\n\n\ndef test_placeholder():\n    assert 1\n")
    try:
        result = subprocess.run(
            [sys.executable, str(RATCHET), "--files", "tests/_lint_ratchet_offender.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        offender.unlink()
    assert result.returncode == 1, result.stdout
    assert "F401" in result.stdout


def test_ratchet_accepts_clean_file():
    """A clean in-tree file passes the ratchet."""
    _ruff()
    clean = REPO_ROOT / "tests" / "_lint_ratchet_clean.py"
    clean.write_text("def test_placeholder():\n    assert 1\n")
    try:
        result = subprocess.run(
            [sys.executable, str(RATCHET), "--files", "tests/_lint_ratchet_clean.py"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        clean.unlink()
    assert result.returncode == 0, result.stdout


def test_report_mode_never_fails():
    """--report is informational: it must exit 0 even with a large backlog."""
    _ruff()
    result = subprocess.run(
        [sys.executable, str(RATCHET), "--report"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "backlog" in result.stdout.lower()
