"""Tests for the FVM-SLIM-GUARD anti-rot trio (D6).

Covers: pre-commit blob gate (block / pass / escape hatch), CI blob gate
over a diff range, pack threshold logic (pure), hook wiring, CI workflow
wiring, and the MIGRATION skeleton.  Stdlib + git only (no torch).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARD = REPO_ROOT / "scripts" / "guard" / "repo_guard.py"
sys.path.insert(0, str(REPO_ROOT / "scripts" / "guard"))

import repo_guard  # noqa: E402


def _git(tmp_path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout


@pytest.fixture()
def bare_repo(tmp_path: Path) -> Path:
    """A throwaway git repo (the guard is pointed at it via env override)."""
    _git(tmp_path, "init", "-q", "-b", "main")
    _git(tmp_path, "config", "user.email", "guard@test")
    _git(tmp_path, "config", "user.name", "guard-test")
    (tmp_path / "seed.txt").write_text("seed\n")
    _git(tmp_path, "add", "seed.txt")
    _git(tmp_path, "commit", "-qm", "seed")
    return tmp_path


def _run_guard(repo: Path, *args: str,
               env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env.pop("FVM_GUARD_ALLOW_LARGE", None)
    env["FVM_GUARD_REPO_ROOT"] = str(repo)
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        [sys.executable, str(GUARD), *args],
        cwd=repo, capture_output=True, text=True, env=env,
    )


def test_staged_blocks_large_blob(bare_repo: Path) -> None:
    big = bare_repo / "big.bin"
    big.write_bytes(b"\0" * (600 * 1024))
    _git(bare_repo, "add", "big.bin")
    proc = _run_guard(bare_repo, "staged")
    assert proc.returncode == 1
    assert "big.bin" in proc.stderr
    assert "512.0 KiB" in proc.stderr


def test_staged_passes_small_blob(bare_repo: Path) -> None:
    small = bare_repo / "small.txt"
    small.write_text("x" * 100)
    _git(bare_repo, "add", "small.txt")
    proc = _run_guard(bare_repo, "staged")
    assert proc.returncode == 0, proc.stderr


def test_staged_escape_hatch(bare_repo: Path) -> None:
    big = bare_repo / "big.bin"
    big.write_bytes(b"\0" * (600 * 1024))
    _git(bare_repo, "add", "big.bin")
    proc = _run_guard(bare_repo, "staged",
                     env_extra={"FVM_GUARD_ALLOW_LARGE": "1"})
    assert proc.returncode == 0, proc.stderr


def test_ci_blob_gate_flags_large_changed_file(bare_repo: Path) -> None:
    big = bare_repo / "big2.bin"
    big.write_bytes(b"\1" * (600 * 1024))
    _git(bare_repo, "add", "big2.bin")
    _git(bare_repo, "commit", "-qm", "add big (pre-guard era)")
    # no origin/main in the throwaway repo -> base falls back to HEAD~1
    proc = _run_guard(bare_repo, "ci")
    assert proc.returncode == 1
    assert "big2.bin" in proc.stderr


def test_ci_blob_gate_passes_small_change(bare_repo: Path) -> None:
    small = bare_repo / "small2.txt"
    small.write_text("y" * 50)
    _git(bare_repo, "add", "small2.txt")
    _git(bare_repo, "commit", "-qm", "add small")
    proc = _run_guard(bare_repo, "ci")
    assert proc.returncode == 0, proc.stderr


def test_evaluate_pack_thresholds() -> None:
    base = 1_000_000_000
    baseline = {"pack_bytes": base}
    warn_delta, hard_delta, cap = 50_000_000, 150_000_000, 2_000_000_000
    ok = repo_guard.evaluate_pack(base + 10_000_000, baseline,
                               warn_delta, hard_delta, cap)
    assert ok["pack_status"] == "OK"
    warn = repo_guard.evaluate_pack(base + 60_000_000, baseline,
                                    warn_delta, hard_delta, cap)
    assert warn["pack_status"] == "WARN"
    fail = repo_guard.evaluate_pack(base + 160_000_000, baseline,
                                    warn_delta, hard_delta, cap)
    assert fail["pack_status"] == "FAIL"
    # absolute backstop fires even with a fresh (small) baseline
    fresh = {"pack_bytes": 100_000_000}
    over = repo_guard.evaluate_pack(2_100_000_000, fresh,
                                    warn_delta, hard_delta, cap)
    assert over["pack_status"] == "FAIL"
    # no baseline -> only the hard cap applies
    nobase = repo_guard.evaluate_pack(1_500_000_000, None,
                                      warn_delta, hard_delta, cap)
    assert nobase["pack_status"] == "OK"


def test_hook_wiring() -> None:
    hook = REPO_ROOT / ".githooks" / "pre-commit"
    assert hook.exists(), ".githooks/pre-commit missing"
    assert os.access(hook, os.X_OK), "pre-commit hook not executable"
    text = hook.read_text()
    assert "repo_guard.py" in text and "staged" in text


def test_ci_workflow_wiring() -> None:
    wf = REPO_ROOT / ".github" / "workflows" / "repo-guard.yml"
    assert wf.exists(), ".github/workflows/repo-guard.yml missing"
    text = wf.read_text()
    assert "repo_guard.py" in text and "ci" in text


def test_migration_skeleton() -> None:
    doc = REPO_ROOT / "docs" / "MIGRATION.md"
    assert doc.exists(), "docs/MIGRATION.md missing"
    text = doc.read_text()
    for section in ("触发条件", "当前基线", "D1", "runbook", "回滚", "状态"):
        assert section in text, f"MIGRATION.md missing section: {section}"


def test_baseline_file_present() -> None:
    bl = REPO_ROOT / "scripts" / "guard" / "pack_baseline.json"
    assert bl.exists(), "pack_baseline.json missing (run repo_guard.py reanchor)"
    data = json.loads(bl.read_text())
    assert data["pack_bytes"] > 0
