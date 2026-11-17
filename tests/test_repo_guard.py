"""Tests for the FVM-SLIM-GUARD anti-rot trio (D6, R4).

Covers: pre-commit blob gate (block / pass / escape hatch / .gitattributes
whitelist), CI blob gate over a diff range, pack threshold logic (pure),
LIVE artifact volume gate (pure + standalone), hook wiring, CI workflow
wiring, and the MIGRATION-m23216 skeleton.  Stdlib + git only (no torch).

NOTE: this worktree is a sparse-checkout (blob:none partial clone); the
guard *wiring* files (.githooks/, .github/) are tracked in git but may not
be materialized in the working directory.  The wiring tests therefore read
the tracked content from the index/HEAD (git show / git ls-files) rather
than the working tree — this is the correct, robust check for a sparse
worktree.
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


def test_live_artifact_bytes_tolerates_submodule(monkeypatch) -> None:
    """`git ls-tree -r -l` prints "-" as the size of a submodule entry.

    live_artifact_bytes must skip it instead of raising ValueError
    (a submodule would otherwise crash the CI size gate).
    """
    fake = "\n".join([
        "100644 blob e69de29bb2d1d6434b8b29ae775ad8c2e48c5391 5\tsmall.txt",
        "160000 commit 1234567890123456789012345678901234567890 -\tsub",
    ])
    monkeypatch.setattr(repo_guard, "_git", lambda *a, **k: fake)
    assert repo_guard.live_artifact_bytes() == 5


def _git(tmp_path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=tmp_path, capture_output=True, text=True, check=True
    ).stdout


def _git_show(path: str) -> str:
    """Read a tracked file's content from HEAD (robust in sparse worktrees)."""
    return subprocess.run(
        ["git", "show", f"HEAD:{path}"], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
    ).stdout


def _git_ls_files(path: str) -> str:
    return subprocess.run(
        ["git", "ls-files", path], cwd=REPO_ROOT,
        capture_output=True, text=True, check=True,
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


def test_staged_whitelist_bypasses_large_blob(bare_repo: Path) -> None:
    """A large blob matching `guard-allow-large` is whitelisted (passes)."""
    # Set up the whitelist attribute in the throwaway repo.
    (bare_repo / ".gitattributes").write_text(
        "docs/verification-*/**/*.npz  guard-allow-large\n")
    _git(bare_repo, "add", ".gitattributes")
    _git(bare_repo, "commit", "-qm", "add gitattributes")
    # A large npz under the whitelisted path.
    big = bare_repo / "docs" / "verification-x" / "f" / "fields.npz"
    big.parent.mkdir(parents=True)
    big.write_bytes(b"\0" * (600 * 1024))
    _git(bare_repo, "add", str(big))
    proc = _run_guard(bare_repo, "staged")
    assert proc.returncode == 0, proc.stderr


def test_staged_whitelist_does_not_bypass_unlisted(bare_repo: Path) -> None:
    """A large blob NOT matching the whitelist is still blocked."""
    (bare_repo / ".gitattributes").write_text(
        "docs/verification-*/**/*.npz  guard-allow-large\n")
    _git(bare_repo, "add", ".gitattributes")
    _git(bare_repo, "commit", "-qm", "add gitattributes")
    big = bare_repo / "docs" / "verification-x" / "f" / "fields.bin"
    big.parent.mkdir(parents=True)
    big.write_bytes(b"\0" * (600 * 1024))
    _git(bare_repo, "add", str(big))
    proc = _run_guard(bare_repo, "staged")
    assert proc.returncode == 1, proc.stderr


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


def test_evaluate_live_thresholds() -> None:
    """Live volume gate: OK / WARN / RED and the no-baseline fallback."""
    base = 183 * 1024 * 1024
    baseline = {"live_bytes": base}
    warn_delta, red_delta = 50 * 1024 * 1024, 100 * 1024 * 1024
    ok = repo_guard.evaluate_live(base + 10_000_000, baseline,
                                  warn_delta, red_delta)
    assert ok["live_status"] == "OK"
    warn = repo_guard.evaluate_live(base + 60_000_000, baseline,
                                    warn_delta, red_delta)
    assert warn["live_status"] == "WARN"
    red = repo_guard.evaluate_live(base + 110_000_000, baseline,
                                   warn_delta, red_delta)
    assert red["live_status"] == "RED"
    # no recorded baseline -> falls back to the frozen 183 MiB reference
    nobase = repo_guard.evaluate_live(base + 10_000_000, None,
                                      warn_delta, red_delta)
    assert nobase["live_status"] == "OK"
    assert nobase["baseline_bytes"] == repo_guard.DEFAULT_LIVE_BASELINE


def test_live_artifact_bytes_dedup(bare_repo: Path) -> None:
    """live_artifact_bytes() dedups by blob sha (same blob, 2 paths = 1)."""
    data = b"\xab" * 1000
    (bare_repo / "a" / "f1.bin").parent.mkdir(parents=True)
    (bare_repo / "a" / "f1.bin").write_bytes(data)
    (bare_repo / "b" / "f2.bin").parent.mkdir(parents=True)
    (bare_repo / "b" / "f2.bin").write_bytes(data)  # same content -> same blob
    _git(bare_repo, "add", "a/f1.bin", "b/f2.bin")
    _git(bare_repo, "commit", "-qm", "two paths one blob")
    # Run the `live` subcommand in the throwaway repo (env points REPO_ROOT
    # at it); seed.txt (5 B) + one 1000 B blob (deduped) = 1005 B.
    proc = _run_guard(bare_repo, "live")
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    assert out["live_bytes"] == 1005


def test_cmd_live_ok(bare_repo: Path) -> None:
    """Standalone `live` subcommand returns 0 on a small repo."""
    proc = _run_guard(bare_repo, "live")
    assert proc.returncode == 0, proc.stderr
    assert "live" in proc.stdout


def test_hook_wiring() -> None:
    """The pre-commit hook is tracked and calls the staged gate."""
    # Robust for sparse worktrees: read the tracked content from HEAD.
    assert ".githooks/pre-commit" in _git_ls_files(".githooks/pre-commit")
    text = _git_show(".githooks/pre-commit")
    assert "repo_guard.py" in text and "staged" in text


def test_ci_workflow_wiring() -> None:
    """The CI workflow is tracked and calls the ci gate."""
    assert ".github/workflows/repo-guard.yml" in _git_ls_files(
        ".github/workflows/repo-guard.yml")
    text = _git_show(".github/workflows/repo-guard.yml")
    assert "repo_guard.py" in text and "ci" in text


def test_gitattributes_whitelist_present() -> None:
    """The .gitattributes whitelist attribute is declared.

    Read from HEAD first: this worktree is a sparse checkout and
    .gitattributes may not be materialized (same strategy as the other
    wiring tests, per the module docstring).
    """
    proc = subprocess.run(
        ["git", "show", "HEAD:.gitattributes"], cwd=REPO_ROOT,
        capture_output=True, text=True,
    )
    if proc.returncode == 0:
        text = proc.stdout
    else:  # no HEAD (bare/empty repo) -> fall back to the working tree
        ga = REPO_ROOT / ".gitattributes"
        assert ga.exists(), ".gitattributes missing in HEAD and working tree"
        text = ga.read_text()
    assert "guard-allow-large" in text


def test_migration_skeleton() -> None:
    doc = REPO_ROOT / "docs" / "MIGRATION-m23216.md"
    assert doc.exists(), "docs/MIGRATION-m23216.md missing"
    text = doc.read_text()
    for section in ("触发条件", "当前基线", "D1", "runbook", "回滚",
                    "commit-map", "旧 SHA 对照表", "状态"):
        assert section in text, f"MIGRATION-m23216.md missing section: {section}"


def test_baseline_files_present() -> None:
    for bl, key in (("pack_baseline.json", "pack_bytes"),
                    ("live_baseline.json", "live_bytes")):
        p = REPO_ROOT / "scripts" / "guard" / bl
        assert p.exists(), f"{bl} missing (run repo_guard.py reanchor)"
        data = json.loads(p.read_text())
        assert data[key] > 0
