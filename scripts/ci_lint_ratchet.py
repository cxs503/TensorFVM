#!/usr/bin/env python3
"""Changed-files ruff ratchet for the TensorFVM lint gate.

The tree still carries a large pre-existing ruff backlog (`ruff check src tests
scripts` reports >8k findings, mostly E702/E701 semicolon-packed statements, and
532 files are not in the formatter's style).  A whole-tree gate therefore cannot
be green without a reformat of every file, which this repository's delivery
pipeline cannot carry (commits are capped at 80 files; see
docs/ci-lint-policy.md).

The gate is narrowed instead of dropped: **new and modified Python must be
clean**, legacy files are left untouched until somebody edits them.  Concretely,
the ratchet checks the Python files this branch touched relative to the merge
base with the default branch:

    ruff check <changed files>
    ruff format --check <changed files>

Usage:
    python scripts/ci_lint_ratchet.py [--base origin/main] [--list-only]

Exit codes: 0 = clean (or nothing to check), 1 = lint/format failure,
2 = cannot determine the changed set (git error), 3 = ruff not installed.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Evidence/report trees are not gated: they hold generated scripts and
# reproducers that are archived as-is by the pusher (docs/<mission>/...).
EXCLUDED_PREFIXES = ("docs/", "artifacts/", "runs/")


def is_gated(path: str) -> bool:
    """True when *path* is a Python file the ratchet should lint."""
    return path.endswith(".py") and not path.startswith(EXCLUDED_PREFIXES)


def select_gated(paths: list[str]) -> list[str]:
    """Filter a raw ``git diff --name-only`` listing down to gated files."""
    return [p for p in paths if is_gated(p)]


def _run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def rev_exists(rev: str, cwd: Path) -> bool:
    return (
        _run(
            ["git", "rev-parse", "--verify", "-q", f"{rev}^{{commit}}"], cwd
        ).returncode
        == 0
    )


def changed_python_files(base: str, cwd: Path) -> list[str]:
    """Gated Python files changed on ``HEAD`` relative to ``base`` (merge-base diff)."""
    proc = _run(
        [
            "git",
            "diff",
            "--name-only",
            "--diff-filter=ACMR",
            f"{base}...HEAD",
            "--",
            "*.py",
        ],
        cwd,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or f"git diff against {base} failed")
    return select_gated(
        [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    )


def ruff_cmd(cwd: Path) -> list[str]:
    exe = shutil.which("ruff")
    if exe is None:  # pragma: no cover - CI installs the pinned release first
        raise FileNotFoundError("ruff not found on PATH (pip install 'ruff==0.15.0')")
    return [exe]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="origin/main", help="git ref to diff against")
    parser.add_argument(
        "--list-only", action="store_true", help="print the gated set and stop"
    )
    parser.add_argument(
        "--root", default=None, help="repository root (defaults to cwd)"
    )
    args = parser.parse_args(argv)

    root = Path(args.root or Path.cwd())
    if not rev_exists(args.base, root):
        print(
            f"ci_lint_ratchet: base ref {args.base!r} is not available", file=sys.stderr
        )
        return 2
    try:
        files = changed_python_files(args.base, root)
    except RuntimeError as exc:
        print(f"ci_lint_ratchet: {exc}", file=sys.stderr)
        return 2

    if not files:
        print(
            f"ci_lint_ratchet: no changed Python files relative to {args.base} — ratchet passes"
        )
        return 0
    print(f"ci_lint_ratchet: {len(files)} changed Python file(s) vs {args.base}")
    for path in files:
        print(f"  {path}")
    if args.list_only:
        return 0

    try:
        ruff = ruff_cmd(root)
    except FileNotFoundError as exc:
        print(f"ci_lint_ratchet: {exc}", file=sys.stderr)
        return 3

    failed = False
    for label, subcommand in (
        ("ruff check", ["check"]),
        ("ruff format --check", ["format", "--check"]),
    ):
        proc = subprocess.run([*ruff, *subcommand, "--no-cache", *files], cwd=root)
        print(f"ci_lint_ratchet: {label} -> exit {proc.returncode}")
        failed |= proc.returncode != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
