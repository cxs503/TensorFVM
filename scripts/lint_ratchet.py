#!/usr/bin/env python3
"""Lint ratchet: stop the ruff backlog from growing while it is paid down.

Why this exists
---------------
``ruff check src tests scripts`` reports ~9.9k findings on this tree, almost all
of them pycodestyle *layout* findings (E701/E702: the project writes dense,
semicolon-packed kernels and one-line test bodies on purpose).  A tree-wide
blocking gate on the default rule set is therefore impossible today, and a
permanently red CI job carries no signal.

The gate is split in two, which is what ``pyproject.toml`` ->
``[tool.ruff.lint].select`` and this script implement:

1. Tree-wide, blocking: only the correctness families that are *already* clean
   (E9 syntax/runtime errors, F63/F7/F82 real pyflakes bugs).  See
   ``ruff check src tests scripts``.
2. Changed files only: the whole pyflakes family (``F``) plus E9.  Editing a
   file therefore cannot introduce unused imports/variables, f-string typos,
   redefinitions or undefined names, so the F backlog strictly shrinks.
   Re-export modules (``__init__.py``) are exempt from F401, as usual.

Usage
-----
    python scripts/lint_ratchet.py                     # files in HEAD commit
    python scripts/lint_ratchet.py --base origin/main  # files vs merge base
    python scripts/lint_ratchet.py --files a.py b.py   # explicit file list
    python scripts/lint_ratchet.py --report            # backlog table, exit 0

Exit codes: 0 = ratchet holds (or nothing to check / ruff missing),
1 = a changed file adds findings, 2 = bad usage.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LINT_DIRS = ("src", "tests", "scripts")
# Rule families applied to *changed* files only.
RATCHET_SELECT = "E9,F"
# Re-export modules legitimately import names they do not use.
PER_FILE_IGNORES = "__init__.py:F401"


def _ruff_cmd(args: list[str]) -> list[str]:
    """Prefer the module entry point so the pinned dev extra is used."""
    return [sys.executable, "-m", "ruff", *args]


def _run(cmd: list[str], cwd: Path = REPO_ROOT) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)


def ruff_available() -> bool:
    result = _run(_ruff_cmd(["--version"]))
    if result.returncode != 0:
        print("lint_ratchet: ruff is not installed - ratchet skipped")
        return False
    return True


def _git(args: list[str]) -> list[str]:
    result = _run(["git", *args])
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _keep(path: str) -> bool:
    candidate = Path(path)
    if candidate.suffix != ".py":
        return False
    return candidate.parts and candidate.parts[0] in LINT_DIRS


def targets_from_git(base: str | None) -> list[str]:
    """Python files to hold to the wider rule set.

    ``base`` is a ref that the merge base is taken against (a PR's target
    branch).  When it cannot be resolved - shallow clone, detached HEAD, no
    remote - fall back to the files touched by the HEAD commit, and finally to
    the staged/working-tree changes, so a local run always has something to say.
    """
    paths: list[str] = []
    if base:
        if not _git(["rev-parse", "--verify", "--quiet", base]):
            print(f"lint_ratchet: base ref '{base}' not found - falling back to HEAD commit")
        else:
            paths = _git(["diff", "--name-only", "--diff-filter=ACMR", f"{base}...HEAD"])
    if not paths:
        paths = _git(["show", "--pretty=format:", "--name-only", "HEAD"])
    if not paths:
        paths = _git(["diff", "--name-only", "--diff-filter=ACMR", "HEAD"])
    return [p for p in dict.fromkeys(paths) if _keep(p) and (REPO_ROOT / p).is_file()]


def check_files(files: list[str]) -> int:
    if not files:
        print("lint_ratchet: no changed python files - nothing to ratchet")
        return 0
    result = _run(
        _ruff_cmd(
            [
                "check",
                "--isolated",
                "--select",
                RATCHET_SELECT,
                "--per-file-ignores",
                PER_FILE_IGNORES,
                "--output-format",
                "concise",
                *files,
            ]
        )
    )
    if result.returncode == 0:
        print(f"lint_ratchet: {len(files)} changed file(s) clean under select={RATCHET_SELECT}")
        return 0
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    print(
        f"lint_ratchet: FAIL - changed files add findings under select={RATCHET_SELECT}."
        " Fix them (or the file's pre-existing backlog before editing it)."
    )
    return 1


def report_backlog() -> int:
    """Print the tree-wide backlog so drift is visible; never fails the build."""
    result = _run(_ruff_cmd(["check", "--isolated", "--statistics", *LINT_DIRS]))
    table = result.stdout.strip() or result.stderr.strip()
    print("ruff backlog (informational, default rule set):")
    for line in table.splitlines()[-15:]:
        print(f"  {line}")
    fmt = _run(_ruff_cmd(["format", "--check", "--isolated", *LINT_DIRS]))
    unformatted = sum(1 for li in fmt.stdout.splitlines() if li.startswith("Would reformat"))
    print(f"  ruff format --check: {unformatted} file(s) would be reformatted")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=None, help="ref to diff against (e.g. origin/main)")
    parser.add_argument("--files", nargs="*", default=None, help="explicit file list")
    parser.add_argument("--report", action="store_true", help="print backlog table and exit 0")
    args = parser.parse_args(argv)

    if not ruff_available():
        return 0
    if args.report:
        return report_backlog()
    files = args.files if args.files is not None else targets_from_git(args.base)
    return check_files([f for f in files if _keep(f)])


if __name__ == "__main__":
    raise SystemExit(main())
