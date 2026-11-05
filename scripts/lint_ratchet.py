"""Lint ratchet: a change must not grow the ruff backlog in the files it touches.

The repository carries a large pre-existing ruff backlog (a deliberately dense
house style plus unused imports in older scripts), and ``ruff`` has no baseline
file, so a tree-wide ``ruff check`` cannot be made blocking without a 550-file
formatting campaign.  The CI lint gate is therefore split in two:

* ``ruff check src tests scripts`` -- blocking, but restricted to the
  correctness families declared in ``[tool.ruff.lint]`` of ``pyproject.toml``;
  those families already measure zero tree-wide, so the gate is satisfiable.
* this ratchet -- blocking, and it applies the *whole* pyflakes set plus the
  ``E9`` syntax family to every Python file the change touches, so new or
  edited code cannot add debt while the backlog is paid down.

Reproduction::

    python scripts/lint_ratchet.py --base origin/main   # gate (changed files)
    python scripts/lint_ratchet.py --report             # informational backlog
    python scripts/lint_ratchet.py --files a.py b.py    # ad-hoc, used by tests

Exit status is 0 when the gate passes, 1 when a touched file has findings, and
0 with a notice when the base revision cannot be resolved: a fork or a shallow
clone must not hard-fail the job for setup reasons.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

#: Rule families enforced on every touched file.  ``E9`` = syntax/runtime
#: errors, ``F`` = all of pyflakes (unused imports and variables, undefined
#: names, always-true comparisons, ...).  Stricter than the tree-wide gate on
#: purpose: that difference is what makes the gate ratchet instead of freeze.
RULES = "E9,F"

#: Directories that carry the production/test/script code guarded by CI.
ROOTS = ("src", "tests", "scripts")

REPO = Path(__file__).resolve().parents[1]


def _ruff() -> list[str]:
    """Command prefix for ruff (``python -m ruff`` when the script is missing)."""
    exe = shutil.which("ruff")
    if exe:
        return [exe]
    return [sys.executable, "-m", "ruff"]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=REPO, capture_output=True, text=True, check=False
    )


def _resolve_base(base: str | None) -> str | None:
    """Return a revision the diff can be taken against, or None if unresolvable.

    No implicit fallback: guessing another base would silently lint the wrong
    file set, so an unknown base degrades to a skip instead (the tree-wide gate
    still runs unconditionally in the same job).
    """
    if not base:
        return None
    probe = _git("rev-parse", "--verify", f"{base}^{{commit}}")
    return base if probe.returncode == 0 else None


def changed_python_files(base: str) -> list[str]:
    """Python files under :data:`ROOTS` that differ between ``base`` and HEAD."""
    merge_base = _git("merge-base", base, "HEAD")
    reference = merge_base.stdout.strip() if merge_base.returncode == 0 else base
    diff = _git(
        "diff", "--name-only", "--diff-filter=ACMR", reference, "HEAD", "--", *ROOTS
    )
    if diff.returncode != 0:
        raise RuntimeError(f"git diff failed against {reference}: {diff.stderr.strip()}")
    kept = []
    for name in diff.stdout.splitlines():
        name = name.strip()
        if name.endswith(".py") and (REPO / name).exists():
            kept.append(name)
    return sorted(kept)


def lint(files: list[str]) -> int:
    """Run the ratchet rule set over ``files``; return the ruff exit status."""
    if not files:
        return 0
    cmd = [*_ruff(), "check", "--isolated", "--no-fix", "--select", RULES,
           "--output-format", "concise", *files]
    return subprocess.run(cmd, cwd=REPO, check=False).returncode


def report() -> int:
    """Print the tree-wide backlog; never fails (informational CI step)."""
    stats = subprocess.run(
        [*_ruff(), "check", "--isolated", "--statistics", *ROOTS],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    lines = (stats.stdout + stats.stderr).strip().splitlines()
    print("ruff backlog over", " ".join(ROOTS), "(informational, not gating):")
    print("\n".join(f"  {line}" for line in lines) if lines else "  (none)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=None,
                        help="revision to diff against (e.g. origin/main)")
    parser.add_argument("--report", action="store_true",
                        help="print the tree-wide backlog and exit 0")
    parser.add_argument("--files", nargs="*", default=None,
                        help="check these files instead of the changed ones")
    args = parser.parse_args(argv)

    if args.report:
        return report()

    if args.files is not None:
        files = [name for name in args.files if name.endswith(".py")]
    else:
        base = _resolve_base(args.base)
        if base is None:
            print(f"lint ratchet: base {args.base!r} unavailable, nothing to compare "
                  "against; skipping (the tree-wide correctness gate still ran).")
            return 0
        files = changed_python_files(base)

    if not files:
        print("lint ratchet: no touched Python files, nothing to check.")
        return 0

    print(f"lint ratchet: {len(files)} touched file(s) must be clean under {RULES}:")
    print("\n".join(f"  {name}" for name in files))
    status = lint(files)
    if status == 0:
        print("lint ratchet: clean.")
    else:
        print("lint ratchet: findings above; fix them, or pay the rule down "
              "tree-wide and promote it to [tool.ruff.lint].select.")
    return status


if __name__ == "__main__":
    raise SystemExit(main())