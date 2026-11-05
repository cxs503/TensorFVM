#!/usr/bin/env python3
"""Audit the ruff backlog that the CI lint gate has to live with.

Extracts the pre-fix tree with ``git archive`` and compares it against the
current checkout with ruff, so every number in the write-up (and in the figure)
is reproducible from a checkout - nothing is transcribed by hand.

    python scripts/audit_lint_gate.py                        # before = merge-base with origin/main
    python scripts/audit_lint_gate.py --before <ref>
    python scripts/audit_lint_gate.py --out docs/ci-lint-gate/lint_gate_audit.json \
        --figure docs/ci-lint-gate/lint_gate_backlog.png

Measured quantities
-------------------
* ``full_backlog``: findings of the ruff *default* rule set (E4,E7,E9,F) over
  ``src tests scripts``.  This is what a naive tree-wide gate would report, and
  why the old CI job was permanently red.
* ``changed_files_ratchet``: findings under the ratchet set (E9 + all of F) for
  the python files the change touches.  This is the half of the gate that new
  code must satisfy.
* ``format_unformatted``: files ``ruff format --check`` would rewrite at the
  project line length (informational step).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("src", "tests", "scripts")
LINE_LENGTH = "120"
RATCHET_SELECT = "E9,F"
ENFORCED_SELECT = "E9,F63,F7,F82"


def ruff(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "ruff", *args], cwd=cwd, capture_output=True, text=True, check=False)


def statistics(tree: Path) -> dict[str, int]:
    """Per-rule findings for the default rule set, isolated from project config."""
    result = ruff(["check", "--isolated", "--statistics", *TARGETS], tree)
    counts: dict[str, int] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit() and parts[1][:1] in "EFW":
            counts[parts[1]] = int(parts[0])
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def findings(paths: list[str], tree: Path, select: str) -> int:
    existing = [p for p in paths if (tree / p).is_file()]
    if not existing:
        return 0
    result = ruff(["check", "--isolated", "--select", select, *existing], tree)
    for line in reversed(result.stdout.splitlines()):
        if line.startswith("Found "):
            return int(line.split()[1])
    return 0


def list_of_targets(tree: Path) -> list[str]:
    """All python files under the lint targets, relative to ``tree``."""
    return [str(p.relative_to(tree)) for d in TARGETS for p in sorted((tree / d).rglob("*.py"))]


def unformatted(tree: Path) -> int:
    result = ruff(["format", "--check", "--isolated", "--line-length", LINE_LENGTH, *TARGETS], tree)
    return sum(1 for line in result.stdout.splitlines() if line.startswith("Would reformat"))


def changed_python_files(before_ref: str) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "diff", "--name-only", before_ref, "--", *TARGETS],
        capture_output=True,
        text=True,
        check=True,
    )
    return [p for p in result.stdout.split() if p.endswith(".py")]


def extract(ref: str) -> Path:
    """Materialise a git tree in a temp dir so ruff can run against it."""
    tmp = Path(tempfile.mkdtemp(prefix="lint-gate-before-"))
    archive = subprocess.run(
        ["git", "-C", str(REPO_ROOT), "archive", "--format=tar", ref], capture_output=True, check=True
    )
    tar_path = tmp / "tree.tar"
    tar_path.write_bytes(archive.stdout)
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp, filter="data")
    tar_path.unlink()
    return tmp


def plot(before: dict[str, int], after: dict[str, int], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rules = [r for r in sorted(set(before) | set(after), key=lambda r: -before.get(r, 0))]
    rules = [r for r in rules if before.get(r, 0) or after.get(r, 0)]
    xs = range(len(rules))
    b = [before.get(r, 0) for r in rules]
    a = [after.get(r, 0) for r in rules]
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    ax.bar([x - 0.2 for x in xs], b, width=0.4, label="before (pre-fix tree)", color="#b03030")
    ax.bar([x + 0.2 for x in xs], a, width=0.4, label="after (this change)", color="#2b7a3d")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(rules, rotation=45, ha="right")
    ax.set_ylabel("findings (symlog scale)")
    ax.set_title(
        "TensorFVM ruff backlog by rule family (src+tests+scripts, default rule set)\n"
        "enforced gate = {E9,F63,F7,F82} -> 0 before and after; new code held to E9+F"
    )
    for x, (bv, av) in enumerate(zip(b, a)):
        ax.annotate(str(bv), (x - 0.2, bv), ha="center", va="bottom", fontsize=7)
        ax.annotate(str(av), (x + 0.2, av), ha="center", va="bottom", fontsize=7)
    ax.legend(loc="upper right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--before", default=None, help="ref to audit as the pre-fix tree")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "docs/ci-lint-gate/lint_gate_audit.json")
    parser.add_argument("--figure", type=Path, default=REPO_ROOT / "docs/ci-lint-gate/lint_gate_backlog.png")
    parser.add_argument("--no-figure", action="store_true")
    args = parser.parse_args(argv)

    before_ref = (
        args.before
        or subprocess.run(
            ["git", "-C", str(REPO_ROOT), "merge-base", "HEAD", "origin/main"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    touched = changed_python_files(before_ref)

    tmp = extract(before_ref)
    try:
        before = statistics(tmp)
        before_touched = findings(touched, tmp, RATCHET_SELECT)
        before_enforced = findings(list_of_targets(tmp), tmp, ENFORCED_SELECT)
        before_fmt = unformatted(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    after = statistics(REPO_ROOT)
    after_touched = findings(touched, REPO_ROOT, RATCHET_SELECT)
    after_enforced = findings(list_of_targets(REPO_ROOT), REPO_ROOT, ENFORCED_SELECT)
    after_fmt = unformatted(REPO_ROOT)

    audit = {
        "before_ref": before_ref,
        "targets": list(TARGETS),
        "full_backlog_default_rules": {
            "before_total": sum(before.values()),
            "after_total": sum(after.values()),
            "before_by_rule": before,
            "after_by_rule": after,
        },
        "enforced_gate": {
            "rule_set": ENFORCED_SELECT,
            "before_findings": before_enforced,
            "after_findings": after_enforced,
        },
        "changed_files": {
            "rule_set": RATCHET_SELECT,
            "files": touched,
            "before_findings": before_touched,
            "after_findings": after_touched,
        },
        "format_unformatted_files": {
            "line_length": int(LINE_LENGTH),
            "before": before_fmt,
            "after": after_fmt,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if not args.no_figure:
        plot(before, after, args.figure)
        print(f"figure -> {args.figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
