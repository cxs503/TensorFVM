#!/usr/bin/env python3
"""Measure the ruff backlog that keeps the CI lint job advisory.

The lint job in ``.github/workflows/ci.yml`` is deliberately non-blocking while
the tree still carries a large pre-existing ruff backlog.  That only stays
honest if the debt is *measurable*: this script produces the numbers quoted in
``docs/ci-lint-advisory/README.md`` and the run-summary snapshot step mirrors
them, so "advisory" cannot quietly become "invisible".

Usage::

    python scripts/audit_ruff_backlog.py                 # markdown table
    python scripts/audit_ruff_backlog.py --png fig.png   # + bar chart
    python scripts/audit_ruff_backlog.py --json out.json # + machine record
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

TARGETS = ["src", "tests", "scripts"]

# One-line description per rule family, used in the report table.
RULE_NOTES = {
    "E701": "statement after colon (packed one-liners)",
    "E702": "statement after semicolon (packed one-liners)",
    "E703": "useless trailing semicolon",
    "E401": "multiple imports on one line",
    "E402": "module import not at top of file",
    "E731": "lambda assigned to a name",
    "E741": "ambiguous variable name (l, I, O)",
    "F401": "imported but unused",
    "F541": "f-string without placeholders",
    "F841": "local variable assigned but unused",
}

# Rules ruff can fix automatically; the rest need judgement.
AUTOFIX_CODES = {"E401", "E703", "F401", "F541"}


def _ruff() -> str:
    exe = shutil.which("ruff")
    if exe is None:  # pragma: no cover - environment guard
        raise SystemExit("ruff not found on PATH (pip install '.[dev]')")
    return exe


def check_counts() -> tuple[collections.Counter, int]:
    """Per-rule finding counts for ``ruff check`` plus the total."""
    proc = subprocess.run(
        [_ruff(), "check", *TARGETS, "--output-format", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    # ruff exits 1 when it finds problems; anything else is a real failure.
    if proc.returncode not in (0, 1):
        raise SystemExit(f"ruff check failed: {proc.stderr.strip()[:400]}")
    findings = json.loads(proc.stdout or "[]")
    counts: collections.Counter = collections.Counter()
    for item in findings:
        counts[item.get("code") or "unknown"] += 1
    return counts, len(findings)


def format_counts() -> tuple[int, int]:
    """(files ruff format would rewrite, total python files scanned)."""
    proc = subprocess.run(
        [_ruff(), "format", "--check", *TARGETS],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode not in (0, 1):
        raise SystemExit(f"ruff format failed: {proc.stderr.strip()[:400]}")
    lines = proc.stdout.splitlines()
    would = sum(1 for line in lines if line.startswith("Would reformat:"))
    # The trailing summary is authoritative for the untouched files.
    tail = re.search(r"(\d+) files? already formatted", proc.stdout)
    already = int(tail.group(1)) if tail else 0
    return would, would + already


def markdown(counts: collections.Counter, total: int, would: int, scanned: int) -> str:
    rows = [
        "| rule | findings | auto-fix | meaning |",
        "|---|---:|---|---|",
    ]
    for code, count in counts.most_common():
        fixable = "yes" if code in AUTOFIX_CODES else "no"
        rows.append(f"| `{code}` | {count} | {fixable} | {RULE_NOTES.get(code, '-')} |")
    rows.append(f"| **total** | **{total}** | | |")
    rows.append("")
    rows.append(
        f"`ruff format --check {' '.join(TARGETS)}`: "
        f"**{would}** of {scanned} Python files would be rewritten."
    )
    return "\n".join(rows)


def figure(counts: collections.Counter, total: int, would: int, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ordered = counts.most_common()
    labels = [code for code, _ in ordered]
    values = [count for _, count in ordered]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    bars = ax.barh(labels[::-1], values[::-1], color="#3b6fa0")
    for bar, value in zip(bars, values[::-1], strict=True):
        ax.text(
            bar.get_width() * 1.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value}",
            va="center",
            fontsize=8,
        )
    ax.set_xscale("log")
    ax.set_xlabel("findings (log scale)")
    ax.set_title(
        "ruff check backlog on src/tests/scripts\n"
        f"total {total} findings; ruff format would rewrite {would} files",
        fontsize=10,
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--png", type=Path, help="write a bar chart of the backlog")
    parser.add_argument("--json", type=Path, help="write a machine-readable record")
    args = parser.parse_args(argv)

    counts, total = check_counts()
    would, scanned = format_counts()
    print(markdown(counts, total, would, scanned))

    if args.png:
        args.png.parent.mkdir(parents=True, exist_ok=True)
        figure(counts, total, would, args.png)
        print(f"\nwrote {args.png}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "targets": TARGETS,
                    "total_findings": total,
                    "by_rule": dict(counts.most_common()),
                    "format_would_rewrite": would,
                    "format_files_scanned": scanned,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
