"""Collect the quantified evidence for the CI lint-gate fix (mission m4241).

Everything the report quotes is produced here from the repository itself:
no number is typed by hand.  The script

  1. measures the pre-existing ruff backlog on PR #103's head commit
     (``908ad91``) by exporting only ``src/``, ``tests/`` and ``scripts/``
     into a scratch directory and running ruff there with ``--isolated``
     (the settings CI effectively uses -- the project declares no
     ``[tool.ruff]`` section);
  2. measures the same quantities on the current worktree, i.e. after the fix;
  3. renders ``figures/lint_backlog.png`` (rule histogram + per-file
     distribution) and ``results.json``;
  4. injects the result table into ``report.md`` between the markers
     ``<!--RESULTS-->`` and ``<!--/RESULTS-->``.
"""

from __future__ import annotations

import json
import subprocess as sp
import sys
import tarfile
import tempfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
DOC = HERE.parent
REPO = HERE.parents[2]  # <repo>/docs/m4241/artifacts -> <repo>
BASE_COMMIT = "908ad91"  # PR #103 head (docs-only deliverable)
DIRS = ("src", "tests", "scripts")


def ruff_json(cwd: Path) -> tuple[list[dict], int]:
    proc = sp.run(
        ["ruff", "check", "--isolated", "--no-cache", "--output-format=json", *DIRS],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout or "[]"), (0 if proc.returncode == 0 else 1)


def ruff_format_check(cwd: Path) -> tuple[int, int, int]:
    proc = sp.run(
        ["ruff", "format", "--isolated", "--check", *DIRS],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    unformatted = sum(
        1 for line in proc.stdout.splitlines() if line.startswith("Would reformat")
    )
    already = 0
    for line in proc.stdout.splitlines():
        if "file" in line and "already formatted" in line:
            already = int(line.split()[0])
    total = sum(1 for d in DIRS for _ in (cwd / d).rglob("*.py"))
    return (0 if proc.returncode == 0 else 1, unformatted, total or already)


def measure(cwd: Path) -> dict:
    findings, check_exit = ruff_json(cwd)
    per_rule = Counter(item.get("code") or "?" for item in findings)
    per_file = Counter(item["filename"] for item in findings)
    fmt_rc, unformatted, total_files = ruff_format_check(cwd)
    return {
        "root": str(cwd),
        "ruff_check_exit": check_exit,
        "findings_total": len(findings),
        "per_rule": dict(per_rule.most_common()),
        "files_total": total_files,
        "files_with_findings": len(per_file),
        "files_clean": total_files - len(per_file),
        "per_file_counts": sorted(per_file.values(), reverse=True),
        "ruff_format_exit": fmt_rc,
        "files_unformatted": unformatted,
    }


def baseline_measure() -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="fvm-base-"))
    archive = tmp / "base.tar"
    with archive.open("wb") as fh:
        sp.run(
            ["git", "archive", "--format=tar", BASE_COMMIT, *DIRS],
            cwd=REPO,
            stdout=fh,
            check=True,
        )
    with tarfile.open(archive) as tar:
        tar.extractall(tmp / "tree")
    return measure(tmp / "tree")


def render_png(base: dict, after: dict, dest: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.5, 4.6), dpi=140)

    rules = list(base["per_rule"].items())[:10]
    codes = [r[0] for r in rules][::-1]
    counts = [r[1] for r in rules][::-1]
    ax0.barh(codes, counts, color="#b23b3b")
    ax0.set_xscale("log")
    ax0.set_xlabel("findings (log scale)")
    ax0.set_title(f"PR #103 head {BASE_COMMIT}: {base['findings_total']} ruff findings")
    for y, value in enumerate(counts):
        ax0.text(value * 1.08, y, str(value), va="center", fontsize=8)
    ax0.set_xlim(right=max(counts) * 3)

    ax1.hist(
        base["per_file_counts"],
        bins=40,
        color="#3b6bb2",
        edgecolor="white",
        log=True,
    )
    ax1.set_xlabel("findings in one file")
    ax1.set_ylabel("files (log scale)")
    ax1.set_title(
        "per-file backlog: %d of %d files carry findings, %d are clean"
        % (
            base["files_with_findings"],
            base["files_total"],
            base["files_clean"],
        )
    )
    ax1.axvline(0, color="#2f7d32", lw=1.2, linestyle="--")
    ax1.annotate(
        "whole-tree gate requires\nall of them at zero",
        xy=(0, 1),
        xytext=(base["per_file_counts"][0] * 0.25, 3.0),
        fontsize=8,
        color="#2f7d32",
        arrowprops=dict(arrowstyle="->", color="#2f7d32", lw=0.8),
    )
    fig.suptitle(
        "Ruff backlog that made the advisory lint job red by construction "
        "(fix: judge the delta of a change set)"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest)
    plt.close(fig)


def table(base: dict, after: dict, gate_cases: dict) -> str:
    rows = [
        (
            "ruff findings, whole tree (src/ tests/ scripts/)",
            base["findings_total"],
            after["findings_total"],
        ),
        ("files under src/ tests/ scripts/", base["files_total"], after["files_total"]),
        (
            "files carrying >= 1 finding",
            base["files_with_findings"],
            after["files_with_findings"],
        ),
        (
            "files that `ruff format --check` would rewrite",
            base["files_unformatted"],
            after["files_unformatted"],
        ),
        (
            "`ruff check src tests scripts` exit code",
            base["ruff_check_exit"],
            after["ruff_check_exit"],
        ),
        (
            "`ruff format --check src tests scripts` exit code",
            base["ruff_format_exit"],
            after["ruff_format_exit"],
        ),
        (
            "gated job: lint on PR #103's change set (exit code)",
            "<job absent>",
            gate_cases["docs_only"],
        ),
        (
            "gated job: control, new file with an unused import (exit code)",
            "-",
            gate_cases["new_dirty"],
        ),
        (
            "gated job: control, legacy file edited with no new finding (exit code)",
            "-",
            gate_cases["legacy_ok"],
        ),
        (
            "gated job: control, legacy file gaining one finding (exit code)",
            "-",
            gate_cases["legacy_debt"],
        ),
    ]
    out = [
        "| quantity | before fix (whole-tree gate) | after fix (delta gate) |",
        "|---|---|---|",
    ]
    for label, before, now in rows:
        b = "-" if before is None else f"{before}"
        out.append(f"| {label} | {b} | {now} |")
    return "\n".join(out)


def _sh(cwd: Path, *args: str, env: dict | None = None) -> sp.CompletedProcess:
    return sp.run(args, cwd=cwd, capture_output=True, text=True, env=env)


def _commit(cwd: Path, message: str) -> str:
    _sh(cwd, "git", "add", "-A")
    _sh(cwd, "git", "commit", "-q", "-m", message)
    return _sh(cwd, "git", "rev-parse", "HEAD").stdout.strip()


def gate_scenarios() -> tuple[dict, dict]:
    """Run the gate on PR #103's change set and on the controls, for real.

    The scenarios are replayed in a throw-away worktree pinned to PR #103's
    head commit, with the gate scripts of this branch dropped in, so the exit
    codes in the report are measured, not asserted.
    """
    wt = Path(tempfile.mkdtemp(prefix="fvm-gate-")) / "head"
    wt.parent.mkdir(parents=True, exist_ok=True)
    sp.run(
        ["git", "worktree", "add", "--detach", str(wt), BASE_COMMIT],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    cases: dict[str, int] = {}
    logs: dict[str, str] = {}
    try:
        dest = wt / "scripts" / "ci"
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("lint_changed.sh", "ruff_ratchet.py"):
            (dest / name).write_bytes((REPO / "scripts" / "ci" / name).read_bytes())
        base0 = (
            _sh(wt, "git", "merge-base", "origin/main", BASE_COMMIT).stdout.strip()
            or BASE_COMMIT
        )
        _commit(wt, "probe: gate scripts")

        def run(label: str, base: str) -> int:
            env = dict(**__import__("os").environ, BASE_SHA=base)
            env.pop("PUSH_BEFORE", None)
            proc = _sh(wt, "bash", "scripts/ci/lint_changed.sh", env=env)
            cases[label] = proc.returncode
            logs[label] = (proc.stdout + proc.stderr).strip()
            return proc.returncode

        run("docs_only", base0)  # PR #103's own change set (docs) -> must be 0

        prev = wt and _commit(wt, "probe: noop")
        (wt / "tests" / "test_wall_traction.py").write_text(
            (wt / "tests" / "test_wall_traction.py").read_text()
            + "\n# probe: comment only\n"
        )
        _commit(wt, "probe: comment-only edit of a legacy file")
        run("legacy_ok", prev)

        prev = _commit(wt, "probe: noop 2")
        (wt / "tests" / "test_wall_traction.py").write_text(
            (wt / "tests" / "test_wall_traction.py").read_text() + "import sys\n"
        )
        _commit(wt, "probe: legacy file gains one finding")
        run("legacy_debt", prev)

        prev = _commit(wt, "probe: noop 3")
        (wt / "tests" / "probe_new_module.py").write_text("import os\n\n\nVALUE = 1\n")
        _commit(wt, "probe: new unclean module")
        run("new_dirty", prev)
    finally:
        sp.run(
            ["git", "worktree", "remove", "--force", str(wt)],
            cwd=REPO,
            capture_output=True,
        )
    return {
        "exit_codes": cases,
        "logs": logs,
        "expected": {"docs_only": 0, "legacy_ok": 0, "legacy_debt": 1, "new_dirty": 1},
    }, cases


def main() -> int:
    base = baseline_measure()
    after = measure(REPO)
    png = HERE / "figures" / "lint_backlog.png"
    render_png(base, after, png)

    evidence, gate_cases = gate_scenarios()
    (HERE / "results.json").write_text(
        json.dumps(
            {"baseline": base, "after_fix": after, "gate_cases": evidence}, indent=2
        )
        + "\n"
    )

    report = DOC / "report.md"
    text = report.read_text()
    start, end = "<!--RESULTS-->", "<!--/RESULTS-->"
    body = table(base, after, gate_cases)
    assert start in text and end in text, "report.md is missing the results markers"
    report.write_text(
        text.split(start)[0] + start + "\n" + body + "\n" + end + text.split(end)[1]
    )
    print(body)
    return 0


if __name__ == "__main__":
    sys.exit(main())
