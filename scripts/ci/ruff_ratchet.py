"""Delta ruff ratchet used by CI (see scripts/ci/lint_changed.py for the driver).

The repository carries a large pre-existing ruff backlog, so a whole-tree
`ruff check src tests scripts` is red by construction -- it fails even for a
docs-only pull request such as PR #103, which is how the lint signal got lost.

A strict "changed files must be clean" ratchet would fix that specific failure
but freeze the repository just as hard: 554 of 556 Python files are unformatted
and thousands of findings predate the workflow, so *any* pull request that
touches existing code would go red.  This module therefore judges the delta
instead of the absolute state:

  * findings  -- for each file in the change set, compare the ruff findings of
                 the base revision against the working copy; fail only when a
                 rule fires *more often* than it did at base (a new file is
                 compared against an empty baseline, so new files must be
                 clean);
  * formatting -- a file must be ruff-format clean when its base revision was
                 clean; files that were already unformatted keep their historic
                 debt (reformatting them is an unrelated, review-hostile diff)
                 but are reported as a notice.

Both sides are measured with the same ruff invocation (``--isolated
--no-cache``), so the comparison cannot drift with a local ruff configuration
and the cache cannot mask a finding.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

RUFF = shutil.which("ruff") or "ruff"


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def findings(path: Path) -> Counter:
    """Rule-code histogram of the ruff findings for ``path``."""
    proc = _run(
        [
            RUFF,
            "check",
            "--isolated",
            "--no-cache",
            "--output-format=json",
            "--",
            str(path),
        ]
    )
    if proc.returncode not in (0, 1):
        raise SystemExit(f"ruff check failed to run on {path}: {proc.stderr.strip()}")
    try:
        payload = json.loads(proc.stdout or "[]")
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise SystemExit(f"unparsable ruff output for {path}: {exc}") from exc
    return Counter((item.get("code") or "?") for item in payload)


def is_formatted(path: Path) -> bool:
    return (
        _run([RUFF, "format", "--isolated", "--check", "--", str(path)]).returncode == 0
    )


def base_copy(repo: Path, base: str, rel: str, workdir: Path) -> Path | None:
    """Materialise the base revision of ``rel`` so both sides can be linted."""
    proc = subprocess.run(
        ["git", "show", f"{base}:{rel}"], cwd=repo, capture_output=True
    )
    if proc.returncode != 0:
        return None  # file did not exist at base (added by this change set)
    target = workdir / rel.replace("/", "__")
    target.write_bytes(proc.stdout)
    return target


def judge(repo: Path, base: str, files: list[str]) -> int:
    workdir = Path(tempfile.mkdtemp(prefix="ruff-ratchet-"))
    regressions: list[str] = []
    notices: list[str] = []
    try:
        for rel in files:
            head = repo / rel
            if not head.is_file():
                continue
            before = base_copy(repo, base, rel, workdir)
            head_findings = findings(head)
            base_findings = findings(before) if before is not None else Counter()
            new_rules = {
                code: head_findings[code] - base_findings.get(code, 0)
                for code in head_findings
                if head_findings[code] > base_findings.get(code, 0)
            }
            red = bool(new_rules)
            if new_rules:
                detail = ", ".join(
                    f"{code} +{n}" for code, n in sorted(new_rules.items())
                )
                regressions.append(f"{rel}: new ruff findings ({detail})")
            elif head_findings:
                notices.append(
                    f"{rel}: {sum(head_findings.values())} pre-existing finding(s), unchanged"
                )

            # Formatting is gated only where it can be enforced: a file whose
            # base revision was already unformatted keeps its historic debt
            # (reformatting it here would be an unrelated, review-hostile diff).
            head_formatted = is_formatted(head)
            if before is not None and not is_formatted(before):
                if not head_formatted:
                    notices.append(f"{rel}: pre-existing format debt, not gated")
            elif not head_formatted:
                regressions.append(
                    f"{rel}: `ruff format --check` fails on a file that was format-clean at base"
                )
                red = True

            print(f"  {'red' if red else 'ok':<4} {rel}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    for notice in notices:
        print(f"notice: {notice}")
    if regressions:
        print("\nnew lint debt introduced by this change set:", file=sys.stderr)
        for item in regressions:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("ruff ratchet: no new lint debt")
    return 0


def main(argv: list[str]) -> int:
    # argv: <base-revision> <relative-path>...
    if len(argv) < 2:
        print("usage: ruff_ratchet.py <base-revision> <path>...", file=sys.stderr)
        return 2
    repo = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True
        ).stdout.strip()
        or "."
    )
    return judge(repo, argv[0], argv[1:])


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
