"""Contract guard for the lint job in ``.github/workflows/ci.yml``.

Background (PR #119): the lint job carried a *job-level*
``continue-on-error: true``.  On GitHub that keeps the workflow run green but
leaves the job's check-run conclusion at ``failure``, so every pull request
reported a red ``lint (advisory)`` check and the advisory intent documented in
the workflow had no effect on the recorded signal.  The tree carries a large
pre-existing ruff backlog (thousands of E701/E702 semicolon-packed statements
plus ~550 files off the ``ruff format`` style), so a whole-tree gate cannot go
green today.

The structure these tests pin down:

* the whole-tree ``ruff`` steps are advisory through a *step-level*
  ``continue-on-error`` (a job-level one must not come back);
* a blocking *ratchet* step lints exactly the ``.py`` files the pull request
  touches, so no new lint debt can enter the tree;
* the checkout fetches enough history for the ratchet to diff against the base
  branch.

The assertions are plain-text scans rather than a YAML parse: PyYAML is not a
dependency of the ``dev`` extra, and this guard has to run in the CI image
without it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

CI_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"


def _lint_job_block() -> list[str]:
    """Return the ``lint:`` job of the workflow as a list of raw lines."""
    lines = CI_WORKFLOW.read_text(encoding="utf-8").splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.rstrip() == "  lint:":
            start = i
            break
    assert start is not None, "no `  lint:` job found in ci.yml"
    block = [lines[start]]
    for line in lines[start + 1 :]:
        # A job ends at the next two-space-indented top-level job key.
        if line.strip() and not line.startswith("    ") and line.startswith("  "):
            break
        block.append(line)
    return block


@pytest.fixture(scope="module")
def lint_job() -> list[str]:
    assert CI_WORKFLOW.is_file(), f"missing workflow: {CI_WORKFLOW}"
    return _lint_job_block()


def test_lint_job_is_not_neutralised_at_job_level(lint_job: list[str]) -> None:
    """`continue-on-error` must never sit at the four-space (job) indent."""
    offenders = [
        line
        for line in lint_job
        if line.startswith("    continue-on-error")
        and line.split(":")[1].strip() == "true"
    ]
    assert not offenders, (
        "job-level continue-on-error keeps the check-run conclusion at "
        f"'failure' (PR #119); move it onto the steps: {offenders}"
    )


def test_whole_tree_ruff_steps_stay_advisory_per_step(lint_job: list[str]) -> None:
    """Both whole-tree ruff steps must carry their own continue-on-error."""
    steps = _steps(lint_job)
    for name in (
        "Lint (ruff, whole tree, advisory)",
        "Format check (ruff format, whole tree, advisory)",
    ):
        step = steps.get(name)
        assert step is not None, f"missing advisory step: {name}"
        assert any("continue-on-error: true" in line for line in step), (
            f"{name} would fail the job on the pre-existing backlog"
        )


def test_changed_file_ratchet_is_blocking(lint_job: list[str]) -> None:
    """The ratchet step must run both ruff modes on the changed files, ungated."""
    steps = _steps(lint_job)
    ratchet = steps.get("Lint changed files (ruff, ratchet gate)")
    assert ratchet is not None, "missing the changed-file ratchet step"
    body = "\n".join(ratchet)
    assert "continue-on-error" not in body, (
        "the ratchet is the gate, it cannot be advisory"
    )
    assert "--diff-filter=ACMR" in body, "ratchet must ignore deleted files"
    assert "*.py" in body, "ratchet must be limited to Python sources"
    assert 'ruff check "${files[@]}"' in body
    assert 'ruff format --check "${files[@]}"' in body


def test_ratchet_has_history_to_diff_against(lint_job: list[str]) -> None:
    """`fetch-depth: 0` is what makes `origin/<base>` resolvable in CI."""
    assert any("fetch-depth: 0" in line for line in lint_job), (
        "the ratchet diffs against the base branch, so the base ref must be fetched"
    )


def _steps(lint_job: list[str]) -> dict[str, list[str]]:
    """Split the job body into ``{step name: lines}`` (unnamed steps dropped)."""
    steps: dict[str, list[str]] = {}
    current: list[str] | None = None
    name: str | None = None
    for line in lint_job:
        stripped = line.strip()
        if stripped.startswith("- name:") or stripped.startswith("- uses:"):
            if name is not None and current is not None:
                steps[name] = current
            name = (
                stripped.split(":", 1)[1].strip()
                if stripped.startswith("- name:")
                else None
            )
            current = [line]
            continue
        if current is not None:
            current.append(line)
    if name is not None and current is not None:
        steps[name] = current
    return steps
