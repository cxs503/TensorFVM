"""Guards the CI lint job's advisory semantics.

Regression context (TensorFVM PR #109): the lint job carried
``continue-on-error: true`` at *job* level.  GitHub keeps the workflow run
green in that case, but the job is still published as a **failed check run**,
so every pull request -- including purely documentary ones -- reported
``lint (advisory) = failure`` and serialised merge gates went red over a
pre-existing ruff backlog it had not created.

Step-level ``continue-on-error`` is what actually expresses "advisory": the
step records ``outcome=failure`` (red step, annotation, counts in the run
summary) while the job/check-run conclusion stays ``success``.

These tests parse ``.github/workflows/ci.yml`` textually so they need no YAML
dependency beyond the optional one used when it happens to be installed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

CI_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci.yml"

# Steps that execute ruff and therefore must never fail the lint job.
RUFF_STEP_NAMES = ("Lint (ruff)", "Format check (ruff format)")


def _job_block(name: str) -> list[str]:
    """Return the raw lines of ``jobs.<name>`` (comments preserved)."""
    lines = CI_PATH.read_text(encoding="utf-8").splitlines()
    header = f"  {name}:"
    for index, line in enumerate(lines):
        if line.rstrip() == header:
            start = index + 1
            break
    else:  # pragma: no cover - defensive
        raise AssertionError(f"job {name!r} not found in {CI_PATH}")
    block: list[str] = []
    for line in lines[start:]:
        # A non-empty line indented <= 2 columns closes the job block.
        if line.strip() and len(line) - len(line.lstrip()) <= 2:
            break
        block.append(line)
    return block


def _job_level_keys(block: list[str]) -> list[str]:
    """Keys that belong to the job mapping itself (4-space indented)."""
    keys = []
    for line in block:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line) - len(line.lstrip()) == 4:
            match = re.match(r"^ {4}([A-Za-z0-9_.-]+):", line)
            if match:
                keys.append(match.group(1))
    return keys


def _steps(block: list[str]) -> list[list[str]]:
    """Split a job block into its ``steps`` entries (list of raw line lists)."""
    steps: list[list[str]] = []
    current: list[str] | None = None
    for line in block:
        stripped = line.lstrip()
        if len(line) - len(stripped) == 6 and stripped.startswith("- "):
            current = [line]
            steps.append(current)
        elif current is not None:
            current.append(line)
    return steps


def _step_name(step: list[str]) -> str | None:
    for line in step:
        match = re.match(r"^ {6}- name:\s*(.+)$", line) or re.match(
            r"^ {8}name:\s*(.+)$", line
        )
        if match:
            return match.group(1).strip()
    return None


def _step_field(step: list[str], key: str) -> str | None:
    """Value of a step-level key (8-space indented under the step item)."""
    pattern = re.compile(r"^ {8}" + re.escape(key) + r":\s*(.*)$")
    for line in step:
        match = pattern.match(line)
        if match:
            return match.group(1).strip()
    return None


def test_lint_job_has_no_job_level_continue_on_error():
    """A job-level flag would turn the advisory job into a failed check run."""
    block = _job_block("lint")
    assert "continue-on-error" not in _job_level_keys(block), (
        "lint job must express advisory semantics per step, not per job: "
        "job-level continue-on-error still publishes a failed check run"
    )


def _lint_job_name() -> str:
    block = _job_block("lint")
    for line in block:
        match = re.match(r"^ {4}name:\s*(.+)$", line)
        if match:
            return match.group(1).strip()
    return ""


def test_ruff_steps_match_the_declared_mode():
    """While the job is named "(advisory)" its ruff steps must be non-blocking.

    Promotion to a gate (job renamed, flags removed) is allowed and expected;
    what must never happen is an advisory-looking job that still fails its
    check run, or a gate whose ruff steps silently ignore failures.
    """
    advisory = "advisory" in _lint_job_name()
    steps = _steps(_job_block("lint"))
    names = [_step_name(step) for step in steps]
    for expected in RUFF_STEP_NAMES:
        assert expected in names, f"missing lint step {expected!r}; found {names}"
    for step in steps:
        body = "\n".join(step)
        if "ruff check" not in body and "ruff format" not in body:
            continue
        flag = _step_field(step, "continue-on-error")
        if advisory:
            assert flag == "true", (
                f"step {_step_name(step)!r} runs ruff but is blocking; the lint "
                "job is declared advisory and must not gate on the backlog"
            )
        else:
            assert flag in (None, "false"), (
                f"step {_step_name(step)!r} runs ruff but is non-blocking; a "
                "promoted lint job must fail on findings"
            )


def test_backlog_stays_measurable_while_advisory():
    """Advisory must not degrade into invisible: counts go to the run summary."""
    if "advisory" not in _lint_job_name():
        pytest.skip("job promoted to a gate; the finding is the failure itself")
    block = _job_block("lint")
    summary_steps = [s for s in _steps(block) if "GITHUB_STEP_SUMMARY" in "\n".join(s)]
    assert summary_steps, "lint job must publish backlog counts into the run summary"


def test_test_job_gate_steps_remain_blocking():
    """The advisory relaxation must not leak into the test gate."""
    steps = _steps(_job_block("test"))
    names = [_step_name(step) for step in steps]
    assert "Run tests" in names
    for step in steps:
        if _step_name(step) in ("Run tests", "Install package with dev extras"):
            assert _step_field(step, "continue-on-error") in (None, "false")


def test_workflow_parses_and_lint_job_is_advisory():
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(CI_PATH.read_text(encoding="utf-8"))
    lint = data["jobs"]["lint"]
    assert "continue-on-error" not in lint
    rent = "advisory" in str(lint.get("name", ""))
    ruff_steps = [s for s in lint["steps"] if "ruff" in str(s.get("run", ""))]
    assert ruff_steps, "no ruff steps found in the lint job"
    if rent:
        assert all(step.get("continue-on-error") is True for step in ruff_steps)
    assert "continue-on-error" not in data["jobs"]["test"]
