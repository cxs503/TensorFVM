"""Contract tests for the ruff lint gate: pin, scope, and the changed-files ratchet.

The gate is two steps (see ``docs/ci-lint-policy.md``): a whole-tree check of the
rules whose findings are real defects, plus ``scripts/ci_lint_ratchet.py`` which
lints/formats only the Python files a change touches.  These tests pin the
contract so a future edit cannot silently drop either half, re-float the ruff
pin, or turn the ratchet into a no-op.
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
POLICY_DOC = ROOT / "docs" / "ci-lint-policy.md"
RATCHET = ROOT / "scripts" / "ci_lint_ratchet.py"
PIN_RE = re.compile(r"ruff==(\d+\.\d+\.\d+)")


def _load_ratchet():
    spec = importlib.util.spec_from_file_location("ci_lint_ratchet", RATCHET)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_lint_ratchet"] = module
    spec.loader.exec_module(module)
    return module


ratchet = _load_ratchet()


def _pyproject() -> dict:
    with open(ROOT / "pyproject.toml", "rb") as fh:
        return tomllib.load(fh)


def _dev_extra() -> list[str]:
    return _pyproject()["project"]["optional-dependencies"]["dev"]


def _pinned_ruff() -> str:
    """The exact ruff version the whole gate is calibrated against."""
    pins = [m.group(1) for req in _dev_extra() if (m := PIN_RE.fullmatch(req.strip()))]
    assert len(pins) == 1, f"dev extra must pin ruff exactly once, got {_dev_extra()}"
    return pins[0]


def _ci_text() -> str:
    return CI_WORKFLOW.read_text()


def test_dev_extra_pins_ruff_to_an_exact_release():
    # A floating `ruff>=0.6` is what made this gate flap: 0.17.0 changed
    # formatter line-length accounting, so an identical tree passed one release
    # and failed the next.  The pin must stay exact.
    assert PIN_RE.fullmatch(next(r for r in _dev_extra() if r.startswith("ruff")))


def test_ci_installs_the_same_pin_as_pyproject():
    pins = set(PIN_RE.findall(_ci_text()))
    assert pins == {_pinned_ruff()}, (
        f"ci.yml pins {pins}, pyproject pins {_pinned_ruff()}"
    )


def test_ci_lint_job_checks_fatal_rules_whole_tree():
    # Syntax errors / undefined names / broken comparisons must not be dodgeable
    # by only linting touched files.
    assert "ruff check --select E9,F63,F7,F82 src tests scripts" in _ci_text()


def test_ci_lint_job_runs_the_changed_files_ratchet():
    assert "scripts/ci_lint_ratchet.py --base origin/main" in _ci_text()
    # The ratchet diffs against origin/main, so the checkout needs the merge base.
    assert "fetch-depth: 0" in _ci_text()


def test_policy_doc_records_the_pin_and_the_ratchet():
    text = POLICY_DOC.read_text()
    assert _pinned_ruff() in text
    assert "ci_lint_ratchet.py" in text


def test_pin_string_is_not_shadowed_by_a_second_install():
    # The lint job must install the pin *after* `pip install -e .[dev]`, i.e. no
    # later step may re-install a floating ruff.
    text = _ci_text()
    dev = text.find('pip install -e ".[dev]"')
    pin = text.find(f'pip install "ruff=={_pinned_ruff()}"')
    assert dev != -1 and pin != -1 and pin > dev


def test_ratchet_skips_evidence_and_non_python_paths():
    assert ratchet.is_gated("src/tensorfvm/solver.py")
    assert ratchet.is_gated("tests/test_x.py")
    assert not ratchet.is_gated("docs/m4253/reproduce.py")
    assert not ratchet.is_gated("artifacts/run.py")
    assert not ratchet.is_gated("runs/study.py")
    assert not ratchet.is_gated("README.rst")
    assert ratchet.select_gated(["src/a.py", "docs/b.py", "notes.md"]) == ["src/a.py"]


def test_ratchet_reports_a_missing_base_ref(tmp_path):
    assert ratchet.main(["--base", "no-such-ref", "--root", str(tmp_path)]) == 2


# --- integration: a throwaway repository exercising the real git + ruff path ---

requires_git_and_ruff = pytest.mark.skipif(
    shutil.which("git") is None or shutil.which("ruff") is None,
    reason="needs git and ruff on PATH",
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _repo_with_branch(tmp_path: Path, changed: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "base.py").write_text('"""Base module."""\n')
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    _git(repo, "checkout", "-qb", "feature")
    for name, body in changed.items():
        target = repo / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "feature work")
    return repo


CLEAN = '"""Sample module."""\n\n\ndef add(a, b):\n    return a + b\n'
UNUSED_IMPORT = '"""Sample module."""\n\nimport os\n'


@requires_git_and_ruff
def test_ratchet_selects_only_gated_changes(tmp_path):
    repo = _repo_with_branch(
        tmp_path, {"src/pkg/mod.py": CLEAN, "docs/m1/repro.py": "x = 1\n"}
    )
    assert ratchet.changed_python_files("main", repo) == ["src/pkg/mod.py"]


@requires_git_and_ruff
def test_ratchet_passes_on_clean_changes(tmp_path):
    repo = _repo_with_branch(tmp_path, {"src/pkg/mod.py": CLEAN})
    assert ratchet.main(["--base", "main", "--root", str(repo)]) == 0


@requires_git_and_ruff
def test_ratchet_fails_on_a_new_lint_finding(tmp_path):
    repo = _repo_with_branch(tmp_path, {"src/pkg/mod.py": UNUSED_IMPORT})
    assert ratchet.main(["--base", "main", "--root", str(repo)]) == 1


@requires_git_and_ruff
def test_ratchet_fails_on_unformatted_changes(tmp_path):
    unformatted = '"""Sample module."""\n\n\ndef add( a,b ):\n    return a+b\n'
    repo = _repo_with_branch(tmp_path, {"src/pkg/mod.py": unformatted})
    assert ratchet.main(["--base", "main", "--root", str(repo)]) == 1


@requires_git_and_ruff
def test_ratchet_passes_when_nothing_relevant_changed(tmp_path):
    repo = _repo_with_branch(tmp_path, {"docs/m1/repro.py": "x=1\n"})
    assert ratchet.main(["--base", "main", "--root", str(repo)]) == 0
