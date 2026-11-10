"""Lock the README claims that were found to drift away from the code.

``scripts/audit_readme_consistency.py`` turns 20 README statements into checks.
Running the full audit needs subprocess ``--help`` probes plus one end-to-end
example run, which is too slow for the default suite; the static mode covers the
same statements minus those two probes, and is what this module asserts.

The three claims guarded explicitly below are the ones that were actually wrong
before the 2026-11 session (documented test runner, missing optional-dependency
instructions, and a documented command whose default output rewrote published
evidence).  They get their own tests so a future README edit fails loudly
instead of silently reintroducing them.
"""

from __future__ import annotations

import importlib.util
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.rst").read_text(encoding="utf-8")


def _audit_module():
    spec = importlib.util.spec_from_file_location(
        "readme_consistency_audit", ROOT / "scripts" / "audit_readme_consistency.py")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves ``from __future__ import annotations`` strings through
    # sys.modules[cls.__module__], so the module must be registered before exec.
    sys.modules.setdefault(spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_static_audit_reports_no_violation():
    result = _audit_module().audit(ROOT, live=False)
    failed = [c for c in result["checks"] if c["status"] == "FAIL"]
    assert not failed, [(c["id"], c["claim"], c["detail"]) for c in failed]
    assert result["summary"]["total"] >= 20


def test_documented_test_runner_matches_the_corpus():
    """The suite is pytest-based; documenting ``unittest discover`` hides that."""
    pytest_files = [p for p in (ROOT / "tests").glob("test_*.py")
                    if re.search(r"^import pytest", p.read_text(errors="ignore"), re.M)]
    unittest_files = [p for p in (ROOT / "tests").glob("test_*.py")
                      if "unittest.TestCase" in p.read_text(errors="ignore")]
    assert len(pytest_files) > len(unittest_files)
    assert "python -m pytest" in README
    assert "标准库 unittest" not in README


def test_install_section_documents_optional_extras():
    """``python -m tensorfvm.verification`` imports numpy/matplotlib at module load."""
    install = README.split("安装\n----")[-1].split("\n---")[0]
    assert ".[benchmark" in install or ".[dev]" in install
    assert "numpy" in install


def test_documented_channel_command_does_not_write_published_evidence(tmp_path):
    """Default output must stay out of the read-only evidence tree (docs/)."""
    source = (ROOT / "src" / "tensorfvm" / "benchmark_suite_channel.py").read_text(encoding="utf-8")
    default = re.search(r"default=Path\('([^']+)'\)", source).group(1)
    assert default.startswith("results/"), default
    assert not (ROOT / default).is_relative_to(ROOT / "docs")
    # The published snapshot is still a valid explicit target.
    assert (ROOT / "docs" / "suite-channel" / "README.rst").is_file()