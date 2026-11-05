"""Artifact gate for ``scripts/audit_gci_band_coverage.py``.

Owner artifact discipline (2026-11-04): recomputable raw outputs stay out of
git, and one PR must stay under the 200 KB artifact cap.  The GCI audit dump is
1.4 MB of pure arithmetic, so the shipped ``audit.json`` is an *excerpt* that
carries the byte size and md5 of the full dump it was cut from, and ``--full``
regenerates that dump.

This gate pins the three properties that make the excerpt trustworthy:
  1. the shipped file is small (it cannot silently grow back to 1.4 MB),
  2. the recorded md5/byte count are the md5/byte count of the real ``--full``
     output (provenance is machine-checkable, not a hand-copied claim),
  3. the generated ``report.md`` describes the excerpt instead of claiming the
     per-case records are all in git.

Both modes are exercised, which costs about 25 s of pure NumPy work.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(ROOT, "scripts", "audit_gci_band_coverage.py")
CAP_BYTES = 200 * 1024
NAME = "gci-band-coverage"


def _run(tmp_path, *extra):
    env = dict(os.environ)
    src = os.path.join(ROOT, "src")
    env["PYTHONPATH"] = os.pathsep.join(
        [src] + [p for p in [env.get("PYTHONPATH")] if p])
    proc = subprocess.run(
        [sys.executable, SCRIPT, NAME, *extra],
        cwd=str(tmp_path), env=env, capture_output=True, text=True,
        timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = tmp_path / "outputs" / NAME
    return out / "audit.json", out / "RESULTS.md", out / "gci_band_coverage.png"


@pytest.fixture(scope="module")
def excerpt_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("excerpt"))


@pytest.fixture(scope="module")
def full_run(tmp_path_factory):
    return _run(tmp_path_factory.mktemp("full"), "--full")


def test_shipped_audit_is_within_an_artifact_budget(excerpt_run, tmp_path):
    audit = excerpt_run[0]
    size = audit.stat().st_size
    assert size < CAP_BYTES, (
        "audit.json is %d bytes; the excerpt policy (EXCERPT_SAMPLES) must keep "
        "it well under the %d byte artifact cap" % (size, CAP_BYTES))
    payload = json.loads(audit.read_text())
    assert payload["_excerpt_notice"]
    assert len(payload["rows"]) <= 2
    assert len(payload["noise_rows"]) <= 2
    # the aggregate tables (what the report quotes) are kept in full
    assert payload["summary"]["cases"] == 1920
    assert payload["summary"]["noise_cases"] == 1000
    assert len(payload["by_alpha"]) == 5
    assert len(payload["by_order"]) == 6


def test_excerpt_provenance_matches_the_full_dump(excerpt_run, full_run):
    payload = json.loads(excerpt_run[0].read_text())
    notice = payload["_excerpt_notice"]
    full_bytes = full_run[0].stat().st_size
    full_md5 = hashlib.md5(full_run[0].read_bytes()).hexdigest()
    assert ("%d bytes" % full_bytes) in notice, notice
    assert full_md5 in notice, notice
    assert full_bytes > CAP_BYTES, "the full dump is the big one; excerpt is the ship"


def test_full_mode_keeps_every_per_case_record(excerpt_run, full_run):
    full = json.loads(full_run[0].read_text())
    assert "rows" not in full or len(full["rows"]) == 1920
    assert len(full["noise_rows"]) == 1000
    excerpt = json.loads(excerpt_run[0].read_text())
    # the excerpt is a prefix cut, not a re-sampling: samples must be identical
    assert excerpt["rows"] == full["rows"][: len(excerpt["rows"])]
    assert excerpt["noise_rows"] == full["noise_rows"][: len(excerpt["noise_rows"])]
    for key in ("summary", "by_alpha", "by_order", "noise"):
        assert excerpt[key] == full[key]


def test_generated_report_describes_the_excerpt(tmp_path):
    _run(tmp_path)
    report = (tmp_path / "docs" / NAME / "report.md").read_text()
    assert "audit.json" in report
    assert "节选" in report, "report must not claim the full dump is in git"
    assert "--full" in report