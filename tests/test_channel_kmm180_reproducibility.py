"""Byte-reproducibility of the CASE-FVM-CHANNEL-KMM180 artifact set.

Why this file exists
--------------------
``manifest.json`` pins ``sha256`` for every durable artifact, and the gate
(``scripts/run_case_fvm_channel_kmm180.py --check``) fails when a byte moves.
That contract only means something if the artifacts are *reproducible*: until
m18277 ``benchmark.json`` embedded ``elapsed_s`` / ``matrix_elapsed_s`` /
``repro_cmd`` (which contains the ``--out`` path) and ``results.json`` embedded
``elapsed_s``, so re-running the documented reproducer changed three pinned
hashes and the gate could never be satisfied after a legitimate regeneration.
An earlier hand edit then appended a "artifact gate clean" sentence to
``REPORT.md`` without regenerating ``manifest.json``; that claim was false and
the gate (and ``test_channel_kmm180_artifacts.py::test_committed_artifacts_pass_
the_gate``) was red on ``main``.

Layers
------
* **unit** -- :func:`canonical_benchmark` must strip exactly the volatile keys,
  and the writer must publish them in ``timings.json`` instead of dropping them;
* **two writes** -- the same case written into two directories, with different
  wall clocks and a different ``repro_cmd``, must give byte-identical
  ``benchmark.json`` / ``metrics.csv`` / ``profiles.csv`` / ``stations.csv``;
* **end to end** -- two ``--quick`` driver runs into different ``--out``
  directories must give a byte-identical pinned artifact set, including
  ``results.json`` and ``manifest.json`` (this is the check that was impossible
  before the fix);
* **committed** -- the checked-in artifacts respect the same contract.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorfvm.benchmark_channel_kmm1987 import run_case
from tensorfvm.verification.kmm180_artifacts import (
    TIMINGS_FILE,
    VOLATILE_BENCHMARK_KEYS,
    build_benchmark,
    canonical_benchmark,
    write_artifacts,
)
from tensorfvm.verification.kmm1987 import load_kmm1987_reference

REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "docs" / "case-fvm-channel-kmm180"
DRIVER = REPO_ROOT / "scripts" / "run_case_fvm_channel_kmm180.py"

#: Files that ``manifest.json`` pins and that must therefore be reproducible.
PINNED = ("benchmark.json", "results.json", "results.csv", "metrics.csv",
          "profiles.csv", "stations.csv", "REPORT.md", "manifest.json")


def _env():
    """Parent environment + this worktree's ``src`` (inherit, do not replace:
    the interpreter needs its own library/back-end environment)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    return env


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Unit: the stripping contract
# --------------------------------------------------------------------------- #

def test_canonical_benchmark_strips_exactly_the_volatile_keys():
    payload = dict(schema="s", cases=[], verdict="FAIL",
                   elapsed_s=1.0, matrix_elapsed_s=2.0, repro_cmd="cmd --out x")
    canonical = canonical_benchmark(payload)
    for key in VOLATILE_BENCHMARK_KEYS:
        assert key not in canonical
    assert canonical == dict(schema="s", cases=[], verdict="FAIL")


def test_writer_publishes_the_volatile_values_instead_of_dropping_them(tmp_path):
    dns = load_kmm1987_reference()
    case = run_case("sst", 64)
    write_artifacts(tmp_path, [case], dns, elapsed_s=12.5, matrix_elapsed_s=9.5,
                    repro_cmd="cmd --out some/other/path")
    written = json.loads((tmp_path / "benchmark.json").read_text())
    assert not [k for k in VOLATILE_BENCHMARK_KEYS if k in written]
    timings = json.loads((tmp_path / TIMINGS_FILE).read_text())
    assert timings["elapsed_s"] == 12.5
    assert timings["matrix_elapsed_s"] == 9.5
    assert timings["repro_cmd"] == "cmd --out some/other/path"


def test_two_writes_with_different_wall_clocks_are_byte_identical(tmp_path):
    dns = load_kmm1987_reference()
    case = run_case("sst", 64)
    a, b = tmp_path / "a", tmp_path / "b"
    write_artifacts(a, [case], dns, elapsed_s=1.0, matrix_elapsed_s=1.0,
                    repro_cmd="cmd --out a")
    write_artifacts(b, [case], dns, elapsed_s=999.0, matrix_elapsed_s=999.0,
                    repro_cmd="cmd --out b")
    for name in ("benchmark.json", "metrics.csv", "profiles.csv", "stations.csv"):
        assert _digest(a / name) == _digest(b / name), name
    # ... while the volatile companion does differ, i.e. the numbers are still
    # published rather than silently lost.
    assert _digest(a / TIMINGS_FILE) != _digest(b / TIMINGS_FILE)


# --------------------------------------------------------------------------- #
# End to end: two driver runs, different --out, identical pinned bytes
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def quick_runs(tmp_path_factory):
    outs = []
    for tag in ("run1", "run2"):
        out = tmp_path_factory.mktemp("kmm180_" + tag)
        proc = subprocess.run(
            [sys.executable, "-m", "tensorfvm.benchmark_channel_kmm1987",
             "--quick", "--out", str(out)],
            cwd=REPO_ROOT, capture_output=True, text=True, env=_env())
        # A FAIL verdict exits non-zero; that is a scientific result, not a
        # run failure, so the artifacts decide.
        assert (out / "manifest.json").is_file(), proc.stderr[-3000:]
        outs.append(out)
    return outs


def test_quick_driver_runs_are_byte_reproducible(quick_runs):
    first, second = quick_runs
    for name in PINNED:
        assert _digest(first / name) == _digest(second / name), name


def test_quick_driver_run_manifest_matches_its_own_files(quick_runs):
    """One run, regenerated: every pinned hash must match that run's bytes."""
    first, _ = quick_runs
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["sha256"], "nothing would be pinned"
    for name, digest in manifest["sha256"].items():
        assert _digest(first / name) == digest, name


# --------------------------------------------------------------------------- #
# Committed artifacts
# --------------------------------------------------------------------------- #

def test_committed_artifacts_obey_the_hash_contract():
    manifest = json.loads((CASE_DIR / "manifest.json").read_text())
    assert TIMINGS_FILE not in (manifest.get("sha256") or {})
    assert TIMINGS_FILE in (manifest.get("volatile_artifacts") or [])
    for name in ("benchmark.json", "results.json"):
        payload = json.loads((CASE_DIR / name).read_text())
        leaked = [k for k in VOLATILE_BENCHMARK_KEYS if k in payload]
        assert not leaked, f"{name} leaks {leaked}"
    timings = json.loads((CASE_DIR / TIMINGS_FILE).read_text())
    assert timings.get("invocation"), "the real invocation must be recorded"


def test_committed_report_is_the_generated_text():
    """``REPORT.md`` is generated; a hand-appended claim invalidates its hash.

    The regression this pins: a "artifact gate clean" sentence was hand-added
    to ``REPORT.md`` while the gate was in fact failing.
    """
    manifest = json.loads((CASE_DIR / "manifest.json").read_text())
    report = CASE_DIR / "REPORT.md"
    assert _digest(report) == manifest["sha256"]["REPORT.md"]
    text = report.read_text()
    assert "re-open" not in text