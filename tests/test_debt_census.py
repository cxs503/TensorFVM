"""Contract tests for the device-debt census (tools/debt_census.py).

The census feeds `docs/p8/FVM_DEBT_REPORT.md`; these tests pin the machine
contract: five required JSON fields, byte-identical reruns (determinism) and an
honest coverage rule (below 70 % the uncovered files must be named in full).
"""
import json
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = REPO / "tools" / "debt_census.py"
REQUIRED_FIELDS = (
    "numpy_only_count",
    "sync_points",
    "no_device_constructs",
    "host_loops",
    "ast_coverage_pct",
)


def run_census(out_path):
    subprocess.run(
        [sys.executable, str(SCRIPT), "--repo-root", str(REPO), "--out", str(out_path), "--quiet"],
        check=True,
        capture_output=True,
    )
    return out_path.read_bytes()


def test_census_schema_and_determinism(tmp_path):
    first = run_census(tmp_path / "a.json")
    second = run_census(tmp_path / "b.json")
    assert first == second, "census must be byte-identical across runs"
    payload = json.loads(first)
    for field in REQUIRED_FIELDS:
        assert field in payload, f"missing required field {field}"
    # Counts are non-negative integers and the population adds up.
    for field in REQUIRED_FIELDS[:-1]:
        assert isinstance(payload[field], int) and payload[field] >= 0, field
    assert payload["files_parsed"] + len(payload["uncovered_files"]) == payload["files_scanned"]
    assert payload["files_scanned"] > 0
    # No wall clock / machine-specific path may leak into the artefact.
    assert str(REPO) not in first.decode()
    assert payload["source_digest"].startswith("sha256:")


def test_census_covers_the_tree(tmp_path):
    payload = json.loads(run_census(tmp_path / "c.json"))
    if payload["ast_coverage_pct"] < 70.0:
        assert payload["uncovered_files"], "coverage below 70% must list every uncovered file"
    assert payload["ast_coverage_pct"] == round(
        100.0 * payload["files_parsed"] / payload["files_scanned"], 2
    )
    # Detail blocks stay consistent with the headline numbers.
    sync = payload["breakdown"]["sync_points"]
    assert payload["sync_points"] == sync["cpu"] + sync["numpy"]
    assert payload["host_loops"] == sum(payload["breakdown"]["host_loops_by_kind"].values())
    assert payload["numpy_only_count"] == len(payload["numpy_only_modules"])
    assert payload["no_device_constructs"] == len(payload["no_device_sites"])


def test_excluded_directories_are_not_debt(tmp_path):
    payload = json.loads(run_census(tmp_path / "d.json"))
    for record in payload["per_file"]:
        parts = record["path"].split("/")
        assert not {"tests", "test", "examples", "example"} & set(parts[:-1]), record["path"]