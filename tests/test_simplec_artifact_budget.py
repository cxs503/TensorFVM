"""Guards on the shipped SIMPLEC-mobility evidence artifact directory.

The merge governor holds a pull request whose artifact-class blobs exceed
200 KB in total, so ``scripts/artifacts/simplec-mobility`` ships a size-capped
form of the verification evidence (see ``scripts/slim_simplec_artifacts.py``).
Capping must not turn evidence into decoration, so these tests pin the two
properties that matter:

* the budget itself (every file, and the directory total), and
* that the numbers which carry meaning survive byte-for-byte: the audit summary
  in ``verification.json`` equals ``evidence/simplec-audit.json``, every
  configured run keeps its metric gates, and every subsampled per-iteration
  trace declares its original length plus the SHA-256 of the full trace so the
  dropped rows stay verifiable against a regeneration.
"""

import hashlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "scripts" / "artifacts" / "simplec-mobility"
FIGURES = ROOT / "docs" / "simplec-mobility"
CAP_BYTES = 200 * 1024

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.slim_simplec_artifacts import STRIDE, slim_summary, subsample  # noqa: E402


def canonical(payload):
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def load(name):
    return json.loads((ARTIFACTS / name).read_text())


@pytest.fixture(scope="module")
def audit():
    return load("evidence/simplec-audit.json")


@pytest.fixture(scope="module")
def verification():
    return load("verification.json")


def test_every_artifact_file_is_under_the_cap():
    sizes = {
        path.name: path.stat().st_size
        for path in ARTIFACTS.rglob("*")
        if path.is_file()
    }
    assert sizes, "artifact directory is empty"
    over = {name: size for name, size in sizes.items() if size > CAP_BYTES}
    assert not over, "individually over the 200 KB governor cap: %s" % over


def test_artifact_directory_total_is_under_the_cap():
    total = sum(path.stat().st_size for path in ARTIFACTS.rglob("*") if path.is_file())
    assert total <= CAP_BYTES, "artifact directory is %.1f KB > 200 KB" % (total / 1024)


def test_json_payloads_are_compact_single_line():
    # indentation was the single largest avoidable cost; one line keeps it gone
    for name in ("verification.json", "evidence/simplec-audit.json"):
        text = (ARTIFACTS / name).read_text().strip()
        assert "\n" not in text, "%s is not compact" % name
        json.loads(text)  # still valid JSON


def test_verification_embeds_the_same_audit_as_the_evidence_file(audit, verification):
    assert canonical(verification["audit"]) == canonical(audit)
    assert (
        verification["guards"] and verification["mutation_control"]["sensitive"] is True
    )


def test_slimming_provenance_is_recorded(verification):
    block = verification["artifact_slimming"]
    assert block["cap_bytes"] == CAP_BYTES
    assert "verify_simplec_mobility.py" in block["regenerate_full"]
    assert "slim_simplec_artifacts.py" in block["apply_cap"]
    originals = block["original_json_bytes"]
    for name, size in originals.items():
        assert size > 0
    assert (
        originals["verification.json"]
        > (ARTIFACTS / "verification.json").stat().st_size
    )
    assert (
        originals["evidence/simplec-audit.json"]
        > (ARTIFACTS / "evidence/simplec-audit.json").stat().st_size
    )


def test_every_configured_run_keeps_its_gate_numbers(audit):
    runs = audit["runs"]
    assert len(runs) == 9
    label_keys = sorted(
        (run["configuration"]["label"], run["resolution"]) for run in runs
    )
    assert label_keys == sorted(
        (label, resolution)
        for label in (
            "simple-default",
            "simple-tuned",
            "simplec-a07-p10",
            "simplec-a08-p10",
        )
        for resolution in (12, 24)
    ) + [("simplec-no-floor", 12)]
    for run in runs:
        assert run["converged"] is True and run["passed"] is True
        metrics = {entry["name"]: entry for entry in run["metrics"]}
        assert set(metrics) == {
            "velocity_l2",
            "velocity_linf",
            "pressure_gradient",
            "pressure_curve_l2",
            "wall_shear_linf",
            "force_balance",
            "relative_mass_imbalance",
            "scaled_divergence",
            "solver_residual",
            "convergence_failure",
        }
        for entry in metrics.values():
            assert isinstance(entry["value"], float)
            assert entry["passed"] is True
            assert (
                entry["limit"] == pytest.approx(audit["error_limit"])
                or entry["limit"] > 0
            )
        # analytic gates quoted in report.md §3/§6 must match the trace record
        assert metrics["velocity_l2"]["value"] == pytest.approx(
            run["velocity_l2"], rel=1e-12
        )
        assert metrics["pressure_gradient"]["value"] == pytest.approx(
            run["pressure_gradient"], rel=1e-12
        )


def test_headline_numbers_are_unchanged(audit):
    assert audit["schema"] == "tensorfvm.simplec-audit/1"
    assert audit["passed"] is True
    assert audit["error_limit"] == pytest.approx(0.03)
    assert audit["summary_resolution"] == 12
    assert audit["max_iterations"] == 2000
    assert audit["fixed_point_agreement_max"] == pytest.approx(
        2.0802836903175553e-06, rel=1e-15
    )
    assert audit["fixed_point_agreement_limit"] == pytest.approx(1e-4)
    assert audit["fixed_point_agreement_passed"] is True
    assert audit["simple_alpha_p_one_failed"] is True
    assert audit["simplec_all_gates_passed"] is True
    assert audit["simplec_sustains_pressure_relaxation_one"] is True
    assert set(audit["simplec_limited_dofs"].values()) == {0}
    assert audit["limiter_observation"]["floor_free_equivalent"] is True
    assert audit["provenance"]["source_sha256"]["src/tensorfvm/solver.py"]
    failed = audit["failed_simple_history"]
    assert list(failed) == ["SIMPLE alpha_u=0.8 alpha_p=1 (ny=12)"]
    assert failed["SIMPLE alpha_u=0.8 alpha_p=1 (ny=12)"]["failure"]
    assert failed["SIMPLE alpha_u=0.8 alpha_p=1 (ny=12)"]["converged"] is False


def test_every_history_trace_is_accounted_for(audit):
    holders = list(audit["runs"]) + list(audit["failed_simple_history"].values())
    sampled = 0
    for holder in holders:
        trace = holder["history"]
        record = holder.get("history_sampling")
        assert trace, "history trace vanished"
        assert record is not None, "sampled trace must declare history_sampling"
        assert record["stride"] == STRIDE
        assert record["total_iterations"] >= len(trace) == record["kept"]
        assert len(record["full_sha256"]) == 64
        iterations = [row["iteration"] for row in trace]
        assert iterations == sorted(set(iterations)), "rows must be unique and ordered"
        assert iterations[0] == 1
        assert iterations[-1] == record["total_iterations"]
        # the retained rows keep the full residual precision of the run
        for row in trace:
            assert set(row) == {
                "iteration",
                "continuity",
                "momentum",
                "velocity_change",
                "mass_imbalance",
            }
            assert isinstance(row["momentum"], float)
        sampled += 1
    assert sampled == 10


def test_traces_dense_enough_to_show_the_transient_and_the_plateau(audit):
    for run in audit["runs"]:
        record = run["history_sampling"]
        assert record["kept"] >= 8
        assert record["kept"] <= 2 + 3 + record["total_iterations"] // STRIDE + 2
        # start-up and final plateau are both represented
        assert run["history"][0]["iteration"] == 1
        assert run["history"][-1]["momentum"] == pytest.approx(
            run["solver_residual"], rel=1e-9, abs=0.0
        )


def test_figures_stay_legible():
    """Figures ship under ``docs/`` (the push gate only lets document
    figures through), so the artifact directory stays numeric-only."""
    from PIL import Image

    for name, minimum_width in (
        ("simplec-convergence.png", 1000),
        ("simplec-profile.png", 900),
        ("simplec-iterations.png", 1000),
    ):
        path = FIGURES / name
        with Image.open(path) as image:
            assert image.size[0] >= minimum_width, "%s was rescaled" % name
            assert image.mode in {"P", "L", "RGB"}
        assert path.stat().st_size < 60 * 1024, "%s is not palette-capped" % name


def test_subsample_keeps_both_ends_and_is_deterministic():
    trace = [{"iteration": index, "momentum": float(index)} for index in range(1, 101)]
    kept, record = subsample(trace)
    assert record["total_iterations"] == 100
    assert kept[0] == trace[0] and kept[-1] == trace[-1]
    assert kept == subsample(trace)[0]
    assert (
        record["full_sha256"] == hashlib.sha256(canonical(trace).encode()).hexdigest()
    )
    assert len(kept) == record["kept"] < len(trace)


def test_slim_summary_is_idempotent_guarded():
    summary = {
        "schema": "tensorfvm.simplec-audit/1",
        "runs": [
            {
                "configuration": {"label": "x"},
                "iterations": 20,
                "history": [
                    {"iteration": index, "momentum": float(index)}
                    for index in range(1, 21)
                ],
            }
        ],
        "failed_simple_history": {},
    }
    slim = slim_summary(summary)
    assert "history_sampling" in slim["runs"][0]
    assert len(summary["runs"][0]["history"]) == 20, "input must not be mutated"
    with pytest.raises(SystemExit):
        slim_summary(slim)
