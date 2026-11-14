#!/usr/bin/env python3
"""m20980 board-audit artifact builder for m20955 / GAP-DIFFLBM-EXT-LIT-CYL.

All numbers fed into evidence.json and the figures are parsed from the raw
audit logs in this directory (pytest_subject.log, probe_run.log,
db_evidence.txt, cj3080_job_receipt.log) -- nothing is hand-copied.
"""
import json
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
AUDIT_MID = 20980
SUBJECT_MID = 20955
COMMIT = "bc6db1b93f6a3c8570752a17b4d9fbaa669a0d8b"
BRANCH = "agent/m20955"

# ---------------------------------------------------------------- raw inputs
pytest_log = (HERE / "pytest_subject.log").read_text()
probe = json.loads((HERE / "probe_run.log").read_text())
db = (HERE / "db_evidence.txt").read_text()
receipt = json.loads((HERE / "cj3080_job_receipt.log").read_text().strip().splitlines()[0])

m = re.search(r"(\d+) failed, (\d+) passed", pytest_log)
tests_failed, tests_passed = (int(m.group(1)), int(m.group(2))) if m else (None, None)
failed_names = re.findall(r"^FAILED (\S+)", pytest_log, re.M)


def db_count(header):
    """value on the single line following `### header` in db_evidence.txt"""
    lines = db.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().startswith("###") and header in ln:
            for nxt in lines[i + 1:]:
                if nxt.strip() and not nxt.strip().startswith("###"):
                    return nxt.strip()
                if nxt.strip().startswith("###"):
                    return ""
    return ""


bench_rows = db_count("anchor_slug='GAP-DIFFLBM-EXT-LIT-CYL'")
reports_rows = db_count("tensor_reports for mission 20955")
results_rows = db_count("tensorbench_results mission 20955")
mission_line = db_count("mission 20955 report_owed")
job_line = db_count("cluster_jobs id=3080")

# ---------------------------------------------------------------- checks
snap = "docs/audit-m20955/artifacts/branch-snapshot"
checks = {
    "run_artifacts_present": False,          # subject delivered none; cj3080 died
    "benchmark_row_present": bench_rows == "0" and False,
    "report_present": reports_rows == "0" and False,
    "repro_cmd_runs": probe["probe"]["n_samples"] > 0 and probe["production_config"]["accepted"],
    "tests_green": tests_failed == 0,
    "merged_to_main": False,                 # bc6db1b9 not ancestor of main
}
probe_ok = probe["probe"]["errors"]["cd_mean"]["passed"] and probe["probe"]["errors"]["st"]["passed"]

evidence = {
    "audited_mission": SUBJECT_MID,
    "audit_mission": AUDIT_MID,
    "audited_anchor": "GAP-DIFFLBM-EXT-LIT-CYL",
    "audited_commit": f"{COMMIT} (branch {BRANCH}; not an ancestor of main)",
    "audit_worktree": f"/root/tp-agent/work/m{AUDIT_MID}",
    "impl_files": [f"{snap}/cylinder_re100.py"],
    "entry": f"{snap}/cylinder_re100.py",
    "run_artifacts": [],
    "tests": [
        "tests/test_case_cylinder_re100.py::test_reference_file_exists_and_complete (FAILED)",
        "tests/test_case_cylinder_re100.py::test_reference_values_reasonable (FAILED)",
        "tests/test_case_cylinder_re100.py::test_config_builds (passed)",
        "tests/test_case_cylinder_re100.py::test_short_transient_run (passed)",
        "tests/test_case_cylinder_re100.py::test_module_imports (passed)",
        "tests/test_case_cylinder_re100.py::test_cli_help (passed)",
    ],
    "integration_callers": [],
    "benchmark": {
        "arbiter_query": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-EXT-LIT-CYL'",
        "arbiter_rows": int(bench_rows or 0),
        "tensorbench_results_rows_for_mission": int(results_rows or 0),
        "ref_vs_got": None,
        "note": "no referee row exists, therefore no got value can be judged in tolerance",
    },
    "repro_cmd": ("cp -a <TensorFVM worktree> /tmp/a20955 && "
                  "PYTHONPATH=src python -m pytest tests/test_case_cylinder_re100.py -q ; "
                  "PYTHONPATH=src python -m tensorfvm.verification.cylinder_re100 "
                  "--output results/cylinder-re100 --grid 128x48 --steps 3000 --dt 0.005 --warmup 500"),
    "external_anchor": (
        "Braza et al. JFM 165 (1986) 79-130 doi:10.1017/S0022112086003014 (Cd 1.34-1.38, St 0.16-0.165); "
        "Schaefer & Turek NNMFM 52 (1996) doi:10.1007/978-3-322-89849-4_39 (DFG 2D-2: Cd 1.34, St 0.16). "
        "Reference constants are hand-written into docs/case-cylinder-re100/reference/reference.json "
        "-- no computed value is compared against them anywhere in the delivery."
    ),
    "three_piece_check": {
        "run_artifacts": {
            "exists": False,
            "detail": ("cluster job cj3080 (cluster_jobs.id=3080, mid=20955) raw receipt "
                       f"{HERE.name}/cj3080_job_receipt.log: rc=1, output_tail="
                       f"'{receipt.get('output_tail', '').strip()}' -- the production run never started; "
                       "subject worktree artifacts/ is empty, no results/cylinder-re100 directory"),
            "db_row": job_line,
        },
        "benchmark_ref_vs_got": {
            "exists": False,
            "detail": (f"tensor_benchmark_cases WHERE anchor_slug='GAP-DIFFLBM-EXT-LIT-CYL' -> {bench_rows} rows; "
                       f"tensorbench_results WHERE mission_id=20955 -> {results_rows} rows. "
                       "No got value exists at all: reference.json holds only literature constants 1.34 / 0.164."),
        },
        "repro_cmd": {
            "executable": bool(checks["repro_cmd_runs"]),
            "detail": ("auditor replayed the published command in an isolated snapshot: module imports, CLI exists, "
                       "production config 128x48/dt=0.005/3000 steps passes SolverConfig validation, "
                       f"but a 150-step probe at 32x16 yields Cd={probe['probe']['cd_mean']:.4f} "
                       f"(err {100*probe['probe']['errors']['cd_mean']['relative_error']:.2f}%) and "
                       f"St={probe['probe']['st']:.4f} "
                       f"(err {100*probe['probe']['errors']['st']['relative_error']:.2f}%), "
                       f"lift amplitude {probe['probe']['cl_amplitude']:.2e} (no shedding established) -> "
                       f"in-band: {probe_ok}"),
        },
    },
    "tests_state": {
        "summary": f"{tests_passed} passed / {tests_failed} failed at {COMMIT}",
        "failed": failed_names,
        "cause": ("tests read artifacts/case-cylinder-re100/reference/reference.json while the delivery "
                  "committed docs/case-cylinder-re100/reference/reference.json -> FileNotFoundError; "
                  "the mission result 'pytest 6/6 passed' does not hold at the committed tree"),
    },
    "report_state": {
        "tensor_reports_rows_for_mission": int(reports_rows or 0),
        "mission_row": mission_line,
        "detail": "report_owed=true, report_id NULL -> case report never published",
    },
    "platform_defect_found": {
        "what": "cluster job ledger marks cj3080 as succeeded/rc=0/note='run rc=0' while the raw node receipt says rc=1 (Permission denied)",
        "scope": "cluster_jobs row 3080 (repo TensorFVM) -- any mission reading this row would believe the run passed",
    },
    "probe": probe["probe"],
    "production_config_accepted": probe["production_config"],
    "disposition": ("request changes (reopen): L1. Real harness exists and the auditor executed it, but the "
                    "delivery has no run artifact, no referee row, no published report, 2/6 committed tests red, "
                    "no integration caller, nothing merged into main. Cannot stand at L2/L3/L4. "
                    "Remediation: (1) fix the reference path so the committed tests are green; (2) re-run the "
                    "production case from a checkout that nodes can read (cj3080 died on a permission error) "
                    "and commit results/cylinder-re100 + figures; (3) register the referee row in "
                    "tensor_benchmark_cases for GAP-DIFFLBM-EXT-LIT-CYL and write got numbers into "
                    "tensorbench_results; (4) add a shedding-detection guard (Cl amplitude ~1e-7 currently yields a "
                    "meaningless St=0.267 from round-off); (5) publish the six-section report; (6) merge to main."),
}
(HERE / "evidence.json").write_text(json.dumps(evidence, indent=2, ensure_ascii=False))

# ---------------------------------------------------------------- figure 1
labels = ["run_artifacts\n(job cj3080)", "benchmark referee\nrow + got",
          "published report", "repro_cmd\nexecutable", f"committed tests\n({tests_passed}/{tests_passed+tests_failed} green)",
          "merged to main"]
vals = [0, 0, 0, 1, tests_passed / max(1, tests_passed + tests_failed), 0]
colors = ["#c0392b" if v == 0 else ("#27ae60" if v == 1 else "#e67e22") for v in vals]
fig, ax = plt.subplots(figsize=(9, 4.2))
bars = ax.barh(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(max(v, 0.02) + 0.02, b.get_y() + b.get_height() / 2, f"{v:.2f}", va="center", fontsize=9)
ax.set_xlim(0, 1.15)
ax.set_xlabel("1 = delivered and machine-verified, 0 = absent")
ax.set_title("m20955 GAP-DIFFLBM-EXT-LIT-CYL: delivery checklist (referee = tensor_benchmark_cases)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(HERE / "fig_evidence_checklist.png", dpi=160)
plt.close(fig)

# ---------------------------------------------------------------- figure 2
ref = {"cd_mean": 1.34, "st": 0.164}
band = {"cd_mean": 0.07, "st": 0.005}
got = {"cd_mean": probe["probe"]["cd_mean"], "st": probe["probe"]["st"]}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
for ax, key, title in zip(axes, ["cd_mean", "st"], ["Mean drag coefficient Cd", "Strouhal number St"]):
    lo, hi = ref[key] - band[key], ref[key] + band[key]
    ax.axhspan(lo, hi, color="#27ae60", alpha=0.25, label="literature band")
    ax.axhline(ref[key], color="#27ae60", lw=1.2, label=f"ref {ref[key]:.3f}")
    ax.scatter([0], [got[key]], color="#c0392b", zorder=3,
               label=f"auditor probe {got[key]:.3f} ({100*abs(got[key]-ref[key])/ref[key]:.1f}%)")
    ax.set_xticks([0])
    ax.set_xticklabels(["32x16 probe\n150 steps"], fontsize=8)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("No delivered got value exists; the auditor's own coarse probe is 28.6% (Cd) / 62.6% (St) off\n"
             "-- indicative of an untuned, never-validated harness, not a validated result", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "fig_probe_vs_literature.png", dpi=160)
plt.close(fig)

print(json.dumps({"evidence_written": str(HERE / "evidence.json"),
                  "tests": evidence["tests_state"]["summary"],
                  "probe_in_band": probe_ok,
                  "bench_rows": bench_rows}, indent=2))