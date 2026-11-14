#!/usr/bin/env python3
"""Insert the board-audit maturity verdict for m20007 (CASE-FVM-CHANNEL-DEVELOP).

Run:  python3 docs/audit_m20007/insert_verdict.py
Calls the platform API (no direct psql write) so the maturity_gate daemon picks
the row up in its next cycle.
"""
from __future__ import annotations

import json
import os
import pathlib
import urllib.request

BASE = os.environ.get("TENSOR_API_BASE", "http://127.0.0.1:8010/api/v1")
USER = os.environ["TENSOR_AGENT_USER"]
PASS = os.environ["TENSOR_AGENT_PASS"]
HERE = pathlib.Path(__file__).resolve().parent


def api(method, path, payload=None, token=None):
    req = urllib.request.Request(BASE + path, method=method)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    data = json.dumps(payload).encode() if payload is not None else None
    try:
        with urllib.request.urlopen(req, data=data, timeout=120) as r:
            return r.status, json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def load_recheck(name):
    p = HERE.parents[1] / "artifacts/m20007_audit" / name
    return json.load(open(p)) if p.exists() else None


def main():
    _, tok = api("POST", "/auth/login", {"username": USER, "password": PASS})
    token = tok["access_token"]

    bench = json.load(open(HERE / "audited_runs/benchmark.json"))
    rows = [{"re_h": c["re_h"], "got_L_e/H": c["L_e/H (computed)"],
             "ref_L_e/H": c["L_e/H (White 2006)"], "err_pct": c["L_e/H error %"],
             "u_max_ubar": c["u_max/u_bar"], "dpdx_err_pct": c["dp/dx error %"],
             "criterion_crossed": False} for c in bench["cases"]]

    recheck = {k: load_recheck(f"recheck_ny{k}_nx128_Re200.json") for k in (8,)}
    run_artifacts = [
        "docs/audit_m20007/audited_runs/benchmark.json",
        "docs/audit_m20007/audited_runs/metrics.csv",
        "docs/audit_m20007/audited_runs/centerline_profiles.csv",
        "docs/audit_m20007/audited_runs/fig_centerline_profiles.png",
        "docs/audit_m20007/audited_runs/manifest.json",
        "artifacts/m20007_audit/repro/benchmark.json",
        "artifacts/m20007_audit/recheck_ny8_nx128_Re200.json",
    ]
    for extra in ("recheck_ny17_nx128_Re100.json", "recheck_ny32_nx128_Re100.json"):
        if load_recheck(extra):
            run_artifacts.append(f"artifacts/m20007_audit/{extra}")

    evidence = {
        "impl_files": [
            "docs/audit_m20007/audited_src/benchmark_entrance_length.py",
            "artifacts/m20007_audit/recheck_entrance.py",
        ],
        "entry": ["docs/audit_m20007/audited_src/benchmark_entrance_length.py"],
        "entry_cmd": ("PYTHONPATH=src python3 -m tensorfvm.benchmark_entrance_length case "
                      "--re-h 200 --nx-per-h 8 --aspect 16 (cli(), audited copy lines 373-430)"),
        "run_artifacts": run_artifacts,
        "tests": [],
        "tests_note": ("none delivered: grep -rn entrance over /root/tp-agent/work/m20007/tests "
                       "returns 0 hits; no pytest entry point for the module"),
        "integration_callers": [],
        "integration_callers_note": (
            "none: the module is a benchmark driver that *calls* the repo kernel "
            "src/tensorfvm/inc_bench/solver.py SteadyFlow, but nothing in TensorFVM imports it "
            "(grep -rn benchmark_entrance_length /root/tensor-ecosystem/TensorFVM -> 0 hits) and "
            "tensorbench_results has 0 rows for mission 20007; the delivered files were never "
            "committed (m20007 result.head_commit=null, git status '?? src/tensorfvm/"
            "benchmark_entrance_length.py'), so the module is absent from the ecosystem repo"),
        "benchmark": {
            "referee": "White 2006 Viscous Fluid Flow 3rd ed §3-3.1 Eq (3-39): L_e/H = 0.06 Re_H",
            "referee_rows_in_tensor_benchmark_cases": 0,
            "referee_note": ("SELECT ... WHERE anchor_slug='CASE-FVM-CHANNEL-DEVELOP' -> 0 rows; "
                             "no platform referee is registered for this anchor, so L4 is "
                             "structurally unreachable and the delivery's own 8% gate is self-declared"),
            "delivered_verdict": bench["verdict"],
            "tolerance": bench["tolerance"],
            "rows": rows,
            "auditor_reproduction_ny8_Re200": recheck[8],
            "key_defect": ("3/3 delivered rows are the no-crossing fallback value x_last "
                           "(=L/H - dx/2: 7.9375 and 15.9375), not measured entrance lengths; the "
                           "99% threshold 1.485 u_bar is never reached because the n_y=8 solution "
                           "peaks at 1.4518 u_bar (3.2% below 1.5) and even the exact parabola "
                           "sampled at cell centres cannot exceed 1.4766 for even n_y"),
        },
        "repro_cmd": (
            "PYTHONPATH=src python3 -m tensorfvm.benchmark_entrance_length case --re-h 200 "
            "--nx-per-h 8 --aspect 16 --name Re200-nx8-repro --out /tmp/repro200  (run by the "
            "auditor in the m20007 worktree, 94 s, deterministic: Le/H=15.9375, "
            "u_max/ubar=1.4518132018201726, iterations=758, err=32.81% -> reproduces "
            "artifacts/m20007_audit/repro/benchmark.json exactly)"),
        "external_anchor": (
            "White, Viscous Fluid Flow 3rd ed (2006) §3-3.1 Eq (3-39) cited by the delivery as the "
            "L_e/H=0.06 Re_H correlation; the auditor did NOT independently verify the coefficient "
            "(no copy of the thesis/book in the estate) -> WITHHELD on the referee coefficient itself; "
            "independent of that, the claim 'computed L_e/H = 7.94/15.94/15.94' is refuted because "
            "those numbers are the fallback domain-end sentinels (auditor harness Le_crossed=False)"),
        "level_rationale": (
            "L2: a real input->solve->output module exists (auditor reproduced its numbers exactly), "
            "so the L2 rung (real code, non-placeholder body, entry point, non-empty run artifacts) "
            "is met. L3 refused: no mainline caller / no platform ingestion (0 tensorbench_results "
            "rows, module not in the ecosystem repo, never committed). L4 refused: the anchor has 0 "
            "referee rows in tensor_benchmark_cases and the delivered benchmark verdict is FAIL with "
            "3/3 rows being sentinel values, i.e. no validated agreement with the referee. "
            "L5 refused: single textbook anchor, no experimental substrate."),
        "findings": [
            "F1 (blocking, correctness) all three delivered L_e/H values are the no-crossing "
            "fallback x_last (benchmark_entrance_length.py:182-184: le_over_h = centerline_x_vals[-1] "
            "when le_over_h is None); they are reported as measured entrance lengths with 165%/165%/33% "
            "errors, which materially mis-describes the run",
            "F2 (blocking, criterion design) 99% x 1.5 u_bar = 1.485 u_bar is above both the discrete "
            "cell-centre ceiling of an even-n_y parabola (1.4766 at n_y=8) and the solver's own "
            "asymptotic peak (1.4518), so the benchmark is unpassable by construction for the whole "
            "declared n_y<=8 set; the criterion must be referenced to the attainability of the "
            "discrete profile (or n_y must be odd / refined)",
            "F3 (major, artifact integrity) metrics.csv and centerline_profiles.csv contain only the "
            "last single-case run (Re200-nx8) while benchmark.json contains 3 cases -> the delivered "
            "run_artifacts bundle is internally inconsistent; manifest.json self-hash is stale "
            "(MISMATCH on manifest.json itself, hash recorded before its own final content)",
            "F4 (major, delivery) nothing was committed: mission result head_commit=null, changed "
            "files all untracked; the module is absent from /root/tensor-ecosystem/TensorFVM, so the "
            "capability cannot be reused by the platform",
            "F5 (major, coverage) PLAN declares 6 cases but only the 3 n_y=8 ones were run; the "
            "discriminating n_y>=16 cases (which the report itself names as the fix) were never "
            "executed, so the report's conclusion 'needs n_y>=16 but compute was unavailable' is "
            "untested rather than measured",
            "F6 (minor, claims) report claims 'zero hand-transcription' while WHITE_COEFF=0.06 is a "
            "hard-coded literal; §5.1's 'n_y=16 verified <0.4% in the Poiseuille benchmark' has no "
            "source doc/test; the referee row for the anchor is not registered",
        ],
        "disposition": (
            "ACCEPT L2 (real functional module, reproducible) / REFUSE L3+L4. Recorded defects F1-F6; "
            "remediation for a future round: (1) make the no-crossing case an explicit failure, never a "
            "length; (2) reference the 99% criterion to the discretely attainable peak or use odd n_y / "
            "n_y>=16; (3) commit the module + a pytest gate; (4) register the referee row in "
            "tensor_benchmark_cases; (5) rebuild metrics.csv/centerline_profiles.csv from the same "
            "multi-case run"),
        "checks_not_run": [
            "the delivered `all` subcommand in one invocation (6 cases, >10 min local budget; only the "
            "3 delivered n_y=8 cases and the auditor's ny=17/32 re-measurements were run)",
            "independent verification of the 0.06 Re_H referee coefficient against the textbook (no copy "
            "of White 2006 in the estate) -> WITHHELD",
            "pixel-level/quantitative inspection of the three delivered PNGs beyond non-blank checks",
        ],
        "audit_artifacts": [
            "docs/audit_m20007/AUDIT-m20007-CASE-FVM-CHANNEL-DEVELOP.md",
            "docs/audit_m20007/audited_src/benchmark_entrance_length.py",
            "docs/audit_m20007/audited_runs/ (delivered artifact snapshot, sha256 in SHA256SUMS.txt)",
            "docs/audit_m20007/audited_docs/REPORT.md",
            "docs/audit_m20007/make_figures.py",
            "docs/audit_m20007/figs/",
            "artifacts/m20007_audit/recheck_entrance.py",
            "artifacts/m20007_audit/repro/",
        ],
        "origin": "evidence copied read-only from /root/tp-agent/work/m20007 (agent/m20007, main 0a075ce8)",
    }

    payload = {
        "repo": "TensorFVM",
        "scope": "m20007 CASE-FVM-CHANNEL-DEVELOP",
        "mission_id": 20103,
        "level": 2,
        "evidence": evidence,
        "verdict_by": "board-audit",
    }
    st, d = api("POST", "/tensor/maturity/verdicts", payload, token)
    print(st, json.dumps(d)[:400])
    if st != 200:
        # fall back: print payload for manual psql insert
        print(json.dumps(payload, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()