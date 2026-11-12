"""Build the tensor_maturity_verdicts evidence payload for m16182 and INSERT it.

Values are read from the machine-produced artifacts in runs/m16204/ (no hand
transcription): repro_compare.json and audit_probe.json.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

HERE = Path("/root/tp-agent/work/m16204/docs/board_audit/m16204/artifacts")
REPO_WT = Path("/root/tp-agent/work/m16182")
CASE_SRC = REPO_WT / "src/tensorfvm/case_blasius_plate.py"

cmp_ = json.loads((HERE / "repro_compare.json").read_text())
probe = json.loads((HERE / "audit_probe.json").read_text())
head = cmp_["headline_numbers"]
p1 = probe["P1_independent_blasius"]
seeds = {row["seed"]: {"cf_err": row["cf_err"], "H_err": row["H_err"],
                       "passed": row["pass"]} for row in probe["P2_seed_independence"]}

evidence = {
    "scheme": "board-audit 2026-11-12 (mission 16204, cross-check of m16182); "
              "level = functional substance, report length irrelevant",
    "declared_level": 3,
    "mission_reviewed": 16182,
    "audited_commit": "cdbb3096 (branch agent/m16182, 2026-11-12 10:37Z); main = 0cd53fa5 = its parent",
    "impl_files": [
        "src/tensorfvm/case_blasius_plate.py:1-512 (case def + LITERATURE + ERROR_LIMIT:67 + "
        "similarity_crosscheck:102 + boundary_layer_integrals:139 + compare_against_reference:239 + "
        "FVM_CONFIGS:259 + run_fvm_case:270 + run_march_case:448 + run_march_sensitivity:488)",
        "scripts/run_case_blasius_plate.py:79-128 (CLI main; --summary-only merge-only mode at :87,:95)",
        "scripts/plot_case_blasius_plate.py (4 figures from artifacts + FVM log parsing)",
        "commit cdbb3096 on branch agent/m16182 only (NOT in main)",
    ],
    "entry": "scripts/run_case_blasius_plate.py:79 main() -> case_blasius_plate.run_march_case:448 "
             "-> rans_ship_resistance.boundary_layer.march_boundary_layer (model='laminar', trip_x=None)",
    "run_artifacts": [
        "docs/case-blasius-plate/artifacts/march.json (committed; cf/H/delta*/theta + continuity_residual 3.32e-13)",
        "docs/case-blasius-plate/artifacts/march-sensitivity.json (committed; ny 128/256/512, stations 101/401/1601)",
        "docs/case-blasius-plate/artifacts/summary.json (committed; literature audit crosscheck)",
        "docs/case-blasius-plate/artifacts/fvm-partial-traces.json (committed; FVM unconverged trace)",
        "docs/case-blasius-plate/artifacts/figure-summary.json (committed)",
        "docs/case-blasius-plate/figures/fig1..fig4 (committed PNG 92/36/63/88 KB)",
        "runs/blasius-plate/*.json in the worktree = byte-identical working copies (untracked)",
        "auditor clean-dir reproduction: /root/tp-agent/work/m16204/runs/m16204/repro-clean/*.json",
    ],
    "tests": [
        "tests/test_case_blasius_plate.py::7 tests (literature refs, similarity crosscheck, exact-profile "
        "integrals + delta99 truncation bias pinned at 0.991, bad-input rejection, gate pass/fail, "
        "march case within gate, fvm smoke asserting converged is False)",
        "auditor run: PYTHONPATH=src python3 -m pytest tests/test_case_blasius_plate.py -q -> "
        "7 passed in 12.02 s (runs/m16204/pytest_audit.log)",
    ],
    "integration_callers": [
        "NONE: grep for case_blasius_plate/CASE-FVM-BLASIUS across the repo finds only the case module, "
        "its CLI, its plot script, its test and its docs; nothing imports it",
        "direction is case -> solver (normal for a benchmark case): calls "
        "rans_ship_resistance.boundary_layer.march_boundary_layer and body_fitted.BodyFittedSolver",
        "git ls-tree -r main | grep case_blasius -> 0 hits (case commit not merged)",
    ],
    "benchmark": {
        "anchor_slug": "CASE-FVM-BLASIUS-PLATE",
        "registry_row": "ABSENT: SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE "
                        "anchor_slug='CASE-FVM-BLASIUS-PLATE' -> 0 rows (platform registration gap, F3)",
        "tolerance": "<3% on both quantities, source = src/tensorfvm/case_blasius_plate.py:67 ERROR_LIMIT=0.03 "
                     "+ case sheet text (machine cannot re-derive it from a registry)",
        "metric": "relative error of local c_f and of shape factor H = delta*/theta at Re_x = 1e5",
        "declared_device": "march = thin-shear-layer boundary-layer march, laminar closure (affordable rung)",
        "not_the_anchor_device": "the 2-D FVM SIMPLE rung named by the anchor never converged (F1)",
        "got_cf": head["cf"], "ref_cf": head["ref_cf"], "cf_relative_error": head["cf_relative_error"],
        "got_H": head["H"], "ref_H": head["ref_H"], "H_relative_error": head["H_relative_error"],
        "passed": head["passed"],
        "grid_convergence_cf": {row["tag"]: row["cf_relative_error"] for row in cmp_["sensitivity"]},
        "fvm_rung": "WITHHELD: 75 outer iterations, cf = 1.353e-2 vs target 2.0998e-3 (+545%), "
                    "momentum = 8.95e-3 still falling (artifacts/fvm-partial-traces.json)",
    },
    "repro_cmd": "cd /root/tp-agent/work/m16182 && PYTHONPATH=src python3 "
                 "scripts/run_case_blasius_plate.py --out /tmp/repro16204c --configs march,march-sensitivity"
                 "  # auditor ran it from an EMPTY out dir: 60.5 s, byte-identical to the committed artifacts"
                 " (runs/m16204/repro_compare.json: 3/3 files n_diffs=0)",
    "external_anchor": {
        "citation": "H. Blasius (1908) Z. Math. Phys. 56, 1-37; H. Schlichting, Boundary-Layer Theory, "
                    "7th ed., Sec.7 Table 7.1 (c_f*sqrt(Re_x)=0.664, delta*/x*sqrt(Re_x)=1.7208, "
                    "theta/x*sqrt(Re_x)=0.664, H=2.5916)",
        "auditor_independent_recompute": {
            "method": "own RK4 + bisection shooting, eta_max=14, h=1e-3, Simpson/trapezoid integrals "
                      "(runs/m16204/audit_probe.py::blasius_independent)",
            "two_fpp0": p1["two_fpp0"], "dev_vs_0.664": p1["dev_two_fpp0_vs_lit"],
            "delta_star_sqrt_re": p1["delta_star_sqrt_re"], "dev_vs_1.7208": p1["dev_delta_star_vs_lit"],
            "theta_sqrt_re": p1["theta_sqrt_re"], "dev_vs_0.664": p1["dev_theta_vs_lit"],
            "H": p1["H"], "dev_vs_2.5916": p1["dev_H_vs_lit"],
        },
        "verdict": "literature constants reproduced to <0.02% by independent code -> reference side is not "
                   "a transcription error; anchor certified for the march rung",
    },
    "auditor_probes": {
        "seed_independence": seeds,
        "seed_independence_note": "F6 refutation of the 'seeded with Blasius => circular' objection: four "
                                  "different initial profiles (default Blasius / linear / 1/7 power law / "
                                  "uniform) all land inside the 3% gate; H sits on the same -0.47% ny=256 "
                                  "discretisation plateau (-0.30% at ny=512)",
        "artifact_recompute": probe["P3_artifact_recompute"],
        "branch_not_merged": "git rev-parse main origin/main -> 0cd53fa5 both; cdbb3096 only on agent/m16182",
    },
    "findings": {
        "F1_high": "anchor-defining device (2-D incompressible SIMPLE, mesh_type='flat-plate') unvalidated: "
                   "75 outer iterations, cf=1.353e-2 vs 2.0998e-3 (+545%), momentum 8.95e-3 still falling; "
                   "delivery honestly marks it WITHHELD (artifacts/fvm-partial-traces.json); ~1000 iterations "
                   "= 1.2-2.5 h/grid level at 4.4-8.5 s/iteration",
        "F2_medium": "delivery is branch-only: main=0cd53fa5 is cdbb3096's parent, git ls-tree main has no "
                     "case_blasius_plate.py, zero importers -> not part of the shipped package",
        "F3_medium": "tensor_benchmark_cases has 0 rows for the anchor -> tolerance only in code/mission text",
        "F4_low": "mission checks line quotes `--configs march,march-sensitivity --summary-only`; --summary-only "
                  "is merge-only, so in an empty --out dir it yields rows=[], all_passed=false, exit 2 "
                  "(reproduced: runs/m16204/repro-clean/summary-summaryonly-only.json). REPORT.md's own repro "
                  "block omits the flag and is correct",
        "F5_low": "flat-plate pressure-correction tolerance hardcoded with no config exit: "
                  "src/tensorfvm/body_fitted.py:783-785 (threshold_floor=1e-8, relative_tolerance=1e-8) - "
                  "line-level confirmation of the delivery's F3 friction item; repo improvement, not this scope",
        "F6_low_refutation": "seed-independence probe rejects the circularity objection (see auditor_probes)",
        "F7_low": "leading-edge band Re_x <~1e3 shows up to 9.53% cf error (artifacts/figure-summary.json); "
                  "disclosed in REPORT.md Sec.5; comparison station 1e5 outside the band",
    },
    "level_gap_vs_L4": "L4 needs the anchor's stated device validated. Error analysis (ny 128->512 monotone "
                       "-1.81%->-0.15%), reproducibility (byte-identical clean-dir rerun in 60.5 s) and the "
                       "external anchor (independently recomputed to <0.02%) are all satisfied, but for the "
                       "march rung, not for the 2-D FVM. Recording L4 on CASE-FVM-* would assert that the FVM "
                       "discretisation is Blasius-validated, which it is not (+545%, unconverged).",
    "reopen_conditions": [
        "FVM rung converges to <3% on |dc_f| and |dH| at Re_x=1e5 -> raise this scope to L4 (after F5 tolerance exit)",
        "or re-scope the anchor to the march rung (e.g. CASE-BLE-BLASIUS-PLATE): its evidence is already L4-grade, "
        "no rerun needed",
    ],
    "checks_not_run": [
        "FVM flagship rerun to convergence (1.2-2.5 h/grid level; outside this audit's window)",
        "cluster execution", "independent re-implementation of march_boundary_layer",
        "independent verification of the per-face _surface_force identity (only the delivery test asserts <1e-12)",
        "third-party/experimental comparison beyond Blasius (none exists for a zero-pressure-gradient laminar plate)",
    ],
    "overstated_claims": [
        "mission result.checks quotes --summary-only as the passing repro command (F4)",
        "the anchor name CASE-FVM-* implies the FVM device was validated; REPORT.md itself does not claim it "
        "(explicit WITHHELD) - the gap is in the anchor naming, not in the report",
    ],
    "disposition": "approve with notes",
    "remediation": [
        "R1 rerun fvm-relaxation --max-iterations 1200 on the cluster after F5 gains a config exit -> then L4",
        "R2 re-scope anchor to the march rung to bank the existing L4-grade evidence at zero cost",
        "R3 register the anchor in tensor_benchmark_cases (metric |dc_f|/ref,|dH|/ref @Re_x=1e5; <3%)",
        "R4 merge agent/m16182 into main (F2)",
    ],
    "artifacts": [
        "docs/board_audit/m16204/CASE-FVM-BLASIUS-PLATE_AUDIT.md",
        "docs/board_audit/m16204/artifacts/audit_probe.py", "docs/board_audit/m16204/artifacts/audit_probe.json",
        "docs/board_audit/m16204/artifacts/repro_compare.py", "docs/board_audit/m16204/artifacts/repro_compare.json",
        "docs/board_audit/m16204/artifacts/repro-clean/ (4 files)",
        "docs/board_audit/m16204/artifacts/pytest_audit.log",
        "docs/board_audit/m16204/artifacts/verdict_evidence.json",
        "docs/board_audit/m16204/figures/fig1..fig3.png"
    ],
}

out = HERE / "verdict_evidence.json"
out.write_text(json.dumps(evidence, indent=2, ensure_ascii=False) + "\n")
print(f"wrote {out} ({out.stat().st_size} bytes)")

level = int(evidence["declared_level"])
payload = out.read_text()
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    "VALUES ('TensorFVM', 'm16182 CASE-FVM-BLASIUS-PLATE', 16204, %d, $jd$%s$jd$::jsonb, 'board-audit') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level = EXCLUDED.level, "
    "evidence = EXCLUDED.evidence, mission_id = EXCLUDED.mission_id, created_at = now() "
    "RETURNING id, repo, scope, mission_id, level, gate_state;" % (level, payload)
)
cmd = ["psql", "dbname=tensorplatform user=foundation host=/var/run/postgresql", "-v", "ON_ERROR_STOP=1", "-c", sql]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
print(res.stderr[-2000:] if res.returncode else "")
raise SystemExit(res.returncode)