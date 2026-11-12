"""Insert the m15813 cross-audit verdict for m15463 CASE-CHANNEL-PERIODIC.

Run:  python3 docs/reports/m15813_insert_verdict.py
Idempotent: re-running deletes the previous board-audit row for this scope first.
"""
from __future__ import annotations

import json
from pathlib import Path

import psycopg2

ROOT = Path(__file__).resolve().parent
AUDIT = json.loads((ROOT / "m15813_audit_m15463_verify.json").read_text())
FIG = json.loads((ROOT / "m15813_audit_m15463_figure_summary.json").read_text())

REPO = "TensorFVM"
SCOPE = "m15463 CASE-CHANNEL-PERIODIC"
MISSION_ID = 15813
LEVEL = 4

prod = next(r for r in AUDIT["rows"] if r["case"] == AUDIT["production_case"])
got = prod["registered_in_json"]
ind = prod["independent_checks"]

evidence = {
    "impl_files": [
        "src/tensorfvm/benchmark_poiseuille_channel.py:63 (TOLERANCE=0.02)",
        "src/tensorfvm/benchmark_poiseuille_channel.py:75 (PLAN, 6 predeclared cases)",
        "src/tensorfvm/benchmark_poiseuille_channel.py:196 (run_case -> SteadyFlow)",
        "src/tensorfvm/benchmark_poiseuille_channel.py:532 (argparse: case/all/plan-index/aggregate)",
        "merged in 653e63a6 'Merge #431 (m15373) via merge-master'",
    ],
    "entry": "PYTHONPATH=src python -m tensorfvm.benchmark_poiseuille_channel plan-index --index 0 "
             "(subcommands: case / all / plan-index / aggregate)",
    "run_artifacts": [
        "/root/tp-agent/work/m15463/runs/poiseuille-channel/benchmark.json (md5 e63581a7, untracked)",
        "/root/tp-agent/work/m15463/runs/poiseuille-channel/metrics.csv",
        "/root/tp-agent/work/m15463/runs/poiseuille-channel/profiles.csv",
        "/root/tp-agent/work/m15463/runs/poiseuille-channel/manifest.json (md5 bf5b47fa)",
        "docs/poiseuille-channel/{benchmark.json,metrics.csv,profiles.csv,manifest.json,figures/*.png} @653e63a6 (committed copy)",
        "absent: runs/poiseuille-channel/cases/ and failures/ (claimed by report section 6 -> F4)",
    ],
    "tests": [
        "tests/test_poiseuille_channel_benchmark.py -q -k 'not tiny_case' -> 5 passed, 1 deselected, 6.04 s (run in m15813 worktree)",
        "test_symbolic_derivation_reproduces_tabulated_targets (sympy: poiseuille=='96', peak_over_bar=='3/2', cf*Re_H==12)",
        "test_reference_values_are_the_exact_plane_channel_relations (dpdx=-0.12, tau_w=0.06, Re_dh=2Re_h)",
        "test_wall_shear_estimators_recover_the_parabola (rel=1e-13, asserts the half-cell first-order bias)",
        "test_observed_order_recovers_a_known_convergence_rate",
        "test_summarise_flags_the_finest_mesh_as_the_production_case (asserts coarse case must be False)",
    ],
    "integration_callers": [
        "src/tensorfvm/benchmark_poiseuille_channel.py:196 run_case -> tensorfvm.inc_bench.solver.SteadyFlow (src/tensorfvm/inc_bench/solver.py:129) = production SIMPLE/MAC kernel",
        "no CI / docs/benchmarks-index.md / registry wiring (grep hit only docs + tests) -> non-blocking recommendation",
    ],
    "benchmark": {
        "ref": {"poiseuille": 96.0, "dpdx": -0.12, "tau_w": 0.06, "u_max": 1.5},
        "got": {
            "velocity_l2": got["velocity_l2"],
            "velocity_linf": got["velocity_linf"],
            "u_max": got["u_max"],
            "pressure_gradient": got["pressure_gradient"],
            "tau_w_quadratic": got["tau_w_quadratic"],
            "poiseuille": got["poiseuille"],
            "flux_conservation_error": got["flux_conservation_error"],
        },
        "audit_recomputed": {
            "velocity_l2_from_profiles_csv": prod["recomputed_from_csv"]["velocity_l2"],
            "velocity_linf_from_profiles_csv": prod["recomputed_from_csv"]["velocity_linf"],
            "dpdx_from_raw_pressure_array": ind["dpdx"],
            "tau_w_from_steady_momentum_balance": ind["tau_w_from_momentum"],
            "poiseuille_from_momentum_balance": ind["poiseuille"],
            "max_relative_error_percent": 100 * max(
                got["velocity_l2"], got["velocity_linf"], got["u_max_relative_error"],
                got["pressure_gradient_relative_error"],
                got["tau_w_quadratic_relative_error"],
                got["poiseuille_relative_error"], ind["dpdx_rel"], ind["poiseuille_rel"]),
        },
        "tolerance": {"value": 0.02, "source": "tensor_case_pool id=37 case_code=CHANNEL-PERIODIC "
                                                  "tol_policy='<2%' repo=TensorFVM status=dispatched "
                                                  "linked_anchor=NULL; code TOLERANCE at "
                                                  "benchmark_poiseuille_channel.py:63"},
        "registry_gap": "tensor_benchmark_cases has 0 rows for anchor CASE-CHANNEL-PERIODIC and 0 rows "
                        "for CASE-POISEUILLE-FVM (22 rows total) -> F1",
        "grid_convergence_percent_L2": dict(zip([str(n) for n in FIG["mesh_ny"]],
                                                [round(v, 6) for v in FIG["velocity_l2_percent"]])),
        "observed_orders": AUDIT["observed_orders"],
        "production_passed": AUDIT["production_passed"],
        "all_cases_passed": AUDIT["all_cases_passed"],
        "honest_failures": ["mesh-ny8 dp/dx 2.832% > 2% gate (disclosed in report section 5)",
                           "re-1000 diverged (disclosed in API report 14548 failure modes)"],
    },
    "repro_cmd": [
        "cd <TensorFVM> && PYTHONPATH=src python -m tensorfvm.benchmark_poiseuille_channel "
        "plan-index --index 0 --out /tmp/m15813-repro  -> iterations=598, Po=93.51461, "
        "dpdx_rel=2.8319e-2, bit-identical to the m15463 artifact (run by this audit)",
        "python3 docs/reports/m15813_audit_m15463_verify.py --artifacts <m15463>/runs/poiseuille-channel",
        "PYTHONPATH=src python -m tensorfvm.benchmark_poiseuille_channel all  (full 6-case plan, ~8 min, not rerun here)",
    ],
    "external_anchor": {
        "literature": "Shah & London 1978, Laminar Flow Forced Convection in Ducts: plane channel f*Re_Dh = 96 "
                      "(circular tube 64); White, Viscous Fluid Flow section 3-3 for u=6 ubar y(H-y)/H^2",
        "reference_provenance": "sympy symbolic integration of mu*u''=-G with u(0)=u(H)=0 "
                                "(symbolic_reference(), reference_derivation.all_checks == true in benchmark.json)",
        "reports": "tensor_reports id=14548 (mission_id=15463, scenario_key=CASE-CHANNEL-PERIODIC, 0 figures); "
                   "repo report docs/poiseuille-channel/REPORT.md (sections 1-6 present)",
        "prior_audit": "docs/reports/m15491_audit_m15373.md (same implementation audited L4 for m15373)",
    },
    "level_rationale": "L4: real solver (SteadyFlow), full case w/ analytic ref + grid convergence + error analysis, "
                       "artifacts reproducible bit-identically, tests assert invariants. Not L5: standard case "
                       "verification, no new physics/numerics result.",
    "findings": {
        "F2_high_zero_filled_record": "runs/.../benchmark.json + metrics.csv record re-1000 with iterations=0, "
                                      "wall_time_s=0.0, all metrics exactly 0.0, converged=False while API report "
                                      "14548 attributes it to RuntimeError('non-finite SIMPLE iterate'); "
                                      "_run_or_record (line 508-520) should have written failures/re-1000.json and no "
                                      "cases/ file - neither directory exists. Not a fake PASS (passed=False) but "
                                      "the zero-filled row cannot evidence an Re sweep.",
        "F3_medium_no_commit": "git log main..agent/m15463 empty; result.head_commit=null; changed_files=[]; "
                               "7 artifacts untracked, ephemeral only. Committed evidence is the #431 docs/ copy whose "
                               "benchmark.json differs (md5 1243b9d2 vs e63581a7: docs has re-10/no re-1000, runs the "
                               "reverse). Declared plan = 6 cases, artifact covers 5 (re-10 missing).",
        "F1_medium_registry_silent": "tensor_benchmark_cases has no CASE-CHANNEL-PERIODIC row (the audit's designated "
                                     "referee table); tolerance only traceable to tensor_case_pool id=37 + code const.",
        "F4_low_report_artifact_mismatch": "docs REPORT.md section 6 lists runs/.../cases/*.json (absent); section 5.2 "
                                           "Re-sensitivity table has only the baseline row, omitting re-10 and re-1000.",
        "F5_low_anchor_drift": "repo report labels the anchor CASE-POISEUILLE-FVM while mission anchor is "
                               "CASE-CHANNEL-PERIODIC; neither slug is registered; pool uses CHANNEL-PERIODIC / "
                               "POISEUILLE-FVM (3 aliases for one implementation).",
    },
    "not_run": [
        "full 'all' plan replay (~8 min incl. the diverging Re=1000 leg)",
        "tests -k tiny_case (real-solve smoke, >115 s locally per m15491; cluster cj1152)",
        "GitHub CI (x86); all readings were taken on loongarch64",
    ],
    "artifact_paths": [
        "docs/reports/m15813_audit_m15463.md",
        "docs/reports/m15813_audit_m15463_verify.py",
        "docs/reports/m15813_audit_m15463_verify.json",
        "docs/reports/m15813_audit_m15463_figure.py",
        "docs/reports/figures/m15813_audit_m15463.png",
    ],
    "disposition": "approve with notes: L4 upheld on independently recomputed evidence; F2 (zero-filled re-1000 "
                   "record) needs artifact rework, F4 report tables need fixing; implementation code is unaffected.",
}

sql = list()
sql.append(
    "DELETE FROM tensor_maturity_verdicts WHERE repo='TensorFVM' "
    "AND scope='m15463 CASE-CHANNEL-PERIODIC' AND verdict_by='board-audit';")
payload = json.dumps(evidence, ensure_ascii=False)
sql.append(
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    f"VALUES ('{REPO}', '{SCOPE}', {MISSION_ID}, {LEVEL}, $json${payload}$json$, 'board-audit') "
    "RETURNING id, gate_state, level;")
sql_path = Path("/tmp/m15813_verdict.sql")
sql_path.write_text("\n".join(sql))
print(f"wrote {sql_path}")

import subprocess
proc = subprocess.run(["psql", "-U", "ship", "-d", "tensorplatform", "-tA", "-f", str(sql_path)],
                      capture_output=True, text=True)
print(proc.stdout.strip())
if proc.stderr.strip():
    print("STDERR:", proc.stderr.strip())
raise SystemExit(proc.returncode)
