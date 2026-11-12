#!/usr/bin/env python3
"""Land the m15456 board-audit verdict for m15331 CASE-DOUBLE-MACH.

Writes one row into tensor_maturity_verdicts; every evidence pointer is a real
file:line / artefact path / endpoint read during the audit.
"""
import json
import subprocess

EVIDENCE = {
    "impl_files": [
        "/root/tp-agent/work/m15331/src/tensorfvm/compressible/cases.py",
        "/root/tp-agent/work/m15331/src/tensorfvm/compressible/solver.py",
        "docs/audit-m15456/audit_double_mach.py",
        "docs/audit-m15456/audit_figure.py",
    ],
    "impl_notes": "audited branch agent/m15331 tip 0e2091f8: cases.py:500 _post_shock_state(), "
                  "cases.py:533 double_mach_reflection()+DoubleMachConfig; solver.py:60 "
                  "_resolve_bc_state() callable inflow BC, :204 resolve in _pad(), :296 "
                  "_resolve_bc_state_1d() 1-D tensor BC; those three edits are the solver-core "
                  "integration surface and are the code the audit reran",
    "entry": "/root/tp-agent/work/m15331/scripts/run_double_mach_reflection.py",
    "run_artifacts": [
        "/root/tp-agent/work/m15331/runs/double_mach_reflection/report.json",
        "/root/tp-agent/work/m15331/runs/double_mach_reflection/density_field.png",
        "/root/tp-agent/work/m15331/runs/double_mach_reflection/initial_condition.png",
        "docs/audit-m15456/dm_fields_120x30.npz",
        "docs/audit-m15456/fig_dm_audit.png",
        "docs/audit-m15456/audit_double_mach_120x30.json",
        "docs/audit-m15456/audit_double_mach_240x60.json",
        "docs/audit-m15456/dm_field_stats.json",
        "docs/audit-m15456/REPORT.md",
    ],
    "run_artifacts_notes": "delivered side: report.json (120x30 row only) + density_field.png "
                           "(1800x600) + initial_condition.png (2250x600), all untracked; "
                           "audit side: rho/p fields recomputed independently "
                           "(dm_fields_120x30.npz), figure (fig_dm_audit.png) and the JSONs the "
                           "numbers come from",
    "run_artifact_provenance": "report.json holds only a 120x30 row while the shipped driver's "
                               "main() runs 240x60/480x120/960x240 and writes density_nx*.npy + "
                               "double_mach_reflection.png (none present); the two PNGs that do "
                               "exist are never written by the shipped script -> artefacts and "
                               "repro_cmd do not corroborate each other. runs/ is entirely "
                               "untracked (git ls-files runs/ = 0), so it is not part of the "
                               "commit that gets pushed",
    "tests": "NONE: no test mentions double_mach in branch agent/m15331 or in main "
             "(grep -rn double_mach tests/ -> 0 hits); the new callable/1-D inflow BC in "
             "solver.py has zero test coverage",
    "integration_callers": [
        "tensorfvm.compressible.solver.EulerSolver used by the case (cases.py:569)",
        "not in main: main (worktree m15456) has 0 hits for double_mach; branch unmerged",
        "platform: GET /tensor/bench/results?mission_id=15331 -> total 0; "
        "GET /tensor/bench/l4-verdict/15331 -> CLAIMED_NOT_PROVEN; "
        "GET /tensor/reports?mission_id=15331 -> total 0 (tensor_reports row absent)",
    ],
    "benchmark": {
        "authoritative_judge": "SELECT ... FROM tensor_benchmark_cases WHERE "
                               "anchor_slug='CASE-DOUBLE-MACH' -> 0 rows (not registered)",
        "mission_internal_ref": {"max_density": 8.44, "mach_stem_height": 0.55,
                                 "triple_point_x": 1.5, "tolerance": 0.08,
                                 "provenance": "no fetched source / sha256 / page image in repo "
                                               "-> WITHHELD (unverified)"},
        "ref_vs_got": {
            "max_density": {"ref": 8.44, "got_reported": 6.3608, "got_audited": 6.3612,
                            "err": 0.246, "pass": False},
            "mach_stem_height": {"ref": 0.55, "got_reported": 0.9833,
                                 "note": "got = y of global max(rho) = top-BC injection corner "
                                         "(0.0167,0.9833); feature never measured",
                                 "err": 0.788, "pass": False},
            "triple_point_x": {"ref": 1.5, "got_reported": 0.0167,
                               "note": "got = x of global max(rho), "
                                       "same boundary corner -> detector invalid",
                               "err": 0.989, "pass": False},
        },
        "grid_trend": {"120x30": {"max_rho": 6.3612, "wall_row_max": 3.484,
                                  "wall_row_mean": 2.518, "max_p": 110.8},
                       "240x60": {"max_rho": 6.3556, "wall_row_max": 3.487,
                                  "wall_row_mean": 2.520, "max_p": 111.9},
                       "note": "2x refinement changes max rho by 0.09% -> the value is the "
                               "smeared top-BC datum, not a physical peak; wall row mean 2.5 "
                               "< post-shock 8.10 = gas driven away from wall"},
        "counterfactual_textbook_datum": {"120x30": {"max_rho_window": 18.577,
                                                     "wall_row_mean": 11.851,
                                                     "max_p": 581.0},
                                          "note": "same solver/grid, only the post-shock datum "
                                                  "replaced by M_n=10 -> wall compression and "
                                                  "reflected-shock structure appear"},
        "verdict": "FAIL: 3/3 metrics outside the 8% tolerance, and 2 of the 3 metrics are "
                   "not physical measurements",
    },
    "repro_cmd": "PYTHONPATH=/root/tp-agent/work/m15331/src OMP_NUM_THREADS=16 python3 "
                 "docs/audit-m15456/audit_double_mach.py --out /tmp/dm120 --reported "
                 "/root/tp-agent/work/m15331/runs/double_mach_reflection/report.json "
                 "(audit-side, reproduced max rho 6.3612 vs reported 6.3608, delta 0.0004); "
                 "delivered side: PYTHONPATH=.../m15331/src python3 "
                 "scripts/run_double_mach_reflection.py (runs 240/480/960, does NOT produce "
                 "report.json's 120x30 row nor the two PNGs present in runs/)",
    "external_anchor": "docs/double_mach_reflection_report.md cites Woodward & Colella (1984) "
                       "JCP 54(1) 115-173 Sec.4, but no fetched document / sha256 / page image "
                       "exists in the repo (contrast docs/woodward-colella-double-blast/"
                       "provenance.md) -> citation only, not an anchor",
    "audit_findings": [
        "BLOCKING: _post_shock_state() is self-contradictory: vs = mach*c1/sin(theta) = 11.547 "
        "and mn = mach/sin(theta) = 11.547 cannot both hold; returned p=155.389 implies normal "
        "Mach 11.547 (audit reverse-solve: 11.5469) vs the cited M=10 -> post-shock pressure "
        "+33.4% (116.500 is the M=10 value)",
        "BLOCKING: post-shock velocity formula wrong: got (u,v) = (3.276, +4.775), |v| = 5.79 = "
        "V*cos(60) (the upstream tangential speed in the shock frame); independent front-frame/"
        "tangential projection gives (7.145, -4.125), |v| = 8.250. The y sign is flipped, so the "
        "gas is driven AWAY from the wall and the wall reflection that defines double Mach "
        "reflection never happens (audit: wall-row max rho 3.484, no cell > 4 on the wall; "
        "wall_shock_x = NaN)",
        "HIGH: top_bc() sweeps at cfg.mach*c1 = 10.0 while the same file uses 11.547 as the "
        "front speed -> prescribed top-BC front lags 0.309 in x at t=0.2 (7.7% of the domain)",
        "HIGH: metrics are argmax coordinates (find_mach_stem_height returns yc[argmax], "
        "xc[argmax]); triple_point_x=0.0167 / mach_stem_height=0.9833 are both the top-left "
        "injection corner -> the report's 98.9%/78.8% 'errors' measure the detector, and the "
        "attribution to 'MUSCL-Hancock diffusion' is unsupported",
        "HIGH: reference triple 8.44/0.55/1.5 has no provenance artefact in the repo "
        "(mission goal demanded zero-hand-copy ref) -> WITHHELD; 8.44 is only 5.5% above the "
        "post-shock rho2=8.0, inconsistent with the strong wall recompression the geometry "
        "requires (audit window max rho 18.58, max p 581 with a correct datum)",
        "MEDIUM: artefact/repro mismatch: report.json holds only a 120x30 row while the shipped "
        "driver only runs 240x60/480x120/960x240 and writes density_nx*.npy + "
        "double_mach_reflection.png, none of which exist; the two PNGs present "
        "(density_field.png, initial_condition.png) are never written by the shipped script and "
        "report.json lacks the keys the script always writes (best/dx/dy/elapsed_seconds/drift_*)",
        "MEDIUM: run_artifacts are untracked (git ls-files runs/ = 0; git status shows '?? runs/'); "
        "tip 0e2091f8 contains only cases.py, solver.py, the driver and a md -> artefacts vanish "
        "on push/merge",
        "MEDIUM: not integrated with the platform: no tensorbench_results row, L4 endpoint "
        "CLAIMED_NOT_PROVEN, no tensor_reports row for 15331 (report exists only as md)",
        "LOW: branch is behind main (cases.py lacks the double-blast datum now on main; solver.py "
        "lacks main's embedded-solid mask) -> a plain merge would regress existing capability",
        "LOW: zero tests anywhere for the case or for the new callable/1-D inflow BC",
    ],
    "what_would_raise_the_level": {
        "L3": "merge/rebase the branch onto main without dropping double-blast/solid-mask code, "
              "add >=1 pytest (initial datum RH consistency + short smoke run) and register the "
              "case in the verification catalog",
        "L4": "fix the datum (M_n=10: rho=8.0, p=116.5, u=7.145, v=-4.125), fix the top-BC sweep "
              "speed, replace argmax metrics by feature detection, run 480x120 within tolerance, "
              "commit runs/ artefacts and bind a cluster job id",
    },
    "verdict_rationale": "L2: real code with input->compute->output verified by an independent "
                         "rerun that reproduces max rho to 0.006% and saves rho/p fields. L3 denied: "
                         "the integration surface (EulerSolver + a solver.py BC patch) is unmerged, "
                         "untested, unregistered and has no platform benchmark row. L4 denied by "
                         "three independent reasons: 3/3 metrics outside the 8% tolerance, two of "
                         "the three metrics are not physical measurements, and the case datum does "
                         "not reproduce the cited Woodward-Colella setup (post-shock pressure +33.4%, "
                         "velocity direction inverted so no wall reflection occurs).",
}

SCOPE = "m15331 CASE-DOUBLE-MACH"
REPO = "TensorFVM"
MISSION = 15456
LEVEL = 2
BY = "board-audit"

ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    f"VALUES ('{REPO}', '{SCOPE}', {MISSION}, {LEVEL}, '{ev}'::jsonb, '{BY}') "
    "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
    "mission_id=EXCLUDED.mission_id, evidence=EXCLUDED.evidence, gate_state='pending', "
    "gate_level=NULL, gate_note=NULL"
)
r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-c", sql],
                   capture_output=True, text=True)
print(r.stdout[-300:], r.stderr[-500:])
r2 = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "|", "-c",
                     "SELECT id, repo, scope, mission_id, level, verdict_by, coalesce(gate_state,'pending'), "
                     "coalesce(gate_level,-1), (SELECT count(*) FROM jsonb_object_keys(evidence)) "
                     f"FROM tensor_maturity_verdicts WHERE scope='{SCOPE}'"], capture_output=True, text=True)
print(r2.stdout, r2.stderr[-300:])