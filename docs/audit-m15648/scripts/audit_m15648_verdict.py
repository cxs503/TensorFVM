#!/usr/bin/env python3
"""m15648: write the board-audit verdict for m15542 (CASE-FORWARD-STEP, TensorFVM).

- builds evidence pointers from docs/audit-m15648/evidence_verification.json (no hand typing)
- INSERT ... ON CONFLICT into tensor_maturity_verdicts (gate_state left pending for maturity_gate)
- registers the missing platform referee row in tensor_benchmark_cases (status=pending, no pass claim)
Writes runs/audit_m15648/{verdict_payload.json,insert_verdict.sql,register_case.sql}
"""
import json
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parents[1]      # docs/audit-m15648
ROOT = HERE.parents[1]                                   # worktree root
RUN = ROOT / "runs" / "audit_m15648"
RUN.mkdir(parents=True, exist_ok=True)
EV = json.loads((HERE / "evidence_verification.json").read_text())
CL = EV["graded_delivered_grid"]
CF = EV["graded_finer_grid_uncommitted"]
W = pathlib.Path("/root/tp-agent/work/m15542")
REV = "docs/audit-m15648/reviewed"

bench = {
    "ref": "closed-form Rankine-Hugoniot / isentropic state for gamma=1.4, M1=3 (WC1984 problem 2 publishes contours, not tables)",
    "got_120x40_delivered": {k: {"reference": v["reference"], "got": v["got"], "rel_error": v["rel_error"], "pass": v["pass"]}
                             for k, v in CL["items"].items()},
    "got_240x80_uncommitted": {k: {"reference": v["reference"], "got": v["got"], "rel_error": v["rel_error"], "pass": v["pass"]}
                               for k, v in CF["items"].items()},
    "pass_count": {"120x40": f"{CL['pass_count']}/{CL['items_total']}", "240x80": f"{CF['pass_count']}/{CF['items_total']}"},
    "worst_error": {"120x40": "M2 -12.66 %", "240x80": "M2 -26.28 %"},
    "error": "delivered grid misses 2 of 6 pre-registered 8 % criteria (u2/u1 -11.53 %, M2 -12.66 %); "
             "refinement does not converge to the anchor (all four shock-jump errors grow), "
             "and the mass identity breaks 5e-16 -> 8.34e-2 between the two grids",
    "grid_of_record": "480x160 declared, never produced (no artifact in any tree)",
}

evidence = {
    "impl_files": [
        f"{REV}/src/tensorfvm/compressible/forward_step_benchmark.py",
        f"{REV}/src/tensorfvm/compressible/forward_step.py",
        f"{REV}/src/tensorfvm/compressible/solver.py",
        f"{REV}/src/tensorfvm/compressible/verification.py",
        f"{REV}/scripts/forward_step_study.py",
    ],
    "impl_files_note": (
        "byte-identical copies (sha256 in docs/audit-m15648/reviewed/SHA256SUMS.txt, verified by "
        "audit_m15648_extract.py: reviewed_copy_all_identical=true) of the audited commit "
        "054dbb84acbebcf7f0aebc7d7e3653d4c2aa8a20; the branch agent/m15542 is NOT merged "
        "(TensorFVM main 4704c5a8 has no forward_step_benchmark.py), so the sources cannot be "
        "cited at their in-repo paths yet.  Key sites: forward_step_benchmark.py:99-101 (graded "
        "criteria+tolerances), :143 grade_run, :193 convergence, :279 CLI; forward_step.py:264 "
        "mass_drift identity; solver.py (positivity floors + counted troubled-cell repair); "
        "verification.py:655 leg_forward_step, :760 run_all registration."
    ),
    "entry": f"{REV}/scripts/forward_step_study.py",
    "entry_note": (
        "audited entries: `python -m tensorfvm.compressible.forward_step_benchmark --runs <dir>` "
        "(regrade, re-executed by this audit: reproduces the committed grading.json "
        "all_graded_pass/per_grid/convergence exactly) and "
        "`python scripts/forward_step_study.py --out <dir> --grids 120x40 --t-end 4` (solve + field dump)."
    ),
    "run_artifacts": [
        str(W / "docs/repro-fvm-forward-step/run/grading.json"),
        str(W / "docs/repro-fvm-forward-step/run/run_120x40.json"),
        str(W / "docs/repro-fvm-forward-step/run/fields_120x40.npz"),
        str(W / "docs/repro-fvm-forward-step/run/table.md"),
        str(W / "docs/repro-fvm-forward-step/REPORT-v2.md"),
        str(W / "runs/forward_step/grading.json"),
        str(W / "runs/forward_step/fields_240x80.npz"),
        str(HERE / "evidence_verification.json"),
        str(HERE / "regrade_summary.json"),
        str(HERE / "figures/fig1_errors_vs_grid.png"),
        str(HERE / "figures/fig2_conservation_and_passcount.png"),
    ],
    "tests": [
        "tests/test_forward_step_benchmark.py::test_exact_field_grades_green",
        "tests/test_forward_step_benchmark.py::test_wrong_jump_grades_red",
        "tests/test_forward_step_benchmark.py::test_budget_identity_fails_when_drift_is_large",
        "tests/test_forward_step_benchmark.py::test_post_shock_sample_skips_the_transition_cell",
        "tests/test_forward_step_benchmark.py::test_troubled_repair_restores_cell_and_skips_solid",
        "tests/test_forward_step_benchmark.py::test_convergence_recovers_first_order",
        "tests/test_repro_fvm_forward_step.py (8 tests incl. test_reference_matches_textbook_relations, "
        "test_wall_is_impermeable_and_conservative) -- 14 passed in 14.89 s, re-run by this audit "
        "in the audited worktree (PYTHONPATH=src python -m pytest ...)",
    ],
    "integration_callers": [
        "tensorfvm.compressible.verification.leg_forward_step (verification.py:655; registered in run_all at :760)",
        "python -m tensorfvm.compressible --legs forward_step (__main__.py legs dispatch)",
    ],
    "integration_note": (
        "in-branch integration only: main 4704c5a8 lacks leg_forward_step, so a grep on main finds no "
        "caller -> maturity_gate may recompute L2; that is the unmerged-branch convention, not an overclaim. "
        "The leg itself grades 240x80 by default and its all_pass is False (honest fail)."
    ),
    "benchmark": bench,
    "repro_cmd": (
        "cd <TensorFVM m15542 tree> && PYTHONPATH=src python scripts/forward_step_study.py "
        "--out /tmp/fs120 --grids 120x40 --t-end 4 --threads 8 && PYTHONPATH=src python -m "
        "tensorfvm.compressible.forward_step_benchmark --runs /tmp/fs120 --json"
    ),
    "repro_note": (
        "the repro_cmd recorded inside the delivered grading.json points at scripts/repro_fvm_forward_step.py, "
        "which writes <out>/artifacts/forward_step.json + forward_step_fields.npz -- NOT the "
        "fields_<tag>.npz/run_<tag>.json the grader consumes (forward_step_benchmark.load_run) => following "
        "that command cannot regenerate the graded artifacts (finding F2). The regrade half of the chain was "
        "re-executed by this audit and is bit-identical to the committed grading.json."
    ),
    "external_anchor": (
        "Woodward & Colella, J. Comput. Phys. 54 (1984) 115-173, problem 2 (pp.161-166, figs.6-7: t=4 density "
        "contours only, no tables); the graded values are closed-form Rankine-Hugoniot/isentropic relations for "
        "M1=3, gamma=1.4 recomputed and checked by this audit (10.3333 / 3.85714 / 0.259259 / 0.475191 / "
        "12.0610 / 4.30749) -- exact, but their applicability to the 2D probe at y=0.1 is unproven: the graded "
        "errors grow with refinement, which is what a mis-placed probe looks like."
    ),
    "findings": {
        "F1_refinement_nonconvergent": {
            "severity": "blocking (L4)",
            "evidence": "graded errors vs grid, from artifact JSONs: u2/u1 -11.53 %->-23.74 %, M2 -12.66 %->-26.28 %, "
                        "p2/p1 +1.30 %->-9.28 %, rho2/rho1 -1.29 %->-15.23 % (120x40 -> 240x80); pass count 6/8 -> 2/8",
            "why": "the mission's stated cause of the miss ('120x40 under-resolved') is falsified: refinement moves the "
                   "graded state away from the closed-form anchor; the post-shock probe (x_s + 1 cell / 0.02 window) or "
                   "the anchor's applicability at y=0.1 is wrong, not the resolution",
        },
        "F2_repro_cmd_mismatch": {
            "severity": "blocking (L4: reproducibility)",
            "evidence": "grading.json case.repro_cmd -> scripts/repro_fvm_forward_step.py:315/321 writes artifacts/forward_step.json; "
                        "forward_step_benchmark.load_run reads fields_<tag>.npz + run_<tag>.json",
            "why": "the recorded command cannot regenerate the graded artifacts; DESIGN-ADDENDUM-r1 documents the correct "
                   "chain (forward_step_study.py) but the machine-readable field was not updated",
        },
        "F3_mass_identity_breaks": {
            "severity": "high (correctness of the solver repair path)",
            "evidence": "mass_budget_drift 5.04e-16 (120x40, 147 positivity fixes) -> 8.34e-2 (240x80, 19 fixes); "
                        "energy drift 2.86e-3 -> -2.92e-2",
            "why": "same code and geometry, 15 orders of magnitude of conservation loss on a refined grid; repair count "
                   "falls while the drift explodes, so the 'repair injects mass' story is a hypothesis, not a diagnosis "
                   "-- unknown root cause, flagged for rework",
        },
        "F4_grid_of_record_never_run": {
            "severity": "high (process)",
            "evidence": "no *480* artifact in the delivered tree, in runs/, or in this audit tree; grading.json "
                        "is_grid_of_record=false; mission result fail_stage=grid_of_record_not_run",
            "why": "the pre-registered benchmark verdict is unreachable by construction; the mission nevertheless closed "
                   "as succeeded",
        },
        "F5_benchmark_not_registered": {
            "severity": "medium (platform ledger)",
            "evidence": "SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FORWARD-STEP' -> 0 rows "
                        "(19 rows exist for other anchors, incl. TensorFVM REPRO-FVM-SOD1978 / CASE-CAVITY-RE1000-FVM)",
            "why": "the tolerance surface existed only in the repo's own grading.json, i.e. the author's own referee; "
                   "fixed by this audit: one pending row registered from the pre-registered DESIGN criteria",
        },
        "F7_same_config_not_reproducible": {
            "severity": "blocking (reproducibility)",
            "evidence": "audit re-run of the identical 120x40/t=4 config (16 threads, 166.8 s, "
                        "docs/audit-m15648/repro_run/): nsteps 2834->2829, positivity fixes 147->2, "
                        "mass drift -5.04e-16->+4.56e-4, energy drift 2.86e-3->-1.26e-4, p2/p1 10.4678->10.2218, "
                        "u2/u1 0.2294->0.2219, M2 0.4150->0.4020; regraded pass 6/8 -> 5/8",
            "why": "the delivered grading depends on thread count / float reduction order; a single run is not a "
                   "reproducible evidence unit and the 5e-16 mass identity is run-specific, not a property of the scheme",
        },
        "F6_delivered_vs_disk": {
            "severity": "medium (evidence hygiene)",
            "evidence": "git status in m15542: ' M docs/repro-fvm-forward-step/REPORT-v2.md', '?? runs/' -- the 240x80 "
                        "report version and the 240x80 fields/grading are NOT in commit 054dbb84",
            "why": "the self-falsifying finer-grid evidence stayed on disk; the delivered tree shows only the more "
                   "favourable 120x40 table",
        },
    },
    "level_rationale": (
        "L3: real code (input=geometry+inflow -> compute=MUSCL-Hancock/HLLC solve + closed-form jump extraction -> "
        "output=grading.json/fields/figures), engineering entry, 14 unit tests re-run green, in-repo integration "
        "(verification.py leg + run_all + __main__). NOT L4: grid of record never run, 2 of 6 pre-registered 8 % "
        "criteria miss, errors grow under refinement, and the recorded repro_cmd does not regenerate the graded "
        "artifacts. NOT L2/L1 only: implementation, artifacts and tests are all genuine. 100-page breadth is "
        "irrelevant; level = functional substance."
    ),
    "verdict_note": (
        "Non-redo audit of m15542 at 054dbb84. Board-audit disposition: request changes -- do NOT book this as a "
        "validated benchmark. Three rework items: (a) run 480x160 grid of record; (b) root-cause the 8.34e-2 mass "
        "drift at 240x80; (c) fix the probe/anchor applicability and point repro_cmd at forward_step_study.py."
    ),
}


_evsrc = json.loads((HERE / "evidence_verification.json").read_text())
evidence["not_run"] = _evsrc["not_run"]
evidence["independence_limitations"] = _evsrc["independence_limitations"]
evidence["independent_rerun_120x40"] = _evsrc["independent_rerun_120x40"]

payload = {"repo": "TensorFVM", "scope": "m15542 CASE-FORWARD-STEP", "mission_id": 15648,
           "level": 3, "evidence": evidence, "verdict_by": "board-audit"}
(RUN / "verdict_payload.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")

ev_json = json.dumps(evidence, ensure_ascii=False).replace("$json$", "\\u0024json\\u0024")
sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) VALUES "
       "('TensorFVM', 'm15542 CASE-FORWARD-STEP', 15648, 3, $json$" + ev_json + "$json$::jsonb, 'board-audit') "
       "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level=EXCLUDED.level, "
       "evidence=EXCLUDED.evidence, mission_id=EXCLUDED.mission_id RETURNING id, gate_state;")
(RUN / "insert_verdict.sql").write_text(sql + "\n")

case_sql = """
INSERT INTO tensor_benchmark_cases (anchor_slug, case_name, ref_source, metric, tolerance, repo, status)
SELECT 'CASE-FORWARD-STEP',
       'Mach 3 wind tunnel with a forward-facing step (Woodward & Colella 1984 problem 2; 3x1 tunnel, step h=0.2 at x=0.6, gamma=1.4, rho=1.4, p=1, t=4)',
       'Woodward & Colella, J. Comput. Phys. 54 (1984) 115-173 problem 2 (t=4 density contours only); graded values = closed-form Rankine-Hugoniot / isentropic state for M1=3 recomputed in forward_step_benchmark._reference',
       'six closed-form criteria (p2/p1, rho2/rho1, u2/u1, M2, p_stag/p1, rho_stag/rho1) + mass drift + energy drift; graded grid of record 480x160',
       'each closed-form criterion <8% (pre-registered in docs/repro-fvm-forward-step/DESIGN.md); mass drift <1e-6; energy drift <1e-2',
       'TensorFVM', 'pending'
WHERE NOT EXISTS (SELECT 1 FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FORWARD-STEP');
"""
(RUN / "register_case.sql").write_text(case_sql)


def psql_file(p):
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1", "-f", str(p)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


if __name__ == "__main__":
    for f in ("insert_verdict.sql", "register_case.sql"):
        rc, out, err = psql_file(RUN / f)
        print(f, "rc=", rc, "out=", out, "err=", err)