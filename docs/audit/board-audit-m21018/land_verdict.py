#!/usr/bin/env python3
"""Land the board-audit verdict for m21018 (GAP-DIFFLBM-EXT-LIT-CYL / TensorFVM).

Inserts one idempotent row into ``tensor_maturity_verdicts`` of the platform
database (postgres ``tensorplatform``, the table ``maturity_gate.py`` reads),
then prints the row back plus the machine-judge replay of the same evidence so
the audit is self-checking.

Level declared: 2 -- independent functional implementation.
  * L2 holds: real code (input -> compute -> output), entry point executed by
    the audit, non-empty run artifacts from a real compute.
  * L3 does NOT hold: the new modules (``tensorfvm.immersed.forces``,
    ``shedding_statistics``) are called only from the case harness
    ``tensorfvm/verification/cylinder_re100.py`` and the tests -- the
    production immersed solver never calls them, ``backend_registry.py`` has no
    immersed entry and no platform route exposes the backend.
  * L4 does NOT hold: both registered benchmark metrics are out of tolerance
    (cd_mean +12.63 % vs 5 %, st -5.44 % vs 3 %, status 'pending') and the
    finest ladder grid never produced physics (non-finite from step ~15).

Usage: python3 docs/audit/board-audit-m21018/land_verdict.py [--dry-run]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCOPE = "m21018 GAP-DIFFLBM-EXT-LIT-CYL"
MISSION_ID = 21105
REPO = "TensorFVM"
LEVEL = 2

EVIDENCE = {
    "impl_files": [
        "src/tensorfvm/immersed/forces.py",
        "src/tensorfvm/immersed/geometry.py",
        "src/tensorfvm/verification/cylinder_re100.py",
        "docs/case-cylinder-re100/ladder.json",
    ],
    "impl_notes": "forces.py (211 lines, added by PR #719): ControlBox:43, budget_terms:116, "
                  "ForceRecorder:175 -- control-volume momentum budget. geometry.py +41 lines: "
                  "circular_cylinder staircase mask + cylinder_mask_area_error. "
                  "verification/cylinder_re100.py:164 shedding_statistics (half-amplitude gate, "
                  "zero-crossing St + FFT cross-check), :340 build_immersed, :726 CLI dispatch. "
                  "docs/case-cylinder-re100/ladder.json = the grid-ladder record rendered into "
                  "docs/case-cylinder-re100/README.md by scripts/case_cylinder_re100.py aggregate.",
    "entry": "src/tensorfvm/verification/cylinder_re100.py",
    "entry_cmd": "PYTHONPATH=src python -m tensorfvm.verification.cylinder_re100 --backend "
                 "immersed --cells-per-diameter {12,20,32} --dt 0.02 --t-end 120 --output <dir>; "
                 "scripts/case_cylinder_re100.py ladder|aggregate. Audit ran `--help` (ok, args "
                 "match the documented repro) and the CLI end-to-end at 8 cells/D: 1200 steps, "
                 "24.4 s, cd_mean 1.5334 (+14.43 %), verdict GATED -- same family as cpd12/20.",
    "run_artifacts": [
        "docs/audit/board-audit-m21018/artifacts/cpd12/result.json",
        "docs/audit/board-audit-m21018/artifacts/cpd20/result.json",
        "docs/audit/board-audit-m21018/artifacts/cpd32/result.json",
        "docs/audit/board-audit-m21018/artifacts/cpd20/force_history.csv",
        "docs/audit/board-audit-m21018/artifacts/cpd32/force_history_head.csv",
        "docs/audit/board-audit-m21018/artifacts/manifest.json",
    ],
    "run_artifact_notes": "cpd12: cd_mean 1.51931, st 0.14415 (FFT 0.14411), 6001 samples; cpd20: "
                          "cd_mean 1.50926, st 0.15508 (FFT 0.15512), 10 periods; cpd32: cd_mean "
                          "nan, shedding_detected false. The mission left runs/ untracked, so the "
                          "copies here (sha256 in manifest.json) are the durable record; "
                          "cpd20/force_history.csv is the full raw signal behind the benchmark.",
    "tests": [
        "tests/test_cylinder_re100_case.py",
        "tests/test_case_cylinder_re100.py",
        "docs/audit/board-audit-m21018/repro_log.md",
        "16 passed in 24.4 s (audit rerun 2026-11-15, /srv/tensor/agent/work/m21018, PYTHONPATH=src). "
        "Invariants asserted, not just execution: force control-volume independence, mask area "
        "convergence to the circle, St recovered from a synthetic shedding signal, St withheld "
        "when nothing sheds, short window flagged, reference file bound to module constants. "
        "GAP: no test feeds a non-finite history -- exactly the cpd32 failure mode.",
    ],
    "integration_callers": "ImmersedFlow:src/tensorfvm/verification/cylinder_re100.py:345 imports and "
                           "runs the m9040 production solver tensorfvm.immersed.solver.ImmersedFlow "
                           "and wraps tensorfvm.immersed.forces.ForceRecorder. Honest scope: the new "
                           "force/statistics modules have no caller outside the case harness and the "
                           "tests; src/tensorfvm/backend_registry.py has 0 immersed hits and no "
                           "platform route (/calculations/*) exposes the immersed backend, so "
                           "system integration (L3) is NOT met.",
    "benchmark": "tensor_benchmark_cases anchor_slug=GAP-DIFFLBM-EXT-LIT-CYL (postgres "
                 "tensorplatform, case cylinder-re100-unbounded, repo TensorFVM, status still "
                 "'pending'): cd_mean tol 5% -> got 1.5093 vs ref 1.3400 = +12.63% FAIL; st tol 3% "
                 "-> got 0.1551 vs 0.1640 = -5.44% FAIL; Cl amplitude 0.2005 vs 0.32 = -37.3% "
                 "(informational). Finest grid 32 cells/D: cd_mean nan, force budget non-finite "
                 "from step ~15 -> no grid-convergence claim. Every grid FAILS -> L4 not met.",
    "repro_cmd": "OMP_NUM_THREADS=4 PYTHONPATH=src python -m tensorfvm.verification."
                 "cylinder_re100 --backend immersed --cells-per-diameter 12 --dt 0.02 --t-end 120 "
                 "--output runs/case-cylinder-re100/cpd12 && python "
                 "scripts/case_cylinder_re100.py aggregate --root runs/case-cylinder-re100 --out "
                 "docs/case-cylinder-re100  (audit: reduced 8 cells/D instance re-run locally, "
                 "step-1 force of the 32 cells/D grid reproduced bit-for-bit 1552.09585999; full "
                 "12 cells/D rerun submitted as compute job cj3192)",
    "external_anchor": "docs/case-cylinder-re100/reference/reference.json",
    "external_anchor_notes": "PR #719 merge-master commit cda4e7f9 (subject branch 69311e24); "
                             "literature Braza et al. JFM 165 (1986) 79-130 "
                             "doi:10.1017/S0022112086003014 and Schaefer & Turek NNFM 52 (1996) "
                             "doi:10.1007/978-3-322-89849-4_39.",
    "verdict_notes": "Honest-FAIL delivery (docs say FAIL, no number dressed up) -> L2. Blocking "
                     "defect: the shedding gate at verification/cylinder_re100.py:200 maps a "
                     "non-finite lift signal onto 'no vortex shedding', so a numerical blow-up is "
                     "reported as a physics gate and ~2639 s were spent stepping a NaN field on the "
                     "finest grid. GAP-DIFFLBM-EXT-LIT-CYL stays open: both registered metrics miss "
                     "tolerance and there is no convergence evidence.",
}



def psql(sql: str) -> str:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-tA", "-F",
                        "\x01", "-v", "ON_ERROR_STOP=1", "-c", sql],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"psql failed: {r.stderr.strip()}")
    return r.stdout


def esc(s: str) -> str:
    return s.replace("'", "''")


def main(argv=None) -> int:
    dry = "--dry-run" in (argv if argv is not None else sys.argv[1:])
    payload = json.dumps(EVIDENCE, ensure_ascii=False)
    exists = psql("SELECT count(*) FROM tensor_maturity_verdicts WHERE scope='%s' "
                  "AND verdict_by='board-audit'" % esc(SCOPE)).strip()
    if exists and int(exists) > 0:
        print(f"already landed ({exists} row(s)) -- idempotent no-op")
    else:
        sql = ("INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, "
               "verdict_by) VALUES ('%s', '%s', %d, %d, '%s'::jsonb, 'board-audit')"
               % (REPO, esc(SCOPE), MISSION_ID, LEVEL, esc(payload)))
        if dry:
            print(sql[:400] + " ...")
        else:
            psql(sql)
            print("INSERT ok")
    row = psql("SELECT id, repo, scope, mission_id, level, verdict_by, gate_state, gate_level "
               "FROM tensor_maturity_verdicts WHERE scope='%s' ORDER BY id DESC LIMIT 1" % esc(SCOPE))
    print("row:", row.strip())
    print("evidence keys:", sorted(EVIDENCE))
    if "--judge" in sys.argv[1:]:
        sys.path.insert(0, "/srv/tensor/agent")
        import maturity_gate as mg
        gl, why = mg.judge(LEVEL, EVIDENCE, REPO, MISSION_ID)
        print(f"maturity_gate replay: gate_level={gl} claim={LEVEL} why={why}")
    return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())