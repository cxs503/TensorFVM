#!/usr/bin/env python3
"""Land the board-audit verdict for m15026 (GAP-FVM-TWOPHASE) into
tensor_maturity_verdicts. Idempotent: same (repo, scope, verdict_by) overwritten.
"""
import json
import subprocess

EVIDENCE = {
    "impl_files": [
        "src/tensorfvm/twophase_gap/__init__.py",
        "src/tensorfvm/twophase_gap/__main__.py",
        "src/tensorfvm/twophase_gap/campaign.py",
        "src/tensorfvm/twophase_gap/zalesak_gate.py",
        "src/tensorfvm/twophase_gap/dambreak_gate.py",
    ],
    "entry": "src/tensorfvm/twophase_gap/__main__.py (python -m tensorfvm.twophase_gap; console script tensorfvm-twophase-gap)",
    "run_artifacts": [
        "docs/GAP-FVM-TWOPHASE/artifacts/gap_twophase.json",
        "docs/GAP-FVM-TWOPHASE/artifacts/leg_zalesak_plic_128.json",
        "docs/GAP-FVM-TWOPHASE/artifacts/fields_zalesak_plic_128.npz",
        "docs/GAP-FVM-TWOPHASE/artifacts/leg_dambreak_plic_240x80.json",
        "docs/GAP-FVM-TWOPHASE/artifacts/figures/fig2_zalesak_geometric_error.png",
    ],
    "tests": [
        "tests/test_twophase_gap.py::test_zalesak_leg_reports_shape_and_conservation",
        "tests/test_twophase_gap.py::test_cicsam_leg_is_conservative",
        "tests/test_twophase_gap.py::test_dambreak_leg_matches_martin_moyce_and_control",
        "tests/test_twophase_gap.py::test_gate_evaluation_is_a_function_of_artefacts",
        "tests/test_twophase_gap.py::test_solver_self_check_uses_installed_scheme",
        "run: PYTHONPATH=src pytest tests/test_twophase_gap.py -q -> 8 passed in 64s (audit m15190)",
    ],
    "integration_callers": [
        "twophase_gap.zalesak_gate:solver.py:130",
        "src/tensorfvm/vof_phase0/solver.py:113 verify_interface_transport()",
    ],
    "benchmark": {
        "source": "tensor_benchmark_cases id=6 anchor_slug='GAP-FVM-TWOPHASE': "
                  "case='Zalesak 旋转缺口盘 + Hysing 上升气泡', metric='界面形状误差/末速 vs 基准数据', "
                  "tolerance='形状 qualitatively+末速<10%'",
        "zalesak_shape": {"ref": "exact(one revolution)=initial field, Zalesak 1979",
                          "got_geometric_error_N128_plic": 0.039548,
                          "got_geometric_error_N64_plic": 0.093023,
                          "observed_order": 1.234, "verdict": "PASS (qualitative)"},
        "hysing_terminal_velocity": {"ref": 0.2342, "got": None,
                                     "tolerance": "<10%",
                                     "verdict": "NOT_MEASURED -> benchmark half unmet"},
        "extra_dambreak_martin_moyce": {"front_max_rel_240x80": 0.8477,
                                        "height_max_rel_240x80": 0.8265,
                                        "front_max_rel_t_star_gt_2.5": 0.2718,
                                        "verdict": "FAIL (>15%); not the benchmark case"},
        "gates_reproduced": "8/10 (independent --summary-only rerun, 12 s)",
        "own_limits_vs_audited": "Zalesak reported/audited delta 0.00% (4 legs); "
                                 "dam break reported/audited delta 0.00% (2 grids)",
        "caveat": "benchmark row created 2026-11-11 19:27 UTC, after mission finished 15:40 UTC",
    },
    "repro_cmd": "PYTHONPATH=src python -m tensorfvm.twophase_gap --summary-only "
                 "(gates 8/10 in 12 s); full: PYTHONPATH=src python -m tensorfvm.twophase_gap "
                 "--case all --out docs/GAP-FVM-TWOPHASE/artifacts --cells 64 128 "
                 "--schemes plic cicsam --dam-grids 120x40 240x80",
    "external_anchor": [
        "src/tensorfvm/vof_phase0/anchors.py",
        "docs/board-audit-m15026/REPORT.md",
    ],
    "audit_artifacts": [
        "docs/board-audit-m15026/REPORT.md",
        "docs/board-audit-m15026/audit_recompute.json",
        "docs/board-audit-m15026/audit_recompute.py",
        "docs/board-audit-m15026/audit_dambreak.json",
        "docs/board-audit-m15026/audit_dambreak.py",
        "docs/board-audit-m15026/fig_audit_m15026.png",
    ],
    "audit_findings": [
        "Zalesak half is clean: reported == audited == independent analytic reference (0.00% delta), "
        "artefact 'exact' field identical to a locally built slotted-disk mask (maxdiff 0)",
        "official benchmark half (b) Hysing terminal velocity <10% has NO data anywhere in the repo "
        "(all docs/in99-n01-two-phase-campaign legs have 2d_reference=None; that campaign is Eo=1.23 "
        "water-air, not Hysing Eo=10)",
        "dam-break headline 84.8%/98.8% is dominated by (i) the t*~0.3 startup where the reference "
        "front is only 0.45 (small denominator) and (ii) a near-constant front offset "
        "x*_num-x*_MM = -0.67 (240x80) / -0.78 (120x40); pure time-rescaling fit (alpha=0.47/0.51) "
        "leaves 51%/45% residual, so it is not a time-normalisation bug; window t*>2.5 still gives "
        "27.2%/31.1% -> real deficiency remains, but the report's single-cause attribution "
        "('first-order momentum advection + void free surface') is an unsupported inference",
        "interface width 12 cells at N=128 with observed order 1.23 for a nominally 2nd-order "
        "geometric reconstruction = real technical debt (transport/reconstruction coupling)",
    ],
    "residual_scope_for_successor": [
        "add Hysing 2-D case (Eo=10, M=1e-3, rho-ratio 10, mu-ratio 10) and grade terminal velocity vs 0.2342 <10%",
        "widen dam-break metric window to t*>=1.0 and report absolute/L1 front error; diagnose the ~0.7a offset",
        "investigate PLIC sweep / time-splitting coupling behind the 12-cell interface width and order 1.23",
    ],
    "verdict_rationale": "L3: real code + entry + reproducible artefacts + genuine integration caller "
                         "(vof_phase0/solver.py:130, covered by test) all verified by rerun; L4 denied "
                         "because the authoritative benchmark case is only half covered (Hysing terminal "
                         "velocity never measured) and the delivered benchmark leg that does exist "
                         "(dam break) fails at 84.8% >> 15%.",
}

SCOPE = "m15026 GAP-FVM-TWOPHASE"
REPO = "TensorFVM"
MISSION = 15190
LEVEL = 3
BY = "board-audit"

ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
sql = (
    "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
    f"VALUES ('{REPO}', '{SCOPE}', {MISSION}, {LEVEL}, '{ev}'::jsonb, '{BY}') "
    f"ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET level={LEVEL}, "
    f"mission_id={MISSION}, evidence='{ev}'::jsonb, gate_state='pending', gate_level=NULL, "
    "gate_note=NULL"
)
r = subprocess.run(["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-c", sql],
                   capture_output=True, text=True)
print(r.stdout[-500:], r.stderr[-800:])
r2 = subprocess.run(
    ["runuser", "-u", "postgres", "--", "psql", "-d", "tensorplatform", "-tA", "-F", "|", "-c",
     f"SELECT id, repo, scope, level, verdict_by, coalesce(gate_state,'pending'), "
     f"jsonb_object_keys_count FROM (SELECT id, repo, scope, level, verdict_by, gate_state, "
     f"(SELECT count(*) FROM jsonb_object_keys(evidence)) AS jsonb_object_keys_count "
     f"FROM tensor_maturity_verdicts WHERE scope='{SCOPE}' AND verdict_by='{BY}') t"],
    capture_output=True, text=True)
print("ROW:", r2.stdout.strip(), r2.stderr[-300:])