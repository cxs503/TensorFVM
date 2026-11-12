#!/usr/bin/env python3
"""Land the m15554 board-audit verdict for mission m15458 (CASE-CAVITY-RE1000-FVM).

Writes one row into ``tensor_maturity_verdicts``.  Every evidence pointer below is a
path/artefact that was actually read while auditing:

=====  ==========================================================================
key     pointer
=====  ==========================================================================
impl    the shipped solver, the Ghia transcription, the runner, the renderer, and
        this audit's own independent recompute script
entry   the repro entry point (exercised end-to-end at reduced scale)
artef   被审单工作树 artifacts/（summary.json / run.log / .pt 场 / 图）+ 本单审计 JSON/PNG
tests   本单实跑的四个测试文件（被审交付测试在审计前 1 failed、修复后 5 passed）
integ   `tensorfvm.lid_cavity` 的仓内调用方（gate 在生态仓 grep 复核）
bench   tensor_benchmark_cases id=18（审计前 0 行）+ ref/got/误差逐项
repro   `scripts/run_cavity_re1000_m15458.py` 的完整命令行
anchor  Ghia et al. (1982) JCP 48:387-411 的仓内转录 + 自校测试
=====  ==========================================================================

Dry-run of the platform's own gate (``/root/tp-agent/maturity_gate.py``) is printed
before the insert, so the declared level is machine-checked here first.

Usage: python3 docs/audit-m15554/land_verdict.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

SCOPE = "m15458 CASE-CAVITY-RE1000-FVM"
MISSION = 15554
REPO = "TensorFVM"
WORK = "/root/tp-agent/work/m15458/artifacts/CASE-CAVITY-RE1000-FVM"
EVIDENCE = {
    "impl_files": [
        "src/tensorfvm/lid_cavity.py",
        "src/tensorfvm/lid_cavity_reference.py",
        "scripts/run_cavity_re1000_m15458.py",
        "scripts/render_cavity_re1000_m15458.py",
        "docs/audit-m15554/verify_cavity_m15458.py",
    ],
    "impl_notes": "lid_cavity.py:151 LidCavitySolver (staggered MAC, van Leer faces, DCT-II "
                  "projection); :260 tendencies / :269 solve_pressure / :289 step / :308 run / "
                  ":397 score_profiles / :445 score_against_ghia / :481 ghia_acceptance. "
                  "Runner :81 run_case / :141 judge_5pct / :166 main. The audit re-read the "
                  "stored velocity fields with its own sampling + linear interpolation and its "
                  "own streamfunction path-integral, i.e. not through score_profiles.",
    "entry": "scripts/run_cavity_re1000_m15458.py",
    "run_artifacts": [
        f"{WORK}/summary.json",
        f"{WORK}/run.log",
        f"{WORK}/fields/Re1000_n32.pt",
        f"{WORK}/fields/Re1000_n64.pt",
        f"{WORK}/fields/Re1000_n128.pt",
        f"{WORK}/fig1_centrelines.png",
        "docs/audit-m15554/cavity_audit.json",
        "docs/audit-m15554/fig_audit_centrelines.png",
        "docs/audit-m15554/fig_audit_fields.png",
        "docs/audit-m15554/fig_audit_convergence.png",
        "docs/audit-m15554/platform_state.json",
    ],
    "run_artifacts_notes": "被审单工作树（branch tip 220c729c, merge 1b7a50e0）: summary.json "
                           "(sha256 3bdafb50…, updated 2026-11-12T00:50:23Z) 含 n=32/64/128 三行、"
                           "全部 converged=True; 场 .pt 三档; run.log 逐次 check 速率. 注意 "
                           "`git ls-files artifacts` = 0 —— 工件未入版本库，CI 上交付证据测试退化为 "
                           "skip（本单在无 artifacts 的工作树复现 5 skipped）。",
    "tests": [
        "tests/test_case_cavity_re1000_m15458.py (交付; 审计时 1 failed/4 passed → 重生成报告后 5 passed)",
        "tests/test_ghia_reference_consistency.py (参考表自校; 19 passed)",
        "tests/test_audit_m15554_cavity_m15458.py::test_all_three_resolutions_are_steady_discrete_solutions (本单新增; 重走一步速率 9.76e-08/9.58e-08/9.89e-08 ≤ 1e-7)",
        "tests/test_audit_m15554_cavity_m15458.py::test_error_decreases_monotonically_with_refinement",
        "tests/test_audit_m15554_cavity_m15458.py::test_finest_row_satisfies_the_3_percent_claim",
        "tests/test_audit_m15554_cavity_m15458.py::test_audit_numbers_match_the_delivered_summary",
        "tests/test_audit_m15554_cavity_m15458.py::test_referee_registration_and_figures_exist",
    ],
    "integration_callers": [
        "tensorfvm.lid_cavity",
        "src/tensorfvm/case_cavity_re100.py:46 imports LidCavityConfig/LidCavitySolver/"
        "score_against_ghia and registers the solver as the case engine at :124; "
        "src/tensorfvm/lid_cavity_3d.py:186 reuses the same 2-D discretisation; "
        "src/tensorfvm/verification/ladder.py drives it as the ladder engine. "
        "平台侧: GET /tensor/bench/results?mission_id=15458 total=0, "
        "GET /tensor/reports?mission_id=15458 total=1 (report id 14539).",
    ],
    "benchmark": {
        "authoritative_judge": "SELECT case_name,metric,tolerance FROM tensor_benchmark_cases "
                               "WHERE anchor_slug='CASE-CAVITY-RE1000-FVM' -> 0 rows at audit "
                               "time (anchor never registered; 17 anchors in table, no CAVITY); "
                               "registered by this audit as id=18 (docs/audit-m15554/"
                               "register_benchmark_case.sql).",
        "ref_source": "Ghia, Ghia & Shin, J. Comput. Phys. 48 (1982) 387-411, Tables I/II "
                      "(committed transcription src/tensorfvm/lid_cavity_reference.py:76/:137, "
                      "vortex centre :162; self-check tests/test_ghia_reference_consistency.py)",
        "n64": {"u_min": {"ref": -0.38289, "got": -0.37128, "rel_err_pct": 3.03},
                "v_max": {"ref": 0.37095, "got": 0.36173, "rel_err_pct": 2.49},
                "u_interior_max_abs_pct_of_U": 1.89, "v_interior_max_abs_pct_of_U": 1.64,
                "vortex_centre": {"ref": [0.5313, 0.5625], "got": [0.5312, 0.5703], "abs_delta_L": 0.0078},
                "verdict": "u_min borderline (delivered Hermite sampling gives 2.80%, audit linear 3.03%)"},
        "n128": {"u_min": {"ref": -0.38289, "got": -0.38511, "rel_err_pct": 0.58},
                 "v_max": {"ref": 0.37095, "got": 0.37368, "rel_err_pct": 0.74},
                 "u_interior_max_abs_pct_of_U": 0.53, "v_interior_max_abs_pct_of_U": 1.01,
                 "vortex_centre": {"ref": [0.5313, 0.5625], "got": [0.5312, 0.5664], "abs_delta_L": 0.0039},
                 "verdict": "all key quantities < 3%"},
        "n32_control": {"u_min": {"ref": -0.38289, "got": -0.31642, "rel_err_pct": 17.36},
                        "verdict": "under-resolved control, FAIL (kept on purpose)"},
        "observed_order_interior": [2.08, 1.82],
        "invariants": {"max_abs_divergence": 1.35e-13, "re_stepped_rate_n128": 9.89e-08,
                       "gate": 1e-07},
        "cross_check": "delivered summary.json numbers re-derived from fields: |d u_min| <= 9.0e-4, "
                       "|d v_max| <= 5.5e-4 -> numbers are real, not transcribed",
    },
    "repro_cmd": ("PYTHONPATH=src python3 scripts/run_cavity_re1000_m15458.py --out "
                  "artifacts/CASE-CAVITY-RE1000-FVM --reynolds 1000 --grids 64,128 --cold-grids 32 "
                  "&& PYTHONPATH=src python3 scripts/render_cavity_re1000_m15458.py "
                  "&& PYTHONPATH=src python3 -m pytest tests/test_case_cavity_re1000_m15458.py -q"),
    "repro_notes": "入口本单实测可跑（缩小算例 --reynolds 100 --grids 16 --cold-grids 8 "
                   "--max-steps 6000 → 27 s 端到端，写出 summary.json/两档/判据）；全量影子成本 "
                   "≈ n=128 1849 s + n=64 442 s（容器 CPU，集群 cj1123 全程排队）。",
    "external_anchor": [
        "src/tensorfvm/lid_cavity_reference.py",
        "tests/test_ghia_reference_consistency.py",
    ],
    "external_anchor_note": "U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411 "
                            "(Table I u on x=L/2, Table II v on y=L/2, primary vortex centre).",
    "level_notes": "L4 = 完整算例(3 分辨率, 全部离散定常) + 误差分析(极值/逐点/内点/收敛阶 1.8-2.1) "
                   "+ 可复现(入口实测). L5 不成立: 标准基准复现, 无新方法.",
    "audit_findings": {
        "fixed": [
            "report.md 只列 n=32/64，bundle 在 00:50 多出已收敛的 n=128 → 交付测试 "
            "test_report_tables_come_from_the_bundle 在审计时失败（1 failed/4 passed）；本单用交付自带 "
            "renderer 从 bundle 重生成报告（零手抄）→ 5 passed，并补上 n=128 这档最强证据。",
            "CASE-CAVITY-RE1000-FVM 在 tensor_benchmark_cases 无登记 → 补登记 id=18。",
        ],
        "open": [
            "交付 headline 的 max|Δu|=2.93% 实际是壁面端点 y=1.0 的 Hermite 外推残差（内点仅 1.89%）；"
            "n=64 的 3% 结论不稳健（线性口径 3.03%），应以 n=128 为生产档。",
            "artifacts/ 未入版本库（git ls-files artifacts = 0），交付证据测试在 CI 退化为 skip；"
            "建议把 summary.json 纳入版本库。",
            "平台 L4 账本（tensorbench_results，全表仅 1 行）无本单行 → /tensor/bench/l4-verdict/15458 "
            "返回 CLAIMED_NOT_PROVEN。本单未代填：需要 cluster_job_id 绑定，而交付数值全部产自容器 CPU。",
            "renderer 把 all_converged 插进「未收敛行」那句话里，三档全收敛后语义别扭（纯文案）。",
            "run_cavity_re1000_m15458.py:200 在 --grids 给 3 档以上时会引用未定义的 fine_solver（默认 2 档不触发）。",
        ],
    },
}


def psql(sql: str) -> tuple[int, str]:
    r = subprocess.run(["psql", "-U", "postgres", "-d", "tensorplatform", "-v", "ON_ERROR_STOP=1",
                        "-c", sql], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, "/root/tp-agent")
    import maturity_gate as mg  # noqa: E402  (platform's own judge, used as a dry run)

    level, reasons = mg.judge(4, EVIDENCE, REPO, MISSION)
    print(f"gate dry-run: declared L4 -> machine-judged L{level}")
    for r in reasons:
        print("  ", r)
    if args.dry_run:
        return 0
    if level < 4:
        print("REFUSING to declare L4: the platform gate does not corroborate it.", file=sys.stderr)
        return 2

    ev = json.dumps(EVIDENCE, ensure_ascii=False).replace("'", "''")
    sql = (
        "INSERT INTO tensor_maturity_verdicts (repo, scope, mission_id, level, evidence, verdict_by) "
        f"VALUES ('{REPO}', '{SCOPE}', {MISSION}, {level}, '{ev}'::jsonb, 'board-audit') "
        "ON CONFLICT (repo, scope, verdict_by) DO UPDATE SET "
        f"mission_id={MISSION}, level={level}, evidence='{ev}'::jsonb, created_at=now() "
        "RETURNING id, repo, scope, level, gate_state;"
    )
    rc, out = psql(sql)
    print(out)
    if rc != 0:
        return rc
    rc, out = psql("SELECT id, level, gate_state, gate_level, left(verdict_by,20) FROM "
                   "tensor_maturity_verdicts WHERE repo='TensorFVM' AND scope="
                   f"'{SCOPE}' ORDER BY id;")
    print(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())