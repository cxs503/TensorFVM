#!/usr/bin/env python3
"""Publish the m20109 board-audit report to the platform (figures embedded)."""
import base64
import json
import os
import subprocess

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ART = os.path.join(REPO, "artifacts", "board_audit_m20109")
FIG = os.path.join(REPO, "docs", "board_audit_m20109", "figures")

rec = json.load(open(os.path.join(ART, "recompute.json")))
ref = json.load(open(os.path.join(ART, "referee.json")))
rep = json.load(open(os.path.join(ART, "report.json")))
repro = json.load(open(os.path.join(ART, "repro.json")))
inv = json.load(open(os.path.join(ART, "inventory.json")))
e = rec["errors_pct"]
rc = rec["reference_condition"]
dr = rec["artifact_vs_audit"]


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return base64.b64encode(f.read()).decode()


md = f"""# 交差核验：m20059 / CASE-FVM-CONVDIFF-PHASE（TensorFVM）—— L4 确认，附 5 条缺陷

**审计单 m20109 · 被审单 m20059（succeeded）· 被审 commit `f2ba8c48`（分支 `agent/m20059`，尚未并入 main）**
进程内产物：`artifacts/board_audit_m20109/*.json`、审计脚本 `scripts/audit/m20109_convdiff_phase_audit.py`、
完整报告 `docs/board_audit_m20109/M20109_BOARD_AUDIT.md`；交付件字节副本在
`docs/board_audit_m20109/audited/`（sha256 见 inventory.json）。

## ① 案例介绍（物理背景/几何/参数）

单模态线性对流扩散 `phi_t + c phi_s = D phi_ss`，闭合环形网格（64 胞/圈，半径 1，厚 0.05，展向 0.05），
初场 `sin(2πi/64)`，用**出厂核** `tensorfvm.moving_mesh.ale_scalar_step`（显式一阶迎风 + 点隐式扩散）
在**出厂网格类** `StructuredCurvilinearMesh(periodic=(True,False,False))` 上推进；测量**相位误差**（色散）
对解析行波 `exp(-Dk²t) sin(k(s-ct))`，并做 Courant / 网格 / 扩散数三轴敏感性。

参考条件：N=64, C=1/4, a=0, 200 步（0.7812 周期）；θ=0.09817；几何因子 ds/(V/A)=1.0002083。

## ② 怎么做的（审计方法，非重做）

1. **取证**：11 个交付件 sha256 与 `f2ba8c48` 逐一比对，字节副本入库（{inv['n_files']}/{inv['n_files']} 存在，副本一致 = {inv['all_copies_match']}）。
2. **基准裁判**：`tensor_benchmark_cases` 中该锚点行数 = **{ref['registered_rows']}（空表）**；
   实际生效判据取任务榜原文 **<3 %，ref=解析正弦**。
3. **独立重算**：在 `git archive f2ba8c48` 的干净检出上，用**自写**环形节点数组 + 自写 `np.linalg.lstsq` 相位估计，
   并用插桩包住 `ale_scalar_step` 统计调用次数 → **{rec['kernel_calls_observed']}/{rec['kernel_calls_expected']}**，证明跑的是产线核。
4. **测试**：`PYTHONPATH=src python -m pytest tests/test_convdiff_phase.py -q` → **13 passed in 5.82 s**。
5. **复现**：重跑 `run_case.py`，`summary.json` 逐叶比较 **n_diffs={repro['n_diffs']}（完全相同）**；
   `REPORT.md` 亦由驱动逐字节重生成（报告非手改）。
6. **报告核对**：六段齐全 = {all(rep['six_section_check'].values())}；§4 头条数字回算一致 = {rep['headline_lag_per_step_matches_artifact']}。

## ③ 计算结果（数字表，程序化注入）

| 量 | 参考 | got（审计重算） | 误差 | 判定 |
|---|---|---|---|---|
| 每步相位滞后（C=1/4,N=64） | 迎风主项 Cθ³(1-C)(1-2C)/6 = {rc['upwind_leading_order_per_step']:.6e} rad | {rc['lag_per_step_vs_index_reference']:.6e} rad | {100*abs(rc['lag_per_step_vs_index_reference']/rc['upwind_leading_order_per_step']-1):.3f} % | **<3 % PASS** |
| 相位误差 vs 解析波（按报告口径，索引参考 C·steps） | 0 | {e['phase_vs_index_reference']:.5f} % | — | **<3 % PASS** |
| 相位误差 vs 解析波（几何参考 C·steps/(ds/(V/A))） | 0 | {e['phase_vs_geometric_reference']:.5f} % | — | **<3 % PASS** |
| 每步滞后随 dx 的阶（16/32/64/128） | 3 | {rec['observed_order_audit']:.4f}（原件 {rec['observed_order_artifact']:.4f}） | {100*abs(rec['observed_order_audit']/3-1):.2f} % | **PASS** |
| 振幅 vs 修正方程 exp(-½C(1-C)θ²n) | {rc['modified_equation_amplitude']:.6f} | {rc['amplitude']:.6f} | {e['amplitude_vs_modified_equation']:.4f} % | **<3 % PASS** |
| 振幅 vs 纯解析正弦（D=0 应为 1） | 1 | {rc['amplitude']:.6f} | {e['amplitude_vs_analytic']:.3f} % | **FAIL（诚实，一阶迎风固有耗散）** |
| 扩散支 a=1e-2 vs 连续解析 | 0 | 24.75 % | — | **FAIL（已声明局限，不算认证）** |

交付件 vs 审计重算的漂移：`lag/step` 相对差 = {dr['lag_per_step_rel_diff']:.1e}，振幅相对差 = {dr['amplitude_rel_diff']:.1e}——两把独立尺子到机器精度一致。

## ④ 结果图

图 1：左＝Courant 扫描（审计独立重算 vs 交付件 vs 迎风主项解析律，C=1/2 与 C=1 处相位差归零）；右＝网格阶数阶梯，斜率 {rec['observed_order_audit']:.3f}（参考线 3）。
图 2：误差预算对 3 % 容差（对数轴）：认证通过的三项 + 两项诚实 FAIL。

## ⑤ 注意点（踩过的坑/局限/适用范围）

* **F1 中** `tensor_benchmark_cases` 该锚点 **0 行**：bench 侧裁判空表，本轮判据只能取任务榜文本；建议补登记一行（metric + `<3 %`）。
* **F2 中** 算例未登记进 `src/tensorfvm/verification/__main__.py` 或任何池清单（`grep -rn CONVDIFF src scripts` 只命中模块自身），平台无法枚举/重跑；唯一消费者是其驱动与测试。
* **F3 低/中** 报告口径的相位参考用 `C·steps`（索引单位），未除几何因子 `ds/(V/A)=1.0002083`；物理参考应为 `0.03945 %` 而非 `0.06028 %`（报告数字偏保守，1.5 倍，结论不变）。有意思的是判据 C4 用的却是物理 `c·ds`，两处口径不一致。
* **F4 低/中** `run_case.py:395` 硬编码句"相位滞后与扩散数无关（a=0…1e-1 变化小于 1.4e+02 度，低于实测滞后本身）"与 §4 表（0.1695°→9.098°→69.61°→−144.40°）直接矛盾，且紧接下一句就写"相位**并非**与扩散无关"。数字是程序注入的，坏的是这句人工模板。
* **F5 低** §2/§3 把探针值 `C=10.202297`（speed=dt=1 时的 A/V）标成"实测 C"，运行/sweep 实际 C=0.25；显式迎风 C=10 物理上荒谬，标签易误导。
* **适用范围**：本算例只认证**相位/色散**（a=0 支）；零梯度 patch 闭合把波速稀释 `1/(1+w)`（实测 w=0.031/0.308/3.08），扩散支是**表征**不是验证；振幅精度未认证。

## ⑥ 误差判定（逐项，显式回答是否 <3 %）

1. 每步相位滞后 vs 迎风主项解析律：{100*abs(rc['lag_per_step_vs_index_reference']/rc['upwind_leading_order_per_step']-1):.3f} % → **是，<3 %**。
2. 相位误差 vs 解析行波（报告口径）：{e['phase_vs_index_reference']:.5f} % → **是，<3 %**。
3. 相位误差 vs 解析行波（几何口径，审计更严）：{e['phase_vs_geometric_reference']:.5f} % → **是，<3 %**。
4. 每步滞后随 dx 的阶：{rec['observed_order_audit']:.4f} vs 3（带 ±0.3）→ **是，落在带内**。
5. 振幅 vs 修正方程：{e['amplitude_vs_modified_equation']:.4f} % → **是，<3 %**。
6. 振幅 vs 纯解析正弦：{e['amplitude_vs_analytic']:.2f} % → **否，>3 %（一阶迎风固有耗散，原件已诚实 FAIL）**。
7. 扩散支 a=1e-2 vs 连续解析：24.75 % → **否，>3 %（patch 闭合导致，原件已声明不可认证）**。

**裁定：L4（基准验证）确认，L5 保留不予**（ref 为教科书闭式、代码内自证，非实验/文献数值表）。审计动作 = approve with notes；缺陷 F1/F2 为登记缺口，F3/F4/F5 为表述缺陷，均不改变任何数值结论。
"""

report = {
    "title": "交差核验 m20059 CASE-FVM-CONVDIFF-PHASE：L4 确认，5 条缺陷（2 登记缺口 + 3 表述）",
    "summary": ("独立重算交付件全部数值（1e-15 级一致）、13 测试通过、复现 n_diffs=0，"
                "插桩证明 200/200 次调用产线核 ale_scalar_step；相位误差 0.060 %（几何口径 0.039 %）<3 % 通过，"
                "振幅 16.5 % 与扩散支 24.7 % 为已声明的诚实 FAIL；bench 裁判表该锚点 0 行、算例未入池。"),
    "scenario_key": "board-audit",
    "mission_id": 20109,
    "tags": ["board-audit", "TensorFVM", "CASE-FVM-CONVDIFF-PHASE", "phase-accuracy",
             "maturity-verdict", "L4", "adversarial-review"],
    "markdown": md,
    "figures": [
        {"name": "audit_fig1_phase_and_order.png",
         "caption": "左：Courant 扫描，审计独立重算 vs 交付件 vs 迎风主项解析律（C=1/2、C=1 相位差归零）；"
                    "右：网格阶数阶梯，实测斜率 %.3f（参考线 3）" % rec["observed_order_audit"],
         "png_base64": b64("audit_fig1_phase_and_order.png")},
        {"name": "audit_fig2_error_budget.png",
         "caption": "误差预算对 3% 容差（对数轴）：相位三项通过，振幅 16.5% 与扩散支 24.7% 为已声明的诚实 FAIL",
         "png_base64": b64("audit_fig2_error_budget.png")},
    ],
}

json.dump(report, open("/tmp/report.json", "w"), ensure_ascii=False)
print("wrote /tmp/report.json", len(json.dumps(report)), "bytes")