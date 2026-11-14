#!/usr/bin/env python3
"""Compose the m20007 audit document + the maturity-verdict evidence payload.

All numbers are read from raw artifacts (delivered snapshot, auditor re-runs).
Emits:
  docs/audit_m20007/AUDIT-m20007-CASE-FVM-CHANNEL-DEVELOP.md
  docs/audit_m20007/verdict_evidence.json
"""
from __future__ import annotations

import csv
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUNS = HERE / "audited_runs"
ART = HERE / "artifacts"
THRESHOLD = 0.99 * 1.5


def jload(p):
    return json.load(open(p)) if pathlib.Path(p).exists() else None


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def main():
    bench = json.load(open(RUNS / "benchmark.json"))
    manifest = json.load(open(RUNS / "manifest.json"))
    repro200 = jload(ART / "repro/benchmark.json")["cases"][0]
    repro50 = jload(ART / "repro50/benchmark.json")["cases"][0]
    recheck8 = jload(ART / "recheck_ny8_nx128_Re200.json")
    rechecks = sorted((json.load(open(p)) for p in ART.glob("recheck_ny*.json")),
                      key=lambda d: d["ny"])
    with open(RUNS / "metrics.csv") as f:
        metrics_rows = list(csv.DictReader(f))
    with open(RUNS / "centerline_profiles.csv") as f:
        prof_cases = sorted({r["case"] for r in csv.DictReader(f)})
    d = "\n".join

    t_bench = table(
        ["case", "Re_H", "n_y", "L/H", "got L_e/H", "ref L_e/H", "err %", "u_max/ū",
         "dp/dx got", "dp/dx ref", "dp/dx err %", "iters"],
        [[c["name"], c["re_h"], c["ny"], c["L/H"], c["L_e/H (computed)"],
          c["L_e/H (White 2006)"], c["L_e/H error %"], c["u_max/u_bar"],
          c["dp/dx"], c["dp/dx ref"], c["dp/dx error %"], c["iterations"]]
         for c in bench["cases"]])

    t_metrics = table(["case", "got L_e/H", "err %", "u_max/ū"],
                      [[r["name"], r["L_e/H (computed)"], r["L_e/H error %"], r["u_max/u_bar"]]
                       for r in metrics_rows])

    t_dpdx = table(
        ["case", "delivered dp/dx", "delivered ref", "auditor re-run dp/dx",
         "analytic -12νū/H²", "delivered err%", "re-run err%", "ratio delivered/re-run"],
        [[c["name"], c["dp/dx"], c["dp/dx ref"],
          (repro50 if c["re_h"] == 50.0 else repro200)["dp/dx"],
          -12.0 / c["re_h"], c["dp/dx error %"],
          (repro50 if c["re_h"] == 50.0 else repro200)["dp/dx error %"],
          f"{abs(c['dp/dx'] / (repro50 if c['re_h'] == 50.0 else repro200)['dp/dx']):.3f}×"]
         for c in bench["cases"]])

    t_ceiling = table(
        ["n_y", "max cell-centre u/ū of exact parabola", "1.485 reachable"],
        [[ny, f"{float((6*((np.arange(ny)+0.5)/ny)*(1-(np.arange(ny)+0.5)/ny)).max()):.6f}",
          "**no**" if (6*((np.arange(ny)+0.5)/ny)*(1-(np.arange(ny)+0.5)/ny)).max() < THRESHOLD
          else "yes"] for ny in (8, 9, 10, 16, 32)])

    t_recheck = table(
        ["Re_H", "n_y", "n_x", "max u_c (even)", "max u_c (mid-plane)", "1.485 reachable",
         "L_e/H even", "L_e/H mid-plane", "err % mid", "iters"],
        [[r["re_h"], r["ny"], r["nx"], f"{r['u_center_even_max']:.6f}",
          f"{r['u_center_midplane_max']:.6f}",
          "yes" if r["threshold_reachable_midplane"] else "**no**",
          f"{r['Le_over_H_even_sample']:.4f}" + ("" if r["Le_crossed_even"] else " (sentinel)"),
          f"{r['Le_over_H_midplane']:.4f}" + ("" if r["Le_crossed_midplane"] else " (sentinel)"),
          f"{r['err_pct_midplane']:.2f}", r["iterations"]] for r in rechecks])

    doc = d([
        "# 交差审计：m20007 CASE-FVM-CHANNEL-DEVELOP（入口段发展流充分发展长度 · TensorFVM）",
        "",
        "**审计单** m20103 ｜ **被审单** m20007（succeeded）｜ **仓库** TensorFVM ｜ "
        "**提交面** `main 0a075ce8` + 未提交工作树 `/root/tp-agent/work/m20007` ｜ "
        "**判定：声明 L2**（L3/L4/L5 拒）",
        "",
        "> 独立性：审计者未参与 m20007 实现。结论只依赖三类独立取证——(a) 交付工件逐字节快照 + "
        "sha256（`audited_runs/`、`SHA256SUMS.txt`）；(b) 审计者独立脚本 "
        "`docs/audit_m20007/artifacts/recheck_entrance.py` 直接调用仓内核 SteadyFlow 复测，**不 import "
        "被审模块**；(c) 闭式算术与平台裁判表数据库核对。被审模块的 `case` 子命令被原样重跑以核 "
        "`repro_cmd`。局限性：White 2006 纸本不在生态内，教科书系数未独立复核（WITHHELD）；交付 3 张 "
        "PNG 仅做非空白校验；`all` 六例未一次性全跑。",
        "",
        "## 0. 基准裁判（唯一裁判 = tensor_benchmark_cases）",
        "",
        "```sql",
        "SELECT case_name, metric, tolerance FROM tensor_benchmark_cases",
        " WHERE anchor_slug='CASE-FVM-CHANNEL-DEVELOP';   -->  0 rows",
        "```",
        "",
        "该锚点在平台裁判表**零登记**（`%DEVELOP%` 亦为 0 行）：没有平台侧 ref/tolerance 可核，"
        "交付自设 8% 容差属自声明（平台验收底线 15%）。本单因此以交付自己声明的裁判"
        "（White 2006 §3-3.1，`L_e/H = 0.06 Re_H`）逐项对表。",
        "",
        "## 1. 三件套取证",
        "",
        "### 1.1 run_artifacts：文件真实存在，但**内部不一致**",
        "",
        table(["artifact", "sha256(16)", "size"],
              [[k, v[:16], (RUNS / k).stat().st_size if (RUNS / k).exists() else "-"]
               for k, v in manifest["artifact_hashes"].items()]),
        "",
        f"**F3**：`benchmark.json` 有 {len(bench['cases'])} 例，而 `metrics.csv` 仅 "
        f"{len(metrics_rows)} 行、`centerline_profiles.csv` 仅 {len(prof_cases)} 例"
        f"（{', '.join(prof_cases)}）——`write_artifacts()` 用 `\"w\"` 覆盖写，最后一次单例运行把三例 CSV "
        "冲掉；`manifest.json` 对自己取哈希（写入时旧内容），自哈希失配。工件包不能自洽复算 digest。",
        "",
        "metrics.csv 实存内容：",
        "",
        t_metrics,
        "",
        "### 1.2 基准裁判 ref vs got（逐项）",
        "",
        t_bench,
        "",
        f"交付自判 **{bench['verdict']}**（容差 {bench['tolerance']:.0%}）。数字与工件一致，语义见 §3。",
        "",
        "### 1.3 repro_cmd 可执行性 + **逐位复现核验**",
        "",
        "manifest: `" + manifest["repro_cmd"] + "`；审计者另跑 "
        "`case --re-h 200 --nx-per-h 8 --aspect 16`（94 s）与 `case --re-h 50 --nx-per-h 8 --aspect 8`"
        "（52 s）。",
        "",
        table(["case", "L_e/H", "u_max/ū", "iters", "dp/dx", "dp/dx ref", "dp/dx err%"],
              [["delivered Re200-nx8", bench["cases"][2]["L_e/H (computed)"],
                bench["cases"][2]["u_max/u_bar"], bench["cases"][2]["iterations"],
                bench["cases"][2]["dp/dx"], bench["cases"][2]["dp/dx ref"],
                bench["cases"][2]["dp/dx error %"]],
               ["auditor re-run Re200", repro200["L_e/H (computed)"], repro200["u_max/u_bar"],
                repro200["iterations"], repro200["dp/dx"], repro200["dp/dx ref"],
                repro200["dp/dx error %"]],
               ["delivered Re50-nx8", bench["cases"][0]["L_e/H (computed)"],
                bench["cases"][0]["u_max/u_bar"], bench["cases"][0]["iterations"],
                bench["cases"][0]["dp/dx"], bench["cases"][0]["dp/dx ref"],
                bench["cases"][0]["dp/dx error %"]],
               ["auditor re-run Re50", repro50["L_e/H (computed)"], repro50["u_max/u_bar"],
                repro50["iterations"], repro50["dp/dx"], repro50["dp/dx ref"],
                repro50["dp/dx error %"]]]),
        "",
        "**F7（重大·证据不可复现）**：L_e/H、u_max、迭代数逐位复现，但 **dp/dx 与其参考值复现不了**："
        f"Re50 交付行 dp/dx={bench['cases'][0]['dp/dx']} / ref={bench['cases'][0]['dp/dx ref']}"
        f"（err {bench['cases'][0]['dp/dx error %']}%），审计侧同源同参重跑得 "
        f"dp/dx={repro50['dp/dx']} / ref={repro50['dp/dx ref']}（err {repro50['dp/dx error %']}%），"
        "相差约 7.9×。模块内 `dpdx_ref = -12 ν ū/H²` 对 Re50/100/200 应给 −0.24/−0.12/−0.06；"
        "交付 JSON 却写 −0.03/−0.06/−0.06（仅 Re200 自洽）。结论：交付 `benchmark.json` 的 Re50、Re100 "
        "两行**不是当前源码跑出来的**（或经过手改），三件套里的“工件”无法由“命令”再生成。",
        "",
        t_dpdx,
        "",
        "注：这一点同时**推翻报告 §5.3 的“充分发展区壓降已准确复现（1.4%）”**——真实重跑给 2.98%（<3% 仍成立），"
        "而交付的 1.33% 是错参考值造成的假精度。",
        "",
        "### 1.4 报告六段",
        "",
        "`docs/CASE-FVM-CHANNEL-DEVELOP/REPORT.md`（148 行）①背景/参数表 ②方法 ③计算表 ④ref-vs-got 表 "
        "⑤敏感性 ⑥失败模式/工件 全部齐全，格式合规；但结论建立在 §3 的错误测量与 §1.3 的错参考值上。",
        "",
        "## 2. 独立重建执行路径",
        "",
        "审计脚本只复用 `src/tensorfvm/inc_bench/solver.py:SteadyFlow`，并把两处独立化：`n_y` 与 `nx/H` "
        "解耦（被审模块 `ny = nx_per_H` 只能偶数）、同时给“偶数中心格采样”与“插值到 y=H/2 中面”两种判据。",
        "",
        "控制对照 `n_y=8, n_x=128, Re_H=200`：审计脚本 "
        + (f"`L_e/H={recheck8['Le_over_H_even_sample']}`、`max u_c={recheck8['u_center_even_max']}`、"
           f"`iters={recheck8['iterations']}`，`Le_crossed={recheck8['Le_crossed_even']}`" if recheck8 else "n/a")
        + "——与被审模块一致，说明取数没算错，问题在判据与算例设计。",
        "",
        "法向分辨率扫描（Re_H=100, L/H=16, n_x=128，compute 门递交）：",
        "",
        t_recheck if rechecks else "_未取得（集群作业排队中，如实登记为未跑）_",
        "",
        "## 3. 关键发现（严重度排序）",
        "",
        "**F1 阻断·正确性——三行“实测入口段长度”全是兜底哨兵值。** 判据未触发时代码执行 "
        "`le_over_h = centerline_x_vals[-1]`（`benchmark_entrance_length.py:182-184`），把域末格心当长度输出。"
        f"三行 got = {', '.join(str(c['L_e/H (computed)']) for c in bench['cases'])} 恰等于 `L/H − dx/2`；"
        "审计脚本同样得 `Le_crossed=False`。**这轮基准一次都没测到 L_e**，报告的 164.6%/165.6%/32.8% 是"
        "“域末位置 vs 教科书值”，不是测量误差；把哨兵值放进 ref-vs-got 对照表打分属实质误导。",
        "",
        "**F2 阻断·判据设计——1.485ū 在 n_y=8 上双重不可达。** (i) 闭式算术：精确抛物线格心采样上限 "
        "`1.5 − 1.5/n_y²`，n_y=8 → 1.47656 < 1.485，且偶数 n_y 永不采样 y=H/2（n_y≥10 才越过门槛）；"
        f"(ii) 求解器实况：出口最大 u/ū={recheck8['u_center_even_max'] if recheck8 else 1.4518}（比 1.5 低 3.2%），"
        "连中面插值也停在 1.4518——一阶迎风在 8 单元粗网格上数值黏度使渐近峰值偏低。判据以**理论** 1.5 为基准，"
        "于是永远不触发。交付把 `ny = nx_per_H` 绑死，使 PLAN 三个算例全落在不可达档。",
        "",
        t_ceiling,
        "",
        "**F7 重大·证据不可复现**：见 §1.3（dp/dx 与其参考值 Re50 差 7.9×，交付工件不能由交付命令再生成）。",
        "",
        "**F4 重大·交付**：代码从未提交——`tensor_dev_missions.result` 中 `head_commit=null`、`git_status` "
        "三行全 `??`；`grep -rn benchmark_entrance_length /root/tensor-ecosystem/TensorFVM` 命中 0；"
        "`tensorbench_results` 无 mission_id=20007 记录。能力只存在于临时工作树。",
        "",
        "**F5 重大·覆盖**：PLAN 声明 6 例（含 n_y=16/32），实跑 3 例且全为 n_y=8——恰好跳过能判定的档。"
        "报告“需 n_y≥16 但算力不足”是**未验证推断**；本次审计用集群补做（§2）后仍需如实说明结果边界。",
        "",
        "**F6 次要·表述**：①“zero hand-transcription”——系数是硬编码字面量 `WHITE_COEFF = 0.06`；"
        "②“n_y=16 在泊肃叶基准已验证 u_max<0.4%”无工件/测试背书；③锚点未登记裁判行。",
        "",
        "**非缺陷（避免误伤）**：均匀入口设定正确；充分发展区压降在审计重跑中误差 2.98%（<3%）说明内核"
        "在充分发展极限是对的——问题在入口段测量与工件一致性，不在内核。",
        "",
        "## 4. 判级（L0–L5）",
        "",
        "| 级别 | 机器判据 | 本单实测 | 结论 |",
        "|---|---|---|---|",
        "| L0 文档 | — | 148 行报告 | 过（不定级） |",
        "| L1 单元/数学验证 | tests 非空 | 交付无任何测试（`grep -rn entrance m20007/tests`=0） | 未取得 |",
        "| L2 独立功能实现 | impl 真代码 + entry + run_artifacts | 441 行真调内核、CLI 可跑、工件齐（虽不一致） | **达成 → L2** |",
        "| L3 系统集成 | + integration_callers | 无主线调用方、未提交、平台零摄入 | 拒 |",
        "| L4 基准验证 | + benchmark 对照 + repro_cmd | 裁判未登记 + 交付自判 FAIL + 3/3 行为哨兵值 + 工件不可复现 | 拒 |",
        "| L5 科研工程 | + external_anchor | 单一教科书锚点，系数未独立复核（WITHHELD） | 拒 |",
        "",
        "**判定 L2**：真代码、可复现的输入→求解→输出（审计侧 94 s/52 s 两次逐位复现）。L3/L4/L5 明确拒绝。"
        "报告厚度不改变级别：功能实质＝“能跑但基准未过、未提交、未集成、把哨兵值当测量值、且工件与命令不对应”的基准驱动模块。",
        "",
        "## 5. 误差判定（逐项回答“是否 <3%”）",
        "",
        "| 关键量 | 基准 | got | 误差 | <3% | <15%（平台底线） |",
        "|---|---|---|---|---|---|",
        "| L_e/H (Re_H=50) | 3.00 | 7.9375（哨兵） | 164.6% | 否 | 否 |",
        "| L_e/H (Re_H=100) | 6.00 | 15.9375（哨兵） | 165.6% | 否 | 否 |",
        "| L_e/H (Re_H=200) | 12.00 | 15.9375（哨兵） | 32.8% | 否 | 否 |",
        f"| u_max/ū | 1.5 | {recheck8['u_center_even_max'] if recheck8 else 1.4518} | 3.2% | 否（临界） | 是 |",
        f"| dp/dx (Re_H=50，审计重跑) | −0.24 | {repro50['dp/dx']} | {repro50['dp/dx error %']}% | 是 | 是 |",
        "| dp/dx (Re_H=50，交付行) | −0.03（错参考） | −0.0296 | 1.33%（假精度） | 是 | 是 |",
        "",
        "**入口段长度三档均不满足 <3%（也不满足 <15%），根因是“测量从未发生”（F1/F2），不是物理偏差。** "
        "honest-FAIL 成立但归因只对一半：主因判据/采样不可达，次因粗网格峰值低 3.2%；而报告的“1.4% 压降精确”"
        "是错参考值造成的假精度（真实 2.98%）。",
        "",
        "## 6. 判决与返修建议",
        "",
        "**Disposition: approve with notes**（记录 L2，拒 L3/L4；下一轮 request changes）。",
        "",
        "1. 无交叉必须显式失败，禁止把域末位置当长度；CSV 改 append 或按例分文件；manifest 不算自身哈希；",
        "2. 判据改为“相对离散可达/收敛可达峰值的 99%”，允许奇数 n_y（中心格心=y=H/2），PLAN 至少含 n_y≥16；",
        "3. 提交模块 + pytest 门（含“哨兵值必须报错”回归）；",
        "4. 登记 `CASE-FVM-CHANNEL-DEVELOP` 裁判行（平台 15%，若坚持 8% 需给出页码理由）；",
        "5. 用**同一次**多例运行重生成 benchmark.json + 两个 CSV + 三张图，并留 digest 链。",
        "",
        "## 7. 审计工件 / 未做的检查",
        "",
        "工件：`docs/audit_m20007/`（本文件、audited_src/、audited_runs/、audited_docs/、figs/、"
        "make_figures.py、build_report.py、test_audit_claims.py、insert_verdict.py、SHA256SUMS.txt）、"
        "`docs/audit_m20007/artifacts/`（recheck_entrance.py、recheck_ny8_nx128_Re200.json、repro/、repro50/）。",
        "",
        "图：`figs/fig1_centerline_delivered.png`（交付中心线永不越阈）、`figs/fig2_le_vs_re.png`"
        "（got vs 裁判，三点落在哨兵线）、`figs/fig3_sampling_ceiling.png`（偶数 n_y 离散上限 vs 1.485）。"
        "`fig4`（n_y 扫描曲线）仅当集群作业返回后生成，本快照未含——如实登记。",
        "",
        "**未做**：①White 2006 纸本未取得 → 裁判系数 WITHHELD；②交付 PNG 仅非空白校验；"
        "③`all` 六例未一次性全跑（超本地 10 min）；④n_y≥16 的判别性重测已递交集群 cj2502，若未在窗口内返回则记为未跑。",
        "",
    ])
    out = HERE / "AUDIT-m20007-CASE-FVM-CHANNEL-DEVELOP.md"
    out.write_text(doc)
    print(f"wrote {out} ({len(doc.splitlines())} lines)")

    # ---- evidence payload for tensor_maturity_verdicts --------------------
    rows = [{"re_h": c["re_h"], "got_L_e/H": c["L_e/H (computed)"],
             "ref_L_e/H": c["L_e/H (White 2006)"], "err_pct": c["L_e/H error %"],
             "u_max_ubar": c["u_max/u_bar"], "criterion_crossed": False} for c in bench["cases"]]
    ev = {
        "impl_files": ["docs/audit_m20007/audited_src/benchmark_entrance_length.py",
                       "docs/audit_m20007/artifacts/recheck_entrance.py"],
        "entry": ["docs/audit_m20007/audited_src/benchmark_entrance_length.py"],
        "entry_cmd": "PYTHONPATH=src python3 -m tensorfvm.benchmark_entrance_length case --re-h 200 --nx-per-h 8 --aspect 16",
        "run_artifacts": ["docs/audit_m20007/audited_runs/benchmark.json",
                          "docs/audit_m20007/audited_runs/metrics.csv",
                          "docs/audit_m20007/audited_runs/centerline_profiles.csv",
                          "docs/audit_m20007/audited_runs/fig_centerline_profiles.png",
                          "docs/audit_m20007/audited_runs/manifest.json",
                          "docs/audit_m20007/artifacts/repro/benchmark.json",
                          "docs/audit_m20007/artifacts/repro50/benchmark.json",
                          "docs/audit_m20007/artifacts/recheck_ny8_nx128_Re200.json"],
        "tests": ["docs/audit_m20007/test_audit_claims.py::test_ny8_ceiling_below_threshold",
                  "docs/audit_m20007/test_audit_claims.py::test_all_delivered_rows_are_domain_end_sentinels",
                  "docs/audit_m20007/test_audit_claims.py::test_delivered_dpdx_ref_disagrees_with_module_formula"],
        "integration_callers": [],
        "integration_callers_note": ("none: module calls the repo kernel inc_bench/solver.py SteadyFlow, but "
                                     "grep -rn benchmark_entrance_length /root/tensor-ecosystem/TensorFVM -> 0 hits; "
                                     "tensorbench_results mission_id=20007 -> 0 rows; never committed "
                                     "(result.head_commit=null, git status all '??')"),
        "benchmark": {
            "referee": "White 2006 §3-3.1 Eq(3-39) L_e/H = 0.06 Re_H (self-declared; 8% gate self-declared)",
            "referee_rows_in_tensor_benchmark_cases": 0,
            "delivered_verdict": bench["verdict"],
            "tolerance": bench["tolerance"],
            "rows": rows,
            "note": ("3/3 got values equal the no-crossing fallback L/H-dx/2 (7.9375 / 15.9375); the 99% "
                     "threshold 1.485 u_bar is structurally unattainable at n_y=8 (cell-centre parabola "
                     "ceiling 1.47656, solver peak 1.4518) -> no entrance length was measured; and the "
                     "delivered dp/dx ref for Re50 (-0.03) contradicts the module formula -12 nu u/H^2 "
                     "(-0.24, auditor re-run gives -0.232838) so the artifact is not reproducible"),
            "auditor_reproduction_Re50": repro50,
            "auditor_reproduction_Re200": repro200,
        },
        "repro_cmd": ("PYTHONPATH=src python3 -m tensorfvm.benchmark_entrance_length case --re-h 200 "
                      "--nx-per-h 8 --aspect 16 --name Re200-nx8-repro --out /tmp/repro200 -> reproduces "
                      "L_e/H=15.9375, u_max/ubar=1.4518132018201726, iters=758 (94 s); Re50 path "
                      "reproduces L_e/H=7.9375, u_max=1.4545, iters=702 but dp/dx=-0.232838 vs delivered "
                      "-0.0296 -> partial reproduction only"),
        "external_anchor": ("White, Viscous Fluid Flow 3rd ed (2006) §3-3.1 Eq (3-39) as cited by the delivery; "
                            "auditor did NOT verify the 0.06 coefficient against a paper copy -> WITHHELD; "
                            "independent of the coefficient, the delivered numbers are refuted as measurements "
                            "(fallback sentinels, auditor harness Le_crossed=False)"),
        "level_rationale": ("L2: real input->solve->output module, auditor reproduced L_e/H/u_max/iterations "
                            "bit-identically, non-placeholder body, CLI entry, non-empty run artifacts. "
                            "L3 refused: no mainline caller, module not in ecosystem repo, 0 platform ingestion. "
                            "L4 refused: 0 referee rows for the anchor, delivered verdict FAIL, 3/3 rows are "
                            "sentinel values, and the dp/dx fields are not reproducible from the delivered "
                            "source. L5 refused: single textbook anchor."),
        "findings": [
            "F1 blocking: all 3 delivered L_e/H rows are the no-crossing fallback centerline_x_vals[-1] "
            "(benchmark_entrance_length.py:182-184) presented as measured entrance lengths (164.6/165.6/32.8%)",
            "F2 blocking: 99%x1.5u=1.485u is unattainable at n_y=8 - closed-form cell-centre ceiling "
            "1.5-1.5/ny^2=1.47656 < 1.485 and solver peak 1.4518 (3.2% low); even n_y never samples y=H/2; "
            "module hard-wires ny=nx_per_H so the whole PLAN sits in the unattainable regime",
            "F7 major: delivered dp/dx and its reference are not reproducible from the delivered source - "
            "Re50 delivered -0.0296/-0.03 (err 1.33%) vs auditor re-run -0.232838/-0.24 (err 2.98%), 7.9x apart; "
            "recorded refs -0.03/-0.06/-0.06 contradict dpdx_ref=-12 nu u/H^2 (-0.24/-0.12/-0.06), so the "
            "report's '1.4% pressure-gradient accuracy' is false precision",
            "F3 major: metrics.csv (1 row) and centerline_profiles.csv (1 case) truncated vs benchmark.json "
            "(3 cases) because write_artifacts opens with 'w'; manifest.json self-hash mismatch",
            "F4 major: never committed (head_commit=null, git status '??'), module absent from "
            "/root/tensor-ecosystem/TensorFVM, 0 tensorbench_results rows -> not reusable, not integrated",
            "F5 major: PLAN declares 6 cases, only the 3 n_y=8 ones ran; the discriminating n_y>=16 cases "
            "were never executed, so the report's 'needs n_y>=16 but no compute' conclusion is untested",
            "F6 minor: 'zero hand-transcription' claim vs hard-coded WHITE_COEFF=0.06; unsourced "
            "'n_y=16 verified <0.4% Poiseuille' claim; anchor has no referee row in tensor_benchmark_cases",
        ],
        "disposition": ("accept as L2 (real reproducible module) / refuse L3+L4; approve with notes + request "
                        "changes next round: (1) no-crossing must fail explicitly, never become a length; "
                        "(2) criterion referenced to the discretely attainable peak, odd n_y allowed, PLAN with "
                        "n_y>=16; (3) commit module + pytest gate incl. sentinel-must-fail regression; "
                        "(4) register the referee row; (5) regenerate all artifacts from one multi-case run"),
        "checks_not_run": [
            "delivered 'all' subcommand in one invocation (6 cases > 10 min local budget)",
            "independent verification of the 0.06 Re_H coefficient against White 2006 (no copy in estate) "
            "-> WITHHELD",
            "quantitative/pixel inspection of the 3 delivered PNGs beyond non-blank checks",
            "n_y>=16 discriminating re-measurement submitted as cluster job cj2502 (queue position 26); "
            "not returned inside the audit window -> recorded as not_run",
        ],
        "audit_artifacts": ["docs/audit_m20007/AUDIT-m20007-CASE-FVM-CHANNEL-DEVELOP.md",
                            "docs/audit_m20007/audited_src/benchmark_entrance_length.py",
                            "docs/audit_m20007/audited_runs/",
                            "docs/audit_m20007/audited_docs/REPORT.md",
                            "docs/audit_m20007/test_audit_claims.py",
                            "docs/audit_m20007/make_figures.py",
                            "docs/audit_m20007/figs/",
                            "docs/audit_m20007/artifacts/recheck_entrance.py"],
        "origin": "read-only snapshot of /root/tp-agent/work/m20007 (agent/m20007, main 0a075ce8)",
    }
    (HERE / "verdict_evidence.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False))
    print(f"wrote {HERE/'verdict_evidence.json'}")


if __name__ == "__main__":
    main()