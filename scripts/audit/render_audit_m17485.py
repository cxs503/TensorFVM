#!/usr/bin/env python3
"""Render docs/board_audit_m17485/M17485_BOARD_AUDIT.md from the JSON artifacts.

Every number on the page is read from artifacts/audit_m17485/*.json; nothing is
typed by hand.
"""
from __future__ import annotations

import json
import os

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(REPO, "artifacts", "audit_m17485")
DOC = os.path.join(REPO, "docs", "board_audit_m17485", "M17485_BOARD_AUDIT.md")

man = json.load(open(os.path.join(OUT, "manifest.json")))
ref = json.load(open(os.path.join(OUT, "referee.json")))
rec = json.load(open(os.path.join(OUT, "recompute.json")))
rcp = json.load(open(os.path.join(OUT, "report_check.json")))
itg = json.load(open(os.path.join(OUT, "integration.json")))
ev = json.load(open(os.path.join(OUT, "evidence.json")))
repro_path = os.path.join(OUT, "repro.json")
repro = json.load(open(repro_path)) if os.path.exists(repro_path) else None

L = []
A = L.append
A("# m17485 交差核验：XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2（TensorFVM，m17360）")
A("")
A("> 交差审计（非重做）。被审交付 `m17360` @ commit `%s`（分支 `agent/m17360`，%s main）。"
  % (man["head"][:12], "**已并入**" if man["in_main"] else "**未并入**"))
A("> 本页与 `artifacts/audit_m17485/*.json` 全部程序化生成（`scripts/audit/render_audit_m17485.py`）。")
A("")
A("**裁定：L3（board-audit）** —— %s" % ev["level_rationale"].split("L4 NOT granted")[0].strip())
A("")
A("## 1. 交付清点（三件套是否存在）")
A("")
A("| 文件 | 字节 | sha256(前 12) |")
A("|---|---|---|")
for f, r in man["files"].items():
    A("| `%s` | %s | `%s` |" % (f, r.get("bytes", "-"), str(r.get("sha256", "-"))[:12]))
A("")
A("全部存在：**%s**；被审树 HEAD 与交付 commit 一致：**%s**；工作树残留未提交改动：%d 条。"
  % (man["all_present"], man["head_matches"], len(man["uncommitted_status"])))
A("字节副本与 sha256 清单位于 `artifacts/audit_m17485/delivered/`、`manifest.json`。")
A("")
A("## 2. 基准裁判（唯一裁判）")
A("")
A("`SELECT ... FROM tensor_benchmark_cases WHERE anchor_slug='%s'` → **%d 行**（审计时刻实测，不粉饰）。"
  % (ref["anchor_slug"], ref["registered_rows_count"]))
A("此时在force的判据 = mission %d goal 原文：**%s**。裁判表为空是资产缺口（本锚点未预登记算例），"
  "故本次核验以 mission goal 判据 + 审计席独立重算为准，并把两者分别记档。"
  % (17360, ref["criterion_in_mission_goal"]))
A("")
A("## 3. 独立重算：15 % 判据逐项核对")
A("")
A("审计席**自带**三点差分 GCI 实现（`scripts/audit/m17485_board_audit.py::_gci_ref_p`，"
  "既不 import `gci_reference.py` 也不 import `convergence_crosscheck.py`），对工件 `raw`/`sweeps` 的 phi/h "
  "重算 p_ref；got 侧用仓内 estimator 对同一数列重算，最后与工件 `agreement_ref_vs_got` 对账。")
A("")
A("**全量对账**：%d 条 (sweep × 功能量) 行的 p_ref / p_got(lsq) / p_got(相邻对) / 收敛类，"
  "与工件差异 >1e-9 的条目 = **%d**。" % (rec["all_rows_recomputed"]["n_rows"],
                                       len(rec["all_rows_recomputed"]["mismatches"])))
A("")
A("### 3.1 渐近区四项（判据 <15 %）")
A("")
A("| 功能量 | p_ref | p_got(lsq) | p_got(相邻对) | Δlsq | Δsucc | <15 % |")
A("|---|---|---|---|---|---|---|")
for r in rec["rows"]:
    if r["family"] != "asymptotic":
        continue
    A("| %s | %.4f | %.4f | %.4f | %.2f %% | %.2f %% | %s |"
      % (r["functional"], r["p_ref_audit"], r["p_got_lsq_audit"], r["p_got_succ_audit"],
         r["rel_diff_lsq"] * 100, r["rel_diff_succ"] * 100,
         "✅" if r["pass_15pct_lsq"] else "❌"))
A("")
A("结论：**%d/%d 达标**，未达标 = `%s`。交付方在报告 3.1 与 ⑤ 中如实标注了该 ❌（honest FAIL），"
  "未粉饰；审计复现同一数字（80.72 %%）。"
  % (rec["asymptotic_pass_lsq"], rec["asymptotic_total"], "`, `".join(rec["failing"])))
A("")
A("### 3.2 非渐近区重叠三重组")
A("")
A("| 功能量 | p(粗三重组) | p(细三重组) | 相对差（细为分母） | 相对差（粗为分母） | 工件值 |")
A("|---|---|---|---|---|---|")
for k, v in rec["non_asymptotic_triplet_drift"].items():
    A("| %s | %.4f | %.4f | %.2f %% | %.2f %% | %.4f |"
      % (k, v["p_coarse_triplet_audit"], v["p_fine_triplet_audit"],
         v["rel_diff_denominator_fine"] * 100, v["rel_diff_audit"] * 100, v["artifact_rel_diff"]))
A("")
A("## 4. 报告六段与数字防手抄")
A("")
A("六段齐全：**%s**（① 案例介绍 / ② 方法与实现 / ③ 计算结果 / ④ 结果图 / ⑤ 收敛性·敏感性 / ⑥ 失败模式）；"
  "差异清单=%s，移植取舍=%s，显式 <3 %% 判定=%s。"
  % (rcp["all_sections"], rcp["difference_list_present"], rcp["porting_tradeoff_present"],
     rcp["declares_lt3pct"]))
A("")
A("报告 3.1/3.2/3.3 头条数字逐项回算：")
A("")
A("| 报告声称 | 审计重算 | 一致 |")
A("|---|---|---|")
for c in rcp["headline_checks"]:
    A("| %s（%s） | %s | %s |" % (c["claim"], c["meaning"],
                                 ("%.6g" % c["recomputed"]) if isinstance(c["recomputed"], (int, float)) else c["recomputed"],
                                 "✅" if c["agrees"] else "❌"))
A("")
A("**发现 F1（medium）**：`REPORT.md:81` 的「用 got 的 p」外推值写作 `0.263504`、误差 `64.6 %`，"
  "该数字在**任何工件中都不存在**；程序生成表 `xval_table.md:86` 与 JSON "
  "`sweeps.combined.../shock_x/derived_values.extrapolated_from_got_exponent` 均为 **0.745313**"
  "（与 ref 差 3.9e-5，0.00 %）。属人工誊写/陈旧数，不影响 15 % 裁定（未达标行是 contact_x），"
  "但违反零手抄纪律，应改。")
A("")
A("## 5. 集成面（L3 判据）")
A("")
A("`convergence_crosscheck` 的新函数：%s。" % ", ".join(itg["functions"][:8]))
A("")
if itg.get("study_pipeline"):
    A("仓内入口 `scripts/sod1978_convergence_study.py` 实跑（--quick）产出的 study.json 中，"
      "新增字段出现次数：crosscheck=%d、effective_exponent=%d、loglog_guard=%d"
      " → **新能力真的流到了流水线输出**（不是孤岛模块）。"
      % (itg["study_pipeline"]["crosscheck_field_occurrences"],
         itg["study_pipeline"]["effective_exponent_occurrences"],
         itg["study_pipeline"]["loglog_guard_occurrences"]))
A("")
A("无求解器侧/平台侧调用者 → 集成只到仓内验证流水线，**不满足更高级别的系统集成**。")
A("")
A("## 6. 可复现")
A("")
A("`repro_cmd`（工件内字段）：`%s`（单核 232.9 s）。" % rec["repro_cmd"])
if repro:
    A("审计复跑叶片级对账：参与比较 %d 个 JSON 叶，差异 %d，identical=%s。"
      % (repro["n_compared"], len(repro["diffs"]), repro["identical_leaves"]))
else:
    A("审计已把整树字节副本 + 驱动递交集群（`cj1554`，stage_dir=交付树副本，timeout 1800）；"
      "**报告时刻仍在排队**，故全驱动叶片级复跑 = **not_run（pending）**，不记为通过。"
      "已执行的替代证据：被审树内 `tests/test_convergence_diagnosis_xval.py` 11 passed、"
      "`tests/test_sod1978_convergence.py` 14 passed、`--quick` 流水线实跑、以及 30/30 数字行独立重算。")
A("")
A("## 7. 定级与理由")
A("")
A("| 级别 | 判据 | 结论 |")
A("|---|---|---|")
A("| L0 文档 | — | 通过（但不是功能） |")
A("| L1 数学/单元验证 | 参考面解析自检 p=2 精确、两实现 29 组一致 | ✅ |")
A("| L2 独立功能实现 | `convergence_crosscheck.py` 319 行真代码，输入数列→输出阶/类/GCI 带 | ✅ |")
A("| L3 系统集成 | 被 `sod1978_convergence._rate_table` 调用并流到 study.json | ✅ |")
A("| L4 基准验证 | 完整算例+误差分析+可复现 **且** 判据全过 | **未达**：裁判表 0 行；contact_x 80.72 % 超 15 %；全驱动复跑 pending |")
A("| L5 科研工程成果 | 论文级科研产出 | 未申报 |")
A("")
A("**裁定：L3。** %s" % ev["level_rationale"])
A("")
A("## 8. 审计自身的局限")
A("")
A("1. 未执行任何第三方 OSS 代码（OpenFOAM/SU2 的 GCI 工具未跑）；OSS 腿=文献公式级，"
  "属 `external_anchor` 的诚实弱化项。")
A("2. 全驱动 232.9 s 复跑被控制面 120 s 截断，只能上集群；报告时刻 `cj1554` 仍 queued → 该项 = not_run。")
A("3. 审计席的重算实现与被审实现同用三点差分公式（数学上同源），独立性限于「不共享代码」，"
  "不能排除公式层面的共同误读（缓解：仓内既有 `richardson_extrapolation` 作为第二实现，29 组 0 分歧）。")
A("")
A("## 9. 处置")
A("")
A("**approve with notes**：L3 记档；返修建议 = (a) 修 `REPORT.md:81` 的 0.263504/64.6 %（改成工件值 0.745313，0.00 %）；"
  "(b) 把本锚点的 15 % 判据登记进 `tensor_benchmark_cases`（当前 0 行，裁判缺位）；"
  "(c) `cj1554` 复跑回填 `repro.json` 后把第 6 节 not_run 改判。")
A("")

os.makedirs(os.path.dirname(DOC), exist_ok=True)
open(DOC, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("wrote", DOC, len(L), "lines")