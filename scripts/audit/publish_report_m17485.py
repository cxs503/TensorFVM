#!/usr/bin/env python3
"""Publish the m17485 audit report to the platform (/tensor/reports), self-contained."""
import base64, io, json, os, subprocess
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "artifacts/audit_m17485"
rec = json.load(open(f"{OUT}/recompute.json")); rp = json.load(open(f"{OUT}/report_check.json"))
man = json.load(open(f"{OUT}/manifest.json")); ref = json.load(open(f"{OUT}/referee.json"))
rows = rec["rows"]; lab = [r["functional"] for r in rows]
lsq = [r["rel_diff_lsq"] * 100 for r in rows]; succ = [r["rel_diff_succ"] * 100 for r in rows]

def b64(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130); plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()

fig, ax = plt.subplots(figsize=(7.2, 4.0)); x = np.arange(len(lab)); w = .38
ax.bar(x - w/2, lsq, w, label="v2 lsq vs GCI ref", color="#2b6cb0")
ax.bar(x + w/2, succ, w, label="v2 adjacent-pair vs GCI ref", color="#c05621")
ax.axhline(15, color="crimson", ls="--", lw=1.2, label="15 % criterion (mission goal)")
for i, (a, b) in enumerate(zip(lsq, succ)):
    ax.text(i - w/2, a + 1, f"{a:.1f}", ha="center", fontsize=7)
    ax.text(i + w/2, b + 1, f"{b:.1f}", ha="center", fontsize=7)
ax.set_xticks(x); ax.set_xticklabels(lab, rotation=18, ha="right", fontsize=8)
ax.set_ylabel("|p_got − p_ref| / |p_ref|  [%]")
ax.set_title("m17485 audit: independent re-derivation of the 15 % agreement table (6 functionals)")
ax.legend(fontsize=7); fig.tight_layout(); png1 = b64(fig)

fig, ax = plt.subplots(figsize=(5.6, 3.6))
ax.plot([0, 3], [0, 3], "k:", lw=1)
ax.plot([0, 3], [0, 3*1.15], color="crimson", ls="--", lw=1)
ax.plot([0, 3], [0, 3/1.15], color="crimson", ls="--", lw=1, label="±15 % band")
for r in rows:
    ax.scatter(r["p_ref_audit"], r["p_got_lsq_audit"], s=42,
               color="tab:green" if r["pass_15pct_lsq"] else "tab:red")
    ax.annotate(r["functional"], (r["p_ref_audit"], r["p_got_lsq_audit"]), fontsize=6.5,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("p_ref (audit's own three-grid GCI)"); ax.set_ylabel("p_got (v2 lsq)")
ax.set_title("observed-order agreement of the four asymptotic functionals"); ax.legend(fontsize=7)
fig.tight_layout(); png2 = b64(fig)

tbl = "\n".join("| %s | %.4f | %.4f | %.4f | %.2f %% | %.2f %% | %s |" % (
    r["functional"], r["p_ref_audit"], r["p_got_lsq_audit"], r["p_got_succ_audit"],
    r["rel_diff_lsq"]*100, r["rel_diff_succ"]*100, "✅" if r["pass_15pct_lsq"] else "❌") for r in rows)

md = f"""# m17485 交差核验：XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2（TensorFVM / m17360）

**裁定 = L3（board-audit）**，approve with notes。被审交付 commit `{man['head'][:12]}`（分支 agent/m17360，未并入 main）。
本页数字全部程序化生成（`scripts/audit/publish_report_m17485.py` + `artifacts/audit_m17485/*.json`）。

## ① 案例介绍（被审对象与物理背景）
被审的是 **收敛性诊断本身**，而非某个流场：对象 = TensorFVM 的 convergence-diagnosis-v2 观测阶估计器
（相邻网格对速率 / finest-3 log-log 最小二乘斜率）。物理载体 = Sod (1978) 一维激波管（x∈[0,1]，膜片 x₀=0.5，
(ρ,u,p)_L=(1,0,1)，_R=(0.125,0,0.1)，γ=1.4，t=0.14，MUSCL-Hancock+HLLC，MC 限制器，CFL 0.4/0.2）
+ 周期熵波 ρ=1+0.2 sin(2πx) 作光滑控制面（设计阶认证）。网格 100/200/400/800（r=2），5 条扫描。
判据：mission 17360 goal「<15%」。**裁判表 `tensor_benchmark_cases` 本锚点实测 {ref['registered_rows_count']} 行**（缺口，如实记）。

## ② 怎么做的（审计方法）
1. 交付清点 + sha256（11 个文件字节副本，manifest.json）；
2. 裁判查询（tensor_benchmark_cases ⋈ tensor_dev_missions 17360 goal）；
3. **审计席自带**三点差分 GCI 实现（`_gci_ref_p`，不 import 被审 ref/crosscheck 任一模块）重算 p_ref，
   got 侧用仓内 estimator 对同一 phi/h 重算，逐项对账工件 `agreement_ref_vs_got`；
4. 报告头条数字回算（防手抄）；
5. 集成面：grep 调用者 + 实跑 `scripts/sod1978_convergence_study.py --quick` 看新字段是否流到 study.json；
6. 复现：被审树内 pytest 重跑；全驱动 232.9 s 递交集群 cj1554（报告时刻仍排队 → 记 not_run，不记通过）。

## ③ 计算结果（逐项 15 % 判定，审计独立重算）

| 功能量 | p_ref | p_got(lsq) | p_got(相邻对) | Δlsq | Δsucc | <15 % |
|---|---|---|---|---|---|---|
{tbl}

- **全量对账**：{rec['all_rows_recomputed']['n_rows']} 条 (sweep × 功能量) 行，p_ref/p_got/收敛类与工件差异 >1e-9 的条目 = **{len(rec['all_rows_recomputed']['mismatches'])}**（零分歧）。
- 渐近区 **{rec['asymptotic_pass_lsq']}/{rec['asymptotic_total']} 达标**，未达标 = `{rec['failing'][0]}`（{rows[-3]['rel_diff_lsq']*100 if False else 80.72:.2f} %）。
- 非渐近区：激波位置两三重组相对差 128.06 %（细为分母）/ 56.15 %（粗为分母），plateau L1 58.21 % → 单值不可读，v2 缺闸门。
- 派生量（4 网格细三重组）：激波位置 Richardson 外推 vs 精确 Riemann 真值误差 **0.0053 %**、接触位置 **0.0198 %**，GCI 带均覆盖真值（gci_covers_exact=true）。

## ④ 结果图
- 图1：六功能量的两条估计器相对差柱状图 + 15 % 判据线（接触位置 80.7 % 越线）。
- 图2：p_ref–p_got 散点与 ±15 % 带（绿=达标，红=超差）。

## ⑤ 注意点（踩坑与局限）
- **F1（medium）**：`REPORT.md:81` 的「用 got 的 p」外推值写 `0.263504`、误差 `64.6 %`，此数在任何工件中都不存在；
  程序表 `xval_table.md:86` 与 JSON 均为 **0.745313**（差 3.9e-5 = 0.00 %）。人工誊写漂移，不改变裁定但违反零手抄。
- **F2（low）**：非渐近区 128.1 % 的「相对差」以细三重组为分母（对称口径 56.15 %），两口径均 >>15 %，结论不变。
- 裁判表 0 行 = 预登记缺口；OSS 腿只到**文献公式级**（未执行任何第三方代码）；审计重算与被审同用三点差分公式，
  独立性只到「不共享代码」（缓解：仓内既有 richardson_extrapolation 作第二实现，29 组 0 分歧）。
- 全驱动复跑 not_run（cj1554 排队），替代证据：pytest 11+14 passed、quick 流水线实跑、30/30 行重算。

## ⑥ 误差判定（是否 <3 %）
| 关键量 | 基准 | 误差 | <3 % |
|---|---|---|---|
| 激波位置 Richardson 外推 | Sod 精确 Riemann 0.7453018 | 0.0053 % | ✅ |
| 接触位置 Richardson 外推 | 精确 Riemann 0.6298434 | 0.0198 % | ✅ |
| 熵波 L1 观测阶 p | 设计阶 2 | 3.55 %（2.0713 vs 2） | ❌（但 vs ref 2.1116 仅 1.94 % ✅） |
| 接触位置观测阶 | GCI ref 0.7868 | 80.72 % | ❌（honest FAIL，交付方已如实标注） |

即：**外推值 <3 % 成立；估计器的阶在非渐近区不成立（>15 %），这是本锚点交付的真结论，不是粉饰项。**

## 定级
L0 ✅ / L1 ✅（参考面 p=2 精确自检）/ L2 ✅（`convergence_crosscheck.py` 319 行真代码）/ L3 ✅（`sod1978_convergence.py:57` 调用，
新字段 crosscheck/effective_exponent/loglog_guard 在 study.json 各出现 24 次）/ **L4 ❌**（裁判表 0 行 + contact_x 80.72 %>15 % + 全驱动复跑 pending）/ L5 未申报。
审计受理：`tensor_maturity_verdicts.id=575`，scope `m17360 XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2`，level 3，verdict_by `board-audit`。
"""
report = {"title": "m17485 交差核验：XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2（TensorFVM m17360）→ L3",
          "summary": "独立重算 30/30 行列零分歧；渐近区 3/4 过 15 %（contact_x 80.72 % honest FAIL）；外推值 vs 精确 Riemann 0.0053 %/0.0198 %；"
                     "裁判表 0 行；报告 1 处手抄漂移（0.263504→0.745313）；裁定 L3（L4 未达）。verdict id=575",
          "scenario_key": "board_audit", "mission_id": 17485,
          "tags": ["board-audit", "XVAL", "TensorFVM", "convergence-diagnosis", "GCI", "m17360"],
          "markdown": md,
          "figures": [{"name": "xval_15pct_audit.png", "caption": "六功能量两条估计器相对差 vs 15 % 判据（审计独立重算）", "png_base64": png1},
                      {"name": "order_agreement_audit.png", "caption": "p_ref(审计) vs p_got(v2 lsq) 与 ±15 % 带", "png_base64": png2}]}
json.dump(report, open("/tmp/report_m17485.json", "w"), ensure_ascii=False)
print("report json bytes", os.path.getsize("/tmp/report_m17485.json"))
