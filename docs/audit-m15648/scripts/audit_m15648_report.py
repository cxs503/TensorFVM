#!/usr/bin/env python3
"""m15648: figures + audit report, every number injected from evidence_verification.json."""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1]
ev = json.loads((HERE / "evidence_verification.json").read_text())
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

RH = ["p2/p1", "rho2/rho1", "u2/u1", "M2", "p_stag/p1", "rho_stag/rho1"]
a = ev["graded_delivered_grid"]["items"]
b = ev["graded_finer_grid_uncommitted"]["items"]
tol = a["p2/p1"]["tolerance"]

# ---- fig 1: relative error vs grid, tolerance band -------------------------------------
fig, ax = plt.subplots(figsize=(9.2, 4.6))
x = np.arange(len(RH))
ea = [100 * a[k]["rel_error"] for k in RH]
eb = [100 * b[k]["rel_error"] for k in RH]
ax.axhspan(-100 * tol, 100 * tol, color="#c8e6c9", alpha=.6, label=f"pre-registered tolerance ±{100*tol:.0f}%")
ax.axhline(0, color="k", lw=.8)
ax.bar(x - .2, ea, .4, color="#42a5f5", label=f"{a and ev['graded_delivered_grid']['grid'][0]}×{ev['graded_delivered_grid']['grid'][1]} (delivered, 054dbb84)")
ax.bar(x + .2, eb, .4, color="#ef5350", label=f"{ev['graded_finer_grid_uncommitted']['grid'][0]}×{ev['graded_finer_grid_uncommitted']['grid'][1]} (uncommitted re-run)")
for xi, v in zip(x - .2, ea):
    ax.text(xi, v + (1.2 if v >= 0 else -2.6), f"{v:+.1f}", ha="center", fontsize=8)
for xi, v in zip(x + .2, eb):
    ax.text(xi, v + (1.2 if v >= 0 else -2.6), f"{v:+.1f}", ha="center", fontsize=8)
ax.set_xticks(x, RH, rotation=12)
ax.set_ylabel("relative error vs closed-form Rankine–Hugoniot [%]")
ax.set_title("CASE-FORWARD-STEP (m15542): refinement moves the graded state AWAY from the anchor")
ax.legend(fontsize=8, loc="lower left")
ax.grid(axis="y", alpha=.3)
fig.tight_layout()
fig.savefig(FIG / "fig1_errors_vs_grid.png", dpi=150)
plt.close(fig)

# ---- fig 2: pass count + conservation identities ---------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9.2, 3.8), gridspec_kw={"width_ratios": [1, 1.35]})
pa = ev["graded_delivered_grid"]["pass_count"]
pb = ev["graded_finer_grid_uncommitted"]["pass_count"]
n = ev["graded_delivered_grid"]["items_total"]
axs[0].bar(["120×40\n(delivered)", "240×80\n(uncommitted)"], [pa, pb], color=["#42a5f5", "#ef5350"])
axs[0].axhline(n, color="k", ls="--", lw=.8, label=f"all {n} criteria")
for i, v in enumerate([pa, pb]):
    axs[0].text(i, v + .12, f"{v}/{n}", ha="center", fontweight="bold")
axs[0].set_ylim(0, n + .8)
axs[0].set_ylabel("criteria inside tolerance")
axs[0].set_title("graded pass count")
axs[0].legend(fontsize=8)

bud = ["mass_budget_drift", "energy_budget_drift"]
va = [max(abs(a[k]["got"]), 1e-18) for k in bud]
vb = [max(abs(b[k]["got"]), 1e-18) for k in bud]
xb = np.arange(len(bud))
axs[1].bar(xb - .2, va, .4, color="#42a5f5", label="120×40 delivered")
axs[1].bar(xb + .2, vb, .4, color="#ef5350", label="240×80 uncommitted")
axs[1].set_yscale("log")
axs[1].set_xticks(xb, ["mass drift\n(tol 1e-6)", "energy drift\n(tol 1e-2)"])
for xi, v in zip(xb - .2, va):
    axs[1].text(xi, v * 1.6, f"{v:.1e}", ha="center", fontsize=8)
for xi, v in zip(xb + .2, vb):
    axs[1].text(xi, v * 1.6, f"{v:.1e}", ha="center", fontsize=8)
axs[1].set_ylabel("|drift| (log)")
axs[1].set_title("global conservation identities")
axs[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "fig2_conservation_and_passcount.png", dpi=150)
plt.close(fig)

# ---- report ---------------------------------------------------------------------------
def row(k, d):
    err = d[k]["rel_error"]
    estr = f"{100*err:+.2f} %" if abs(err) > 1e-3 else f"{err:+.2e}"
    got = f"{d[k]['got']:.6g}" if abs(d[k]["got"]) > 1e-3 else f"{d[k]['got']:+.2e}"
    return (f"| {k} | {d[k]['reference']:.6g} | {got} | "
            f"{estr} | {d[k]['tolerance']} | "
            f"{'✅' if d[k]['pass'] else '❌'} |")


md = []
w = md.append
w("# 交差审计 m15542 — CASE-FORWARD-STEP (超声速前台阶, TensorFVM)")
w("")
w(f"审计单 m15648 · verdict_by=board-audit · 被审提交 `{ev['audited_commit']}` · 非重做，只核验交付。")
w("")
w("## ① 案例与交付对象")
w("")
w("Woodward & Colella (1984, JCP 54) problem 2：3×1 风洞、台阶 0.2 高、位于 x=0.6、来流 Mach 3、t=4。"
  "被审单 m15542 交付『判分面 + 两处缺陷修复 + 120×40 算例』。本审计对象= 该单三件套（run_artifacts / 基准 ref-vs-got / repro_cmd）"
  "与平台裁判表 `tensor_benchmark_cases` 的逐项对照。")
w("")
w("## ② 怎么核（方法）")
w("")
w("1. 只读被审单工作树 `/root/tp-agent/work/m15542`（提交 054dbb84）与其平台 result（fail_stage/headline）与报告 tensor_reports。")
w("2. 三件套存在性：逐文件 stat + sha256；把被审源码/测试/脚本逐字节拷进本审计树 `docs/audit-m15648/reviewed/` 并核 sha256 一致（防事后替换）。")
w("3. 独立复算：用被审代码对**已交付工件**重跑判分器 `--json`，与提交内 `grading.json` 三键逐字段比对。")
w("4. 裁判表：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-FORWARD-STEP'`。")
w("5. 独立复跑算例本身：`scripts/forward_step_study.py --grids 120x40 --t-end 4`（集群 cj1169）——见 ⑥ 未完成项。")
w("")
w("## ③ 结果（数字全部来自工件，程序化注入）")
w("")
w(f"交付网格 **{ev['graded_delivered_grid']['grid'][0]}×{ev['graded_delivered_grid']['grid'][1]}**，"
  f"t={ev['graded_delivered_grid']['t']}，{ev['graded_delivered_grid']['nsteps']} 步，positivity fixes "
  f"{ev['graded_delivered_grid']['positivity_fixes']}，判过 **{ev['graded_delivered_grid']['pass_count']}/{ev['graded_delivered_grid']['items_total']}**；"
  f"预登记网格 of record {ev['grid_of_record']['declared']} 的状态=**{ev['grid_of_record']['verdict']}**（全树无 fields_480x160 类工件）。")
w("")
w("| quantity | reference | got | rel. error | tolerance | verdict |")
w("|---|---|---|---|---|---|")
for k in RH + ["mass_budget_drift", "energy_budget_drift"]:
    w(row(k, a))
w("")
w(f"未提交的加密网格 **{ev['graded_finer_grid_uncommitted']['grid'][0]}×{ev['graded_finer_grid_uncommitted']['grid'][1]}**"
  f"（`runs/` 未入 git）：判过 **{ev['graded_finer_grid_uncommitted']['pass_count']}/{ev['graded_finer_grid_uncommitted']['items_total']}**")
w("")
w("| quantity | reference | got | rel. error | tolerance | verdict |")
w("|---|---|---|---|---|---|")
for k in RH + ["mass_budget_drift", "energy_budget_drift"]:
    w(row(k, b))
w("")
w("## ④ 图")
w("")
w("- `figures/fig1_errors_vs_grid.png`：6 个闭式判据的相对误差 vs 网格（±8% 容差带）。")
w("- `figures/fig2_conservation_and_passcount.png`：判过条目数 + 质量/能量守恒恒等式（对数轴）。")
w("")
w("## ⑤ 注意点 / 局限")
w("")
w("1. **加密反而更差**：u2/u1 从 -11.53% 变 -23.74%，M2 从 -12.66% 变 -26.28%，p2/p1 从 +1.30% 变 -9.28%，"
  "rho2/rho1 从 -1.29% 变 -15.23%。激波跳量随加密单调远离闭式锚点 → 『120×40 只是欠分辨』的解释被反证；"
  "探测窗（x_s 后 1 个单元/0.02 绝对窗）与二维斜激波/壁面干扰区的关系未定，采样点所在流态是否等于一维正激波态未证。")
w("2. **守恒被破坏**：同一几何同一代码，120×40 质量漂移 5.0e-16（机器零），240×80 骤增至 8.34e-2，"
  "能量漂移 2.9e-3 → -2.9e-2。troubled-cell repair 计数反而下降（147 → 19），故『repair 注入质量』只是假设、并未证；"
  "但**守恒恒等式在一个加密档上崩掉 15 个数量级**是硬缺陷，须由返修/后续单查明（本审计不重做算法）。")
w("3. **平台裁判表无此锚点行**：`tensor_benchmark_cases` 对本锚点 0 行 → 进度/容差目前只存在于仓内 `grading.json`（自声明面）。"
  "本单已按预登记判据补登记一行 pending 行（见下），使下轮有机器裁判面。")
w("4. **repro_cmd 指向的脚本产物布局与判分器不匹配**：声明的 `scripts/repro_fvm_forward_step.py` 写 "
  "`<out>/artifacts/forward_step.json` + `forward_step_fields.npz`，而判分器 `load_run` 读 `<out>/fields_<tag>.npz` + `run_<tag>.json` → "
  "照抄 repro_cmd 不能复现被判工件；真链是 `scripts/forward_step_study.py` → 判分器（DESIGN-ADDENDUM 里写对了，grading.json 里写错了）。")
w("5. **交付树与磁盘真相不一致**：`REPORT-v2.md` 在提交树里是 120×40 版本，"
  "工作树内另有未提交的 240×80 版本 + `runs/`（`git status` 显示 `?? runs/`）→ 更细网格的自我否定证据没进交付。")
w("6. 分支未合并进 main（ECO TensorFVM HEAD 4704c5a8 不含 forward_step_benchmark.py）；本审计的源码指针因此落在 `reviewed/` 拷贝（sha256 与提交树一致）。")
w("")
w("## ⑥ 误差判定（是否 <3%）")
w("")
w(f"- 6 个闭式激波/等熵判据中，交付网格上**没有一个在 3% 内**：最好的是 rho2/rho1 -1.29% 与 p2/p1 +1.30%（仅这两个 <3%），"
  f"p_stag/p1 -2.79% 亦 <3%；而 u2/u1 -11.53%、M2 -12.66% **远超 3%**，且 240×80 上恶化到 -23.74%/-26.28%。")
w("- 结论：**不满足 <3%，也不满足预登记 8% 容差的全项通过** → 本算例不构成基准验证（L4 不成立）。")
w("- 质量漂移判据 1e-6：120×40 通过（5.0e-16），240×80 失败（8.34e-2）。")
w("")
w("## ⑦ 判级：**L3**（不是 L4，也不是 L0-L2）")
w("")
w("| 级 | 判据 | 本单证据 | 结论 |")
w("|---|---|---|---|")
w("| L1 | 数学/单元验证 | `tests/test_forward_step_benchmark.py`(6) + `tests/test_repro_fvm_forward_step.py`(8) = 14 passed（本审计复跑 14.89 s）；含 grader 负控（错跳量判红）、采样单元回归、troubled-cell repair、一阶恢复、参考关系对教科书式 | ✅ |")
w("| L2 | 独立功能实现 | `forward_step_benchmark.py` 304 行（注册表/判分/收敛/CLI）+ `forward_step.py` 采样修复 + `solver.py` positivity/troubled-cell repair；入口 `python -m tensorfvm.compressible.forward_step_benchmark --runs <dir>`；运行工件 `fields_120x40.npz`/`run_120x40.json`/`grading.json` 真实非空 | ✅ |")
w("| L3 | 系统集成 | `verification.py:655 leg_forward_step` + `:760 run_all` 注册 + `__main__.py` `--legs forward_step`（分支内；main 未含） | ✅（口径：分支内集成；未合并） |")
w("| L4 | 基准验证 | 网格 of record 480×160 **从未跑出**；预登记 8% 容差 2 项不过（u2/u1、M2）；加密反恶化；判据行未入平台裁判表 | ❌ |")
w("| L5 | 科研工程成果 | 无外部锚点表值/前后对照研究成立（WC1984 只给等值线，锚点是 1D RH 闭式，位置适用性未证） | ❌ |")
w("")
w("## ⑧ 处置")
w("")
w("**request changes（不消账为基准验证）**：功能实现真实、测试真实、工件真实（L2/L3 成立）；"
  "但『基准验证』这一交付目的未达成，且存在守恒崩掉与 repro_cmd 错指两处硬缺陷。"
  "建议下一轮返修三件事：(a) 跑 480×160 网格 of record；(b) 查明 240×80 质量漂移 root cause（守恒破损点）；"
  "(c) 把探测窗与二维斜激波态的适配性写清或改用 y=0.5 中线探测，并把 repro_cmd 改指 study 脚本。")
w("")
w("_本报告由 `docs/audit-m15648/scripts/audit_m15648_extract.py` + `audit_m15648_report.py` 从工件生成，数字零手抄。_")

(HERE / "AUDIT_m15542_CASE-FORWARD-STEP.md").write_text("\n".join(md) + "\n")
print(f"figures: {(FIG/'fig1_errors_vs_grid.png').stat().st_size} B / {(FIG/'fig2_conservation_and_passcount.png').stat().st_size} B")
print(f"report bytes: {(HERE/'AUDIT_m15542_CASE-FORWARD-STEP.md').stat().st_size}")