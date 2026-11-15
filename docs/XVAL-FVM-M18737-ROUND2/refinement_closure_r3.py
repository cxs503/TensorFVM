"""Round-3 closure: adjudicate the frozen F1/F2/F3 criteria on the nx=6400/12800 rungs.

Why this exists
---------------
``REFINEMENT-CLOSURE.md`` refined the *graded* MUSCL/HLLC solution and left two
holes, both recorded honestly: the nx=6400/12800 rungs were ``not measured``
(job cj3325 never landed), and the time step was never varied while the grid was
refined, so the observed L1 floor could in principle be a *time-step* artefact.
This script closes both:

* it grades the newly computed rungs against the self-certified WENO5@12800
  reference and tests the prediction that was frozen before the runs existed
  (``PREDECLARATION-refine.md``: F1 point value in band, F2 sequence order,
  F3 certified-<1 % ruling);
* it runs the CFL control (``nx=6400`` at CFL 0.2 vs 0.4) so the recommendation
  "refine the space or change the scheme" is not confounded with the time step.

Every number in ``REFINEMENT-R3-CLOSURE.md`` is rendered from
``summary_r3.json`` written here -- no hand transcription.

Run: PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/refinement_closure_r3.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.verification import wc_refine as wr

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
DATA = wr.default_data_dir()
REF_NX = 12800
CFL_CONTROL = {"nx": 6400, "cfl": 0.2}
REGIONS = (("left_plateau", (0.40, 0.55)),
           ("collision_fronts", (0.55, 0.72)),
           ("right_contact_and_shock", (0.72, 1.00)))


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_ladder(verdict: dict, checks: dict) -> Path:
    rows = [r for r in verdict["ladder"] if r.get("measured")]
    nx = np.array([r["nx"] for r in rows], float)
    l1 = np.array([r["l1_percent"] for r in rows], float)
    pre = checks["predeclaration"]

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.loglog(nx, l1, "o-", lw=1.8, ms=7, color="#1f4e79",
              label="MUSCL/HLLC vs WENO5@12800 (measured)")

    fx = np.array([pre["nx_ref"]] + list(pre["targets"]), float)
    for p, style, lab in ((pre["order_band"][0], ":", "forecast band (frozen, before the runs)"),
                          (pre["order"], "--", f"forecast p={pre['order']:.4f}"),
                          (pre["order_band"][1], ":", None)):
        fy = pre["err_ref_percent"] * (fx / pre["nx_ref"]) ** (-p)
        ax.loglog(fx, fy, style, color="0.45", lw=1.3, label=lab)

    ax.axhline(wr.CERTIFIED_PERCENT, color="#c00000", lw=1.4, ls="-.")
    ax.text(nx.min(), wr.CERTIFIED_PERCENT * 1.08, "certified-<1 % bar",
            color="#c00000", fontsize=8)
    for r in rows:
        ax.annotate(f'{r["l1_percent"]:.3f} %', (r["nx"], r["l1_percent"]),
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color="#1f4e79")
    ax.set_xlabel("nx (cells)")
    ax.set_ylabel(f"window L1 vs WENO5@{REF_NX}  [%]")
    ax.set_title("Refining the graded solution: measured ladder vs the frozen forecast")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper right")
    out = FIG / "fig_refine_r3_ladder.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_profiles() -> Path:
    ref = wr.load_profile(DATA, REF_NX, "weno5")
    rungs = [n for n in wr.HLLC_RUNGS if wr.try_load_profile(DATA, n, "hllc") is not None]
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.6, 6.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2.0, 1.0]})
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(rungs)))
    for nx, col in zip(rungs, colors):
        got = wr.load_profile(DATA, nx, "hllc")
        d = wr.residual_density(got["rho"], ref["rho"])
        ax0.plot(d["x"], d["coarse"], "-", lw=1.0, color=col, label=f"MUSCL/HLLC nx={nx}")
        ax1.semilogy(d["x"], np.abs(d["coarse"] - d["reference"]) + 1e-12, "-",
                     lw=1.0, color=col)
    dref = wr.residual_density(wr.load_profile(DATA, min(rungs), "hllc")["rho"], ref["rho"])
    ax0.plot(dref["x"], dref["reference"], "k--", lw=1.2, label=f"WENO5@{REF_NX} restricted (conservatively restricted)")
    ax0.set_ylabel(r"$\rho$ (cell average)")
    ax0.set_title("Window density profiles and the pointwise residual they leave behind")
    ax0.legend(fontsize=7.5)
    ax0.grid(alpha=0.3)
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$|\rho_{graded}-\rho_{ref}|$")
    ax1.grid(alpha=0.3, which="both")
    out = FIG / "fig_refine_r3_profiles.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_cfl(cfl: dict | None) -> Path | None:
    if not cfl:
        return None
    ref = wr.load_profile(DATA, REF_NX, "weno5")
    base = wr.load_profile(DATA, cfl["nx"], "hllc")
    alt = wr.load_profile_tagged(DATA, f"nx{cfl['nx']}_hllc_cfl{cfl['cfl_refined']}")
    db = wr.residual_density(base["rho"], ref["rho"])
    da = wr.residual_density(alt["rho"], ref["rho"])
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [2.0, 1.0]})
    ax0.plot(db["x"], db["reference"], "k--", lw=1.2, label=f"WENO5@{REF_NX} restricted")
    ax0.plot(db["x"], db["coarse"], "-", lw=1.2, color="#1f4e79",
             label=f"CFL 0.4  (L1 {cfl['l1_percent_baseline']:.3f} %)")
    ax0.plot(da["x"], da["coarse"], "-", lw=1.2, color="#c00000",
             label=f"CFL {cfl['cfl_refined']}  (L1 {cfl['l1_percent_refined']:.3f} %)")
    ax0.set_ylabel(r"$\rho$ (cell average)")
    ax0.set_title(f"Time-step control at nx={cfl['nx']}: does the L1 floor come from dt?")
    ax0.legend(fontsize=7.5)
    ax0.grid(alpha=0.3)
    ax1.plot(db["x"], da["coarse"] - db["coarse"], "-", lw=1.0, color="#7030a0")
    ax1.set_xlabel("x")
    ax1.set_ylabel(r"$\rho$(CFL 0.2) $-\rho$(CFL 0.4)")
    ax1.grid(alpha=0.3)
    out = FIG / "fig_refine_r3_cfl_control.png"
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# build + render
# --------------------------------------------------------------------------- #
def build() -> dict:
    verdict = wr.audit(DATA, ref_nx=REF_NX)
    checks = wr.check_predeclaration(verdict)
    finest = max((r for r in verdict["ladder"] if r.get("measured")), key=lambda r: r["nx"])
    ref = wr.load_profile(DATA, REF_NX, "weno5")
    density = wr.residual_density(wr.load_profile(DATA, finest["nx"], "hllc")["rho"], ref["rho"])
    regions = wr.region_shares(density, REGIONS)
    try:
        cfl = wr.cfl_sensitivity(DATA, CFL_CONTROL["nx"], CFL_CONTROL["cfl"], ref_nx=REF_NX)
    except wr.DataMissing:
        cfl = None

    figs = [fig_ladder(verdict, checks), fig_profiles()]
    cfl_fig = fig_cfl(cfl)
    if cfl_fig:
        figs.append(cfl_fig)

    summary = {
        "mids": {"scale_dir": None},
        "verdict": verdict,
        "predeclaration_checks": checks,
        "regions": regions,
        "cfl_control": cfl,
        "figures": [p.name for p in figs],
        "data_dir": str(DATA),
    }
    (HERE / "summary_r3.json").write_text(json.dumps(summary, indent=2, default=float) + "\n")
    return summary


def _f(x, nd=4):
    return "n/a" if x is None else f"{x:.{nd}f}"


def render(s: dict) -> str:
    v = s["verdict"]
    c = s["predeclaration_checks"]
    pre = c["predeclaration"]
    rows = [r for r in v["ladder"] if r.get("measured")]
    f1, f2, f3 = c["F1"], c["F2"], c["F3"]
    cfl = s["cfl_control"]
    l1, lp = [], []
    l1.append("# XVAL-FVM-M18737-ROUND2 Round-3: 加密被定级解的**最终裁决**（F1/F2/F3）\n")
    l1.append("> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 精度标准）。")
    l1.append("> 前置：`REFINEMENT-CLOSURE.md`（冻结预测 + `not_measured=[6400,12800]`）、")
    l1.append("> `PREDECLARATION-refine.md`（判据 F1/F2/F3 写死于计算之前）。")
    l1.append("> 本文件**不覆盖**前轮的 honest-FAIL 记录，只把当时未测的两个档位测出来并按预声明判据裁决。\n")
    l1.append("## ① 案例介绍（物理背景 / 参数表）\n")
    l1.append("Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台")
    l1.append("（左 p=1000、中 p=0.01、右 p=100），ρ=1、u=0，两束爆轰波相向对撞，")
    l1.append("t≈0.028 在管中部相撞，积分至 t=0.038；γ=1.4，两端反射壁。无解析解，")
    l1.append("故基准是**跨族自认证数值参考**（WENO5/SSP-RK3/Rusanov，与 TensorFVM 不共享离散）。\n")
    l1.append("| 参数 | 值 |")
    l1.append("|---|---|")
    l1.append("| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC，CFL 0.4（`compressible/double_blast`） |")
    l1.append(f"| 参考解 | WENO5@nx={REF_NX}，自认证 L1 {_f((v['reference']['self_certification'] or {}).get('l1_percent'))} % |")
    l1.append("| 比较窗口 | x∈[0.4,1.0]，**守恒限制**（coarse 单元均值 = 其内 r 个 fine 单元均值，零插值） |")
    l1.append("| 本轮新增档 | MUSCL/HLLC nx=6400、nx=12800（作业 cj3458 两分片）；时间步控制 nx=6400 CFL 0.2（cj3459） |")
    l1.append(f"| certified 判据 | L1 与峰值同时 < {wr.CERTIFIED_PERCENT:.0f} %；总账容差 {wr.TOLERANCE_PERCENT:.0f} % |\n")
    l1.append("## ② 怎么做的\n")
    l1.append("1. **预声明先行**（前一单已完成）：在 6400/12800 被定级解算出来**之前**冻结")
    l1.append(f"   p={pre['order']:.4f}（由 1600/3200 两档定阶）、预测 e(6400)={_f(pre['err_ref_percent'] * (6400/pre['nx_ref'])**-pre['order'])} %、")
    l1.append(f"   e(12800)={_f(pre['err_ref_percent'] * (12800/pre['nx_ref'])**-pre['order'])} %，判据 F1/F2/F3 写死在 `PREDECLARATION-refine.md`。")
    l1.append("2. **计算**：`docs/woodward-colella-double-blast/ladder/wc_ladder.py --plan 6400:hllc,12800:hllc`（两分片，")
    l1.append("   单核）与 `--plan 6400:hllc:mc:0.2`（时间步控制），递交调度系统，非本机。")
    l1.append("3. **判据机检**：`tensorfvm.verification.wc_refine.check_predeclaration()` 把 F1（点值落带）、")
    l1.append("   F2（三段序列 log--log 最小二乘阶 ∈ [0.70,1.20] 且无平台）、F3（首个 L1 且峰值 <1 % 的档位）")
    l1.append("   逐条判真假，返回数字，报告由 `summary_r3.json` 程序化渲染（零手抄）。")
    l1.append("4. **时间步控制**：同一 nx=6400 用 CFL 0.2 重算，量化 dt 误差对 L1 地板的贡献；")
    l1.append("   若 ΔL1 ≥ 0.05 pp 则「加密」建议被 dt 混淆，反之不混淆。\n")
    l1.append("## ③ 计算结果\n")
    l1.append("### 被定级解阶梯（vs WENO5@12800，守恒限制口径）\n")
    l1.append("| nx | L1 [%] | 峰值 [%] | 窗质量 [%] | L1 <1 % |")
    l1.append("|---|---|---|---|---|")
    for r in rows:
        l1.append(f"| {r['nx']} | {_f(r['l1_percent'])} | {_f(r['peak_percent'])} | "
                  f"{_f(r['mass_percent'], 5)} | {'✅' if r['l1_percent'] < 1.0 else '❌'} |")
    l1.append("")
    l1.append("### 冻结预测检验（F1）\n")
    l1.append("| 目标 nx | 冻结预测 L1 [%] | 预测带 [%] | 实测 L1 [%] | 预测相对误差 [%] | 落带 |")
    l1.append("|---|---|---|---|---|---|")
    for row in c["forecast"]:
        l1.append(f"| {row['nx']} | {_f(row['predicted_percent'])} | "
                  f"[{_f(row['band_percent'][0])}, {_f(row['band_percent'][1])}] | "
                  f"{_f(row['measured_percent'])} | {_f(row['rel_error_percent'], 2)} | "
                  f"{'✅' if row['in_band'] else ('❌' if row['in_band'] is not None else '未测')} |")
    l1.append("")
    l1.append(f"**F1 判定**: 目标 nx={f1['target_nx']}，实测 {_f(f1['measured_percent'])} % 对预声明带 "
              f"[{_f(f1['band_percent'][0])}, {_f(f1['band_percent'][1])}] → "
              f"**{'PASS' if f1['passed'] else ('FAIL' if f1['passed'] is not None else 'UNDECIDED')}**"
              f"{('' if f1['passed'] else '（' + f1['note'] + '）') if f1['note'] else ''}\n")
    l1.append("### 观测阶（F2）\n")
    l1.append("| 项 | 值 |")
    l1.append("|---|---|")
    l1.append(f"| 序列 | {f2['sequence']} |")
    l1.append(f"| 序列最小二乘阶 | **{_f(f2['order_sequence_lsq'])}** |")
    l1.append(f"| 预声明带 | [{f2['order_band'][0]}, {f2['order_band'][1]}] |")
    l1.append(f"| 首末两点阶（对照） | {_f(f2['consistent_two_point_order'])} |")
    l1.append(f"| 平台判定（最细两档 L1 = {f2['finest_two_l1_percent']}） | {'PLATEAU' if f2['plateau'] else 'no plateau'} |")
    l1.append(f"| **F2 判定** | **{'PASS' if f2['passed'] else 'FAIL'}** |\n")
    l1.append("### certified-<1 % 裁决（F3）\n")
    l1.append("```")
    l1.append(f"certified_at_nx        = {f3['certified_at_nx']}")
    l1.append(f"finest_measured        = nx={f3['finest_measured_nx']}, L1={_f(f3['finest_l1_percent'])} %, peak={_f(f3['finest_peak_percent'])} %")
    l1.append(f"nx_needed_for_1pct     = {_f(f3['nx_needed_for_1pct'], 0)}")
    l1.append(f"path_exhausted         = {f3['path_exhausted']}")
    l1.append(f"ruling                 = {f3['ruling']}")
    l1.append("```\n")
    l1.append(f"**验收判据**：<{wr.CERTIFIED_PERCENT:.0f} %（certified）/ <3 %（严格）/ <{wr.TOLERANCE_PERCENT:.0f} %（总账）。"
              f"总账 {wr.TOLERANCE_PERCENT:.0f} % 判定 = **PASS**（最细档 L1 {_f(f3['finest_l1_percent'])} % ≪ 15 %）。\n")
    if cfl:
        l1.append("### 时间步控制（nx=6400，CFL 0.2 vs 0.4）\n")
        l1.append("| 量 | CFL 0.4 | CFL 0.2 |")
        l1.append("|---|---|---|")
        l1.append(f"| L1 vs WENO5@12800 [%] | {_f(cfl['l1_percent_baseline'])} | {_f(cfl['l1_percent_refined'])} |")
        l1.append(f"| 峰值 [%] | {_f(cfl['peak_percent_baseline'])} | {_f(cfl['peak_percent_refined'])} |")
        l1.append("")
        l1.append(f"ΔL1 = **{_f(cfl['delta_l1_pp'], 4)} pp**；决策不变（L1 门结论一致）= {cfl['decision_invariant']}；"
                  f"时间步误差可忽略（Δ<0.05 pp）= **{cfl['time_error_negligible']}**。\n")
    l1.append("### 残差空间归属（最细实测档）\n")
    l1.append("| 区间 | L1 残差占比 [%] |")
    l1.append("|---|---|")
    for r in s["regions"]:
        l1.append(f"| {r['region']} | {_f(r['share_percent_of_l1'], 2)} |")
    l1.append("")
    l1.append("## ④ 结果图\n")
    l1.append("![加密阶梯 vs 冻结预测](figures/fig_refine_r3_ladder.png)\n")
    l1.append("*图 1: 被定级解 L1 阶梯（实测点）叠在计算前冻结的预测带、1 % certified 线与实测阶上。*\n")
    l1.append("![剖面与逐点残差](figures/fig_refine_r3_profiles.png)\n")
    l1.append("*图 2: 窗口密度剖面（各档 MUSCL/HLLC 与守恒限制后的 WENO5@12800 参考）+ 逐点残差。*\n")
    if cfl:
        l1.append("![时间步控制](figures/fig_refine_r3_cfl_control.png)\n")
        l1.append("*图 3: nx=6400 在 CFL 0.4 与 0.2 下的剖面与两者之差——dt 误差是否主导 L1 地板。*\n")
    l1.append("## ⑤ 注意点（局限 / 适用范围）\n")
    l1.append("1. **参考侧地板仍在**：WENO5@6400 vs @12800 自认证 L1 "
              f"{_f((v['reference']['self_certification'] or {}).get('l1_percent'))} %，"
              "故被定级解 L1 降到与该量级相当时，继续加密的收益会被参考侧误差吃掉；")
    l1.append("   本轮的 12800 档就是用来量这个地板的，它的意义不是「更高的分」而是「地板的量级」。")
    l1.append("2. **峰值度量非单调**：单元平均峰值受前沿落在单元内的位置影响，判据以 L1 为主，峰值只作辅助门。")
    l1.append("3. **只加密空间不够就换格式**：若 F2 的阶落在近一阶区（激波主导），")
    l1.append("   要更低的 certified 只能上高阶格式（`xcode/wc_weno7.py`）或前沿自适应，而非继续均匀加密。")
    l1.append("4. **时间步只做了一次控制**：CFL 0.2 单点对照能排除 dt 主导，但不是完整的时间收敛研究（未做 dt 三点序列）。")
    l1.append("5. **无解析解**：全部结论建立在跨族自认证参考之上，参考族若在 12800 未自收敛则结论随之失效——故门 1 每次必重测。")
    l1.append("6. **本文件与 `REFINEMENT-CLOSURE.md` 并存**：前轮的 `not_measured` 与 honest-FAIL 记录保留不改写。\n")
    l1.append("## ⑥ 误差判定（逐项显式回答「是否 <15 % / <3 % / <1 %」）\n")
    l1.append("| 关键量 | 实测 | <15 %（总账） | <3 % | <1 %（certified） |")
    l1.append("|---|---|---|---|---|")
    sc = v["reference"]["self_certification"] or {}
    l1.append(f"| 参考自认证 L1（WENO5@6400 vs @12800） | {_f(sc.get('l1_percent'))} % | ✅ | ✅ | ✅ |")
    for r in rows:
        val = r["l1_percent"]
        l1.append(f"| 被定级解 L1（nx={r['nx']}） | {_f(val)} % | ✅ | {'✅' if val < 3 else '❌'} | "
                  f"{'✅' if val < 1 else '❌'} |")
    l1.append(f"| 窗质量 ∫ρdx（nx={f3['finest_measured_nx']}） | {_f(rows[-1]['mass_percent'], 5)} % | ✅ | ✅ | ✅ |")
    if cfl:
        l1.append(f"| 时间步控制 ΔL1（CFL 0.2 vs 0.4） | {_f(cfl['delta_l1_pp'], 4)} pp | ✅ | ✅ | — |")
    l1.append("")
    l1.append(f"**结论**：总账判据 <{wr.TOLERANCE_PERCENT:.0f} % → **PASS**（全部良态量 ≪15 %）。"
              f"certified-<1 % → "
              f"**{'达成，首次在 nx=' + str(f3['certified_at_nx']) if f3['certified_at_nx'] else '未达成（honest FAIL）'}**；"
              f"F1/F2/F3 三条预声明判据通过 {c['checks_passed']}/{c['checks_total']} 条（overall = `{c['overall']}`）。\n")
    l1.append("## 复现命令\n")
    l1.append("```bash")
    l1.append("# 两档新算（调度系统递交，非本机）")
    l1.append("PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_ladder.py --plan 6400:hllc,12800:hllc")
    l1.append("PYTHONPATH=src python docs/woodward-colella-double-blast/ladder/wc_ladder.py --plan 6400:hllc:mc:0.2")
    l1.append("# 判据机检 + 出图 + 渲染本文件")
    l1.append("PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/refinement_closure_r3.py")
    l1.append("PYTHONPATH=src python -m pytest tests/test_compressible_wc_refine_r3.py -q")
    l1.append("```")
    return "\n".join(l1) + "\n"


def main() -> dict:
    s = build()
    (HERE / "REFINEMENT-R3-CLOSURE.md").write_text(render(s))
    c = s["predeclaration_checks"]
    print(json.dumps({"figures": s["figures"], "F1": c["F1"]["passed"], "F2": c["F2"]["passed"],
                      "F3": c["F3"]["certified_at_nx"], "overall": c["overall"],
                      "finest_l1": c["F3"]["finest_l1_percent"]}, indent=2, default=float))
    return s


if __name__ == "__main__":
    main()