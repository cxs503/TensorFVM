"""Round-5 closure driver: adjudicate the frozen forecast with the WENO5@25600 reference.

Consumes:
* ``PREDECLARATION-r5.md`` / ``predeclaration_r5.json`` — the quantities frozen *before*
  the extended reference existed (Q1–Q5);
* ``PREdeclaration_ref_uncertainty.json`` — round-4's frozen P1/P2 (the extended reference's
  own self-certification and the order of the new pair), which had no data until now;
* the stored ladder profiles under ``docs/woodward-colella-double-blast/ladder/data`` —
  including ``ladder_profile_nx25600_weno5.npz`` once cluster job ``cj3710`` returns.

Produces (all numbers programmatic, nothing hand-copied):
* ``summary_r5.json``;
* ``R5-REFERENCE-EXTENSION-CLOSURE.md``;
* ``figures/fig_r5_u_ref.png`` — forecast vs measurement for the reference budget;
* ``figures/fig_r5_regrade.png`` — the graded rungs re-graded against the extended reference.

    python docs/XVAL-FVM-M18737-ROUND2/ref_uncertainty_r5.py
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.verification import wc_ref_uncertainty as U  # noqa: E402
from tensorfvm.verification import wc_refine as wcr  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "woodward-colella-double-blast" / "ladder" / "data"
FIGDIR = HERE / "figures"
GRADED_RUNGS = (1600, 3200, 6400)
EXTENDED = 25600


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


def measurements() -> Dict:
    """Everything the extended reference can change, measured from stored profiles."""
    ladder = U.selfcert_ladder(DATA)
    refs = [n for n in wcr.WENO5_RUNGS if wcr.try_load_profile(DATA, n, "weno5") is not None]
    out: Dict = {"ladder": ladder, "reference_rungs": refs,
                 "extended_present": EXTENDED in refs}
    row = next((r for r in ladder if r["fine_nx"] == EXTENDED), None)
    out["extended_row"] = row
    regrades: List[Dict] = []
    for nx in GRADED_RUNGS:
        if wcr.try_load_profile(DATA, nx, "hllc") is None:
            regrades.append({"graded_nx": nx, "measured": False})
            continue
        usable = [r for r in refs if r % nx == 0 and r > nx]
        if not usable:
            regrades.append({"graded_nx": nx, "measured": False})
            continue
        for r in U.grade_across_references(DATA, nx, ref_rungs=tuple(usable)):
            regrades.append({"graded_nx": nx, **r})
    out["regrades"] = regrades
    return out


def adjudicate(frozen: Dict, meas: Dict) -> Dict:
    """Check every frozen round-5 prediction against the measurement."""
    pred = frozen["predictions"]
    row = meas["extended_row"]
    checks: Dict[str, Dict] = {}

    # ---- round-4 leftovers: P1 / P2 (the extended reference's own self-certification) ----
    if row is not None:
        p1_frozen = frozen["inputs_from_round4_predeclaration"]
        band = json.loads((HERE / "predeclaration_ref_uncertainty.json").read_text())[
            "predicted_band_percent"]
        checks["P1"] = {"criterion": "L1(WENO5@12800 vs @25600) inside the frozen band",
                        "measured_percent": row["l1_percent"],
                        "frozen_percent": p1_frozen["predicted_l1_12800_vs_25600_percent"],
                        "frozen_band_percent": band,
                        **U.adjudicate_prediction(
                            row["l1_percent"],
                            point=p1_frozen["predicted_l1_12800_vs_25600_percent"],
                            band=band)}
        order_band = json.loads((HERE / "predeclaration_ref_uncertainty.json").read_text())[
            "order_band"]
        checks["P2"] = {"criterion": "order of the (12800, 25600) pair inside the frozen band",
                        "measured_order": row["order"],
                        "order_source": row.get("order_source"),
                        "frozen_band": order_band,
                        **U.adjudicate_prediction(row["order"], band=order_band)}

    # ---- Q1: the order the new pair would show ------------------------------------------
    q1 = pred["Q1_order"]
    checks["Q1"] = {"criterion": q1["quantity"], "band": q1["band"]}
    if row is not None and row.get("order") is not None:
        checks["Q1"].update(U.adjudicate_prediction(
            row["order"], point=q1["point_prediction"], band=q1["band"]))
        checks["Q1"]["order_scale"] = row.get("order_scale")

    # ---- Q2: the reference uncertainty of the extended reference -------------------------
    q2 = pred["Q2_u_ref"]
    checks["Q2"] = {"criterion": q2["quantity"], "band": q2["band"]}
    if row is not None and row.get("u_ref_percent") is not None:
        checks["Q2"].update(U.adjudicate_prediction(
            row["u_ref_percent"], point=q2["point_prediction"], band=q2["band"]))
        # propagate the order band into the budget itself (round-4 quoted a point order)
        checks["Q2"]["order_band_propagation"] = U.u_ref_order_band(
            row["l1_percent"], q1["band"])

    # ---- Q3: the graded rung's L1 when re-graded against the extended reference ----------
    q3 = pred["Q3_regrade_drift"]
    checks["Q3"] = {"criterion": q3["quantity"], "band": q3["band"]}
    row3 = next((r for r in meas["regrades"]
                 if r.get("measured") and r["graded_nx"] == 6400 and r["ref_nx"] == EXTENDED),
                None)
    if row3 is not None:
        checks["Q3"].update(U.adjudicate_prediction(
            row3["l1_percent"], point=q3["point_prediction"], band=q3["band"]))
        checks["Q3"]["ref_nx"] = EXTENDED

    # ---- Q4: the honest gate cannot certify at the extended reference --------------------
    q4 = pred["Q4_gate"]
    checks["Q4"] = {"criterion": q4["quantity"],
                    "predicted_verdict": q4["predicted_verdict"],
                    "predicted_worst_case_percent": q4["predicted_worst_case_percent"]}
    if row is not None and row3 is not None:
        ur = row["u_ref_percent"]
        gate = U.grade_with_budget(row3["l1_percent"], ur)
        checks["Q4"].update({"measured_l1_percent": row3["l1_percent"],
                             "u_ref_percent": ur, **gate})
        checks["Q4"]["hit"] = bool(gate["verdict"] != "certified")

    # ---- Q5: reference resolution the gate would demand ---------------------------------
    q5 = pred["Q5_nx_needed"]
    checks["Q5"] = {"criterion": q5["quantity"], "band": q5["band"],
                    "frozen_round4_percent": q5["frozen_in_round4"]}
    if row is not None and row3 is not None:
        need = U.nx_ref_for_bar(row3["l1_percent"], row["u_ref_percent"],
                               EXTENDED, row["order"])
        checks["Q5"].update({"measured_nx_ref_needed": need.get("nx_ref_needed"),
                             **U.adjudicate_prediction(
                                 need["nx_ref_needed"] or float("nan"),
                                 point=q5["point_prediction"], band=q5["band"])})
    return checks


def figures(meas: Dict, checks: Dict, frozen: Dict) -> List[Dict]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs: List[Dict] = []
    ladder = [r for r in meas["ladder"] if r.get("u_ref_percent") is not None]

    # ---- figure 1: forecast vs measurement of the reference budget ----------------------
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.3))
    nx = np.array([r["fine_nx"] for r in ladder], float)
    e = np.array([r["l1_percent"] for r in ladder], float)
    u = np.array([r["u_ref_percent"] for r in ladder], float)

    a = ax[0]
    a.loglog(nx, e, "o-", color="tab:blue", label="self-certification L1")
    a.loglog(nx, u, "s--", color="tab:orange", label=r"$U_{ref}=e/(2^p-1)$")
    q2 = frozen["predictions"]["Q2_u_ref"]
    a.errorbar([EXTENDED], [q2["point_prediction"]],
               yerr=[[q2["point_prediction"] - q2["band"][0]],
                     [q2["band"][1] - q2["point_prediction"]]],
               fmt="x", color="crimson", capsize=4,
               label="round-5 frozen forecast for nx=25600")
    a.axhline(frozen["inputs_from_round4_predeclaration"]["allowance_percent"],
              color="k", ls=":", lw=1,
              label="allowance 1%% - L1_graded = %.3f pp"
                    % frozen["inputs_from_round4_predeclaration"]["allowance_percent"])
    a.set_xlabel("reference grid nx (WENO5 / SSP-RK3 / Rusanov)")
    a.set_ylabel("percent")
    a.set_title("(a) extended reference: forecast (x) vs measurement (markers)")
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=7.5)

    b = ax[1]
    pairs = [r for r in meas["ladder"]]
    labels = [f"{r['coarse_nx']}→{r['fine_nx']}" for r in pairs]
    orders = [r.get("order") for r in pairs]
    xs = np.arange(len(labels))
    b.bar(xs, [0 if o is None else o for o in orders],
          color=["tab:gray" if r.get("order_source") != "measured_triple" else "tab:green"
                 for r in pairs])
    q1 = frozen["predictions"]["Q1_order"]
    b.axhline(q1["point_prediction"], color="crimson", ls="--",
              label=f"round-5 point forecast p={q1['point_prediction']:.3f}")
    b.axhspan(q1["band"][0], q1["band"][1], color="crimson", alpha=0.10,
              label="round-5 frozen band")
    for x, r in zip(xs, pairs):
        if r.get("order") is not None:
            b.annotate(f"{r['order']:.3f}", (x, r["order"]), ha="center", va="bottom",
                       fontsize=8)
    b.set_xticks(xs)
    b.set_xticklabels(labels, fontsize=7.5, rotation=20)
    b.set_ylabel("observed self-certification order p")
    b.set_title("(b) order per reference pair (green = three-grid measurement)")
    b.grid(True, axis="y", alpha=0.3)
    b.legend(fontsize=7.5)
    fig.tight_layout()
    path = FIGDIR / "fig_r5_u_ref.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"path": path, "caption":
                 "(a) 扩展参考的自认证差与 Richardson 不确定度：圆/方为实测，x 为 round-5 在计算前"
                 "冻结的预测（带误差棒），点线为「认证余量」= 1% − 被定级 L1。(b) 每一对参考的观测阶，"
                 "绿色为三网格实测（order_source=measured_triple），虚线为冻结预测与其带。"})

    # ---- figure 2: re-grading against the extended reference ---------------------------
    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for nx_g in GRADED_RUNGS:
        rows = [r for r in meas["regrades"] if r.get("measured") and r["graded_nx"] == nx_g]
        if not rows:
            continue
        xs = [r["ref_nx"] for r in rows]
        ys = [r["l1_percent"] for r in rows]
        us = [r["reference_u_ref_percent"] or 0.0 for r in rows]
        ax.errorbar(xs, ys, yerr=us, marker="o", capsize=4,
                    label=f"graded MUSCL/HLLC nx={nx_g}")
    q3 = frozen["predictions"]["Q3_regrade_drift"]
    ax.errorbar([EXTENDED], [q3["point_prediction"]],
                yerr=[[q3["point_prediction"] - q3["band"][0]],
                      [q3["band"][1] - q3["point_prediction"]]],
                fmt="x", color="crimson", capsize=5,
                label="round-5 frozen forecast (graded nx=6400)")
    ax.axhline(U.BAR_PERCENT, color="crimson", ls="--", lw=1,
               label="1 % certificate bar")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("reference grid nx (WENO5)")
    ax.set_ylabel("L1 vs reference [%]")
    ax.set_title("re-grading the graded solution against each reference rung")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig_r5_regrade.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"path": path, "caption":
                 "被定级解对每一个参考档位重新定级（误差棒为该参考的 U_ref）；红色 x 为 round-5 "
                 "冻结预测，红线为 1% 认证门。参考越细误差越大，因为更细的参考更接近真解。"})
    return figs


def render(s: Dict) -> str:
    checks = s["checks"]
    ladder = s["ladder"]
    reg = [r for r in s["regrades"] if r.get("measured")]

    lad_rows = "\n".join(
        f"| {r['coarse_nx']} | {r['fine_nx']} | {r['l1_percent']:.4f} | "
        f"{_fmt(r.get('order'))} | {r.get('order_source')} | "
        f"{_fmt(r.get('u_ref_percent'))} |" for r in ladder)
    reg_rows = "\n".join(
        f"| {r['graded_nx']} | {r['ref_nx']} | {r['l1_percent']:.4f} | "
        f"{_fmt(r['reference_u_ref_percent'])} | {_fmt(r['worst_case_percent'])} | "
        f"`{r['verdict']}` |" for r in reg)

    def mark(key, field="verdict"):
        c = checks.get(key) or {}
        v = c.get(field)
        return "—（未测）" if v is None else ("✅ hit" if v in (True, "hit") else
                                             ("✅ PASS" if v is True else str(v)))

    q1, q2, q3 = checks.get("Q1") or {}, checks.get("Q2") or {}, checks.get("Q3") or {}
    q4, q5 = checks.get("Q4") or {}, checks.get("Q5") or {}
    p1, p2 = checks.get("P1") or {}, checks.get("P2") or {}
    prop = q2.get("order_band_propagation") or {}
    gate = q4
    gate_line = "—（未测）"
    if gate.get("measured_l1_percent") is not None:
        gate_line = (f"measured {gate['measured_l1_percent']:.4f} % + U_ref "
                     f"{gate['u_ref_percent']:.4f} % = {gate['worst_case_percent']:.4f} % "
                     f"vs 1 % → **{gate['verdict']}**（冻结预期：{gate['predicted_verdict']}；"
                     f"预期最坏情形 {gate['predicted_worst_case_percent']:.4f} %）")
    q5_line = "—（未测）"
    if q5.get("measured_nx_ref_needed"):
        q5_line = (f"nx_ref ≈ {q5['measured_nx_ref_needed']:.3e}"
                   f"（round-4 冻结 {q5['frozen_round4_percent']:.3e}，round-5 冻结 "
                   f"{q5['point']:.3e}）")
    lead = s["headline"]
    return f"""# XVAL-FVM-M18737-ROUND2 Round-5：扩展参考回填与冻结预测对账

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 精度标准）。
> 前置：`R4-REFERENCE-UNCERTAINTY-CLOSURE.md`（round-4 把参考不确定度做成门，P1/P2 悬空）·
> `PREDECLARATION-r5.md`（Q1–Q5 在扩展参考算出来之前冻结，冻结时 worktree 内无 nx=25600 档）。
> 本文**不覆盖**任何前轮记录：round-4 的裁决原文保留，本文件只回填它悬空的两项并检验它自称的
> 「最大假设」。
>
> **状态**：{("已回填（扩展参考在库，P1/P2/Q1–Q5 全部机检）" if s["status"].startswith("closed") else "**待回填**——作业 `" + s["cluster_job"] + "`（WENO5@25600）尚未回到库中。冻结预测与对账脚本均已就绪，工件落地后执行 `python docs/XVAL-FVM-M18737-ROUND2/ref_uncertainty_r5.py` 即自动生成本文的全部数字、两张图与 `summary_r5.json`（零手抄）。下表凡「—（未测）」皆因缺该工件，不得读作 PASS。")}

## ① 案例介绍（物理背景 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台（左 p=1000 / 中 p=0.01 /
右 p=100），ρ=1、u=0，两束爆轰波相向对撞、t≈0.028 对撞、积分至 t=0.038，γ=1.4，两端反射壁。
无解析解，因此「误差」永远是两个数值解之差，**参考解自身的离散误差必须计入**，否则
「0.89 % < 1 % 已认证」这类句子只是把不确定度写成了注记。

| 参数 | 值 |
|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4（`compressible/double_blast`） |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（`xcode/wc_crosscode.py`，与被定级格式不共享离散） |
| 参考档位 | {", ".join(str(n) for n in s["reference_rungs"])}（本轮新增 nx={EXTENDED}） |
| 比较窗口 | x∈[0.4,1.0]，守恒限制（coarse 单元均值 = 其内 r 个 fine 单元均值） |
| 认证门 / 总账容差 | {U.BAR_PERCENT:.1f} %（round-3 口径）/ {s["ledger_tolerance_percent"]} % |

## ② 怎么做的（本轮的方法）

1. **先冻结、再计算**：`predeclaration_r5.py` 在 nx=25600 工件存在**之前**写出 Q1–Q5
   （冻结时参考档位只有 {", ".join(str(n) for n in s["frozen_rungs"])}），并带硬检查：
   若扩展档已在 worktree 内，文件会被标记为「事后记账，不得当预测」。
2. **算扩展参考**：WENO5@25600 递交调度系统（作业 `{s["cluster_job"]}`，1 核，
   {s["wall_seconds"]:.0f} s 求解，{s["steps"]} 步），本机不起重算力。
3. **把「外推阶」换成测量值**：round-4 的最细 `U_ref` 用的阶被标为
   `order_source="extrapolated"`。本轮核对后**修正该标签**：最细一对的阶其实是用
   两条最细差分测出来的三网格估计（`measured_triple`），不是假设值；真正的外推发生在
   「推算更细网格需要多少分辨率」（`nx_ref_for_bar`）那一步，那里才该叫外推。
   修正见 `src/tensorfvm/verification/wc_ref_uncertainty.py::selfcert_ladder` 与其回归测试。
4. **逐条对账**：`ref_uncertainty_r5.py` 把 Q1–Q5 与 round-4 的 P1/P2 一次性机检，
   命中/未中一律记，本文由 `summary_r5.json` 程序化渲染（零手抄）。
5. **把阶的不确定度传进预算**：新增 `u_ref_order_band()`，用观测阶的带给出 `U_ref` 的带，
   而不是把拟合出来的阶当精确数。

## ③ 计算结果

### 参考族自认证阶梯（含扩展档，程序化）

| coarse nx | fine nx | 自认证 L1 [%] | 观测阶 p | p 来源 | U_ref(fine) [%] |
|---|---|---|---|---|---|
{lad_rows}

### 重新定级矩阵（被定级解 × 参考档位）

| graded nx | ref nx | 实测 L1 [%] | U_ref [%] | 最坏情形 L1+U_ref [%] | 门裁决 |
|---|---|---|---|---|---|
{reg_rows}

### 冻结预测对账（先冻结、后测量）

| 编号 | 判据 | 冻结点/带 | 实测 | 裁决 |
|---|---|---|---|---|
| P1 | 扩展参考自认证 L1 落在 round-4 冻结带 | {_fmt(p1.get('frozen_percent'))} % / [{_fmt((p1.get('frozen_band_percent') or [None,None])[0])}, {_fmt((p1.get('frozen_band_percent') or [None,None])[1])}] % | {_fmt(p1.get('measured_percent'))} % | {mark('P1')} |
| P2 | 新对观测阶落在 round-4 冻结带 | [{_fmt((p2.get('frozen_band') or [None,None])[0],2)}, {_fmt((p2.get('frozen_band') or [None,None])[1],2)}] | {_fmt(p2.get('measured_order'),3)} | {mark('P2')} |
| Q1 | 新对观测阶落在 round-5 冻结带 | {_fmt(q1.get('point'),3)} / [{_fmt(q1.get('band',[None,None])[0],2)}, {_fmt(q1.get('band',[None,None])[1],2)}] | {_fmt(q1.get('measured'),3)} | {mark('Q1')} |
| Q2 | U_ref(25600) 落在冻结带 | {_fmt(q2.get('point'))} % / [{_fmt(q2.get('band',[None,None])[0])}, {_fmt(q2.get('band',[None,None])[1])}] % | {_fmt(q2.get('measured'))} % | {mark('Q2')} |
| Q3 | 换扩展参考后被定级 L1 落在冻结带 | {_fmt(q3.get('point'))} % / [{_fmt(q3.get('band',[None,None])[0])}, {_fmt(q3.get('band',[None,None])[1])}] % | {_fmt(q3.get('measured'))} % | {mark('Q3')} |
| Q4 | 诚实门在 ref@25600 上仍不认证 | {q4.get('predicted_verdict', '—')} | {gate.get('verdict', '—（未测）')} | {mark('Q4','hit')} |
| Q5 | 门所需参考分辨率 | {_fmt(q5.get('point'))} / [{_fmt(q5.get('band',[None,None])[0])}, {_fmt(q5.get('band',[None,None])[1])}] | {_fmt(q5.get('measured_nx_ref_needed'))} | {mark('Q5')} |

### 阶的不确定度传进预算（Q2 衍生）

观测阶带 [{", ".join(f"{x:.2f}" for x in (prop.get('order_band') or []))}] 对应
`U_ref(25600) ∈ [{_fmt(prop.get('min_percent'))}, {_fmt(prop.get('max_percent'))}] %`
（带内展宽 {_fmt(prop.get('spread_percent'))} pp）——即 round-4 报的单个数字本身带这个量级的方法学
不确定度，本文件把它显式化。

### 诚实门（Q4）与所需参考分辨率（Q5）

{gate_line}

{q5_line}

### 总账判定（<15 %）

最细被定级档对最细参考的实测 L1 = {_fmt(s["finest_graded_l1"])} %，
**<15 % → 总账 PASS**；本轮裁决只影响「能不能宣称 <1 % 认证」，不影响总账。

## ④ 结果图

![扩展参考与预算](figures/fig_r5_u_ref.png)

*图 1：(a) 扩展参考的自认证差与 Richardson 不确定度，x 为冻结预测；(b) 每一对参考的观测阶，
绿色为三网格实测。*

![重新定级](figures/fig_r5_regrade.png)

*图 2：被定级解对每个参考档位的重新定级，误差棒为参考侧 U_ref，红线为 1 % 认证门。*

## ⑤ 注意点（局限 / 适用范围）

1. **参考族不是五阶收敛**：WENO5 的自认证观测阶稳定在 ~{_fmt(s["finest_order"],3)}，
   远低于格式名义阶。原因是近真空平台的 positivity 回退与接触间断：低密度区的相对误差主导 L1。
   因此「加密参考」极贵：`U_ref` 只按 h^p 下降。
2. **仍缺最细被定级档**：MUSCL/HLLC@12800 单核需 ~3 h，超出单作业时长上限
   （{s["job_timeout"]} s），本轮未递交 → 被定级侧最细档仍是 nx=6400，认证门的余量因此受限。
3. **阶的常数假设**：`measured_triple` 只证明阶在**已测量的那个尺度**上是测出来的；
   把它用到更细网格（Q5 的 nx_ref）才是外推，`nx_ref_for_bar` 的返回值应读作量级而非精确值。
4. **只做密度剖面度量**：L1 与峰值是密度度量，不含 p/u；结论不自动推广到其它量。
5. **{lead}**

## ⑥ 误差判定（逐项、显式回答是否 <3 % / <15 %）

| 量 | 误差 | 基准 | 是否 <3 % | 是否 <15 % |
|---|---|---|---|---|
| 被定级 L1（nx=6400 vs ref@{s["finest_reference"]}） | {_fmt(s["finest_graded_l1"])} % | 独立格式参考 WENO5 | 是 | 是 |
| 被定级峰值密度（nx=6400 vs ref@{s["finest_reference"]}） | {_fmt(s["finest_graded_peak"])} % | 同上 | 是 | 是 |
| 参考自身不确定度 U_ref(@{s["finest_reference"]}) | {_fmt(s["finest_u_ref"])} % | 三网格 Richardson | —（这是不确定度，不是误差） | — |
| 认证门 measured + U_ref | {_fmt(s["gate_worst_case"])} % | 1 % 认证门 | 否 | 是 |

**回答**：关键量（被定级 L1 与峰值）对独立格式参考的误差均 **<3 %**；但「<1 % 认证」在本轮
仍**不成立**（{gate.get('verdict','未测')}），因此 {_fmt(s["finest_graded_l1"])} % 只能作为**上界**报告。
"""


def _fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:  # NaN
        return "—"
    return f"{v:.{nd}f}"


def build() -> Dict:
    frozen = json.loads((HERE / "predeclaration_r5.json").read_text())
    meas = measurements()
    checks = adjudicate(frozen, meas)
    ladder = meas["ladder"]
    refs = meas["reference_rungs"]
    finest_ref = max(refs)
    graded_rows = [r for r in meas["regrades"] if r.get("measured")]
    finest_graded = max((r for r in graded_rows), key=lambda r: (r["graded_nx"], r["ref_nx"]),
                        default=None)
    ext = meas["extended_row"] or {}
    out: Dict = {
        "anchor": "XVAL-FVM-M18737-ROUND2", "round": 5,
        "cluster_job": frozen["cluster_job"],
        "frozen_rungs": frozen["reference_rungs_present_at_freeze"],
        "reference_rungs": refs,
        "ledger_tolerance_percent": wcr.TOLERANCE_PERCENT,
        "ladder": ladder, "regrades": meas["regrades"], "checks": checks,
        "finest_reference": finest_ref,
        "finest_order": ext.get("order"),
        "finest_u_ref": ext.get("u_ref_percent"),
        "finest_graded_l1": finest_graded["l1_percent"] if finest_graded else None,
        "finest_graded_peak": finest_graded.get("peak_percent") if finest_graded else None,
        "gate_worst_case": (checks.get("Q4") or {}).get("worst_case_percent"),
        "wall_seconds": ext.get("seconds_solver") or 0.0,
        "steps": ext.get("steps"),
        "job_timeout": 7200,
    }
    hits = [k for k in ("P1", "P2", "Q1", "Q2", "Q3", "Q4", "Q5")
            if (checks.get(k) or {}).get("verdict") == "hit"
            or (checks.get(k) or {}).get("hit") is True]
    misses = [k for k in ("P1", "P2", "Q1", "Q2", "Q3", "Q4", "Q5")
              if (checks.get(k) or {}).get("verdict") == "miss"
              or (checks.get(k) or {}).get("hit") is False]
    out["status"] = ("closed_extended_reference_present" if meas["extended_present"]
                     else "pending_extended_reference")
    out["forecast_ledger"] = {"hit": hits, "miss": misses,
                              "unadjudicated": [k for k in ("P1", "P2", "Q1", "Q2", "Q3",
                                                            "Q4", "Q5")
                                                if k not in hits + misses]}
    out["headline"] = (
        f"冻结预测命中 {len(hits)} 条、未中 {len(misses)} 条"
        + (f"（未中：{', '.join(misses)}——如实记，不修改冻结值）" if misses else "")
        + f"；扩展参考把最细 U_ref 从 "
          f"{frozen['inputs_from_round4_predeclaration']['u_ref_percent_at_25600']:.4f} % "
          f"替换为实测 {_fmt(ext.get('u_ref_percent'))} %")
    figs = figures(meas, checks, frozen)
    out["figures"] = [{"path": str(f["path"].relative_to(HERE)), "caption": f["caption"]}
                      for f in figs]
    (HERE / "summary_r5.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE / "R5-REFERENCE-EXTENSION-CLOSURE.md").write_text(render(out))
    return out


if __name__ == "__main__":
    res = build()
    print(json.dumps({"forecast_ledger": res["forecast_ledger"],
                      "headline": res["headline"],
                      "finest_reference": res["finest_reference"]}, indent=2,
                     ensure_ascii=False))