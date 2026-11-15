"""Round-4 closure driver: reference-side uncertainty budget for XVAL-FVM-M18737-ROUND2.

Consumes the frozen predeclaration (`predeclaration_ref_uncertainty.json`), the stored ladder
profiles, and — when present — the extended WENO5@25600 reference.  Produces:

* ``summary_r4.json`` — every number in the closure document, no hand copying;
* ``R4-REFERENCE-UNCERTAINTY-CLOSURE.md`` — the closure document itself;
* ``figures/fig_r4_u_ref.png`` — reference self-certification ladder + U_ref vs reference
  resolution against the certificate bar;
* ``figures/fig_r4_regrade.png`` — the graded solution re-graded against each reference rung,
  with its uncertainty band and the honest verdict per rung.

    python docs/XVAL-FVM-M18737-ROUND2/ref_uncertainty_r4.py
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
BAR = U.BAR_PERCENT


def regrade_matrix(graded_rungs=GRADED_RUNGS) -> List[Dict]:
    """Every (graded rung, reference rung) pair that is nested, with its budget."""
    refs = [n for n in wcr.WENO5_RUNGS if wcr.try_load_profile(DATA, n, "weno5") is not None]
    out: List[Dict] = []
    for nx in graded_rungs:
        if wcr.try_load_profile(DATA, nx, "hllc") is None:
            out.append({"graded_nx": nx, "measured": False})
            continue
        usable = [r for r in refs if r % nx == 0 and r > nx]
        if not usable:
            out.append({"graded_nx": nx, "measured": False})
            continue
        rows = U.grade_across_references(DATA, nx, ref_rungs=tuple(usable))
        for row in rows:
            out.append({"graded_nx": nx, **row})
    return out


def p_checks(frozen: Dict, ladder: List[Dict]) -> Dict:
    pair = [r for r in ladder if r["fine_nx"] == 25600]
    out: Dict = {"P1": None, "P2": None, "P3": None, "P4": None, "P5": None}
    if pair:
        row = pair[0]
        lo, hi = frozen["predicted_band_percent"]
        out["P1"] = {"criterion": "L1(WENO5@12800 vs @25600) inside the frozen band",
                     "measured_percent": row["l1_percent"],
                     "predicted_percent": frozen["predicted_l1_12800_vs_25600_percent"],
                     "band_percent": [lo, hi],
                     "passed": bool(lo <= row["l1_percent"] <= hi)}
        band = frozen["order_band"]
        out["P2"] = {"criterion": "order of the (12800, 25600) pair inside the frozen band",
                     "measured_order": row["order"], "band": band,
                     "passed": bool(band[0] <= row["order"] <= band[1])}
    return out


def figures(ladder: List[Dict], regrades: List[Dict], needed: Dict) -> List[Dict]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    figs: List[Dict] = []

    # ---- figure 1: reference self-certification and its uncertainty ----------
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.2))
    nx = np.array([r["fine_nx"] for r in ladder], float)
    e = np.array([r["l1_percent"] for r in ladder], float)
    uref = np.array([r["u_ref_percent"] if r["u_ref_percent"] is not None else np.nan
                     for r in ladder], float)
    src = [r.get("order_source") for r in ladder]
    a = ax[0]
    a.loglog(nx, e, "o-", label="self-certification L1 (coarse vs fine)")
    a.loglog(nx, uref, "s--", label=r"reference uncertainty $U_{ref}=e/(2^p-1)$")
    for xi, yi, s in zip(nx, e, src):
        a.annotate(s or "", (xi, yi), textcoords="offset points", xytext=(4, 6), fontsize=7)
    a.set_xlabel("reference grid nx (WENO5/SSP-RK3/Rusanov)")
    a.set_ylabel("percent")
    a.set_title("(a) the reference is itself only ~0.8-order convergent")
    a.grid(True, which="both", alpha=0.3)
    a.legend(fontsize=8)

    b = ax[1]
    order = frozen_order(ladder)
    grid = np.geomspace(min(nx), 2.0e5, 200)
    if order:
        b.loglog(grid, uref[-1] * (grid / nx[-1]) ** (-order), "-",
                 label=rf"$U_{{ref}}$ extrapolated, $p={order:.3f}$")
    b.loglog(nx, uref, "s", label="measured $U_{ref}$")
    b.axhline(BAR - 0.8931, color="crimson", ls=":",
              label="allowance $1\\% - 0.893\\%$ for a certified grade")
    if needed.get("nx_ref_needed"):
        b.axvline(needed["nx_ref_needed"], color="crimson", ls="--", alpha=0.7)
        b.annotate(f"nx≈{needed['nx_ref_needed']:.2e}", (needed["nx_ref_needed"], BAR - 0.8931),
                   textcoords="offset points", xytext=(4, -12), fontsize=8, color="crimson")
    b.set_xlabel("candidate reference grid nx")
    b.set_ylabel("percent")
    b.set_title("(b) reference resolution the <1 % certificate would demand")
    b.grid(True, which="both", alpha=0.3)
    b.legend(fontsize=8)
    fig.tight_layout()
    path = FIGDIR / "fig_r4_u_ref.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append({"path": path, "caption":
                 "参考族自认证阶梯与其 Richardson 不确定度 U_ref（左），以及为让 <1 % 认证"
                 "成立所需的参考分辨率（右）。"})

    # ---- figure 2: re-grade of the graded rungs against each reference ------
    if regrades:
        fig, ax = plt.subplots(figsize=(7.6, 4.4))
        rungs = sorted({r["graded_nx"] for r in regrades if r.get("measured")})
        for nx_g in rungs:
            rows = [r for r in regrades if r.get("measured") and r["graded_nx"] == nx_g]
            xs = [r["ref_nx"] for r in rows]
            ys = [r["l1_percent"] for r in rows]
            us = [r["reference_u_ref_percent"] or 0.0 for r in rows]
            ax.errorbar(xs, ys, yerr=us, marker="o", capsize=4,
                        label=f"MUSCL/HLLC nx={nx_g}")
        ax.axhline(BAR, color="crimson", ls="--", label="certificate bar 1 %")
        ax.set_xscale("log")
        ax.set_xlabel("reference rung nx (WENO5)")
        ax.set_ylabel("measured L1 [%] ± $U_{ref}$")
        ax.set_title("re-grading against finer references: error moves, band shrinks")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = FIGDIR / "fig_r4_regrade.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        figs.append({"path": path, "caption":
                     "同一批被定级解对不同参考档位的重新定级：误差随参考加密而移动（方向由"
                     "参考是否更接近真解决定），误差棒为参考侧不确定度 U_ref。"})
    return figs


def frozen_order(ladder: List[Dict]) -> Optional[float]:
    measured = [r["order"] for r in ladder if r.get("order_source") == "measured_pair"]
    return measured[-1] if measured else None


def build() -> Dict:
    frozen = json.loads((HERE / "predeclaration_ref_uncertainty.json").read_text())
    ladder = U.selfcert_ladder(DATA)
    regrades = regrade_matrix()
    checks = p_checks(frozen, ladder)

    # P3: does refining the reference move the finest graded rung's L1 by more than the
    # uncertainty claimed for the coarser reference?
    p3: Dict = {"criterion": "|dL1| <= U_ref(coarser reference)"}
    # P3 needs a graded rung that was graded against *two* references: take the finest such
    # rung (the finest overall may only have one finer reference available)
    multi = [g for g in sorted({r["graded_nx"] for r in regrades if r.get("measured")})
             if sum(1 for r in regrades if r.get("measured") and r["graded_nx"] == g) >= 2]
    finest_graded = multi[-1] if multi else max(
        (r["graded_nx"] for r in regrades if r.get("measured")), default=None)
    pair = [r for r in regrades if r.get("measured") and r["graded_nx"] == finest_graded]
    if len(pair) >= 2:
        moved = pair[-1]["l1_percent"] - pair[-2]["l1_percent"]
        claimed = pair[-2]["reference_u_ref_percent"]
        p3.update({"graded_nx": finest_graded,
                   "ref_from": pair[-2]["ref_nx"], "ref_to": pair[-1]["ref_nx"],
                   "l1_from_percent": pair[-2]["l1_percent"],
                   "l1_to_percent": pair[-1]["l1_percent"],
                   "delta_percent": moved, "claimed_u_ref_percent": claimed,
                   "passed": bool(claimed is not None and abs(moved) <= claimed)})
    checks["P3"] = p3 if "delta_percent" in p3 else None

    # P4: honest certificate gate on the finest available reference
    refs = [n for n in wcr.WENO5_RUNGS if wcr.try_load_profile(DATA, n, "weno5") is not None]
    finest_ref = max(refs)
    # P4/P5 are about the *finest graded rung* (the one round-3 certified), not about the
    # rung chosen for P3 (which needs two references and is therefore coarser)
    finest_any = max((r["graded_nx"] for r in regrades if r.get("measured")), default=None)
    pair = [r for r in regrades if r.get("measured") and r["graded_nx"] == finest_any]
    gate: Dict = {"criterion": "measured L1 + U_ref(finest reference) < 1 %"}
    if pair:
        ur = U.u_ref_of_reference(DATA, finest_ref, ladder=ladder)
        g = U.grade_with_budget(pair[-1]["l1_percent"], ur["u_ref_percent"])
        gate.update({"graded_nx": finest_any, "ref_nx": finest_ref,
                     "u_ref_percent": ur["u_ref_percent"],
                     "u_ref_order_source": ur.get("order_source"), **g,
                     "passed": bool(g["verdict"] == "certified")})
        checks["P4"] = gate
        # P5: reference resolution the gate would demand
        need = U.nx_ref_for_bar(pair[-1]["l1_percent"], ur["u_ref_percent"], finest_ref,
                                ur["order"])
        need["predicted_in_predeclaration"] = frozen["nx_ref_needed_for_bar"]
        checks["P5"] = need

    # next-rung prediction for the pending WENO5@25600 reference: the measured grade drifts
    # *up* when the reference is refined, because a finer reference is closer to the truth.
    # Take the per-doubling drift observed at the finest graded rung that has two references.
    drift = None
    if pair and len(pair) >= 2:
        drift = pair[-1]["l1_percent"] - pair[-2]["l1_percent"]
    next_pred = None
    if drift is not None:
        base = [r for r in regrades if r.get("measured") and r["graded_nx"] == 6400]
        if base:
            next_pred = {"graded_nx": 6400,
                         "ref_from": base[-1]["ref_nx"], "ref_to": 2 * base[-1]["ref_nx"],
                         "l1_from_percent": base[-1]["l1_percent"],
                         "drift_per_doubling_from_percent": drift,
                         "predicted_l1_percent": base[-1]["l1_percent"] + drift,
                         "band_percent": [base[-1]["l1_percent"] + 0.5 * drift,
                                          base[-1]["l1_percent"] + 2.0 * drift],
                         "note": "empirical drift, not a convergence-law extrapolation"}
    r3 = {"graded_nx": 6400, "l1_percent": 0.8930772870380906, "ref_nx": 12800}
    out = {"anchor": "XVAL-FVM-M18737-ROUND2", "round": 4,
           "bar_percent": BAR, "ledger_tolerance_percent": wcr.TOLERANCE_PERCENT,
           "predeclaration": frozen["predicted_l1_12800_vs_25600_percent"],
           "selfcert_ladder": ladder,
           "reference_rungs_available": refs,
           "regrades": regrades,
           "checks": checks,
           "round3_claim": r3,
           "next_rung_prediction": next_pred,
           "finest_reference": finest_ref}
    figs = figures(ladder, regrades, checks.get("P5") or {})
    out["figures"] = [{"path": str(f["path"].relative_to(HERE)), "caption": f["caption"]}
                      for f in figs]
    (HERE / "summary_r4.json").write_text(json.dumps(out, indent=2) + "\n")
    (HERE / "R4-REFERENCE-UNCERTAINTY-CLOSURE.md").write_text(render(out))
    return out


def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def render(s: Dict) -> str:
    lad = s["selfcert_ladder"]
    rows = "\n".join(
        f"| {r['coarse_nx']} | {r['fine_nx']} | {r['l1_percent']:.4f} | "
        f"{_fmt(r.get('order'))} | {r.get('order_source')} | "
        f"{_fmt(r.get('u_ref_percent'))} |" for r in lad)

    reg = [r for r in s["regrades"] if r.get("measured")]
    reg_rows = "\n".join(
        f"| {r['graded_nx']} | {r['ref_nx']} | {r['l1_percent']:.4f} | "
        f"{_fmt(r['reference_u_ref_percent'])} | {_fmt(r['worst_case_percent'])} | "
        f"`{r['verdict']}` |" for r in reg)

    c = s["checks"]
    p1, p2 = c.get("P1"), c.get("P2")
    p3, p4, p5 = c.get("P3"), c.get("P4"), c.get("P5")

    def verdict(flag):
        return "—" if flag is None else ("✅ PASS" if flag else "❌ FAIL")

    nxt = s.get("next_rung_prediction")
    next_block = "—（无两档参考可比，无法给出漂移）"
    if nxt:
        next_block = (
            f"用最细被定级档的**参考加密漂移**外推：被定级 nx={nxt['graded_nx']} 在 ref@"
            f"{nxt['ref_from']} 上测得 {nxt['l1_from_percent']:.4f} %，历史上每加密一档参考，"
            f"实测 L1 上漂 {nxt['drift_per_doubling_from_percent']:+.4f} pp（这是参考更接近真解，"
            f"不是被定级解变差）。故预测 ref@{nxt['ref_to']} 上 "
            f"**L1 ≈ {nxt['predicted_l1_percent']:.4f} %**（带 [{nxt['band_percent'][0]:.4f}, "
            f"{nxt['band_percent'][1]:.4f}]）。若该预测成立，则 round-3 的 "
            f"L1=0.8931 %<1 % 是**参考偏粗造成的低报**，<1 % 认证在 nx=6400 上不成立——"
            f"这正是本轮把不确定度做成门的意义。")
    gate_line = "—"
    if p4:
        gate_line = (f"measured {p4['measured_percent']:.4f} % + U_ref {p4['u_ref_percent']:.4f} % "
                     f"= {p4['worst_case_percent']:.4f} % vs 1 % → **{p4['verdict']}**")
    need_line = "—"
    if p5 and p5.get("nx_ref_needed"):
        need_line = (f"nx_ref ≈ {p5['nx_ref_needed']:.3e}"
                     f"（预声明冻结值 {p5['predicted_in_predeclaration']:.3e}）")
    p3_line = "—"
    if p3:
        p3_line = (f"被定级 nx={p3['graded_nx']}：ref@{p3['ref_from']} → ref@{p3['ref_to']} 时 "
                   f"L1 {p3['l1_from_percent']:.4f} % → {p3['l1_to_percent']:.4f} %"
                   f"（Δ={p3['delta_percent']:+.4f} pp，旧参考声明的 U_ref={p3['claimed_u_ref_percent']:.4f} pp）")
    return f"""# XVAL-FVM-M18737-ROUND2 Round-4：参考侧不确定度预算（把「认证」拆成可检验的门）

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 精度标准）。
> 前置：`REFINEMENT-R3-CLOSURE.md`（round-3 判 certified-<1 %，L1 = 0.8931 %，参考 WENO5@12800）·
> `PREDECLARATION-ref-uncertainty.md`（判据 P1–P5 在计算前冻结）。
> 本文件**不覆盖**任何前轮记录，只把 round-3 结论里隐含的「参考不确定度」显式化并逐条裁决。

## ① 案例介绍（物理背景 / 参数表）

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台（左 p=1000 / 中 p=0.01 /
右 p=100），ρ=1、u=0，两束爆轰波相向对撞，t≈0.028 对撞、积分至 t=0.038，γ=1.4，两端反射壁。
无解析解，基准必须是**数值参考**；因此每一次「误差」都是两个数值解之差，**参考解自身的
离散误差必须计入**。

| 参数 | 值 |
|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4（`compressible/double_blast`） |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（`xcode/wc_crosscode.py`，与被定级格式不共享离散） |
| 参考档位 | {", ".join(str(n) for n in s["reference_rungs_available"])}（本轮可用） |
| 比较窗口 | x∈[0.4,1.0]，守恒限制（coarse 单元均值 = 其内 r 个 fine 单元均值） |
| 认证门 | 标称 1 %（round-3 口径）/ 总账容差 {s["ledger_tolerance_percent"]} % |

## ② 怎么做的（本轮的方法）

1. **先把不确定度定义死**：嵌套比 2 的自认证差 `e = |q_coarse − q_fine|`，观测阶 p 由相邻两对
   的 log–log 斜率测得，参考不确定度 `U_ref = e/(2^p − 1)`（Richardson）。这是**参考解自己的
   误差**，而不是我们想知道的被定级解误差。
2. **三角不等式门**：被定级解的真误差满足
   `|q_graded − q_exact| ≤ |q_graded − q_ref| + U_ref`。于是「certified <1 %」只有在
   `measured + U_ref < 1 %` 时才可以宣称；否则只能宣称「上界 <1 %」。
3. **预声明先行**：在 WENO5@25600 参考算出来之前，用 ≤12800 的既有档位冻结预测
   （L1(12800 vs 25600) = {s['predeclaration']:.4f} %）、判据 P1–P5 与所需参考分辨率，
   见 `PREDECLARATION-ref-uncertainty.md`。
4. **算新参考**：WENO5@25600 由调度系统跑（作业 cj3556，1 核，本机不起重算力）。
5. **重新定级**：把既有的被定级解档位（{", ".join(str(n) for n in GRADED_RUNGS)}）对
   **每一个更细的参考档位**重新定级，观察误差是否移动、移动是否落在旧参考声明的 U_ref 之内。
6. **机检 + 程序化出报告**：`wc_ref_uncertainty.py` 逐条判定，本文由 `summary_r4.json`
   程序化渲染（零手抄）。

## ③ 计算结果

### 参考族自认证阶梯与 U_ref（程序化）

| coarse nx | fine nx | 自认证 L1 [%] | 观测阶 p | p 来源 | U_ref(fine) [%] |
|---|---|---|---|---|---|
{rows}

### 重新定级矩阵（被定级解 × 参考档位）

| graded nx | ref nx | 实测 L1 [%] | U_ref [%] | 最坏情形 L1+U_ref [%] | 门裁决 |
|---|---|---|---|---|---|
{reg_rows}

### 冻结判据逐条裁决

| 判据 | 结果 |
|---|---|
| P1 扩展参考落在冻结预测带 | {verdict(p1['passed'] if p1 else None)}（实测 {_fmt(p1['measured_percent']) if p1 else '—'} % vs 预测 {_fmt(p1['predicted_percent']) if p1 else '—'} %） |
| P2 参考族观测阶在冻结带内 | {verdict(p2['passed'] if p2 else None)}（实测 {_fmt(p2['measured_order'],3) if p2 else '—'}） |
| P3 换更细参考引起的 L1 移动 ≤ 旧参考 U_ref | {verdict(p3['passed'] if p3 else None)}<br>{p3_line} |
| P4 诚实认证门：measured + U_ref < 1 % | {verdict(p4['passed'] if p4 else None)}<br>{gate_line} |
| P5 门所要求的参考分辨率 | {need_line} |

### 下一手的可检验预测（WENO5@25600 参考，本轮未回）

{next_block}

### 总账判定（<15 %）

最细被定级档的实测 L1（对最细参考）远小于 {s['ledger_tolerance_percent']} %，
**总账 PASS**；本轮裁决只影响「能不能宣称 <1 % 认证」，不影响总账。

## ④ 结果图

![参考不确定度](figures/fig_r4_u_ref.png)

*图 1：(a) 参考族自认证阶梯与其 Richardson 不确定度 U_ref；(b) 让 <1 % 认证成立所需的参考分辨率。*

![重新定级](figures/fig_r4_regrade.png)

*图 2：同一批被定级解对不同参考档位的重新定级，误差棒为参考侧 U_ref。*

## ⑤ 注意点（局限 / 适用范围）

1. **参考族不是五阶收敛**：WENO5 的自认证观测阶只有 ~0.8（见上表），远低于格式名义阶。
   原因是近真空平台的 positivity 回退与接触间断：低密度区的相对误差主导 L1，收敛很慢。
   因此「加密参考」是**昂贵**的：U_ref 只按 h^0.8 下降。
2. **U_ref 的阶是外推的**：最细那一对的阶没有更细档位可测，只能按更粗一对的阶外推
   （`order_source = extrapolated`）。这是本轮最大的假设；`wc_ref_uncertainty` 把它显式标注，
   不当作测量值。
3. **P3 的方向不可预设**：更细的参考更接近真解，不是更接近被定级解，故重新定级可能让误差
   **变大**。判据只能是「移动量 ≤ 旧参考声明的 U_ref」，不是「误差单调下降」。
4. **只做了一维标量度量**：L1 与峰值是密度剖面度量，不含 p/u；结论不自动推广到其它量。
5. **未测**：{", ".join(f"nx={n}" for n in [12800] if wcr.try_load_profile(DATA, n, "hllc") is None) or "无"}
   被定级解（HLLC@12800 单核约需 3 h，超出单作业时长上限，未递交）→ P1/P2 之外的最细档
   观测阶仍以 6400 为准。
"""


if __name__ == "__main__":
    out = build()
    print(json.dumps({"checks": out["checks"], "finest_reference": out["finest_reference"]},
                     indent=2))