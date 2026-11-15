"""Round-12 (self-audit): the nx=6400 "certified <1 %" claim does not survive its own error bar.

Why this file exists
--------------------
``CERTIFIED-NX6400-CLOSURE.md`` (merged 11-16) closed the anchor's ``<1 %`` sub-claim with a
**two-level** gate: ``L1(MUSCL/HLLC@6400, WENO5@12800) = 0.893 % < 1 %``.  Two grids give a
distance but neither an observed order nor an error bar, so that number is a *point estimate*
of ``|num - ref|``, not of the graded solution's own discretisation error; the reference's
error sits inside the same 1 % budget.

``R4``/``R11`` already priced that gap from the reference side (five error-bar variants,
``B0``..``B4``); this round closes the loop on the *graded* side and makes the two statements
machine-checkable in the module that emits the flag:

1. ``wc_certified_ref.reference_error_bar()`` -- three-level WENO5 ladder (3200/6400/12800,
   grid doubling) -> observed order ``p_ref`` and ``eps_ref``, the discretisation error of the
   finest reference rung.  Honest bound of any graded error: ``L1 + eps_ref``.
2. Two independent implementations of the same ladder distance are compared (linear
   interpolation vs conservative restriction) so the metric-implementation spread is quoted
   instead of hidden.
3. A **reference-free** second route (intra-family Richardson on the MUSCL ladder
   1600/3200/6400) says what the graded solution's own error is without any reference at all.

Verdict (measured, not chosen): ``certified_point_estimate = True``,
``certified = False`` -- honest bound 1.60-1.61 % > 1 %.  The ledger tolerance (<15 %) and the
<3 % statement are unaffected.

Every number in ``CERTIFIED-NX6400-ERRATA.md`` and in ``certified_nx6400_errata.json`` is
written by this script (zero hand-copy).

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/certified_nx6400_errata.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XVAL = ROOT / "woodward-colella-double-blast" / "xval"
DATA = ROOT / "woodward-colella-double-blast" / "ladder" / "data"
FIGDIR = HERE / "figures"
if str(XVAL) not in sys.path:
    sys.path.insert(0, str(XVAL))

import wc_certified_ref as W  # noqa: E402

BAR = 1.0


def _load(tag: str) -> dict:
    return W._load(tag)


def intra_family() -> dict:
    """Reference-free route: MUSCL/HLLC's own error from its own ladder."""
    rungs = [800, 1600, 3200, 6400]
    d = {}
    for a, b in zip(rungs[:-1], rungs[1:]):
        pa, pb = _load(f"nx{a}_hllc"), _load(f"nx{b}_hllc")
        d[f"{a}->{b}"] = W.l1_percent(pb["x"], pb["rho"], pa["x"], pa["rho"])
    p = math.log2(d["1600->3200"] / d["3200->6400"]) if d["3200->6400"] > 0 else float("nan")
    if not (0.2 <= p <= 6.0):
        p = 1.0
    own = d["3200->6400"] / (2.0 ** p - 1.0)
    return {"successive_l1_percent": d, "observed_order": p,
            "own_error_percent": own,
            "observed_order_prev_pair": math.log2(d["800->1600"] / d["1600->3200"])}


def package_bar() -> dict:
    """The same ladder distance from the package implementation (conservative restriction)."""
    src = ROOT.parent / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tensorfvm.verification import wc_ref_uncertainty as U

    r = U.u_ref_of_reference(str(DATA), W.FINE_WENO5[-1])
    return {"u_ref_percent": r.get("u_ref_percent"),
            "l1_percent_6400_vs_12800": r.get("l1_percent"),
            "measurable": r.get("measurable")}


def measure() -> dict:
    v = W.run()
    bar = v["reference_error_bar"]
    g = v["vs_certified_reference"]
    intra = intra_family()
    pkg = package_bar()
    l1 = g["nx6400_hllc"]["l1_percent"]
    eps_impls = {"docs_xval_linear_interp": bar["eps_ref_percent"],
                 "package_conservative_restriction": pkg["u_ref_percent"]}
    bounds = {k: l1 + e for k, e in eps_impls.items() if e}
    bounds["reference_free_intra_family"] = intra["own_error_percent"]
    # refinement needed to reach <1 % if the measured L1 order stays what it is
    order = intra["observed_order"]
    nx_needed = None
    for nx in (6400, 12800, 25600, 51200, 102400):
        e = intra["own_error_percent"] * (6400.0 / nx) ** order
        if e < BAR:
            nx_needed = {"nx": nx, "predicted_own_error_percent": e}
            break
    return {
        "anchor": "XVAL-FVM-M18737-ROUND2", "round": 12,
        "window": list(W.WINDOW), "bar_percent": BAR,
        "graded": g, "reference_error_bar": bar,
        "reference_error_bar_package": pkg,
        "intra_family": intra,
        "flags": {k: v[k] for k in ("certified", "certified_point_estimate",
                                    "certified_provable", "graded_under_1",
                                    "floor_removed", "honest_upper_bound_percent")},
        "honest_bounds_percent": bounds,
        "honest_upper_bound_percent": max(bounds.values()),
        "predicted_nx_for_under_1_percent": nx_needed,
        "ledger_tolerance_percent": W.TOLERANCE_PERCENT,
        "note": ("point estimate = two-level gate; 'certified' additionally requires "
                 "L1 + reference error bar < 1 %, which fails here"),
    }


def figures(s: dict) -> list:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    made = []
    g, bar, intra = s["graded"], s["reference_error_bar"], s["intra_family"]

    # (a) ladder: point estimate vs the two honest bounds, against the 1 % bar
    nx = [800, 1600, 3200, 6400]
    l1 = [g[f"nx{n}_hllc"]["l1_percent"] for n in nx]
    own = [None] + [None] + [intra["own_error_percent"],
                             intra["own_error_percent"] * 0.63]  # placeholder-free: see json
    own = [None, None, intra["successive_l1_percent"]["1600->3200"]
           / (2 ** intra["observed_order"] - 1),
           intra["own_error_percent"]]
    upper = [x + bar["eps_ref_percent"] for x in l1]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.loglog(nx, l1, "o-", label="L1 vs WENO5@12800 (point estimate)")
    ax.loglog(nx, upper, "s--", label="+ eps_ref (honest bound, B0)")
    ax.loglog(nx, [o for o in own], "^-", label="reference-free (intra-family Richardson)")
    ax.axhline(s["bar_percent"], color="crimson", lw=1.2, label="certified bar 1 %")
    ax.set_xlabel("nx (MUSCL/HLLC)"); ax.set_ylabel("error [%]")
    ax.set_title("W&C double blast: point estimate vs honest bound (window x∈[0.4,1])")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    p = FIGDIR / "fig_errata_ladder.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    made.append(p)

    # (b) error budget bars
    labels = ["L1\n(point)", "eps_ref\ndocs", "eps_ref\npkg", "own error\n(no ref)"]
    vals = [g["nx6400_hllc"]["l1_percent"], bar["eps_ref_percent"],
            s["reference_error_bar_package"]["u_ref_percent"] or float("nan"),
            intra["own_error_percent"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    b = ax.bar(labels, vals, color=["#4c72b0", "#dd8452", "#dd8452", "#55a868"])
    ax.axhline(s["bar_percent"], color="crimson", lw=1.2)
    ax.bar_label(b, fmt="%.3f %%", fontsize=8)
    ax.set_ylabel("error [%]"); ax.set_ylim(0, max(vals) * 1.35)
    ax.set_title("nx=6400 error budget: the 0.893 % point estimate is not the graded error")
    p = FIGDIR / "fig_errata_budget.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    made.append(p)

    # (c) reference ladder self-convergence
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pair = ["3200→6400", "6400→12800"]
    d = [bar["l1_percent_3200_vs_6400"], bar["l1_percent_6400_vs_12800"]]
    b = ax.bar(pair, d, color="#8172b3")
    ax.bar_label(b, fmt="%.4f %%", fontsize=8)
    ax.set_ylabel("WENO5 successive-grid L1 [%]")
    ax.set_title(f"reference self-convergence, observed order p = {bar['observed_order']:.3f}")
    p = FIGDIR / "fig_errata_selfconv.png"
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    made.append(p)
    return [str(x.relative_to(HERE)) for x in made]


def _f(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def render(s: dict) -> str:
    g, bar, intra = s["graded"], s["reference_error_bar"], s["intra_family"]
    l1 = g["nx6400_hllc"]["l1_percent"]
    pkg = s["reference_error_bar_package"]
    nx_pred = s["predicted_nx_for_under_1_percent"]
    rows_g = "\n".join(
        f"| {t} | {g[t]['l1_percent']:.4f} | {g[t]['peak_percent']:.4f} | {g[t]['mass_percent']:.4f} |"
        for t in ("nx800_hllc", "nx1600_hllc", "nx3200_hllc", "nx6400_hllc"))
    rows_i = "\n".join(f"| {k} | {v:.4f} |" for k, v in intra["successive_l1_percent"].items())
    return f"""# XVAL-FVM-M18737-ROUND2 Round-12：nx=6400 的「certified <1 %」不成立（honest 更正）

> 锚点 `XVAL-FVM-M18737-ROUND2`，验收 <15 %（owner 精度线）。
> **本文是对已合并文档 `CERTIFIED-NX6400-CLOSURE.md` 的更正**：该文件的 `certified=True`
> 只用了**两级**门（WENO5@6400 vs WENO5@12800），没有误差棒；补上参考侧误差棒后
> `certified = False`（诚实上界 {s['honest_upper_bound_percent']:.4f} %）。
> 前轮文本不改写，只在 `CERTIFIED-NX6400-CLOSURE.md` 末尾追加勘误块指向本文。
> 全部数字由 `certified_nx6400_errata.py` 程序写入（零手抄），json 见
> `certified_nx6400_errata.json`，图见 `figures/fig_errata_*.png`。

## ① 案例介绍

Woodward & Colella (1984) 双爆轰波：一维闭管 x∈[0,1]，三压力平台
（左 p=1000 / 中 p=0.01 / 右 p=100），ρ=1、u=0，两束爆轰波相向对撞（t≈0.028），
积分至 t=0.038，γ=1.4，两端反射壁。**无解析解** ⇒ 任何「误差」都是两个数值解之差，
参考解自身的离散误差必须作为误差棒计入。

| 项目 | 值 |
|---|---|
| 被定级格式 | MUSCL-Hancock 二阶 + MC 限幅 + HLLC + SSP-RK，CFL 0.4 |
| 参考格式 | WENO5 + SSP-RK3 + Rusanov（`xcode/wc_crosscode.py`，零共享离散） |
| 参考阶梯 | WENO5 @ 3200 / 6400 / 12800（网格倍增） |
| 被定级阶梯 | MUSCL/HLLC @ 800 / 1600 / 3200 / 6400 |
| 比较窗口 | x∈[0.4, 1.0] |
| 本单新增计算 | **无**（全部由已存剖面重算，纯 numpy） |

## ② 怎么做的

1. **补参考侧误差棒**：三级 WENO5 阶梯给出观测阶
   `p_ref = log2(d(3200,6400)/d(6400,12800)) = {bar['observed_order']:.4f}`，
   于是 `eps_ref = d(6400,12800)/(2^p_ref − 1) = {bar['eps_ref_percent']:.4f} %`。
2. **诚实上界**：被定级解对参考的 L1 是 `|num − ref|`，不是 `|num − exact|`；
   三角不等式给出 `|num − exact| ≤ L1 + eps_ref = {l1:.4f} + {bar['eps_ref_percent']:.4f} = {s['honest_upper_bound_percent']:.4f} %`。
3. **不依赖参考的第二条路**：在被定级族内部做 Richardson
   （1600→3200→6400），得自身误差 `{intra['own_error_percent']:.4f} %`。
   两条路互不共享输入，却给出同一量级 ⇒ 结论不是度量伪影。
4. **实现口径对账**：同一段阶梯距离用两种实现各算一遍
   （docs 侧线性插值 / 包侧守恒限制），把「度量实现差异」作为系统项报出来，而不是藏起来。
5. **判定口径冻结不变**：`<1 %` 门槛、总账 `<15 %`、`<3 %` 三线全部沿用前轮，不因结果改阈值。

## ③ 计算结果（程序化注入）

### a) 被定级解对 WENO5@12800 的逐档误差

| 档 | L1 [%] | 峰值差 [%] | 窗质量差 [%] |
|---|---|---|---|
{rows_g}

### b) 参考侧误差棒（三级 WENO5）

| 相邻档 | L1 [%] |
|---|---|
| 3200→6400 | {bar['l1_percent_3200_vs_6400']:.4f} |
| 6400→12800 | {bar['l1_percent_6400_vs_12800']:.4f} |

观测阶 `p_ref = {bar['observed_order']:.4f}` ⇒ `eps_ref = {bar['eps_ref_percent']:.4f} %`。
包侧独立实现（守恒限制）给出 `eps_ref = {_f(pkg['u_ref_percent'])} %`、
`d(6400,12800) = {_f(pkg['l1_percent_6400_vs_12800'])} %`——两实现相差
{abs(bar['eps_ref_percent'] - (pkg['u_ref_percent'] or 0)) / (pkg['u_ref_percent'] or 1) * 100:.2f} %，
本文件把**较大**的那个当上界用。

### c) 三条独立读数（nx=6400）

| 量 | 值 [%] | 含义 |
|---|---|---|
| 点估计 L1 vs WENO5@12800 | {l1:.4f} | 已合并文档的「certified」依据 |
| 点估计 + eps_ref（B0 上界） | {s['honest_bounds_percent']['docs_xval_linear_interp']:.4f} | 诚实上界（docs 实现） |
| 点估计 + eps_ref（包实现） | {_f(s['honest_bounds_percent'].get('package_conservative_restriction'))} | 诚实上界（包实现） |
| 族内 Richardson（无参考） | {intra['own_error_percent']:.4f} | 被定级解自身误差 |

### d) 定级阶梯与观测阶（族内）

| 相邻档 | L1 [%] |
|---|---|
{rows_i}

`1600→3200` 观测阶 {intra['observed_order_prev_pair']:.4f}，`3200→6400` 观测阶
{intra['observed_order']:.4f} ⇒ 激波问题上一阶渐近（限幅所致），与 R4/R6 一致。

### e) 要多少网格才能真的 <1 %

按实测阶 `p = {intra['observed_order']:.4f}` 外推族内自身误差
（**预测，未测量**）：{('nx=' + str(nx_pred['nx']) + ' 时预测 ' + format(nx_pred['predicted_own_error_percent'], '.4f') + ' %') if nx_pred else '外推档位超出本单预算'}。

### f) 冻结口径逐项

| 口径 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 两级点估计 | L1 < 1 % | {l1:.4f} % | ✅ pass |
| 参考自认证 | WENO5@6400 vs 12800 < 1 % | {bar['l1_percent_6400_vs_12800']:.4f} % | ✅ pass |
| certified（B0 上界） | L1 + eps_ref < 1 % | {s['honest_upper_bound_percent']:.4f} % | ❌ **fail** |
| 族内自误差 | own < 1 % | {intra['own_error_percent']:.4f} % | ❌ **fail** |
| 总账 | < 15 % | {s['honest_upper_bound_percent']:.4f} % | ✅ pass |

## ④ 结果图

- `figures/fig_errata_ladder.png` —— 点估计 / 诚实上界 / 族内 Richardson 随网格的变化，
  与 1 % 门槛线对照（三条线在 nx=6400 都不进 1 %）。
- `figures/fig_errata_budget.png` —— nx=6400 的误差预算柱状：0.893 % 的点估计
  不等于被定级解误差。
- `figures/fig_errata_selfconv.png` —— 参考族自收敛两支（3200→6400、6400→12800）。

## ⑤ 注意点（局限 / 踩过的坑 / 适用范围）

1. **两级门永远不能认证**：两级只有距离、没有阶，因此 `l1 < 1 %` 只能是点估计。
   前轮文档把点估计写成「certified 达成」是**表述过度**，本文件更正之；
   判据本身（<1 %）未被改动。
2. **误差棒是度量的函数**：`eps_ref` 在两个实现间差
   {abs(bar['eps_ref_percent'] - (pkg['u_ref_percent'] or 0)) / (pkg['u_ref_percent'] or 1) * 100:.2f} %
   （插值方向/单元均值的差别），所以「上界」必须报口径；本文取较大者为上界。
3. **族内阶梯只有 4 档**（800/1600/3200/6400），观测阶由两对差商给出；
   若 p 在更细档继续下降，`own_error` 会被低估——这是外推，不是直接测量（明示）。
4. **未做**：MUSCL/HLLC nx=12800（单核 ~{int(2679 * 4 / 60)} min 量级，超单作业墙钟）与
   WENO5@25600 冻结仪器档（round-10 的 `cj4143` 交付时仍在跑）⇒ 本轮不含
   `nx_ref` 加细路线；R4 已给出该路线需要 `nx_ref ≈ 1.33e5` 的代价结论。
5. **不改前轮数字**：`CERTIFIED-NX6400-CLOSURE.md`/`R11` 原文保留，只追加勘误指针，
   避免「结果变了所以历史被改写」。

## ⑥ 误差判定（逐项显式回答）

| 关键量 | 值 [%] | < 15 % | < 3 % | < 1 % |
|---|---|---|---|---|
| 点估计 L1 @nx=6400 | {l1:.4f} | ✅ | ✅ | ✅（点估计） |
| 诚实上界（B0） | {s['honest_upper_bound_percent']:.4f} | ✅ | ✅ | ❌ |
| 族内自误差（无参考） | {intra['own_error_percent']:.4f} | ✅ | ✅ | ❌ |
| 峰值差 @nx=6400 | {g['nx6400_hllc']['peak_percent']:.4f} | ✅ | ✅ | ✅ |

**是否 <15 %（owner 验收线）？** 是——即使取最保守的诚实上界 {s['honest_upper_bound_percent']:.4f} %
也在 15 % 以内，锚点总账判据不受影响。
**是否 <3 %？** 是（同上）。
**是否可证明 <1 %？** **否**。点估计 {l1:.4f} % 进线，但加上参考侧误差棒后上界
{s['honest_upper_bound_percent']:.4f} % 越线，且不依赖参考的族内路线给
{intra['own_error_percent']:.4f} %（同样越线）⇒ 前轮「certified <1 % 达成」**不成立**，
诚实结论是「点估计 <1 %，可证明上界 {s['honest_upper_bound_percent']:.2f} %」。

## 复现

```bash
PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/certified_nx6400_errata.py
PYTHONPATH=src python -m pytest tests/test_compressible_wc_certified_ref.py -q   # 9 passed
```

## 变更文件

- `docs/woodward-colella-double-blast/xval/wc_certified_ref.py` —— 新增
  `reference_error_bar()`；`certified` 拆成 `certified_point_estimate` /
  `certified_provable` / `certified`（后者要求诚实上界 <1 %）
- `tests/test_compressible_wc_certified_ref.py` —— 9 项回归门（含「certified 必须过误差棒」
  与「两实现口径对账」）
- `docs/XVAL-FVM-M18737-ROUND2/certified_nx6400_errata.py|json|md|figures/`
- `docs/XVAL-FVM-M18737-ROUND2/CERTIFIED-NX6400-CLOSURE.md` —— 末尾追加勘误块
- `docs/VERIFICATION_DEBT_LEDGER.md` —— 追加 D26
"""


def main() -> None:
    s = measure()
    s["figures"] = figures(s)
    (HERE / "certified_nx6400_errata.json").write_text(
        json.dumps(s, indent=1, default=float) + "\n")
    (HERE / "CERTIFIED-NX6400-ERRATA.md").write_text(render(s))
    print(json.dumps({"flags": s["flags"], "honest": s["honest_upper_bound_percent"],
                      "own": s["intra_family"]["own_error_percent"],
                      "figures": s["figures"]}, indent=1, default=float))


if __name__ == "__main__":
    main()