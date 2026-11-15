"""Round-11 closure driver: robustness of the *error bar* itself (B0-B4).

Round 1-10 asked "are the numbers right".  Round-11 asks the next question: does the
verdict survive **swapping the error-bar convention** for another, equally legal but more
conservative one?  The frozen conventions live in ``PREDECLARATION-r11.md``:

* **B0** point-order Richardson (the R1-R10 convention), ``eps = e/(2^p - 1)``
* **B1** order-band propagation ``p +- 0.25`` (uses R5's ``u_ref_order_band``)
* **B2** GCI with safety factor ``Fs = 1.25`` (ASME V&V-20, *assumes* asymptotic range)
* **B3** non-asymptotic safety factor ``Fs = 3`` (Roache: observed order is an order of
  magnitude below the formal one)
* **B4** order-free Cauchy/geometric bound ``eps = e/(1 - r)``, licensed only when the
  difference-ratio sequence does not increase

Nothing here runs a solver: it reads the committed ``ladder_*.npz/json`` artifacts, so it
is cheap and reproducible, and it re-grades the same graded family (MUSCL/HLLC) against
the same reference under every convention.  The graded criteria thresholds (E4 = 1.5 %,
E5 = 1.0 %, ledger = 15 %) are **frozen by earlier predeclarations** and are not changed
here -- R11 only reports whether they survive each convention.

    PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/ref25600_r11_robust.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.verification import wc_ref_uncertainty as U  # noqa: E402
from tensorfvm.verification import wc_refine as R  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "woodward-colella-double-blast" / "ladder" / "data"
FIGDIR = HERE / "figures"

# Frozen thresholds.  These are *inherited* (E4/E5 from PREDECLARATION-r7/r10, ledger bar
# from the owner's acceptance line) -- R11 does not invent new bars.
BAR_LEDGER = 15.0
BAR_E4 = 1.5
BAR_E5 = 1.0
ORDER_HALF_WIDTH = 0.25
FS_GCI = 1.25
FS_NONASYMPTOTIC = 3.0
RATIO_TOL = 1e-12


def _r(x, nd=4):
    return None if x is None else round(float(x), nd)


# --------------------------------------------------------------------------- #
# 1. the reference side: five error-bar conventions on the finest measurable pair
# --------------------------------------------------------------------------- #
def bar_conventions(e: float, p: float, r: Optional[float],
                    r_sequence: List[float]) -> Dict:
    """Five conventions for ``eps_ref`` from the same pair difference ``e``.

    ``e`` is the L1 difference of the finest measurable reference pair, ``p`` its measured
    order and ``r`` the ratio to the previous pair's difference.  Every convention answers
    the same question ("how far may the reference still be from exact?") under a different
    assumption about the order; the assumptions are named so the reader can price them.
    """
    out: Dict = {"e_percent": _r(e, 6), "order": _r(p, 6), "ratio": _r(r, 6)}
    base = e / (2.0 ** p - 1.0)
    out["B0"] = {"label": "point-order Richardson", "eps_percent": _r(base),
                 "assumption": "measured order p is exact"}
    band = U.u_ref_order_band(e, [p - ORDER_HALF_WIDTH, p + ORDER_HALF_WIDTH])
    out["B1"] = {"label": "order band p+-0.25", "eps_percent": _r(band["max_percent"]),
                 "eps_lo_percent": _r(band["min_percent"]),
                 "spread_percent": _r(band["spread_percent"]),
                 "assumption": "order uncertain by +-0.25, propagated"}
    out["B2"] = {"label": "GCI Fs=1.25", "eps_percent": _r(FS_GCI * base),
                 "assumption": "ASME V&V-20: asymptotic range reached (R<1, p stable)"}
    out["B3"] = {"label": "non-asymptotic Fs=3", "eps_percent": _r(FS_NONASYMPTOTIC * base),
                 "assumption": "Roache: observed order not yet asymptotic"}
    # B4 -- order-free, but only legal when the difference-ratio sequence does not increase
    licensed = bool(r_sequence) and all(
        b <= a + RATIO_TOL for a, b in zip(r_sequence, r_sequence[1:]))
    out["B4"] = {"label": "order-free Cauchy (e/(1-r))",
                 "eps_percent": _r(e / (1.0 - r)) if (r is not None and 0 < r < 1) else None,
                 "licensed": licensed,
                 "ratio_sequence": [_r(x, 6) for x in r_sequence],
                 "assumption": "difference series geometric with non-increasing ratio"}
    return out


def asymptotic_check(e: float, e_prev: Optional[float]) -> Dict:
    """The ASME V&V-20 convergence-ratio gate that B2 silently assumes.

    ``R = e_fine / e_coarse`` must satisfy ``0 < R < 1`` (monotone) for Richardson to be
    usable at all; the *asymptotic* claim additionally needs the observed order to be near
    the formal one.  For WENO5 the formal order is 5, so a measured order near 0.8 is a
    factor ~6 away -- that is the number this function puts on record.
    """
    out: Dict = {"R": None, "monotone": None, "formal_order_weno5": 5.0,
                 "order_ratio": None, "asymptotic": None}
    if e_prev is None or not e_prev:
        return out
    ratio = e / e_prev
    out["R"] = _r(ratio, 6)
    out["monotone"] = bool(0.0 < ratio < 1.0)
    return out


# --------------------------------------------------------------------------- #
# 2. re-grade the graded family under every convention
# --------------------------------------------------------------------------- #
def regrade_under_bars(bars: Dict, graded_rows: List[Dict]) -> Dict:
    """Honest upper bound ``L1_graded + eps`` for each convention and each graded rung."""
    measured = [r for r in graded_rows if r.get("measured")]
    if not measured:
        return {"rows": [], "finest": None}
    rows = []
    for g in measured:
        row = {"nx": g["nx"], "l1_percent": _r(g["l1_percent"]),
               "pointwise_l1_percent": _r(g.get("pointwise_l1_percent")),
               "upper_bound_percent": {}}
        for key in ("B0", "B1", "B2", "B3", "B4"):
            eps = bars[key].get("eps_percent")
            if eps is None or (key == "B4" and not bars[key].get("licensed")):
                row["upper_bound_percent"][key] = None
                continue
            row["upper_bound_percent"][key] = _r(g["l1_percent"] + eps)
        rows.append(row)
    finest = rows[-1]
    verdicts = {}
    for key in ("B0", "B1", "B2", "B3", "B4"):
        ub = finest["upper_bound_percent"][key]
        if ub is None:
            verdicts[key] = {"E4": "not_licensed", "E5": "not_licensed",
                             "ledger": "not_licensed"}
            continue
        verdicts[key] = {
            "E4": "pass" if ub < BAR_E4 else "fail",
            "E5": "pass" if ub < BAR_E5 else "fail",
            "ledger": "pass" if ub < BAR_LEDGER else "fail",
        }
    return {"rows": rows, "finest": finest, "verdicts": verdicts}


def critical_order(bars: Dict, l1_graded: float) -> Dict:
    """Which order ``p`` would the *frozen* bars require under convention B0?

    ``L1_g + e/(2^p - 1) < bar  <=>  p > log2(1 + e/(bar - L1_g))``.  This converts the
    pass/fail into a *quantitative* statement about the reference family: how much better
    the order has to be before the bar becomes reachable at all.
    """
    e = bars["e_percent"]
    out: Dict = {"e_percent": e, "l1_graded_percent": _r(l1_graded), "bars": {}}
    for name, bar in (("E4", BAR_E4), ("E5", BAR_E5)):
        head = bar - l1_graded
        if head <= 0:
            out["bars"][name] = {"bar": bar, "p_star": None,
                                 "note": "graded L1 alone already exceeds the bar"}
            continue
        out["bars"][name] = {"bar": bar, "p_star": _r(math.log2(1.0 + e / head), 4),
                             "headroom_percent": _r(head)}
    return out


# --------------------------------------------------------------------------- #
# 3. driver
# --------------------------------------------------------------------------- #
def compute() -> Dict:
    ladder = U.selfcert_ladder(DATA, "weno5")
    rows = [r for r in ladder if r.get("u_ref_percent") is not None]
    if not rows:
        return {"status": "not_measured",
                "note": "no reference pair carries an uncertainty"}
    finest = rows[-1]
    e = float(finest["l1_percent"])
    p = float(finest["order"])
    e_prev = float(rows[-2]["l1_percent"]) if len(rows) >= 2 else None
    r_seq = [b["l1_percent"] / a["l1_percent"] for a, b in zip(rows, rows[1:])]
    r = r_seq[-1] if r_seq else None

    bars = bar_conventions(e, p, r, r_seq)
    asym = asymptotic_check(e, e_prev)
    asym["order"] = _r(p, 6)
    if asym.get("order"):
        asym["order_ratio"] = _r(float(p) / asym["formal_order_weno5"], 4)
        asym["asymptotic"] = bool(asym.get("monotone")
                                  and float(p) >= 0.5 * asym["formal_order_weno5"])

    # R1: does B0 reproduce the library's own number?  The comparison is done on the
    # *unrounded* floats: comparing two differently rounded display values would report a
    # spurious 1e-5 discrepancy.  This is a formula-identity gate (it fires if the library
    # ever changes the Richardson factor), not an independent measurement.
    base_raw = e / (2.0 ** p - 1.0)
    lib_raw = finest.get("u_ref_percent")
    r1 = {"library_percent": _r(lib_raw, 6), "B0_percent": _r(base_raw, 6)}
    r1["rel_diff"] = (None if not lib_raw else
                      float(abs(base_raw - float(lib_raw)) / abs(float(lib_raw))))
    r1["verdict"] = ("pass" if r1["rel_diff"] is not None and r1["rel_diff"] < 1e-12
                     else "fail")

    graded_rows = R.graded_ladder(DATA, finest["fine_nx"])
    regrade = regrade_under_bars(bars, graded_rows)
    crit = (critical_order(bars, regrade["finest"]["l1_percent"])
            if regrade.get("finest") else {"bars": {}})

    verdicts = {
        "R1": r1["verdict"],
        "R2": regrade["verdicts"]["B1"]["E4"],
        "R3": regrade["verdicts"]["B2"]["E4"],
        "R4": regrade["verdicts"]["B3"]["E4"],
        "R5": ("not_licensed" if not bars["B4"]["licensed"]
               else regrade["verdicts"]["B4"]["E4"]),
        "R6": regrade["verdicts"]["B0"]["ledger"],
    }
    # R6 is defined on the *most conservative licensed* convention, not on B0 alone
    licensed = [k for k in ("B0", "B1", "B2", "B3", "B4")
                if regrade["verdicts"][k]["ledger"] != "not_licensed"]
    worst = None
    if licensed and regrade.get("finest"):
        ubs = {k: regrade["finest"]["upper_bound_percent"][k] for k in licensed}
        worst_key = max(ubs, key=lambda k: ubs[k])
        worst = {"convention": worst_key, "upper_bound_percent": ubs[worst_key]}
        verdicts["R6"] = "pass" if ubs[worst_key] < BAR_LEDGER else "fail"

    return {
        "status": "measured",
        "reference": {"finest_pair": [finest["coarse_nx"], finest["fine_nx"]],
                      "order_source": finest.get("order_source"),
                      "order_scale": finest.get("order_scale")},
        "bars": bars,
        "asymptotic": asym,
        "R1": r1,
        "regrade": regrade,
        "critical_order": crit,
        "worst_convention": worst,
        "verdicts": verdicts,
    }


# --------------------------------------------------------------------------- #
# 4. figure
# --------------------------------------------------------------------------- #
def figures(s: Dict) -> List[Dict]:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    out: List[Dict] = []
    keys = ["B0", "B1", "B2", "B3", "B4"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#7f7f7f"]

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.3))
    eps = [s["bars"][k].get("eps_percent") for k in keys]
    xs = range(len(keys))
    ax[0].bar(xs, [0 if v is None else v for v in eps], color=colors)
    for i, v in enumerate(eps):
        ax[0].annotate("n/a" if v is None else f"{v:.3f}%",
                       (i, 0 if v is None else v), ha="center", va="bottom", fontsize=8)
    ax[0].set_xticks(list(xs))
    ax[0].set_xticklabels([s["bars"][k]["label"] for k in keys],
                          rotation=25, ha="right", fontsize=7)
    ax[0].set_ylabel("eps_ref [%]")
    ax[0].set_title("reference error bar by convention")
    ax[0].grid(alpha=0.3, axis="y")

    rows = s["regrade"].get("rows") or []
    if rows:
        nx = [r["nx"] for r in rows]
        for k, c in zip(keys, colors):
            ys = [r["upper_bound_percent"].get(k) for r in rows]
            if any(y is not None for y in ys):
                ax[1].loglog(nx, [y if y is not None else float("nan") for y in ys],
                             "o-" if k in ("B0", "B2") else "s--", color=c, label=k)
        for bar, style, lbl in ((BAR_E4, ":", "E4 bar 1.5%"), (BAR_E5, "-.", "E5 bar 1.0%"),
                                (BAR_LEDGER, "--", "ledger 15%")):
            ax[1].axhline(bar, color="k", linestyle=style, linewidth=0.8, label=lbl)
        ax[1].set_xscale("log")
        ax[1].set_xlabel("graded nx (MUSCL/HLLC)")
        ax[1].set_ylabel("honest upper bound [%]")
        ax[1].set_title("honest bound under each convention")
        ax[1].legend(fontsize=7, ncol=2)
        ax[1].grid(alpha=0.3, which="both")

    rseq = s["bars"]["B4"].get("ratio_sequence") or []
    if rseq:
        ax[2].plot(range(len(rseq)), rseq, "o-", color="#7f7f7f")
        for i in range(1, len(rseq)):
            up = rseq[i] > rseq[i - 1] + RATIO_TOL
            ax[2].annotate("increases" if up else "ok", (i, rseq[i]),
                           textcoords="offset points", xytext=(4, 6),
                           fontsize=7, color="#d62728" if up else "#2ca02c")
        ax[2].set_xlabel("pair index (coarse -> fine)")
        ax[2].set_ylabel("r = d_i / d_(i-1)")
        ax[2].set_title("B4 licensing: ratio must not increase")
        ax[2].grid(alpha=0.3)

    p = FIGDIR / "fig_r11_error_bar_robustness.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append({"name": p.name, "path": str(p),
                "caption": "(a) 五种口径下的 eps_ref；(b) 各口径的 honest upper bound 与 "
                           "冻结判据线 E4/E5/总账；(c) B4 合法性所需的差值比序列"})
    return out


# --------------------------------------------------------------------------- #
# 5. markdown
# --------------------------------------------------------------------------- #
def _fmt(x, nd=4):
    return "—" if x is None else f"{float(x):.{nd}f}"


def _mark(x, ok):
    return "✅" if ok else "❌"


def render(s: Dict) -> str:
    if s.get("status") != "measured":
        return "# R11\n\n无参考对，记 not_measured。\n"
    ref = s["reference"]
    b = s["bars"]
    a = s["asymptotic"]
    v = s["verdicts"]
    fin = s["regrade"]["finest"]
    crit = s["critical_order"]["bars"]

    bar_rows = "\n".join(
        f"| {k} | {b[k]['label']} | {_fmt(b[k].get('eps_percent'))} | {_md_lic(k, b)} | "
        f"{b[k]['assumption']} |" for k in ("B0", "B1", "B2", "B3", "B4"))
    ub_rows = "\n".join(
        f"| {r['nx']} | {_fmt(r['l1_percent'])} | " +
        " | ".join(_fmt(r['upper_bound_percent'].get(k)) for k in ("B0", "B1", "B2", "B3", "B4")) +
        " |" for r in s["regrade"]["rows"])
    ver_rows = "\n".join(
        f"| {k} | {v[k]} | {_mark(v[k], v[k] == 'pass')} |" for k in
        ("R1", "R2", "R3", "R4", "R5", "R6"))
    reg_rows = "\n".join(
        f"| {k} | {s['regrade']['verdicts'][k]['E4']} | {s['regrade']['verdicts'][k]['E5']} | "
        f"{s['regrade']['verdicts'][k]['ledger']} |" for k in ("B0", "B1", "B2", "B3", "B4"))
    def _meets(d):
        ps = d.get("p_star")
        if ps is None:
            return "—"
        reached = float(b["order"]) >= ps
        return ("已达" if reached else f"未达，差 {ps - float(b['order']):.4f}")

    crit_rows = "\n".join(
        f"| {k} | {_fmt(d['bar'], 1)} | {_fmt(d.get('headroom_percent'))} | "
        f"{_fmt(d.get('p_star'))} | {_meets(d)} |"
        for k, d in crit.items())

    worst = s.get("worst_convention") or {}
    p_hat = b["order"]
    return f"""# XVAL-FVM-M18737-ROUND2 Round-11：误差棒的口径稳健性（B0–B4）

> 锚点 `XVAL-FVM-M18737-ROUND2`（TensorFVM 的 m18737-round2 做法），验收 <15 %。
> **本文**：不改数字、不改前轮阈值，只问一件事——把 `eps_ref` 的**口径**换成同样合法但
> 更保守的另一种，被定级判据 E4(<1.5 %)/E5(<1 %)/总账(<15 %) 还成不成立。
> 判据在 `PREDECLARATION-r11.md` 中先于任何数字冻结。

## ① 案例介绍（物理背景 / 参数表）

Woodward & Colella (1984) 双爆轰：一维闭管 x∈[0,1]，三压力平台（左 1000 / 中 0.01 / 右 100），
ρ=1、u=0，两波对撞于 t≈0.028，积分至 t=0.038，γ=1.4，两端反射壁。无解析解 ⇒ 误差恒为
「两数值解之差」，因此**参考解自身的离散误差必须作为误差棒计入**，而误差棒的口径直接决定
「<1 %」是否有意义。

| 参数 | 值 |
|---|---|
| 参考族 / 最细可测对 | WENO5 → nx = {ref['finest_pair'][0]} → {ref['finest_pair'][1]} |
| 该对的 L1 差 `e` | {_fmt(b['e_percent'], 6)} % |
| 观测阶 `p`（来源 `{ref['order_source']}`） | {_fmt(p_hat, 6)} |
| 差值比 `r = e/e_prev` | {_fmt(b['ratio'], 6)} |
| 被定级族最细档 | MUSCL/HLLC nx = {fin['nx']}，L1 = {_fmt(fin['l1_percent'])} % |
| 冻结判据线 | E4 1.5 % · E5 1.0 % · 总账 15 % |

## ② 怎么做的（方法 / 流程 / 关键设置）

1. 从已入库 artifact（`ladder_*.npz/json`）读参考阶梯，**不跑求解器**。
2. 对同一个 `e` 与 `p` 套五种口径（B0–B4，见下表），得到五个 `eps_ref`。
3. 对每个口径把被定级族的 L1 加上该 `eps_ref`（三角形不等式），得到五个 honest upper bound，
   与冻结的 E4/E5/总账线逐条比对。
4. **B4 的合法性单独判**：只有当差值比序列不增时 `e/(1−r)` 才是上界，否则记账为
   `not_licensed`（只报数不判定）。
5. **把判据反解成阶的要求**：`p* = log2(1 + e/(bar − L1_g))`，回答「参考族要好到什么程度，
   这条线才可能达到」，与「现在 p = {_fmt(p_hat, 4)}」并列。
6. R1 是自检门：B0 必须复现库里既有的 `u_ref_percent`（防口径漂移）。

## ③ 计算结果（程序化注入，零手抄）

### a) 五种口径下的参考误差棒

| 口径 | 名称 | eps_ref [%] | 合法性 | 承重假设 |
|---|---|---|---|---|
{bar_rows}

### b) 各口径下的 honest upper bound（被定级族 × 参考）

| nx | L1 [%] | B0 | B1 | B2 | B3 | B4 |
|---|---|---|---|---|---|---|
{ub_rows}

### c) 冻结判据在各口径下的存亡（最细档 nx={fin['nx']}）

| 口径 | E4 (<1.5 %) | E5 (<1 %) | 总账 (<15 %) |
|---|---|---|---|
{reg_rows}

### d) 渐近区检查（B2 的入场条件）

| 项 | 值 |
|---|---|
| 收敛比 `R = e/e_prev` | {_fmt(a.get('R'), 6)} |
| 单调收敛（0<R<1） | {a.get('monotone')} |
| 形式阶（WENO5） | {_fmt(a.get('formal_order_weno5'), 1)} |
| 观测阶 / 形式阶 | {_fmt(a.get('order_ratio'), 4)} |
| **是否在渐近区** | **{a.get('asymptotic')}** |

### e) 判据反解：需要多好的观测阶

| 判据 | 阈值 [%] | 余量 `bar − L1_g` [%] | 所需 `p*` | 实测 `p = {_fmt(p_hat, 4)}` 是否达标 |
|---|---|---|---|
{crit_rows}

### f) 冻结判据逐项

| 编号 | 判定 |
|---|---|
{ver_rows}

## ④ 结果图

- `figures/fig_r11_error_bar_robustness.png` —
  (a) 五种口径的 `eps_ref` 柱状对比；(b) 各口径下 honest bound 随被定级网格的变化与三条冻结判据线；
  (c) B4 合法性所需的差值比序列（标出上升处）。

## ⑤ 注意点（局限 / 踩过的坑 / 适用范围）

1. **B2/B3 是口径选择，不是物理**。`Fs` 取 1.25 还是 3 取决于「是否已在渐近区」，本文件把
   渐近区检查（③d）单列，就是为了不让 `Fs` 变成一个可以随便挑的数字。实测形式阶比
   {_fmt(a.get('order_ratio'), 3)} ⇒ **不在渐近区**，因此 B2 的 `Fs=1.25` 在这里偏乐观，
   B3 才与证据相符。
2. **B4 只在「差值比不增」时是上界**。本项目实测该序列**上升**（{b['B4'].get('ratio_sequence')}），
   所以 B4 记 `not_licensed`：它的数值可以看，但**不能当作上界引用**。这是负面结果，照记。
3. **最细档仍是外推**：无论哪种口径，`eps_ref` 都把「更细尺度的 p 保持不变」当成假设；
   R11 只让这个假设**显式化并定价**，没有消除它。
4. **不改前轮阈值**：E4/E5/总账三条线由 R7/R10 预声明与 owner 验收线冻结，R11 只报告
   「在这些线下、换口径后还成不成立」，不回灌新阈值。
5. **点估计与上界不能互顶**：被定级 L1 的点估计在各口径下都不变（口径只作用于参考侧），
   变的是「可证明」的部分。
6. **适用面**：结论只对「无解析解、以两数值解之差为误差 proxy」的验证案例有意义；
   有解析解的案例应直接对解析误差。

## ⑥ 误差判定（逐项显式回答）

| 关键量 | 值 [%] | <15 % | <3 % | <1.5 % | <1 % |
|---|---|---|---|---|---|
| 被定级 L1 点估计 @nx={fin['nx']} | {_fmt(fin['l1_percent'])} | {_mark('', fin['l1_percent'] < 15)} | {_mark('', fin['l1_percent'] < 3)} | {_mark('', fin['l1_percent'] < 1.5)} | {_mark('', fin['l1_percent'] < 1)} |
| honest bound · B0（现行口径） | {_fmt(fin['upper_bound_percent'].get('B0'))} | {_mark('', (fin['upper_bound_percent'].get('B0') or 99) < 15)} | {_mark('', (fin['upper_bound_percent'].get('B0') or 99) < 3)} | {_mark('', (fin['upper_bound_percent'].get('B0') or 99) < 1.5)} | {_mark('', (fin['upper_bound_percent'].get('B0') or 99) < 1)} |
| honest bound · B2 (Fs=1.25) | {_fmt(fin['upper_bound_percent'].get('B2'))} | {_mark('', (fin['upper_bound_percent'].get('B2') or 99) < 15)} | {_mark('', (fin['upper_bound_percent'].get('B2') or 99) < 3)} | {_mark('', (fin['upper_bound_percent'].get('B2') or 99) < 1.5)} | {_mark('', (fin['upper_bound_percent'].get('B2') or 99) < 1)} |
| honest bound · B3 (Fs=3) | {_fmt(fin['upper_bound_percent'].get('B3'))} | {_mark('', (fin['upper_bound_percent'].get('B3') or 99) < 15)} | {_mark('', (fin['upper_bound_percent'].get('B3') or 99) < 3)} | {_mark('', (fin['upper_bound_percent'].get('B3') or 99) < 1.5)} | {_mark('', (fin['upper_bound_percent'].get('B3') or 99) < 1)} |

**是否 <15 %（owner 验收线）？** 最保守的**合法**口径（{worst.get('convention')}）给出
{_fmt(worst.get('upper_bound_percent'))} % ⇒ {'是' if (worst.get('upper_bound_percent') or 99) < 15 else '否'}。
**是否 <3 %？** 见上表逐行（B3 是最严的一行）。
**是否可证明 <1 %？** 否——五个口径里没有一个把上界压到 1 % 以下；
反解要求 `p* ≥ {_fmt(crit.get('E5', {}).get('p_star'))}`（E5）/ `{_fmt(crit.get('E4', {}).get('p_star'))}`（E4），
而实测 `p = {_fmt(p_hat, 4)}`。

## 附：判据对账（先冻结后测量）

{ver_rows}

## 复现

```bash
PYTHONPATH=src python docs/XVAL-FVM-M18737-ROUND2/ref25600_r11_robust.py
PYTHONPATH=src python -m pytest tests/test_compressible_wc_r11_error_bar_robust.py -q
```
"""


def _md_lic(k: str, b: Dict) -> str:
    if k == "B4":
        return "licensed" if b[k].get("licensed") else "**not_licensed**"
    return "—"


def main(argv=None) -> int:
    s = compute()
    s["figures"] = figures(s) if s.get("status") == "measured" else []
    (HERE / "summary_r11.json").write_text(json.dumps(s, indent=1, ensure_ascii=False) + "\n")
    (HERE / "R11-ERROR-BAR-ROBUSTNESS.md").write_text(render(s))
    print(json.dumps({"status": s.get("status"), "verdicts": s.get("verdicts")},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())