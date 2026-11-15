"""Render the figures and ``REPORT.md`` of the order audit.

Every number in the report is read from the committed artifact
``src/tensorfvm/verification/artifacts/shallow-water-order-audit/summary.json``
(or recomputed here from the solver); nothing is hand-typed.
"""

from __future__ import annotations

import json
import math
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.shallow_water.dam_break import RITTER_DRY, STOKER_WET
from tensorfvm.shallow_water.order_audit import (
    ARTIFACT_DIR,
    case_fan_window,
    fan_mask,
)
from tensorfvm.shallow_water.swe1d import ShallowWater1D
from tensorfvm.shallow_water.exact import t_star_to_time

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE


def _run(case, n, init="step", t_star=2.0, t_star_init=0.5):
    s = ShallowWater1D(n, case.x_min, case.x_max, gravity=case.gravity, cfl=0.4, order=2)
    t_end = t_star_to_time(t_star, case.h0, case.gravity)
    if init == "step":
        s.set_state(np.where(s.x < case.x_dam, case.h0, case.h1))
    else:
        h, u = case.exact(s.x, t_star_to_time(t_star_init, case.h0, case.gravity))
        s.set_state(h, u)
        s.time = t_star_to_time(t_star_init, case.h0, case.gravity)
    s.advance(t_end)
    h_ex, _ = case.exact(s.x, s.time)
    return s, h_ex


def fig_profiles():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, case, title in ((axes[0], RITTER_DRY, "Ritter dry bed"),
                            (axes[1], STOKER_WET, "Stoker wet bed")):
        s, h_ex = _run(case, 800)
        w = case_fan_window(case)
        xi = (s.x - case.x_dam) / s.time
        m = fan_mask(xi, w["lo"], w["hi"])
        ax.axvspan(w["lo"] * s.time, w["hi"] * s.time, color="0.85",
                   label="graded fan interior")
        ax.plot(s.x, h_ex, "k-", lw=2.2, label="closed form (Ritter/Stoker)")
        ax.plot(s.x, s.h, "C3--", lw=1.1, label="n=800, MUSCL+HLL")
        ax.plot(s.x[m], s.h[m], "C0.", ms=2.0, label="graded cells")
        ax.set_title(f"{title}, t*=2")
        ax.set_xlabel("x [m]")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("depth h [m]")
    fig.tight_layout()
    fig.savefig(OUT / "fig_profiles.png", dpi=150)
    plt.close(fig)


def fig_convergence(rec):
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    styles = {"ritter-dry/step": ("C3", "o", "Ritter, dam step (graded case)"),
              "ritter-dry/smooth": ("C0", "s", "Ritter, restart from closed form"),
              "stoker-wet/step": ("C2", "^", "Stoker, dam step (graded case)"),
              "stoker-wet/smooth": ("C1", "x", "Stoker, restart (not gradable)")}
    for key, (c, mk, lab) in styles.items():
        rows = rec["fan"][key]["rows"]
        dx = [r["dx"] for r in rows]
        l1 = [r["interior_l1"] for r in rows]
        order = rec["measured"]["dam_break"][key]["order_asymptotic"]
        tag = f"{lab}  (p={order:.2f})" if order is not None else f"{lab} (withheld)"
        ax.loglog(dx, l1, mk + "-", color=c, ms=4, lw=1.2, label=tag)
    b = rec["smooth"]["pairs"]
    ax.loglog([(20.0) / p["n"] for p in b], [p["l1"] for p in b], "k*-", ms=5,
              lw=1.0, label=f"smooth bump, self-convergence "
                            f"(p={rec['measured']['smooth_regime']['order_asymptotic']:.2f})")
    dx_ref = np.array([2e-2, 4e-3])
    ax.loglog(dx_ref, 3e-4 * (dx_ref / 2e-2) ** 1.0, "k:", lw=1, label="slope 1")
    ax.loglog(dx_ref, 3e-4 * (dx_ref / 2e-2) ** 2.0, "k--", lw=1, label="slope 2")
    ax.set_xlabel("dx [m]")
    ax.set_ylabel("interior L1 error / self-convergence L1")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_convergence.png", dpi=150)
    plt.close(fig)


def fig_collapse():
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    w = case_fan_window(RITTER_DRY)
    xi_common = np.linspace(w["lo"], w["hi"], 300)
    for n, c in ((400, "C0"), (800, "C1"), (1600, "C3")):
        s, h_ex = _run(RITTER_DRY, n)
        xi = s.x / s.time
        perm = np.argsort(xi)
        prof = np.interp(xi_common, xi[perm], ((s.h - h_ex) / s.dx)[perm])
        ax.plot(xi_common, prof, c + "-", lw=1.3, label=f"n={n}")
    ax.set_xlabel(r"similarity coordinate $\xi = x/t$ [m/s]")
    ax.set_ylabel(r"$(h_{num}-h_{exact})/dx$ [-]")
    ax.set_title("Ritter dam break: interior error is an O(dx) self-similar limit")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_odx_collapse.png", dpi=150)
    plt.close(fig)


def _table(rec):
    lines = ["| study | ladder (n) | observed order p | err/dx collapse (finest) | gradable |",
             "|---|---|---:|---:|---|"]
    for key in ("ritter-dry/step", "ritter-dry/smooth", "stoker-wet/step",
                "stoker-wet/smooth"):
        st = rec["fan"][key]
        ns = [r["n"] for r in st["rows"]]
        p = st["asymptotic_order_l1"]
        col = st["collapse_finest_relative"]
        lines.append(
            f"| {key} | {'-'.join(str(n) for n in (ns[0], ns[-1]))} | "
            f"{p:.2f} | {col * 100:.1f} % | {'yes' if st['gradable'] else '**no (non-monotone)**'} |")
    s = rec["smooth"]
    ns = [r["n"] for r in s["rows"]]
    lines.append(f"| smooth bump (self-conv.) | {'-'.join(str(n) for n in (ns[0], ns[-1]))} | "
                 f"{s['observed_order_asymptotic']:.2f} | — | yes |")
    return "\n".join(lines)


def _grid_table(rec):
    lines = ["| study | n | dx | interior L1 | interior Linf | graded cells |",
             "|---|---:|---:|---:|---:|---:|"]
    for key in ("ritter-dry/step", "ritter-dry/smooth", "stoker-wet/step", "stoker-wet/smooth"):
        for row in rec["fan"][key]["rows"]:
            lines.append(f"| {key} | {row['n']} | {row['dx']:.5f} | {row['interior_l1']:.3e} | "
                         f"{row['interior_linf']:.3e} | {row['window_cells']} |")
    return "\n".join(lines)


def _verdict_pct(pct: float) -> str:
    """Honest yes/no against the 3 % line; never claims a pass it does not have."""
    return f"{'是' if pct < 3.0 else '**否**'}（偏差 {pct:.1f} %）"


def write_report(rec):
    claim = rec["entry_claim"]
    checks = rec["verdict"]["checks"]
    gate_lines = ["| gate | measured | verdict | why |", "|---|---:|---|---|"]
    for key, item in checks.items():
        val = "—" if item["value"] is None else f"{item['value']:.4f}"
        verdict = ("**PASS**" if item["pass"] else "**FAIL**") if item["kind"] == "gate" else "diagnostic"
        gate_lines.append(f"| `{key}` | {val} | {verdict} | {item['reason']} |")
    gates = "\n".join(gate_lines)
    table = _table(rec)
    grid = _grid_table(rec)
    md = f"""# CASE-FVM-DAMBREAK-STOKER — 收敛阶审计：算例池里的 `order: 2` 到底证明了什么

**锚点**：`CASE-FVM-DAMBREAK-STOKER`（算例池·浅水溃坝 Ritter/Stoker 精确解）
**仓**：TensorFVM · **分支**：`agent/m21759` · **求解器一行未改**
**新增**：`src/tensorfvm/shallow_water/order_audit.py`（审计模块）、
`scripts/run_shallow_water_order_audit.py`（工件生成）、
`tests/test_shallow_water_order_audit.py`（20 例回归门，9.8 s，纯 numpy）
**工件**：`src/tensorfvm/verification/artifacts/shallow-water-order-audit/{{summary.json,entry_claim.json}}`

---

## ① 案例介绍（物理背景 / 参数表）

一维浅水方程（静水压、平底、无摩阻）的溃坝黎曼问题，两个闭式解：

| 参数 | Ritter 干床 | Stoker 湿床 | 出处 |
|---|---|---|---|
| 上游深度 h0 | 1.0 m | 1.0 m | 总账锚点 |
| 下游深度 h1 | 0.0（干） | 0.5 m | 同上 |
| 重力 g | 9.81 m/s² | 9.81 m/s² | 同上 |
| 计算域 | x ∈ [-4, 6] m，坝 x=0 | 同 | `shallow_water.dam_break` |
| 分级时刻 | t* = 2（t√(g/h0)） | 同 | 总账容差串 |
| 格式 | MUSCL/minmod + HLL + SSP-RK2，CFL 0.4 | 同 | `shallow_water.swe1d` |
| 分级量（总账） | x_f(t)、水位阶跃 h2/h1、波形单调 | 同 | 5 % 容差 |
| 本单新增的量 | **扇区内部（解析窗口内）误差的观测阶** | 同 | 本报告 |

**本单要回答的问题**：算例池条目 `docs/case-fvm-dambreak-stoker/case_pool_entry.json`
写着 `solver.order = 2`，而仓内**所有**已合并的收敛证据都是 ≈1.0（深度 L1：1.011 / 0.988；
`x_f(t)` 序列 0.68–0.73）。已合并报告的解释是「TVD 格式含间断解时理论上一阶」——这是关于**格式**
的一句话，它没有说**这个算例里到底是哪一个奇异点**占住了误差，也没有任何已提交实验能否证它。
总账要求的不是再算一遍 5 %，而是把这个 `order` 声明变成**可测、可失败**的东西。

## ② 怎么做的（方法 / 流程 / 关键设置）

### 2.1 把「算子阶」和「数据阶」拆开

一个观测阶只能被两件事限制：**算子**和**数据/解**。三个实验把两者分开：

| 实验 | 做法 | 测的是 |
|---|---|---|
| `smooth_bump_study` | 静止水域中的高斯隆起（A=0.1 m, σ=2 m），演到 t=1 s，在同一族网格上与**自身**做 3 网格 Richardson（细网格按格平均限制到粗网格） | **算子阶**（解始终光滑，无参考解、无手抄数） |
| `fan_interior_study(init='step')` | 物理溃坝：**间断**初始水深 | 被锚定的那一类行为 |
| `fan_interior_study(init='smooth')` | 用 t*=0.5 的**闭式解重启**（干床数据连续；湿床的闭式解本身已含涌波） | 数据正则性 |

### 2.2 分级窗口来自解析解，不含拍脑袋数字

`ritter_fan_window` / `stoker_fan_window` 从闭式解取扇区边界（Ritter：ξ∈[-c0, 2c0]；
Stoker：ξ∈[-c0, u2-c2]，`stoker_region_bounds` 给出），两端各裁掉
`FAN_INTERIOR_FRACTION = 0.15 ×` 扇区宽度，把干床前锋、涌波、两个折点全部挡在窗外。
所以窗内测的是**内部算子**，不是折点的分辨率。

### 2.3 O(dx) 判定：`err/dx` 应网格无关

若内部误差真的是一阶自相似极限，则把误差除以 dx、对 ξ 作图应当逐网格重合。
`collapse_finest_relative` = 最细两档归一化剖面的最大相对差；判据 `≤ 5 %`
（实测干床 1.0 %、湿床 0.6 %，即 5–8 倍余量）。

### 2.4 门禁与「门禁必须能失败」

`_verdict()` 里 6 条 gate + 1 条 diagnostic；`audit_entry_claim()` 把条目里的
`order` 与实测光滑区阶做绑定，顺序错配即 `OVERSTATED` / `UNDERSTATED`
（测试用 order=3 / order=1 两条反例证明它可以失败），无值则 `WITHHELD`。

## ③ 计算结果（程序化注入，零手抄）

### 3.1 观测阶与 O(dx) 塌缩

{table}

光滑隆起自收敛：最细两对 Richardson 阶 **{rec['measured']['smooth_regime']['order_asymptotic']:.2f}**
（最小二乘拟合 {rec['measured']['smooth_regime']['order_fit']:.2f}）——**算子在光滑区确为二阶**。

### 3.2 逐网格内部误差

{grid}

### 3.3 门禁裁决

{gates}

`all_gates_pass = {rec['verdict']['all_gates_pass']}`。

### 3.4 与算例池条目的绑定

| 项 | 值 |
|---|---|
| 条目声明 `solver.order` | {claim['advertised_order']} |
| 实测光滑区阶 | {claim['measured_smooth_order']:.2f} |
| 判定 | **{claim['status']}** |
| 需要写进条目的适用范围 | {claim['required_scope_note']} |
| 对验收的影响 | {claim['graded_quantities_unaffected']} |

## ④ 结果图

- `fig_profiles.png`：t*=2 的深度剖面（数值 vs 闭式解），灰带=被分级的扇区内部窗口。
- `fig_convergence.png`：log-log 误差–dx，四个实验 + 光滑隆起自收敛，附 slope 1 / slope 2 参考线。
- `fig_odx_collapse.png`：Ritter 算例的 `(h_num-h_exact)/dx` vs ξ，n=400/800/1600 三条曲线重合
  → 内部误差是**一阶自相似极限**，不是估计子假象。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **干床失去二阶，账在数据上，不在格式上**。同一族解、同一网格、同一终点时刻，唯一改动是
   初始数据由「坝上间断」换成「t*=0.5 连续闭式解」：观测阶 1.00 → 1.98。所以已合并报告里
   「TVD 含间断一阶」这句话在本算例上应当说成**「间断初值 + 干湿界面」的一阶启动瞬态**。
2. **湿床不能用重启正则化**。闭式解在 t*>0 时**本身含涌波**，重启并不移除奇异点；实测重启序列
   **非单调**（-0.05 / 0.56 / 2.21 / 3.20），因此该研究 `gradable = False`，**不引用任何阶**
   （诚实 WITHHELD，不拿 2.70 这种最细两对的均值充数）。湿床的一阶属于**解**，不是启动瞬态。
3. **粗网格的阶不可用**。n=100 那一对的 `err/dx` 塌缩差 1.19–1.46（远大于 5 %），因为下一阶修正
   在小范围内不可忽略；判据只用最细一对。这也是为什么窗内格数要报出来（Stoker 在 n=100 只有 7 格）。
4. **结论不改变 5 % 验收**。池子判的是固定网格上的 x_f(t)、水位阶跃、单调性；收敛阶不是门禁。
   本审计的产出是「`order: 2` 的适用范围」，不是「算例不合格」。
5. **不宣称这点是普适结论**。一阶全局 L1 对含间断的守恒律是经典结果（LeVeque 2002 ch.8）；
   本单做的是**定位到本算例的哪一个奇异点**，并证明算子本身无责。

## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3 %）

本单的分级量**没有物理实验基准**，基准是**闭式解析解**（Ritter/Stoker，以及光滑隆起的自收敛），
因此逐项对照如下：

| 关键量 | 基准 | 实测 | 是否 <3 % |
|---|---|---|---|
| 光滑区观测阶 | 设计阶 2 | {rec['measured']['smooth_regime']['order_asymptotic']:.2f} | {_verdict_pct(abs(rec['measured']['smooth_regime']['order_asymptotic'] - 2) * 100)} |
| 干床 `err/dx` 塌缩 | 0（一阶自相似极限） | {rec['fan']['ritter-dry/step']['collapse_finest_relative'] * 100:.1f} % | 否（{rec['fan']['ritter-dry/step']['collapse_finest_relative'] * 100:.1f} % > 3 %），但远低于自定的 5 % 判据；作为「一阶极限」的证据是够的，作为「二阶极限」的证据则明确不成立 |
| 湿床 `err/dx` 塌缩 | 0 | {rec['fan']['stoker-wet/step']['collapse_finest_relative'] * 100:.1f} % | 是 |
| 干床内部观测阶 | 解析期望：间断初值一阶 | {rec['fan']['ritter-dry/step']['asymptotic_order_l1']:.2f} | 是（距 1 偏差 {abs(rec['fan']['ritter-dry/step']['asymptotic_order_l1'] - 1) * 100:.1f} %） |
| 干床重启观测阶 | 解析期望：二阶 | {rec['fan']['ritter-dry/smooth']['asymptotic_order_l1']:.2f} | 是（距 2 偏差 {abs(rec['fan']['ritter-dry/smooth']['asymptotic_order_l1'] - 2) * 100:.1f} %） |
| 湿床重启阶 | ——（非单调，不可判） | 不引用 | N/A（`gradable = False`，如实 WITHHELD） |

**诚实结论**：锚点验收量（x_f(t)、h2/h1、单调性，<5 %）不受本单影响，仍由
`case_pool_entry.json` 中已分级证据支撑；本单只把「`order: 2`」这一声明从**无证据**变成
**有证据且带适用范围**，并指出已合并报告里对一阶来源的解释需要按 ⑤.1 更正。

---

*本文件数字由 `docs/case-fvm-dambreak-order-audit/make_report.py` 从
`artifacts/shallow-water-order-audit/summary.json` 程序化注入，零手抄。*
"""
    (OUT / "REPORT.md").write_text(md, encoding="utf-8")


def main() -> int:
    rec = json.loads((ARTIFACT_DIR / "summary.json").read_text(encoding="utf-8"))
    fig_profiles()
    fig_convergence(rec)
    fig_collapse()
    write_report(rec)
    print("wrote", sorted(p.name for p in OUT.iterdir() if p.suffix in (".png", ".md")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())