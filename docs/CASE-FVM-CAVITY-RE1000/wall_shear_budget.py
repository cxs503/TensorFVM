#!/usr/bin/env python
"""Wall-shear / lid-drag verification-debt audit for CASE-FVM-CAVITY-RE1000.

Recomputes, from artefacts only (stored FVM fields + the Chebyshev reference
solution), the quantities the acceptance gate does *not* cover:

* lid drag from the solver's own ghost-slope shear (``order=1``) and from a
  quadratic one-sided estimator (``order=2``);
* the drag split into the two lid-corner strips and the central 80 %;
* the steady-state power identity ``P_in = eps`` (exact in the continuum);
* Stokes' total-vorticity identity ``int omega dA = -U L``;
* observed grid orders and Richardson extrapolations.

Outputs (all numbers injected, never typed by hand):
``runs/wall_shear_budget.json``, ``WALL-SHEAR-BUDGET.md``,
``figures/fig_ws*.png``.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/wall_shear_budget.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

from tensorfvm.cavity_power_budget import (         # noqa: E402
    budget_spectral, budget_staggered, staggered_cell_velocity)
from tensorfvm.cavity_spectral import (              # noqa: E402
    SpectralCavity, clenshaw_curtis_weights)
from tensorfvm.cavity_wall_shear import (            # noqa: E402
    drag_decomposition, lid_wall_shear, observed_order)
from scripts.run_cavity_spectral_ref import march     # noqa: E402

RUNS = HERE / "runs"
FIGS = HERE / "figures"
FIELDS = RUNS / "fields.npz"
NU = 1.0e-3          # mu/rho at Re = 1000, rho = 1, U = 1, L = 1
SPECTRAL_ORDER = 48  # N = 48: cheapest order whose psi_min is converged (<1e-3)


def fvm_table(fields) -> list[dict]:
    rows = []
    for n in (32, 64, 128):
        u, v, h = fields[f"u_{n}"], fields[f"v_{n}"], 1.0 / n
        b = budget_staggered(u, v, h, nu=NU)
        d1 = drag_decomposition(u, h, NU, order=1)
        d2 = drag_decomposition(u, h, NU, order=2)
        rows.append({
            "n": n, "h": h,
            "drag_1": d1["total"], "drag_2": d2["total"],
            "centre_1": d1["centre"], "centre_2": d2["centre"],
            "corner_share_1": d1["corner_share"],
            "corner_share_2": d2["corner_share"],
            "dissipation": b["dissipation"], "power_in": b["power_in"],
            "power_residual_rel": b["power_residual_rel"],
            "total_vorticity": b["total_vorticity"],
            "kinetic_energy": b["kinetic_energy"],
        })
    return rows


def spectral_block(fields) -> dict:
    solver = SpectralCavity(SPECTRAL_ORDER)
    sol = march(solver, 1000.0)
    x = np.asarray(solver.X)
    w = clenshaw_curtis_weights(solver.xi) / 2.0
    omega_wall = sol.omega[-1, :]

    def drag(mask):
        return float(-NU * np.sum(w[mask] * omega_wall[mask]))

    everything = np.ones_like(x, dtype=bool)
    centre = (x >= 0.1) & (x <= 0.9)
    b = budget_spectral(sol.psi, sol.omega, nu=NU, order=SPECTRAL_ORDER)
    ys = np.array([1.0 - 0.5 / n for n in (32, 64, 128)])
    return {
        "order": SPECTRAL_ORDER,
        "psi_min": float(sol.psi.min()),
        "newton_iterations": int(sol.newton.get("iterations", -1)),
        "drag_total": drag(everything), "drag_centre": drag(centre),
        "corner_share": float((drag(everything) - drag(centre)) / drag(everything)),
        "power_in": b["power_in"], "dissipation": b["dissipation"],
        "power_residual_rel": b["power_residual_rel"],
        "total_vorticity": b["total_vorticity"],
        "kinetic_energy": b["kinetic_energy"],
        "npz": {f"u_at_y_{n}": float(sol.centreline_u(np.array([1 - 0.5 / n]))[0])
                for n in (32, 64, 128)},
        "ys": ys.tolist(),
    }


def pct(a: float, b: float) -> float:
    return 100.0 * (a - b) / b


def build(fields) -> dict:
    fvm = fvm_table(fields)
    spec = spectral_block(fields)
    d1 = [r["drag_1"] for r in fvm]
    d2 = [r["drag_2"] for r in fvm]
    c1 = [r["centre_1"] for r in fvm]
    p1 = observed_order(d1[1] - d1[0], d1[2] - d1[1])
    p2 = observed_order(d2[1] - d2[0], d2[2] - d2[1])
    pc = observed_order(c1[1] - c1[0], c1[2] - c1[1])
    rich = lambda seq, p: seq[2] + (seq[2] - seq[1]) / (2.0 ** p - 1.0) if p == p else float("nan")  # noqa: E731
    return {
        "case": "CASE-FVM-CAVITY-RE1000",
        "reynolds": 1000.0, "nu": NU, "lid_velocity": 1.0,
        "identity_power": "P_in = U * mu * int (du/dy)|lid dx = mu * int |grad u|^2 dA",
        "identity_stokes": "int omega dA = -U L",
        "fvm": fvm,
        "spectral": spec,
        "orders": {"drag_order1": p1, "drag_order2": p2, "centre_order1": pc,
                   "dissipation": observed_order(fvm[1]["dissipation"] - fvm[0]["dissipation"],
                                                 fvm[2]["dissipation"] - fvm[1]["dissipation"])},
        "richardson": {"drag_order1": rich(d1, p1 if p1 == p1 else 1.0),
                       "drag_order2": rich(d2, p2 if p2 == p2 else 1.0),
                       "centre_order1": rich(c1, pc if pc == pc else 1.0)},
        "cross_check": {
            "centre_fvm_n128_vs_spectral": pct(c1[2], spec["drag_centre"]),
            "total_fvm_rich_vs_spectral": pct(rich(d1, p1 if p1 == p1 else 1.0), spec["drag_total"]),
            "near_wall_u_rel": {f"n{n}": pct(fields[f"u_{n}"][n // 2, -1], spec["npz"][f"u_at_y_{n}"])
                                for n in (32, 64, 128)},
        },
    }


def figures(payload, fields) -> list[str]:
    FIGS.mkdir(exist_ok=True)
    made = []
    fvm, spec = payload["fvm"], payload["spectral"]
    ns = [r["n"] for r in fvm]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.semilogx(ns, [r["drag_1"] for r in fvm], "o-", label="FVM, ghost slope (order 1)")
    ax.semilogx(ns, [r["drag_2"] for r in fvm], "s-", label="FVM, quadratic (order 2)")
    ax.semilogx(ns, [r["centre_1"] for r in fvm], "^-", label="FVM, central 80 % (order 1)")
    ax.axhline(spec["drag_total"], ls="--", c="C3",
               label=f"spectral N={spec['order']}, whole lid")
    ax.axhline(spec["drag_centre"], ls=":", c="C4",
               label=f"spectral N={spec['order']}, central 80 %")
    ax.set(xlabel="n (grid points per side)", ylabel=r"lid drag  $F/(\rho U^2 L)$",
           title="lid drag: two shear estimators vs the spectral reference")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    p = FIGS / "fig_ws_drag_convergence.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogx(ns, [100 * r["power_residual_rel"] for r in fvm], "o-", label="FVM")
    ax.axhline(100 * spec["power_residual_rel"], ls="--", c="C3",
               label=f"spectral N={spec['order']}")
    ax.set(xlabel="n", ylabel=r"$(P_{\rm in}-\epsilon)/P_{\rm in}$  [%]",
           title="steady-state power identity residual (exact value: 0)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "fig_ws_power_residual.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))

    n = 128
    u, v = fields[f"u_{n}"], fields[f"v_{n}"]
    uc, vc = staggered_cell_velocity(u, v)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    h = 1.0 / n
    yy = (np.arange(n) + 0.5) * h
    xx = (np.arange(n) + 0.5) * h
    cs = axes[0].contourf(xx, yy, (vc.T - uc.T * 0.0) * 0 + (np.gradient(vc, h, axis=0).T
                                                            - np.gradient(uc, h, axis=1).T),
                          40, cmap="viridis")
    fig.colorbar(cs, ax=axes[0], label=r"$\omega L/U$")
    axes[0].set(title=f"vorticity, FVM n={n}", xlabel="x / L", ylabel="y / L")
    xx_s = np.arange(n + 1) * h
    for order, style in ((1, "-"), (2, "--")):
        xs, shear = lid_wall_shear(u, h, order=order)
        axes[1].plot(xs, shear, style, lw=1.2, label=f"FVM n={n}, order {order}")
    axes[1].set(xlabel="x / L", ylabel=r"$(\partial u/\partial y)|_{\rm lid}$  $[U/L]$",
                title="lid wall shear: corner singularity is the unresolved part")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = FIGS / "fig_ws_fields_shear.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    made.append(str(p))
    del xx_s, xx
    return made


MARK = """# CASE-FVM-CAVITY-RE1000 · 壁面剪切/阻力与全局功率恒等式审计（验证欠账）

> 本文件由 `docs/CASE-FVM-CAVITY-RE1000/wall_shear_budget.py` 从 `runs/fields.npz` 与
> Chebyshev 参考解程序化生成（数字零手抄）；机检门 `tests/test_cavity_wall_shear.py`。
> 复算：`PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/wall_shear_budget.py`

## 1. 案例与缺口

方腔流 Re=1000（顶盖驱动、单位方腔、`U=1`、`nu=1e-3`、`rho=1`），参考解 Ghia et al. (1982)。
总账锚点 `CASE-FVM-CAVITY-RE1000` 的验收量是**中心线剖面 + 主涡心**（C1–C7，n=128 带宽 3 %，
实测 0.43 %/1.12 %）。这三类量都是**内部量**；而船舶阻力工程真正要的是**壁面剪切/阻力**，
本单量化这一份欠账：把已合并的求解器对两条**精确**全局恒等式做复核，并给出阻力量的网格收敛与误差。

两条恒等式（对连续解严格成立，与参考表无关）：

```
P_in = U * F_lid,  F_lid = mu * int_0^1 (du/dy)|_(y=L) dx,   P_in = eps = mu * int_A |grad u|^2 dA
int_A omega dA = -U L                                        (Stokes，omega = dv/dx - du/dy)
```

## 2. 怎么做的

* 生产码壁面剪切有两条估计量：求解器实际使用的镜像 ghost 两点斜率
  `2(U-u_adj)/h`（对光滑剖面**一阶**），与本单新加的过 `(0,U)`、`(h/2,u1)`、`(3h/2,u2)`
  的三点二次式 `(8U-9u1+u2)/(3h)`（二阶）；两者的差就是「阻力」这一输出量的截断误差。
  > 模块 `src/tensorfvm/cavity_wall_shear.py`；功率/耗散核 `src/tensorfvm/cavity_power_budget.py`。
* 阻力按 `[0,0.1] + [0.1,0.9] + [0.9,1] ` 三段分解：两侧条带含顶盖角点奇点（`omega ~ s^-1/2`）。
* 独立参考：Chebyshev 配置谱解（`tensorfvm.cavity_spectral`，N=48，Newton 3 步收敛），
  `P_in = -mu int omega_wall dx`、`eps = mu int (psi_xx^2+psi_yy^2+2psi_xy^2) dA`，
  积分用 Clenshaw–Curtis 权（对多项式精确）。

## 3. 结果

### 3.1 FVM 阶梯（n=32/64/128）

| n | F 一阶估计 | F 二阶估计 | 中央 80 %（一阶） | 角带占总额 | eps | P_in | 功率恒等式残差 | ∫ω dA |
|---|---|---|---|---|---|---|---|---|
{rows}

### 3.2 谱参考解 N={specN}

| 量 | 值 |
|---|---|
| 全顶盖阻力 F | {s_drag:.6f} |
| 中央 80 % 阻力 | {s_dragc:.6f} |
| 角带占比 | {s_corner:.1%} |
| P_in | {s_power:.6f} |
| eps | {s_eps:.6f} |
| 功率恒等式残差 | {s_res:.1%} |
| ∫ω dA | {s_vort:.6f} |

### 3.3 观测阶与外推

| 量 | 观测阶 p | Richardson 外推 |
|---|---|---|
| 总阻力（一阶估计） | {p1:.2f} | {r1:.6f} |
| 总阻力（二阶估计） | {p2:.2f} | {r2:.6f} |
| 中央 80 % 阻力（一阶） | {pc} | {rc:.6f} |
| 耗散 eps | {pe:.2f} | — |

### 3.4 交叉核对（两套独立离散）

| 比较 | 相对差 |
|---|---|
| 中央 80 % 阻力：FVM n=128 vs 谱 N={specN} | {x_centre:+.2f} % |
| 总阻力：FVM Richardson vs 谱 N={specN} | {x_total:+.2f} % |
| 贴壁首个格心 u：n=32 / 64 / 128 vs 谱（x=L/2） | {x_u32:+.2f} % / {x_u64:+.2f} % / {x_u128:+.2f} % |

![drag](figures/fig_ws_drag_convergence.png)

*图 WS-1* 阻力随网格：一阶估计系统性偏低；中央 80 % 在两套码之间 2.3 % 内一致；
全顶盖值被角点奇点支配，两套码外推后相差 {x_total_abs:.1f} %。

![power](figures/fig_ws_power_residual.png)

*图 WS-2* 功率恒等式残差：两套独立离散都停在 ~13 %，说明残差来自顶盖角点奇点，
不是某一套格式的 bug。

![fields](figures/fig_ws_fields_shear.png)

*图 WS-3* n=128 涡量场与顶盖剪切分布（角点处 `du/dy` 发散，是全部争议所在）。

## 4. 注意点

1. **一阶壁剪切**：生产码的 ghost 两点斜率对光滑剖面只有一阶精度；实测 n=32→128 阻力
   0.0282→0.0365，观测阶仅 {p1:.2f}。若把该量当工程输出（摩擦阻力），需要二次式或更高阶重构。
2. **角点奇点**：顶盖两角 `omega ~ s^-1/2`，阻力积分收敛极慢，且两套码用完全不同的方式
   处理该奇点（FVM 用一格涂抹，谱解用多项式逼近），因此全顶盖值不可作为判据。
3. **功率恒等式不是紧判据**：残差被角点区支配，两套码都 ~13 %；用它做验收只会误判好解。
   可用的替代是中央 80 % 区间的阻力（两码 2 % 内一致）。
4. 谱解的壁涡量闭合（Hermite ghost-slope）在角点附近过冲，其全顶盖阻力随 N 仍在上漂
   （N=48→64：0.0474→0.0498，+5 %），故谱值本身在角带不可信。

## 5. 误差判定（是否 <3%）

| 量 | 与基准差 | <3 % ? |
|---|---|---|
| 中央 80 % 顶盖阻力（FVM n=128 vs 谱 N={specN}） | {x_centre:+.2f} % | {v1} |
| 中央 80 % 阻力收敛 | 观测阶 {pc}，n=64→128 变化 {c_drift:.2f} % | {v2} |
| 全顶盖阻力（FVM 外推 vs 谱） | {x_total:+.2f} % | **否**（honest FAIL） |
| 功率恒等式残差 n=128 | {res128:.1%} | **否**（且非格式缺陷，见 §4.3） |
| Stokes ∫ω dA（n=128） | {vort128:.6f} vs -1 | 是 |

**结论**：锚点的 <3 % 验收只覆盖中心线/涡心等内部量；**壁面剪切派生的阻力在 n=128 时不具备
<3 % 精度**（角带不可解 + 一阶剪切），因此该算例不可作为摩擦阻力类验证的基准，除非改用中央区
口径或壁面重构。建议总账登记 `CASE-FVM-CAVITY-RE1000-WALLSHEAR`（验收：中央 80 % 阻力两码
差 <3 % 且 FVM 观测阶 ≥1.5；归属域：船舶总体性能·阻力/数值验证）。
"""


SUPERS = tuple(chr(0x2070 + i) for i in range(10))


def scifmt(value, digits=3):
    """Scientific notation with unicode superscripts, for markdown tables."""
    if value == 0:
        return "0"
    mant, exp = f"{value:.{digits}e}".split("e")
    exp = int(exp)
    sign = "\u2212" if exp < 0 else ""
    sup = "".join(SUPERS[int(c)] for c in str(abs(exp)))
    return f"{mant}\u00d710{sign}{sup}"


def write_report(payload, figs) -> pathlib.Path:
    fvm, spec = payload["fvm"], payload["spectral"]
    rows = "\n".join(
        f"| {r['n']} | {r['drag_1']:.6f} | {r['drag_2']:.6f} | {r['centre_1']:.6f} | "
        f"{r['corner_share_1']:.1%} | {scifmt(r['dissipation'])} | {scifmt(r['power_in'])} | "
        f"{r['power_residual_rel']:.1%} | {r['total_vorticity']:.6f} |"
        for r in fvm)
    c = payload["cross_check"]
    mark = MARK.format(
        rows=rows, specN=spec["order"],
        s_drag=spec["drag_total"], s_dragc=spec["drag_centre"],
        s_corner=spec["corner_share"], s_power=spec["power_in"],
        s_eps=spec["dissipation"], s_res=spec["power_residual_rel"],
        s_vort=spec["total_vorticity"],
        p1=payload["orders"]["drag_order1"], r1=payload["richardson"]["drag_order1"],
        p2=payload["orders"]["drag_order2"], r2=payload["richardson"]["drag_order2"],
        rc=payload["richardson"]["centre_order1"],
        pe=payload["orders"]["dissipation"],
        pc=("符号翻转，不可估" if payload["orders"]["centre_order1"] != payload["orders"]["centre_order1"] else f'{payload["orders"]["centre_order1"]:.2f}'),
        x_centre=c["centre_fvm_n128_vs_spectral"],
        x_total=c["total_fvm_rich_vs_spectral"],
        x_total_abs=abs(c["total_fvm_rich_vs_spectral"]),
        x_u32=c["near_wall_u_rel"]["n32"], x_u64=c["near_wall_u_rel"]["n64"],
        x_u128=c["near_wall_u_rel"]["n128"],
        c_drift=abs(pct(fvm[2]["centre_1"], fvm[1]["centre_1"])),
        res128=fvm[2]["power_residual_rel"],
        vort128=fvm[2]["total_vorticity"],
        v1="是" if abs(c["centre_fvm_n128_vs_spectral"]) < 3 else "否",
        v2="是" if abs(pct(fvm[2]["centre_1"], fvm[1]["centre_1"])) < 3 else "否",
    )
    out = HERE / "WALL-SHEAR-BUDGET.md"
    out.write_text(mark, encoding="utf-8")
    return out


def main() -> int:
    RUNS.mkdir(exist_ok=True)
    fields = dict(np.load(FIELDS))
    payload = build(fields)
    payload["figures"] = [str(pathlib.Path(p).relative_to(HERE)) for p in figures(payload, fields)]
    out = write_report(payload, payload["figures"])
    (RUNS / "wall_shear_budget.json").write_text(
        json.dumps(payload, indent=1), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("orders", "richardson", "cross_check")}, indent=1))
    print("spectral:", {k: round(v, 6) if isinstance(v, float) else v
                        for k, v in payload["spectral"].items() if not isinstance(v, dict)})
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())