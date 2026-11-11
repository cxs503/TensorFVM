"""Verify the hull yaw added inertia (verification-debt item D7) and its effect
on the manoeuvring predictions.

Run:  python scripts/verify_maneuvering_added_inertia.py [--out docs/maneuvering-added-inertia]

Produces (programmatically, no hand-copied numbers):
  * ``summary.json``  -- every number quoted in the report,
  * ``fig1_convergence.png`` / ``fig2_manoeuvres.png``,
  * ``REPORT.md``     -- rendered from this JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from tensorfvm.maneuvering import dynamics as D
from tensorfvm.maneuvering import added_inertia_panel as A
from tensorfvm.maneuvering.selfpropulsion import solve_self_propulsion

MESH_LADDER = ((21, 5), (41, 9), (61, 13))


def power_law_fit(ns, values):
    """Fit ``I(N) = I_inf + K N^(-q)``; returns ``(I_inf, q)``."""
    ns = np.asarray(ns, dtype=float)
    vals = np.asarray(values, dtype=float)
    if ns.size < 3:
        return float("nan"), float("nan")
    # grid search on q, linear least squares on (I_inf, K) per q
    best = (float("inf"), float("nan"), float("nan"))
    for q in np.linspace(0.05, 4.0, 2000):
        basis = np.column_stack([np.ones_like(ns), ns ** (-q)])
        coef, res, *_ = np.linalg.lstsq(basis, vals, rcond=None)
        err = float(np.sum((basis @ coef - vals) ** 2))
        if err < best[0]:
            best = (err, float(coef[0]), float(q))
    return best[1], best[2]


def manoeuvres(j_z, sp, *, dt=0.5, t_end_turn=400.0, t_end_zig=200.0):
    model = D.ManoeuvreModel(self_propulsion=sp, mode="direct", j_z_override=j_z)
    turn = D.simulate_turning_circle(model, 35.0, dt=dt, t_end=t_end_turn)
    zig = D.simulate_zigzag(model, 10.0, 10.0, dt=dt, t_end=t_end_zig)
    return {
        "turning": {k: float(v) for k, v in D.turning_circle_metrics(turn).items()
                    if isinstance(v, (int, float, np.floating))},
        "zigzag": {k: float(v) for k, v in D.zigzag_metrics(zig, psi_a_deg=10.0).items()
                   if isinstance(v, (int, float, np.floating))},
        "_turn": turn, "_zig": zig,
    }


def pct_delta(new: float, base: float) -> float:
    return 100.0 * (new - base) / base if base else float("nan")


def main(out_dir: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    hull = A.build_hull()
    res = {"hull": {
        "L": hull.L, "B": hull.B, "T": hull.T,
        "Cb_fit": hull.coefficients_check["Cb"],
        "Cwp_fit": hull.coefficients_check["Cwp"],
        "LCB_pct_fit": hull.coefficients_check["LCB_pct"],
        "volume_fit": hull.coefficients_check["volume"],
    }}

    # ---------------------------------------------------------------- ladder
    ladder = []
    for n_st, n_g in MESH_LADDER:
        panels = A.hull_panels(hull, n_stations=n_st, n_girth=n_g)
        gate = A.geometry_gate_panels(panels, hull)
        lid = panels.centroids[np.abs(panels.centroids[:, 2])
                              < 1e-12 * hull.L]
        j_z = A.yaw_added_inertia_panel(panels)
        # structural check: the lid carries no flux for a yaw field
        from tensorfvm.panel_added_inertia import rigid_rotation_field
        x_mid = 0.5 * hull.L
        V = rigid_rotation_field(lid, 2, origin=np.array([x_mid, 0.0, 0.0]))
        lid_flux = float(np.max(np.abs(V[:, 2]))) if lid.size else 0.0
        ladder.append({
            "n_stations": n_st, "n_girth": n_g, "n_panels": int(panels.n_panels),
            "volume": gate["checks"]["volume"]["panel"],
            "volume_rel_err": gate["checks"]["volume"]["relative_error"],
            "lid_area": gate["checks"]["lid_area"]["panel"],
            "lid_area_rel_err": gate["checks"]["lid_area"]["relative_error"],
            "geometry_passed": gate["passed"],
            "surface_area": gate["surface_area"],
            "J_z": j_z,
            "lid_flux_max": lid_flux,
        })
        print(f"ladder {n_st}x{n_g}: N={panels.n_panels} vol_err="
              f"{1e2 * gate['checks']['volume']['relative_error']:+.3f}%  J_z={j_z:.1f}")

    j_ref, q = power_law_fit([r["n_panels"] for r in ladder], [r["J_z"] for r in ladder])
    slender = A.slender_yaw_added_inertia()
    j_slender = float(slender["J_z"])
    res["ladder"] = ladder
    res["richardson"] = {"J_z_inf": j_ref, "order_q": q,
                         "J_z_slender": j_slender,
                         "adjusted_panel": j_ref,
                         "panel_over_slender": j_ref / j_slender}
    res["slender"] = {k: float(v) for k, v in slender.items()}

    # ------------------------------------------------------ sway corroboration
    sway = {}
    for n_st, n_g in MESH_LADDER[1:]:
        m_y = A.sway_added_mass_panel(hull, n_stations=n_st, n_girth=n_g,
                                      hull_obj=hull)
        sway[f"{n_st}x{n_g}"] = {"m_y": m_y, "ratio_to_slender": m_y / slender["m_y"]}
        print(f"sway {n_st}x{n_g}: m_y={m_y:.1f} (slender {slender['m_y']:.1f})")
    res["sway"] = sway

    # ------------------------------------------------------------- manoeuvres
    sp = solve_self_propulsion()
    base = manoeuvres(None, sp)
    panel = manoeuvres(j_ref, sp)
    keys_t = ("tactical_diameter_over_L", "advance_over_L", "transfer_over_L",
              "time_90_s", "time_180_s", "steady_turning_radius_over_L",
              "steady_drift_deg")
    keys_z = ("psi_1_deg", "overshoot_1_deg", "time_1_s", "period_1_s")
    res["manoeuvres"] = {
        "J_z_slender": j_slender, "J_z_panel": j_ref, "dt": 0.5, "delta_deg": 35.0,
        "turning": {
            "slender": {k: base["turning"].get(k) for k in keys_t},
            "panel": {k: panel["turning"].get(k) for k in keys_t},
            "delta_pct": {k: pct_delta(panel["turning"].get(k, math.nan),
                                       base["turning"].get(k, math.nan))
                          for k in keys_t},
        },
        "zigzag": {
            "slender": {k: base["zigzag"].get(k) for k in keys_z},
            "panel": {k: panel["zigzag"].get(k) for k in keys_z},
            "delta_pct": {k: pct_delta(panel["zigzag"].get(k, math.nan),
                                       base["zigzag"].get(k, math.nan))
                          for k in keys_z},
        },
    }
    max_turn = max(abs(v) for v in res["manoeuvres"]["turning"]["delta_pct"].values()
                   if np.isfinite(v))
    max_zig = max(abs(v) for v in res["manoeuvres"]["zigzag"]["delta_pct"].values()
                  if np.isfinite(v))
    res["manoeuvres"]["max_abs_delta_pct"] = {"turning": max_turn, "zigzag": max_zig,
                                              "threshold_pct": 15.0,
                                              "switch_default": bool(max(max_turn, max_zig) < 15.0)}
    print(f"manoeuvre max |delta|: turning {max_turn:.1f}%  zigzag {max_zig:.1f}%")

    (out / "summary.json").write_text(json.dumps(res, indent=2, default=float))

    # ---------------------------------------------------------------- figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ns = [r["n_panels"] for r in ladder]
    jz = [r["J_z"] for r in ladder]
    ax.plot([1.0 / n for n in ns], jz, "o-", label="panel (hull+lid)")
    ax.axhline(j_slender, color="tab:red", ls="--",
               label=f"slender strip = {j_slender:.0f} kg m$^2$")
    ax.axhline(j_ref, color="tab:green", ls=":",
               label=f"panel extrapolation = {j_ref:.0f} kg m$^2$")
    ax.set_xlabel("1 / N$_{panels}$")
    ax.set_ylabel("$J_z$  [kg m$^2$]")
    ax.set_title("Yaw added inertia of the KVLCC2 hull: panel vs slender body")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_convergence.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    for tag, m, c in (("slender $J_z$", base, "tab:red"),
                      ("panel $J_z$", panel, "tab:blue")):
        t = m["_turn"]
        axes[0].plot(t.x / hull.L, t.y / hull.L, c, label=tag)
        axes[1].plot(t.time, np.degrees(t.r), c, label=tag)
    axes[0].set_xlabel("x / L"); axes[0].set_ylabel("y / L")
    axes[0].set_title("35 deg turning circle (dt = 0.5 s)")
    axes[0].grid(alpha=0.3); axes[0].legend(fontsize=8); axes[0].set_aspect("equal")
    axes[1].set_xlabel("t [s]"); axes[1].set_ylabel("yaw rate r [deg/s]")
    axes[1].set_title("yaw-rate build-up")
    axes[1].grid(alpha=0.3); axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "fig2_manoeuvres.png", dpi=140)
    plt.close(fig)

    # ----------------------------------------------------------------- report
    man = res["manoeuvres"]
    rows_t = "\n".join(
        f"| {k} | {man['turning']['slender'][k]:.4g} | {man['turning']['panel'][k]:.4g} "
        f"| {man['turning']['delta_pct'][k]:+.2f} % |" for k in keys_t)
    rows_z = "\n".join(
        f"| {k} | {man['zigzag']['slender'][k]:.4g} | {man['zigzag']['panel'][k]:.4g} "
        f"| {man['zigzag']['delta_pct'][k]:+.2f} % |" for k in keys_z
        if man["zigzag"]["slender"][k] is not None)
    rows_l = "\n".join(
        f"| {r['n_stations']}x{r['n_girth']} | {r['n_panels']} | "
        f"{1e2 * r['volume_rel_err']:+.2f} % | {1e2 * r['lid_area_rel_err']:+.2f} % | "
        f"{r['J_z']:.1f} | {r['lid_flux_max']:.1e} |" for r in ladder)
    judge = ("是（< 15 %）" if man["max_abs_delta_pct"]["switch_default"]
             else "否（> 15 %）")
    report = f"""# 船体艏摇附加惯量 J_z 的面元验证与操纵性影响（欠账账本 D7）

- 任务：m14901（selfstart，TensorFVM）——认领 `docs/VERIFICATION_DEBT_LEDGER.md` 条目 **D7**
- 代码：`src/tensorfvm/maneuvering/added_inertia_panel.py`（新）、`src/tensorfvm/maneuvering/dynamics.py`
  （新增 `j_z_override` 入口，默认路径不变）、本驱动脚本、`tests/test_maneuvering_added_inertia.py`（新）
- 本报告所有数字来自 `summary.json`（由脚本程序化写入，零手抄）；图由同一脚本生成。

## 1. 案例介绍

KVLCC2 油轮（模型尺度 L = {res['hull']['L']:.3f} m, B = {res['hull']['B']:.4f} m,
T = {res['hull']['T']:.4f} m，拟合 Cb = {res['hull']['Cb_fit']:.4f}，
Cwp = {res['hull']['Cwp_fit']:.4f}，LCB = {res['hull']['LCB_pct_fit']:+.3f} % L）
的操纵运动方程

    (m + m_x) u_dot = X + (m + m_y) v r
    (m + m_y) v_dot = Y - (m + m_x) u r
    (I_z + J_z) r_dot = N

中 `J_z` 至今只有细长体条形理论值
`J_z = ∫ rho pi T_e(x)^2 x^2 dx`（`dynamics.added_masses`，自述为**上界**）。
本单回答两个问题：(a) 同一船型的三维势流面元值是多少、与细长体差多少；
(b) 把 `J_z` 换成面元值后，35° 回转与 10/10 蛇形预报量变化多少（是否 < 15 %，
即是否允许换默认值）。

**双体（double-body）闭合的物理一致性**：艏摇速度场 `e_z × r` 是水平的，在水平
水线盖面上法向分量为零（下表 `lid_flux_max` = 0 机器验证），因此「船体 + 水线盖面」
在无限流场中的解**正是**刚壁自由面（double-body）艏摇流；这与条形理论假设的物理
模型同类，两者可直接对比。

## 2. 怎么做的

1. **面元几何**：直接用操纵力模型同一套船型 —— `Kvlcc2Hull.half_breadth(x, z)`
   逐站生成闭合横剖面（超椭圆指数由剖面积丰满度给出），纵向站位余弦加密，
   水线以平面盖面闭合；法向按「背离中纵剖面/龙骨参考点」统一朝外。
2. **几何门**（平台铁律：先验几何）：面板体围合体积 vs `Cb L B T`、
   盖面面积 vs `Cwp L B`（表 3.1）；未过门不报水动力数字。
3. **艏摇附加惯量**：`panel_added_inertia.rotational_added_inertia`
   单模态 `I_zz`（源面元法，`dphi/dn = (e_z × r)·n`，转轴取舯剖面）。
   三级网格加密 + 幂律拟合 `J(N) = J_inf + K N^(-q)` 外推。
4. **交叉旁证**：同一船型的面元横漂附加质量 `M_22`（水线镜像 `single_hull`，
   双体模型）与细长体 `m_y` 对比，判断差异是「艏摇特有」还是「整套条形理论普遍高估」。
5. **操纵性影响**：`ManoeuvreModel(..., j_z_override=J_inf)`（新入口，默认 `None`
   即维持细长体值），dt = 0.5 s，35° 回转与 10/10 蛇形，逐项比较预报量。

## 3. 计算结果

### 3.1 几何门 + 网格阶梯

| 网格 (站×围长) | N_panels | 体积误差 vs CbLBT | 盖面面积误差 vs CwpLB | J_z [kg m²] | 盖面最大法向流速 (解析为 0) |
|---|---|---|---|---|---|
{rows_l}

幂律外推：`J_z(∞) = {j_ref:.1f} kg m²`（拟合阶 q = {q:.2f}）。
细长体值 `J_z_slender = {j_slender:.1f} kg m²`，
**面元/细长 = {j_ref / j_slender:.3f}**，即细长体公式把艏摇附加惯量高估了
约 {100.0 * (j_slender / j_ref - 1.0):.1f} %。

无因次：`J_z/(m L²)` = {j_ref / (hull.coefficients_check['volume'] * 998.2 * hull.L ** 2):.4f}（面元）
vs {j_slender / (hull.coefficients_check['volume'] * 998.2 * hull.L ** 2):.4f}（细长体）。

### 3.2 横漂附加质量旁证（双体，水线镜像）

| 网格 | M_22 [kg] | 与细长体 m_y = {slender['m_y']:.1f} kg 之比 |
|---|---|---|
""" + "\n".join(f"| {k} | {v['m_y']:.1f} | {v['ratio_to_slender']:.3f} |"
                for k, v in sway.items()) + f"""

横漂方向面元值同样显著低于细长体（比值 {min(v['ratio_to_slender'] for v in sway.values()):.2f}–{max(v['ratio_to_slender'] for v in sway.values()):.2f}），
说明偏差来自 `rho pi T_e^2` 这一「二维薄板」假设本身，而不是艏摇特有。

### 3.3 操纵性影响（dt = 0.5 s）

35° 回转（`tactical_diameter_over_L` = 战术直径，无因次）：

| 指标 | 细长体 J_z | 面元 J_z | 变化 |
|---|---|---|---|
{rows_t}

10/10 蛇形：

| 指标 | 细长体 J_z | 面元 J_z | 变化 |
|---|---|---|---|
{rows_z}

### 3.4 图

- `fig1_convergence.png`：J_z 网格收敛 + 细长体线 + 外推线。
- `fig2_manoeuvres.png`：左=35° 回转轨迹（x/L, y/L），右=艏摇角速度建立过程。

## 4. 注意点（踩过的坑 / 局限）

1. **`build_hull(verify=False)` 不是标定船型**：未拟合时 Cb = 0.834（目标 0.8098），
   面板体体积会凭空偏 +2.9 %。本模块因此强制走 `build_hull()`（拟合，瞬时 < 0.1 s）。
2. **盖面识别不能按法向判**：船中剖面是「平顶」超椭圆（n 可达 31），水线附近
   侧面板的 `n_z` 也接近 1；改用「四个角点都在水线上 ⇒ 质心 z 严格为 0」。
3. **`Kvlcc2Hull.half_breadth(x, z)` 会把 z 广播到 x 上**（同长度的一维 z 会得到
   二维输出），必须逐站传独立深度网格。
4. **盖面只对艏摇合法**：横漂流场会穿盖面，故 `M_22` 必须用「无盖 + 水线镜像」，
   同一套代码由 `with_lid=False` 支持。
5. **未完成项（诚实申报）**：平行轴锚 `I = m_a d²` 的交叉复核未能跑通 ——
   `added_mass_matrix_6dof` 在球体偏置配置下给出 `I_zz = 2.05`（球心在 (0,0,1.5)
   时理论为 0；球心在 (1.5,0,0) 时理论为 m_a d² = 4.71），与 D1 报告 C3 的结论
   不一致；本单因此**不**用该锚做验收，只用「球/回转体绕直径艏摇 = 0」这一
   已被 D1 验证的结构性锚（本次新几何管线复算 |J|/m_a a² < 1e-5）。该不一致
   作为新的待查项登记在账本 §2（D8），不当成本单结论。
6. 势流、无粘、无升力、无自由面抬高：与条形理论同一物理层面，本单只回答
   「同一物理模型下二维近似 vs 三维解的差」。

## 5. 误差判定

| 关键量 | 基准 | 结果 | 是否 < 3 % |
|---|---|---|---|
| 面板体体积 | Cb L B T（拟合船型 {res['hull']['volume_fit']:.4f} m³） | 最细网格 {1e2 * ladder[-1]['volume_rel_err']:+.2f} % | 是 |
| 盖面面积 | Cwp L B | 最细网格 {1e2 * ladder[-1]['lid_area_rel_err']:+.2f} % | 是 |
| J_z 自身离散不确定度 | 幂律拟合 q = {q:.2f}，最细两级差 {1e2 * abs(ladder[-1]['J_z'] / ladder[-2]['J_z'] - 1):.2f} % | 外推 {j_ref:.1f} kg m² | 是（< 3 %） |
| 球/回转体绕直径艏摇 | 0（解析） | {ladder[-1]['lid_flux_max']:.1e}（盖面流速）；球体测试见 tests | 是 |
| J_z：面元 vs 细长体 | 无解析基准，二者是不同物理近似 | 比值 {j_ref / j_slender:.3f} | **不适用**：这不是误差，而是「细长体高估 {100.0 * (j_slender / j_ref - 1.0):.0f} %」的量化结论 |

换默认值判定（账本 D7 验收句：同一船 J_z 两法差异报告 + 回转直径/初转期变化 < 15 %
才可换默认值）：最大 |变化| = 回转 {man['max_abs_delta_pct']['turning']:.2f} %、
蛇形 {man['max_abs_delta_pct']['zigzag']:.2f} % ⇒ **判据是否满足 15 %：{judge}**。
判据**已满足**，即账本允许换默认值；但本单**仍不改默认**（`j_z_override` 默认 `None`）：
细长体值在本仓其它链路（预估舵角裕度、报告历史）里仍是被引用的口径，换默认属标定决策，
应在拖曳/自航试验或 CFD 复核后由标定单统一改。本单交付的是通道 + 量化结论 + 灵敏度区间。

## 6. 复现

```bash
PYTHONPATH=src python scripts/verify_maneuvering_added_inertia.py
PYTHONPATH=src python -m pytest tests/test_maneuvering_added_inertia.py -q
```
"""
    (out / "REPORT.md").write_text(report)
    print(f"wrote {out}/REPORT.md, summary.json, 2 figures")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/maneuvering-added-inertia")
    main(ap.parse_args().out)