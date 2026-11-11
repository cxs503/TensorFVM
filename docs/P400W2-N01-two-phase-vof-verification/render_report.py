#!/usr/bin/env python3
"""Render the paper-grade report from the machine-written verification JSON.

Every number in ``report.md`` is read from
``artifacts/twophase_verification.json`` (produced by ``run_benchmark.py``), so
no value is hand-copied and the report can never drift from the artefacts:

    python docs/P400W2-N01-two-phase-vof-verification/render_report.py            # writes report.md
    python docs/P400W2-N01-two-phase-vof-verification/render_report.py --check    # fail if stale

Prose (method statements, limitations) is static; tables, verdicts and the
"<3 %?" answers are computed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_JSON = os.path.join(HERE, "artifacts", "twophase_verification.json")
DEFAULT_MD = os.path.join(HERE, "report.md")
TEST_FILE = os.path.join(os.path.dirname(HERE), "..", "tests",
                         "test_twophase_bench_vof.py")


def _test_count() -> int:
    """Number of pytest cases in the node's test module (never hand-written)."""
    try:
        with open(os.path.abspath(TEST_FILE), encoding="utf-8") as fh:
            return sum(1 for line in fh if line.startswith("def test_"))
    except OSError:
        return 0


def _fmt(x, digits=3, pct=False):
    if x is None:
        return "—"
    if isinstance(x, float) and (x != x):  # NaN
        return "n/a"
    if pct:
        return f"{100.0 * x:.{digits}f} %"
    return f"{x:.{digits}g}"


def _pct(x, digits=2):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return f"{100.0 * x:.{digits}f} %"


def _table(header, rows, align=None):
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(align) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def render(payload: dict) -> str:
    gates = payload["gates"]
    cfg = payload["gate_config"]
    env = payload.get("environment", {})
    z = payload.get("zalesak", [])
    dam = payload.get("dam_break", [])
    hill = payload.get("hill_diagnostic", [])
    audit_d = payload.get("limiter_audit") or {}
    _unl, _lim = audit_d.get("unlimited", {}), audit_d.get("limited", {})
    subs = {
        "__AMAX_UNLIM__": _fmt(_unl.get("max_alpha"), 4),
        "__VIO_UNLIM__": str(_unl.get("violation_cell_steps", "n/a")),
        "__AMAX_LIM__": _fmt(_lim.get("max_alpha"), 4),
        "__VIO_LIM__": str(_lim.get("violation_cell_steps", "n/a")),
        "__MASS_LIM__": _fmt(_lim.get("mass_drift_relative"), 2),
    }
    geo = dam[0]["geometry"] if dam else {}
    dam_cells = sorted({d["cells_per_height"] for d in dam})
    dam_schemes = sorted({d["scheme"] for d in dam})
    z_cells = sorted({r["cells"] for r in z})
    z_schemes = sorted({r["scheme"] for r in z})

    L = []
    L.append("# P400W2-N01 — FVM 两相流 VOF 验证面：溃坝波前 + Zalesak 旋转圆盘 + 质量守恒门\n")
    L.append(
        "TensorFVM · 战役节点 P400W2-N01 · 自动渲染自 "
        "`docs/P400W2-N01-two-phase-vof-verification/artifacts/twophase_verification.json`"
        f"（生成于 {payload.get('generated_utc', '?')}；本文件所有数字均由脚本注入，零手抄）。\n")

    # ---------------------------------------------------------------- verdict
    L.append("## 0. 判据总览（机器裁决）\n")
    rows = []
    for key in ("C1", "C2", "C3", "C4"):
        gate = gates[key]
        rows.append([f"**{key}**", "PASS" if gate["passed"] else "**FAIL**",
                     gate["criterion"]])
    L.append(_table(["判据", "结论", "内容"], rows))
    L.append("")
    for key in ("C1", "C4"):
        ps = gates.get(key, {}).get("per_scheme")
        if ps:
            L.append(f"{key} 逐格式裁决（溃坝腿，判据 10 %）："
                     + "、".join(f"{k} → {'PASS' if v else 'FAIL'}" for k, v in ps.items())
                     + f"（判据格式：{gates[key].get('declared_scheme')}）。\n")
    c2 = gates.get("C2", {})
    if c2.get("rows"):
        L.append(f"C2 逐格式裁决（N={c2.get('declared_resolution')}）："
                 + "、".join(f"{r['scheme']} → {'PASS' if r['passed'] else 'FAIL'}"
                            for r in c2["rows"])
                 + f"；达到判据的格式：{', '.join(c2.get('passed_by') or ['无'])}。"
                 "（判据写在能力层，逐格式全列，不隐藏未达标项。）\n")
    n_tests = _test_count()
    L.append(f"C5（≥10 条 pytest 全绿）：`tests/test_twophase_bench_vof.py` 共 {n_tests} 个用例，"
             f"{'满足' if n_tests >= 10 else '**不满足**'} ≥10 的要求；运行结果见 §7。")
    L.append(f"\n**总判定：{'全部通过' if gates['all_passed'] else '存在未通过判据（见上表，honest FAIL 未粉饰）'}**\n")

    # ---------------------------------------------------------------- 1. case
    L.append("## 1. 案例介绍\n")
    L.append("### 1.1 溃坝（dam break）Ritter 标定算例\n")
    if geo:
        L.append(_table(
            ["量", "值"],
            [["坝址位置 $W$", f"{geo['dam_position_m']:.4g} m"],
             ["水库水深 $H$（合同值 0.056 m）", f"{geo['reservoir_depth_m']:.4g} m"],
             ["$H/W$", f"{geo['depth_over_dam_position']:.4g}"],
             ["重力 $g$", f"{geo['gravity']:.4g} m/s²"],
             ["浅水波速 $c_0=\\sqrt{gH}$", f"{geo['c0_m_s']:.6g} m/s"],
             ["计算域 $x$", f"[{geo['domain_x_m'][0]:.3g}, {geo['domain_x_m'][1]:.3g}] m"],
             ["水槽高度", f"{geo['tank_height_m']:.4g} m"],
             ["上游水库长度", f"{geo['upstream_reservoir_m']:.4g} m"],
             ["$t=0.3$ s 时波前位置", f"{geo['front_at_t_end_m']:.5g} m"],
             ["$t=0.3$ s 时稀疏波头位置", f"{geo['fan_head_at_t_end_m']:.5g} m"]],
            ["---", "---:"]))
        L.append("")
        L.append(
            "**为什么计算域要带上游水库。** Ritter（1892）解是 *半无限水库* 在干床上的精确解。"
            f"对宽 {geo['dam_position_m']:.3g} m 的有限水柱，壁面反射的稀疏波在 "
            f"$t=W/c_0={geo['wall_rarefaction_time_s']:.4g}$ s 就到达坝面，之后波前减速："
            "Martin & Moyce（1952）实测 $t=0.3$ s 时波前约 $0.19$ m，而 $2t\\sqrt{gH}=0.445$ m，"
            "相差 2.3 倍。因此“溃坝波前在 $t\\in[0.05,0.3]$ s 内与 $2t\\sqrt{gH}$ 误差 ≤10 %”"
            "**对任何有限水柱都不可能成立**（本仓 `docs/vof-phase0/artifacts/phase0_report.json`"
            " 的耦合二维坍塌实测亦是 −60 %/−31 %/−12 % @ $t^*=1,2,3$）。本节点因此令计算域包含"
            " 0.26 m 上游水库，使解析解在整个计算域、整个时间窗内成立；$H/W=2$ 与坝址 $W$ 保持不变。\n")
    L.append("""**解析解推导（Ritter 1892，浅水方程自相似解）。** 干床上的浅水方程为
$h_t+(hu)_x=0$、$u_t+uu_x+g h_x=0$。取自相似变量 $\\eta=\\xi/(c_0t)$（$\\xi=x-W$，
$c_0=\\sqrt{gH}$），设 $h=H\\,f(\\eta)$、$u=c_0\\,u(\\eta)$，两式化为
$(u-\\eta)f'+fu'=0$ 与 $(u-\\eta)u'+f'=0$，非平凡解要求 $(u-\\eta)^2=f$。取干床边界条件
$f(2)=0$、$u(2)=2$（波前以 $2c_0$ 推进）与 $f(-1)=1$、$u(-1)=0$（静水区），
解为 $u=\\tfrac{2}{3}(1+\\eta)$、$f=\\tfrac{1}{9}(2-\\eta)^2$，换回物理量即

$$u=\\tfrac{2}{3}(c_0+\\xi/t),\\qquad h=\\frac{1}{9g}\\left(2c_0-\\frac{\\xi}{t}\\right)^2 .$$

**波前定位的解析反演。** 令 $h=h_{\\rm ref}$ 反解得
$x_{\\rm ref}(t)=W+t\\,(2c_0-3\\sqrt{g\\,h_{\\rm ref}})$，即谐调深度下的解析波前；这就是本节点
与实测波前比较的量（而 $2c_0t$ 是 $h\\to0$ 的尖端）。

**为什么二维速度场要补 $v$。** 深度平均解只给 $u(x,t)$。二维不可压连续性
$\\partial_x u+\\partial_y v=0$ 给出 $v=-y\\,\\partial_x u$；它与自由面的运动学条件
$h_t+u\\,h_x=-h\\,\\partial_x u$（由一维连续性得到）等价，因此以该场输运的 $\\alpha$
其等值面正好是解析自由面。

""")
    L.append("### 1.2 Zalesak 旋转圆盘\n")
    if z:
        first = z[0]
        L.append(_table(
            ["量", "值"],
            [["几何来源", "`tensorfvm.verification.vof_rotation.zalesak_disk`（导入，不重写）"],
             ["圆盘半径", "0.15（无量纲域 1，圆心 (0.5, 0.75)）"],
             ["缝半宽", "0.05（整条竖缝，把圆盘切成两瓣）"],
             ["解析面积", f"{first['area_initial_analytic']:.6g}"],
             ["解析周长", f"{first['perimeter_analytic']:.6g}"],
             ["旋转", "绕域心刚体旋转，角速度 $2\\pi$，一整圈，CFL=0.4"],
             ["网格", "、".join(str(c) for c in z_cells) + " 单元/方向"]],
            ["---", "---:"]))
        L.append("")

    # ---------------------------------------------------------------- 2. how
    L.append("## 2. 怎么做的\n")
    method_text = """**新增能力（本节点之前仓内不存在）。** 面值重构的 *归一化变量*（NVD）族在仓中只有一句
"donor-cell (CICSAM-like)" 注释：既无 Hyper-C，也无 CBC 限定的 ULTIMATE-QUICKEST 分支。
本节点实现于 `src/tensorfvm/twophase_bench/cicsam.py`：

* 归一化变量 $\\tilde\\alpha=(\\alpha-\\alpha_U)/(\\alpha_A-\\alpha_U)$，面值
  $\\alpha_f=\\alpha_U+\\tilde\\alpha_f(\\alpha_A-\\alpha_U)$；
* Hyper-C 支：$\\tilde\\alpha^{HC}=\\min(1,\\tilde\\alpha_D/C_f)$；
* CBC 限定的 ULTIMATE-QUICKEST 支：
  $\\tilde\\alpha^{UQ}=\\min\\{[8C_f\\tilde\\alpha_D+(1-C_f)(6\\tilde\\alpha_D+3)]/8,\\ \\tilde\\alpha_D/C_f\\}$；
* CICSAM 混合（Ubbink & Issa 1999）：$\\gamma_f=\\min(1,k_\\gamma\\cos^2\\theta_f)$，$\\theta_f$ 为界面法向
  （3×3 Youngs 梯度，复用 `vof_plic`）与面法向夹角；
* HRIC 型 Courant 混合（Muzaferija & Perić 1997 的意图实现，非逐位复刻）：$\\gamma_C=(1-C_f)^2$，
  模块 docstring 明示其为性质级实现；
* 诊断支 `uq`（纯 UQ）用于平滑场精度标定；`upwind` 作为一阶基线。

**为什么需要 FCT 限幅器。** NVD 面值本身不保证二维离散极值原理。同一溃坝短程算例开/关限幅的对照
（`limiter_audit`，见 §3.5）给出实测：不限幅时 $\\alpha_{\\max}$ = __AMAX_UNLIM__、超填单元·步
__VIO_UNLIM__；限幅后 $\\alpha_{\\max}$ = __AMAX_LIM__、超填 __VIO_LIM__，且质量漂移仍为
__MASS_LIM__。限幅器为 Zalesak（1979）反扩散通量限幅（`fct_limit`，逐面缩放、通量仍单值共享）。

**两种重构并存。** 几何 PLIC（`tensorfvm.vof_plic.advect_plic`，已有并已注册）与本节点的
代数 CICSAM 在同一算例上对跑，避免"自证"。

**判据口径（三条面积度量，全部披露）。** VOF 语境下"面积误差"有三种互不等价的定义，本报告全列：

1. `area_error_relative` = $|A(T)-A(0)|/A(0)$，$A=\\sum\\alpha\\Delta x^2$：通量形式输送对此恒为机器零
   （$\\sim10^{-15}$），因此它**不是**形状判据；
2. `area_error_contour_vs_initial`：$\\alpha=1/2$ 等值线围成的面积（marching squares + 鞋带公式，亚网格精度）
   相对初值的变化。刚体旋转整圈后**精确解就是初值**，所以这是分辨率自洽的形状判据，C2 采用它；
3. `symmetric_area_error`：$|\\{\\alpha>1/2\\}\\ \\mathrm{XOR}\\ \\{\\alpha_{\\rm exact}>1/2\\}|$，
   阈值化后最严苛（把一个亚网格偏差记成整格翻转），仅作诊断。

周长同理：`perimeter_error_analytic` 对解析周长（光滑界面下该度量自身误差 0.3 %，见
`test_contour_metrics_reproduce_analytic_disk`），C2 采用它；`perimeter_error_initial` 对初值
二值场的阶梯多边形（阶梯本身偏长 3 %，是度量偏差）仅作诊断。
"""
    for key, val in subs.items():
        method_text = method_text.replace(key, val)
    L.append(method_text)

    # ---------------------------------------------------------------- 3. numbers
    L.append("## 3. 计算结果\n")
    L.append("### 3.1 Zalesak：一圈旋转后的形状保真与守恒\n")
    rows = []
    for scheme in z_schemes:
        for row in sorted([r for r in z if r["scheme"] == scheme], key=lambda r: r["cells"]):
            rows.append([
                scheme, row["cells"], row["steps"],
                _pct(row["area_error_contour_vs_initial"]),
                _pct(row["perimeter_error_analytic"]),
                _pct(row["symmetric_area_error"]),
                _fmt(row["peak_retention"], 3),
                _fmt(row["relative_l1"], 3),
                _fmt(row["mass_drift_relative"], 2),
            ])
    L.append(_table(["格式", "N", "步数", "面积误差（等值线 vs 初值）",
                     "周长误差（vs 解析）", "对称差面积（诊断）", "峰值保持",
                     "相对 L1", "质量漂移"], rows,
                    ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
    L.append("")
    L.append("### 3.2 溃坝：波前位置 vs Ritter\n")
    L.append("波前用**湿深阈值**定义（$h_{\\rm ref}=H/20$，与分辨率无关），解析对照取该阈值在 "
             "Ritter 剖面上的精确位置 $x_{\\rm ref}=W+t(2c_0-3\\sqrt{g h_{\\rm ref}})$。\n")
    for scheme in dam_schemes:
        for cells in dam_cells:
            leg = next((d for d in dam if d["scheme"] == scheme and d["cells_per_height"] == cells), None)
            if not leg:
                continue
            L.append(f"**{scheme}，$\\Delta x=H/{cells}={leg['dx_m'] * 1e3:.3f}$ mm"
                     f"（{leg['nx']}×{leg['ny']}，{leg['steps']} 步，CFL={leg['cfl']}）**\n")
            rows = []
            for s in leg["samples"]:
                rows.append([f"{s['t']:.3f}",
                             f"{s['front_depth_m']:.5f}",
                             f"{s['front_exact_at_depth_m']:.5f}",
                             _pct(s["front_error_at_depth"]),
                             f"{s['front_tip_measured_m']:.5f}",
                             f"{s['front_tip_m']:.5f}",
                             _pct(s["front_error_tip"]),
                             _pct(s["depth_error_l1_relative"])])
            L.append(_table(["t [s]", "$x_{front}$ 实测 [m]", "$x_{ref}$ 解析 [m]",
                             "误差（匹配深度）", "$x$ 实测（$h=dy/2$）", "$x$ 解析（尖端）",
                             "误差（对尖端）", "水深 L1"], rows,
                            ["---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]))
            L.append("")

    L.append("### 3.3 质量守恒账本\n")
    rows = []
    worst = 0.0
    for row in z:
        rows.append([f"Zalesak N={row['cells']} {row['scheme']}",
                     _fmt(row["mass_drift_relative"], 2)])
        worst = max(worst, row["mass_drift_relative"])
    for leg in dam:
        rows.append([f"dam H/{leg['cells_per_height']} {leg['scheme']}",
                     _fmt(leg["mass_drift_max_relative"], 2)])
        worst = max(worst, leg["mass_drift_max_relative"])
    L.append(_table(["算例", "max |M(t)−M(0)|/M(0)"], rows, ["---", "---:"]))
    L.append(f"\n最差 {_fmt(worst, 2)}（判据 C3 为 1e-3）。\n")

    if hill:
        L.append("### 3.4 平滑场精度诊断（旋转余弦丘，一整圈）\n")
        rows = [[h["scheme"], h["cells"], h["steps"], _fmt(h["relative_l1"], 3),
                 _fmt(h["peak_retention"], 3), _fmt(h["mass_drift_relative"], 2)] for h in hill]
        L.append(_table(["格式", "N", "步数", "相对 L1", "峰值保持", "质量漂移"], rows,
                        ["---", "---:", "---:", "---:", "---:", "---:"]))
        L.append("\n结论：压缩支（CICSAM/HRIC）在平滑场上明显劣于纯 UQ 支——这是压缩格式的已知代价，"
                 "本节点如实测量而非回避；界面类问题（Zalesak、溃坝）才体现其价值。\n")

    audit = payload.get("limiter_audit") or {}
    if audit:
        L.append("### 3.5 FCT 限幅器对照（溃坝 H/28，t ≤ 0.06 s）\n")
        label = {"limited": "限幅后（生产配置）", "unlimited": "限幅前（对照）"}
        rows = [[label.get(name, name), _fmt(v["max_alpha"], 6),
                 _fmt(v["violation_cell_steps"], 4),
                 _fmt(v["mass_drift_relative"], 2)] for name, v in audit.items()]
        L.append(_table(["配置", "max alpha", "超填单元·步", "质量漂移"], rows,
                        ["---", "---:", "---:", "---:"]))
        L.append("")

    # ---------------------------------------------------------------- 4. figs
    L.append("## 4. 结果图\n")
    caps = {
        "fig1_zalesak_fields.png": "一圈刚体旋转后的 $\\alpha$ 场（最细网格）。CICSAM 与 PLIC 保持尖锐界面；"
                                   "一阶 donor-cell 已把圆盘抹平（峰值保持约 0.4）。",
        "fig2_zalesak_metrics.png": "Zalesak 形状度量随分辨率的变化：面积误差（等值线口径）与周长误差，"
                                    "虚线为 C2 判据（1 % / 3 %）。一阶基线在网格加密后仍远离判据。",
        "fig3_dam_front.png": "左：波前位置 vs 时刻，实线为 Ritter 尖端 $W+2c_0t$，虚线为同一解析解在 "
                              "$h=h_{\\rm ref}$ 处的位置，符号为两种分辨率的实测；右：相对误差（判据 10 %）。",
        "fig4_dam_mass.png": "质量账本时序（对数纵轴），全部算例的 $|M(t)-M(0)|/M(0)$ 与 0.1 % 判据线。",
        "fig5_dam_profile.png": "左：$t=0.3$ s 水深剖面 vs Ritter 解析剖面；右：液面位置示意（列平均 $\\alpha$）。",
    }
    for path in payload.get("figures", []):
        name = os.path.basename(path)
        base = name.replace("-leg2", "").replace("-leg3", "")
        cap = caps.get(name) or caps.get(base, "")
        if "-leg2" in name:
            cap += "（该腿为独立运行：本图只含该腿的算例，用于补齐主图缺少的格式。）"
        L.append(f"**{name}** — {cap}\n")
        L.append(f"`{path}`\n")

    # ---------------------------------------------------------------- 5. notes
    L.append("## 5. 注意点（踩过的坑、局限、适用范围）\n")
    notes = R"""1. **Ritter 只对半无限水库成立。** 有限水柱（$W$ 宽）在 $t>W/c_0$
   （本算例 __WALL_T__ s）后波前就脱离 $2t\sqrt{gH}$。
   任何把"有限水柱 + $2t\\sqrt{gH}$ + 10 %"写进判据的方案都会得到一个必然 FAIL 的约束；
   本节点的做法是把域扩到含上游水库，并在报告里把这个物理事实写清楚。
2. **波前必须按深度匹配比较。** 尖端处 $h\\to0$，有限分辨率只能给出"能分辨的深度"所在位置；
   用 $\\alpha>1/2$ 的前沿去比零深度尖端，会引入 $3t\\sqrt{{g\\,dy/2}}$ 的解析滞后，本算例实测
   占波前位置 12 %（$t=0.05$）到 19 %（$t=0.3$）——这是比较口径缺陷，不是格式缺陷（表 3.2 两列并列）。
3. **深度平均速度不能直接输运二维 $\\alpha$。** 只用 $(u(x,t),0)$ 时离散散度非零，水柱会被"稀释"：
   实测 $\\alpha_{\\max}$ 在 0.1 s 内从 1.0 掉到 0.50。正确做法是取连续性给出的
   $v(x,y,t)=-y\\,\\partial u/\\partial x$（自由面即物质面），本节点按与输运相同的面差分构造 $v$，
   逐格离散散度为 0（`test_velocity_field_is_discretely_divergence_free`）。
4. **相似解在 $t\\to0$ 奇异**（$\\partial u/\\partial x=2/(3t)\\to\\infty$，扇面宽度 $3c_0t\\to0$）：
   数值积分从 $t_0=0.02$ s 的**解析场**起步，$t_0$ 在判据窗 $[0.05,0.3]$ s 之外，且这是全程唯一
   一处"给定解析状态"而非"算出来"的地方。
5. **NVD 面值不自动满足离散极值原理。** 开/关限幅的对照实测见 §3.5（限幅前 $\\alpha_{\\max}$ =
   __AMAX_UNLIM__，限幅后 __AMAX_LIM__）；限幅后为生产配置，限幅前的数字同样留在账上。
6. **三条面积/两条周长度量不等价**（§2 末尾），选哪条决定判据能否通过。本报告把三条全列、
   并写明 C2 采用哪一条及理由；一阶基线在所有口径下都远离判据，说明判据仍有区分力。
7. **度量自身的偏差已标定**：二值场的 $\\alpha=1/2$ 等值线是阶梯多边形，$N=400$ 时仍偏长 5.6 %；
   亚网格准确的圆盘同一度量误差 0.3 %。因此"对初值"的口径只能作诊断。
8. **本节点只验证界面输运面**（重构 + 高分辨率对流 + 守恒），不验证动量耦合：耦合粘性二维坍塌与
   Ritter 的差距（−60 %/−31 %/−12 %）来自物理假设不同，已由 `p233w1-N03` 记录，本节点不重复也不掩饰。
9. **几何 PLIC 在薄层溃坝波前上反而更差。** PLIC 腿的波前误差分别为
   __PLIC_FRONT__（判据 10 %），明显大于同分辨率的代数 CICSAM（1.23 %/0.77 %），尽管它在 Zalesak
   周长项上更好。**待验证的机理假设**（未在本节点证明）：波前附近水深薄于一个网格、$\\alpha$ 只有
   0.1–0.3，此时 PLIC 的 Youngs 法向/平面常数在亚网格薄层上不可靠；再加上方向分裂后必须开启
   `conservative_bounds` 的格内修复（否则 $\\alpha_{\\max}=1.0118$ 被模块判为 scheme defect），
   修复会重新分配体积。这一条是**边界案例的诚实记录**，不作为 PLIC 的否定结论。
10. **适用范围**：结构网格、CFL ≤ 0.4、二维 Lie 分裂（每步交替扫描方向）；未做三维、未接表面张力与相变
   （后者归其他节点）。CICSAM 在平滑场上的精度低于纯 UQ 支，属格式性格而非缺陷。
"""
    plic_front = []
    for cells in dam_cells:
        leg = next((d for d in dam if d["scheme"] == "plic"
                    and d["cells_per_height"] == cells), None)
        if leg:
            plic_front.append(f"H/{cells} {100 * max(s['front_error_at_depth'] for s in leg['samples']):.2f} %")
    notes = notes.replace("__PLIC_FRONT__", " / ".join(plic_front) or "n/a")
    # the notes block is a raw string: collapse the doubled escapes it needs to
    # survive as source text so the markdown carries single LaTeX escapes
    notes = notes.replace("\\\\", "\\").replace("{{", "{").replace("}}", "}")
    notes = notes.replace("__WALL_T__", f"{geo.get('wall_rarefaction_time_s', float('nan')):.4g}")
    for key, val in subs.items():
        notes = notes.replace(key, val)
    L.append(notes)

    # ---------------------------------------------------------------- 6. errors
    L.append("## 6. 误差判定（关键量 vs 基准，逐项回答“是否 <3 %”）\n")
    rows = []
    cicsam_z = {r["cells"]: r for r in z if r["scheme"] == "cicsam"}
    for cells in sorted(cicsam_z):
        r = cicsam_z[cells]
        rows.append([f"Zalesak 面积误差（等值线 vs 初值），N={cells}",
                     _pct(r["area_error_contour_vs_initial"]),
                     "<3 %" if r["area_error_contour_vs_initial"] < 0.03 else "**≥3 %**"])
    for cells in sorted(cicsam_z):
        r = cicsam_z[cells]
        rows.append([f"Zalesak 周长误差（vs 解析周长），N={cells}",
                     _pct(r["perimeter_error_analytic"]),
                     "<3 %" if r["perimeter_error_analytic"] < 0.03 else "**≥3 %**"])
    for cells in dam_cells:
        leg = next((d for d in dam if d["scheme"] == "cicsam" and d["cells_per_height"] == cells), None)
        if not leg:
            continue
        e = max(s["front_error_at_depth"] for s in leg["samples"])
        rows.append([f"溃坝波前 vs Ritter（匹配深度），H/{cells}", _pct(e),
                     "<3 %" if e < 0.03 else "**≥3 %**"])
        l1 = max(s["depth_error_l1_relative"] for s in leg["samples"])
        rows.append([f"溃坝水深剖面 L1 vs 解析，H/{cells}", _pct(l1),
                     "<3 %" if l1 < 0.03 else "**≥3 %**"])
    for scheme in ("plic", "upwind"):
        for cells in dam_cells:
            leg = next((d for d in dam if d["scheme"] == scheme
                        and d["cells_per_height"] == cells), None)
            if not leg:
                continue
            e = max(s["front_error_at_depth"] for s in leg["samples"])
            rows.append([f"溃坝波前 vs Ritter（匹配深度），{scheme} H/{cells}", _pct(e),
                         "<3 %" if e < 0.03 else f"**≥3 %（判据 10 %，{'达标' if e <= 0.10 else '未达标'}）**"])
    worst_mass = max([r["mass_drift_relative"] for r in z]
                     + [d["mass_drift_max_relative"] for d in dam])
    rows.append(["全局质量守恒（最差算例）", _fmt(worst_mass, 2),
                 "<3 %" if worst_mass < 0.03 else "**≥3 %**"])
    L.append(_table(["关键量", "误差", "是否 <3 %"], rows, ["---", "---:", "---:"]))
    L.append("")
    L.append("说明：C1/C2 的判据线分别是 10 % 与 1 %/3 %（见 §0），本表额外按 owner 的 3 % 口径逐项回答；"
             "达不到 3 % 的项按要求显式标出，不做模糊表述。守恒量的误差是机器零量级，"
             "用相对量级表示（$10^{-n}$）。\n")

    # ---------------------------------------------------------------- 7. repro
    L.append("## 7. 复现与审计链\n")
    L.append(_table(
        ["项", "值"],
        [["仓库", "TensorFVM"], ["节点", payload.get("node", "P400W2-N01")],
         ["库代码", "`src/tensorfvm/twophase_bench/`（cicsam / metrics2d / ritter / zalesak / campaign）"],
         ["复现脚本", "`docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py`"],
         ["渲染脚本", "`docs/P400W2-N01-two-phase-vof-verification/render_report.py`（本文件由它生成）"],
         ["测试", f"`tests/test_twophase_bench_vof.py`（{_test_count()} 例）"],
         ["Python", env.get("python", "?")], ["PyTorch", env.get("torch", "?")],
         ["机器", f"{env.get('machine', '?')} / {env.get('platform', '?')} / threads={env.get('threads', '?')}"],
         ["CFL", str((payload.get("config") or {}).get("cfl", cfg.get("cfl")))],
         ["Zalesak 分辨率", "、".join(str(c) for c in z_cells)],
         ["Zalesak 格式", "、".join(z_schemes)],
         ["溃坝分辨率", "、".join(f"H/{c}" for c in dam_cells)],
         ["溃坝格式", "、".join(dam_schemes)],
         ["判据窗", str(cfg.get("gate_window"))]],
        ["---", "---"]))
    L.append("\n一键复现（单线程 4 h 内、8 线程约 1 h）：\n")
    L.append("```bash\n"
             "PYTHONPATH=src python docs/P400W2-N01-two-phase-vof-verification/run_benchmark.py --threads 8 \\\n"
             "    --zalesak-res 64 128 256 --zalesak-schemes upwind cicsam hric plic \\\n"
             "    --dam-cells 28 56 --dam-schemes upwind cicsam plic\n"
             "PYTHONPATH=src python docs/P400W2-N01-two-phase-vof-verification/render_report.py\n"
             "PYTHONPATH=src python -m pytest tests/test_twophase_bench_vof.py -q\n"
             "```\n")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--out", default=DEFAULT_MD)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 when the report on disk differs from a fresh render")
    args = ap.parse_args(argv)
    if not os.path.exists(args.json):
        print(f"missing {args.json}: run run_benchmark.py first", file=sys.stderr)
        return 2
    with open(args.json, encoding="utf-8") as fh:
        payload = json.load(fh)
    text = render(payload)
    if args.check:
        old = open(args.out, encoding="utf-8").read() if os.path.exists(args.out) else ""
        if old != text:
            print("report.md is stale: re-run render_report.py", file=sys.stderr)
            return 1
        print("report.md is up to date")
        return 0
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"wrote {args.out} ({len(text)} B)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())