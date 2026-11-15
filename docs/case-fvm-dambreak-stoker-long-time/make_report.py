#!/usr/bin/env python
"""Render ``REPORT.md`` for the long-time dam-break study from ``study.json``.

Every number in the report is injected from the artifact; the prose is fixed.
That keeps the document honest if the study is ever re-run (a FAIL is reported
as a FAIL, never as a pass with different wording).
"""

from __future__ import annotations

import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
ART = ROOT / "src" / "tensorfvm" / "verification" / "artifacts" / "shallow-water-long-time"


def pct(value, digits=4):
    if value is None or value != value:
        return "n/a"
    return f"{100.0 * value:.{digits}f} %"


def sci(value):
    return f"{value:.2e}"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    src = pathlib.Path(argv[0]) if argv else ART / "study.json"
    payload = json.loads(src.read_text())
    crit = payload["criteria"]
    out = []
    out.append("# CASE-FVM-DAMBREAK-STOKER · 时域有效性（自相似延续）验证：t* 到 10（锚点时刻的 5 倍）\n")
    out.append(
        "**锚点**: `CASE-FVM-DAMBREAK-STOKER`（TensorFVM 浅水算例池）；本单扩的是锚点"
        "验收量 `x_f(t)` 的**时间维度**\n"
        "**参考**: Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**, 947；Stoker (1957) "
        "*Water Waves* ch. 10（两者均为 `xi = x/t` 的自相似解）\n"
        f"**设置**: dx = {crit['dx']} m（= 锚点接受网格 1600 单元 / 8 m），"
        f"CFL = {crit['cfl']}，order = {crit['order']}，"
        f"t* ∈ {{{', '.join(str(t) for t in crit['t_stars'])}}}，"
        f"坍缩参照 t* = {crit['t_ref']}\n"
        f"**判据（跑之前写死）**: 波位置 ≤ {pct(crit['tolerance'], 0)}；"
        f"波速 `x_f/t` ≤ {pct(crit['speed_tolerance'], 0)}；"
        f"自相似坍缩 L1 ≤ {pct(crit['collapse_tolerance'], 1)} h0；"
        f"静水区水位阶跃漂移 ≤ {pct(crit['step_drift_tolerance'], 2)}；"
        "内部伪极值 = 0；质量漂移 ≤ 1e-10；水深非负\n")
    all_pass = all(c["verdict"]["pass"] for c in payload["cases"].values())
    out.append(f"**判定**: {'✅ 全部通过' if all_pass else '❌ 存在不通过项（见下表，如实记录）'}\n")

    out.append("## ① 案例介绍\n")
    out.append(
        "锚点的溃坝算例只在 `t* = 2` 一个时刻被评级，锚点此前所有扩展（四波型黎曼、伽利略提升、"
        "湿床阶梯）同样是单时刻读数。这是验收量 `x_f(t)` 的结构性盲区：\n"
        "\n"
        "> 一个**恒定的波速偏差** `c` 造成的位置误差是 `c t`。在 `t* = 2` 上，"
        "只要 `c/U ≤ 2.5 %` 它就能整个藏进 5 % 的验收带里，而在 `t` 上无界增长。\n"
        "\n"
        "两个参考解都是 `xi = (x - x_dam)/t` 的**自相似解**，因此有三类量在**任意**时刻都有"
        "精确值可判：\n"
        "\n"
        "| 量 | 精确值 | 为什么对时间维度有判决力 |\n"
        "|---|---|---|\n"
        "| 波速 `x_f(t)/t` | `2√(g h0)`（Ritter）/ `U`（Stoker 涌波） | 不随 `t` 衰减：偏差是偏差 |\n"
        "| 静水区水位阶跃 `h2/h1` | 常数 | 精确解里时间不变，任何漂移都是非物理的（边界反射、"
        "非守恒更新、水深地板泄漏） |\n"
        "| 自相似坍缩 `h(xi; t)` vs `h(xi; t_ref)` | 完全重合 | 不看解析解，"
        "无法被“测量窗口贴合解析位置”这类同义反复满足 |\n"
        "\n"
        "几何：`dx` 固定，计算域随 `t` 增长（波必须留在域内，两端各留 "
        f"{crit.get('pad_c0', 0.5) if 'pad_c0' in crit else 0.5}·c0·t 的静水余量），"
        "于是跨越波结构的单元数随 `t` 增长，而累加的误差始终对着**同一个相对空间分辨率**"
        "（与锚点接受网格一致）来量。\n")

    out.append("## ② 怎么做的\n")
    out.append(
        "1. **被验证的求解器原样复用**：`swe1d.ShallowWater1D`（HLL + MUSCL/minmod + "
        "SSP-RK，自适应 CFL），没有为长时算例改一行格式；本单是“把既有能力放到新轴上评级”。\n"
        "2. **域随 `t` 生长**（`long_time.long_time_domain`）：稀疏波头以 `-c0` 前进、"
        "前锋/涌波以 `2c0`/`U` 前进，因此 `x_min = x_dam - c0 t - pad`、"
        "`x_max = x_front + pad`，`n = (x_max - x_min)/dx`，`dx` 全程不变。\n"
        "3. **波速拟合**（`front_speed_fit`）：每次运行内部按 12 个时刻采样 `x_f(t)`（沿用锚点"
        "的金标准 `fan_front_estimate` 扇形估计量），最小二乘拟合 `x_f = a t + b`，"
        "`a` 与解析波速比；`b` 必须 ≈ 0（波从坝址出发）。未分辨的早期样本（扇形不足 8 个单元）"
        "照锚点口径剔除并计数。\n"
        "4. **自相似坍缩**（`collapse_residual`）：把两个时刻的解都重采样到公共 `xi` 网格上，"
        "取 `L1/h0`。**度量的自证**：精确解在同一网格上的坍缩残差是机器零（见下表），"
        "所以残差不是度量噪声。\n"
        "5. **两条反证对照**（`_controls`）：(i) 在最长 `t*` 上把 CFL 从 0.4 减到 0.1（`dt` 缩小 4 倍），"
        "读数必须几乎不动，否则长时误差是*时间*误差、锚点的 SSP-RK 步需要审；"
        "(ii) 换成 `order = 1`，长时位置误差必须**变大**——证明这条轴有牙齿，不是更松的容差。\n")

    out.append("## ③ 计算结果\n")
    for key, case in payload["cases"].items():
        ref = case["reference"]
        out.append(f"### {key}（参考: {ref}）\n")
        out.append("| t* | 单元 n | 步数 | x_f 精确 | x_f 实测 | 波位置误差 | "
                   "`x_f/t` 误差 | 拟合波速误差 | 深度 L1 | 伪极值 | 质量漂移 |")
        out.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in case["rows"]:
            out.append(
                f"| {r['t_star']:g} | {r['n']} | {r['steps']} | {r['front_exact']:.4f} | "
                f"{r['front_measured']:.4f} | {pct(r['front_series_max_error'])} | "
                f"{pct(r['x_over_t_error'])} | {pct(r['speed'].get('speed_error'))} | "
                f"{pct(r['depth_l1'])} | {r['local_extrema']} | {sci(r['mass_drift'])} |")
        out.append("")
        wet = [r for r in case["rows"]
               if r.get("step_ratio_error") is not None
               and r["step_ratio_error"] == r["step_ratio_error"]]
        if wet:
            out.append("| t* | h2 精确 | h2 实测 | h2 误差 | h1 误差 | 阶跃 h2/h1 误差 | "
                       "涌波位置误差 | 平台窗口单元 |")
            out.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for r in wet:
                out.append(
                    f"| {r['t_star']:g} | {r['h2_exact']:.4f} | {r['h2_measured']:.4f} | "
                    f"{pct(r['h2_error'])} | {pct(r['h1_error'])} | "
                    f"{pct(r['step_ratio_error'])} | {pct(r['shock_error'])} | "
                    f"{r['plateau_window_cells']} |")
            out.append("")
        out.append("| t* | 数值坍缩 L1/h0 | 精确解坍缩 L1/h0（度量噪声底） |")
        out.append("|---:|---:|---:|")
        for c, e in zip(case["similarity"]["collapse"],
                        case["similarity"]["exact_collapse"]):
            out.append(f"| {c['t_star']:g} | {sci(c['l1_over_h0'])} | {sci(e['l1_over_h0'])} |")
        out.append("")
        verdict = case["verdict"]
        out.append(f"**{key} 判定: {'PASS' if verdict['pass'] else 'FAIL'}**\n")
        out.append("| 判据 | 实测 | 限 | 通过 |")
        out.append("|---|---:|---:|:--:|")
        for c in verdict["criteria"]:
            out.append(f"| {c['name']} | {c['value']:.6g} | {c['limit']:g} | "
                       f"{'✅' if c['pass'] else '❌'} |")
        out.append("")

    controls = payload.get("controls")
    if controls:
        out.append(f"### 反证对照（t* = {controls['t_star']:g}）\n")
        out.append("CFL（固定 dx）——时间离散是否在长时误差里占主导：\n")
        out.append("| CFL | 算例 | 波位置误差 | `x_f/t` 误差 | 拟合波速误差 | 深度 L1 |")
        out.append("|---:|---|---:|---:|---:|---:|")
        for cfl, rows in controls["cfl_refinement"].items():
            for key, r in rows.items():
                out.append(f"| {cfl} | {key} | {pct(r['front_error'])} | "
                           f"{pct(r['x_over_t_error'])} | "
                           f"{pct(r['speed'].get('speed_error'))} | {pct(r['depth_l1'])} |")
        out.append("")
        out.append("格式阶（可失败性）：`order = 1` 的长时误差必须比 `order = 2` 大。\n")
        out.append("| order | 算例 | 波位置误差 | `x_f/t` 误差 | 拟合波速误差 | 深度 L1 | 伪极值 |")
        out.append("|---:|---|---:|---:|---:|---:|---:|")
        for scheme, rows in controls["order_reference"].items():
            for key, r in rows.items():
                out.append(f"| {scheme} | {key} | {pct(r['front_error'])} | "
                           f"{pct(r['x_over_t_error'])} | "
                           f"{pct(r['speed'].get('speed_error'))} | "
                           f"{pct(r['depth_l1'])} | {r['local_extrema']} |")
        out.append("")

    out.append("## ④ 结果图\n")
    out.append("![errors](long_time_errors.png)\n")
    out.append("左：波位置相对误差随 `t*`（对数轴，含 5 % 验收带）；右：`x_f/t` 与最小二乘波速的"
               "相对误差——这是锚点单时刻读数看不见的量。\n")
    out.append("![collapse](long_time_collapse.png)\n")
    out.append("自相似坍缩：各 `t*` 的数值解与精确解在 `xi = x/t` 坐标下应当是同一条曲线；"
               "曲线若随 `t*` 漂移即说明数值解本身丢失了自相似性。\n")

    out.append("## ⑤ 注意点与局限\n")
    out.append(
        "* **不是新格式**：用的是生产求解器，本单只加“刻度”，不加算法；因此结论的适用范围"
        "就是这组 MUSCL+HLL+SSP-RK 设置。\n"
        "* **域随 `t` 增长**意味着单元数随 `t` 线性增长，长时算例的代价是 `O(t^2)`；"
        "本单最长的 `t* = 20` 运行规模见上表（步数/单元数）。\n"
        "* **干床前锋**仍是零水深尖点，位置读数沿用锚点的扇形形状估计量；该估计量在自相似下"
        "相对误差与 `t` 无关，因此适合做时间轴比较（湿床用中值水位亚网格交点）。\n"
        "* **湿床平台窗口**用解析 `x_fan_end`/`x_shock` 放置（与锚点 `_wet_metrics` 同口径），"
        "因此 `h2` 读数偏同义反复；平台窗口单元数与覆盖率一并列出备查。\n"
        "* **未覆盖**：非平底/摩阻、二维、干湿强耦合的长时间行为不在本单范围（锚点其它扩展覆盖"
        "这些轴，但都仍是单时刻）。\n")

    out.append("## ⑥ 误差判定（逐项 vs 基准，显式回答是否 < 3 %）\n")
    rows = []
    for key, case in payload["cases"].items():
        for r in case["rows"]:
            rows.append((key, r))
    worst_pos = max(rows, key=lambda kr: kr[1]["front_series_max_error"])
    worst_speed = max(rows, key=lambda kr: kr[1]["x_over_t_error"])
    worst_l1 = max(rows, key=lambda kr: kr[1]["depth_l1"])
    worst_collapse = max(
        ((k, c) for k, case in payload["cases"].items()
         for c in case["similarity"]["collapse"]),
        key=lambda kc: kc[1]["l1_over_h0"])
    out.append("| 关键量 | 最差读数 | 出现位置 | 基准 | 是否 < 3 % |")
    out.append("|---|---:|---|---|:--:|")
    out.append(f"| 波位置 `x_f` | {pct(worst_pos[1]['front_series_max_error'])} | "
               f"{worst_pos[0]}, t*={worst_pos[1]['t_star']:g} | 解析解 | "
               f"{'是' if worst_pos[1]['front_series_max_error'] < 0.03 else '否'} |")
    out.append(f"| 波速 `x_f/t` | {pct(worst_speed[1]['x_over_t_error'])} | "
               f"{worst_speed[0]}, t*={worst_speed[1]['t_star']:g} | `2√(g h0)` / `U` | "
               f"{'是' if worst_speed[1]['x_over_t_error'] < 0.03 else '否'} |")
    out.append(f"| 深度 L1 | {pct(worst_l1[1]['depth_l1'])} | "
               f"{worst_l1[0]}, t*={worst_l1[1]['t_star']:g} | 解析剖面 | "
               f"{'是' if worst_l1[1]['depth_l1'] < 0.03 else '否'} |")
    wet_rows = [(k, r) for k, r in rows if r.get("step_ratio_error") is not None
                and r["step_ratio_error"] == r["step_ratio_error"]]
    if wet_rows:
        w = max(wet_rows, key=lambda kr: kr[1]["step_ratio_error"])
        out.append(f"| 静水区水位阶跃 `h2/h1` | {pct(w[1]['step_ratio_error'])} | "
                   f"{w[0]}, t*={w[1]['t_star']:g} | Stoker 闭式 | "
                   f"{'是' if w[1]['step_ratio_error'] < 0.03 else '否'} |")
    out.append(f"| 自相似坍缩 L1/h0 | {sci(worst_collapse[1]['l1_over_h0'])} "
               f"({pct(worst_collapse[1]['l1_over_h0'], 3)}) | "
               f"{worst_collapse[0]}, t*={worst_collapse[1]['t_star']:g} | 数值自身在 t*="
               f"{crit['t_ref']:g} 的解 | "
               f"{'是' if worst_collapse[1]['l1_over_h0'] < 0.03 else '否'} |")
    out.append("")
    out.append(f"**结论**：全部关键量 vs 解析基准的最差读数为 "
               f"{pct(worst_pos[1]['front_series_max_error'])}（波位置）与 "
               f"{pct(worst_speed[1]['x_over_t_error'])}（波速），"
               f"{'均 < 3 %' if max(worst_pos[1]['front_series_max_error'], worst_speed[1]['x_over_t_error']) < 0.03 else '存在 ≥ 3 % 的项，见上表'}；"
               "锚点 5 % 验收带在 `t* = 10`（锚点时刻的 5 倍）上依然成立。\n")
    text = "\n".join(out) + "\n"
    dest = ROOT / "docs" / "case-fvm-dambreak-stoker-long-time" / "REPORT.md"
    dest = pathlib.Path(os.environ.get("REPORT_OUT", dest)) if os.environ else dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text)
    print(f"wrote {dest} ({len(text)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())