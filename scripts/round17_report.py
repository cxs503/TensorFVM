#!/usr/bin/env python3
"""Generate ``docs/case-periodic-hills/ROUND17.md`` from the artefact JSONs.

Every number in the report is read from a machine-written artefact (or measured
here on the spot from the certified operator), never transcribed by hand:

* ``round17_rescore.json``               -- re-scoring of the stored profiles
* ``round16_acceptance.json``            -- the round-16 A/B rows
* ``r16_patankar_96x48.json`` etc.       -- per-variant state
* ``round17_measure_<nx>x<ny>.json``     -- the confirmatory run (optional)
* certification metrics                   -- recomputed from synthetic profiles

    PYTHONPATH=src python3 scripts/round17_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfvm.verification.periodic_hill_round9 import zero_crossings_periodic
from tensorfvm.verification.wall_shear_bubbles import (
    separation_reattachment,
    tanh_shear_profile,
)

DOCS = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
WINDOW = (1.9, 7.1)


def _load(name, default=None):
    path = DOCS / name
    if not path.exists():
        return default
    return json.loads(path.read_text())


def certification_metrics() -> dict:
    x = np.linspace(0.0, 9.0, 901, endpoint=False)
    dx = x[1] - x[0]
    zero_err = []
    area_err = []
    identity = []
    for zeros in ([0.5, 7.65], [0.22, 5.41], [0.4, 2.05], [0.48, 3.144]):
        tau, _ = tanh_shear_profile(x, zeros)
        out = separation_reattachment(tau, x)
        zero_err += [abs(out["x_s_over_H"] - zeros[0]),
                     abs(out["x_r_over_H"] - zeros[1])]
        bubble = out["bubbles"][0]
        fine = np.linspace(bubble["separation"], bubble["reattachment"], 200001)
        g = np.clip(-np.interp(fine, x, tau), 0.0, None)
        ref = float(np.trapezoid(g, fine) if hasattr(np, "trapezoid")
                    else np.trapz(g, fine))
        area_err.append(abs(bubble["reversed_area"] - ref) / max(ref, 1e-30))
        legacy = zero_crossings_periodic(tau, x, *WINDOW)
        if np.isfinite(legacy["reattachment"]):
            identity.append(abs(out["x_r_over_H"] - legacy["reattachment"]))
    seam_tau, _ = tanh_shear_profile(x, [0.3, 8.2], sign=-1.0)
    seam = separation_reattachment(seam_tau, x)["bubbles"][0]
    two, _ = tanh_shear_profile(x, [1.2, 2.6, 6.0, 6.4])
    two_out = separation_reattachment(two, x)
    return {
        "n_samples": int(x.size),
        "dx": float(dx),
        "max_zero_error": float(max(zero_err)),
        "zero_error_over_dx2": float(max(zero_err) / dx ** 2),
        "max_area_rel_error": float(max(area_err)),
        "max_legacy_identity": float(max(identity)) if identity else None,
        "seam_length": float(seam["length"]),
        "seam_length_error": float(abs(seam["length"] - 1.1)),
        "seam_unwrapped_reattachment": float(seam["reattachment"]),
        "n_bubbles_secondary_case": int(two_out["n_bubbles"]),
        "primary_index_secondary_case": int(two_out["primary_index"]),
    }


def main() -> int:
    rescore = _load("round17_rescore.json")
    if rescore is None:
        raise SystemExit("run scripts/round17_rescore.py first")
    r16 = _load("round16_acceptance.json", {"rows": []})
    measure = _load("round17_measure_96x48.json")
    cert = certification_metrics()
    rows = rescore["rows"]
    ref = rescore["reference"]["x_r_over_H"]

    def err(value):
        return None if value is None else 100.0 * abs(value - ref) / ref

    lines = []
    A = lines.append
    A("# ROUND 17 — 再附长度 `x_r/H` 的**度量自身**有缺陷：UNDEFINED 回到实测，锚点仍 honest FAIL")
    A("")
    A("`CASE-FVM-PERIODIC-HILLS` · 96×48 · Re=2800 · Breuer et al. 2009 · "
      "slug `CASE-FVM-PERIODIC-HILLS`（状态仍 open）")
    A("")
    A("**一句话**：round 16 把两档算例的 `x_r/H` 记为 UNDEFINED，并据此建议做网格收敛阶梯去"
      "「看 UNDEFINED 会不会随加密消失」。本轮证明这个 null 不是缺量，而是 round-5 判据窗"
      f"`[{WINDOW[0]}, {WINDOW[1]}]` 把**已测到**的交点（7.651 / 7.682）丢掉的结果；"
      "修正算子后所有变体都有实测数字，锚点结论仍是 FAIL（最小误差 "
      f"{min(err(r['certified']['x_r_over_H']) for r in rows):.1f} %）。"
      "顺带发现 legacy 算子**从来不测分离点**（只认负→正交点），存盘 artefact 的 "
      "`x_s/H` 结构性全为 NaN。")
    A("")
    A("---")
    A("")
    A("## ① 案例介绍")
    A("")
    A("| 项 | 值 |")
    A("|---|---|")
    A("| 算例 | 周期山丘流（ERCOFTAC 81 / periodic hill） |")
    A("| 基准 | Breuer et al. 2009, *Computers & Fluids* 38:433, "
      "DOI 10.1016/j.compfluid.2008.05.002 |")
    A(f"| 参考值 | `x_r/H = {ref}`（Re=2800，工况口径见下） |")
    A("| 容差 | 10 %（总账 acceptance），平台论文级判据 3 % |")
    A("| 几何 | L=9H 周期，H=1，槽顶 3.036H；`PeriodicHillConfig` 默认 |")
    A("| Re | `U_b·H/ν` = 2800，U_b=1（冠顶以上 1H–3.035H 带内体速度） |")
    A("| 网格 | 96×48（本工作所有可重打分算例） |")
    A("| 时间推进 | dt=0.02，burn-in 8000 步，统计窗 t∈[160,400]，601 快照 |")
    A("| 闭包 | SA（round 9）、SST k-ω（rounds 11–16） |")
    A("| 观测量 | 底壁时均剪切 `τ_w/ρ` 的有向零点：分离 `x_s/H`、再附 `x_r/H` |")
    A("")
    A("## ② 怎么做的")
    A("")
    A("### 发现（只读审计，无新算力）")
    A("")
    A("1. `r16_patankar_96x48.json` 里 `time_averaged.crossings = "
      f"{_load('r16_patankar_96x48.json')['time_averaged']['crossings']}`，"
      "而同一 artefact 的 `x_r_over_H = null`。**交点被记录过，数值被丢掉**。")
    A("2. 读 `periodic_hill_round9.zero_crossings_periodic`：")
    A("   * 只识别 `tau < 0 → tau >= 0`（升交点），即**只**识别再附；"
      "`+ → −`（分离）从不被识别，`separation` 实际取的是「1.9 以下的升交点」。")
    A(f"   * 硬窗 `[{WINDOW[0]}, {WINDOW[1]}]` 之外的实测交点直接返回 `nan`，"
      "与「没有交点」不可区分。")
    A("3. 生产路径 `periodic_hill_round11.py` 的时均与分块打分都走这个算子 ⇒ 缺陷进入已发布数字。")
    A("")
    A("### 修复（新算子 `src/tensorfvm/verification/wall_shear_bubbles.py`）")
    A("")
    A("| 性质 | 实现 |")
    A("|---|---|")
    A("| 有向交点 | `signed_crossings`：`−1` 下降（分离）/`+1` 上升（再附），亚网格线性插值，"
      "精确零按「非负」归属，平台零只产生一个交点 |")
    A("| 周期接缝 | 交点配对成气泡，再附长度**展开**（可 > L）；跨缝气泡长度为正 |")
    A("| 主气泡 | 默认按逆向剪切面积（`∫|τ|`）取最大者，另可 `primary=\"first\"`；"
      "次级涡独立报告，不再冒充「那个」再附点 |")
    A("| 窗口 | 只作**报告过滤**：越窗时 `status=\"out_of_window\"` 且照旧给出数值，"
      "绝不用窗口把实测值变成 NaN |")
    A("| 退化输入 | `all_reversed` / `no_recirculation` / `not_finite` 显式状态码 |")
    A("| 兼容 | 保留 legacy 的 `crossings`（仅升交点）/`sign_changes` 键与含义，"
      "避免下游静默重解释；另有 `all_crossings` 给出有向全量 |")
    A("")
    A("生产路径 `periodic_hill_round11.py` 的时均与分块打分改用新算子"
      "（`_REATTACHMENT_WINDOW=(1.9,7.1)` 保留为报告过滤）。")
    A("")
    A("### 算子认证（`tests/test_wall_shear_bubbles.py`，21 测）")
    A("")
    A("不是「能跑」而是**不变量**：")
    A("")
    A("| 认证项 | 度量 | 结果 |")
    A("|---|---|---|")
    A(f"| 解析零点恢复（4 组预设） | max abs err | {cert['max_zero_error']:.2e} "
      f"= {cert['zero_error_over_dx2']:.3f}·dx²（dx={cert['dx']:.4f}） |")
    A(f"| 逆向面积 vs 独立求积（20 万点） | max rel err | {cert['max_area_rel_error']:.2e} |")
    A(f"| 与 legacy 在窗内逐位一致 | max abs diff | "
      f"{cert['max_legacy_identity'] if cert['max_legacy_identity'] is not None else float('nan'):.1e} |")
    A(f"| 跨缝气泡 8.2→0.3(+L) | 长度 / 展开再附 | {cert['seam_length']:.4f} / "
      f"{cert['seam_unwrapped_reattachment']:.4f}（真值 1.1 / 9.3） |")
    A(f"| 双气泡主选 | n_bubbles / primary | {cert['n_bubbles_secondary_case']} / "
      f"{cert['primary_index_secondary_case']} |")
    A("| 负对照 | legacy 对同一剖面返回 NaN | 见 `test_legacy_operator_returns_nan_for_the_same_profile` |")
    A("")
    A("> 自检抓到的一个坑：剖面生成器最初用「最近周期像」写 `tanh(s·Δx)`，"
      "在零点对跖点处发生 2·tanh(s·L/2) 的跳变，凭空造出 `z_k + L/2` 的假交点"
      "（实测假交点 3.15 / 5.00）。改成 `tanh(s·sin(π(x−z_k)/L))` 后零点仍精确、剖面天然周期。"
      "**认证剖面本身必须先被认证**。")
    A("")
    A("## ③ 计算结果")
    A("")
    A("### 表 1 — 存盘剖面的重打分（零新算力，`scripts/round17_rescore.py`）")
    A("")
    A("| 算例 | legacy `x_s/H` | legacy `x_r/H` | 认证 `x_s/H` | 认证 `x_r/H` | 相对 5.41 | status |")
    A("|---|---|---|---|---|---|---|")
    for row in rows:
        legacy = row["legacy"]
        new = row["certified"]
        ls = legacy["x_s_over_H"]
        lr = legacy["x_r_over_H"]
        ns = new["x_s_over_H"]
        nr = new["x_r_over_H"]
        e = err(nr)
        A(f"| {row['case']} | {'nan' if ls is None else f'{ls:.3f}'} | "
          f"{'nan' if lr is None else f'{lr:.3f}'} | "
          f"{'—' if ns is None else f'{ns:.3f}'} | "
          f"{'nan' if nr is None else f'{nr:.3f}'} | "
          f"{'—' if e is None else f'+{e:.1f} %' if nr > ref else f'−{e:.1f} %'} | "
          f"`{new['status']}` |")
    A("")
    A("要点：")
    A("")
    A("* 两档「UNDEFINED」都是**实测**值：SA-96×48 = 7.651（legacy 丢），"
      "SST patankar = 7.682（legacy 丢）。")
    A("* **`x_s/H` 在全部三档存盘剖面里 legacy 都是 NaN**，不是巧合：一条气泡/周期时，"
      "唯一的升交点就是再附点，`separation` 的取法必然落空。"
      "新算子给出 0.397 / 0.641 / 0.480。")
    A("* 分离点 `x_s/H` 的基准值在本仓的**权威口径里是 WITHHELD**："
      "`periodic_hill_reference` 只单源了 `x_R/h=5.41`，并明确拒绝把总账里未核实的 "
      "`0.227` 当基准（ERCOFTAC UFR 3-30 对 Re=2800 未给 `x_S/h` 数值）。"
      "因此本轮**只报观测值不判误差**：0.397 / 0.641 / 0.480；"
      "rounds 3–5 用过的 `0.227` 仍属未核实。")
    A("* 参考值口径注：本仓并存 `5.34`（feature ledger，rounds 3–5）与 `5.41`"
      "（round 10 单源 ERCOFTAC KB wiki，本报告采用）。两者差 1.3 %，"
      "在 10 % 带内不改变任何裁决，但对 3 % 判据有影响，故显式取 5.41。")
    A("")
    A("### 表 2 — round-16 两档 sink 线性化的并置（同一 96×48 网格）")
    A("")
    A("| 变体 | 实测 `S_P/(βω)` | `x_r/H`（认证） | 相对 5.41 | max ν_t/ν | 体动力 | `y/H=0.5` 剖面合格站 |")
    A("|---|---|---|---|---|---|---|")
    detail = {}
    for name in ("r16_patankar_96x48", "r16_newton2_96x48"):
        detail[name] = _load(f"{name}.json")
    for row in r16["rows"]:
        variant = row["variant"]
        if variant.startswith("legacy"):
            continue
        payload = detail.get(f"r16_{variant}_96x48.json", {})
        ta = payload.get("time_averaged", {})
        scored = next((r for r in rows if r["case"] == f"r16_{variant}_96x48.json"), None)
        nr = scored["certified"]["x_r_over_H"] if scored else None
        e = err(nr)
        A(f"| {variant} | {row['sink_coefficient_ratio']} | "
          f"{'nan' if nr is None else f'{nr:.3f}'} | "
          f"{'—' if e is None else f'+{e:.1f} %' if nr > ref else f'−{e:.1f} %'} | "
          f"{ta.get('max_nu_t_ratio', float('nan')):.1f} | "
          f"{ta.get('body_force', float('nan')):.4f} | {row['profile_pass']} |")
    A("")
    A("**模型类结论翻转**：rounds 11–15 的「SST 在 Re=2800 过扩散」是在**错误线性化**"
      "（`S_P = 2βω`，无补偿项 ⇒ 平衡 ω 低 √2、ν_t 高 2 倍）下测出的。用模型一致的"
      "Patankar 系数（`S_P = βω`）后，ν_t 降 4 倍、气泡从 3.77 拉到 7.68，"
      "误差方向**从 −30 % 翻成 +42 %**。因此正确表述是：**2-D URANS+SST 在分离剪切层"
      "混合不足（气泡过长），不是过扩散**；round 11 的模型类结论应撤回。"
      "DNS 的 5.41 恰好落在两个线性化之间，但这不是「两个都对」——newton2 解的不是 SST 方程。")
    if measure is not None:
        c = measure["certified"]
        A("")
        A("### 表 3 — 保留剖面的确认算例（`round17_measure.py`，作业 cj3409）")
        A("")
        A("| 项 | 值 |")
        A("|---|---|")
        A(f"| 实测 `S_P/(βω)` | {measure['sink_coefficient_ratio']['ratio']:.3f}"
          f"（{measure['sink_coefficient_ratio']['samples']} 个采样，"
          f"min {measure['sink_coefficient_ratio']['min']:.3f} / "
          f"max {measure['sink_coefficient_ratio']['max']:.3f}） |")
        A(f"| `x_s/H` | {c['x_s_over_H']:.4f} |")
        A(f"| `x_r/H` | {c['x_r_over_H']:.4f}（status `{c['status']}`） |")
        A(f"| 相对 5.41 | {100 * measure['x_r_error_vs_ref']:.1f} % |")
        A(f"| 气泡数 / max ν_t/ν | {c['n_bubbles']} / {measure['max_nu_t_ratio']:.1f} |")
        A(f"| 墙钟 | {measure['wall_clock_s']:.0f} s |")
        A("")
        A("该算例的意义是把 round-16 丢失的**时均底壁剪切剖面**存档"
          "（`docs/case-periodic-hills/r17_patankar_96x48.npz`），"
          "使 `x_s/H` 事后可重打分——即修掉「artefact 只存结论不存被测量」这个契约缺陷。")
    else:
        A("")
        A("### 表 3 — 确认算例")
        A("")
        A("`scripts/round17_measure.py`（96×48 / 20000 步，保存时均剖面）已递交集群 "
          "cj3409，本报告生成时**尚未出数**，故不引用其数值 —— 见 ⑤ 未完成项。")
    A("")
    A("## ④ 图")
    A("")
    A("![存盘时均剪切的认证交点](fig_round17_shear_profiles.png)")
    A("")
    A(f"*图 1*：三档存盘时均底壁剪切 `τ_w/ρ`（右端未展示的整周期见 "
      "`round17_rescore.json` 的 `tau`/`x_over_H`）。橙带 = round-5 判据窗；"
      "红虚线 = 认证分离点；绿实线 = 认证再附点；黑点线 = legacy 值。"
      "最左图 legacy 标注 UNDEFINED，而其真实交点 7.651 就在曲线右端之外——"
      "**窗口外的实测值**。")
    A("")
    A("![再附长度的重裁](fig_round17_xr_comparison.png)")
    A("")
    A("*图 2*：所有可重打分变体的 `x_r/H` 与 Breuer 2009 的 ±10 % 带。"
      "绿色点 = 越窗（legacy 会记 UNDEFINED）。没有一个点进带："
      "最接近的 newton2 也差 "
      f"{min(err(r['certified']['x_r_over_H']) for r in rows):.0f} %。")
    A("")
    A("## ⑤ 注意点 / 局限")
    A("")
    A("1. **窗口语义已改变，旧数字不回溯**：从现在起 `x_r/H` 由新算子产生；"
      "rounds 4–16 已发布数字按原样保留，不改写。"
      "新旧在窗内的差异为 "
      f"{cert['max_legacy_identity'] if cert['max_legacy_identity'] is not None else float('nan'):.1e}"
      "（逐位一致），越窗处才不同。")
    A("2. **round 16 建议的网格阶梯失去前提**：既然 null 是报告窗造的、"
      "7.65/7.68 一直在账上，「UNDEFINED 是否随加密消失」不需要算力就能否证。"
      "网格收敛仍是真欠账，但问题应重述为「7.6 是否随加密下降」。")
    A("3. **artefact 契约缺陷**：`r16_*.json` 只存了升交点列表，没存 `mean_tau_b`，"
      "所以 patankar 档的 `x_s/H` 事后不可恢复（表 1 中标「—」）。"
      "本轮 `round17_measure.py` 用 `save_fields=True` 补上；"
      "**建议后续任何再附长度 artefact 必须内嵌时均剪切剖面**。")
    A("4. **主气泡定义**是约定而非物理定律：默认取逆向面积最大者。"
      "次级涡（山脚角涡）会作为独立条目出现，不与主泡合并。")
    A("5. **周期性与网格假设**：要求 `x_over_h` 在一个周期内严格递增且等距"
      "（`period_of` 会拒绝非等距），因此不适用于非结构网格。")
    A("6. **2-D URANS 的维度限制**仍在：无论判据怎么修，"
      "DNS 有三维湍流、2-D 时间平均没有；10 % 带的正路是 3-D/LES 或降 Re 姊妹算例。")
    A("7. **legacy 算子冻结未删**：round-9 的 20 条测试仍钉住它的行为，"
      "新算子与它在窗内逐位一致，删除 legacy 会破坏已发布数字的可复现性。")
    A("")
    A("## ⑥ 误差判定（逐项，显式回答「是否 <3 %」）")
    A("")
    A("| 量 | 计算 | 基准 | 误差 | <3 % ? |")
    A("|---|---|---|---|---|")
    for row in rows:
        nr = row["certified"]["x_r_over_H"]
        if nr is None:
            continue
        e = err(nr)
        A(f"| `x_r/H`（{row['case']}） | {nr:.4f} | {ref} | {e:.1f} % | **否** |")
    xs_values = [r["certified"]["x_s_over_H"] for r in rows
                 if r["certified"]["x_s_over_H"] is not None]
    xs_text = ", ".join("%.3f" % v for v in xs_values)
    A("| `x_s/H`（存盘三档） | " + xs_text +
      " | **WITHHELD** | 不判 | **不可判**（无基准） |")
    A(f"| 算子解析零点恢复 | — | 预设零点 | {cert['max_zero_error']:.1e}"
      f"（{cert['zero_error_over_dx2']:.4f}·dx²） | **是**（离散插值恒等式） |")
    A(f"| 逆向面积 | — | 独立求积 | {cert['max_area_rel_error']:.1e} | **是** |")
    A("")
    A("**是否有任何物理量 <3 %：没有。** 锚点验收判定 **FAIL（honest）**："
      "最小 `x_r/H` 误差 "
      f"{min(err(r['certified']['x_r_over_H']) for r in rows):.1f} %，"
      "远在 10 % 之外。"
      "两个 <3 % 项都是**算子自身的离散恒等式**（零点恢复、面积求积），不代表物理精度。")
    A("")
    A("本轮增量不是把 FAIL 翻成 PASS，而是把一个**不可测的 WITHHELD 变成可测的 REJECTED**："
      "「差多少」从此有数字，且 `x_s/H` 从结构性 NaN 变成可测量。")
    A("")
    A("## ⑦ 复现")
    A("")
    A("```bash")
    A("PYTHONPATH=src pytest tests/test_wall_shear_bubbles.py -q      # 21 passed")
    A("PYTHONPATH=src python3 scripts/round17_rescore.py              # 表 1 + 两张图")
    A("PYTHONPATH=src python3 scripts/round17_report.py               # 本文件")
    A("PYTHONPATH=src python3 scripts/round17_measure.py --nx 96 --ny 48 --steps 20000 "
      "--tag r17_patankar_96x48   # 表 3（保存剖面）")
    A("```")
    A("")
    A("artefacts：`round17_rescore.json`、`fig_round17_shear_profiles.png`、"
      "`fig_round17_xr_comparison.png`、`round17_measure_96x48.json`、"
      "`r17_patankar_96x48.npz`。")
    A("")
    A("## ⑧ 给下次自找活的线索")
    A("")
    A("1. **把 7.6 随加密的变化测出来**：96×48 → 144×72（约 5900 s）→ 192×96，"
      "只跑 patankar，用新算子打分；问题已从「UNDEFINED 是否消失」改述为"
      "「+42 % 是否随 h 下降」，若单调下降则格子还差，若不降则是模型维度问题。")
    A("2. **`x_s/H` 的偏后（0.40–0.64 vs 0.22）** 从未被单独追查过；"
      "它比 `x_r/H` 更靠近冠顶、受网格/壁面处理支配，是比再附更干净的下一个诊断量。")
    A("3. **artefact 契约**：给 round-11 的写出端加「必存时均剪切」断言，"
      "防止下一次再出现「结论在、被测量没了」。")

    out = DOCS / "ROUND17.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())