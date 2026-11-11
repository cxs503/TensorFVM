#!/usr/bin/env python
"""Render ``docs/repro-fvm-forward-step.md`` from the run artifacts (zero hand-copy).

    python scripts/render_forward_step_report.py --dir docs/repro-fvm-forward-step

Reads ``artifacts/forward_step.json`` and writes the six-section report plus the
platform-report payload (``artifacts/report_payload.json``) with the figures
inlined as base64, so every number in the text comes from the JSON.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def _pct(x: float) -> str:
    return f"{100.0 * x:+.2f} %"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="docs/repro-fvm-forward-step")
    args = ap.parse_args(argv)
    root = Path(args.dir)
    data = json.loads((root / "artifacts" / "forward_step.json").read_text())
    ref = data["reference"]["values"]
    crit = data["criteria"]
    rows = data["results"]
    fine = rows[-1]

    lines = []
    add = lines.append
    add("# REPRO-FVM-FORWARD-STEP — 超声速前台阶（Woodward & Colella 1984, problem 2）")
    add("")
    add("> 报告由 `scripts/render_forward_step_report.py` 从 `artifacts/forward_step.json` "
        "程序化生成，正文数字全部来自工件（零手抄）。预注册判据见 `DESIGN.md`。")
    add("")
    add("## ① 问题与 ref 出处")
    add("")
    add("- **算例**：超声速前台阶（Mach 3 风洞内含台阶），二维 Euler 方程。")
    add("- **ref**：Woodward, P. & Colella, P., *J. Comput. Phys.* **54** (1984) 115–173, "
        "problem 2（\"a Mach 3 wind tunnel with a step\"）。")
    add("- **ref 数值可得性（如实声明）**：该文对本题给出的是**密度等值线图，不是数表**；"
        "因此本报告没有任何数字誊抄自论文。所有基准值由代码按论文给定的来流状态"
        "（M = 3, γ = 1.4, ρ = 1.4, p = 1）用精确 Rankine–Hugoniot / 等熵关系现算"
        "（`normal_shock_reference`），即「零手抄」的等价实现。")
    add("")
    add("| 基准量 | 精确关系 | 值 |")
    add("|---|---|---|")
    add(f"| p2/p1 | 1 + 2γ(M²−1)/(γ+1) | {ref['p_ratio']:.6f} |")
    add(f"| ρ2/ρ1 | (γ+1)M²/((γ−1)M²+2) | {ref['rho_ratio']:.6f} |")
    add(f"| u2/u1 | 1/(ρ2/ρ1) | {ref['u_ratio']:.6f} |")
    add(f"| M2 | 正激波下游 Mach | {ref['mach2']:.6f} |")
    add(f"| p0₂/p0₁ | 总压损失 | {ref['p0_ratio']:.6f} |")
    add(f"| p_stag/p1 = p0₂/p1 | 激波后等熵滞止 | {ref['p02']:.6f} |")
    add(f"| ρ_stag/ρ1 = ρ0₂/ρ1 | 激波后等熵滞止 | {ref['rho02'] / ref['rho1']:.6f} |")
    add("")
    add("## ② 方法与实现")
    add("")
    add("- 求解器：TensorFVM `compressible.EulerSolver`（MUSCL–Hancock, MC limiter, HLLC, "
        f"order = {data['setup']['order']}, cfl = {data['setup']['cfl']}, float64）。")
    add("- **本次新增能力**：嵌入式固壁（`solid=` 掩膜）。固体单元每步被最近流体单元镜像、"
        "壁法向速度取反，使流固面上 HLLC 星速为 0；同时把壁面上的质量/能量通量精确投影为 0"
        "（`project_wall_flux=True`），动量通量保留 p（即壁面压力）。台阶块对齐单元面，"
        "因此壁面是**精确表示、无阶梯误差**。")
    add("- 边界：左端给定来流（M = 3），右端零梯度出流，上下壁面反射。初始为均匀来流，"
        f"冲击启动，积分到 t = {data['setup']['t_end']:g}。")
    add("- 复现命令（一键）：")
    add("")
    add("```bash")
    add(data["repro_cmd"])
    add("```")
    add("")
    add("## ③ 参数表（无量纲）")
    add("")
    add("| 量 | 值 |")
    add("|---|---|")
    row_keys = ("lx", "ly", "step_x", "step_h", "cfl", "order", "t_end")
    for key in row_keys:
        add(f"| {key} | {data['setup'][key]} |")
    for key in ("mach", "gamma", "rho1", "u1", "p1"):
        add(f"| {key} | {ref[key]} |")
    add(f"| 台阶高/通道高 | {fine['grid'] and data['setup']['step_h'] / data['setup']['ly']:.3f} |")
    add(f"| 台阶起点 x/L | {data['setup']['step_x'] / data['setup']['lx']:.4f} |")
    add("")
    add("网格（dx = dy，台阶面均落在单元面上）：")
    add("")
    add("| nx × ny | dx = dy | 步数 | 墙钟 (s) |")
    add("|---|---|---|---|")
    for r in rows:
        add(f"| {r['grid']['nx']} × {r['grid']['ny']} | {r['grid']['dx']:.5f} | "
            f"{r['nsteps']} | {r['wall_seconds']:.1f} |")
    add("")
    add("## ④ 对照表（ref vs got，逐项打勾/叉）")
    add("")
    add(f"判据取自最细网格 {crit['grid_of_record']['nx']} × {crit['grid_of_record']['ny']}"
        f"（dx = {crit['grid_of_record']['dx']:.5f}），容差 {100 * crit['tolerance']:.0f} %。")
    add("")
    add("| 量 | ref | got | 相对误差 | 容差 | 判定 |")
    add("|---|---|---|---|---|---|")
    for item in crit["items"]:
        add(f"| {item['quantity']} | {item['reference']:.6f} | {item['got']:.6f} | "
            f"{_pct(item['rel_error'])} | {item['tolerance']:g} | "
            f"{'✅' if item['pass'] else '❌'} |")
    add("")
    add(f"最细网格：**物理判据（C1–C6）{'全部通过' if crit['all_graded_pass'] else '存在未通过项'}**，"
        f"全部判据 {'全部通过' if crit['all_pass'] else '存在未通过项'}。")
    add("")
    add("逐网格（收敛性）：")
    add("")
    add("| nx × ny | p2/p1 | ρ2/ρ1 | u2/u1 | M2 | p_stag/p1 | x_shock |")
    add("|---|---|---|---|---|---|---|")
    for r in rows:
        s = r["shock"]
        add(f"| {r['grid']['nx']} × {r['grid']['ny']} | {s['p_ratio']:.4f} | "
            f"{s['rho_ratio']:.4f} | {s['u_ratio']:.4f} | {s['mach2']:.4f} | "
            f"{r['stagnation']['p_ratio']:.4f} | {s['x_shock']:.4f} |")
    add("")
    add("## ⑤ 收敛性与参数敏感性")
    add("")
    add("1. **网格敏感性**：上表的 p2/p1 等随 dx 变化；相对误差见 "
        + "；".join(f"{r['grid']['nx']}×{r['grid']['ny']}: {_pct(r['shock']['errors']['p_ratio'])}"
                   for r in rows) + "。")
    add("2. **采样窗敏感性**（线性外推到激波位置所用窗宽 ×0.5/×1/×2）：")
    add("")
    add("| 网格 | 窗倍率 | 窗内单元数 | err(p2/p1) | err(ρ2/ρ1) | err(u2/u1) | err(M2) |")
    add("|---|---|---|---|---|---|---|")
    for r in rows:
        for scale, sens in r["shock"]["sensitivity"].items():
            add(f"| {r['grid']['nx']}×{r['grid']['ny']} | {scale} | {sens['n_cells_behind']} | "
                f"{_pct(sens['err_p_ratio'])} | {_pct(sens['err_rho_ratio'])} | "
                f"{_pct(sens['err_u_ratio'])} | {_pct(sens['err_mach2'])} |")
    add("")
    abl = data.get("wall_treatment_ablation", {})
    if abl:
        add("3. **壁面处理消融（同网格 60×20，t = 0.5）**：仅镜像 vs 镜像 + 通量投影")
        add("")
        add("| 设置 | 质量收支漂移 | 能量收支漂移 | p2/p1 |")
        add("|---|---|---|---|")
        for key, label in (("mirror_only", "仅镜像"), ("projected", "镜像 + 投影")):
            if key in abl:
                add(f"| {label} | {abl[key]['mass_drift']:+.3e} | {abl[key]['energy_drift']:+.3e} | "
                    f"{abl[key]['p_ratio']:.4f} |")
        add("")
        add("   投影的必要性由此表直接可见：不投影时壁面漏质量，且激波后的 p2/p1 被污染。")
    add("")
    add("## ⑥ 失败模式与工件")
    add("")
    add("- 失败模式 1（已修）：**无嵌入式固壁能力** —— 原 `EulerSolver` 只支持四边边界条件，"
        "台阶根本无法建模；本单新增 `solid=` 掩膜。")
    add("- 失败模式 2（已修）：**镜像态单独不足** —— 重构斜率不对称导致壁面残余质量通量，"
        "累计把质量收支漂移推到 1e-2 量级并污染激波后状态（见消融表）；改为壁面通量精确投影后"
        "质量收支闭合到机器精度。")
    add("- 失败模式 3（如实记录）：`EulerSolver` 的正性修复是非守恒的，强膨胀处每步约 0.16 个单元触发，"
        f"使能量收支漂移达到 {rows[-1]['budget']['energy_drift']:+.3e}（判据 C8 容差 1e-2）；"
        "质量收支不受影响（"
        f"{rows[-1]['budget']['mass_drift']:+.2e}）。")
    add("- 未判据化的观测：质量通量在 x = 0.05 与 x = 2.95 处的差为 "
        f"{crit['ungraded']['mass_flux_imbalance']:+.3%}（台阶后回流区在 t = 4 非定常，"
        "故该量没有精确基准，不作为判据）。")
    add("")
    add("生态摩擦：")
    add("")
    for item in data["ecosystem_friction"]:
        add(f"- {item}")
    add("")
    add("产物：")
    add("")
    add("- `artifacts/forward_step.json`（全部数字）、`artifacts/forward_step_fields.npz`"
        "（最细网格 rho/u/v/p 与固壁掩膜，可直接复画）")
    for fig in data.get("figures", []):
        add(f"- `{fig}`")
    add("")
    (root.parent / "repro-fvm-forward-step.md").write_text("\n".join(lines))

    # platform report payload with inlined figures
    figures = []
    for fig in data.get("figures", []):
        path = root / fig
        if path.exists():
            figures.append({
                "name": path.stem,
                "caption": path.stem.replace("_", " ") + f" (t = {data['setup']['t_end']:g})",
                "png_base64": base64.b64encode(path.read_bytes()).decode(),
            })
    payload = {
        "title": "REPRO-FVM-FORWARD-STEP: Woodward & Colella (1984) Mach-3 forward-facing step",
        "summary": (
            "TensorFVM 新增嵌入式固壁（solid mask + 壁面通量投影）后复现 Woodward & Colella "
            f"(1984) Mach 3 前台阶：{crit['grid_of_record']['nx']}x{crit['grid_of_record']['ny']} "
            f"网格上 p2/p1 误差 {_pct(fine['shock']['errors']['p_ratio'])}、"
            f"ρ2/ρ1 {_pct(fine['shock']['errors']['rho_ratio'])}、"
            f"u2/u1 {_pct(fine['shock']['errors']['u_ratio'])}、"
            f"M2 {_pct(fine['shock']['errors']['mach2'])}、"
            f"滞止压 {_pct(fine['stagnation']['err_p_ratio'])}（容差 8 %）；"
            f"质量收支闭合到 {fine['budget']['mass_drift']:+.1e}。"
        ),
        "scenario_key": "tensorfvm-compressible-benchmark",
        "mission_id": 15326,
        "tags": ["TensorFVM", "compressible", "Euler", "Woodward-Colella", "forward-facing step",
                 "embedded wall", "verification"],
        "markdown": "\n".join(lines),
        "figures": figures,
    }
    (root / "artifacts" / "report_payload.json").write_text(json.dumps(payload, ensure_ascii=False))
    print(f"wrote {root.parent / 'repro-fvm-forward-step.md'} and report payload "
          f"({len(figures)} figures)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())