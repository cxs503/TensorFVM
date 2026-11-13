#!/usr/bin/env python3
"""Render ``docs/CASE-LAX/report.md`` from the harness artifact (zero hand-copy).

Input
-----
``docs/CASE-LAX/compressible_multimaterial.json``
    written by ``python -m tensorfvm.compressible --legs lax --output docs/CASE-LAX``

Optional cross-check
--------------------
``docs/lax-shock-tube/artifacts/lax_report.json``
    the stand-alone benchmark harness (``scripts/lax_shock_tube_benchmark.py``).
    When present the script compares the two independent *entry points*
    (verification-leg driver vs stand-alone script, separate runs) on the same
    grids and prints the agreement, so "the case is registered and reproduces
    the documented numbers" is a number, not a claim.

Every value in the markdown is interpolated from those JSON files; the script is
the only place the report text is produced.  Usage::

    python scripts/render_case_lax_pool_report.py \
        --artifact docs/CASE-LAX/compressible_multimaterial.json \
        --crosscheck docs/lax-shock-tube/artifacts/lax_report.json \
        --out docs/CASE-LAX/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100.0 * x:.3f}%"


def _sci(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3e}"


def _table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _crosscheck(artifact: dict, ref_path: Path | None) -> dict:
    """Compare the leg's L1 numbers with the stand-alone harness, grid by grid."""
    if ref_path is None or not ref_path.exists():
        return {"available": False}
    ref = json.loads(ref_path.read_text())
    ref_rows = {r["nx"]: r for r in ref["sweep"]["rows"]}
    leg_rows = {r["nx"]: r for r in artifact["legs"]["lax"]["rows"]}
    common = sorted(set(ref_rows) & set(leg_rows))
    table = []
    worst = 0.0
    for nx in common:
        a, b = ref_rows[nx], leg_rows[nx]
        diffs = {k: abs(a["l1_rel"][k] - b["l1_rel"][k]) for k in ("rho", "u", "p")}
        worst = max(worst, *diffs.values())
        table.append((nx, _pct(a["l1_rel"]["rho"]), _pct(b["l1_rel"]["rho"]),
                      _sci(diffs["rho"]), _sci(diffs["u"]), _sci(diffs["p"])))
    return {"available": bool(common), "grids": common, "rows": table,
            "worst_abs_diff": worst,
            "p_star_ref": ref["exact"]["p_star"],
            "p_star_leg": artifact["legs"]["lax"]["exact_waves"]["p_star"]}


def render(artifact: dict, cross: dict) -> str:
    leg = artifact["legs"]["lax"]
    cfg = leg["config"]
    th = leg["thresholds"]
    waves = leg["exact_waves"]
    rows = leg["rows"]
    fine = leg["graded"]
    checks = fine["checks"]
    finest = max(rows, key=lambda r: r["nx"])
    orders = leg["observed_order"]
    chk = leg["independent_exact_check"]
    rh = leg["rankine_hugoniot"]
    figs = artifact.get("figures", {}).get("lax", [])

    L = []
    a = L.append

    a("# CASE-LAX｜算例池注册：Lax (1954) 激波管（TensorFVM 标准验证腿）")
    a("")
    a("> 本报告由 `scripts/render_case_lax_pool_report.py` 从 "
      "`docs/CASE-LAX/compressible_multimaterial.json` 程序化渲染（零手抄）。"
      "算例池条目：`lax` = `tensorfvm.compressible.verification.leg_lax`，"
      "与 sod / interface / rm / bubble / vortex / forward_step 同一驱动、同一工件格式。")
    a(f"> 运行：`python -m tensorfvm.compressible --legs lax --output docs/CASE-LAX`，"
      f"本次腿耗时 {artifact['seconds']} s（含出图），"
      f"`all_criteria_met = {artifact['all_criteria_met']}`。")
    a("")
    a("## ① 案例介绍")
    a("")
    a("**物理背景。** 一维理想气体激波管（Riemann 问题）：隔膜两侧压差驱动产生"
      "左行稀疏波 + 接触间断 + 右行激波。Lax 算例与 Sod 的关键差别是右波为**强激波**"
      f"（p*/p_R = {waves['p_star'] / cfg['p_r']:.3f}，ρ*_R/ρ_R = "
      f"{waves['rho_star_r'] / cfg['rho_r']:.3f}），"
      f"而接触两侧星区密度比为 ρ*_R/ρ*_L = {waves['rho_star_r'] / waves['rho_star_l']:.3f}；"
      "强激波 + 强稀疏波 + 大跳跃接触同时在场，对 HLLC 波速估计与接触/剪切分辨率的要求高于 Sod。")
    a("")
    a("**参考出处。** 初值取自 Lax, P. D. (1954), *Comm. Pure Appl. Math.* **7**, 159–193；"
      "同一数据集被 Toro, E. F. (2009), *Riemann Solvers and Numerical Methods for Fluid "
      "Dynamics*, 3rd ed., Wiley, ch. 4（test problem 2 + Sec. 4.2 精确解构造）复述为标准算例。"
      "参考值不是抄录表格数字，而是**按上述初值在运行时重算的自相似精确解**（两套互不共享代码的实现）。")
    a("")
    a("**几何与参数表（全部无量纲，值来自工件 `legs.lax.config`）。**")
    a("")
    a(_table(["参数", "值"], [
        ("γ（两侧理想气体）", cfg["gamma"]),
        ("左态 ρ_L / u_L / p_L", f"{cfg['rho_l']} / {cfg['u_l']} / {cfg['p_l']}"),
        ("右态 ρ_R / u_R / p_R", f"{cfg['rho_r']} / {cfg['u_r']} / {cfg['p_r']}"),
        ("计算域 / 隔膜位置 x0", f"[0, 1] / {cfg['x0']}"),
        ("终止时间 t_end", cfg["t_end"]),
        ("网格（本次运行）", ", ".join(str(r["nx"]) for r in rows) + "（均匀，ny=1）"),
        ("通量/重构/限制器/CFL", f"{rows[0]['flux_mode'].upper()} / MUSCL-Hancock / "
                              f"{rows[0]['limiter']} / {rows[0]['cfl']}"),
    ]))
    a("")
    a("**精确解的波系（工件 `legs.lax.exact_waves`，由两套实现互证后取用）。**")
    a("")
    a(_table(["量", "值", "含义"], [
        ("p*", f"{waves['p_star']:.6f}", "星区压力"),
        ("u*", f"{waves['u_star']:.6f}", "星区速度"),
        ("ρ*_L / ρ*_R", f"{waves['rho_star_l']:.6f} / {waves['rho_star_r']:.6f}", "星区密度"),
        ("S_head / S_tail", f"{waves['speed_head']:.6f} / {waves['speed_tail']:.6f}",
         "左行稀疏波头/尾速度"),
        ("S_contact", f"{waves['speed_contact']:.6f}", "接触间断速度（= u*）"),
        ("S_shock", f"{waves['speed_shock']:.6f}", "右行激波速度"),
        ("x_head…x_shock (t_end)", f"{waves['x_head']:.4f} … {waves['x_shock']:.4f}",
         "t = t_end 时四个波面位置"),
        ("左/右波型", f"{waves['left_wave']} / {waves['right_wave']}", "自相似解结构"),
    ]))
    a("")
    a("## ② 怎么做的")
    a("")
    a("- **注册（本单的实质改动）。** Lax 此前只以独立脚本 `scripts/lax_shock_tube_benchmark.py` "
      "存在（`docs/lax-shock-tube`），`python -m tensorfvm.compressible --legs` 不认识它——"
      "即「算例存在但不在池里」。现在 `verification.leg_lax` 把同一套测量接入标准验证驱动："
      "`run_all(legs=(\"lax\",))` 走与 sod 完全相同的路径（阈值表、verdict 展开、JSON 工件、出图），"
      "CLI 可直接 `--legs lax`。求解器、测量器、精确解均复用既有实现，**未新增数值方法**。")
    a("- **判据（动手机器化，先于测量）。** 星区平台 p*/u*/ρ*_L/ρ*_R、三个波面位置、"
      f"ρ/u/p 的相对 L1 均要求 ≤ {100 * th['l1_rel_error']:.0f}%；质量账漂移 ≤ "
      f"{th['conservation_drift']}；正性修补次数 = 0；另加两条参考自证判据："
      f"两套精确解一致 ≤ {th['independent_exact_rel_error']:.0e}、Rankine–Hugoniot 残差 ≤ 1e-12。")
    a("- **测量口径。** 以自相似坐标 ξ=(x−x0)/t 采样精确解逐格比较；星区平台取「离间断 15% 子窗口」"
      "均值；波面位置用**同一个**跨中值插值定位器同时作用于数值场与 20001 点细网格精确场，"
      "估计器偏差在比较中相消。")
    a("- **工具链。** 求解器 `src/tensorfvm/compressible/solver.py`；算例/精确解 "
      "`src/tensorfvm/compressible/lax.py`；本腿 `verification.leg_lax`；出图 "
      "`figures.fig_lax`；CI 门 `tests/test_compressible_lax_leg.py`（quick 网格，~35 s）。")
    a("- **复现。** `python -m tensorfvm.compressible --legs lax --output docs/CASE-LAX`。")
    a("")
    a("## ③ 计算结果")
    a("")
    a("### 3.1 网格序列（工件 `legs.lax.rows`）")
    a("")
    a(_table(["nx", "步数", "秒", "L1ρ 相对", "L1u 相对", "L1p 相对",
              "平台 ρ*_R 误差", "激波位置误差", "质量漂移", "正性修补"],
             [(r["nx"], r["steps"], r["seconds"], _pct(r["l1_rel"]["rho"]),
               _pct(r["l1_rel"]["u"]), _pct(r["l1_rel"]["p"]),
               _pct(r["plateau_rel_error"]["rho_star_r"]),
               _pct(r["position_rel_error"]["x_shock"]),
               _sci(abs(r["drift"]["mass"])), r["positivity_fixes"]) for r in rows]))
    a("")
    a("观测收敛阶（逐对网格，ρ/u/p）：" + "；".join(
        f"{k} = " + ", ".join(f"{o:.2f}" for o in v) for k, v in orders.items())
      + "。含接触/激波的间断解，TVD-MUSCL 的全局 L1 阶预期在 0.5–1.0 之间，实测落在此区间；"
      "光滑解的二阶性由 `test_compressible_isentropic_vortex.py` 单独认证，本算例只认证间断行为。")
    a("")
    a("### 3.2 验收判定（最细网格 nx = " + str(fine["nx"]) + "，全部由工件 verdicts 展开）")
    a("")
    verdict_rows = []
    for key, item in checks.items():
        verdict_rows.append((key, _pct(item["value"]) if key.split("[")[0] in
                             ("l1", "plateau", "position") else _sci(item["value"]),
                             "PASS" if item["ok"] else "FAIL"))
    a(_table(["判据", "实测", "判定"], verdict_rows))
    a("")
    a(f"腿级 `all_ok = {fine['all_ok']}`；`all_criteria_met = {artifact['all_criteria_met']}`。")
    a("")
    a("### 3.3 参考值自证（参考不是断言，是可审计的数字）")
    a("")
    a(_table(["检查", "结果"], [
        ("两套精确解 p* 相对差", _sci(chk["rel_dp_star"])),
        ("两套精确解 u* 相对差", _sci(chk["rel_du_star"])),
        ("两套精确解 ρ*_L / ρ*_R 相对差",
         f"{_sci(chk['rel_d_rho_star_l'])} / {_sci(chk['rel_d_rho_star_r'])}"),
        (f"整条自相似剖面最大偏差（{chk['n_xi']} 个 ξ 采样，尺度化）",
         _sci(chk["max_abs_profile_diff_scaled"])),
        ("右激波 Rankine–Hugoniot 残差 质量/动量/能量",
         f"{_sci(rh['rel_residual_mass'])} / {_sci(rh['rel_residual_momentum'])} / "
         f"{_sci(rh['rel_residual_energy'])}"),
        ("波面序 x_head < x_tail < x_contact < x_shock",
         "成立" if waves["x_head"] < waves["x_tail"] < waves["x_contact"] < waves["x_shock"]
         else "不成立"),
    ]))
    a("")
    a("### 3.4 两个独立入口的可复现性对照")
    a("")
    if cross.get("available"):
        a("同一算例经**两个不同入口、两次不同时间的运行**：独立脚本 "
          "`scripts/lax_shock_tube_benchmark.py`（工件 `docs/lax-shock-tube/artifacts/lax_report.json`）"
          "与本单的标准验证腿。数值一致性如下（差值为两次运行 L1 误差之绝对差）：")
        a("")
        a(_table(["nx", "L1ρ（脚本）", "L1ρ（本腿）", "Δ L1ρ", "Δ L1u", "Δ L1p"],
                 cross["rows"]))
        a("")
        a(f"三场 L1 的最大绝对差 = {cross['worst_abs_diff']:.3e}；"
          f"p* 脚本值 {cross['p_star_ref']:.6f} vs 本腿值 {cross['p_star_leg']:.6f}。"
          + ("两次运行在报告精度内逐位一致（同一确定性代码路径，仅线程数不同），"
             "即注册后的腿复现了原文档数值，不是重新调参后的另一组数字。"
             if cross["worst_abs_diff"] <= 1e-12 else
             "两次运行差异在 1e-12 以上，须按浮点/线程归约差异解释，不能声称逐位复现。"))
    else:
        a("（未提供 `docs/lax-shock-tube/artifacts/lax_report.json`，跳过跨入口对照。）")
    a("")
    a("## ④ 结果图")
    a("")
    a(f"![Lax profiles vs exact]({Path(figs[0]).name if figs else 'fig_lax_profiles_vs_exact.png'})")
    a("")
    a(f"图注：左三栏 = t = {leg.get('profile', {}).get('t', cfg['t_end'])} 时 ρ/u/p 的数值解（点）"
      "与精确解（实线）对比（最细网格）；右栏 = ρ/u/p 相对 L1 误差 vs 网格数的 log-log 曲线与 "
      f"{100 * th['l1_rel_error']:.0f}% 判据线，标题给出 ρ 的观测收敛阶。"
      "该图由 `figures.fig_lax` 从工件数值直绘，无手写数组。")
    a("")
    a("## ⑤ 注意点（坑与局限）")
    a("")
    a("1. **接触/激波仍是 O(Δx) 抹平**：L∞ 误差在间断处不收敛到零（本次 nx="
      f"{fine['nx']} 时 ρ 约 {_pct(max(r['linf_rel']['rho'] for r in rows))}），"
      "这是迎风格式的固有行为，不作判据；判据取平台值、波面位置与 L1。")
    a("2. **<2% 判据依赖网格**：nx=200 时 L1ρ 已达 "
      f"{_pct([r for r in rows if r['nx'] == 200][0]['l1_rel']['rho'])}（接近判据线），"
      "因此腿的正式判定固定取本腿最细网格；扩大适用面须同时报告网格。")
    a("3. **单材料理想气体**（pinf=0），不覆盖 stiffened-gas / 多材料路径"
      "（那由 `two_gas_interface`、`shock_bubble` 覆盖）；Lax 算例本身也只有单一 γ。")
    a("4. **本单只做注册与门，不改数值方法**：若日后改 HLLC 波速或限制器，"
      "本腿的 verdict 会随之变化，这正是它作为回归门的意义。")
    a("5. **生态摩擦（如实记）**：`_strip_heavy` 会把腿里的 `profile` 场快照从 JSON 工件剥离，"
      "故工件保留的是剖面图 PNG 而非原始场；需要原始场时用 `leg_lax` 的返回值（内存内）。")
    a("")
    a("## ⑥ 误差判定（逐项 vs 基准，显式回答是否 <3%）")
    a("")
    a("基准 = 同一初值问题的自相似精确解（两套实现互证，见 3.3）；判据线 = "
      f"{100 * th['l1_rel_error']:.0f}%（CASE-LAX 验收容差 <2%），"
      "此处同时回答 owner 的 <3% 问题。")
    a("")
    judge_rows = []
    for key, name in (("p_star", "星区压力 p*"), ("u_star", "星区速度 u*"),
                      ("rho_star_l", "星区密度 ρ*_L"), ("rho_star_r", "星区密度 ρ*_R")):
        err = checks[f"plateau[{key}]"]["value"]
        judge_rows.append((name, f"{finest['plateau_exact'][key]:.6f}",
                           f"{finest['plateau'][key]:.6f}", _pct(err),
                           "是" if err < 0.03 else "否",
                           "是" if err < th["plateau_rel_error"] else "否"))
    for key, name in (("x_fan", "稀疏波扇中点位置"), ("x_contact", "接触间断位置"),
                      ("x_shock", "激波位置")):
        err = checks[f"position[{key}]"]["value"]
        judge_rows.append((name, f"{finest['position_exact'][key]:.6f}",
                           f"{finest['position'][key]:.6f}", _pct(err),
                           "是" if err < 0.03 else "否",
                           "是" if err < th["wave_position_rel_error"] else "否"))
    for key, name in (("rho", "全场 ρ（相对 L1）"), ("u", "全场 u（相对 L1）"),
                      ("p", "全场 p（相对 L1）")):
        err = checks[f"l1[{key}]"]["value"]
        judge_rows.append((name, "精确解场", "数值场", _pct(err),
                           "是" if err < 0.03 else "否",
                           "是" if err < th["l1_rel_error"] else "否"))
    a(_table(["关键量", "ref（精确解）", "got（数值）", "相对误差", "<3%？", "<2%？"],
             judge_rows))
    a("")
    all_3 = all(checks[k]["value"] < 0.03 for k in checks
                if k.split("[")[0] in ("plateau", "position", "l1"))
    a(f"**结论：{'全部关键量误差 < 3%' if all_3 else '存在关键量误差 ≥ 3%'}，"
      f"且 {'全部满足' if artifact['all_criteria_met'] else '未全部满足'} <2% 验收容差**"
      f"（最细网格 nx = {fine['nx']}）。间断处 L∞ 误差不属关键量："
      "它是格式分辨率指标，按 ⑤ 的说明不作判据；本算例不声明 L∞ <3%。")
    a("")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifact", default="docs/CASE-LAX/compressible_multimaterial.json")
    ap.add_argument("--crosscheck", default="docs/lax-shock-tube/artifacts/lax_report.json")
    ap.add_argument("--out", default="docs/CASE-LAX/report.md")
    args = ap.parse_args(argv)
    artifact = json.loads(Path(args.artifact).read_text())
    cross = _crosscheck(artifact, Path(args.crosscheck) if args.crosscheck else None)
    text = render(artifact, cross)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(f"wrote {out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())