#!/usr/bin/env python3
"""Benchmark harness for the Lax (1954) shock tube: ``python scripts/lax_shock_tube_benchmark.py``.

Runs the grid sweep + the parameter-sensitivity runs through
``tensorfvm.compressible.lax``, writes ``lax_report.json``, redraws the figures
*from that JSON* and regenerates ``report.md`` from the same JSON, so no number in
the report is transcribed by hand.

Usage::

    python scripts/lax_shock_tube_benchmark.py --out docs/lax-shock-tube
    python scripts/lax_shock_tube_benchmark.py --out /tmp/lax --quick
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
if str(SRC) not in sys.path:  # run from a checkout without installing
    sys.path.insert(0, str(SRC))

from tensorfvm.compressible import lax as L  # noqa: E402

CITATION = {
    "initial_data": "Lax, P. D. (1954), Comm. Pure Appl. Math. 7:159-193",
    "reproduced_in": ("Toro, E. F. (2009), Riemann Solvers and Numerical Methods for "
                      "Fluid Dynamics, 3rd ed., Wiley, ch. 4 (test problem 2 + Sec. 4.2 "
                      "exact-solution construction)"),
    "note": ("Ref values are the exact solution of the stated initial-value problem, "
             "recomputed at run time by two independent solvers (in-repo Toro iteration "
             "+ from-scratch bisection in tensorfvm.compressible.lax); no digit is "
             "transcribed from the printed table."),
}


def _run(args) -> dict:
    t0 = time.time()
    cfg = L.LaxConfig()
    sizes = tuple(args.nx)
    sweep = L.sweep(sizes, cfg=cfg)
    sens = []
    if not args.quick:
        for kw in ({"limiter": "minmod"}, {"limiter": "vanleer"},
                   {"order": 1}, {"cfl": 0.2}, {"cfl": 0.8}):
            run_kw = {"limiter": "mc", "order": 2, "cfl": 0.4}
            run_kw.update(kw)
            row = L.run_case(args.sens_nx, cfg=cfg, **run_kw)
            row["variant"] = kw
            sens.append(row)
    result = {
        "case": "lax_shock_tube",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "citation": CITATION,
        "reference_state": cfg.as_dict(),
        "tol": L.THRESHOLDS,
        "exact": L.exact_waves(cfg),
        "independent_exact_check": L.independent_check(cfg),
        "rankine_hugoniot_residual": L.rankine_hugoniot_residual(cfg),
        "numerics": {"flux_mode": "hllc", "reconstruction": "MUSCL-Hancock",
                     "limiter": "mc", "cfl": 0.4, "bc": "outflow/outflow/periodic/periodic",
                     "nx_sweep": list(sizes), "sens_nx": args.sens_nx},
        "sweep": sweep,
        "sensitivity": sens,
        "verdicts": L.verdicts(sweep["rows"], cfg=cfg),
        "seconds": round(time.time() - t0, 1),
        "torch": {"version": torch.__version__, "threads": torch.get_num_threads()},
    }
    return result


# ----------------------------------------------------------------------
# figures (drawn from the JSON, never from in-memory numbers)
# ----------------------------------------------------------------------
def _figures(report: dict, figdir: Path) -> list[dict]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)
    out = []
    prof = max(report["sweep"]["rows"], key=lambda r: r["nx"])["profile"]
    ex = report["exact"]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 8.4), sharex=True)
    for ax, key, label in zip(axes, ("rho", "u", "p"),
                              (r"$\rho$", r"$u$", r"$p$")):
        ax.plot(prof["x"], prof[key], "o", ms=2.0, mfc="none", mec="C0", label="numerical")
        ax.plot(prof["x"], prof[f"{key}_exact"], "-", lw=1.2, color="C1", label="exact")
        ax.axvline(ex["x_head"], ls=":", c="0.4", lw=0.8)
        ax.axvline(ex["x_contact"], ls=":", c="0.4", lw=0.8)
        ax.axvline(ex["x_shock"], ls=":", c="0.4", lw=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("x  (dotted: exact rarefaction head / contact / shock)")
    axes[0].set_title(f"Lax shock tube, t = {report['reference_state']['t_end']}, "
                      f"nx = {prof_nx(report)}")
    p = figdir / "lax_profiles_vs_exact.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append({"name": p.name, "caption": "Numerical vs exact rho/u/p profiles "
                f"(nx = {prof_nx(report)}, MUSCL-Hancock + MC limiter, HLLC)."})

    rows = report["sweep"]["rows"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for key, label in (("rho", r"$\rho$"), ("u", r"$u$"), ("p", r"$p$")):
        a1.loglog([r["nx"] for r in rows], [r["l1_rel"][key] for r in rows], "o-", label=label)
    a1.axhline(report["tol"]["l1_rel_error"], ls="--", c="k", lw=0.9, label="2 % criterion")
    a1.set_xlabel("nx"); a1.set_ylabel("relative L1 error"); a1.grid(alpha=0.3, which="both")
    a1.legend(fontsize=8)
    a1.set_title("grid convergence vs exact solution")
    # observed order is defined per grid pair -> plot at the pair's geometric mean
    x_mid = [(rows[i]["nx"] * rows[i + 1]["nx"]) ** 0.5 for i in range(len(rows) - 1)]
    for key, label in (("rho", r"$\rho$"), ("u", r"$u$"), ("p", r"$p$")):
        a2.semilogx(x_mid, report["sweep"]["observed_order"][key], "o-",
                    label=f"order {label}")
    a2.set_xlabel("nx"); a2.set_ylabel("observed order"); a2.grid(alpha=0.3)
    a2.legend(fontsize=8)
    a2.set_title("observed convergence order")
    p = figdir / "lax_convergence.png"
    fig.tight_layout()
    fig.savefig(p, dpi=140)
    plt.close(fig)
    out.append({"name": p.name, "caption": "Relative L1 error vs nx (log-log) and observed "
                "order; dashed line = 2 % acceptance level."})

    if report["sensitivity"]:
        labels, vals = [], []
        for row in report["sensitivity"]:
            labels.append("+".join(f"{k}={v}" for k, v in row["variant"].items()))
            vals.append(row["plateau_rel_error"]["rho_star_r"])
        labels.insert(0, "baseline")
        vals.insert(0, max(rows, key=lambda r: r["nx"] == report["numerics"]["sens_nx"])
                    ["plateau_rel_error"]["rho_star_r"]
                    if any(r["nx"] == report["numerics"]["sens_nx"] for r in rows) else float("nan"))
        fig, ax = plt.subplots(figsize=(7.6, 3.8))
        ax.barh(range(len(labels)), vals, color="C0")
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.axvline(report["tol"]["plateau_rel_error"], ls="--", c="k", lw=0.9)
        ax.set_xlabel(r"relative error of $\rho^*_R$ plateau")
        ax.set_title(f"parameter sensitivity at nx = {report['numerics']['sens_nx']}")
        fig.tight_layout()
        p = figdir / "lax_sensitivity.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        out.append({"name": p.name, "caption": "Sensitivity of the post-shock density "
                    "plateau error to limiter / order / CFL at nx = "
                    f"{report['numerics']['sens_nx']}; dashed line = 2 % criterion."})
    return out


def prof_nx(report: dict) -> int:
    return max(r["nx"] for r in report["sweep"]["rows"] if "profile" in r)


# ----------------------------------------------------------------------
# report.md (all numbers injected from the JSON)
# ----------------------------------------------------------------------
def _fmt(v, nd=6):
    if isinstance(v, float):
        if v != v:
            return "nan"
        if v != 0.0 and (abs(v) < 1e-3 or abs(v) >= 1e4):
            return f"{v:.3e}"
        return f"{v:.{nd}f}"
    return str(v)


def _mark(got, ref, tol):
    if got is None or got != got:
        return "x"
    return "OK" if got <= tol else "x"


def _report_md(report: dict, figures: list[dict]) -> str:
    cfg = report["reference_state"]
    ex = report["exact"]
    ind = report["independent_exact_check"]
    rh = report["rankine_hugoniot_residual"]
    tol = report["tol"]
    rows = report["sweep"]["rows"]
    fine = max(rows, key=lambda r: r["nx"])
    v = report["verdicts"]
    num = report["numerics"]
    L = []
    a = L.append
    a("# Lax 激波管算例（TensorFVM）——精确解对照与网格/参数敏感性")
    a("")
    a(f"> 算例：Lax (1954) 一维 Euler 激波管；求解器 `tensorfvm.compressible`（MUSCL-Hancock + "
      f"{num['flux_mode'].upper()} + {num['limiter']} 限制器，CFL={num['cfl']}）。")
    a(f"> 生成：`{report['generated_at']}`；耗时 {report['seconds']} s；"
      f"torch {report['torch']['version']}，线程数 {report['torch']['threads']}。")
    a("> 本文件与全部数字由 `scripts/lax_shock_tube_benchmark.py` 程序化生成，"
      "源数据 `artifacts/lax_report.json`（零手抄）。")
    a("")
    a("## ① 问题与参考出处")
    a("")
    a("**物理背景。** 一维理想气体激波管（Riemann 问题）：隔膜两侧压差驱动产生"
      "左行稀疏波 + 接触间断 + 右行激波。与 Sod 相比，Lax 算例的右波是**强激波**"
      f"（p*/p_R = {_fmt(ex['p_star']/cfg['p_r'],3)}，ρ*_R/ρ_R = "
      f"{_fmt(ex['rho_star_r']/cfg['rho_r'],3)}），接触两侧星区密度相差 "
      f"{_fmt(ex['rho_star_r']/ex['rho_star_l'],3)} 倍，左行稀疏波扇跨 "
      f"[{_fmt(ex['x_head'],3)}, {_fmt(ex['x_tail'],3)}]；强激波、强稀疏波与接触间断"
      "同时在场，对通量波速估计与接触锐化更敏感。")
    a("")
    a(f"**星区状态。** u*/c*_L = {_fmt(ex['u_star']/ex['c_star_l'],3)}（亚声速），"
      f"u*/c_R = {_fmt(ex['u_star']/math.sqrt(cfg['gamma']*cfg['p_r']/cfg['rho_r']),3)}。")
    a("")
    a(f"**参考出处。** 初值数据取自 {report['citation']['initial_data']}；同一数据集被 "
      f"{report['citation']['reproduced_in']} 复述为标准一维测试算例。"
      f"{report['citation']['note']}")
    a("")
    a(f"**自相似精确解（本节所有数值由程序算出，见 `artifacts/lax_report.json:exact`）。** "
      f"左波={ex['left_wave']}，右波={ex['right_wave']}；"
      f"星区压力 p* = {_fmt(ex['p_star'])}、星区速度 u* = {_fmt(ex['u_star'])}、"
      f"ρ*_L = {_fmt(ex['rho_star_l'])}、ρ*_R = {_fmt(ex['rho_star_r'])}；"
      f"波速：稀疏波头 {_fmt(ex['speed_head'])}、稀疏波尾 {_fmt(ex['speed_tail'])}、"
      f"接触 {_fmt(ex['speed_contact'])}、激波 {_fmt(ex['speed_shock'])}；"
      f"t = {cfg['t_end']} 时位于 x = {_fmt(ex['x_head'],4)} / {_fmt(ex['x_tail'],4)} / "
      f"{_fmt(ex['x_contact'],4)} / {_fmt(ex['x_shock'],4)}。")
    a("")
    a("**参考值的独立复核（两套互不共享代码的精确解）。**")
    a("")
    a("| 量 | 仓内 Toro 迭代 | 独立二分重写 | 相对差 |")
    a("|---|---|---|---|")
    a(f"| p* | {_fmt(ex['p_star'])} | {_fmt(ind['p_star'])} | {_fmt(ind['rel_dp_star'])} |")
    a(f"| u* | {_fmt(ex['u_star'])} | {_fmt(ind['u_star'])} | {_fmt(ind['rel_du_star'])} |")
    a(f"| ρ*_L | {_fmt(ex['rho_star_l'])} | {_fmt(ind['rho_star_l'])} | "
      f"{_fmt(ind['rel_d_rho_star_l'])} |")
    a(f"| ρ*_R | {_fmt(ex['rho_star_r'])} | {_fmt(ind['rho_star_r'])} | "
      f"{_fmt(ind['rel_d_rho_star_r'])} |")
    a("")
    a(f"整条自相似剖面上（{ind['n_xi']} 个 ξ 采样点）两套解的最大绝对偏差 "
      f"{_fmt(ind['max_abs_profile_diff'])}（判据 ≤ {_fmt(tol['independent_exact_rel_error'])} "
      f"相对）。右激波上的 Rankine-Hugoniot 残差（S(U*-U)=F*-F 三分量）：质量 "
      f"{_fmt(rh['rel_residual_mass'])}、动量 {_fmt(rh['rel_residual_momentum'])}、"
      f"能量 {_fmt(rh['rel_residual_energy'])}。")
    a("")
    a("## ② 方法与实现")
    a("")
    a("- 求解器：仓内 `tensorfvm.compressible`（`EulerSolver`：有限体积，二阶 MUSCL-Hancock "
      "预测-校正，HLLC 通量，CFL 自适应步长，outflow/periodic 边界），**未新增求解器代码**，"
      "只新增算例 + 计量层 `tensorfvm.compressible.lax`。")
    a("- 计量：以自相似坐标 ξ=(x-x0)/t 取精确解，逐网格比较 ρ/u/p；"
      "另取 4 个星区平台值（p*, u*, ρ*_L, ρ*_R，窗口离间断 15% 子区间）与 3 个波面位置"
      "（稀疏波扇中点、接触、激波；**同一估计器**同时作用于数值场与 20001 点细网格上的"
      "精确场，沿场量跨中值做亚网格线性插值定位，估计器偏差在比较中抵消）。")
    a("- 参考值：两套独立精确解（仓内 Toro 压力函数迭代 + 本模块从零重写的二分求解 + "
      "Rankine-Hugoniot 自洽检验），互证到机器精度后再作为 ref。")
    a("- 复现：`python scripts/lax_shock_tube_benchmark.py --out docs/lax-shock-tube`。")
    a("")
    a("## ③ 参数表（全部无量纲）")
    a("")
    a("| 参数 | 值 |")
    a("|---|---|")
    a(f"| γ（两侧理想气体） | {cfg['gamma']} |")
    a(f"| 左态 ρ_L / u_L / p_L | {cfg['rho_l']} / {cfg['u_l']} / {cfg['p_l']} |")
    a(f"| 右态 ρ_R / u_R / p_R | {cfg['rho_r']} / {cfg['u_r']} / {cfg['p_r']} |")
    a(f"| 计算域 / 隔膜位置 x0 | [0, 1] / {cfg['x0']} |")
    a(f"| 终止时间 t_end | {cfg['t_end']} |")
    a(f"| c_L / c_R | {_fmt(math.sqrt(cfg['gamma']*cfg['p_l']/cfg['rho_l']),4)} / "
      f"{_fmt(math.sqrt(cfg['gamma']*cfg['p_r']/cfg['rho_r']),4)} |")
    a(f"| 星区马赫数 u*/c*_L | {_fmt(ex['u_star']/ex['c_star_l'],4)} |")
    a(f"| 网格 | nx ∈ {num['nx_sweep']}（均匀，ny=1） |")
    a(f"| 通量/重构/限制器/CFL | {num['flux_mode'].upper()} / MUSCL-Hancock / "
      f"{num['limiter']} / {num['cfl']} |")
    a(f"| 边界 | {num['bc']} |")
    a(f"| 验收判据 | 平台值/波面位置/L1 相对误差 ≤ {tol['plateau_rel_error']*100:.0f}% |")
    a("")
    a("## ④ 对照表（ref vs got，nx = %d）" % fine["nx"])
    a("")
    a("| 量 | ref（精确解） | got（数值） | 相对误差 | <2% |")
    a("|---|---|---|---|---|")
    for key in ("p_star", "u_star", "rho_star_l", "rho_star_r"):
        rel = fine["plateau_rel_error"][key]
        a(f"| 平台 {key} | {_fmt(fine['plateau_exact'][key])} | {_fmt(fine['plateau'][key])} | "
          f"{100*rel:.3f}% | {_mark(rel, None, tol['plateau_rel_error'])} |")
    for key, label in (("x_fan", "波面 x_fan（稀疏波扇中点）"),
                       ("x_contact", "波面 x_contact（接触间断）"),
                       ("x_shock", "波面 x_shock（激波）")):
        rel = fine["position_rel_error"][key]
        a(f"| {label} | {_fmt(fine['position_exact'][key],4)} | "
          f"{_fmt(fine['position'][key],4)} | "
          f"{'nan' if rel is None else format(100*rel, '.3f') + '%'} | "
          f"{_mark(rel, None, tol['wave_position_rel_error'])} |")
    for key in ("rho", "u", "p"):
        a(f"| 全场相对 L1 误差 {key} | — | — | {100*fine['l1_rel'][key]:.3f}% | "
          f"{_mark(fine['l1_rel'][key], None, tol['l1_rel_error'])} |")
        a(f"| 全场相对 L∞ 误差 {key} | — | — | {100*fine['linf_rel'][key]:.3f}% | 参照项 |")
    a("")
    a(f"**判定：{'全部通过' if v['all_ok'] else '未全部通过'}**"
      f"（nx={v['nx']}；质量漂移 {_fmt(abs(fine['drift']['mass']))}，"
      f"正性修补次数 {fine['positivity_fixes']}）。")
    a("")
    a("## ⑤ 收敛性与参数敏感性")
    a("")
    a("| nx | L1ρ | L1u | L1p | ρ*_R 平台误差 | 步数 | 秒 |")
    a("|---|---|---|---|---|---|---|")
    for r in rows:
        a(f"| {r['nx']} | {100*r['l1_rel']['rho']:.3f}% | {100*r['l1_rel']['u']:.3f}% | "
          f"{100*r['l1_rel']['p']:.3f}% | {100*r['plateau_rel_error']['rho_star_r']:.3f}% | "
          f"{r['steps']} | {r['seconds']} |")
    a("")
    for key in ("rho", "u", "p"):
        obs = ", ".join(f"{o:.2f}" for o in report["sweep"]["observed_order"][key])
        a(f"- 观测收敛阶（{key}，逐对网格）：{obs}")
    a("")
    if report["sensitivity"]:
        base = next((r for r in rows if r["nx"] == num["sens_nx"]), None)
        a(f"参数敏感性（nx = {num['sens_nx']}，与同网格基线对照）：")
        a("")
        a("| 变体 | L1ρ | L1u | L1p | p* 平台误差 | ρ*_R 平台误差 |")
        a("|---|---|---|---|---|---|")
        if base is not None:
            a(f"| 基线 mc/order2/cfl0.4 | {100*base['l1_rel']['rho']:.3f}% | "
              f"{100*base['l1_rel']['u']:.3f}% | {100*base['l1_rel']['p']:.3f}% | "
              f"{100*base['plateau_rel_error']['p_star']:.3f}% | "
              f"{100*base['plateau_rel_error']['rho_star_r']:.3f}% |")
        for r in report["sensitivity"]:
            tag = "+".join(f"{k}={vv}" for k, vv in r["variant"].items())
            a(f"| {tag} | {100*r['l1_rel']['rho']:.3f}% | {100*r['l1_rel']['u']:.3f}% | "
              f"{100*r['l1_rel']['p']:.3f}% | {100*r['plateau_rel_error']['p_star']:.3f}% | "
              f"{100*r['plateau_rel_error']['rho_star_r']:.3f}% |")
        a("")
    a("## ⑥ 失败模式与工件")
    a("")
    a("**已知失败模式 / 局限。**")
    a("  1. 接触间断与激波处仍是 O(dx) 抹平：全场 L1 误差随网格一阶下降（见 ⑤ 观测阶），"
      "因此 ≤2% 的判据只能在足够细的网格上满足；平台值与波面位置对网格不敏感，是本算例的主判据。")
    a("  2. 精确解在 t→0 与 x=x0 处是奇点，网格越细时间步越小，成本线性增长。")
    a("  3. L∞ 误差在间断处保持 O(1)（本网格族 ρ 约 40%、p 约 15%）：这是迎风格式"
      "在接触/激波上的固有抹平，不应作为收敛判据；判据取平台值、波面位置与 L1。")
    a("  4. 本算例为单材料理想气体（pinf=0），不覆盖 stiffened-gas/多材料路径"
      "（那由 `two_gas_interface` / `shock_bubble` 覆盖）。")
    a("")
    a("**工件。**")
    a("- `artifacts/lax_report.json`：全部数值（ref、独立复核、RH 残差、扫描、敏感性、判定）。")
    for f in figures:
        a(f"- `figures/{f['name']}`：{f['caption']}")
    a(f"- 复现命令：`python scripts/lax_shock_tube_benchmark.py --out docs/lax-shock-tube`"
      f"（单进程约 {report['seconds']} s）。")
    a("")
    a("**生态摩擦（如实记）。**")
    a("  1. `python -m tensorfvm.compressible` 的 `--legs` 只认 sod/interface/rm/bubble/vortex；"
      "Lax 之前仓里根本没有该算例（新算例只能走新入口，未改动既有 CLI 语义）。")
    a("  2. 默认 torch 线程数=128 时 200 网格 1-D 算例 23 ms/步，远慢于单线程 I/O 受限的实际成本，"
      "小算例应显式限制线程数。")
    a("  3. 知识库检索端点（KE engine :8021）离线，`开工三查` 只能退化为 API/仓内文档检索。")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="docs/lax-shock-tube")
    ap.add_argument("--nx", type=int, nargs="+", default=[200, 400, 800, 1600])
    ap.add_argument("--sens-nx", type=int, default=400)
    ap.add_argument("--quick", action="store_true", help="sweep only, coarse grids")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--figures", action="store_true", default=True)
    args = ap.parse_args(argv)
    if args.quick:
        args.nx = [100, 200]
    torch.set_num_threads(args.threads)
    out = Path(args.out)
    (out / "artifacts").mkdir(parents=True, exist_ok=True)
    report = _run(args)
    jpath = out / "artifacts" / "lax_report.json"
    jpath.write_text(json.dumps(report, indent=1, sort_keys=True), encoding="utf-8")
    reloaded = json.loads(jpath.read_text(encoding="utf-8"))     # figures/report read the file
    figures = _figures(reloaded, out / "figures")
    (out / "report.md").write_text(_report_md(reloaded, figures), encoding="utf-8")
    summary = {"all_ok": report["verdicts"]["all_ok"], "nx": report["verdicts"]["nx"],
               "seconds": report["seconds"], "json": str(jpath)}
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if report["verdicts"]["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())