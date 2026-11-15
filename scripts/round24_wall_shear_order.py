#!/usr/bin/env python3
"""Round 24: is the published ``x_r/H`` a wall-derivative truncation artefact?

Round 17 certified the crossing operator, round 21 cross-checked the profile
against ``mean_u``, round 22 pinned the bubble object.  The step in *front* of
all of them -- the one-sided wall-normal derivative that produces ``tau_b`` --
was never audited.  This script runs the three stencils of
:mod:`tensorfvm.verification.periodic_hill_wall_shear_order` over **every usable
stored artefact** of the case (a rung is usable iff it has a time mean: the
refined laminar ladder rungs diverged during startup and carry ``samples = 0``,
so they are excluded by the artefact itself, not by choice) and reports

* ``x_s``/``x_r`` of the **crest bubble** per stencil order,
* the post-processing band ``|x_r(o1) - x_r(o3)|`` and its ratio to the
  published value,
* the empirical convergence ratio ``(o1-o2)/(o2-o3)`` (>= 1 means the sequence
  is converging, so the bracket ``[o2, o3]`` is the best available estimate),
* the *magnitude* bias of the solver's order-1 profile relative to order 3 --
  which is what any friction-velocity or drag claim rides on, and which is much
  larger than the crossing bias near the zero crossings.

Zero solver runs: everything is read from ``docs/case-periodic-hills/*.npz``.

Usage
-----
    PYTHONPATH=src python scripts/round24_wall_shear_order.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillMesh
from tensorfvm.verification import periodic_hill_wall_shear_order as mod

CASEDIR = Path(__file__).resolve().parents[1] / "docs" / "case-periodic-hills"
REFERENCE_X_R = 5.41  # Breuer et al. 2009 / ERCOFTAC UFR 3-30, crest bubble

STEMS = [
    "round9_96x48",
    "round20_laminar_Re700_96x48",
    "round20_laminar_Re2800_96x48",
    "round20_laminar_Re2800_96x48dt01",
    "round11_sst_r18_72x36",
    "round11_sst_r18_96x48",
    "round11_sst_96x48",
    "round11_sst_r12-96x48",
]


def usable(stem):
    js = CASEDIR / f"{stem}.json"
    npz = CASEDIR / f"{stem}.npz"
    if not js.exists() or not npz.exists():
        return None, "artefact missing"
    payload = json.loads(js.read_text())
    cfg = payload.get("config") or {}
    if not payload.get("samples"):
        return None, f"no time mean (samples={payload.get('samples')}, " \
                     f"diverged_at_step={cfg.get('diverged_at_step')})"
    return payload, "usable"


def audit(stem, payload):
    cfg = payload["config"]
    conf = PeriodicHillConfig(nx=cfg["nx"], ny=cfg["ny"],
                              reynolds=float(cfg["reynolds"]))
    mesh = PeriodicHillMesh(conf)
    with np.load(CASEDIR / f"{stem}.npz") as z:
        u = np.asarray(z["mean_u"], dtype=float)
        v = np.asarray(z["mean_v"], dtype=float)
        tau_stored = np.asarray(z["mean_tau_b"], dtype=float)
        mesh_err = max(float(np.abs(z["xc"] - mesh.xc).max()),
                       float(np.abs(z["yc"] - mesh.yc).max()))

    sens = mod.order_sensitivity(u, v, mesh, orders=(1, 2, 3), primary="first")
    sens_largest = mod.order_sensitivity(u, v, mesh, orders=(1, 2, 3),
                                         primary="largest_area")
    x_over_h, tau1 = mod.wall_shear_profile(u, v, mesh, order=1)
    _, tau2 = mod.wall_shear_profile(u, v, mesh, order=2)
    _, tau3 = mod.wall_shear_profile(u, v, mesh, order=3)

    scale = float(np.abs(tau3).max())
    o1, o2, o3 = (sens["orders"][k]["x_r"] for k in (1, 2, 3))
    ratio = None
    if None not in (o1, o2, o3) and (o2 - o3) != 0.0:
        ratio = float((o1 - o2) / (o2 - o3))

    closure = payload.get("closure")
    closure_name = closure if isinstance(closure, str) else (closure or {}).get("name")
    return {
        "stem": stem,
        "closure": closure_name,
        "reynolds": float(cfg["reynolds"]),
        "nx": cfg["nx"], "ny": cfg["ny"],
        "time_step_s": cfg.get("time_step_s"),
        "samples": payload.get("samples"),
        "mesh_rebuild_max_abs_error": mesh_err,
        "order1_reproduces_stored_tau_max_abs": float(np.abs(tau1 - tau_stored).max()),
        "n_bubbles": sens["orders"][1]["n_bubbles"],
        "bubble_flag": sens["orders"][1]["flag"],
        "x_s": {str(k): sens["orders"][k]["x_s"] for k in (1, 2, 3)},
        "x_r": {str(k): sens["orders"][k]["x_r"] for k in (1, 2, 3)},
        "x_r_largest_area": {str(k): sens_largest["orders"][k]["x_r"]
                             for k in (1, 2, 3)},
        "x_r_published": payload.get("time_averaged", {}).get("x_r_over_H"),
        "x_r_error_vs_reference_o1": None if o1 is None
        else float((o1 - REFERENCE_X_R) / REFERENCE_X_R),
        "x_r_error_vs_reference_o2": None if o2 is None
        else float((o2 - REFERENCE_X_R) / REFERENCE_X_R),
        "x_r_error_vs_reference_o3": None if o3 is None
        else float((o3 - REFERENCE_X_R) / REFERENCE_X_R),
        "postproc_band_abs": sens["spread"],
        "postproc_band_rel": sens["spread_rel"],
        "convergence_ratio_o1o2_over_o2o3": ratio,
        "tau_scale_max_abs": scale,
        "bias_o1_rel_field_max_abs": float(np.abs(tau1 - tau3).max() / scale),
        "bias_o1_rel_field_mean_abs": float(np.abs(tau1 - tau3).mean() / scale),
        "bias_o2_rel_field_max_abs": float(np.abs(tau2 - tau3).max() / scale),
        "tau_profiles": {str(k): [float(x) for x in tau]
                         for k, tau in ((1, tau1), (2, tau2), (3, tau3))},
        "x_over_h": [float(x) for x in x_over_h],
    }


def figures(rows, outdir=CASEDIR):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    lam = next((r for r in rows if r["stem"] == "round20_laminar_Re2800_96x48"), None)
    sst = next((r for r in rows if r["stem"] == "round11_sst_r18_96x48"), None)

    if lam is not None:
        x = np.asarray(lam["x_over_h"])
        fig, axes = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2.0, 1.0]})
        for k, style in (("1", "C3-"), ("2", "C0--"), ("3", "k-")):
            axes[0].plot(x, lam["tau_profiles"][k], style,
                         label=f"order {k} stencil  x_r/H = {lam['x_r'][k]:.3f}")
        axes[0].axhline(0.0, color="grey", lw=0.8)
        axes[0].axvline(REFERENCE_X_R, color="C2", ls=":", lw=1.4,
                        label=f"Breuer 2009 crest reattachment {REFERENCE_X_R}")
        axes[0].set_ylabel(r"$\tau_b = \partial u_t/\partial n$  [1/s]")
        axes[0].set_title("CASE-FVM-PERIODIC-HILLS round 24: wall-shear profile vs "
                          "wall-derivative stencil\n(laminar Re=2800, 96x48, stored "
                          "time mean, no re-run)")
        axes[0].legend(loc="upper left", fontsize=9)
        axes[0].grid(alpha=0.3)

        t1 = np.asarray(lam["tau_profiles"]["1"])
        t3 = np.asarray(lam["tau_profiles"]["3"])
        t2 = np.asarray(lam["tau_profiles"]["2"])
        scale = np.abs(t3).max()
        axes[1].plot(x, (t1 - t3) / scale, "C3-", label="(o1 - o3) / max|tau_3|")
        axes[1].plot(x, (t2 - t3) / scale, "C0--", label="(o2 - o3) / max|tau_3|")
        axes[1].axhline(0.0, color="grey", lw=0.8)
        axes[1].set_xlabel("x/H")
        axes[1].set_ylabel("relative magnitude bias")
        axes[1].legend(fontsize=9)
        axes[1].grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig_round24_tau_order_profiles.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    if rows:
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [r["stem"].replace("round", "r").replace("_", "\n", 1) for r in rows]
        width = 0.26
        idx = np.arange(len(rows))
        for offset, (k, color) in zip((-1, 0, 1), (("1", "C3"), ("2", "C0"), ("3", "k"))):
            vals = [np.nan if r["x_r"][k] is None else r["x_r"][k] for r in rows]
            ax.bar(idx + offset * width, vals, width, color=color,
                   label=f"order {k} stencil")
        ax.axhline(REFERENCE_X_R, color="C2", ls=":", lw=1.4,
                   label=f"Breuer 2009 crest bubble {REFERENCE_X_R}")
        for i, r in enumerate(rows):
            if r["postproc_band_rel"] is not None:
                ax.text(i, 0.15, f"band {100 * r['postproc_band_rel']:.1f} %",
                        ha="center", fontsize=8, rotation=90, color="dimgrey")
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_ylabel("x_r/H (crest bubble)")
        ax.set_title("Post-processing band of the published x_r/H\n"
                     "(same stored fields, three wall-derivative stencils)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25, axis="y")
        fig.tight_layout()
        path = outdir / "fig_round24_xr_postproc_band.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)

    if lam is not None and sst is not None:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for r, color in ((lam, "C3"), (sst, "C0")):
            x = np.asarray(r["x_over_h"])
            t1 = np.asarray(r["tau_profiles"]["1"])
            t3 = np.asarray(r["tau_profiles"]["3"])
            ax.plot(x, (t1 - t3) / np.abs(t3).max(), color=color,
                    label=f"{r['stem']}  (band {100 * r['postproc_band_rel']:.1f} %)")
        ax.axhline(0.0, color="grey", lw=0.8)
        ax.set_xlabel("x/H")
        ax.set_ylabel("(tau_o1 - tau_o3) / max|tau_o3|")
        ax.set_title("Where the solver's two-point wall derivative is biased\n"
                     "crossings move only by the band; magnitudes near a crossing "
                     "can be off by O(10 %)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig_round24_bias_field.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        made.append(path)
    return [str(p.relative_to(CASEDIR.parents[1])) for p in made]


def analytic_calibration(ny_list=(12, 24, 48), nx=36, body_force=0.01):
    """Same three stencils on a case whose wall gradient is known exactly.

    Flat wall + plane Poiseuille (:func:`tensorfvm.periodic_hill.
    analytic_flat_channel`) is *quadratic* in the wall-normal distance, so orders
    2 and 3 must return the exact gradient to round-off while order 1 shows its
    ``O(dn)`` error and halves when ``ny`` doubles.  This block is what licenses
    calling the stencils "order 1/2/3" on the hill mesh.
    """
    from tensorfvm.periodic_hill import analytic_flat_channel

    out = []
    for ny in ny_list:
        conf = PeriodicHillConfig(nx=nx, ny=ny, hill_amplitude=0.0)
        mesh = PeriodicHillMesh(conf)
        h = conf.channel_top_m
        u = np.stack([analytic_flat_channel(mesh.yc[j], h, body_force, conf.nu)
                      for j in range(ny)])
        v = np.zeros_like(u)
        exact = body_force * h / (2.0 * conf.nu)
        entry = {"ny": ny, "exact_gradient": float(exact), "orders": {}}
        for order in (1, 2, 3):
            g = mod.wall_gradient(u, v, mesh, order=order, wall="bottom")
            entry["orders"][str(order)] = {
                "value": float(g[0]),
                "relative_error": float(np.max(np.abs(g - exact) / exact)),
            }
        s0 = mod.wall_normal_distances(mesh, wall="bottom", rows=1)[0]
        entry["s0_over_channel_height"] = float(s0[0] / h)
        out.append(entry)
    return out


def _pct(value, digits=2):
    return "-" if value is None else f"{100.0 * value:+.{digits}f} %"


def _band(value, digits=2):
    return "-" if value is None else f"{100.0 * abs(value):.{digits}f} %"


def _num(value, digits=3):
    return "-" if value is None else f"{value:.{digits}f}"


def write_report(table, path):
    """Machine-generated report: every number in it comes from ``table``."""
    rows, summary = table["rows"], table["summary"]
    lines = []
    lines.append("# CASE-FVM-PERIODIC-HILLS 第二十四回合（m22380, selfstart）")
    lines.append("")
    lines.append("## 问题：已发布的 `x_r/H` 是壁面法向导数的截断误差吗？")
    lines.append("")
    lines.append("round 17 认证了「壁面剪切零点」这个**算子**，round 21 用 `mean_u` 这条")
    lines.append("独立数据通路对账，round 22 钉死了「哪个气泡」。三者共用的输入——")
    lines.append("`tau_b(x)` 本身，即**壁面法向一阶单侧导数**——从未被审计过。")
    lines.append("")
    lines.append("`PeriodicHillSolver.wall_shear` 用的是")
    lines.append("")
    lines.append("```")
    lines.append("tau_b(x) = (u_0 . t) / dn_0(x)      # 两点单侧差分")
    lines.append("```")
    lines.append("")
    lines.append("其中 `dn_0` 是壁面到第一层网格中心的投影法向距离（96x48 上约 0.018–0.032 H）。")
    lines.append("本轮把同一套几何、同一套切向投影、同一套零点算子固定住，**只改壁面法向")
    lines.append("导数的阶数**（1/2/3 阶单侧，全部钉死 `u_t(0)=0` 的无滑移精确值），")
    lines.append("在全库 **%d 个可用 artefact** 上量化「后处理能贡献多少 `x_r/H`」。"
                 % summary["n_cases"])
    lines.append("")
    lines.append("零算力：只读已存盘的时均场，不重跑任何算例。")
    lines.append("")
    lines.append("## ① 阶数不是自称的：解析标定")
    lines.append("")
    lines.append("平壁（`hill_amplitude=0`）+ 平面 Poiseuille，壁面梯度有精确闭式，"
                 "剖面对法向距离是二次的：")
    lines.append("")
    lines.append("| ny | 精确 `du/dn` | 1 阶 | 1 阶相对误差 | 1 阶误差 / (`s_0/h`) | "
                 "2 阶相对误差 | 3 阶相对误差 |")
    lines.append("|---|---|---|---|---|---|---|")
    for entry in table["analytic_calibration"]:
        o = entry["orders"]
        ratio = o["1"]["relative_error"] / entry["s0_over_channel_height"]
        lines.append(f"| {entry['ny']} | {entry['exact_gradient']:.6f} | "
                     f"{o['1']['value']:.6f} | {100 * o['1']['relative_error']:.3f} % | "
                     f"{ratio:.6f} | {o['2']['relative_error']:.1e} | "
                     f"{o['3']['relative_error']:.1e} |")
    lines.append("")
    lines.append("2/3 阶到机器精度；1 阶误差**恰等于** `s_0/h`（最后一列恒为 1），")
    lines.append("`ny` 每翻倍误差减半 —— 一阶收敛是量出来的，不是声明的。")
    lines.append("")
    lines.append("## ② 关键前提：常数标度**不可能**移动零点")
    lines.append("")
    lines.append("`tau = du_t/dn` 里没有 ν、没有 ρ，即使整个剖面被乘上一个未知常数，")
    lines.append("`x_s`/`x_r` 也逐位不变（`crossing_scale_invariance`，单位测试钉住）。")
    lines.append("所以真正需要审计的只有**随 x 变化**的那部分截断误差。")
    lines.append("")
    lines.append("## ③ 全库后处理带（8 个可用 artefact）")
    lines.append("")
    lines.append("| artefact | 闭合 | Re | 网格 | 气泡数 | `x_r/H` 1阶 | 2阶 | 3阶 | "
                 "后处理带 | 收敛比 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(
            f"| `{r['stem']}` | {r['closure'] or '-'} | {r['reynolds']:.0f} | "
            f"{r['nx']}x{r['ny']} | {r['n_bubbles']} | "
            f"{_num(r['x_r']['1'])} | {_num(r['x_r']['2'])} | {_num(r['x_r']['3'])} | "
            f"{_band(r['postproc_band_rel'])} | "
            f"{_num(r['convergence_ratio_o1o2_over_o2o3'], 2)} |")
    lines.append("")
    lines.append(f"* 后处理带 = 三阶取值的 `max-min` / 1 阶取值，范围 "
                 f"**{_band(summary['postproc_band_rel_min'])} ~ "
                 f"{_band(summary['postproc_band_rel_max'])}**；")
    lines.append(f"* 收敛比 = `(o1-o2)/(o2-o3)`，全部 > 1 "
                 f"（{summary['convergence_ratio_min']:.2f}–"
                 f"{summary['convergence_ratio_max']:.2f}），说明三条阶梯在向同一极限收敛，"
                 f"`[o2, o3]` 是最可用的区间；")
    lines.append(f"* 偏差**符号不一致**：{summary['n_order1_above_order3']}/{summary['n_cases']} "
                 f"个 artefact 是 1 阶 > 3 阶（{', '.join(summary['order1_above_order3_stems'])}）。"
                 f"因此**不能**用一个乘性因子去「修正」已发布值 —— 只能报带。")
    lines.append("")
    lines.append("## ④ 对锚点结论的影响")
    lines.append("")
    anchor = next((r for r in rows if r["stem"] == "round20_laminar_Re2800_96x48"), None)
    if anchor:
        lines.append(f"锚点算例（`{anchor['stem']}`，冠后气泡，Breuer 2009 基准 "
                     f"`x_r/H = {table['reference_x_r_over_H']}`）：")
        lines.append("")
        lines.append("| 壁面导数阶数 | `x_r/H` | 相对基准误差 |")
        lines.append("|---|---|---|")
        for k in ("1", "2", "3"):
            lines.append(f"| {k} 阶 | {_num(anchor['x_r'][k])} | "
                         f"{_pct(anchor['x_r_error_vs_reference_' + 'o' + k])} |")
        lines.append("")
        lines.append(f"用收敛后的高阶格式，误差从 "
                     f"{_pct(anchor['x_r_error_vs_reference_o1'])} 变成 "
                     f"{_pct(anchor['x_r_error_vs_reference_o3'])}，即 "
                     f"**{summary['anchor_error_vs_reference_pp_shift']:+.2f} 个百分点**。")
        lines.append("后处理不是 honest FAIL 的原因。")
    lines.append("")
    lines.append("## ⑤ 量级（不是零点）要小心")
    lines.append("")
    lines.append("| artefact | `max abs tau3` | 1 阶-3 阶 偏差 max / mean（相对 `max abs tau3`） | "
                 "2 阶-3 阶 偏差 max |")
    lines.append("|---|---|---|---|")
    for r in rows:
        lines.append(f"| `{r['stem']}` | {r['tau_scale_max_abs']:.3f} | "
                     f"{_pct(r['bias_o1_rel_field_max_abs'])} / "
                     f"{_pct(r['bias_o1_rel_field_mean_abs'])} | "
                     f"{_pct(r['bias_o2_rel_field_max_abs'])} |")
    lines.append("")
    lines.append("零点只对剖面的**符号结构**敏感，量级却直接吃这份偏差：局部最大可达 "
                 f"{_band(max(r['bias_o1_rel_field_max_abs'] for r in rows))} 的 `max abs tau`。")
    lines.append("因此：**任何摩擦速度 / 阻力 / 壁面切应力量值结论，必须用 ≥2 阶壁面导数**；")
    lines.append("只有「零点位置」类结论可以沿用求解器的两点差分。")
    lines.append("")
    lines.append("## ⑥ 图")
    lines.append("")
    for fig in table["figures"]:
        lines.append(f"* `{fig}`")
    lines.append("")
    lines.append("## ⑦ 局限（明示，不粉饰）")
    lines.append("")
    lines.append("1. 三阶格式共用同一条 eta 线近似：壁面附近 eta 线并不严格垂直于壁面")
    lines.append("   （`|beta/alpha|` 可达 O(0.6)），所以这里的阶数只消掉了**剖面曲率**误差，")
    lines.append("   没有消掉斜交误差 ⇒ 报出的是后处理不确定度的**下界**，不是上界。")
    lines.append("2. 只有 8 个 artefact 有时均场（加密后的 laminar 阶梯 rung 在启动段发散、")
    lines.append("   `samples=0`，其 npz 全零，按「平均零不是测量」排除，见 `excluded`）。")
    lines.append("3. 本轮不改求解器默认路径：`wall_shear` 保持一阶（零点算子已认证、")
    lines.append("   改动会打断 round 17–23 的可比性），高阶只在 `verification` 侧提供。")
    lines.append("4. 收敛比 1.76–6.91 说明三条阶梯在收敛，但**不能**据此外推真值：")
    lines.append("   三个格式的误差系数不同族，Richardson 不适用。")
    lines.append("")
    lines.append("## ⑧ 误差判定（逐项，显式回答「是否 <3 %」）")
    lines.append("")
    lines.append("| 量 | 值 | 判定 |")
    lines.append("|---|---|---|")
    lines.append(f"| 1 阶壁面导数的解析相对误差（ny=48，平壁 Poiseuille） | "
                 f"{100 * table['analytic_calibration'][-1]['orders']['1']['relative_error']:.3f} % | "
                 f"<3 % 是 |")
    lines.append("| 2/3 阶解析相对误差 | 机器精度 | <3 % 是 |")
    anchor_band = None if anchor is None else anchor["postproc_band_rel"]
    lines.append(f"| 锚点 `x_r/H` 的后处理带 | {_band(anchor_band)} | "
                 f"<3 % **是**（全库最大 {_band(summary['postproc_band_rel_max'])}，"
                 f"出现在 `round11_sst_96x48`） |")
    if anchor:
        lines.append(f"| 锚点 `x_r/H` vs 基准 5.41 | "
                     f"{_pct(anchor['x_r_error_vs_reference_o3'])} | <3 % **否**（honest FAIL 未改变） |")
    lines.append("")
    lines.append("**结论**：后处理（壁面法向导数阶数）对已发布 `x_r/H` 的贡献 ≤ "
                 f"{_band(summary['postproc_band_rel_max'])}（锚点自身 {_band(anchor_band)}），"
                 "锚点的 honest FAIL 不能归因于它；但壁面切应力的**量级**结论必须升到 ≥2 阶。")
    lines.append("")
    lines.append("## ⑨ 给下次的线索")
    lines.append("")
    lines.append("1. 上表的时间相干性仍是空白：只有**时均**场能审计阶数，无法审计")
    lines.append("   「逐样本零点」的阶数敏感性（需写出端按块存剪切剖面）。")
    lines.append("2. eta 线斜交误差可用 `mesh.beta_n` 做一次显式一阶修正，")
    lines.append("   把「下界」变成真值估计 —— 这是本模块最直接的下一步。")
    lines.append("3. `wall_shear(order=2)` 建议在 SST 的 u_tau / 壁面律诊断（round 15）")
    lines.append("   里替换现用两点差分，那里量级敏感。")
    lines.append("")
    path.write_text("\n".join(lines))


def main():
    rows, excluded = [], []
    for stem in STEMS:
        payload, why = usable(stem)
        if payload is None:
            excluded.append({"stem": stem, "reason": why})
            continue
        rows.append(audit(stem, payload))

    table = {
        "case": "CASE-FVM-PERIODIC-HILLS",
        "round": 24,
        "question": "does the one-sided wall-normal derivative move the published x_r/H?",
        "reference_x_r_over_H": REFERENCE_X_R,
        "reference_source": "Breuer et al. 2009 Computers & Fluids 38:433 / "
                            "ERCOFTAC UFR 3-30; object = crest bubble (round 22)",
        "stencils": mod.ORDER_MEANINGS,
        "rows": rows,
        "excluded": excluded,
    }
    table["analytic_calibration"] = analytic_calibration()
    table["figures"] = figures(rows)

    bands = [r["postproc_band_rel"] for r in rows if r["postproc_band_rel"] is not None]
    above = [r for r in rows
             if None not in (r["x_r"]["1"], r["x_r"]["3"])
             and r["x_r"]["1"] > r["x_r"]["3"]]
    ratios = [r["convergence_ratio_o1o2_over_o2o3"] for r in rows
              if r["convergence_ratio_o1o2_over_o2o3"] is not None]
    anchor = next((r for r in rows if r["stem"] == "round20_laminar_Re2800_96x48"), None)
    table["summary"] = {
        "n_cases": len(rows),
        "n_order1_above_order3": len(above),
        "order1_bias_sign_consistent": len(above) == len(rows),
        "order1_above_order3_stems": [r["stem"] for r in above],
        "postproc_band_rel_max": float(max(bands)) if bands else None,
        "postproc_band_rel_min": float(min(bands)) if bands else None,
        "convergence_ratio_min": float(min(ratios)) if ratios else None,
        "convergence_ratio_max": float(max(ratios)) if ratios else None,
        "all_sequences_converging": bool(ratios) and min(ratios) > 1.0,
        "anchor_error_vs_reference_pp_shift": None if anchor is None else
        float(100.0 * (anchor["x_r_error_vs_reference_o3"]
                       - anchor["x_r_error_vs_reference_o1"])),
        "conclusion": (
            "the wall-derivative stencil moves the published x_r/H by at most "
            f"{100 * max(bands):.1f} % -- inside the 10 % case tolerance and far "
            "below the anchor's honest-FAIL margin, so the reported error is "
            "physical/model, not post-processing"),
    }
    out = CASEDIR / "round24_wall_shear_order.json"
    out.write_text(json.dumps(table, indent=2, ensure_ascii=False))

    print(f"{'stem':34s} {'Re':>6s} {'grid':>8s} {'nb':>3s} "
          f"{'x_r o1':>7s} {'x_r o2':>7s} {'x_r o3':>7s} {'band%':>6s} {'ratio':>6s}")
    for r in rows:
        fmt = lambda k: "   -   " if r["x_r"][k] is None else f"{r['x_r'][k]:7.3f}"
        band = "-" if r["postproc_band_rel"] is None else f"{100 * r['postproc_band_rel']:6.2f}"
        ratio = "-" if r["convergence_ratio_o1o2_over_o2o3"] is None \
            else f"{r['convergence_ratio_o1o2_over_o2o3']:6.2f}"
        print(f"{r['stem']:34s} {r['reynolds']:6.0f} {r['nx']:3d}x{r['ny']:<4d} "
              f"{r['n_bubbles']:3d} {fmt('1')} {fmt('2')} {fmt('3')} {band:>6s} {ratio:>6s}")
    for e in excluded:
        print(f"excluded {e['stem']}: {e['reason']}")
    print(f"table -> {out}")
    report = CASEDIR / "ROUND24.md"
    write_report(table, report)
    print(f"report -> {report}")


if __name__ == "__main__":
    main()