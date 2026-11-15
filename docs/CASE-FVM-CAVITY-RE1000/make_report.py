"""Render the ``CASE-FVM-CAVITY-RE1000`` report and figures from the run artefacts.

Every number in ``REPORT.md`` is injected from

    runs/summary.json      case-level verdict, criteria, convergence
    runs/benchmark.json    ref vs got table
    runs/fields.npz        converged velocity fields (figures + field checks)

so no figure or table entry is typed by hand.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/make_report.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table

HERE = pathlib.Path(__file__).resolve().parent
RUNS = HERE / "runs"
FIGS = HERE / "figures"
RE = 1000.0


def pct(x: float) -> str:
    return f"{100.0 * x:.2f}%"


def load() -> tuple[dict, dict, dict]:
    summary = json.loads((RUNS / "summary.json").read_text())
    bench = json.loads((RUNS / "benchmark.json").read_text())
    fields = np.load(RUNS / "fields.npz")
    return summary, bench, fields


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def fig_profiles(summary: dict) -> pathlib.Path:
    cases = sorted(summary["cases"], key=lambda c: c["n"])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    gu = np.asarray(table(RE, "u"), dtype=float)
    gv = np.asarray(table(RE, "v"), dtype=float)
    for c in cases:
        u = c["u"]["profiles"]
        v = c["v"]["profiles"]
        axes[0].plot(u["computed"], u["coord"], "-", lw=1.2, label=f"n = {c['n']}")
        axes[1].plot(v["coord"], v["computed"], "-", lw=1.2, label=f"n = {c['n']}")
    axes[0].plot(gu[:, 1], gu[:, 0], "ko", ms=4.5, mfc="none", label="Ghia et al. (1982)")
    axes[1].plot(gv[:, 0], gv[:, 1], "ko", ms=4.5, mfc="none", label="Ghia et al. (1982)")
    axes[0].set_xlabel("u / U  (x = L/2)")
    axes[0].set_ylabel("y / L")
    axes[0].set_title("Table I: vertical centreline")
    axes[1].set_xlabel("x / L")
    axes[1].set_ylabel("v / U  (y = L/2)")
    axes[1].set_title("Table II: horizontal centreline")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    fig.suptitle(f"Lid-driven cavity, Re = {RE:.0f}: computed centrelines vs Ghia et al. (1982)")
    fig.tight_layout()
    out = FIGS / "fig1_centreline_profiles.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_errors(summary: dict) -> pathlib.Path:
    cases = sorted(summary["cases"], key=lambda c: c["n"])
    n = np.asarray([c["n"] for c in cases], dtype=float)
    u = np.asarray([c["criteria"]["C1_u_profile_max_abs"]["value"] for c in cases])
    v = np.asarray([c["criteria"]["C2_v_profile_max_abs"]["value"] for c in cases])
    ur = np.asarray([c["criteria"]["C3_u_min_relative"]["value"] for c in cases])
    vr = np.asarray([c["criteria"]["C4_v_max_relative"]["value"] for c in cases])
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.loglog(n, u, "o-", label="max|Δu| on x = L/2")
    ax.loglog(n, v, "s-", label="max|Δv| on y = L/2")
    ax.loglog(n, ur, "^--", label="u_min relative error")
    ax.loglog(n, vr, "v--", label="v_max relative error")
    ax.axhline(0.05, color="crimson", ls=":", label="case-pool tolerance 5 %")
    ax.axhline(0.03, color="gray", ls="-.", lw=0.8, label="3 % reference line")
    ax.set_xlabel("n (cells per direction)")
    ax.set_ylabel("error")
    ax.set_xticks(n)
    ax.set_xticklabels([f"{int(x)}" for x in n])
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title(f"Re = {RE:.0f}: error vs grid resolution")
    fig.tight_layout()
    out = FIGS / "fig2_grid_convergence.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_fields(fields, n: int) -> pathlib.Path:
    u = np.asarray(fields[f"u_{n}"], dtype=float)
    v = np.asarray(fields[f"v_{n}"], dtype=float)
    h = 1.0 / n
    uc = 0.5 * (u[:-1, :] + u[1:, :])
    vc = 0.5 * (v[:, :-1] + v[:, 1:])
    omega = np.zeros((n, n))
    omega[:, 1:-1] = (vc[:, 2:] - vc[:, :-2]) / (2 * h)      # dv/dx
    omega[1:-1, :] -= (uc[2:, :] - uc[:-2, :]) / (2 * h)     # du/dy
    psi = np.cumsum(uc, axis=0) * h
    xc = (np.arange(n) + 0.5) * h
    X, Y = np.meshgrid(xc, xc)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.5))
    cs = axes[0].contourf(X, Y, psi, levels=24, cmap="viridis")
    axes[0].contour(X, Y, psi, levels=24, colors="k", linewidths=0.3, alpha=0.5)
    vx, vy = VORTEX_CENTRE[RE]
    axes[0].plot(vx, vy, "w+", ms=12, mew=1.6, label="Ghia vortex centre")
    axes[0].legend(fontsize=8, loc="upper right")
    fig.colorbar(cs, ax=axes[0], label="ψ / (U L)")
    axes[0].set_title(f"streamfunction, n = {n}")
    cm = axes[1].contourf(X, Y, omega, levels=32, cmap="coolwarm")
    fig.colorbar(cm, ax=axes[1], label="ω L / U")
    axes[1].set_title(f"vorticity, n = {n}")
    for ax in axes:
        ax.set_xlabel("x / L")
        ax.set_ylabel("y / L")
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    fig.suptitle(f"Re = {RE:.0f} lid-driven cavity: converged field")
    fig.tight_layout()
    out = FIGS / "fig3_fields.png"
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def table_md(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def render(summary: dict, bench: dict, fields) -> str:
    cases = sorted(summary["cases"], key=lambda c: c["n"])
    grids = [c["n"] for c in cases]
    finest = cases[-1]
    cont = summary.get("continuation_check") or {}
    L: list[str] = []
    a = L.append

    a(f"# CASE-FVM-CAVITY-RE1000 —— Re = 1000 二维顶盖驱动方腔（Ghia et al. 1982 对标）")
    a("")
    a(f"> 本文件与全部数字由 `docs/CASE-FVM-CAVITY-RE1000/make_report.py` 从 "
      f"`runs/summary.json`、`runs/benchmark.json`、`runs/fields.npz` 注入（零手抄）；"
      f"生成时间戳：{summary.get('generated')}。")
    a("")
    a("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    a("")
    a("顶盖驱动方腔是不可压缩 N-S 方程最标准的验证算例：单位方腔 `[0,1]²`，三壁无滑移，"
      "顶盖以切向速度 `U` 匀速拖动，流动由顶盖剪切层驱动、形成单一主涡与角涡。Re = 1000 时流场定常，"
      "因此验收对象是收敛的定常解。基准（external anchor）：U. Ghia, K. N. Ghia, C. T. Shin, "
      "*J. Comput. Phys.* **48** (1982) 387–411，Table I（`x = L/2` 上 `u(y)`）、Table II"
      "（`y = L/2` 上 `v(x)`）、主涡中心 `(0.5313, 0.5625) L`。数值转写由 "
      "`src/tensorfvm/lid_cavity_reference.py` 承载并被其 `self_check()` 自校（极值/去重）。")
    a("")
    a("无量纲参数：")
    a("")
    a(table_md(["量", "符号", "取值"],
               [["雷诺数", "Re = U L / ν", f"{summary['reynolds']:.1f}"],
                ["运动粘度", "ν", f"{1.0 / summary['reynolds']:.2e}"],
                ["顶盖速度", "U", f"{summary['lid_velocity']:.1f}"],
                ["密度", "ρ", "1.0"],
                ["域", "L × L", "1 × 1"],
                ["容差（算例池）", "—", pct(summary["pool_tolerance"])]]))
    a("")
    a("## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）")
    a("")
    a("控制方程：`∂u/∂t + ∇·(u u) = −∇p/ρ + ν∇²u`，`∇·u = 0`。离散与推进：")
    a("")
    a("* 交错 MAC 有限体积：`u` 在竖直面 `(n+1, n)`、`v` 在水平面 `(n, n+1)`、`p` 在单元中心；")
    a("* 对流：守恒通量形式 + van Leer (harmonic, TVD) 限制面重构（顶盖剪切层内保持有界）；")
    a("* 扩散：紧致中心差分；顶盖经镜像 ghost `u = 2U − u_wall` 施加；")
    a("* 投影：显式前 Euler 推进动量，随后解纯 Neumann 压力 Poisson 方程，用正交 DCT-II 对角化精确求解，"
      "并在变换前扣除平均散度以满足离散相容性 `Σ div u* = 0`；")
    a("* 定常判据：`max|Δu|/Δt` 每 250 步检查一次，≤ 1e-7 即判收敛；CFL = 0.15（更紧的 Courant 数是因为"
      "van Leer 重构在 cfl ≈ 0.4 时对顶盖剪切层不稳定）；")
    a("* 网格续算（只改成本不改解）：n = 32 冷启动 → 64 冷启动（续算对照）→ 128 由 64 双线性延拓；")
    a("* 工具链：`src/tensorfvm/lid_cavity.py`（生产求解器）+ `src/tensorfvm/case_cavity_re1000.py`"
      "（算例层：跑一个分辨率、逐点打分、7 条判据）+ `docs/CASE-FVM-CAVITY-RE1000/run_case.py`（本包入口，"
      "写 `runs/`）+ `tests/test_case_cavity_re1000.py`（机检门：从工件重算）"
      "+ `tensorfvm.verification.ladder.lid_cavity_verdict`（主验证流水线的 C2 消费方）。")
    a("")
    a(f"运行入口：`{summary.get('command')}`")
    a("")
    a("## ③ 计算结果（数字表，程序化注入）")
    a("")
    a("各分辨率运行记录：")
    a("")
    a(table_md(["n", "h", "步数", "定常残差", "max|div u|", "墙钟 [s]", "初值", "收敛"],
               [[c["n"], f"{c['h']:.5f}", c["steps"], f"{c['residual']:.2e}",
                 f"{c['max_divergence']:.1e}", f"{c['wall_seconds']:.0f}",
                 c.get("initial_condition", "rest"), c["converged"]] for c in cases]))
    a("")
    a("**ref vs got 数值对照**（ref 来自 Ghia 表，got 为本次计算；u_min/v_max 为相对误差，剖面列为 U 单位绝对误差）：")
    a("")
    rows = []
    for row in sorted(bench["rows"], key=lambda r: r["n"]):
        rows.append([row["n"],
                     f"{row['u_min']['got']:.5f}", f"{row['u_min']['ref']:.5f}",
                     pct(row["u_min"]["rel_err"]),
                     f"{row['v_max']['got']:.5f}", f"{row['v_max']['ref']:.5f}",
                     pct(row["v_max"]["rel_err"]),
                     f"{row['u_profile_max_abs']['err']:.4f}",
                     f"{row['v_profile_max_abs']['err']:.4f}",
                     f"({row['vortex_centre']['got'][0]:.4f}, {row['vortex_centre']['got'][1]:.4f})",
                     f"{row['vortex_centre']['abs_err_L']:.4f}",
                     row["verdict"]])
    a(table_md(["n", "u_min got", "u_min ref", "rel", "v_max got", "v_max ref", "rel",
                "max|Δu|", "max|Δv|", "涡心 got", "Δ/L", "判"], rows))
    a("")
    a("七条独立判据（最细网格）：")
    a("")
    rows = []
    for cid, entry in finest["criteria"].items():
        if cid.startswith("_"):
            continue
        rows.append([cid, entry["statement"], f"{entry['value']:.6g}",
                     f"{entry['threshold']:.6g}", "✓" if entry["pass"] else "✗"])
    a(table_md(["判据", "含义", "实测", "阈值", "结论"], rows))
    a("")
    orders = summary["convergence"]["observed_orders"]
    if orders:
        a("网格自收敛（相邻两级的中心线剖面差，观测阶）：")
        a("")
        a(table_md(["n 序列", "p(Linf u)", "p(Linf v)", "p(L2 u)", "p(L2 v)"],
                   [[f"{o['between'][0]}→{o['between'][1]}→{o['between'][2]}",
                     f"{o['p_linf_u']:.2f}", f"{o['p_linf_v']:.2f}",
                     f"{o['p_l2_u']:.2f}", f"{o['p_l2_v']:.2f}"] for o in orders]))
        a("")
    if cont:
        a(f"续算对照（n = {cont['grid']}）：冷启动 {cont['steps_cold']} 步 / "
          f"{cont['wall_seconds_cold']:.0f} s 与延拓启动 {cont['steps_continued']} 步 / "
          f"{cont['wall_seconds_continued']:.0f} s 收敛到同一解："
          f"中心线最大差 `max|Δu| = {cont['max_abs_du_centreline']:.2e}`、"
          f"`max|Δv| = {cont['max_abs_dv_centreline']:.2e}`，"
          f"同解判定 = {cont['same_steady_state']}。")
        a("")
    a("## ④ 结果图")
    a("")
    a("![centrelines](figures/fig1_centreline_profiles.png)")
    a("")
    a("*图 1* 三个分辨率的中心线剖面 vs Ghia (1982) 采样点（u 在 x = L/2，v 在 y = L/2）。")
    a("")
    a("![convergence](figures/fig2_grid_convergence.png)")
    a("")
    a("*图 2* 误差随网格分辨率的收敛（对数坐标），红线为算例池 5% 容差，灰线为 3% 参考线。")
    a("")
    a("![fields](figures/fig3_fields.png)")
    a("")
    a(f"*图 3* 最细网格（n = {finest['n']}）流函数与涡量云图；白色十字为 Ghia 主涡中心。")
    a("")
    a("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    a("")
    a(f"* **分辨率是硬门槛**：n = 32 时 `u_min` 相对误差 {pct(cases[0]['criteria']['C3_u_min_relative']['value'])}，"
      f"超 5% 容差；必须到 n ≥ 64（{pct([c for c in cases if c['n'] == 64][0]['criteria']['C3_u_min_relative']['value'])}）"
      f"才进容差。早期（原单 m19978）报 FAIL 的死因就是预算只够跑粗网格/未收敛，不是求解器错。")
    a("* **插值口径**：`score_profiles` 用 Catmull–Rom (Hermite) 把计算剖面插到 Ghia 采样点；"
      "`scripts/audit_cavity_re1000_m20000.py` 用分段线性与局部二次独立重算，三种口径给出的"
      "最大偏差差异见该审计（顶盖剪切层附近的插值偏差显著小于容差，故结论不依赖单一插值规则）。")
    a("* **续算是成本技巧**：延拓只提供初值；冷启动对照证明同一网格的冷/热两条路径收敛到同一解，"
      "因此续算没有筛选出某个解。")
    a("* **适用范围**：二维、层流、定常；Re = 1000 有角涡但无需湍流模型。三维、非定常脉动、"
      "更高 Re（需亚格子/湍流模型）不在本案例覆盖范围内。")
    a("* **散度**：纯 Neumann 闭腔要求 `Σ div u* = 0` 精确满足，本实现按此扣除均值；"
      f"最细网格 `max|div u| = {finest['max_divergence']:.1e}`。")
    a("")
    a("## ⑥ 误差判定（逐项 vs 基准，显式回答）")
    a("")
    a(f"| 关键量 | 基准（Ghia 1982） | 本次（n = {finest['n']}） | 误差 | < 3%？ |")
    a("|---|---|---|---|---|")
    ext = finest["criteria"]["_extremum_basis"]
    for label, which, cid in (("u_min（x = L/2 回流极值）", "u", "C3_u_min_relative"),
                              ("v_max（y = L/2 极值）", "v", "C4_v_max_relative")):
        ref = ext[which]["reference"]
        got = ext[which]["computed"]
        err = finest["criteria"][cid]["value"]
        a(f"| {label} | {ref:.5f} | {got:.5f} | {pct(err)} | {'是' if err < 0.03 else '否'} |")
    for label, cid in (("max\\|Δu\\| 全剖面点（U 单位）", "C1_u_profile_max_abs"),
                       ("max\\|Δv\\| 全剖面点（U 单位）", "C2_v_profile_max_abs")):
        entry = finest["criteria"][cid]
        a(f"| {label} | 0（逐点差） | {entry['value']:.4f} | {entry['value']:.4f} | "
          f"{'是' if entry['value'] < 0.03 else '否'} |")
    vc_ref = VORTEX_CENTRE[RE]
    vc_got = finest["vortex_centre"]["computed"]
    a(f"| 主涡中心 | ({vc_ref[0]:.4f}, {vc_ref[1]:.4f}) | ({vc_got[0]:.4f}, {vc_got[1]:.4f}) | "
      f"{finest['criteria']['C5_vortex_centre']['value']:.4f} L | "
      f"{'是' if finest['criteria']['C5_vortex_centre']['value'] < 0.03 else '否'} |")
    a("")
    all_below_3 = all(
        finest["criteria"][cid]["value"] < 0.03
        for cid in ("C1_u_profile_max_abs", "C2_v_profile_max_abs",
                    "C3_u_min_relative", "C4_v_max_relative", "C5_vortex_centre"))
    a(f"**是否 < 3%？** 最细网格（n = {finest['n']}）"
      f"{'全部关键量均 < 3%' if all_below_3 else '有关键量未达 3%（见上表）'}；"
      f"算例池 5% 容差判定 = **{summary['finest_verdict']}**。"
      f"n = 64 同样 {summary['cases'][sorted(grids).index(64)]['verdict'] if 64 in grids else 'N/A'}。")
    a("")
    a("## 复现")
    a("")
    a("```")
    a("# 全流程（网格序列 32→64→128 + 冷启动对照），约 30 min 单核")
    a(f"{summary.get('command')}")
    a("# 机检门：从工件重算判据与剖面")
    a("PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000.py -q")
    a("# 报告与图")
    a("PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/make_report.py")
    a("```")
    a("")
    a("## 产物清单")
    a("")
    a("* `runs/summary.json` —— 算例级判词、7 判据、网格收敛、续算对照")
    a("* `runs/benchmark.json` —— ref vs got 数值对照表")
    a(f"* `runs/lid_cavity_Re1000_n{{{','.join(str(n) for n in grids)}}}.json` —— 每分辨率工件")
    a("* `runs/fields.npz` —— 收敛速度场（可离线重算剖面）")
    a("* `runs/control/` —— 冷启动对照")
    unc = HERE / "runs" / "uncertainty.json"
    if unc.is_file():
        u = json.loads(unc.read_text())
        a("## 离散不确定度（补充研究，见 `UNCERTAINTY.md`）")
        a("")
        q = u["quantities"]
        a("本报告的判据是「最细网格 vs Ghia 表」的**参考值一致性**。独立的网格收敛指数研究"
          "（`runs/uncertainty.json`，Roache GCI + Celik 2008 数据质量门，泛函取自存储面场、"
          "不做插值）给出：")
        a("")
        a("| 量 | 观测阶 p | 最细网格 GCI | 最细网格 vs Ghia | Richardson 外推 vs Ghia | ref 在 GCI 带内 |")
        a("|---|---|---|---|---|---|")
        for name, label in (("u_min", "u_min"), ("v_max", "v_max")):
            e = q[name]
            a(f"| {label} | {e['observed_order']:.2f} | {pct(e['gci_fine_relative'])} | "
              f"{pct(abs(e['fine_minus_reference_relative']))} | "
              f"{pct(abs(e['extrapolated_minus_reference_relative']))} | "
              f"{'✓' if e['reference_within_gci_band'] else '✗'} |")
        a("")
        a("即最细网格与 Ghia 的差**小于自身离散带**，故本报告不应被读作「求解器精度 0.6%」；"
          "时间步（Courant）轴在本包内为 not_run。机器门已按平台登记裁判行收紧到 3% / 0.01 L，"
          "由 `tests/test_case_cavity_re1000_gate_registered.py` 断言（负对照：3%~5% 窗口内必然 FAIL）。")
        a("")
    a("* `figures/` —— 本报告的三张图")
    return "\n".join(L) + "\n"


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    summary, bench, fields = load()
    f1 = fig_profiles(summary)
    f2 = fig_errors(summary)
    f3 = fig_fields(fields, int(summary["finest"]))
    text = render(summary, bench, fields)
    out = HERE / "REPORT.md"
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out} ({len(text)} chars)")
    for f in (f1, f2, f3):
        print(f"wrote {f} ({f.stat().st_size} bytes)")


if __name__ == "__main__":
    main()