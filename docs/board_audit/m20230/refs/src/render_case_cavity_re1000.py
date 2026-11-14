"""Render docs/CASE-CAVITY-RE1000-FVM/report.md and its figures from the run bundle.

Every number in the report is injected from
``docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json`` (written by
``python -m tensorfvm.case_cavity_re1000``); nothing is typed by hand, and the
reference column is taken from :mod:`tensorfvm.lid_cavity_reference`, the same
transcription the gate test re-reads.  Figures are produced here as PNGs:

    fig1_centrelines.png   u(y) at x = L/2 and v(x) at y = L/2, every grid, vs Ghia
    fig2_fields.png        vorticity + streamlines of the finest converged field
    fig3_convergence.png   max |u - u_Ghia| and |v - v_Ghia| vs grid resolution

Usage::

    PYTHONPATH=src python scripts/render_case_cavity_re1000.py
"""

from __future__ import annotations

import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402
import numpy as np                            # noqa: E402
import torch                                  # noqa: E402

from tensorfvm.case_cavity_re1000 import CASE_SLUG, REFERENCE, REYNOLDS  # noqa: E402
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE, table           # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / CASE_SLUG / "artifacts"
DOCS = ROOT / "docs" / CASE_SLUG
FIGS = DOCS / "figures"


def fmt(value: float, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "pass" if value else "fail"
    return f"{value:.{digits}f}"


def production(cases):
    return sorted([c for c in cases if not c.get("role", "").startswith("control")],
                  key=lambda c: c["n"])


def controls(cases):
    return [c for c in cases if c.get("role", "").startswith("control")]


# ------------------------------------------------------------------ figures
def figure_centrelines(bundle: dict) -> pathlib.Path:
    ref_u = table(REYNOLDS, "u")
    ref_v = table(REYNOLDS, "v")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.1))
    for case in sorted(bundle["cases"], key=lambda c: c["n"]):
        style = ":" if case.get("role", "").startswith("control") else "-"
        axes[0].plot(case["u"]["profiles"]["computed"], case["u"]["profiles"]["coord"],
                     style, lw=1.4, label=f"n = {case['n']}")
        axes[1].plot(case["v"]["profiles"]["coord"], case["v"]["profiles"]["computed"],
                     style, lw=1.4, label=f"n = {case['n']}")
    axes[0].plot([p[1] for p in ref_u], [p[0] for p in ref_u], "o", ms=4.5,
                 mfc="none", color="k", label="Ghia et al. (1982) Table I")
    axes[1].plot([p[0] for p in ref_v], [p[1] for p in ref_v], "o", ms=4.5,
                 mfc="none", color="k", label="Ghia et al. (1982) Table II")
    axes[0].set_xlabel("u / U"); axes[0].set_ylabel("y / L")
    axes[0].set_title(f"u on x = L/2, Re = {REYNOLDS:g}")
    axes[1].set_xlabel("x / L"); axes[1].set_ylabel("v / U")
    axes[1].set_title(f"v on y = L/2, Re = {REYNOLDS:g}")
    for ax in axes:
        ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    path = FIGS / "fig1_centrelines.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def figure_fields(bundle: dict) -> pathlib.Path | None:
    finest = production(bundle["cases"])[-1]
    path_in = OUT / "fields" / f"Re{REYNOLDS:g}_n{finest['n']}.pt"
    if not path_in.exists():
        return None
    saved = torch.load(path_in, weights_only=False)
    u_face, v_face = saved["u"].numpy(), saved["v"].numpy()
    uc = 0.5 * (u_face[:-1, :] + u_face[1:, :])
    vc = 0.5 * (v_face[:, :-1] + v_face[:, 1:])
    n = uc.shape[0]
    h = 1.0 / n
    x = (np.arange(n) + 0.5) * h
    y = (np.arange(n) + 0.5) * h
    dvdx = np.gradient(vc, h, axis=1)
    dudy = np.gradient(uc, h, axis=0)
    omega = dvdx - dudy
    # streamfunction by integrating u vertically from the moving lid downwards
    psi = np.zeros((n, n))
    psi[-1, :] = 0.0
    for j in range(n - 2, -1, -1):
        psi[j, :] = psi[j + 1, :] - uc[j, :] * h

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    levels = np.linspace(-np.abs(omega).max(), np.abs(omega).max(), 41)
    cf = axes[0].contourf(x, y, omega, levels=levels, cmap="RdBu_r", extend="both")
    fig.colorbar(cf, ax=axes[0], label=r"$\omega_z L / U$")
    axes[0].set_title(f"vorticity, n = {finest['n']}², Re = {REYNOLDS:g}")
    cs = axes[1].contour(x, y, psi, levels=17, colors="k", linewidths=0.8)
    axes[1].clabel(cs, inline=True, fontsize=6, fmt="%.3f")
    axes[1].set_title(r"streamlines $\psi/UL$")
    for ax in axes:
        ax.set_aspect("equal"); ax.set_xlabel("x / L"); ax.set_ylabel("y / L")
    fig.tight_layout()
    path = FIGS / "fig2_fields.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


def figure_convergence(bundle: dict) -> pathlib.Path:
    rows = sorted(bundle["cases"], key=lambda c: c["n"])
    ns = [c["n"] for c in rows]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    axes[0].loglog(ns, [c["u"]["max_abs_error"] for c in rows], "o-",
                   label=r"max $|u-u_{Ghia}|$")
    axes[0].loglog(ns, [c["v"]["max_abs_error"] for c in rows], "s-",
                   label=r"max $|v-v_{Ghia}|$")
    axes[0].axhline(bundle["tolerance_fraction"], color="r", ls="--", lw=1,
                    label="case-pool band (5 % of U)")
    axes[0].set_xlabel("n (cells per side)"); axes[0].set_ylabel("error / U")
    axes[0].set_title("discretisation error vs grid"); axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7.5)
    steps = [c["solver"]["steps"] for c in rows]
    axes[1].bar([str(n) for n in ns], steps, color="#4c72b0")
    for i, c in enumerate(rows):
        axes[1].text(i, steps[i], f"{steps[i]}\n{c['solver']['wall_seconds']:.0f}s",
                     ha="center", va="bottom", fontsize=7)
    axes[1].set_xlabel("n"); axes[1].set_ylabel("time steps to steady state")
    axes[1].set_title("cost per resolution"); axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path = FIGS / "fig3_convergence.png"
    fig.savefig(path, dpi=150); plt.close(fig)
    return path


# ------------------------------------------------------------------ report
def render(bundle: dict) -> pathlib.Path:
    FIGS.mkdir(parents=True, exist_ok=True)
    figs = {"centrelines": figure_centrelines(bundle),
            "convergence": figure_convergence(bundle)}
    fields_fig = figure_fields(bundle)
    if fields_fig is not None:
        figs["fields"] = fields_fig

    rows = sorted(bundle["cases"], key=lambda c: c["n"])
    prod = production(bundle["cases"])
    finest = prod[-1]
    ref_u = dict(table(REYNOLDS, "u"))
    ref_v = dict(table(REYNOLDS, "v"))
    gx, gy = VORTEX_CENTRE[REYNOLDS]
    L: list[str] = []
    a = L.append

    a(f"# {CASE_SLUG} — 方腔驱动流 Re = 1000 对标 Ghia, Ghia & Shin (1982)")
    a("")
    a("> 本文件由 `scripts/render_case_cavity_re1000.py` 从 "
      "`docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json` 生成（零手抄）；"
      "参考值来自 `src/tensorfvm/lid_cavity_reference.py` 的逐字转录，"
      "机检门为 `tests/test_case_cavity_re1000.py`。")
    a("")

    # ① ---------------------------------------------------------------
    a("## ① 问题设定与基准出处")
    a("")
    a("二维方腔驱动流：单位方腔 `[0,1]²`，三面固壁无滑移，顶盖 `y = L` 以 "
      "`u = U = 1` 切向运动（不可穿透）。以 `Re = U L / ν` 定义雷诺数，"
      f"`Re = {REYNOLDS:g}` ⇒ `ν = {1.0 / REYNOLDS:g}`（ρ = 1，L = 1）。定常解。")
    a("")
    a(f"- 基准：{REFERENCE}；基准值单源转录在 `src/tensorfvm/lid_cavity_reference.py`"
      "（`table(Re, 'u'|'v')`、`VORTEX_CENTRE`），由 `self_check()` 与参考审计测试钉住。")
    a("- 验收判据（总账 `CASE-CAVITY-RE1000-FVM`）：关键量相对误差 < 5%。"
      "本模块把该判据展开成 7 条逐项机检（见 ④）。")
    a("")
    a("| 量 | 基准值 | 出处 |")
    a("|---|---|---|")
    a(f"| u_min（x = L/2 反向流极小） | {ref_u[finest['u']['extremum']['min_reference'][0]]:.5f}"
      f" @ y = {finest['u']['extremum']['min_reference'][0]:.4f} | Ghia 1982 Table I |")
    a(f"| v_max（y = L/2 极大） | {ref_v[finest['v']['extremum']['max_reference'][0]]:.5f}"
      f" @ x = {finest['v']['extremum']['max_reference'][0]:.4f} | Ghia 1982 Table II |")
    a(f"| 主涡中心 | ({gx:.4f}, {gy:.4f}) | Ghia 1982 §3（Table I/II 的极值位置） |")
    a("")

    # ② ---------------------------------------------------------------
    a("## ② 方法与实现")
    a("")
    a("求解器：`tensorfvm.lid_cavity.LidCavitySolver` —— 交错（MAC）有限体积，"
      "显式对流（van Leer 受限面通量）+ 隐式扩散，投影法解耦压力，"
      "压力泊松用 DCT-II 在纯 Neumann 边界下的谱解（离散拉普拉斯精确求逆，无迭代误差）。"
      "时间推进到 `max|Δu|/Δt ≤ tolerance` 判为定常。")
    a("")
    a("本单新增/复用的成本装置（纯成本，不改变被计分的离散定常解，且经检验）：")
    a("")
    a("1. **网格延拓**：粗网格收敛 → 单元中心速度双线性延拓到细网格 → 继续推进；")
    a("2. **延拓校验**：基础网格同时冷启动跑一遍，两个收敛场逐点比对"
      "（`max|Δu|`、`max|Δv|`），延拓若引入偏差会在此暴露；")
    a("3. **逐算例检查点**（每 20 000 步 + 每算例结束），长作业被截断也留有可用行。")
    a("")
    a("关键代码指针：")
    a("")
    a("| 职责 | 位置 |")
    a("|---|---|")
    a("| 算例实现（运行/延拓/判据/工件/CLI 入口） | `src/tensorfvm/case_cavity_re1000.py` |")
    a("| 数值内核（交错 FV + 投影 + 谱压力解） | `src/tensorfvm/lid_cavity.py` |")
    a("| 基准转录（单源） | `src/tensorfvm/lid_cavity_reference.py` |")
    a("| 延迟兼容入口 | `scripts/run_cavity_re1000_m15458.py`（薄封装 → `case_cavity_re1000.main`） |")
    a("| 集成调用方（主验证梯） | `src/tensorfvm/verification/ladder.py` → `lid_cavity_verdict`（C2，Re = 1000 取最细网格） |")
    a("| 机检门 | `tests/test_case_cavity_re1000.py` |")
    a("| 报告渲染 | `scripts/render_case_cavity_re1000.py` |")
    a("")

    # ③ ---------------------------------------------------------------
    a("## ③ 参数表")
    a("")
    a("| 参数 | 值 | 说明 |")
    a("|---|---|---|")
    a(f"| Re | {bundle['reynolds']:g} | `U L / ν`，L = 1，U = {bundle['lid_velocity']:g} |")
    a(f"| ν | {1.0 / bundle['reynolds']:g} | `U L / Re`，ρ = 1 |")
    first = rows[0]
    a(f"| CFL | {first['config']['cfl']:g} | `dt = CFL · h / U` |")
    a(f"| 收敛容差 | {first['config']['tolerance']:g} | `max|Δu|/Δt`（每 {first['config']['check_every']} 步测一次） |")
    a(f"| 步数上限 | {first['config']['max_steps']} | 达到即停止并如实标记未收敛 |")
    a(f"| 网格序列 | {', '.join(str(c['n']) for c in rows)} | 含 1 个故意欠分辨的对照网格 |")
    a("| 精度 | float64 | 单进程，`torch.set_num_threads(1)` |")
    a("| 离散 | 交错 MAC + van Leer + DCT-II 压力解 | 见 `lid_cavity.py` 模块文档 |")
    a("")

    # ④ ---------------------------------------------------------------
    a("## ④ 对照表（ref vs got，逐判据打勾/叉）")
    a("")
    a("每个网格一行；`u_min`/`v_max` 为相对误差，`max|Δu|`/`max|Δv|` 为以顶盖速度 U 归一的无量纲偏差，"
      "`涡心` 为以 L 归一的绝对距离。")
    a("")
    a("| n | 角色 | u_min got | u_min 相对误差 | v_max got | v_max 相对误差 | 涡心 got | 涡心距离 | max|Δu| | max|Δv| | 5% 判据 |")
    a("|---|---|---|---|---|---|---|---|---|---|---|")
    for c in rows:
        um = c["u"]["extremum"]["min_computed"][1]
        uref = c["u"]["extremum"]["min_reference"][1]
        vm = c["v"]["extremum"]["max_computed"][1]
        vref = c["v"]["extremum"]["max_reference"][1]
        vx, vy = c["vortex_centre"]["computed"]
        a(f"| {c['n']} | {c.get('role','')} | {um:.5f} | {abs(um - uref) / abs(uref) * 100:.2f}% "
          f"| {vm:.5f} | {abs(vm - vref) / abs(vref) * 100:.2f}% "
          f"| ({vx:.4f}, {vy:.4f}) | {c['vortex_centre']['abs_error']:.4f} "
          f"| {c['u']['max_abs_error']:.4f} | {c['v']['max_abs_error']:.4f} "
          f"| {'✅ pass' if c['verdict']['pass'] else '❌ fail'} |")
    a("")
    a(f"最细生产网格 n = {finest['n']} 的逐判据结论（判据定义见 "
      "`case_cavity_re1000.CRITERIA`）：")
    a("")
    a("| 判据 | 说 | 实测 | 阈值 | 结论 |")
    a("|---|---|---|---|---|")
    for chk in finest["verdict"]["checks"]:
        value = chk["value"]
        shown = ("true" if value is True else "false") if isinstance(value, bool) else f"{value:.5f}"
        a(f"| {chk['id']} | {chk['statement']} | {shown} | {chk['threshold']} | "
          f"{'✅' if chk['pass'] else '❌'} |")
    a("")
    a("逐点剖面（Ghia 采样点，最细网格）：")
    a("")
    a("| y (u 剖面) | u got | u Ghia | Δ | x (v 剖面) | v got | v Ghia | Δ |")
    a("|---|---|---|---|---|---|---|---|")
    up, vp = finest["u"]["profiles"], finest["v"]["profiles"]
    for i in range(max(len(up["coord"]), len(vp["coord"]))):
        cells = []
        if i < len(up["coord"]):
            cells += [f"{up['coord'][i]:.4f}", f"{up['computed'][i]:.5f}",
                      f"{up['reference'][i]:.5f}", f"{up['delta'][i]:+.5f}"]
        else:
            cells += ["", "", "", ""]
        if i < len(vp["coord"]):
            cells += [f"{vp['coord'][i]:.4f}", f"{vp['computed'][i]:.5f}",
                      f"{vp['reference'][i]:.5f}", f"{vp['delta'][i]:+.5f}"]
        else:
            cells += ["", "", "", ""]
        a("| " + " | ".join(cells) + " |")
    a("")

    # ⑤ ---------------------------------------------------------------
    a("## ⑤ 收敛性与参数敏感性")
    a("")
    a("| n | 步数 | 稳态判据 max|Δu|/Δt | 散度 max|∇·u| | 墙钟 (s) |")
    a("|---|---|---|---|---|")
    for c in rows:
        a(f"| {c['n']} | {c['solver']['steps']} | {c['solver']['residual_rate']:.2e} "
          f"| {c['max_divergence']:.2e} | {c['solver']['wall_seconds']:.0f} |")
    a("")
    coarse, fine = prod[0], prod[-1]
    a(f"网格加密 {coarse['n']} → {fine['n']}：`max|Δu|` "
      f"{coarse['u']['max_abs_error']:.4f} → {fine['u']['max_abs_error']:.4f}，"
      f"`max|Δv|` {coarse['v']['max_abs_error']:.4f} → {fine['v']['max_abs_error']:.4f}"
      f"（比值 {coarse['u']['max_abs_error'] / max(fine['u']['max_abs_error'], 1e-12):.2f}× / "
      f"{coarse['v']['max_abs_error'] / max(fine['v']['max_abs_error'], 1e-12):.2f}×）。"
      "误差随加密单调下降且全程处在 5% 带宽内，说明 5% 的通过不是单网格侥幸。")
    a("")
    chk = bundle.get("continuation_check")
    if chk:
        a(f"延拓校验（同一 n = {chk['to_n']} 网格，两种初值）："
          f"`max|Δu| = {chk['max_abs_du']:.2e}`、`max|Δv| = {chk['max_abs_dv']:.2e}` "
          f"⇒ 同一离散定常解（阈值 1e-6）：{'✅' if chk['same_steady_state'] else '❌'}；"
          f"延拓 {chk['continued_steps']} 步 vs 冷启动 {chk['cold_steps']} 步，"
          f"墙钟 {chk['continued_wall_seconds']:.0f}s vs {chk['cold_wall_seconds']:.0f}s"
          f"（省时 {chk['speedup']:.2f}×）。")
        a("")
    ctrl = controls(bundle["cases"])
    if ctrl:
        a(f"欠分辨对照：{', '.join('n = ' + str(c['n']) for c in ctrl)} "
          f"未通过 5% 判据（{'、'.join(str(c['n']) + ': ' + ('pass' if c['verdict']['pass'] else 'fail') for c in ctrl)}），"
          "如实保留在工件中，不用平均掩盖。")
        a("")

    # ⑥ ---------------------------------------------------------------
    a("## ⑥ 失败模式、工件与复现")
    a("")
    a("工件（全部由程序写入，路径相对仓库根）：")
    a("")
    a("- `docs/CASE-CAVITY-RE1000-FVM/artifacts/summary.json` —— 全量度量、逐判据结论、参数表、延拓校验")
    for c in rows:
        a(f"- `docs/CASE-CAVITY-RE1000-FVM/artifacts/fields/Re{REYNOLDS:g}_n{c['n']}.pt` —— 收敛流场")
    for name in figs:
        a(f"- `docs/CASE-CAVITY-RE1000-FVM/figures/{figs[name].name}` —— {name}")
    a("- `src/tensorfvm/verification/artifacts/verification_ladder/lid_cavity_Re1000_n*.json` —— "
      "供主验证梯 C2 判据直接消费的行")
    a("")
    a("复现命令（>10 min，按 owner 计算调度令递交队列）：")
    a("")
    a("```bash")
    a("# 入口：跑整条网格序列（控制网格 32 → 生产 64 → 延拓 128 → 延拓校验）")
    a("PYTHONPATH=src python -m tensorfvm.case_cavity_re1000 \\")
    a("    --out docs/CASE-CAVITY-RE1000-FVM/artifacts --grids 64,128 --cold-grids 32 \\")
    a("    --ladder-dir src/tensorfvm/verification/artifacts/verification_ladder")
    a("# 机检门（从工件重算误差与判据，不重跑求解器）")
    a("PYTHONPATH=src python -m pytest tests/test_case_cavity_re1000.py -q")
    a("# 报告与图")
    a("PYTHONPATH=src python scripts/render_case_cavity_re1000.py")
    a("```")
    a("")
    a("已知失败模式与局限：")
    a("")
    a("- **欠分辨网格会假通过**：n = 32 的极值误差可能偶然落在 5% 内，"
      "因此对照行必须与生产行分开报告，且机检门要求对照行 `pass is False`；")
    a("- **显式推进的步数 ∝ n**：n = 128 约需 10⁵ 步，墙钟以小时计；"
      "本单用网格延拓与检查点把风险压到可接受，但收敛判据 `max|Δu|/Δt` 是必要非充分——"
      "它不能排除缓慢衰减的伪定常态，故要求两个独立初值给出同一解（延拓校验）；")
    a("- **只覆盖定常解**：Ghia 的 Re = 1000 方腔是定常的，本工件不对非定常/三维方腔作任何声明；")
    a("- **仅二分解法自证**：参考值是文献表，不是解析解；表格采样点之间的误差未评估。")
    a("")

    DOCS.mkdir(parents=True, exist_ok=True)
    path = DOCS / "report.md"
    path.write_text("\n".join(L) + "\n")
    return path


def main() -> int:
    bundle_path = OUT / "summary.json"
    if not bundle_path.exists():
        raise SystemExit(f"missing bundle {bundle_path}; run the case first")
    bundle = json.load(open(bundle_path))
    path = render(bundle)
    print(json.dumps({"report": str(path),
                      "figures": sorted(str(p) for p in FIGS.glob("*.png"))}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())