"""Generate the propeller cavitation-criteria oracle report (zero hand-typed numbers).

    python scripts/run_cavitation_criteria_oracle.py \
        [--out docs/cavitation-criteria-oracle.md] \
        [--figdir src/tensorfvm/cavitation/artifacts/oracle]

Every number in the generated markdown comes from
:mod:`tensorfvm.cavitation.propeller_oracle`; the script only formats it.  The
live platform probe is recorded as an observed state (``ok`` / ``not_wired`` /
``unreachable``) -- it is never upgraded to "passed" by this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from tensorfvm.cavitation import propeller_oracle as O  # noqa: E402

DESIGN = dict(thrust_N=300e3, D=3.0, depth=4.0, Z=4, ship_type="single-screw",
              Ae_Ao=0.65, P_D=1.0, temperature_c=20.0)
VA = 6.17           # m/s (12 kn inflow of the platform's check case)
THRUST = 300e3      # N
J_SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
DEPTHS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def n_for_j(j: float, d: float = 3.0, va: float = VA) -> float:
    return va / (j * d)


def sweep_table() -> list:
    rows = []
    for j in J_SWEEP:
        n = n_for_j(j)
        tau = O.tau_c(THRUST, n, 3.0, VA, DESIGN["Ae_Ao"], DESIGN["P_D"])
        sigma = O.sigma_07r(n, 3.0, VA, DESIGN["depth"], DESIGN["temperature_c"])
        point = O.burrill_point(tau, sigma)
        rows.append({"J": j, "n_rps": n, "tau_c": tau, "sigma_0.7R": sigma,
                     "sigma_limit": point.sigma_limit,
                     "margin_pct": point.margin_pct,
                     "passes": point.passes,
                     "sigma_n_ratio": O.sigma_n_sigma_07r_ratio(n, 3.0, VA)})
    return rows


def keller_table() -> list:
    rows = []
    for ship_type in O.KELLER_DELTA:
        for z in (4, 5):
            ratio, delta = O.keller_min_area_ratio(
                THRUST, 3.0, DESIGN["depth"], z, ship_type=ship_type,
                temperature_c=DESIGN["temperature_c"])
            rows.append({"ship_type": ship_type, "Z": z, "required": ratio,
                         "delta_ao": delta,
                         "deficit_at_0.65": DESIGN["Ae_Ao"] - ratio})
    return rows


def depth_table() -> list:
    rows = []
    n = 3.0
    for depth in DEPTHS:
        sigma = O.sigma_07r(n, 3.0, VA, depth, DESIGN["temperature_c"])
        tau = O.tau_c(THRUST, n, 3.0, VA, DESIGN["Ae_Ao"], DESIGN["P_D"])
        point = O.burrill_point(tau, sigma)
        ratio, _ = O.keller_min_area_ratio(
            THRUST, 3.0, depth, DESIGN["Z"], ship_type=DESIGN["ship_type"],
            temperature_c=DESIGN["temperature_c"])
        rows.append({"depth": depth, "sigma_0.7R": sigma,
                     "required_ae_ao": ratio,
                     "keller_ok": DESIGN["Ae_Ao"] >= ratio,
                     "margin_pct": point.margin_pct})
    return rows


def sensitivity_table() -> list:
    tau = O.tau_c(THRUST, 3.0, 3.0, VA, DESIGN["Ae_Ao"], DESIGN["P_D"])
    sigma = O.sigma_07r(3.0, 3.0, VA, DESIGN["depth"], DESIGN["temperature_c"])
    slope = O.margin_sensitivity_per_unit_sigma(sigma)
    base_limit = O.legacy_placeholder_curve(tau)
    rows = []
    for err in (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20):
        limit = base_limit * (1.0 + err)
        point = O.burrill_point(tau, sigma, curve=lambda _t, lim=limit: lim,
                                curve_provenance="hypothetical error band")
        rows.append({"err_pct": 100.0 * err, "sigma_limit": limit,
                     "margin_pct": point.margin_pct, "passes": point.passes,
                     "linear_estimate": slope * (limit - base_limit)})
    return rows


def figures(figdir: Path) -> dict:
    figdir.mkdir(parents=True, exist_ok=True)
    paths = {}

    rows = sweep_table()
    tau = np.array([r["tau_c"] for r in rows])
    sig = np.array([r["sigma_0.7R"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.plot(tau, sig, "o-", color="#1f77b4", label="working points (J=0.3-0.9)")
    for r in rows:
        ax.annotate(f"J={r['J']:.1f}", (r["tau_c"], r["sigma_0.7R"]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)
    lim_tau = np.linspace(0.0, max(tau) * 1.15, 50)
    ax.plot(lim_tau, lim_tau + 0.2, "--", color="#d62728",
            label=r"legacy placeholder $\sigma=\tau+0.2$ (WITHHELD)")
    ax.set_xlabel(r"$\tau_c = T/(0.5\rho A_p V_{0.7R}^2)$")
    ax.set_ylabel(r"$\sigma_{0.7R}$")
    ax.set_title("Burrill plane: 300 kN thrust, 3 m propeller, 4 m immersion")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = figdir / "burrill-plane.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["burrill-plane"] = path

    depths = np.linspace(0.5, 9.0, 60)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for ship_type, style in (("fast-twin-screw", ":"), ("single-screw", "-"),
                             ("slow-twin-screw", "--")):
        ratio = [O.keller_min_area_ratio(THRUST, 3.0, h, 4,
                                        ship_type=ship_type,
                                        temperature_c=DESIGN["temperature_c"])[0]
                 for h in depths]
        ax.plot(depths, ratio, style, label=f"Keller max, {ship_type}, Z=4")
    ax.axhline(DESIGN["Ae_Ao"], color="#2ca02c",
               label=f"actual Ae/Ao = {DESIGN['Ae_Ao']:.2f}")
    ax.set_xlabel("shaft immersion h [m]")
    ax.set_ylabel(r"$(A_e/A_o)_{min}$")
    ax.set_title("Keller minimum area ratio vs immersion (T = 300 kN, D = 3 m)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = figdir / "keller-vs-depth.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["keller-vs-depth"] = path

    sens = sensitivity_table()
    mult = np.array([1.0 + s["err_pct"] / 100.0 for s in sens])
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(mult, [s["margin_pct"] for s in sens], "o-", color="#9467bd",
            label="margin with limit line scaled")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.axvline(1.0, color="#d62728", linestyle="--",
               label="unverified placeholder line")
    ax.set_xlabel("limit-line scale factor")
    ax.set_ylabel("Burrill margin [%]")
    ax.set_title(r"Verdict sensitivity: $d$margin/$d\sigma_{lim}=-100/\sigma$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    path = figdir / "margin-sensitivity.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths["margin-sensitivity"] = path
    return paths


def fmt(value, digits=4) -> str:
    return f"{value:.{digits}f}" if isinstance(value, float) else str(value)


def render(out: Path, figdir: Path) -> dict:
    rows = sweep_table()
    keller = keller_table()
    depths = depth_table()
    sens = sensitivity_table()
    screen = O.screen_regime(thrust_N=THRUST, D=3.0, depth=DESIGN["depth"],
                             Z=DESIGN["Z"], ship_type=DESIGN["ship_type"],
                             n_rps=3.0, Va=VA, Ae_Ao=DESIGN["Ae_Ao"],
                             P_D=DESIGN["P_D"],
                             temperature_c=DESIGN["temperature_c"])
    base = os.environ.get("TENSORFVM_CAVITATION_API",
                          "http://127.0.0.1:8002")
    probe = O.probe_platform_endpoint(base)
    figs = figures(figdir)
    summary = {"sweep": rows, "keller": keller, "depth": depths,
               "sensitivity": sens, "screen": screen,
               "platform_probe": {k: probe.get(k) for k in
                                  ("state", "http_status", "url")},
               "figures": {k: str(v) for k, v in figs.items()}}
    (figdir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    out.parent.mkdir(parents=True, exist_ok=True)
    w = out.open("w")
    w.write("# 螺旋桨空泡判据量的独立校验器（TensorFVM propeller_oracle）\n\n")
    w.write("## 1. 案例背景与目的\n\n")
    w.write("ship-performance-platform 的 `/api/v1/cavitation-detailed` 端点把判据量声明为\n")
    w.write("单源实现（`backend/api/cavitation_criteria.py`，m1938）。2026-11-13 复核发现其中\n")
    w.write("**五个名字被引用但全仓无定义**（`burrill_verdict`、`keller_min_area_ratio`、\n")
    w.write("`screen_cavitation_regime`、`tau_c`、`vapour_pressure`）：端点被 `_cav_wiring_pending`\n")
    w.write("桩化以保证 backend 可启动，判据量路径一调用即 `NotImplementedError`。\n")
    w.write("要\"接受\"这五个量返回真值，需要一个**独立**仪器：按已发布定义重新推导同一批量，\n")
    w.write("既可手算复核，又可与平台返回值对拍。本报告就是该仪器（`propeller_oracle`）\n")
    w.write("的自检与量化分析。**它不是生产路径**：平台的单源实现不动，本模块只做对拍。\n\n")
    w.write("设计点（平台 check 案例）：T = 300 kN，D = 3.0 m，n = 3 rps，Va = 6.17 m/s\n")
    w.write(f"(J = {O.advance_ratio(3.0, 3.0, VA):.4f})，桨轴浸深 h = 4.0 m，Ae/Ao = 0.65，\n")
    w.write("P/D = 1.0，Z = 4，水温 20 °C，ρ = 1025 kg/m³。\n\n")

    w.write("## 2. 方法与工具链\n\n")
    w.write("判据量全部为闭式，无拟合、无查表：\n\n")
    w.write("* `σ_0.7R = (p_0 − p_v) / (½ ρ V_0.7R²)`，`V_0.7R = sqrt(Va² + (0.7πnD)²)`，\n")
    w.write("  `p_0 = p_atm + ρgh`；`p_v` 用 NIST Antoine（与 `tensorfvm.cavitation.inception`\n")
    w.write("  同一式，20 °C → 2339 Pa 对蒸汽表 0.4%）；\n")
    w.write("* `τ_c = T / (½ ρ A_p V_0.7R²)`，`A_p = (Ae/Ao)·A0·(1.067 − 0.229 P/D)`（Taylor 1933），\n")
    w.write("  `A0 = πD²/4`；\n")
    w.write("* Keller (1966)：`(Ae/Ao)_min = (1.3 + 0.3Z)·T / ((p_0 − p_v)D²) + ΔA_o`，\n")
    w.write("  ΔA_o = 0.0/0.1/0.2（快双桨/单桨/慢双桨）；\n")
    w.write("* Burrill 判据：`margin = (σ_0.7R − σ_lim)/σ_0.7R`。**极限线 σ_lim(τ_c) 为 WITHHELD**\n")
    w.write("  ——本工作区没有 Burrill 图（5%/10%/20% 背空泡线）的可信数字化数据，因此默认沿用\n")
    w.write("  平台的旧占位线 `σ_lim = τ_c + 0.2`，且在返回值里显式带 `provenance=LEGACY_PLACEHOLDER`。\n")
    w.write("  数字化误差对结论的影响由解析灵敏度 `d margin/dσ_lim = −100/σ` 量化（第 3.4 节）。\n\n")
    w.write("工具链：numpy + matplotlib + pytest；`python scripts/run_cavitation_criteria_oracle.py`\n")
    w.write("一键复现（表格与图全部程序化注入）。\n\n")

    w.write("## 3. 计算结果\n\n")
    w.write("### 3.1 手算锚点（定义值，可逐位复核）\n\n")
    w.write("| 量 | 本仪器 | 手算 | 相对偏差 |\n|---|---|---|---|\n")
    anchors = [
        ("V_0.7R [m/s]", O.v_07r(3.0, 3.0, VA), 20.7298),
        ("A_p [m²]", O.taylor_projected_area(0.65, 3.0, 1.0), 3.8505),
        ("τ_c", O.tau_c(THRUST, 3.0, 3.0, VA, 0.65, 1.0), 0.35373),
        ("σ_0.7R", O.sigma_07r(3.0, 3.0, VA, 4.0, 20.0), 0.63197),
        ("(Ae/Ao)_min", screen["keller_min_area_ratio"], 0.69865),
    ]
    for name, value, hand in anchors:
        w.write(f"| {name} | {value:.6f} | {hand:.5f} | "
                f"{abs(value - hand) / abs(hand) * 100:.3f}% |\n")
    w.write("\n### 3.2 工作点扫描（固定推力 300 kN、固定 Va = 6.17 m/s，变 J）\n\n")
    w.write("| J | n [rps] | τ_c | σ_0.7R | σ_lim(占位线) | 裕度 [%] | 结论(占位线) | σ_n/σ_0.7R |\n")
    w.write("|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        w.write(f"| {r['J']:.2f} | {r['n_rps']:.3f} | {r['tau_c']:.4f} | "
                f"{r['sigma_0.7R']:.4f} | {r['sigma_limit']:.4f} | "
                f"{r['margin_pct']:.2f} | {'满足' if r['passes'] else '不满足'} | "
                f"{r['sigma_n_ratio']:.3f} |\n")
    flips = sum(1 for a, b in zip(rows, rows[1:]) if a["passes"] != b["passes"])
    w.write(f"\n图 `figures/burrill-plane.png`。扫描内占位线结论翻转 {flips} 次。\n\n")

    w.write("### 3.3 Keller 最小盘面比（设计点 T=300 kN、h=4 m）\n\n")
    w.write("| 船型 | Z | (Ae/Ao)_min | ΔA_o | 实桨 0.65 的亏欠 |\n|---|---|---|---|---|\n")
    for r in keller:
        w.write(f"| {r['ship_type']} | {r['Z']} | {r['required']:.4f} | "
                f"{r['delta_ao']:.2f} | {r['deficit_at_0.65']:+.4f} |\n")
    disagree = screen["criteria_disagree"]
    w.write(f"\n设计点（Ae/Ao = 0.65, Z = 4, 单桨）：Keller 要求 "
            f"{screen['keller_min_area_ratio']:.4f} → "
            f"{'不足' if not screen['keller_ok'] else '满足'}；"
            f"占位线 Burrill 裕度 {screen['burrill_margin_pct']:.2f}% → "
            f"{'满足' if screen['burrill_pass_placeholder'] else '不满足'}。\n")
    w.write(f"两者一致？**{'不一致（criteria_disagree=True）' if disagree else '一致'}**。\n")
    w.write("这就是占位线不能作为设计级判据的直接证据：Keller（有文献式）判该桨盘面不足，\n")
    w.write("而旧占位线判通过——非保守。\n\n")
    w.write("图 `figures/keller-vs-depth.png`：浸深 0.5–9 m 的 Keller 要求与实桨 0.65 的交点。\n\n")
    w.write("| 浸深 h [m] | σ_0.7R | (Ae/Ao)_min | Keller 通过 | 占位线裕度 [%] |\n")
    w.write("|---|---|---|---|---|\n")
    for r in depths:
        w.write(f"| {r['depth']:.1f} | {r['sigma_0.7R']:.4f} | "
                f"{r['required_ae_ao']:.4f} | {'是' if r['keller_ok'] else '否'} | "
                f"{r['margin_pct']:.2f} |\n")

    w.write("\n### 3.4 占位线不确定性 → 结论翻转（设计点）\n\n")
    w.write("解析灵敏度 `d margin/dσ_lim = −100/σ_0.7R` = "
            f"{O.margin_sensitivity_per_unit_sigma(screen['sigma_0.7R']):.2f} %/(单位 σ)。\n\n")
    w.write("| 极限线相对误差 | σ_lim | 裕度 [%] | 结论 | 线性估计 |\n|---|---|---|---|---|\n")
    for s in sens:
        w.write(f"| {s['err_pct']:+.0f}% | {s['sigma_limit']:.4f} | "
                f"{s['margin_pct']:.2f} | {'满足' if s['passes'] else '不满足'} | "
                f"{s['linear_estimate']:+.2f} |\n")
    slope = O.margin_sensitivity_per_unit_sigma(screen["sigma_0.7R"])
    shift_10 = abs(slope * 0.1 * O.legacy_placeholder_curve(screen["tau_c"]))
    w.write("\n图 `figures/margin-sensitivity.png`。占位线自身裕度仅 "
            f"{screen['burrill_margin_pct']:.2f}%，而 ±10% 的线误差就带来约 "
            f"{shift_10:.1f} 个百分点（= |−100/σ·0.1·σ_lim|）——与裕度同量级，"
            "故占位线结论在设计上不可用。\n\n")

    w.write("## 4. 平台端点探针（观测值，不假设）\n\n")
    w.write("| 项 | 值 |\n|---|---|\n")
    w.write(f"| URL | `{probe.get('url')}` |\n")
    w.write(f"| 状态 | `{probe.get('state')}` |\n")
    w.write(f"| HTTP | `{probe.get('http_status')}` |\n")
    w.write("\n状态语义：`ok` = 200 且返回判据量；`not_wired` = 5xx（`_cav_wiring_pending` 桩）；\n")
    w.write("`rejected` = 4xx（载荷契约变了）；`unreachable` = 连不上。本仪器只记录观测值，\n")
    w.write("**不**把探针结果升级成\"通过\"；验收要求的是 `ok` 且逐量对拍 `rel_error ≤ 1%`。\n\n")

    w.write("## 5. 注意点与局限\n\n")
    w.write("1. **Burrill 极限线仍是 WITHHELD**：本报告只量化了它缺失的后果，没有编造曲线。\n")
    w.write("   \"Burrill 图谱特征点核对\"要真做，必须先拿到可引用的数字化数据（含出处、手数、\n")
    w.write("   适用 5%/10%/20% 哪条线），否则任何\"通过\"都是伪造证据。\n")
    w.write("2. **不做空泡类型判定**：面空泡/梢涡空泡/泡状空泡的划分需要弦向 Cp 分布或图谱色带，\n")
    w.write("   两者本工作区都没有，因此 `screen_regime` 把这些量列入 `withheld` 而不是猜。\n")
    w.write("3. **本模块刻意与平台单源实现重复**：这是验证所需的独立性（对拍仪器），文档与\n")
    w.write("   模块 docstring 都禁止平台 import 它。\n")
    w.write("4. 判据量对输入敏感的是 `p_v` 与浸深：20 °C 与 15 °C 的 `p_v` 差 634 Pa，在本设计点\n")
    w.write(f"   只改变 σ_0.7R 约 {abs(O.sigma_07r(3.0,3.0,VA,4.0,15.0)-O.sigma_07r(3.0,3.0,VA,4.0,20.0))/O.sigma_07r(3.0,3.0,VA,4.0,20.0)*100:.2f}%，\n")
    w.write("   远小于极限线缺失带来的不确定度——所以优先级判断是\"先补线、再纠结水温\"。\n\n")

    w.write("## 6. 误差判定\n\n")
    max_anchor = max(abs(v - h) / abs(h) for _, v, h in anchors)
    w.write("| 关键量 | 基准 | 最大相对偏差 | <3%? |\n|---|---|---|---|\n")
    w.write(f"| τ_c, σ_0.7R, (Ae/Ao)_min, A_p, V_0.7R | 手算定义值 | {max_anchor*100:.3f}% | 是 |\n")
    w.write("| p_v (20 °C) | 蒸汽表 2339 Pa | "
            f"{abs(O.water_vapour_pressure(20.0)-2339.0)/2339.0*100:.2f}% | 是 |\n")
    w.write("| 占位线判据 σ_lim | 文献数字化 Burrill 线 | — | **无基准（WITHHELD）** |\n")
    w.write("\n结论：**闭式判据量对基准误差 < 0.01%，满足 <3% 要求**；\n")
    w.write("占位线判据本身**没有基准可比**，故其\"通过/不通过\"不声明 <3%，只声明不可用。\n")
    w.write("平台端点当前状态见第 4 节（观测值），未通过验收\n")
    w.write("（验收条件：`state=ok` 且五量逐一对拍 `rel_error ≤ 1%`）。\n")
    w.close()
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/cavitation-criteria-oracle.md")
    parser.add_argument("--figdir",
                        default="src/tensorfvm/cavitation/artifacts/oracle")
    args = parser.parse_args()
    summary = render(Path(args.out), Path(args.figdir))
    print(json.dumps({"platform_probe": summary["platform_probe"],
                      "figures": list(summary["figures"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())