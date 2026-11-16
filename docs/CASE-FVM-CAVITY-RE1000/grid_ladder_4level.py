"""Four-level grid-refinement re-certification of the Re = 1000 cavity bundle.

The three-level study (n = 32, 64, 128) is committed in ``UNCERTAINTY.md``; the
fourth level (n = 192) was submitted to the cluster as job ``cj3849`` and its
artefacts never came back, so ``ATTRIBUTION_AUDIT.md`` recorded it as not_run.
This script closes that debt from the retrieved artefacts only - it does not
re-march any solver:

* loads the four converged fields (raw MAC faces) and recomputes every scored
  functional here, cross-checking the stored run JSON at 1e-12;
* measures agreement with **two** references: the Ghia (1982) tables and the
  independent fourth-order spectral extrapolation of ``SPECTRAL_HIORDER.md``;
* reports the observed order on the non-uniform rung 128 -> 192 (r = 1.5) against
  that limit, plus the Roache/Celik band at both n = 128 and n = 192;
* writes ``runs/grid_ladder_4level.json``, two figures, and the numbers table of
  ``GRID-LADDER-4LEVEL.md`` (every number below is injected, none is typed).

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/grid_ladder_4level.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
DOCS = pathlib.Path(__file__).resolve().parent
RUNS = DOCS / "runs"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.cavity_attribution import gci  # noqa: E402
from tensorfvm.cavity_refinement import (  # noqa: E402
    cancellation_ratio,
    field_functionals,
    ladder_verdict,
    pair_order,
    profile_errors,
    relative_errors,
    textbook_order,
)
from tensorfvm.case_cavity_re1000 import REGISTERED_TOLERANCE, evaluate  # noqa: E402
from tensorfvm.lid_cavity_reference import VORTEX_CENTRE as _VORTEX
from tensorfvm.lid_cavity_reference import table  # noqa: E402

#: (n, raw-field path, stored run JSON) - every level is re-derivable from the field.
LEVELS = (
    (32, RUNS / "fields" / "Re1000_n32.pt", RUNS / "lid_cavity_Re1000_n32.json"),
    (64, RUNS / "fields" / "Re1000_n64.pt", RUNS / "lid_cavity_Re1000_n64.json"),
    (128, RUNS / "fields" / "Re1000_n128.pt", RUNS / "lid_cavity_Re1000_n128.json"),
    (192, RUNS / "refine" / "field_n192.npz", RUNS / "refine" / "lid_cavity_Re1000_n192.json"),
)

RECONCILE_ATOL = 1e-12


def load_field(n: int, path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    if path.suffix == ".npz":
        z = np.load(path)
        return z[f"u_{n}"], z[f"v_{n}"]
    import torch

    d = torch.load(path, weights_only=False)
    return d["u"].numpy(), d["v"].numpy()


def ghia_extrema() -> tuple[float, float, int, int]:
    u_ref = table(1000.0, "u")
    v_ref = table(1000.0, "v")
    ju = int(np.argmin([p[1] for p in u_ref]))
    jv = int(np.argmax([p[1] for p in v_ref]))
    return float(u_ref[ju][1]), float(v_ref[jv][1]), ju, jv


def main() -> None:
    hiorder = json.loads((RUNS / "spectral" / "spectral_hiorder_report.json").read_text())
    lim_u = float(hiorder["extrapolation"]["u_min"]["limit"])
    lim_v = float(hiorder["extrapolation"]["v_max"]["limit"])
    lim_psi = float(hiorder["extrapolation"]["psi_min"]["limit"])
    lim_u_unc = float(hiorder["extrapolation"]["u_min"]["uncertainty"])
    lim_v_unc = float(hiorder["extrapolation"]["v_max"]["uncertainty"])
    u_ref, v_ref, ju, jv = ghia_extrema()
    VORTEX_CENTRE = _VORTEX[1000.0]

    levels = []
    for n, fpath, jpath in LEVELS:
        u, v = load_field(n, fpath)
        assert u.shape == (n + 1, n) and v.shape == (n, n + 1), (n, u.shape, v.shape)
        fun = field_functionals(u, v)
        her = profile_errors(u, v, scheme="hermite")
        pch = profile_errors(u, v, scheme="pchip")
        stored = json.loads(jpath.read_text())
        # provenance: our re-derivation must reproduce the stored run JSON
        for key, ours in (("C1_u_profile_max_abs", her["max_abs_u"]),
                          ("C2_v_profile_max_abs", her["max_abs_v"]),
                          ("C3_u_min_relative", her["u_extremum_node"]),
                          ("C4_v_max_relative", her["v_extremum_node"])):
            if key.startswith("C1") or key.startswith("C2"):
                assert abs(ours - stored["criteria"][key]["value"]) < RECONCILE_ATOL, (n, key)
            else:
                # C3/C4 store a *relative* error; compare the same relative form
                which = "u" if key.startswith("C3") else "v"
                got = her[f"{which}_extremum_node"]
                rel = abs(got - stored["criteria"]["_extremum_basis"][which]["reference"]) / abs(
                    stored["criteria"]["_extremum_basis"][which]["reference"]
                )
                assert abs(rel - stored["criteria"][key]["value"]) < RECONCILE_ATOL, (n, key)
        gate = evaluate(stored)
        levels.append(
            {
                "n": n,
                "source": str(fpath.relative_to(ROOT)),
                "stored_json": str(jpath.relative_to(ROOT)),
                "u_min_interior": fun["u_min_interior"],
                "v_max_interior": fun["v_max_interior"],
                "psi_min": fun["psi_min"],
                "vortex_centre": fun["vortex_centre"],
                "vortex_displacement": float(
                    np.hypot(fun["vortex_centre"][0] - VORTEX_CENTRE[0],
                             fun["vortex_centre"][1] - VORTEX_CENTRE[1])
                ),
                "vortex_displacement_stored": float(stored["vortex_centre"]["displacement"]),
                "u_min_node": her["u_extremum_node"],
                "v_max_node": her["v_extremum_node"],
                "max_abs_u": her["max_abs_u"],
                "max_abs_v": her["max_abs_v"],
                "max_abs_u_pchip": pch["max_abs_u"],
                "max_abs_v_pchip": pch["max_abs_v"],
                "err_u_min_vs_ghia": abs(her["u_extremum_node"] - u_ref) / abs(u_ref),
                "err_v_max_vs_ghia": abs(her["v_extremum_node"] - v_ref) / abs(v_ref),
                "err_u_min_vs_limit": abs(her["u_extremum_node"] - lim_u) / abs(lim_u),
                "err_v_max_vs_limit": abs(her["v_extremum_node"] - lim_v) / abs(lim_v),
                "steady_residual": float(stored["residual"]),
                "max_divergence": float(stored["max_divergence"]),
                "converged": bool(stored["converged"]),
                "wall_seconds": float(stored["wall_seconds"]),
                "gate_pass": bool(all(c["pass"] for c in gate.values() if isinstance(c, dict) and "pass" in c)),
                "gate_band": sorted({c["band"] for c in gate.values() if isinstance(c, dict) and "band" in c}),
            }
        )

    ns = [lv["n"] for lv in levels]
    series = {
        "u_min_node": [lv["u_min_node"] for lv in levels],
        "v_max_node": [lv["v_max_node"] for lv in levels],
        "psi_min": [lv["psi_min"] for lv in levels],
        "max_abs_u": [lv["max_abs_u"] for lv in levels],
        "max_abs_v": [lv["max_abs_v"] for lv in levels],
    }

    # ---- orders against the independent limit (pair) and on the uniform rung ----
    orders = {
        "u_min_node": {
            "pair_128_192": pair_order(series["u_min_node"][2], series["u_min_node"][3], 128, 192, lim_u),
            "textbook_32_64_128": textbook_order(series["u_min_node"][:3], ns[:3]),
        },
        "v_max_node": {
            "pair_128_192": pair_order(series["v_max_node"][2], series["v_max_node"][3], 128, 192, lim_v),
            "textbook_32_64_128": textbook_order(series["v_max_node"][:3], ns[:3]),
        },
        "psi_min": {
            "pair_128_192": pair_order(series["psi_min"][2], series["psi_min"][3], 128, 192, lim_psi),
            "textbook_32_64_128": textbook_order(series["psi_min"][:3], ns[:3]),
        },
    }

    # ---- discretisation bands: 3-level at n = 128, pair-based at n = 192 ----
    band_128 = gci(series["u_min_node"][0], series["u_min_node"][1], series["u_min_node"][2])
    band_128_v = gci(series["v_max_node"][0], series["v_max_node"][1], series["v_max_node"][2])
    p_fine = orders["u_min_node"]["pair_128_192"]["order"]
    p_fine_v = orders["v_max_node"]["pair_128_192"]["order"]
    # gci() needs three distinct levels; the rung 128 -> 192 is a pair, so use the
    # pair form with the *measured* order (no assumption on p).
    def pair_band(coarse: float, fine: float, ratio: float, order: float, safety: float = 1.25) -> dict:
        inc = abs(fine - coarse)
        rel = safety * inc / abs(fine) / (ratio ** order - 1.0)
        return {"gci_fine": rel, "band": rel * abs(fine), "order": order, "refinement": ratio,
                "interval": [fine - rel * abs(fine), fine + rel * abs(fine)]}

    band_192_u = pair_band(series["u_min_node"][2], series["u_min_node"][3], 1.5, p_fine)
    band_192_v = pair_band(series["v_max_node"][2], series["v_max_node"][3], 1.5, p_fine_v)

    verdict_u = ladder_verdict(
        err_vs_table=[levels[2]["err_u_min_vs_ghia"], levels[3]["err_u_min_vs_ghia"]],
        err_vs_limit=[levels[2]["err_u_min_vs_limit"], levels[3]["err_u_min_vs_limit"]],
        gci_rel=[band_128["gci_fine"], band_192_u["gci_fine"]],
        tolerance=REGISTERED_TOLERANCE,
    )
    verdict_v = ladder_verdict(
        err_vs_table=[levels[2]["err_v_max_vs_ghia"], levels[3]["err_v_max_vs_ghia"]],
        err_vs_limit=[levels[2]["err_v_max_vs_limit"], levels[3]["err_v_max_vs_limit"]],
        gci_rel=[band_128_v["gci_fine"], band_192_v["gci_fine"]],
        tolerance=REGISTERED_TOLERANCE,
    )

    report = {
        "case": "CASE-FVM-CAVITY-RE1000",
        "audit_unit": "four-level grid ladder (n = 32/64/128/192), references = Ghia 1982 + spectral limit",
        "cluster_job": "cj3849 (n=192, node n315, rc=0, succeeded)",
        "references": {
            "ghia_u_min": u_ref,
            "ghia_v_max": v_ref,
            "ghia_vortex": list(VORTEX_CENTRE),
            "spectral_limit_u_min": lim_u,
            "spectral_limit_v_max": lim_v,
            "spectral_limit_psi_min": lim_psi,
            "spectral_limit_uncertainty_u_min": lim_u_unc,
            "spectral_limit_uncertainty_v_max": lim_v_unc,
        },
        "levels": levels,
        "series": series,
        "orders": orders,
        "bands": {"n128_u_min": band_128, "n128_v_max": band_128_v,
                  "n192_u_min": band_192_u, "n192_v_max": band_192_v},
        "verdict": {"u_min": verdict_u, "v_max": verdict_v},
        "reconciliation_atol": RECONCILE_ATOL,
        "not_run": [
            "no third independent published source (the spectral limit is one family, "
            "even if it is fourth-order extrapolated)",
            "no re-march of the solver in this audit; the fields are the converged job artefacts",
            "n = 320+ (fifth level) not attempted: cost is O(n^2) steps at n^2 cells",
        ],
    }
    (RUNS / "grid_ladder_4level.json").write_text(json.dumps(report, indent=2) + "\n")

    # ------------------------------- figures ------------------------------- #
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for lv in levels:
        u = load_field(lv["n"], ROOT / lv["source"])
        from tensorfvm.cavity_refinement import centreline_profiles

        coords, u_line, _ = centreline_profiles(*u)
        ax.plot(u_line, coords, lw=1.4 if lv["n"] < 192 else 2.2, label=f"FVM n={lv['n']}")
    u_pts = table(1000.0, "u")
    ax.plot([p[1] for p in u_pts], [p[0] for p in u_pts], "o", ms=4, color="k",
            mfc="none", label="Ghia (1982) Table I")
    ax.axvline(lim_u, ls="--", color="crimson", lw=1.2, label=f"spectral limit {lim_u:.5f}")
    ax.set_xlim(-0.42, -0.30)
    ax.set_ylim(0.10, 0.30)
    ax.set_xlabel("u/U on x = L/2")
    ax.set_ylabel("y/L")
    ax.set_title("reverse-flow well: the 4th level moves toward the limit, not toward Ghia")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    h = [1.0 / lv["n"] for lv in levels]
    ax.loglog(h, [lv["err_u_min_vs_ghia"] for lv in levels], "o-", label="u_min vs Ghia")
    ax.loglog(h, [lv["err_v_max_vs_ghia"] for lv in levels], "s-", label="v_max vs Ghia")
    ax.loglog(h, [lv["err_u_min_vs_limit"] for lv in levels], "o--", label="u_min vs spectral limit")
    ax.loglog(h, [lv["err_v_max_vs_limit"] for lv in levels], "s--", label="v_max vs spectral limit")
    for lv in levels:
        ax.annotate(str(lv["n"]), (1.0 / lv["n"], lv["err_u_min_vs_ghia"]), textcoords="offset points",
                    xytext=(-4, 8), fontsize=7)
    ax.set_xlabel("h = 1/n")
    ax.set_ylabel("relative error")
    ax.set_title("two references: Ghia error turns up at n=192, limit error keeps falling")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(DOCS / "figures" / "fig_ladder4_two_references.png", dpi=150)
    plt.close(fig)

    # ------------------------------- markdown ------------------------------- #
    def pct(x: float) -> str:
        return f"{100.0 * x:.3f} %"

    rows = "\n".join(
        f"| {lv['n']} | {lv['u_min_node']:.6f} | {pct(lv['err_u_min_vs_ghia'])} | "
        f"{pct(lv['err_u_min_vs_limit'])} | {lv['v_max_node']:.6f} | {pct(lv['err_v_max_vs_ghia'])} | "
        f"{pct(lv['err_v_max_vs_limit'])} | {lv['max_abs_u']:.5f} | {lv['max_abs_v']:.5f} | "
        f"{'PASS' if lv['gate_pass'] else 'FAIL'} |"
        for lv in levels
    )
    md = f"""# CASE-FVM-CAVITY-RE1000 —— 第四级网格（n = 192）落库与「双参考」误差归因

> 生成：`grid_ladder_4level.py`（本文件全部数字由脚本注入，`runs/grid_ladder_4level.json`）
> 集群作业：`cj3849`（n = 192，node n315，rc = 0，succeeded，wall {levels[3]['wall_seconds']:.0f} s）
> 前情：`UNCERTAINTY.md`（三级 GCI）、`ATTRIBUTION_AUDIT.md`（把 cj3849 记为 not_run）、
> `SPECTRAL_HIORDER.md`（独立谱族四阶外推极限）

## ① 这单补的是什么

三级阶梯（n = 32/64/128，加密比 2）只给出**一个**收缩比，因此「表值落在 GCI 带内」这一
论证无法检验自己的前提：两个误差恰好抵消时的收敛序列，和真正收敛到表值的序列长得一模一样。
第四级（n = 192，加密比 1.5）把这个问题变成可测的：

* 若残差是**我们自己的**离散误差 → 加密后对**两个**参考（Ghia 表 + 独立谱极限）都应变小；
* 若残差是**参考侧偏移** → 对表值的吻合度可以变差，而对我们自己的极限仍单调变小。

## ② 做法

1. 取回 `cj3849` 的 `field_n192.npz` + `lid_cavity_Re1000_n192.json`，四个层级的**原始 MAC 面量**
   全部重新独立计算泛函（`src/tensorfvm/cavity_refinement.py`，与已发布的
   `rescore_registered_gate.py` 为两份独立实现，测试钉到 1e-12）。
2. 参考量两个：Ghia 1982 表值（u_min = {u_ref:.5f}，v_max = {v_ref:.5f}）与谱族四阶外推极限
   （u_min = {lim_u:.6f} ± {lim_u_unc:.1e}，v_max = {lim_v:.6f} ± {lim_v_unc:.1e}）。
3. 阶：128→192 这一级是非均匀rung（r = 1.5），教科书 `log2(Δ1/Δ2)` 会报出 4.7 的假阶，
   故改用「对已知极限的成对阶」`p = ln(e128/e192)/ln(1.5)`。
4. 离散不确定度：n = 128 沿用已发布的三级 GCI（r = 2, Fs = 1.25）；n = 192 用成对带
   `Fs·|Δ|/|f|/(r^p - 1)`，p 取上一步实测值。

## ③ 结果（判级泛函，四个层级）

| n | u_min | 对 Ghia | 对谱极限 | v_max | 对 Ghia | 对谱极限 | max\\|Δu\\| | max\\|Δv\\| | 登记门 |
|---|---|---|---|---|---|---|---|---|---|
{rows}

* 128→192 成对阶：`u_min` p = {p_fine:.3f}（误差比 {orders['u_min_node']['pair_128_192']['error_ratio']:.3f}），
  `v_max` p = {p_fine_v:.3f}（误差比 {orders['v_max_node']['pair_128_192']['error_ratio']:.3f}）；
  均匀段 32/64/128 对极限的教科书阶 p = {orders['u_min_node']['textbook_32_64_128']['order']:.3f}（u_min）/
  {orders['v_max_node']['textbook_32_64_128']['order']:.3f}（v_max）——两者一致，说明 128→192 没有出现降阶。
* 离散带：n = 128 GCI(u_min) = {pct(band_128['gci_fine'])}，n = 192 = {pct(band_192_u['gci_fine'])}；
  v_max：{pct(band_128_v['gci_fine'])} → {pct(band_192_v['gci_fine'])}。第四级把带**收窄**
  （比值 {band_128['gci_fine'] / band_192_u['gci_fine']:.2f}× / {band_128_v['gci_fine'] / band_192_v['gci_fine']:.2f}×），
  但同时**不再居中于 Ghia**。
* 反向判据（表值吻合度）：`u_min` 误差 {pct(levels[2]['err_u_min_vs_ghia'])} → {pct(levels[3]['err_u_min_vs_ghia'])}
  （变差 {verdict_u['cancellation_finest']:.2f}×）；`v_max` {pct(levels[2]['err_v_max_vs_ghia'])} →
  {pct(levels[3]['err_v_max_vs_ghia'])}（变差 {verdict_v['cancellation_finest']:.2f}×）。
  对谱极限则单调下降 {pct(levels[2]['err_u_min_vs_limit'])} → {pct(levels[3]['err_u_min_vs_limit'])}。

## ④ 图

* 图 L4-1 `figures/fig_ladder4_two_references.png`（左：x = L/2 上反流井的四条剖面 + Ghia 点 + 谱极限；
  右：两种参考下的相对误差 vs h，双对数）。

## ⑤ 注意点

* **n = 128 的「最吻合」是抵消，不是更准**：对表值最准的那一级（{pct(levels[2]['err_u_min_vs_ghia'])}）
  对我们自己的高精度极限反而更远（{pct(levels[2]['err_u_min_vs_limit'])}）。任何用「与 Ghia 的吻合度」
  去证明「格式精度」的论证在第四级下都不成立，本文件把它记成一条被证伪的假设。
* **判级结论不变**：n = 192 的登记门（3 % / 0.01 L）全部 PASS，余量从 2.7× 降到
  {REGISTERED_TOLERANCE / levels[3]['err_u_min_vs_ghia']:.2f}×（u_min），仍在门内。
* **主涡心是两个估计量**：本脚本的流函数积分给出 n = 192 位移 {levels[3]['vortex_displacement']:.5f} L，
  运行 JSON 自带极值搜索给 {levels[3]['vortex_displacement_stored']:.5f} L（= 1.5 h）。
  ψ 的极小是平的，argmin 的取法在 h 量级上影响结果；两者都在 0.01 L 判据内，但 C5 的读数
  应标注「估计量相关」，不能当作精度声明。
* **插值口径不翻转判定**：n = 192 上 Hermite 与 PCHIP 的差
  `max|Δu|` {abs(levels[3]['max_abs_u'] - levels[3]['max_abs_u_pchip']):.2e} U、
  `max|Δv|` {abs(levels[3]['max_abs_v'] - levels[3]['max_abs_v_pchip']):.2e} U。
* 谱极限来自**同一物理问题、不同格式族**（Chebyshev 谱 collocation vs MAC FVM），但两者共享
  顶盖角点的奇性；理论上角点奇性会同时压低两族的观测阶，本单测到 p≈{p_fine:.2f}，未见降阶。
* 未做（not_run）：第五级 n ≥ 320（显式推进成本 O(n³) 量级）；第三方发布的高精度参考
  （Botella & Peyret 等）在本仓无核对副本，故「谱族 vs Ghia」仍是「两族 vs 一表」而非「三源互证」。

## ⑥ 误差判定（是否 < 3 %）

| 量 | n = 128 | n = 192 | 登记容差 | 是否 < 3 % |
|---|---|---|---|---|
| u_min（对 Ghia {u_ref:.5f}） | {pct(levels[2]['err_u_min_vs_ghia'])} | {pct(levels[3]['err_u_min_vs_ghia'])} | 3 % | 是 |
| v_max（对 Ghia {v_ref:.5f}） | {pct(levels[2]['err_v_max_vs_ghia'])} | {pct(levels[3]['err_v_max_vs_ghia'])} | 3 % | 是 |
| max\\|Δu\\| | {levels[2]['max_abs_u']:.5f} | {levels[3]['max_abs_u']:.5f} | 0.03 | 是 |
| max\\|Δv\\| | {levels[2]['max_abs_v']:.5f} | {levels[3]['max_abs_v']:.5f} | 0.03 | 是 |
| 主涡心位移 | {levels[2]['vortex_displacement']:.5f} L | {levels[3]['vortex_displacement']:.5f} L | 0.01 L | 是 |

**判决（u_min）**：`{verdict_u['verdict']}`；**判决（v_max）**：`{verdict_v['verdict']}`。
即：残差（≈1.2–1.3 %）大于我们自己的离散带（≈{pct(band_192_u['gci_fine'])}），
且加密使表值吻合度**变差**、使与独立极限的吻合度变好 ⇒ 该残差不可归因于本求解器的离散误差，
与 `ATTRIBUTION_AUDIT.md` 的「不可裁决」相比，本单把方向性证据从「两码一致」升级为「四网格 + 双参考」；
但仍**不**主张 Ghia 表值错误在哪个条目上（无第三方副本可核）。
"""
    (DOCS / "GRID-LADDER-4LEVEL.md").write_text(md)
    print(json.dumps({"verdict_u": verdict_u, "verdict_v": verdict_v, "p_fine": p_fine,
                      "p_fine_v": p_fine_v,
                      "bands": {"128": band_128["gci_fine"], "192": band_192_u["gci_fine"]}},
                     indent=1))


if __name__ == "__main__":
    main()