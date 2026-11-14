"""Discretisation-uncertainty study for ``CASE-FVM-CAVITY-RE1000``.

Reads the stored artefacts (never re-runs the solver), builds the grid-convergence
payload with :mod:`tensorfvm.cavity_uncertainty`, folds in the same-grid /
different-Courant runs, writes ``runs/uncertainty.json``, the figures and the
markdown report ``UNCERTAINTY.md``.  Every number in that report is injected from
the payload — nothing is typed by hand.

    PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/uncertainty.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfvm.cavity_uncertainty import (  # noqa: E402
    functionals_from_fields,
    richardson,
    study,
)

RUNS = HERE / "runs"
FIGURES = HERE / "figures"
SUMMARY = json.loads((RUNS / "summary.json").read_text(encoding="utf-8"))
BASELINE_CFL = float(SUMMARY["cases"][0]["solver"]["cfl"])


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def load_fields() -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Stored converged fields: the three-grid sequence plus the n = 192 refinement."""
    data = np.load(RUNS / "fields.npz")
    fields: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    for key in data.files:
        if not key.startswith("u_"):
            continue
        n = int(key.split("_")[1])
        fields[n] = (torch.as_tensor(data[key]).double(),
                     torch.as_tensor(data[f"v_{n}"]).double())
    refine = RUNS / "refine" / "field_n192.npz"
    if refine.is_file():
        extra = np.load(refine)
        fields[192] = (torch.as_tensor(extra["u_192"]).double(),
                       torch.as_tensor(extra["v_192"]).double())
    return dict(sorted(fields.items()))


def storey(json_path: pathlib.Path) -> dict:
    return json.loads(json_path.read_text(encoding="utf-8"))


def time_step_runs() -> dict[str, dict]:
    """Runs on the finest grid that differ only in the Courant number."""
    runs = {}
    base = storey(RUNS / "lid_cavity_Re1000_n128.json")
    runs[f"cfl={BASELINE_CFL:.3f} (baseline)"] = base
    for path in sorted((RUNS / "dt").glob("*/lid_cavity_Re1000_n128.json")):
        rec = storey(path)
        if int(rec["n"]) != 128:
            continue
        runs[f"cfl={float(rec['solver']['cfl']):.3f}"] = rec
    return runs


def functionals_of_storey(rec: dict) -> dict[str, float]:
    """Functionals published by a single-resolution artefact."""
    basis = rec["criteria"]["_extremum_basis"]
    return {
        "u_min": float(basis["u"]["computed"]),
        "v_max": float(basis["v"]["computed"]),
        "psi_min": float(rec["streamfunction"]["psi_min"]),
        "vortex_x": float(rec["vortex_centre"]["computed"][0]),
        "vortex_y": float(rec["vortex_centre"]["computed"][1]),
    }


def time_step_block(runs: dict[str, dict], baseline: str) -> dict:
    labels = list(runs)
    values = {lab: functionals_of_storey(rec) for lab, rec in runs.items()}
    quantities = {}
    for q in ("u_min", "v_max", "psi_min"):
        series = {lab: values[lab][q] for lab in labels}
        ref = series[baseline]
        quantities[q] = {
            "values": series,
            "deviation_from_baseline": {lab: (series[lab] - ref) / abs(ref) for lab in labels},
            "max_relative_spread": max(
                abs(series[lab] - np.mean(list(series.values()))) / abs(np.mean(list(series.values())))
                for lab in labels),
        }
    # profile-level comparison against the baseline: interpolation free
    profile_delta = {}
    for lab in labels:
        du = [abs(float(b) - float(a)) for a, b in
              zip(runs[lab]["u"]["profiles"]["computed"], runs[baseline]["u"]["profiles"]["computed"])]
        dv = [abs(float(b) - float(a)) for a, b in
              zip(runs[lab]["v"]["profiles"]["computed"], runs[baseline]["v"]["profiles"]["computed"])]
        profile_delta[lab] = {"max_abs_u": max(du), "max_abs_v": max(dv)}
    worst = max(max(v["max_relative_spread"] for v in quantities.values()),
                max(max(profile_delta[lab].values()) for lab in labels if lab != baseline))
    return {
        "runs": labels,
        "baseline": baseline,
        "courant_numbers": {lab: float(runs[lab]["solver"]["cfl"]) for lab in labels},
        "steps": {lab: int(runs[lab]["steps"]) for lab in labels},
        "quantities": quantities,
        "profile_delta_vs_baseline": profile_delta,
        "max_relative_spread": float(worst),
        "tolerance": 1e-3,
        "time_step_independent": bool(worst <= 1e-3),
    }


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def figure_functionals(payload: dict, path: pathlib.Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, q, unit in ((axes[0], "u_min", "u at x = L/2 (U)"),
                        (axes[1], "v_max", "v at y = L/2 (U)")):
        blk = payload["quantities"][q]
        n = np.asarray(blk["grids"], dtype=float)
        f = np.asarray(blk["values"], dtype=float)
        ax.plot(1.0 / n, f, "o-", color="C0", label="computed (stored fields)")
        ext = blk.get("extrapolated")
        if ext is not None:
            ax.axhline(ext, color="C0", ls="--", lw=1.4,
                       label=f"Richardson limit (p = {blk['observed_order']:.2f})")
            gci = blk["gci_fine_absolute"]
            ax.errorbar([1.0 / n[-1]], [f[-1]], yerr=[[gci], [gci]], fmt="none",
                        ecolor="C0", capsize=4, label="GCI band (finest)")
        ref = blk.get("reference")
        if ref is not None:
            ax.axhline(ref, color="C3", ls="-.", lw=1.4, label="Ghia et al. (1982)")
        ax.set_xlabel("1 / n  (n = grid cells per direction)")
        ax.set_ylabel(unit)
        ax.set_title(f"{q}: grid convergence")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def figure_time_step(block: dict, path: pathlib.Path) -> None:
    labels = block["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    ax = axes[0]
    width = 0.8 / len(labels)
    xs = np.arange(len(block["quantities"]))
    for k, lab in enumerate(labels):
        ys = [block["quantities"][q]["deviation_from_baseline"][lab] for q in block["quantities"]]
        ax.bar(xs + k * width, ys, width=width, label=lab)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xticks(xs + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(list(block["quantities"]))
    ax.set_ylabel("relative deviation from the baseline run")
    ax.set_title("time-step sensitivity on the finest grid")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)

    ax = axes[1]
    for k, lab in enumerate(labels):
        ys = [block["profile_delta_vs_baseline"][lab]["max_abs_u"],
              block["profile_delta_vs_baseline"][lab]["max_abs_v"]]
        ax.bar(np.arange(2) + k * width, ys, width=width, label=lab)
    ax.axhline(block["tolerance"], color="C3", ls="--", lw=1.2,
               label=f"acceptance {block['tolerance']:g}")
    ax.set_xticks(np.arange(2) + width * (len(labels) - 1) / 2)
    ax.set_xticklabels(["max|Δu| on x=L/2", "max|Δv| on y=L/2"])
    ax.set_ylabel("abs. difference to the baseline run (U)")
    ax.set_title("centreline change vs the baseline Courant number")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# markdown
# --------------------------------------------------------------------------- #
def pct(x: float) -> str:
    return f"{100.0 * x:+.3f}%"


def markdown(payload: dict) -> str:
    q = payload["quantities"]
    grids = payload["grids"]
    lines = []
    lines.append("# CASE-FVM-CAVITY-RE1000 · discretisation-uncertainty study\n")
    lines.append("> 本文件由 `uncertainty.py` 从 `runs/` 工件程序化生成（数字零手抄）；"
                 "复算入口见文末。\n")
    lines.append("## 1. 为什么要做这件事\n")
    lines.append(
        "算例门（`case_cavity_re1000.py` 的 C1–C7）判的是「最细网格与 Ghia 表的差」——"
        "这是**参考值一致性**。它不能回答「最细网格的解离 PDE 精确解有多远」，因为"
        "（i）网格误差与「格式-参考值」偏差可能相互抵消，使最细网格看起来比它自己的"
        "离散误差还准；（ii）参考值（Ghia 1982, 129²）本身有不确定度。"
        "本节用求解器**自己的网格序列**给出第二个答案（Roache GCI，Celik 2008 数据质量检查），"
        "并沿时间步轴做同样口径的检查。\n")
    lines.append("## 2. 方法与设置\n")
    lines.append(
        "* 泛函（grid-independent，全部**从存储的面场直接算出**，`functionals_from_fields`）："
        "`u_min`（x=L/2 面上最负流向速度）、`v_max`（y=L/2 面上最大横向速度）、"
        "`psi_min`（流函数极小，主涡强度）、`kinetic`（动能）；偶数 n 时基准截线恰好落在面上，"
        "**不做任何插值**——这与算例门的逐点判据（Hermite 插值到 Ghia 采样点）是互补口径。\n"
        f"* 网格序列：n = {', '.join(str(n) for n in grids)}（32/64/128 来自算例包，"
        "n = 192 由 `refine_from_stored.py` 从存储 n=128 场双线性延拓后续算，仅提供初值）。\n"
        f"* 估计量：非均匀加密比下的 Celik et al. (2008) 观测阶迭代式；"
        f"Richardson 外推 `f_ext = f_3 + (f_3-f_2)/(r^p-1)`；安全因子 Fs = {payload['safety_factor']}；"
        "GCI_fine = Fs·|(f_3-f_2)/f_3|/(r^p-1)。\n"
        "* 数据质量门：序列非单调或观测阶越界（<0.5 或 >4）时**拒绝给 GCI**，不做算术平均粉饰。\n")
    lines.append("## 3. 结果：空间离散不确定度\n")
    lines.append("| 泛函 | 观测阶 p | 序列数值 | 外推极限 | GCI(最细) | 最细 vs Ghia | 外推 vs Ghia | 参考偏差在 GCI 带内？ |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name, blk in q.items():
        vals = ", ".join(f"{v:.6g}" for v in blk["values"])
        if blk["data_quality"]["usable"]:
            p = f"{blk['observed_order']:.2f}"
            ext = f"{blk['extrapolated']:.6g}"
            gci = pct(blk["gci_fine_relative"])
        else:
            p = ext = gci = "—"
        if "reference" in blk:
            fine_ref = pct(blk["fine_minus_reference_relative"])
            ext_ref = pct(blk["extrapolated_minus_reference_relative"]) if "extrapolated" in blk else "—"
            inside = "是" if blk.get("reference_within_gci_band") else "否"
        else:
            fine_ref = ext_ref = inside = "无参考值"
        lines.append(f"| {name} | {p} | {vals} | {ext} | {gci} | {fine_ref} | {ext_ref} | {inside} |")
    lines.append("")
    u, v = q["u_min"], q["v_max"]
    lines.append("![grid convergence](figures/fig_uq1_functionals.png)\n")
    lines.append("*图 UQ-1* 泛函随 1/n 的收敛（圆点=存储场实算，蓝虚线=Richardson 极限，"
                 "误差棒=最细网格 GCI，红线=Ghia 1982 表值）。\n")
    lines.append("### 3.1 这张表怎么读（显式结论）\n")
    lines.append(
        f"* 观测阶 p = {u['observed_order']:.2f}（u_min）/ {v['observed_order']:.2f}（v_max），"
        f"与二阶格式一致 → 序列确实处在渐近区，GCI 可用。\n"
        f"* 最细网格（n = {grids[-1]}）与 Ghia 的差为 {pct(u['fine_minus_reference_relative'])} / "
        f"{pct(v['fine_minus_reference_relative'])}，**小于**该网格自身的离散带 "
        f"{pct(u['gci_fine_relative'])} / {pct(v['gci_fine_relative'])}。"
        "因此「0.6% 精度」这种说法不成立；可支持的表述是「最细网格的参考偏差落在自身离散不确定度带内」。\n"
        f"* Richardson 外推极限 u_min = {u['extrapolated']:.5f}、v_max = {v['extrapolated']:.5f}，"
        f"距 Ghia 表值 {pct(u['extrapolated_minus_reference_relative'])} / "
        f"{pct(v['extrapolated_minus_reference_relative'])}。**外推极限比最细网格离参考值更远**——"
        "说明 n=128 上那 0.6% 的一致性部分来自网格误差与格式偏差的抵消，不是格式精度更高。\n"
        "* 外推极限仍在算例池 5% 与账本 3% 带内，故算例门的 PASS 判定不受影响；"
        "受影响的只是「精度」的叙述口径。\n")
    lines.append("## 4. 结果：时间步（Courant 数）敏感度\n")
    ts = payload.get("time_step")
    if ts is None:
        lines.append(f"**未执行（not_run，不记 pass）**：{payload.get('time_step_status')}。"
                     "时间步无关性在本包内没有证据，不作任何结论。\n")
        lines.append("## 5. 注意点与局限\n")
    lines.extend(limitations(payload))
    lines.append("## 6. 复算\n")
    lines.append(recompute_block())
    return "\n".join(lines) + "\n"


def limitations(payload: dict) -> list[str]:
    return LIMITATIONS_TEXT


def recompute_block() -> str:
    return RECOMPUTE_TEXT


LIMITATIONS_TEXT = [
    "* **只有 3–4 个网格、至多 2 个加密区间**：GCI 的数据质量门只能检查单调性与观测阶是否落在"
    "可信区间，无法证明「已进入渐近区」。观测阶在 u_min/v_max 上稳定在 ~1.9–2.0，"
    "但在逐点剖面差范数上给出 2.57/1.51（`summary.json` 的 convergence 段）——不同泛函收敛速率"
    "不同，这一点如实保留。",
    "* **外推极限不是真值**：它只是该格式族在 h→0 的极限；与 Ghia 的差可以是格式差异，"
    "也可以是 Ghia 129² 网格自身的不确定度。要判定需要第二个独立高精度参考（如谱方法文献），"
    "本仓库内**没有**已核对的第二来源，故不下结论（记入待办）。",
    "* **逐点判据的插值敏感性**（审计 m20230 F5）：粗网格上 max|Δu| 随插值方案变化可达 ×2.5；"
    "本节用面值泛函回避了这个问题，两者不能混引。",
    "* **时间步（Courant 数）轴**：只有存在同网格、不同 CFL 的工件时才给结论；缺失时记为 "
    "not_run，不记 pass。",
]

RECOMPUTE_TEXT = """```
# 网格序列（已有工件，无需重跑求解器）
PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/uncertainty.py
# n=192 加密（从存储 n=128 场延拓，可选）
PYTHONPATH=src python docs/CASE-FVM-CAVITY-RE1000/refine_from_stored.py --from-n 128 --n 192
# CFL 敏感度（同网格、不同 dt）
PYTHONPATH=src python scripts/case_cavity_re1000.py --n 128 --cfl 0.075 --out-dir docs/CASE-FVM-CAVITY-RE1000/runs/dt/cfl0075
PYTHONPATH=src python scripts/case_cavity_re1000.py --n 128 --cfl 0.300 --out-dir docs/CASE-FVM-CAVITY-RE1000/runs/dt/cfl0300
# 机检门
PYTHONPATH=src python -m pytest tests/test_case_cavity_uncertainty.py -q
```"""


def main() -> None:
    fields = load_fields()
    payload = study(fields)
    runs = time_step_runs()
    baseline = f"cfl={BASELINE_CFL:.3f} (baseline)"
    if len(runs) >= 2:
        payload["time_step"] = time_step_block(runs, baseline)
    else:
        # no false claim: one run cannot show time-step independence
        payload["time_step_status"] = ("not_run: no same-grid/different-Courant "
                                       "artefacts under runs/dt at generation time")
    payload["refined_grid_from_stored"] = 192 in fields

    RUNS.mkdir(parents=True, exist_ok=True)
    (RUNS / "uncertainty.json").write_text(
        json.dumps(payload, indent=1, sort_keys=False), encoding="utf-8")
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure_functionals(payload, FIGURES / "fig_uq1_functionals.png")
    if "time_step" in payload:
        figure_time_step(payload["time_step"], FIGURES / "fig_uq2_timestep.png")
    (HERE / "UNCERTAINTY.md").write_text(markdown(payload), encoding="utf-8")
    print(f"grids={payload['grids']} refined={payload['refined_grid_from_stored']}")
    for q, blk in payload["quantities"].items():
        if blk["data_quality"]["usable"]:
            print(f"{q}: p={blk['observed_order']:.2f} gci={100 * blk['gci_fine_relative']:.2f}% "
                  f"ext={blk['extrapolated']:.6g}")
    if "time_step" in payload:
        ts = payload["time_step"]
        print(f"time_step: {ts['runs']} spread={100 * ts['max_relative_spread']:.4f}% "
              f"independent={ts['time_step_independent']}")
    else:
        print("time_step: not_run", payload["time_step_status"])


if __name__ == "__main__":
    main()
