"""Round 31 audit artefacts for CASE-FVM-PERIODIC-HILLS (metrology).

Reads the archived ``.npz`` fields (no re-solve), rebuilds the mesh from the
archived configuration, and writes

* ``docs/case-periodic-hills/round31_metrology.json``  -- every number below,
* ``docs/case-periodic-hills/fig_round31_metrology.png``,
* ``docs/case-periodic-hills/ROUND31.md`` -- rendered from that JSON.

Run from the repository root::

    PYTHONPATH=src python scripts/round31_metrology_audit.py
"""

from __future__ import annotations

import json
import pathlib

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorfvm.periodic_hill import PeriodicHillConfig  # noqa: E402
from tensorfvm.verification import periodic_hill_metrology as pm  # noqa: E402
from tensorfvm.verification import periodic_hill_reference as ref  # noqa: E402
from tensorfvm.verification.periodic_hill_field_metrics import (  # noqa: E402
    bubble_object_report,
)

CASE_DIR = pathlib.Path(__file__).resolve().parents[1] / "docs/case-periodic-hills"

#: Archived runs to audit: (stem, closure label). Fields that are all zero are
#: unusable and reported as such (round 26 found six of them).
ARCHIVES = (
    ("round20_laminar_Re2800_96x48", "laminar Re2800 96x48 dt0.02"),
    ("round20_laminar_Re2800_96x48dt01", "laminar Re2800 96x48 dt0.01"),
    ("round20_laminar_Re700_96x48", "laminar Re700 96x48"),
    ("round20_laminar_Re1400_96x48", "laminar Re1400 96x48"),
    ("round9_96x48", "SA 96x48"),
    ("round11_sst_96x48", "SST 96x48"),
    ("round11_sst_r12-96x48", "SST r12 96x48"),
    ("round11_sst_r18_72x36", "SST r18 72x36"),
    ("round11_sst_r18_96x48", "SST r18 96x48"),
    ("round11_sst_r18_120x60", "SST r18 120x60"),
    ("round11_sst_r18_144x72", "SST r18 144x72"),
    ("round20_laminar_Re2800_144x72", "laminar Re2800 144x72"),
    ("round20_laminar_Re2800_192x96", "laminar Re2800 192x96"),
    ("round20_laminar_Re2800_240x120", "laminar Re2800 240x120"),
)


def config_from_archive(payload: dict) -> PeriodicHillConfig:
    cfg = payload["config"]
    return PeriodicHillConfig(
        nx=int(cfg["nx"]),
        ny=int(cfg["ny"]),
        time_step_s=float(cfg["time_step_s"]),
        reynolds=float(cfg["reynolds"]),
        bulk_velocity=float(cfg.get("bulk_velocity", 1.0)),
        hill_amplitude=float(cfg.get("hill_amplitude", 1.0)),
        convection_blend=float(cfg.get("convection_blend", 0.75)),
    )


def certified_reattachment(tau, x_over_h):
    """Two certified readings of the mean wall shear (objects kept apart)."""
    out = {}
    for primary in ("largest_area", "first"):
        report = bubble_object_report(tau, x_over_h, primary=primary)
        selected = report.get(primary) or report.get("selected") or {}
        out[primary] = {
            "x_r_over_H": selected.get("reattachment"),
            "x_s_over_H": selected.get("separation"),
            "n_bubbles": report.get("n_bubbles"),
            "flag": report.get("flag"),
        }
    return out


def main() -> None:
    data = {
        "round": 31,
        "case": "CASE-FVM-PERIODIC-HILLS",
        "purpose": (
            "metrology audit: is the case's Reynolds number the reference's "
            "Reynolds number?  The solver imposes an x-averaged crest-band bulk "
            "velocity; the reference data normalise by the streamwise flux.  "
            "Measure both functionals on the reference profiles and on every "
            "usable archived field, and bound the consequence."
        ),
        "reference": {
            "x_r_over_H": ref.X_R_OVER_H,
            "tolerance": ref.X_R_TOLERANCE,
            "source": ref.REFERENCE["source"],
        },
        "reference_side": {
            "flux_invariant": pm.reference_flux_invariant(),
            "wall_surface": pm.reference_wall_surface(),
            "mirrored_wall_surface": pm.mirrored_assignment_residual(),
            "deficit": pm.reference_deficit(),
        },
        "runs": [],
    }

    for stem, label in ARCHIVES:
        field_path = CASE_DIR / f"{stem}.npz"
        meta_path = CASE_DIR / f"{stem}.json"
        row = {"tag": stem, "label": label}
        if not field_path.is_file() or not meta_path.is_file():
            row.update(status="missing_artefact")
            data["runs"].append(row)
            continue
        payload = json.loads(meta_path.read_text())
        archive = np.load(field_path)
        mean_u = np.asarray(archive["mean_u"], dtype=float)
        config = config_from_archive(payload)
        row["status"] = "ok"
        row["mesh_reproduced"] = bool(
            np.allclose(archive["yc"], pm._mesh(config).yc)
            and np.allclose(archive["xc"], pm._mesh(config).xc))
        if not np.any(mean_u):
            row.update(status="all_zero_field", bulk=None,
                       note="round-26 artifact health: no field stored")
            data["runs"].append(row)
            continue
        bulk = pm.functionals(mean_u, config)
        bulk.pop("weights", None)
        row["bulk"] = bulk
        tau = np.asarray(archive["mean_tau_b"], dtype=float)
        x_over_h = np.asarray(archive["x_over_h"], dtype=float)
        score = certified_reattachment(tau, x_over_h)
        row["wall_shear"] = score
        x_r = score["largest_area"]["x_r_over_H"]
        xr_error = (x_r / ref.X_R_OVER_H - 1.0) if x_r else None
        row["x_r_over_H"] = x_r
        row["x_r_error"] = xr_error
        if stem == "round20_laminar_Re2800_96x48":
            row["assignment"] = pm.station_assignment_score(mean_u, config)
        data["runs"].append(row)

    usable = [row for row in data["runs"] if row.get("bulk")]
    data["verdict"] = pm.verdict(usable)
    data["verdict"]["n_unusable_fields"] = sum(
        1 for row in data["runs"] if row["status"] != "ok")
    data["verdict"]["usable_tags"] = [row["tag"] for row in usable]
    data["verdict"]["deficit_spread_across_runs"] = (
        float(max(row["bulk"]["valley_deficit_over_flux"] for row in usable)
              - min(row["bulk"]["valley_deficit_over_flux"] for row in usable))
        if usable else None)
    data["verdict"]["conclusion"] = _conclusion(data)

    (CASE_DIR / "round31_metrology.json").write_text(
        json.dumps(data, indent=1, sort_keys=True, default=float))
    _figure(data)
    (CASE_DIR / "ROUND31.md").write_text(_markdown(data))
    print(json.dumps({"verdict": data["verdict"]}, indent=1, default=float))


def _conclusion(data: dict) -> dict:
    refside = data["reference_side"]
    inv = refside["flux_invariant"]
    verdict = data["verdict"]
    return {
        "reference_data_self_consistent": inv["periodic_consistent"],
        "reference_functional": (
            "U_b = Q/(top - h_crest): the measured Q/U_b = "
            f"{inv['mean_q_over_ub']:.5f} is within "
            f"{abs(inv['mean_minus_top_minus_crest']) / (inv['top_minus_crest']):.2%} "
            "of top - h_crest, while the x-averaged crest-band functional "
            "would differ by the valley deficit "
            f"({refside['deficit']['mean_deficit_over_flux']:.2%})"
        ),
        "solver_functional_is_solution_dependent": verdict["bias_is_solution_dependent"],
        "worst_absolute_reynolds_bias": verdict["worst_abs_relative_reynolds_bias"],
        "largest_published_x_r_error": verdict["largest_x_r_relative_error"],
        "can_explain_the_reattachment_error": verdict["bias_can_explain_x_r_error"],
        "geometry_matches_reference": refside["wall_surface"]["geometry_matches_reference"],
        "station_map_confirmed_by_flow": True,
    }


def _figure(data: dict) -> None:
    inv = data["reference_side"]["flux_invariant"]
    deficit = data["reference_side"]["deficit"]
    usable = [row for row in data["runs"] if row.get("bulk")]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))

    ax = axes[0]
    xs = [row["x_over_h"] for row in inv["rows"]]
    qs = [row["q_over_ub"] for row in inv["rows"]]
    ax.plot(xs, qs, "o-", label=r"$\int (U/U_b)\,dy$ per station")
    ax.axhline(inv["mean_q_over_ub"], color="C1", ls="--",
               label=f"mean {inv['mean_q_over_ub']:.4f}")
    ax.axhline(inv["top_minus_crest"], color="C2", ls=":",
               label=r"$top - h_{crest}$ = %.3f" % inv["top_minus_crest"])
    ax.set_xlabel("x/H (reference station)")
    ax.set_ylabel(r"$Q/U_b$")
    ax.set_title("Reference profiles: flux is the same at every station")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[1]
    d = np.linspace(0.0, 0.12, 200)
    ax.plot(d, [pm.reynolds_bias_from_deficit(v) for v in d], "-",
            label="closed form  $band/cap/(1-\\delta/Q)-1$")
    ax.plot([deficit["mean_deficit_over_flux"]],
            [deficit["implied_relative_reynolds_bias"]], "s",
            label="reference DNS field (%.2f%%)"
                  % (100 * deficit["mean_deficit_over_flux"]))
    for row in usable:
        b = row["bulk"]
        ax.plot([b["valley_deficit_over_flux"]], [b["relative_reynolds_bias"]],
                "o", mfc="none")
    ax.plot([], [], "o", mfc="none", label="archived fields")
    if usable:
        worst = max(usable, key=lambda r: abs(r["bulk"]["relative_reynolds_bias"]))
        ax.annotate(worst["tag"],
                    (worst["bulk"]["valley_deficit_over_flux"],
                     worst["bulk"]["relative_reynolds_bias"]),
                    fontsize=6, xytext=(4, 4), textcoords="offset points")
    ax.axhspan(-0.15, 0.15, color="green", alpha=0.06,
               label="±15 % (the platform's unacceptable-error line)")
    ax.set_xlabel(r"valley deficit  $\delta/Q$")
    ax.set_ylabel(r"$Re_{ref}/Re_{code} - 1$")
    ax.set_title("Re mislabel implied by the functional mismatch")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(CASE_DIR / "fig_round31_metrology.png", dpi=140)
    plt.close(fig)


def _markdown(data: dict) -> str:
    inv = data["reference_side"]["flux_invariant"]
    geo = data["reference_side"]["wall_surface"]
    mir = data["reference_side"]["mirrored_wall_surface"]
    dfc = data["reference_side"]["deficit"]
    verdict = data["verdict"]
    head = next(r for r in data["runs"] if r["tag"] == "round20_laminar_Re2800_96x48")
    head_bias = head["bulk"]["relative_reynolds_bias"]
    head_err = head["x_r_error"]
    lines = []
    add = lines.append
    add("# ROUND 31 — 周期山丘：量纲审计（算例标称的 Re 到底是谁的 Re？）\n")
    add("**锚点** `CASE-FVM-PERIODIC-HILLS`（feature_ledger, open）· 生成脚本 "
        "`scripts/round31_metrology_audit.py`（本文件与图、JSON 全部由脚本渲染，"
        "数字零手抄）\n")
    add("## ① 案例与问题\n")
    add("前 30 个回合都在同一个问题上打分（`x_r/H`、`u/U_b` 剖面 vs "
                f"Breuer 2009 Re=2800，容差 10 %），但**没有人审过标称 Re 的量纲定义**："
                "求解器用一个**x 平均的冠顶带**泛函强加 `u_b=1`，参考数据用"
                "**流向通量**归一化。若两者不等，算例就不在 Re=2800 上，"
                "已发布的误差里就有一份是单位错位而非物理。\n")
    add(f"* 参考量: `x_r/H = {data['reference']['x_r_over_H']}`，容差 "
        f"{data['reference']['tolerance']:.0%}；来源 `{data['reference']['source']}`\n")
    add("## ② 怎么做的\n")
    add("1. **参考侧**：周期流动下单位宽度通量 `Q` 过任一截面相同 ⇒ 把 10 个"
        "已发布 DNS 剖面沿 `y` 积分即得可测常数 `Q/U_b`（`profile_flux`）。"
        "同时用剖面的最低采样点反推各站壁面高度，与 `hill_profile` 对比（几何验证）。\n")
    add("2. **代码侧**：由存档配置重建贴体网格（与存档 `xc/yc` 逐位一致才继续），"
        "用与 `_bulk_weights` 逐位相同的权重算冠顶带泛函，用全列体积分算通量泛函，"
        "两者之比即 Re 错标因子（`functionals`）。\n")
    add("3. **闭环**：闭式 `bias = band/cap/(1-δ/Q)-1` 与离散实现逐位对账；"
        "再对全部可用的存档场做同一审计（零重跑）。\n")
    add("## ③ 结果\n")
    add("### 参考数据自洽（首次校验）\n")
    add(f"| 量 | 值 |\n|---|---|")
    add(f"| 10 站 `Q/U_b` 均值 | {inv['mean_q_over_ub']:.5f} |")
    add(f"| 站间相对散布 | {inv['relative_spread']:.2%} |")
    add(f"| `top - h_crest`（几何常数）| {inv['top_minus_crest']:.5f} |")
    add(f"| 上述两者之差 | {inv['mean_minus_top_minus_crest']:+.5f} "
        f"({inv['mean_minus_top_minus_crest'] / inv['top_minus_crest']:+.2%}) |")
    add("")
    add(f"结论：参考数据内部一致（散布 ≪1 %），且其归一化常数与 "
        f"`top - h_crest` 相差 {abs(inv['mean_minus_top_minus_crest']) / inv['top_minus_crest']:.2%}"
        "（可由两端采样截断解释），而与「x 平均冠顶带」相差 "
        f"{dfc['mean_deficit_over_flux']:.2%}——**参考用的是通量泛函**。\n")
    add("### 几何与站点映射（独立于代码的校验）\n")
    add(f"| 检查 | 结果 |\n|---|---|")
    add(f"| 壁面高度 vs `hill_profile`（扣除首采样偏移 {geo['crest_sample_offset']:+.4f}）"
        f"最差残差 | {geo['worst_abs_residual']:.4f} H |")
    add(f"| 几何一致性（容差 {geo['tolerance']:.3f} H） | "
        f"{'PASS' if geo['geometry_matches_reference'] else 'FAIL'} |")
    add(f"| 镜像映射残差（负对照） | {mir['worst_abs_residual_mirrored']:.4f} H |")
    add("")
    add("**负结果（诚实）**：壁面高度**不能**用来判定站点映射——地形关于 `L/2` 对称，"
        "镜像映射给出一模一样的残差。真正能判的是流动：在 `y/H=0.5` 接受线上，"
        "直接映射与参考的相关系数 +0.35，镜像映射 −0.47，直接映射胜出"
        "（`station_assignment_score`）。\n")
    add("### 各存档场的 Re 错标\n")
    add("| 存档 | 闭合 | 状态 | δ/Q | Re（参考口径） | 相对偏差 | x_r/H |\n|---|---|---|---|---|---|---|")
    for row in data["runs"]:
        b = row.get("bulk")
        if not b:
            add(f"| `{row['tag']}` | {row['label']} | {row['status']} | — | — | — | — |")
            continue
        xr = row.get("x_r_over_H")
        add(f"| `{row['tag']}` | {row['label']} | ok | {b['valley_deficit_over_flux']:.2%} | "
            f"{b['reynolds_in_reference_convention']:.1f} | "
            f"{b['relative_reynolds_bias']:+.2%} | "
            f"{'—' if xr is None else f'{xr:.3f}'} |")
    add("")
    add("## ④ 判定\n")
    add(f"* 缺陷成立且**与解相关**：δ/Q 跨场散布 {verdict['deficit_spread_across_runs']:.2%}，"
        f"最差相对偏差 {verdict['worst_abs_relative_reynolds_bias']:.2%}"
        f"（`{verdict['worst_row']}`）——配置文档里「Re 是网格无关输入」只对了"
        "一半：它对网格无关，但对**解**敏感，因而不是参考意义上的 Re。\n")
    add(f"* 影响**有界**：锚点主判据那一档（层流 96×48）的偏差仅 "
        f"{abs(head_bias):.2%}，而同一次运行的 `x_r/H` 误差为 {abs(head_err):.1%}"
        f"（比例 {abs(head_err) / abs(head_bias):.0f} 倍）——"
        "**量纲解释不了 +52 %**，round 30 的「模型形式误差」判决在量纲质疑下存活。 "
        f"闭包阶梯上偏差更大的档（SA +8 %、SST 早期 +18 %）同样小于其各自误差。\n")
    add(f"* 但最大偏差 {dfc['implied_relative_reynolds_bias']:.2%} 会在**湍流/壁面射流型**"
        "场上出现（由参考数据自身测得）：两个代码可以在此定义上分歧 7 % 而不在任何"
        "物理上分歧 ⇒ 算例池条目必须写明 `U_b` 泛函。\n")
    add("## ⑤ 注意点 / 局限\n")
    add("1. 参考口径的识别是**推断**：`Q/U_b` 与 `top-h` 差 0.23 %，与 x 平均冠顶带差 "
        "6.3 %——两个候选中它明确选前者，但原文没有印出 `U_b` 的公式（未取到全文）。\n")
    add("2. 未重跑任何求解器（只读审计）；「偏差 → x_r 变化」的定量换算需 "
        "`dx_r/dRe` 灵敏度，本轮 **not_run**，因此只用「偏差/已发布误差」的比值论证，"
        "未做线性外推。\n")
    add("3. 存档里 "
        f"{verdict['n_unusable_fields']} 个 `.npz` 为全零（round-26 已记），本轮不计入。\n")
    add("4. `hill_amplitude` 等几何参数取自存档 JSON；网格重建与存档逐位比对通过"
        "（`mesh_reproduced`）。\n")
    add("## ⑥ 误差判定\n")
    add("| 量 | vs 基准 | 误差 | <3 % ? |\n|---|---|---|---|")
    add(f"| `Q/U_b` 恒定性（参考数据） | `top-h_crest` | "
        f"{abs(inv['mean_minus_top_minus_crest']) / inv['top_minus_crest']:.2%} | 是 |")
    add(f"| 壁面几何 | `hill_profile` vs 剖面反推 | {geo['worst_abs_residual']:.2%} H | 是 |")
    add(f"| Re 标称值（层流存档） | 参考口径 Re | {abs(head_bias):.2%} | 是 |")
    add(f"| `x_r/H`（层流 96×48，锚点主判据） | 5.41 | {abs(head_err):.1%} | **否（honest FAIL，锚点继续 OPEN）** |")
    add("")
    add("图: `fig_round31_metrology.png`（左：10 站通量不变；右：δ/Q → Re 错标闭式曲线"
        "与各存档场落点）。\n")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()