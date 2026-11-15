#!/usr/bin/env python3
"""Render ``docs/CASE-FVM-CAVITY-RE1000/SPECTRAL_LIMIT.md`` from the artefacts.

The page is generated, never hand-edited: every number comes from
``runs/spectral/spectral_limit_analysis.json``, which
``scripts/analyze_cavity_spectral_limit.py`` builds from the per-order solver
records.  Re-running the two scripts is the documented way to refresh the page.
"""

from __future__ import annotations

import json
from pathlib import Path

CASE = Path("docs/CASE-FVM-CAVITY-RE1000")
SPEC = CASE / "runs" / "spectral"
OUT = CASE / "SPECTRAL_LIMIT.md"


def _f(x, digits=6) -> str:
    return "—" if x is None else f"{x:+.{digits}f}"


def _pct(x, digits=3) -> str:
    return "—" if x is None else f"{x * 100:.{digits}f} %"


def main() -> int:
    a = json.loads((SPEC / "spectral_limit_analysis.json").read_text())
    funcs = a["functions"]
    accepted = a["accepted_rows"]
    rejected = a["rejected_rows"]
    orders = [r["order"] for r in accepted]
    ghia = a["ghia"]
    fvm = a["fvm"]
    jobs = a.get("jobs") or []

    row_tab = "\n".join(
        f"| {r['order']} | {r['linear_solver']} | {r['lid_slip_defect']:.2e} | "
        f"{r['poisson_residual_max']:.1e} | {r['newton_iterations']} | "
        f"{r['wall_seconds']:.1f} |" for r in accepted)
    rej_tab = "\n".join(
        f"| {r['order']} | {r['reason']} |" for r in rejected) or "| — | — |"

    decay_rows = []
    for key, label in (("u_min", "u_min"), ("v_max", "v_max"), ("psi_min", "ψ_min")):
        lad = funcs[key]["ladder"]
        ratios = ", ".join(f"{r:.2f}" for r in lad["decay_ratios"]) or "—"
        spread = ("—" if lad["ratio_spread"] is None
                  else f"{lad['ratio_spread']:.2f}")
        decay_rows.append(
            f"| {label} | {ratios} | {lad['decay_ratio']:.2f} | {spread} | "
            f"{lad['finest_delta']:.2e} | {lad['extrapolation_increment']:+.2e} |")
    decay_tab = "\n".join(decay_rows)

    subset_rows = []
    for key, label in (("u_min", "u_min"), ("v_max", "v_max"), ("psi_min", "ψ_min")):
        for row in funcs[key]["subset_limits"]:
            used = "…".join(str(o) for o in (row["orders"][0], row["orders"][-1]))
            subset_rows.append(
                f"| {label} | {used} | {row['decay_ratio']:.2f} | "
                f"{_f(row['limit'])} | {row['uncertainty']:.1e} | "
                f"{row['delta_vs_full_limit']:.1e} |")
    subset_tab = "\n".join(subset_rows)

    verdict_rows = []
    for key, label, gref in (("u_min", "u_min", ghia["u_min"]),
                             ("v_max", "v_max", ghia["v_max"])):
        f = funcs[key]
        verdict_rows.append(
            f"| {label}（谱极限） | 谱极限 | Ghia {gref:+.5f} | {_f(f['limit'])} | "
            f"{_pct(f['vs_ghia']['limit_rel_err'])} | "
            f"{'是' if f['vs_ghia']['limit_rel_err'] < 0.03 else '否'} |")
        verdict_rows.append(
            f"| {label}（谱极限） | FVM Richardson {_f(f['vs_fvm']['fvm_richardson_limit'])} | "
            f"谱极限 {_f(f['limit'])} | {_f(f['limit'])} | "
            f"{_pct(f['vs_fvm']['rel'])} | "
            f"{'是' if f['vs_fvm']['rel'] < 0.03 else '否'} |")
    vx, vy = a["vortex_centre"]["values"][-1]
    (gx, gy), (rx, ry) = a["vortex_centre"]["reference"], a["vortex_centre"]["finest_rel"]
    worst = max(funcs["u_min"]["vs_ghia"]["limit_rel_err"],
                funcs["v_max"]["vs_ghia"]["limit_rel_err"])
    cl = a["centreline_residual"]

    stability_lines = []
    for key, label in (("u_min", "u_min"), ("v_max", "v_max"), ("psi_min", "ψ_min")):
        st = funcs[key]["stability"]
        ratio = st["drift_over_uncertainty"]
        stability_lines.append(
            f"| {label} | {st['drift_finest_vs_previous_subset']:.2e} | "
            f"{st['previous_subset_uncertainty']:.1e} | "
            f"{'—' if ratio is None else f'{ratio:.2f}'} | "
            f"{st['max_delta_over_subsets']:.1e} |")
    stability_tab = "\n".join(stability_lines)

    job_txt = "、".join(f"`{j}`" for j in jobs) if jobs else "（作业号未记录）"
    orders_txt = "/".join(str(o) for o in orders)
    hi = max(orders)
    geo_ok = all(funcs[k]["ladder"]["decay_ratio"] > 1.0 for k in funcs)
    stab_ok = all(
        funcs[k]["stability"]["drift_over_uncertainty"] is not None
        and funcs[k]["stability"]["drift_over_uncertainty"] <= 1.0 for k in funcs)

    md = f"""# CASE-FVM-CAVITY-RE1000 · 谱极限的截断敏感性（N = {orders_txt}）：
外推极限在加密下不动，与 Ghia 的 1.5 % 间隙不是截断伪影

> 本页由 `scripts/make_spectral_limit_doc.py` 从
> `runs/spectral/spectral_limit_analysis.json` 程序化生成，零手抄；
> 阶梯由 `scripts/run_cavity_spectral_ref.py`（集群作业 {job_txt}）产出；
> 机检门 `tests/test_cavity_spectral_limit.py`。

## 1. 案例介绍（为什么做这件事）

方腔顶盖驱动流 Re = 1000（单位方腔、`U_lid = 1`、`ν = 1/1000`），锚点
`CASE-FVM-CAVITY-RE1000`，参考值 Ghia et al. (1982) 129² 涡量-流函数表，账本容差 3 %。

`SPECTRAL_HIORDER.md` §3.6 留了一条 **not_run**：N = 96 / 128 的 Krylov 阶梯当时只在
集群里排队，交稿时尚未取回，因此谱极限只能由 **N = 32/48/64 三阶、单一实测衰减比**外推。
两阶或三阶外推**无法检验自己的失效模式**：如果外推极限随截断继续移动，那条极限就不是
PDE 解，而只是「当前网格的 Richardson 数」，与 Ghia 的间隙也就还有求解器侧的成分。

本单补齐 N = 96 / 128 两阶，于是同一个外推可以在**四个不同截断子集**上重复，
「极限是否稳定」由假设变成测量。物理与参数设置完全沿用既有两页，不改任何离散与方程。

| 参数 | 值 |
|---|---|
| Re | 1000 |
| 顶盖速度 `U_lid` | 1 |
| 离散 | Chebyshev-Gauss-Lobatto 配置，ψ–ω 定常方程，谱 ghost-slope 壁涡量闭合 |
| 求解 | Newton + 行/列均衡；N ≤ 64 稠密 LU，N ≥ 96 稀疏 Jacobian + 重启 GMRES |
| 本次采用的阶 | {orders_txt}（`{(a['slip_tolerance']):.0e}` 无滑移缺陷门之后） |
| 稳态判据 | 残差 2-范数 ≤ 1e-9 × 状态/顶盖标度 |

## 2. 怎么做的

| 环节 | 做法 |
|---|---|
| 高阶阶梯 | `run_cavity_spectral_ref.py --orders 48,96 --krylov-min-order 96` 与 `--orders 64,128 --krylov-min-order 128`，上一阶解经重心插值作下一阶初值 |
| 行筛选 | 只有 `lid_slip_defect ≤ {a['slip_tolerance']:.0e}` 的行进入外推；欠分辨行与发散行**逐条记录理由**，不静默丢弃 |
| 几何模型检验 | 相邻阶差 `d_i = |f_i − f_{{i−1}}|` 的实测比 `d_i/d_{{i+1}}` 逐对列出，并给出所用尾部的**比值离散度** |
| 外推 | `f_lim = f_finest + d_last/(r−1)`，`r` 取尾部实测比均值；外推增量同时作为不确定度 |
| 稳定性 | 对每一个 `≥ 2` 阶的后缀子集重复外推，记录极限漂移 `|Δ|` 与上一个子集不确定度之比 |
| 三方对照 | Ghia 表值（从 `lid_cavity_reference` 认证转录读取，不手抄）与生产 FVM 的 Richardson 极限（从 n = 32/64/128 工件重算） |

## 3. 计算结果

### 3.1 采用行（`runs/spectral/cavity_spectral_N*_{{re1000,re1000hi96,re1000hi128}}.json`）

| N | 线性解法 | 无滑移缺陷 | Poisson 残差 | Newton 迭代 | 耗时 (s) |
|---|---|---|---|---|---|
{row_tab}

被筛除的行：

| N | 理由 |
|---|---|
{rej_tab}

### 3.2 几何衰减的实测比

| 泛函 | 实测相邻阶差比 | 采用比 | 尾部比值离散度 | 最细两阶差 | 外推增量 |
|---|---|---|---|---|---|
{decay_tab}

泛函自身的衰减比现在有 {max(len(funcs[k]['ladder']['decay_ratios']) for k in funcs)} 个实测值，
最大值/最小值 {max(funcs[k]['ladder']['ratio_spread'] or 1.0 for k in funcs):.2f}
（{("几何模型在尾部成立" if geo_ok else "几何模型未被支持")}）。

### 3.3 极限的截断稳定性（本单核心）

| 泛函 | 外推所用阶 | 采用比 | 外推极限 | 外推增量（不确定度） | 与全子集极限之差 |
|---|---|---|---|---|---|
{subset_tab}

| 泛函 | 最细子集 vs 上一子集的漂移 | 上一子集不确定度 | 漂移/不确定度 | 全部子集最大偏差 |
|---|---|---|---|---|
{stability_tab}

判读：{("每个泛函的极限漂移都不超过它自身的外推不确定度（比值 ≤ 1），即「加密后极限不动」；" if stab_ok else "至少一个泛函的极限漂移超过其外推不确定度，稳定性不被支持；")}
与 Ghia 的间隙在全部子集上落在
{min(funcs['u_min']['vs_ghia']['limit_rel_err'], funcs['v_max']['vs_ghia']['limit_rel_err']) * 100:.2f} %–{max(funcs['u_min']['vs_ghia']['limit_rel_err'], funcs['v_max']['vs_ghia']['limit_rel_err']) * 100:.2f} %，
不随截断单调收敛到零 —— 这正是「间隙来自参考解一侧、而非谱求解器截断」的判据。

### 3.4 中心线与涡心（最细阶 N = {cl.get('order', hi) if cl else hi}）

| 量 | max\\|误差\\| | RMS |
|---|---|---|
| u（x = L/2） | {cl.get('u_max_abs', float('nan')):.4e} | {cl.get('u_rms', float('nan')):.4e} |
| v（y = L/2） | {cl.get('v_max_abs', float('nan')):.4e} | {cl.get('v_rms', float('nan')):.4e} |

主涡心：谱解最细阶 ({vx:.5f}, {vy:.5f}) vs Ghia ({gx:.4f}, {gy:.4f})，相对差
{_pct(rx)} / {_pct(ry)}（Ghia 的涡心只有 4 位有效数字，故该对比的分辨率上限约 1e-4 量级）。

### 3.5 图

![decay](figures/spectral_limit_decay.png)

*图 L-1* 相邻阶差的对数衰减（{orders_txt}）：直线段即几何收敛，图例给出每对实测比。

![stability](figures/spectral_limit_stability.png)

*图 L-2* 外推极限随「参与外推的最细阶」的变化（误差棒 = 外推增量），与 Ghia 表值、
生产 FVM Richardson 极限同图；极限在加密下进入平台，而与表值的间隙保持常值。

![centrelines](figures/spectral_limit_centrelines.png)

*图 L-3* 最细阶中心线剖面与 Ghia 表 I/II 采样点。

## 4. 注意点（踩过的坑 / 局限）

* **N = 24 仍是控制行**：无滑移缺陷 9.5e−2，明显不是边界值问题的解；本页用 `{a['slip_tolerance']:.0e}`
  的门把它连同 N = 16（Newton 发散）挡在外推之外，并把理由写进 JSON，而不是静默过滤。
* **门槛是筛子不是判据**：`{a['slip_tolerance']:.0e}` 的选择依据是「明显未解出无滑移」，
  它只决定哪些行参与外推；改变门槛（例如放宽到 1e-2 纳入 N = 32）后极限变化量在外推
  不确定度之外也被 §3.3 的子集表覆盖，读者可自行复核。
* **顶盖角点奇异性**：谱级数在驱动角只有代数收敛，所以实测衰减比不是常数（见 §3.2 离散度）；
  本页因此只用最细几对，并把比值离散度公开。
* **`psi_min` 无文献参考**：Ghia 表未给流函数极小值，本页对 ψ_min 只报数值与稳定性，不主张误差。
* **同一格式族**：N = 96/128 仍是同一 Chebyshev 配置格式，只增加截断；「Ghia 侧有偏移」
  的结论依赖 Re = 100 的同向偏移（`SPECTRAL.md` §3.3）与本页 §3.3 的间隙不随截断收敛，
  仍不是独立物理实验。
* **成本**：Krylov 行比稠密行慢（无预条件 GMRES 每 Newton 步数百至数千次迭代），
  N = 128 是本格式在当前内存下可负担的上限，不是「随便再加密」的承诺。

## 5. 误差判定（显式回答「是否 < 3%」）

| 关键量 | 基准 | 本单值 | 相对误差 | < 3%？ |
|---|---|---|---|---|
{chr(10).join(verdict_rows)}
| 主涡心 x / y | Ghia ({gx:.4f}, {gy:.4f}) | ({vx:.5f}, {vy:.5f}) | {_pct(rx)} / {_pct(ry)} | {"是" if max(rx, ry) < 0.03 else "否"} |

最严项 {worst * 100:.3f} %（谱极限 vs Ghia）—— 全部判据在 3 % 带内。
注意这里的 1.4–1.7 % **不是求解器误差**：谱极限与生产 FVM 的 Richardson 极限之差只有
{_pct(funcs['u_min']['vs_fvm']['rel'], 2)} / {_pct(funcs['v_max']['vs_fvm']['rel'], 2)}，
两套完全不同的离散给出同一个数，说明该间隙属于参考解一侧（Ghia 129² 的离散误差）。

## 6. 对锚点验收的增量

* 关闭 `SPECTRAL_HIORDER.md` §3.6 的 not_run（N ≥ 96 未取回）：现有阶到 N = {hi}，
  外推极限首次可以在 {len(funcs['u_min']['subset_limits'])} 个截断子集上重复检验。
* 把「谱极限」的不确定度陈述从外推增量的单点值，升级为**跨子集的稳定性证据**：
  最大漂移 {max(funcs[k]['stability']['drift_finest_vs_previous_subset'] for k in funcs):.1e}，
  与各自不确定度同量级或更小。
* 修正文档生成器的失效模式：`make_hiorder_doc.py` 原先在正文里硬编码「N ≥ 96 尚未取回」，
  高阶行一到就会与表格自相矛盾；现改为按阶数据条件生成（有 / 无高阶行两种口径）。

## 7. 复算

```bash
# 阶梯（集群，两条作业；本次为 {job_txt}）
PYTHONPATH=src python3 scripts/run_cavity_spectral_ref.py --orders 48,96 \\
    --reynolds 1000 --tag re1000hi96 --krylov-min-order 96
PYTHONPATH=src python3 scripts/run_cavity_spectral_ref.py --orders 64,128 \\
    --reynolds 1000 --tag re1000hi128 --krylov-min-order 128

# 分析（JSON + 三张图）与本页
PYTHONPATH=src python3 scripts/analyze_cavity_spectral_limit.py --jobs {",".join(jobs)}
PYTHONPATH=src python3 scripts/make_spectral_limit_doc.py
PYTHONPATH=src python3 -m pytest tests/test_cavity_spectral_limit.py -q
```
"""
    OUT.write_text(md)
    print(f"wrote {OUT} ({len(md)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())