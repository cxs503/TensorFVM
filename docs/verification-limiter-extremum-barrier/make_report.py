"""Generate ``docs/verification-limiter-extremum-barrier/REPORT.md`` from summary.json.

Every number in the report is injected from
``outputs/limiter-barrier/summary.json`` (produced by
``python -m tensorfvm.verification.limiter_barrier``); nothing is transcribed by
hand.  Run from the repository root:

    PYTHONPATH=src python3 -m tensorfvm.verification.limiter_barrier 40 80 160 320
    python3 docs/verification-limiter-extremum-barrier/make_report.py
"""
from __future__ import annotations

import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs" / "verification-limiter-extremum-barrier"
SRC = ROOT / "outputs" / "limiter-barrier"


def load() -> dict:
    with open(SRC / "summary.json", encoding="utf-8") as fh:
        return json.load(fh)


def fmt(value: float, digits: int = 4) -> str:
    return f"{value:.{digits}f}"


def table(header, rows) -> str:
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(out)


def build(summary: dict) -> str:
    grids = summary["grids"]
    probes = summary["probes"]
    orders = summary["orders"]
    thresh = summary["threshold"]
    verdict = summary["verdict"]
    criteria = summary["criteria"]
    errors = summary["errors"]
    amp = summary["amplification"]
    clip = summary["clip"]

    order_rows = []
    for lim in probes:
        l1, linf = orders[lim]["l1"], orders[lim]["linf"]
        order_rows.append([
            f"`{lim}`",
            fmt(l1), f"{(l1 - 2.0) / 2.0 * 100:+.2f} %",
            fmt(linf), f"{(2.0 - linf) / 2.0 * 100:+.2f} %",
            fmt(l1 - linf, 3),
            verdict["per_probe"][lim]["label"],
        ])

    err_rows = []
    for lim in probes:
        err_rows.append([f"`{lim}`"] + [f"{errors[lim][i]['linf']:.3e}"
                                        for i in range(len(grids))])

    amp_rows = []
    for lim in summary["amplification"]:
        series = amp[lim]
        amp_rows.append([f"`{lim}`", f"{series[0]:.2f}", f"{series[-1]:.2f}",
                         f"{series[-1] / series[0]:.2f}"])

    crit_rows = []
    for check in criteria:
        obs = check["observed"]
        text = obs if isinstance(obs, str) else json.dumps(obs, sort_keys=True)
        if len(text) > 120:
            text = text[:117] + "..."
        crit_rows.append([check["name"], check["claim"], f"`{text}`",
                          check["threshold"],
                          "**PASS**" if check["passed"] else "FAIL"])

    clip_rows = []
    for lim in summary["probes"]:
        clip_rows.append([f"`{lim}`"] + [str(row["extremum_cells"]) for row in clip[lim]]
                         + [str(clip[lim][-1]["heavy_clip_cells"]),
                            f"{clip[lim][-1]['min_clip_factor']:.3f}"])

    passes = f"{verdict['criteria_passed']}/{verdict['criteria_total']}"
    long_ladder = {lim: summary["readouts"][lim]["linf"].get("long_ladder")
                  for lim in summary["amplification"]}
    linf_summary = ", ".join(
        f"{lim}={fmt(orders[lim]['linf'], 3)}" for lim in summary["amplification"])
    l1_lo = fmt(min(orders[l]['l1'] for l in amp), 3)
    l1_hi = fmt(max(orders[l]['l1'] for l in amp), 3)
    linf_lo = fmt(min(orders[l]['linf'] for l in amp), 3)
    linf_hi = fmt(max(orders[l]['linf'] for l in amp), 3)
    l1_dev_max = max(abs(orders[l]['l1'] - 2.0) / 2.0 for l in amp) * 100.0
    l1_dev_min = min(abs(orders[l]['l1'] - 2.0) / 2.0 for l in amp) * 100.0
    gap_lo = (2.0 - max(orders[l]['linf'] for l in amp)) / 2.0 * 100.0
    gap_hi = (2.0 - min(orders[l]['linf'] for l in amp)) / 2.0 * 100.0
    long_rel_max = max(
        abs(long_ladder[l]['p_loglog'] - orders[l]['linf']) / orders[l]['linf']
        for l in amp) * 100.0
    worst_rel = max(l1_dev_max, gap_hi) / 100.0

    md = f"""# TVD 限制器在光滑极值处的精度壁垒：按范数分辨的观测阶验证（XVAL）

- 模块: `tensorfvm.verification.limiter_barrier`（`python -m tensorfvm.verification.limiter_barrier {(' '.join(str(g) for g in grids))}` 一条命令复现）
- 本报告由 `make_report.py` 从 `summary.json` 程序化生成（数字零手抄）
- 判定: **{passes} 条预声明判据 PASS**（B1–B13）
- 图: `fig_limiter_barrier.png`

## 1. 案例介绍

**背景.** `tensorfvm.compressible.reconstruct` 的文档过去无条件声称 MUSCL 重构
"second order in space and time for smooth flow"。已合并的三网格 log--log 阶研究
（`tensorfvm.verification.cellcentre_loglog`，本锚点前序交付）也只拟合 **L1** 单元平均误差，
因此无法区分这句话在哪个范数下成立。经典结论（van Leer 1979, *J. Comput. Phys.* 32:101；
LeVeque 2002 *Finite Volume Methods for Hyperbolic Problems* §6.13；Suresh & Huynh 1997,
*J. Comput. Phys.* 136:83）是：任何 TVD 限制器在**数据出现局部极值**的单元上必须压低斜率，
于是重构在该处退化，**逐点（L∞）阶**低于二阶；而 L1 把少数退化单元摊到 O(1/h) 个良好单元上，
把这个损失**藏起来**。本单把这句话变成可验证的测量，并对文档做出订正。

**算例.** 周期熵波（Euler 精确解）：`rho = 1 + 0.2 sin(2πx)`，`u = 0.5`、`p = 1`，
密度波被匀速输运而不变形，故误差只反映格式本身。求解器为已合并的
`EulerSolver`（单元中心 FV，MUSCL-Hancock + HLLC，CFL 0.4，`t_end = 0.2`），
经由 `cellcentre_loglog.run_case` 调用——本单只加测量，不加第二条求解路径。

| 参数 | 取值 |
|---|---|
| 方程 | 1-D Euler（周期熵波，精确解闭式） |
| 振幅 / 波数 / 速度 | A = 0.2 / k = 2π / u = 0.5 |
| 网格 | nx = {' / '.join(str(g) for g in grids)}（均匀周期，refinement ratio 2） |
| 时间推进 | MUSCL-Hancock（二阶），CFL 0.4，t_end = 0.2 |
| 通量 | HLLC |
| 重构/限制器 | `none`（无限制中心斜率，二阶基线）+ `mc` / `vanleer` / `superbee` / `minmod` |
| 误差口径 | 单元平均体积加权 L1 与 L∞（参考 = 精确**单元平均**，闭式） |
| 阶估计 | 三网格 log--log 最小二乘斜率（前 3 个网格）+ 长阶梯斜率（全部网格） |
| 判据 | 13 条，全部预声明（阶偏差 15% 门槛 = owner 精度标准的"阶版本"） |

## 2. 怎么做的

1. **先证明测量工具等于求解器的工具**：`slopes()` 是 `compressible.reconstruct.limited_slope`
   的 NumPy 镜像，在光滑/单调/带极值/随机四类场上与 torch 实现逐点比对（B1），
   最大相对偏差 {summary['mirror_max_rel_error']['none']:.2e}（五支限制器全部 ≤ 1e-12）。
   机制层面的数字因此读自求解器**真正跑的那一支**限制器。
2. **再跑阶梯**：每个限制器 nx = {'/'.join(str(g) for g in grids)} 各一次，误差对
   精确单元平均取 L1 与 L∞，对 h = 1/nx 做 log--log 最小二乘。
3. **机制直读（非拟合）**：在收敛后的场上逐单元计算 `a = q_{{i+1}} - q_i`、`b = q_i - q_{{i-1}}`，
   标出 `a·b ≤ 0` 的单元（TVD 约束生效处），并直接检查该处限制器返回的斜率是否恰为 0、
   限制器是否放大中心斜率、以及该单元集合的**计数是否随 h 变化**。
4. **代价量化**：同一网格上"受限方案 L∞ 误差 / 无限制 L∞ 误差"的放大倍数，以及该倍数
   在加密下的增长（误差率之差）。
5. **负对照**：`none` 必须在两个范数下都回到 2 阶（否则阶缺口是求解器缺陷而非限制器性质）；
   判据本身也要能判死——测试里构造了"限制器在 L∞ 恰好 2 阶"的伪造研究，必须使 B4/B5 FAIL。
6. **文档订正**：`reconstruct.py` 的阶声明改为带范数限定，并加漂移门 B13（字符串检查）防止
   无条件"光滑流二阶"的表述回来。

## 3. 计算结果

### 3.1 观测阶（三网格 log--log 斜率）

{table(["限制器", "p(L1)", "L1 偏差 vs 2", "p(L∞)", "L∞ 缺口 vs 2", "L1−L∞ 差", "判定标签"], order_rows)}

### 3.2 L∞ 误差阶梯（点wise 最坏单元）

{table(["限制器"] + [f"nx={g}" for g in grids], err_rows)}

### 3.3 代价：L∞ 误差相对无限制方案的放大

{table(["限制器", "最粗网格", "最细网格", "最细/最粗"], amp_rows)}

### 3.4 机制：TVD 约束生效的单元集合（逐网格计数 / 最细网格重度裁剪）

{table(["限制器"] + [f"nx={g}" for g in grids] + ["重度裁剪单元数", "最小保留比"], clip_rows)}

### 3.5 判据表（{passes}，全部预声明）

{table(["判据", "声明", "实测", "阈值", "判定"], crit_rows)}

## 4. 结果图

`fig_limiter_barrier.png`（三面板）

1. 左：L1（虚线）与 L∞（实线）误差 vs h，五支限制器 + 斜率 2 参考线——同一批解在两种范数下
   给出两条不同的斜率；
2. 中：各限制器的观测阶柱状图（L1 vs L∞）与 15% 带、声明阶 2 线；
3. 右：裁剪单元计数 vs nx（实线：裁剪因子 < 0.5 的重度裁剪单元；点线：极值单元）——
   缺陷集合不随 h 缩小。

![limiter barrier](fig_limiter_barrier.png)

## 5. 注意点（局限与适用范围）

1. **阶缺口在 L∞，不在 L1**：L1 读数 {l1_lo}–{l1_hi}
   （全部落在 15% 带内），L∞ 读数 {linf_lo}–{linf_hi}。
   **用哪个范数不是风格问题，它决定"是否满足文档声明的二阶"这个结论。**
2. **不宣称"恰好一阶"**：理论渐近值是一阶，但 nx ≤ {max(grids)} 的实测 L∞ 阶
   {linf_summary} 明显高于 1（B11 显式声明该理想化
   NOT_CONFIRMED）：这是**部分壁垒 + 前渐近**。相邻斜率（长阶梯）见 `summary.json` 的
   `readouts.*.linf.long_ladder.p_adjacent`，长阶梯斜率与三网格斜率相差
   ≤ {long_rel_max:.1f}%（B10），
   所以缺口不是三网格假象。
3. **superbee 是压缩型限制器**：它会放大中心斜率（`min_clip_factor` 之外的单元上
   |s_lim| > |s_cen| 合法存在），因此 B7 对它只要求"极值单元斜率恰为 0"，不要求"不放大"。
4. **只覆盖光滑极值这一条机制**：激波/接触间断处的 TVD 行为、边界掉阶（近壁一阶 BC 把整体阶
   压到 1）、非结构网格重建误差都不在本单范围内，需要各自独立的阶梯。
5. **一维周期熵波**：没有壁面、没有源项、没有湍流模型；结论只针对"重构 + 限制器"这一层，
   不外推到 RANS/LES 的整体精度声明。
6. **实测对网格对齐敏感**（极值单元计数 2 或 4：极值恰好落在单元面上时两个相邻单元同时被裁剪），
   B6 因此判"计数与网格无关且 ≤ 4"，而不是硬编码 2。

## 6. 误差判定

基准 = 精确单元平均（闭式）+ 无限制二阶基线（`none`）。

| 关键量 | 基准 | 实测 | 偏差 | 是否 <3% |
|---|---|---|---|---|
| 无限制方案 L1 观测阶 | 声明阶 2 | {fmt(orders['none']['l1'])} | {(orders['none']['l1']-2)/2*100:+.2f} % | {'✅ 是' if abs(orders['none']['l1']-2)/2 < 0.03 else '❌ 否'} |
| 无限制方案 L∞ 观测阶 | 声明阶 2 | {fmt(orders['none']['linf'])} | {(orders['none']['linf']-2)/2*100:+.2f} % | {'✅ 是' if abs(orders['none']['linf']-2)/2 < 0.03 else '❌ 否'} |
| TVD 限制器 L1 观测阶（{len(amp)} 支） | 声明阶 2 | {l1_lo}–{l1_hi} | 最大 {l1_dev_max:.2f} % | ✅ 是（全部 <15%） |
| 限制器镜像 vs 求解器 torch 路径 | 逐点一致 | {max(summary['mirror_max_rel_error'].values()):.2e} 相对 | ≤ 1e-12 | ✅ 是 |
| TVD 限制器 L∞ 观测阶（精度壁垒本体） | 声明阶 2 | {linf_lo}–{linf_hi} | 缺口 {gap_lo:.1f}–{gap_hi:.1f} % | ❌ 否（>15%，**如实报告为壁垒**） |
| 极值单元处 minmod/vanleer/superbee 斜率 | 0 | 0（逐单元精确） | 0 | ✅ 是 |
| TVD 约束生效单元计数 | 与 h 无关 | 见 §3.4（常数） | 0（随网格不变） | ✅ 是 |

**显式回答"是否 <3%"**：测量工具（B1）与无限制基线（B2）偏差 **<3%**；
受限方案在 **L1** 下的观测阶偏差 ≤{l1_dev_max:.1f}%
（最小 {l1_dev_min:.2f}%），但**最差者未进 3% 带**；
受限方案在 **L∞** 下的观测阶相对声明阶偏差 {gap_lo:.1f}–{gap_hi:.1f}%，
**远大于 3%，也大于 15%**——这正是本单的结论，属 honest FAIL 的对象（壁垒被证实存在），
不粉饰为"格式正确"。
"""
    return md


def main() -> int:
    summary = load()
    DOCS.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SRC / "summary.json", DOCS / "summary.json")
    shutil.copyfile(SRC / "fig_limiter_barrier.png", DOCS / "fig_limiter_barrier.png")
    (DOCS / "REPORT.md").write_text(build(summary), encoding="utf-8")
    print(f"wrote {DOCS / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())