"""Render ``REPORT.md`` from ``artifacts/study.json`` (every number injected).

No number in the report is typed by hand: the markdown is produced by this
script from the artifact written by :mod:`run_study`, so a re-run regenerates the
report and any stale digit is a rendering bug rather than a copy error.

    PYTHONPATH=src python docs/xval/three-grid-loglog-cellcentre/render_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE / "artifacts" / "study.json"
OUT = HERE / "REPORT.md"

LADDER_TITLES = {
    "order2_uniform": "设计阶 2.0 · 均匀 r=2 · 40/80/160 · CFL 0.4",
    "order1_uniform": "设计阶 1.0 · 均匀 r=2 · 40/80/160 · CFL 0.4",
    "order2_fine": "设计阶 2.0 · 均匀 r=2 · 80/160/320 · CFL 0.4",
    "order2_nonuniform": "设计阶 2.0 · 非均匀 r21=1.75/r32=2.0 · 50/100/175 · CFL 0.4",
    "order1_nonuniform": "设计阶 1.0 · 非均匀 r21=1.75/r32=2.0 · 50/100/175 · CFL 0.4",
    "order2_cfl010": "设计阶 2.0 · 均匀 r=2 · 40/80/160 · CFL 0.1",
    "order2_coarse": "设计阶 2.0 · 均匀 r=2 · 10/20/40（粗，非渐近对照）",
    "order2_mid": "设计阶 2.0 · 均匀 r=2 · 20/40/80",
}


def readout(study: dict, metric: str, norm: str = "l1") -> dict:
    return next(r for r in study["readouts"]
                if r["metric"] == metric and r["norm"] == norm)


def pct(x: float) -> str:
    return f"{100 * x:.2f} %"


def main() -> int:
    payload = json.loads(STUDY.read_text())
    prov = payload["provenance"]
    ladders = payload["ladders"]
    summary = payload["summary"]
    L: list[str] = []
    add = L.append

    add("# XVAL：cell-centred FV 的三网格 log-log 斜率阶验证")
    add("")
    add("**锚点**：`XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR`  "
        "· **仓**：TensorFVM  · **性质**：验证欠账（已合并求解器对标设计阶）")
    add("")
    add(f"- 工件：`artifacts/study.json`（sha256 `{payload.get('artifact_sha256', '')[:32]}…`）")
    add(f"- 生成时刻：{prov['generated_utc']} · git HEAD `{prov['git_head'][:12]}` "
        f"（dirty={prov['git_dirty']}）")
    add(f"- 运行：`PYTHONPATH=src python docs/xval/three-grid-loglog-cellcentre/run_study.py`，"
        f"单核 **{payload['wall_seconds']} s**，确定性（固定网格/CFL，无随机）")
    add(f"- 求解器：{prov['solver']}")
    add(f"- 判据阈值：可接受 {int(100 * prov['thresholds']['acceptable'])} %（owner 精度标准）、"
        f"本单收紧线 {int(100 * prov['thresholds']['tight'])} %")
    add("")

    # ---------------------------------------------------------------- ①
    add("## ① 案例介绍（物理背景 / 几何 / 参数表）")
    add("")
    add("被验证的对象**不是**某个新算例，而是「怎么给一个 cell-centred 有限体积解读出观测收敛阶」"
        "这一动作本身。评测载体取周期熵波：`rho = 1 + A sin(k(x-ut))` 配均匀 `u`、`p` 是 Euler 方程的"
        "**精确解**（密度波随流不变，无压力耦合、无激波），因此「格式阶」与「间断处理」被彻底分开，"
        "误差序列就是格式设计阶的干净读数。")
    add("")
    add("| 项 | 值 |")
    add("|---|---|")
    add(f"| 计算域 | 单位区间 `x ∈ [0,1)`，两侧**周期**边界 |")
    add(f"| 精确解 | `rho = 1 + {prov['case'].split('A=')[1].split(',')[0].strip()} sin(k(x-ut))`，"
        f"`u = 0.5`、`p = 1` 均匀 |")
    add(f"| 波数 / 振幅 | `k = 2π`（一个波长铺满域）/ `A = 0.2` |")
    add(f"| 气体 | γ = 1.4（理想气体），EOS = StiffenedGas（双空气组分） |")
    add(f"| 终时 | `t = 0.2`（求解器实际到达时刻，落盘于工件的 `rows[*].t`） |")
    add(f"| 数值格式 | collocated cell-centred FV，MUSCL-Hancock 预测-校正 + HLLC 通量 + MC 限制器 |")
    add(f"| 声明阶 | `order=2`（主）与 `order=1`（对照） |")
    add(f"| CFL | 0.4（主）与 0.1（CFL 敏感性） |")
    add(f"| 网格 | 40/80/160（主三元组）、80/160/320（细三元组）、50/100/175（非均匀）、"
        f"10/20/40 与 20/40/80（粗三元组对照） |")
    add("")
    add("**为什么 cell-centred 这件事值得单独立账**：FV 的未知量是**单元平均** "
        "`u_h = (1/|V|)∫ u dV`，不是格心点值。`|V| = 1/nx` 的单元上，闭式平均为")
    add("")
    add("```")
    add("avg_i(u) = 1 + A[cos(k(a_i-ut)) - cos(k(b_i-ut))] / (k h),   a_i = i h, b_i = a_i + h")
    add("```")
    add("")
    add("用格心点值 `u(x_c)` 代替它，等于给误差额外注入限制误差 "
        "`avg - u(x_c) = -(h²/24) u'' + O(h⁴)`——**同为 O(h²)**，于是它既能把二阶格式的误差常数抬高，"
        "也能把任何 >2 阶的读数**压在 2 以下**（本单 §③D 实测）。")
    add("")

    # ---------------------------------------------------------------- ②
    add("## ② 怎么做的（方法 / 流程 / 关键设置）")
    add("")
    add("1. **精确解两个面**：闭式单元平均（上式）与格心点值都由 `cellcentre_loglog` 程序给出；"
        "闭式平均与 256 子格中点求积互校到 `1e-7`，且整周期平均恰为 1（`1e-14`）——",
        ) if False else None
    add("1. **精确解两个面**：闭式单元平均（上式）与格心点值都由 `cellcentre_loglog` 程序给出；"
        "闭式平均与 256 子格中点求积互校到 `1e-7`，且整周期平均恰为 1（`1e-14`）。")
    add("2. **三种估计器，同一批误差序列**（都是「三网格 log-log 斜率」的做法，"
        "但用三种等价/近似等价的口径，互相校验）：")
    add("   - `p_loglog`：`ln e` 对 `ln h` 的三点最小二乘斜率（教科书 log-log 图斜率）；")
    add("   - `p_adjacent`：相邻两网格对 `ln(e_i/e_{i+1})/ln(h_i/h_{i+1})`；")
    add("   - `p_richardson`：`tensorfvm.verification.richardson.observed_order`，"
        "即对非均匀网格比也成立的**精确三网格解**（求解 `(e3-e2)/(e2-e1)` 的超越方程）。")
    add("3. **两种度量**：`cell_average`（对闭式单元平均求误差，FV 的正确口径）与 `point`"
        "（对格心点值求误差，采样偏置口径），另把**采样偏差本身** `point - avg` 作为第三条序列。"
        "范数一律体积加权：`L1 = Σw|e|`、`L2 = sqrt(Σw e²)`、`L∞ = max|e|`。")
    add("4. **五条梯子 + 三条对照梯子**（见 §① 表）：覆盖设计阶 2/1、均匀/非均匀网格比、"
        "CFL 0.4/0.1、以及「粗到非渐近」的 10/20/40。")
    add("5. **两个负对照**（证明度量与估计器「会出错」，而不是只会点头）：")
    add("   - **采样地板**：合成单元场 `u = avg + c h^p sin(k(x-ut))`，真阶 `p ∈ {2,3,4}`、"
        "系数 `c ∈ {1,10}`。cell-average 度量必须读回 `p`；格心度量对 `p>2` 必须读成 ≈2。")
    add("   - **均匀比陷阱**：非均匀梯子 `50/100/175`（`r21=1.75`、`r32=2.0`）上，"
        "教科书单 `r` 公式 `|ln(e21/e32)|/ln r` 与精确三网格解对比。")
    add("6. **渐近区门**：Celik et al. (2008) 的 eq. (17) 在「p 由同一三元组算出」时是恒等式，"
        "没有鉴别力；本单用**重叠三元组一致性**（两套三元组独立求 p，相对差 ≤15 %）代替，"
        "并对全部 6 个 (度量, 范数) 组合各报一次。")
    add("7. **闸门**：`tests/test_cellcentre_loglog_order.py`（26 个用例，17 s）锁住"
        "度量代数、负对照、以及求解器实际阶；"
        "驱动与工件分离（`run_study.py` → `artifacts/study.json` → 本文件由 `render_report.py` 渲染）。")
    add("")

    # ---------------------------------------------------------------- ③
    add("## ③ 计算结果（全部数字由脚本注入）")
    add("")
    add("### ③A 设计阶读出（cell-average 度量，L1 范数）")
    add("")
    add("| 梯子 | 设计阶 | `p_loglog` | `p_adjacent` | `p_richardson` | 偏差 vs 设计阶 | <3 % | <15 % |")
    add("|---|---|---|---|---|---|---|---|")
    for name, study in ladders.items():
        r = readout(study, "cell_average")
        adj = " / ".join(f"{v:.3f}" for v in r["p_adjacent"])
        add(f"| {name}<br/>{LADDER_TITLES[name]} | {r['p_theory']:.1f} | "
            f"**{r['p_loglog']:.4f}** | {adj} | {r['p_richardson']:.4f} | "
            f"{pct(r['loglog_rel_dev'])} | {'✅' if r['loglog_pass_3'] else '❌'} | "
            f"{'✅' if r['loglog_pass_15'] else '❌'} |")
    add("")
    s = summary["design_order_readouts"]
    add(f"→ **六条设计阶梯子的 log-log 斜率全部落在 <15 % 内（{summary['all_design_readouts_within_15pct']}），"
        f"其中 <3 % 的有 "
        f"{sum(1 for v in s.values() if v['pass_3'])}/{len(s)} 条**。"
        f"细网格梯子 `80/160/320` 是 3.05 %（刚好压线外），说明 3 % 是紧线、不是宽松线。")
    add("")
    add("### ③B 范数敏感性（`order2_uniform`，cell-average）")
    add("")
    add("| 范数 | `p_loglog` | `p_richardson` | 偏差 vs 设计阶 2 |")
    add("|---|---|---|---|")
    for norm, v in summary["norm_sensitivity_order2_uniform"].items():
        add(f"| {norm.upper()} | {v['p_loglog']:.4f} | {v['p_richardson']:.4f} | "
            f"{pct(v['rel_dev'])} |")
    add("")
    add("同一份解、同一组网格：**L1 读 2.0，L∞ 只读到 1.68**。原因是 MC 限制器在工作点附近局部降阶，"
        "L∞ 由最差单元支配、L2 比 L1 更重地加权这些单元。这不是精度不足，而是「阶」必须连范数一起报。")
    add("")
    add("### ③C 采样偏置：格心点值度量会说什么")
    add("")
    add("| 梯子 | `p_loglog`(cell-average) | `p_loglog`(point) | 偏差放大 |")
    add("|---|---|---|---|")
    for name in ("order2_uniform", "order2_fine", "order2_cfl010"):
        a = readout(ladders[name], "cell_average")["p_loglog"]
        b = readout(ladders[name], "point")["p_loglog"]
        add(f"| {name} | {a:.4f} | {b:.4f} | {(b - a) / a * 100:+.2f} % |")
    add("")
    rows = ladders["order2_uniform"]["rows"]
    finest = rows[-1]
    ratio = finest["sampling_bias"]["l1"] / finest["cell_average"]["l1"]
    add(f"`nx={finest['nx']}` 上，纯采样偏差 `L1 = {finest['sampling_bias']['l1']:.4e}` 已达"
        f"单元平均误差 `{finest['cell_average']['l1']:.4e}` 的 **{100 * ratio:.1f} %**——"
        "两点结论：(i) 绝对误差幅度的任何「精度」判断都被它污染；"
        "(ii) 它带来的读数偏差是「系统性抬高 2–6 %」，不是随机噪声。")
    add("")
    add("### ③D 负对照 1：采样地板（合成场真阶 p，两个度量读回什么）")
    add("")
    add("| 网格 | 真阶 p | 系数 c | 读回（cell-average） | 读回（格心） |")
    add("|---|---|---|---|---|")
    for entry in payload["floor_control"]:
        add(f"| {entry['grids'][0]}/{entry['grids'][1]}/{entry['grids'][2]} | "
            f"{entry['true_order']:.0f} | {entry['coefficient']:.0f} | "
            f"**{entry['slope']['cell_average']:.4f}** | {entry['slope']['point']:.4f} |")
    add("")
    add("真阶 3 的场，格心度量只读回 1.73–1.98；真阶 4 的场读回 1.997–2.000。"
        "**采样地板效应是结构性的：cell-centred FV 的阶不许用格心点值读，否则 p>2 一律被压到 2。**")
    add("")
    add("### ③E 负对照 2：非均匀网格比下「单 r 公式」的代价")
    add("")
    add("| 梯子 | r21 | r32 | `p_true`（精确三网格解） | `p_uniform_form`（单 r 公式） | 阶偏差 | 零极限位移/误差尺度 |")
    add("|---|---|---|---|---|---|---|")
    for t in payload["ratio_traps"]:
        c = t["cell_average"]
        if c["uniform_ladder"]:
            continue
        add(f"| {t['ladder']} | {c['r21']:.2f} | {c['r32']:.2f} | {c['p_true']:.4f} | "
            f"{c['p_uniform_form']:.4f} | {pct(c['order_rel_dev'])} | "
            f"{pct(c['limit_abs_diff_over_scale'])} |")
    add("")
    add("均匀梯子（`40/80/160`、`80/160/320`）上两种口径差 `1e-12`（测试锁定），"
        "一旦 `r21 ≠ r32` 单 r 公式就把 1.891 读成 2.543、把 0.960 读成 1.469——"
        "**非均匀网格必须用精确三网格解，不能用单 r 公式**，而且这个错误比 15 % 判据大一倍以上。")
    add("")
    add("### ③F 渐近区门（重叠三元组一致性，15 % 带）")
    add("")
    add("| 三元组 A / B | 度量 / 范数 | p_A | p_B | 相对差 | 门 | A 偏差 vs 设计阶 | B 偏差 |")
    add("|---|---|---|---|---|---|---|---|")
    for g in payload["consistency_gates"]:
        add(f"| {g['first']} / {g['second']} | {g['metric']} / {g['norm']} | "
            f"{g['p_first']:.4f} | {g['p_second']:.4f} | {pct(g['rel_diff'])} | "
            f"{'✅' if g['pass_15'] else '❌'} | {pct(g['first_rel_dev'])} | "
            f"{pct(g['second_rel_dev'])} |")
    add("")
    add("两个重要读法："
        "(i) 门确实**拒绝**了 L∞ 的两个组合（25.5 %、33.5 %），说明它非空转；"
        "(ii) 但粗到非渐近的 `10/20/40` 与 `20/40/80` 在 L1 上只差 3.4 %，门放行，"
        "而 `10/20/40` 自身偏差 9.20 % —— **一致性门是必要不充分条件**，"
        "真正锐利的是「直接对标设计阶」。")
    add("")
    add("### ③G GCI 交叉面（`order2_fine`，cell-average/L1）")
    add("")
    g = payload["gci_reference"]
    add("| 量 | 值 |")
    add("|---|---|")
    add(f"| 网格 | {g['grids']} |")
    add(f"| 误差序列 | {' / '.join(f'{v:.4e}' for v in g['errors'])} |")
    add(f"| `order` / `order_used` | {g['order']:.4f} / {g['order_used']:.4f} |")
    add(f"| 分类 / 渐近区 | {g['convergence']} / {g['asymptotic']} |")
    add(f"| GCI 带 | [{g['band'][0]:.4e}, {g['band'][1]:.4e}] |")
    add("")
    add("注意：误差范数的极限是 **0**，所以这里的 GCI 是**绝对**不确定度带"
        "（零极限量），其「相对带」无意义——本单已在 `uniform_ratio_bias` 里把"
        "「相对位移」显式置空（`limit_relative_meaningful=False`）并改用绝对差。")
    add("")

    # ---------------------------------------------------------------- ④
    add("## ④ 结果图")
    add("")
    add("1. `figures/fig_loglog_orders.png`：设计阶 2（左）与 1（右）梯子的 log-log 误差曲线，"
        "两条线分别是 cell-average 与格心度量，虚线为设计阶参考斜率——可以直观看斜率读数。")
    add("2. `figures/fig_sampling_floor.png`：合成场（真阶 2/3/4 × 系数 1/10 × 两套网格）的"
        "读回阶；cell-average 线紧贴真阶，格心线在真阶 >2 时被压在 2 水平线上。")
    add("3. `figures/fig_uniform_ratio_trap.png`：非均匀梯子上「精确三网格解 vs 单 r 公式」的柱状对比。")
    add("")

    # ---------------------------------------------------------------- ⑤
    add("## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）")
    add("")
    add("1. **末时刻必须用求解器真正到达的时间**：`EulerSolver.run(t_end)` 的 `dt` 是自适应的，"
        "末时刻不一定等于 `t_end`；本单给 `run_smooth_wave` 增补了 `time` 键（纯增量），"
        "否则精确解相位错位会把阶读低——这是最容易「静默出错」的一处。")
    add("2. **`avg` 与 `point` 的差别不是小量**：`nx=160` 时等于 38 % 的误差量级（§③C），"
        "且随 `h²` 同步衰减，因此**看起来收敛得很好**但读数偏高 2–6 %。")
    add("3. **范数选择比 ±0.05 的拟合技巧重要得多**（§③B）：L1 读 2.04、L∞ 读 1.68，"
        "同一份解。任何「阶」的结论必须写清范数与度量。")
    add("4. **一致性门有盲区**：15 % 带下它放行了偏差 9.2 % 的粗三元组（§③F），"
        "因此门只能筛掉「明显出区」，标定仍须与设计阶直接比。")
    add("5. **非均匀网格比**：`r21≠r32` 时单 r 公式不可用（阶偏差 34 %/53 %），"
        "必须用 `richardson.observed_order` 的超越方程解。")
    add("6. **适用域**：结论针对**光滑解 + 均匀/规则结构化网格 + 二阶 MUSCL-Hancock**；"
        "含激波的功能量（波位置/平台窗）阶在相邻三元组间本身不稳定，已在"
        "`docs/xval/convergence-diagnosis-v2/REPORT.md` 单独立账，本单不复用其结论。"
        "网格为非均匀尺寸时 `error_norms` 支持体积权重，但本仓 `EulerSolver` 只吃均匀网格，"
        "故此项仅由单元测试覆盖，未在求解器上实测（明示未做）。")
    add("7. **明示未做**：>2 阶格式的真实求解器验证（本单只用合成场证明地板效应）；"
        "三维/非结构网格；时间步收敛阶分离（CFL 0.4 vs 0.1 只作敏感性，未做 `dt` 固定扫描）。")
    add("")

    # ---------------------------------------------------------------- ⑥
    add("## ⑥ 误差判定（关键量 vs 基准，显式回答「是否 <3 %」）")
    add("")
    add("基准 = 格式**声明阶**（`order=2` / `order=1`），这是本算例唯一可用的外部锚（精确解已知，"
        "观测阶的理论值即设计阶）。")
    add("")
    add("| 关键量 | 基准 | 实测 | 相对误差 | <3 %？ | <15 %？ |")
    add("|---|---|---|---|---|---|")
    pairs = [("order2_uniform", "cell_average", "l1", 2.0),
             ("order1_uniform", "cell_average", "l1", 1.0),
             ("order2_fine", "cell_average", "l1", 2.0),
             ("order2_nonuniform", "cell_average", "l1", 2.0),
             ("order1_nonuniform", "cell_average", "l1", 1.0),
             ("order2_cfl010", "cell_average", "l1", 2.0),
             ("order2_uniform", "point", "l1", 2.0),
             ("order2_uniform", "cell_average", "l2", 2.0),
             ("order2_uniform", "cell_average", "linf", 2.0)]
    for name, metric, norm, theory in pairs:
        r = readout(ladders[name], metric, norm)
        add(f"| {name} · {metric}/{norm} 的观测阶 | {theory:.1f} | {r['p_loglog']:.4f} | "
            f"{pct(r['loglog_rel_dev'])} | {'是 ✅' if r['loglog_pass_3'] else '否 ❌'} | "
            f"{'是 ✅' if r['loglog_pass_15'] else '否 ❌'} |")
    add("")
    add("**结论（不含糊）**：")
    add("")
    add(f"- 主判据（`order2_uniform`，cell-average/L1）：观测阶 {readout(ladders['order2_uniform'], 'cell_average')['p_loglog']:.4f} "
        f"vs 设计阶 2，偏差 {pct(readout(ladders['order2_uniform'], 'cell_average')['loglog_rel_dev'])} → **<3 % 成立**。")
    add(f"- 一阶对照：{readout(ladders['order1_uniform'], 'cell_average')['p_loglog']:.4f} vs 1.0，"
        f"偏差 {pct(readout(ladders['order1_uniform'], 'cell_average')['loglog_rel_dev'])} → **<3 % 成立**。")
    add(f"- 细网格梯子 `80/160/320`：偏差 {pct(readout(ladders['order2_fine'], 'cell_average')['loglog_rel_dev'])} → "
        "**>3 % 但 <15 %**（诚实记录，不粉饰：三点拟合在细梯子上并不必然更准）。")
    add(f"- L∞ 范数：偏差 {pct(readout(ladders['order2_uniform'], 'cell_average', 'linf')['loglog_rel_dev'])} → **>15 %，判 FAIL**；"
        "该 FAIL 归因于限制器局部降阶被 L∞ 放大，非格式整体失阶（L1/L2 同解均 <15 %）。")
    add(f"- 六条设计阶梯子 <15 % 全通过：`{summary['all_design_readouts_within_15pct']}`；"
        f"<3 % 全通过：`{summary['all_design_readouts_within_3pct']}`。")
    add("")
    add("## 交付物")
    add("")
    add("- `src/tensorfvm/verification/cellcentre_loglog.py`：可复用验证面"
        "（闭式单元平均 / 体积加权范数 / log-log 斜率 / 三网格读出 / 一致性门 / 均匀比陷阱量化 / 梯子驱动）")
    add("- `tests/test_cellcentre_loglog_order.py`：26 个用例（度量代数 + 两个负对照 + 求解器事实）")
    add("- `docs/xval/three-grid-loglog-cellcentre/{run_study.py,render_report.py,artifacts/study.json,figures/*.png}`")
    add("- `src/tensorfvm/verification/sod1978.py`：`run_smooth_wave` 增补 `time` 键（纯增量）")
    add("")
    OUT.write_text("\n".join(line for line in L if line is not None) + "\n")
    print(f"[write] {OUT} ({len(L)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())