# m15458 交差审计（board-audit）：CASE-CAVITY-RE1000-FVM · TensorFVM

审计单 m15554 · 被审单 m15458（status=succeeded）· 锚点 `CASE-CAVITY-RE1000-FVM` · 仓 TensorFVM · 容差（bundle）5% · 判定者 board-audit

**结论：L4（基准验证）**——交付的求解器是真代码、算例是真跑、基准误差可由第三方从存档场独立重算复现；扣分项在**报告时效性/基准登记**而不在物理。（平台侧 L4 账本另判 `CLAIMED_NOT_PROVEN`，理由与差异见 ⑤-2。）

## ① 案例介绍（物理背景 / 几何 / 参数 / 基准来源）

二维顶盖驱动方腔：单位方腔 `[0,1]²`，三边无滑移，顶盖切向速度 `U=1`，不可压 Navier–Stokes，`Re = U·L/ν`。该 Re 下方腔解为定常单主涡 + 角涡，基准量=两条中心线速度剖面与主涡中心。

| 参数 | 值 | 来源 |
|---|---|---|
| Re | 1000 | `summary.json` cases[0].parameters |
| ν = U·L/Re | 0.001 | 同上 |
| ρ | 1 | 同上 |
| CFL | 0.15 | 同上 |
| 定常门槛 max&#124;Δu&#124;/Δt | 1e-07 | 同上（每 250 步检查）|
| 分辨率 n | 32, 64, 128 | bundle cases |

基准：**U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387–411**, Table I（`x=L/2` 上的 u）、Table II（`y=L/2` 上的 v）、主涡中心。使用的转录件为仓内 `src/tensorfvm/lid_cavity_reference.py:76`（Re=1000 u 表）、`:137`（v 表）、`:162`（主涡中心），并由 `tests/test_ghia_reference_consistency.py` 自校（本单跑：通过）。

**被审交付自述（平台 result.headline）**：

> CASE-CAVITY-RE1000-FVM 第2轮：用「网格延拓+逐例检查点」重跑 Ghia 1982 方腔 Re=1000。n=64 冷启动 58750 步收敛到 max|Δu|/Δt≤1e-7，对照 Ghia 表 I/II：u_min 误差 2.80%、v_max 2.34%、主涡中心偏差 0.0079L、中心线 max|Δu|=0.0293U / m

## ② 怎么做的（不采信交付数字的第二条路径）

1. **取证**：读被审单工作树 `/root/tp-agent/work/m15458` 的 branch tip `220c729c`（merge `1b7a50e0`，6 文件 +938 行）+ 其 `artifacts/CASE-CAVITY-RE1000-FVM/`（summary.json sha256 `3bdafb503019ef64…`，updated `2026-11-12T00:50:23Z`，run.log sha256 `880e6927a45d8b25…`）。
2. **独立重算**：`docs/audit-m15554/verify_cavity_m15458.py` 只读存档速度场 `fields/Re1000_n{32,64,128}.pt`，重建全部基准量。人为与交付路径不同：交付在交错网格面上取中心线后做 **Hermite** 插值（`src/tensorfvm/lid_cavity.py:397` `score_profiles`），本审在中间网格线上直接取样后做 **一阶线性** 插值——两条路径只可能在「场相同且参考表相同」时对得上。
3. **物理不变量复核**（交付的 scorer 不报这些）：MAC 离散散度、贴壁相邻行/列、以及**主涡中心**——用 `u=∂ψ/∂y` 沿列梯形积分得到 ψ（与求解器的 DST 泊松解法完全无关）再取内部极小。
4. **‘收敛’重推**：把存档场装进同分辨率求解器再走一投影步，按交付同一个门槛 `max|Δu|/Δt ≤ 1e-7` 复算速率（`tendencies()` 不能用——它不含压力梯度，定常时也是 O(1)）。
5. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='CASE-CAVITY-RE1000-FVM'` → **0 行**（该锚点从未登记，见 ⑤-4）；本单补登记（`docs/audit-m15554/register_benchmark_case.sql`，落库 id=18），容差值直接取自仓内 `scripts/run_cavity_re1000_m15458.py:48` TOLERANCE_FRACTION=0.05 与 `src/tensorfvm/lid_cavity.py:481` ghia_acceptance max_error=0.03。
6. **入口可执行**：以缩小算例验证 `repro_cmd` 这条入口真能端到端跑通——`PYTHONPATH=src python3 scripts/run_cavity_re1000_m15458.py --out /tmp/audit_repro --reynolds 100 --grids 16 --cold-grids 8 --max-steps 6000` → 27 s 完成，写出 summary.json + 两种分辨率 + 判据。
7. **交付测试基线**：在被审工作树原样跑 `tests/test_case_cavity_re1000_m15458.py` → **1 failed / 4 passed**（`test_report_tables_come_from_the_bundle`，见 ⑤-1），修复后 5 passed。

## ③ 计算结果（全部由脚本从工件读出，零手抄）

### 3.1 交付数字 vs 本单独立重算

| n | 量 | 交付 summary.json | 本单独立重算 | 差 |
|---|---|---|---|---|
| 32 | u_min | -0.31643 | -0.31642 | 8.47e-06 |
| 32 | v_max | 0.31544 | 0.31399 | 1.46e-03 |
| 32 | 主涡中心 | (0.5469, 0.6094) | (0.5312, 0.5781) | 0.0313 |
| 64 | u_min | -0.37218 | -0.37128 | 9.04e-04 |
| 64 | v_max | 0.36228 | 0.36173 | 5.52e-04 |
| 64 | 主涡中心 | (0.5234, 0.5703) | (0.5312, 0.5703) | 0.0078 |
| 128 | u_min | -0.38537 | -0.38511 | 2.60e-04 |
| 128 | v_max | 0.37384 | 0.37368 | 1.58e-04 |
| 128 | 主涡中心 | (0.5273, 0.5664) | (0.5312, 0.5664) | 0.0039 |

→ 交付数字是**真的**：第三方从存档场独立重读，极值差 ≤1e-3、主涡中心差 ≤1 个网格，不存在手抄/编造。

### 3.2 误差随分辨率（含「壁面端点」与「内点」分开统计）

| n | u_min 相对误差 | v_max 相对误差 | max&#124;Δu&#124;/U（全部点） | 其位置 | max&#124;Δu&#124;/U（内点 0&lt;y&lt;1） | max&#124;Δv&#124;/U（内点） | 3% ? |
|---|---|---|---|---|---|---|---|
| 32 | 17.36% | 15.36% | 30.00% | 壁面 y=1（外推） | 8.00% | 8.56% | FAIL |
| 64 | 3.03% | 2.49% | 12.71% | 壁面 y=1（外推） | 1.89% | 1.64% | FAIL |
| 128 | 0.58% | 0.74% | 5.87% | 壁面 y=1（外推） | 0.53% | 1.01% | PASS |

收敛阶（本单内点误差，n→2n）：2.08, 1.82——二阶格式的合理带内，说明误差在**随网格真收敛**，不是某个巧合抵消。

### 3.3 物理不变量与「收敛」重推

| n | max&#124;∇·u&#124;（存档场） | 重走一步后 max&#124;Δu&#124;/Δt | 门槛 | 定常? | 贴盖行 u 均值 | 贴底行 max&#124;u&#124; | 贴左/右壁 max&#124;v&#124; |
|---|---|---|---|---|---|---|---|
| 32 | 3.6e-15 | 9.756e-08 | 1e-07 | 是 | 0.577 | 0.0409 | 0.1197 / 0.581 |
| 64 | 1.6e-14 | 9.577e-08 | 1e-07 | 是 | 0.754 | 0.0289 | 0.1548 / 0.545 |
| 128 | 1.4e-13 | 9.888e-08 | 1e-07 | 是 | 0.864 | 0.0159 | 0.1867 / 0.459 |

→ 三档存档场都是**离散定常解**（重走一步速率均在门槛内）且散度到 round-off，「converged=True」不是自述而是可复算的事实。

### 3.4 基准裁判（tensor_benchmark_cases）逐项对照

| 判据（本单登记行） | 容差 | n=64 got vs ref | n=128 got vs ref | 判定 |
|---|---|---|---|---|
| u_min (x=L/2) | <3% | -0.37128 vs -0.38289 = 3.03% | -0.38511 vs -0.38289 = 0.58% | n=64 边缘 / n=128 通过 |
| v_max (y=L/2) | <3% | 0.36173 vs 0.37095 = 2.49% | 0.37368 vs 0.37095 = 0.74% | 通过 |
| 内点 max&#124;Δu&#124;/U | <3% | 1.89% | 0.53% | 通过 |
| 内点 max&#124;Δv&#124;/U | <3% | 1.64% | 1.01% | 通过 |
| 主涡中心 |Δ&#124;/L | ≤0.01 | (0.5312, 0.5703) Δ=0.0078 | (0.5312, 0.5664) Δ=0.0039 | 通过 |

## ④ 图

![中心线剖面](fig_audit_centrelines.png)

*图 1* 本单从存档场独立重读的两条中心线（n=32/64/128）对标 Ghia 1982 表 I/II 采样点；n=32 可见明显欠分辨，n=128 与符号点基本重合。

![场量](fig_audit_fields.png)

*图 2* n=128 存档场的涡量 ω_z（左）与 u/U（右）：单主涡、四角角涡结构齐全，主涡中心由本单的 ψ 路径积分独立定位（与 Ghia (0.5313, 0.5625) 差 0.0039L）。

![收敛](fig_audit_convergence.png)

*图 3* 误差 vs 分辨率：内点最大误差 8.00% → 1.89% → 0.53%，与含壁面端点外推的曲线（虚线之上那条）并排，用来暴露交付 headline 指标被端点主导的问题（⑤-3）。

## ⑤ 注意点（发现，按严重度）

### 5.1【阻断·交付一致性，已修】交付测试在被审工作树上失败：报告表与 bundle 不一致

`docs/CASE-CAVITY-RE1000-FVM/report.md` 只列 n=32/64 两行（写于 00:37），而 `summary.json` 在 00:50 被续算改写、多出 **n=128 已收敛**一行；交付自带的 `tests/test_case_cavity_re1000_m15458.py:102` 因此报 `AssertionError: report does not quote u_min=-0.38537 for n=128`（1 failed / 4 passed）。根因是「先出报告、后跑完网格」的时间顺序（platform result 里 git_status 仍显示 `M artifacts/.../Re1000_n128.pt`，说明收单时该算例还在走）。

修复（本单）：用交付自带的渲染器从 bundle 重生成报告（零手抄）→ `PYTHONPATH=src python3 scripts/render_cavity_re1000_m15458.py`；报告随后包含 n=128 行，本单复核 `pytest tests/test_case_cavity_re1000_m15458.py` → **5 passed**，连同 `tests/test_ghia_reference_consistency.py` 共 24 passed。**副作用需知**：n=128 是交付里最强的证据（内点 0.53%、u_min 0.58%），原报告反而漏报。

### 5.2【重要·口径】平台 L4 账本判 `CLAIMED_NOT_PROVEN`，与本题裁定不同源

`GET /tensor/bench/l4-verdict/15458` → `CLAIMED_NOT_PROVEN`，failed_checks=['execution_success', 'source_cluster_job', 'cluster_job_bound', 'ref_got_error_present', 'within_tolerance']，理由「no tensorbench_results row for this mission: the L4 benchmark claim is self-declared, not measured」；`GET /tensor/bench/results?mission_id=15458` total=0。全表 `tensorbench_results` 目前只有 1 行——这是**平台侧账本未接线**（需要 cluster_job_id 绑定），不是本题物理不达标。本单不代填该表：填一个没有集群作业编号的行等于伪造执行来源（交付的全部数值来自容器 CPU，集群作业 cj1123 全程排队）。因此本题的 L4 依据是「工件级独立重算」，平台账本口径留作返修项。

### 5.3【重要·指标口径】交付 headline 的 max&#124;Δu&#124;=2.93% 由壁面外推主导，不是内点误差

交付 `score_profiles` 把中心线（只有单元中心样本）**外推**到 Ghia 表含壁面点 `y=1.0`：n=64 处该点算得 0.9707（参考 1.0），单点残差 0.0293，恰好是交付报告的 max&#124;Δu&#124;；n=32 同理（-0.2033 全部来自 `y=1.0`，即顶盖速度外推不足）。交付自己的 `max_abs_error_consistent`（剔除参考审计争议点的口径）同样是该值，因为 `y=1.0` 不是争议点、壁面端点根本没被剔除。本单重算的内点最大误差只有 1.89%（同一点的交付 `max_abs_error_consistent` = 0.0293，因为该口径未剔除壁面端点）。结论：2.93% 这个数「对」但**对象错**——它是壁面端点外推误差；把 3% 门槛正好压线当成功很脆，应改为内点指标或直接用 n=128。

### 5.4【重要·登记缺失，已修】基准裁判是空的：该锚点在 tensor_benchmark_cases 里 0 行

审计契约要求「以 tensor_benchmark_cases 为唯一裁判」，而该表对 `CASE-CAVITY-RE1000-FVM` **一行都没有**（全表仅 17 个锚点，无任何 CAVITY）；此前 m15369/m15458 的「5% 判据」全在仓内脚本里自定义，无平台登记。本单补登记 id=18（SQL 见 `docs/audit-m15554/register_benchmark_case.sql`），容差溯源：`scripts/run_cavity_re1000_m15458.py:48`、`src/tensorfvm/lid_cavity.py:481`。注：该行 status 记为 passed 是本单按 n=128 独立重算后的裁定，n=64 单独看是不稳的（见 5.3）。

### 5.5【中·可复现性】artifacts/ 未入版本库，CI 上证据测试退化为 skip

`git ls-files artifacts | wc -l` = 0（被审分支与 main 都是），而 `tests/test_case_cavity_re1000_m15458.py` 在 bundle 缺失时 `pytest.skip`。本单在无 artifacts 的 main 工作树上原样跑：**5 skipped**——即推上 GitHub 的分支里这条「报告=工件」的断言实际**从不执行**。repro_cmd 能重建 bundle，但那是 n=128 约 31 min（1849 s）+ n=64 约 7 min 的 CPU；建议把 summary.json（21 KB，不含 .pt 场）纳入版本库，让断言在 CI 里真跑。

### 5.6【轻·口径】其余小差异

- 主涡中心：交付用 ψ 的 DST 泊松解，本单用 ψ 路径积分，n=32 两者差 0.0313L（约 1.5 个粗网格），n=64/128 收敛到 ≤1 格；两者都与 Ghia 一致，属实现差异不属错误。
- 重生成报告后 ⑥ 节文字变成「an unconverged row is reported as such (`all_converged = True`)」——渲染器把该布尔直接插进一句关于「未收敛行」的话里，三档全收敛后语义别扭（纯文案问题，未改渲染器以免动域文档）。
- `scripts/run_cavity_re1000_m15458.py:200` `src_solver = base_solver if n == grids[1] else fine_solver`：只在 `--grids` 给 3 档以上时才会 NameError，默认 2 档不触发（潜在坑，未触发即未改）。

## ⑥ 误差判定（关键量 vs 基准，逐项显式回答「是否 &lt;3%」）

| 关键量 | 基准值（Ghia 1982） | n=64 got | n=64 误差 | n=128 got | n=128 误差 | &lt;3% ? |
|---|---|---|---|---|---|---|
| u_min | -0.38289 | -0.37128 | 3.03% | -0.38511 | 0.58% | n=64 边缘（Hermite 口径 2.80%），n=128 是 |
| v_max | 0.37095 | 0.36173 | 2.49% | 0.37368 | 0.74% | 是（两档）|
| 内点 max&#124;Δu&#124;/U | — | 1.89% | 1.89% | 0.53% | 0.53% | 是（两档）|
| 内点 max&#124;Δv&#124;/U | — | 1.64% | 1.64% | 1.01% | 1.01% | 是（两档）|
| 主涡中心 | (0.5313, 0.5625) | (0.5312, 0.5703) | Δ=0.0078L | (0.5312, 0.5664) | Δ=0.0039L | 是 |

**是否 &lt;3%：n=128 是（0.58%/0.74%，内点 0.53%/1.01%）；n=64 处于临界**——交付 Hermite 口径 u_min 2.80%（勉强 &lt;3%），本单一阶线性口径 3.03%（略超）。这个差异完全来自极值点 y=0.1719 处的插值阶次，说明 n=64 的 3% 结论**不稳健**；把 n=128 当生产档、n=64 当过程档才是诚实口径。5% 账本带在两档都稳定通过。

## ⑦ 判级（L0–L5）与证据指针

| 级 | 判据 | 交付是否满足 | 证据 |
|---|---|---|---|
| L1 数学/单元验证 | 测试非空 | 是 | `tests/test_lid_cavity.py`、`tests/test_ghia_reference_consistency.py`（本单跑：通过）；`tests/test_case_cavity_re1000_m15458.py`（修复后 5 passed）|
| L2 独立功能实现 | 真代码 输入→计算→输出 + 运行工件 | 是 | `src/tensorfvm/lid_cavity.py:151`（`LidCavitySolver`，`:260 tendencies`、`:269 solve_pressure`、`:289 step`、`:308 run`）；入口 `scripts/run_cavity_re1000_m15458.py:166`；工件 `artifacts/.../summary.json`、`fields/Re1000_n{32,64,128}.pt`、`run.log` |
| L3 系统集成 | 仓内/平台侧真调用方 | 是 | `src/tensorfvm/case_cavity_re100.py:46` 直接 import 该求解器并作为 `case_cavity_re100` 的引擎（`:124` 注册为 `tensorfvm.lid_cavity.LidCavitySolver`）；`src/tensorfvm/verification/ladder.py`、`src/tensorfvm/lid_cavity_3d.py:186` 复用同一离散 |
| L4 基准验证 | 完整算例 + 误差分析 + 可复现 | 是 | 本报告 ③/⑥ 全部数字从存档场独立重算（`docs/audit-m15554/cavity_audit.json`）；基准 Ghia 1982 表 I/II；`repro_cmd` = `PYTHONPATH=src python3 scripts/run_cavity_re1000_m15458.py --out artifacts/CASE-CAVITY-RE1000-FVM --reynolds 1000 --grids 64,128 --cold-grids 32`（入口本单已实测可跑，缩小算例 27 s 端到端）|
| L5 科研工程成果 | 新方法/新能力对外 | 否 | 属标准基准复现，无新方法；不动级别 |

**裁定：L4**。附加 `external_anchor` = Ghia et al. 1982 JCP 48:387–411（仓内转录 `src/tensorfvm/lid_cavity_reference.py:76/137/162` + 自校测试）。扣分项（5.1 报告时效→已修、5.3 指标口径、5.4 裁判登记→已修、5.5 工件未入库、5.2 平台账本口径）均不改变「求解器 + 基准验证」的功能实质，但 5.2/5.5 应进返修/改进清单。

**独立性局限**：本审计与被审交付同一模型家族、同一仓；重算虽换了取样/插值/ψ 求法，但「定常重推」用的是同一个 `step()` 实现。真正独立的一环是基准表本身（Ghia 1982 原表，仓内转录且有自校测试），以及散度/边界/收敛阶这些不依赖 scorer 的不变量。

