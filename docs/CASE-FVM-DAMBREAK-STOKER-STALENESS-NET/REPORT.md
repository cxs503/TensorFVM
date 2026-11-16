# CASE-FVM-DAMBREAK-STOKER · 把溃坝算例纳入冻结网（D31 FAIL-1/FAIL-2 闭环），并查出目录 §19.1 与自身工件不一致

**锚点**: `CASE-FVM-DAMBREAK-STOKER`（feature_ledger，本单开工核验状态 `open`）· mission m23193 · 分支 `agent/m23193`
**参考**: Ritter (1892) *Z. Ver. Deutsch. Ing.* **36**, 947-954（干床稀疏波 `x_f = 2t√(g h0)`）; Stoker (1957) *Water Waves* ch. 10（湿床稀疏波+恒定态+水跃，`U` 由 Riemann 不变量定）
**容差**: 5 %（锚点同口径），`t* = 2`，池网格 `n = 1600`
**本单性质**: 补证据链欠账（**验证欠账类**），不重做锚点验收数字；认领 `docs/VERIFICATION_DEBT_LEDGER.md` D31 明确留下的两条 FAIL。

## ① 案例介绍（缺的是什么）

D31 的结论是「池条目的数字全部可用当前源码逐位复现」，但留了两条**机制级**欠账：

| 欠账 | D31 原文要点 | 后果 |
|---|---|---|
| **FAIL-1** | `docs/shallow-water-dam-break/` 没有 `manifest.json`，该算例**完全不在** `tests/test_source_staleness.py` 的冻结网里（53 例中无 dam-break） | 改 `shallow_water/dam_break.py`（或被它 import 的 `swe1d.py`/`exact.py`/驱动脚本）而**不重跑工件**，仓内没有任何门会红：存档数字变成不可核对的孤儿 |
| **FAIL-2** | 池条目缺 `dry_depth`（存档 `metrics.json` 生成于 `run_case` 记录该键之前） | 「第三方只凭条目能否重跑」= 否：12 个能推动被判数字的输入只记了 11 个 |

两者都是**可引用性**问题，不是物理精度问题，所以本单不动任何判据、不动容差、不动求解器。

参数（与池条目一致，程序注入）：`g = 9.81 m/s²`，`x = 0` 处间断；干床 `h0 = 1.0 m`（Ritter）、湿床 `h0 = 1.0 / h1 = 0.5 m`（Stoker）；`t* = t√(g h0)/h0 = 2`；域 `[-4, 6] m`；MUSCL(minmod)+HLL+SSP-RK3，CFL 0.4，order 2。

## ② 怎么做的

1. **`scripts/register_dam_break_staleness.py`（新增，`--write` / `--check` / `--check --json`）**：把 6 个生产文件
   （`pyproject.toml`、`shallow_water/__init__.py`、`swe1d.py`、`exact.py`、`dam_break.py`、`scripts/run_shallow_water_dam_break.py`）
   的 sha256 写进 `docs/shallow-water-dam-break/manifest.json`，并把**逐字节副本**冻结到
   `docs/shallow-water-dam-break/source-snapshot/`（156 KB）。digest 全部由脚本现算，无手抄。
2. **选「冻结」而不是「裸绑定」**：`source_staleness.freeze_violations()` 把「绑了生产代码却没冻结源码」判为欠账，
   裸绑定只会把新算例塞进 `known_unfrozen_cases`（增账）。冻结是仓内已文档化的可持续修法：存档数字永远可由同源的代码副本重导出。
   代价是「活树漂移」不再由 digest 报红 —— 那条由 `scripts/audit_dambreak_pool_provenance.py` P1（用**当前**源码重跑并与存档比对）
   负责，本单的 `--check` 也把活树 vs 冻结的差异列成可读漂移表（当前为空）。
3. **FAIL-2 用「重跑驱动」而不是手写补键**：`scripts/run_shallow_water_dam_break.py` 重跑（52.8 s 单核，n=1600/800 + 收敛 + 敏感性 + 前缘定义研究），
   `metrics.json` 的 `discretisation` 于是自带 `dry_depth`，再 `case_pool_dambreak.render_entry_json()` 重生成
   `docs/case-fvm-dambreak-stoker/case_pool_entry.json`，`scripts/generate_catalog_verification_entries.py --write` 重生成目录 §19.1。
4. **顺带修的真缺陷（生成器里的手抄数字）**：目录 §19.1 的结论句硬编码了「coarse dry-bed rows at n = 200/400 miss the front *series* (6.61 %)」。
   重生成后该句与自己上面的表格直接矛盾（表格里 n=200/400 已是 PASS），故把该句改为由条目自己的 `honest_fail` **派生**
   （`case_pool_dambreak._failing_cell_sentence`），不再有可手改的网格名与百分数。
5. **工具链**: `--check --json` 状态机 + `docs/CASE-FVM-DAMBREAK-STOKER-STALENESS-NET/{make_report.py,summary.json,fig_staleness_closure.png}`（本文件所有数字由 `make_report.py` 从仓内状态注入）。

## ③ 计算结果（`summary.json` 程序化注入）

**冻结网（`source_staleness.audit`，前=main HEAD，后=本单）**

| 量 | 前 | 后 |
|---|---:|---:|
| 被审算例数 | 53 | **54** |
| 绑定条目数 | 1378 | **1384** |
| fresh 绑定 | 1253 | **1259** |
| stale 算例 | 18 | **18（未增）** |
| 该算例状态 | 未注册 | **frozen=True, stale=False, 6/6 fresh** |
| 该算例是否冻结策略违规 | — | **否** |

账本 `docs/verification-archived-source-staleness/staleness.json` 由 `scripts/audit_archived_source_staleness.py --update-ledger`
复位，`ledger_matches_tree = True`；`known_stale_cases` / `known_unfrozen_cases` 两条棘轮都**没有**新增条目
（`unregistered_stale = []`、`unregistered_unfrozen = []`、`retired_unfrozen = []`）。

**池条目输入完备性（FAIL-2 验收）**

| case | 记全的输入 | 缺 |
|---|---:|---|
| `ritter-dry` | **12/12** | 无 |
| `stoker-wet` | **12/12** | 无 |

`dry_depth = 1e-12` 与 `dam_break.run_case` 的签名默认值一致（测试钉住）。重跑后**全部判据逐位不变**：除新增的
`dry_depth`、`wall_time_s`、`meta.generated_at/git_commit/command` 外，`metrics.json` 与 main 版本零差异
（1e-12 比较，`convergence.json`/`exactness.json`/`sensitivity.json`/`front_definitions.json` 逐字节未变）。

**意外收获：目录 §19.1 在 main 上就与自己的工件不一致**

| n | 目录里印的 x_f(t) 序列误差 | `convergence.json` 里的值 |
|---:|---:|---:|
| 100 | 4.695 % | **0.958 %** |
| 200 | 6.613 % | **0.639 %** |
| 400 | 6.613 % | **0.680 %** |
| 800 | 1.398 % | **0.532 %** |
| 1600 | 0.425 % | 0.425 % |

5 个网格里 4 个与工件不符，且结论句据此宣称「n = 200/400 未过序列门、PASS 在 n ≥ 800」——实际
`accepted_grid_min` 是 **400**（真正的 coarse-grid 失败只剩 `stoker-wet` n=100/200 的单调性与 `ritter-dry` n=100 的诊断量
`front_wetted`）。这属于**同一类欠账**（手抄数字活过了一次工件更新），本单以「重新生成 + 结论句派生」修掉。

## ④ 结果图

`fig_staleness_closure.png`（左：冻结网前后计数，stale 数不变即「没借新债」；中：目录印的 vs 工件里的干床序列误差，
>5 % 门线可见——红线是 main 上被印出去的旧值；右：池条目 12/12 输入完备性）。

## ⑤ 注意点（局限与适用范围）

* **冻结 vs 漂移是两个问题**：冻结保证「存档数字永远可由同源代码副本重导出」，**不**保证「当前源码仍产出这些数字」。
  后者由 P1 溯源码保证（本单 `--check` 报活树漂移为空）。若 CI 要自动化，应接 P1 而不是指望 digest 门。
* **`dry_depth` 是隐藏自变量**（D30 已量化：`1e-6` 时 `front_wetted` 破 5 %）。本单只是把它**记进条目**，
  没有改变其取值（`1e-12`，出厂默认），因此锚点数字不受影响；若将来改默认值，条目会立刻可见差异。
* **`forward-compat`**：`manifest.json` 的 `impl_files` 只覆盖溃坝算例的直接生产链；`numpy` 版本、BLAS 线程数等运行环境
  不在 digest 范围内（存档 `manifest.json` 已记 `python/numpy/platform`，但仍不是按位可复现的强保证）。
* 未做（not_run）：二维/斜置溃坝的同一冻结；`hires.json`/`bore_subcell.json` 的重新派生（它们由另外的驱动生成，本单未重跑，仍按原样存档）。

## ⑥ 误差判定（vs 基准逐项，显式回答「是否 <3%」）

| 判据（primary） | 解析基准 | 数值（池网格 n=1600，重跑后与 main 逐位一致） | 相对误差 | <3 % ? |
|---|---|---:|---:|---|
| `front_fan<5%`（干床波前 `x_f(t*)`） | `2 t√(g h0) = 4.0070 m` | 见条目 | 0.0173 % | **是** |
| `front_series<5%`（整条 `x_f(t)`） | Ritter 序列 | — | 0.0425 % | **是** |
| `shock_position<5%`（湿床水跃 `x_s(t*)`） | `x_dam + U t*`, `U = 2.9579 m/s` | — | 0.0210 % | **是** |
| `step_ratio<5%`（静水区水位阶跃） | `h2/h1 = 1.45384` | — | 5.76e-5 % | **是** |
| `monotone(no_extrema)`（无伪振荡） | 0 个内部极值（>1e-4 h0） | 0 | — | **是** |

四项数值判据全部通过，最大 0.0425 %，**远小于 3 %**（比锚点 5 % 门低两个量级）；本单未改动任何判据或读数路径，
锚点验收结论维持 PASS。**本单自身的验收**（D31 两条 FAIL）也逐条闭合：冻结 `covered=True`（唯一此前为
`false` 的 P5 检查项）、条目 12/12 输入、棘轮零增长。
