# 交差审计 m17411：XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR（TensorFVM 三网格 log-log 斜率阶验证）

**审计单**: m17446（board-audit）｜**被审单**: m17411（succeeded, auto-approve dev_acceptance）｜**判级**: **L4**（基准验证：完整算例+误差分析+可复现）

## ① 案例介绍

m17411 是任务榜 XVAL 单：把 TensorFVM `tensorfvm.verification.ladder` 的三网格 log-log 斜率阶验证
（`observed_order`/`gci_three_grid`/`gci_study`，cell-centred FV）与独立参考实现（ASME V&V 20 /
Roache 1994 / Celik 2008 公式重实现）交叉对标，判据 <15%。物理载体：1D 对流-扩散 Pe=1
（精确解 T=(e^{Pe x}-1)/(e^{Pe}-1)，upwind 32/64/128、central 16/32/64，r=2，探测点 x=0.5）
+ Kovasznay Re=40 文献数值（溯源 m17018 CASE-FVM-KOVASZNAY-ORDER，零手抄）。

## ② 怎么做的（审计流程）

1. **取证**：API 拉 mission 17411 全记录（changed_files/headline/quality_audit）；读 m17411 工作树
   `docs/xval-three-grid-loglog-slope/`（REPORT.md/run_xval.py/reference_impl.py/xval_results.json/figures×3）
   与 `tests/test_xval_three_grid_loglog_slope.py`。
2. **基准裁判**：`SELECT case_name,metric,tolerance FROM tensor_benchmark_cases WHERE anchor_slug='XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR'`
   → **0 行**（表内共 53 行，无本锚点）→ NO_BENCHMARK_CASE，参考退化为独立重实现+精确解+溯源数值。
3. **复算**：loongarch64 本机重跑 `pytest tests/test_xval_three_grid_loglog_slope.py`（17 passed, 5.54s）
   与 `run_xval.py`（17/17 PASS），重生成 `xval_results.json` 与入库版 **JSON 逐键相等（byte-identical）**。
4. **指针核验**：REPORT.md 引用的 `ladder.py:417/468/506` 行号逐一 grep 命中；Kovasznay 三档 L2
   （1.25748e-02/3.83969e-03/1.21724e-03）与 `docs/CASE-FVM-KOVASZNAY-ORDER/VERIFICATION_REPORT.md:86-88` 一致。
5. **落表**：`tensor_maturity_verdicts` INSERT（repo=TensorFVM, scope=m17411…, level=4, verdict_by=board-audit）。

## ③ 计算结果（程序化注入自 xval_results.json，零手抄）

**T1 两网格观测阶（6 组）**

| 描述 | TF | REF | 误差 | 判定 |
|---|---:|---:|---:|:--:|
| 2nd order, r=2 | 2.0000000000 | 2.0000000000 | 0.000e+00% | PASS |
| 1st order, r=2 | 1.0000000000 | 1.0000000000 | 0.000e+00% | PASS |
| 3rd order, r=2 | 3.0000000000 | 3.0000000000 | 0.000e+00% | PASS |
| 2nd order, r=3 | 2.0000000000 | 2.0000000000 | 0.000e+00% | PASS |
| Kovasznay vel 16->32 | 1.7114737106 | 1.7114737106 | 0.000e+00% | PASS |
| Kovasznay vel 32->64 | 1.6573761902 | 1.6573761902 | 0.000e+00% | PASS |

**T2 三网格观测阶（2 组）**

| 描述 | TF | REF | 误差 | 判定 |
|---|---:|---:|---:|:--:|
| Upwind Pe=1 probe(0.5) | 0.9389422010 | 0.9389422010 | 0.000e+00% | PASS |
| Central Pe=1 probe(0.5) | 2.0007658807 | 2.0007658807 | 0.000e+00% | PASS |

**T3 GCI（2 组，fine 网格）**

| 描述 | TF GCI_fine | REF GCI_fine | 误差 | 判定 |
|---|---:|---:|---:|:--:|
| upwind GCI | 1.590742e-03 | 1.590742e-03 | 1.363e-14% | PASS |
| central GCI | 7.910829e-06 | 7.910829e-06 | 0.000e+00% | PASS |

**T4 完整 GCI 研究（1D 对流-扩散，端到端过 TF 求解器）**

| 方案 | TF 观测阶 | REF 三网格阶 | REF LS 斜率 | 误差 | 判定 |
|---|---:|---:|---:|---:|:--:|
| upwind | 0.938942 | 0.938942 | -0.970616 | 0.000e+00% | PASS |
| central | 2.000766 | 2.000766 | -2.000383 | 0.000e+00% | PASS |

**T5 log-log 斜率公式（5 组）**

| 预期 p | 观测 p | R² | 误差 | 判定 |
|---:|---:|---:|---:|:--:|
| 1.0 | 1.0000000000 | 1.0000000000 | 0.000e+00% | PASS |
| 1.5 | 1.5000000000 | 1.0000000000 | 0.000e+00% | PASS |
| 2.0 | 2.0000000000 | 1.0000000000 | 0.000e+00% | PASS |
| 2.5 | 2.5000000000 | 1.0000000000 | 0.000e+00% | PASS |
| 3.0 | 3.0000000000 | 1.0000000000 | 2.961e-14% | PASS |

**总计 17/17 PASS，最大误差 2.961e-14%（机器精度级）。**

## ④ 结果图

三图取自 m17411 工件（make_figures.py 生成，审计未改动）：fig1 收敛曲线（TF vs 精确解，upwind/central）、
fig2 观测阶对比（TF vs REF vs 名义阶）、fig3 GCI 不确定带（带含精确解）。

## ⑤ 注意点（审计发现，按严重度）

1. **工件未入库（流程缺口，非功能缺陷）**：m17411 交付物在 worktree 中为 untracked（`git status` 两条 `??`，
   mission `head_commit=null, branch=null`）——平台自动推送只带 commit，若 worktree 清理则工件丢失。
   功能实质不受影响（本审计已复算验证），但建议补 commit 固化。
2. **tensor_report id=16403 markdown 为空**（markdown_len=0，仅 title/summary/figures）：六段正文只存在于
   worktree REPORT.md，报告纪律欠账。
3. **质量门缺失**：mission `server_quality_audit: gate_valid=false, key_present=false`（auto-approve 时门未生效）。
4. **参考为公式级重实现**：非外部 OSS CFD 代码（loong64 无 OpenFOAM/SU2 预编译包，REPORT.md §8 已如实披露）；
   17/17 机器精度一致证明 ladder.py 公式实现正确，但不构成跨求解器物理对标。
5. Kovasznay 数值为继承值（m17018 溯源），m17411 未重跑该算例（n=64 需 7-10min，n=128 需 1-1.7h，集群档）。

## ⑥ 误差判定

| 关键量 | 基准 | 实测（审计复跑） | 是否 <3% |
|---|---|---:|:--:|
| 两网格阶 6 组 | 闭式/REF | 误差 0.00e+00% | ✅ 是 |
| 三网格阶 2 组 | Roache 公式 REF | 0.00e+00% | ✅ 是 |
| GCI_fine 2 组 | REF | ≤1.42e-14% | ✅ 是 |
| 完整研究阶 2 组 | REF | 0.00e+00% | ✅ 是 |
| log-log 斜率 5 组 | 已知阶 | ≤2.96e-14% | ✅ 是 |
| upwind 观测阶 vs 名义 1 | 名义阶 | 0.9389（偏差 6.1%，渐近区一阶格式正常） | 声明：格式阶非误差项 |
| central 观测阶 vs 名义 2 | 名义阶 | 2.0008（偏差 0.04%） | ✅ 是 |

**基准声明**：tensor_benchmark_cases 无本锚点正式参考行（NO_BENCHMARK_CASE），参考退化为独立重实现+精确解；
全部 17 项复跑误差 ≤2.96e-14% < 3%，**显式回答：是，<3%**。honest 披露：对标对象为本仓自身实现 vs 标准公式
（非第三方求解器），级别按功能实质判 L4，不虚标 L5。
