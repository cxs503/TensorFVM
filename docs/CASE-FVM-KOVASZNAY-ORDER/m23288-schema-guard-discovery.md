# CASE-FVM-KOVASZNAY-ORDER · schema 新鲜度门从「手写目录表」改为「按内容自发现」（并诚实记 n=256 仍 WITHHELD）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER` · 判据: L2 收敛阶 ≥ 1.8，指标 = 32/64/128 网格 log-log 斜率
**外部基准**: Kovasznay, L. I. G. (1948), *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/S0305004100023999
**席位 / 仓**: m23288（重开 PR#856 的 m20571 交付）· TensorFVM · 分支 `agent/m23288`

---

## 1. 案例介绍

Kovasznay 流是二维定常不可压 Navier–Stokes 方程的解析解（`u = 1 − e^{λx}cos 2πy`，
`v = (λ/2π)e^{λx}sin 2πy`，`λ² − Re λ = 4π²`，λ = −0.9637405），单位域 [0,1]²，
对称挤出 n×n×1 网格。台账判据只看 32→64→128 加密时体积加权相对 L2 误差的 log-log 斜率
是否 ≥ 1.8。本单的物理内容与 m20571 交付**完全一致**（该交付已在 main 上，见 §4），
本单动的是**证据与门之间的绑定方式**。

| 参数 | 值 |
|---|---|
| Re | 40 |
| 域 / 网格 | [0,1]²，对称挤出 n×n×1（cells = n²）|
| 对流格式 | 2 阶 hybrid |
| 对称面闭合 | `zero_gradient`（du/dz = dv/dz = 0）|
| 算法 / 线性 | coupled + reused_lu，松弛 0.7，tol 1e−8 |
| 网格序列 | 32 / 64 / 128（n=256 第四层本单未取得，见 §5.1）|

## 2. 怎么做的

### 2.1 先确认 m20571 交付在当前 main 上的实际状态

任务书列出的六件交付（诊断书、`report_payload_m20571.py`、三网格与两网格 `RESULTS.md`、
`summary-32-64/summary.json`、`n128/fields-error.png`）在本单基线 `cc0ba3b0` 上**全部存在**，
且门是绿的（§3 表）。因此本单**不做重复交付**，只处理该交付链上仍然敞着的一处结构性缺陷。

### 2.2 缺陷：schema 新鲜度门自己会漂移

m20571 新增的门（`tests/test_kovasznay_32_64_128_acceptance.py`）用一张**手写目录表**
`SUMMARY_DIRS` 指定"哪些 `summary.json` 必须与当前 summarizer 同代"。这张表有两个缺陷，
而且两个都属于它自己要防的那一类：

1. **表里有一条不存在的路径**（`summary-32-64-128-256/`，n=256 从未跑成）——该参数位永远
   `skip`，看上去"覆盖了"，实际什么都没检查；
2. **新增的汇总目录不会被覆盖**：任何人（包括未来的四层总结）一旦提交新的
   `summary.json` 目录，除非同时改测试，否则**不在门内**——这正是 m20239→m20571
   那次事故（重新生成的证据被提交过滤器滤掉，树上留旧代工件）的同构复发路径。

### 2.3 修法：按内容契约发现，而不是按路径枚举

把 `SUMMARY_DIRS` 换成 `_summariser_documents()`：递归扫描算例目录下所有 `summary.json`，
**凡自称是当前 summarizer 输出的**（带 `provenance.summarizer == scripts/summarize_kovasznay_32_64_128.py`，
或带有仅该脚本写出的字段 `observed_orders / fits / verdict / pair_consistency /
three_level_estimates` 之一）都必须满足完整 schema 键集。

这样"哪一代"由**文件自身内容**决定，不由维护者记性决定：

* 同目录下不同来源的汇总（每层 `<n>/summary.json` 由 runner 写、`inplane-decomposition/summary.json`
  由另一脚本写）不会被误判——判据是内容契约而非路径；
* 新提交的汇总目录（例如未来的 `summary-32-64-128-256/`）**自动进闸**，无需改测试；
* 被过滤器滤掉的"未提交"仍由既有新鲜度/可复算门负责；本门负责"提交了就一定是同代"。

配一条**非空性控制** `test_schema_guard_discovers_the_committed_summariser_outputs`：发现
到的目录必须覆盖已提交的已知目录（当前 3 个：三网格根、`summary-32-64/`、`selftest-4-8-16/`），
否则报错——**发现逻辑本身坏掉不许让门变绿**。

## 3. 计算结果（门与数字）

### 3.1 数值未被本单改写（先证明再动手之外的改动都是浪费）

从已提交 npz 现场重跑 summarizer 到 `/tmp`，与仓库中的 `summary.json` **逐叶值相等**
（去掉随机器路径不同的 `directory` 字段后 `a == b` 为 True），即本单没有、也不该移动任何
已发布数字：

| 量 | 数值 | 判据 | 判定 |
|---|---:|---|:--|
| 速度 L2 斜率（32/64/128，全 3D） | 1.9623452 | ≥ 1.8 | PASS |
| 速度 L2 斜率（面内 u,v） | 1.9623452 | ≥ 1.8 | PASS |
| 去均值压力 L2 斜率 | 1.8777243 | ≥ 1.8 | PASS |
| 三层拟合再识别阶 | 1.9194465 | ≥ 1.8 | PASS（佐证）|
| n=128 速度 L2 相对误差 | 2.262325e−04 | 解析解 | 0.0226% ≪ 3% |
| n=128 去均值压力 L2 相对误差 | 2.136311e−03 | 解析解 | 0.214% < 3% |

### 3.2 门前后对比（同一批文件，只差这一处改动）

| 门 | 改动前（基线 `cc0ba3b0`）| 改动后 |
|---|---|---|
| `tests/test_kovasznay_32_64_128_acceptance.py` | **22 passed, 10 skipped** | **23 passed, 9 skipped** |
| 其中 schema 新鲜度参数位 | 4 个（1 个是永远 skip 的不存在目录）| 3 个（= 实际提交的 summarizer 目录数）+ 1 条非空性控制 |

skip 数少一个，正是那条**永远不可能失败**的空目录参数被删除；新增的一条是同名控制的**真实**检查。

### 3.3 结果图

**本单无新图**，理由：本单没有产生任何新场量——被审计的对象是测试代码与工件绑定关系，
不是解；已提交的三网格图（`runs-32-64-128-zero_gradient/order.png`、`n128/fields-error.png`）
在本单前后**逐字节不变**。拿它们充当"本单结果图"会是伪造新证据，故不附。

## 4. 与 m20571 交付的关系（重开单的落点）

| m20571 交付件 | 当前 main 上的状态 | 本单动作 |
|---|---|---|
| `m20571-artifact-schema-drift-diagnosis.md` | 在库 | 不改（其 §4 门对比表在今日树上仍成立）|
| `report_payload_m20571.py` | 在库 | 不改 |
| `runs-32-64-128-zero_gradient/{RESULTS.md,summary.json}` | 在库、与 npz 一致 | 不改（§3.1 复算为证）|
| `summary-32-64/{RESULTS.md,summary.json}` | 在库、schema 同代 | 不改 |
| `n128/fields-error.png` | 在库 | 不改 |
| **该交付引入的 schema 门** | 在库，但**枚举式覆盖** | **本单改为自发现 + 非空性控制** |

## 5. 注意点与局限

1. **n=256 第四层仍未取得（honest WITHHELD）**。本单递交 `cj4739`（8 核，上限 7100 s）
   `scripts/verify_kovasznay_order.py --n 256 --symmetry-closure zero_gradient`，递交时排队位次
   138；到本单收尾时（递交后 20+ 分钟）仍在 `queued`、无任何产物，故本单在收尾时
   **主动 `kill` 该作业**（`cj4739` → killed），不占着 8 核让别人的排队变长；第四层需后续单重新递交。因此：
   `summary-32-64-128-256/` 不生成、四层阶（含标准误/GCI）一律不报，依赖它的
   `tests/test_kovasznay_256_ladder.py` 5 条如实 **skip**（`4 passed, 5 skipped`），
   not measured ≠ pass。这不是数值结论，是资源结论，不得引用四层任何数字。
2. **自发现判据不是"路径白名单"**：任何**新的**、内容上自称 summarizer 输出的
   `summary.json` 都会被按完整 schema 判——包括将来某个只想练手的临时目录。这是有意的，
   代价是临时目录必须真的同代，或者不提交。
3. **本门管"提交即同代"，不管"该不该提交"**：证据被提交过滤器滤除的场景由
   `tests/test_kovasznay_committed_artifacts_fresh.py` 与可复算门负责，两者互补，不可互相替代。
4. **三层 GCI 依然不可识别**（`f(h)=L+C h^p` 三点无残差自由度，外推极限为负），
   m20571 §5.4 的欠账本单**未偿还**，只记下：偿还它需要 n=256 实算落地。
5. 全仓测试未在本机全量跑（6509 条；本机策略只允许 <8 核、<10 min 短测试）。本单跑的是
   本算例相关门集（§6），跨算例影响面仅限一个测试文件的参数化方式。

## 6. 复现清单

```bash
# 1) 工件与 summarizer 同代（自发现）
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py -q
#    -> 23 passed, 9 skipped
# 2) 文档数字 vs 工件、四层文档可复现（无四层工件时如实 skip）
PYTHONPATH=src python -m pytest tests/test_kovasznay_doc_claims.py \
  tests/test_kovasznay_committed_artifacts_fresh.py tests/test_kovasznay_order_budget.py \
  tests/test_summarize_kovasznay_mixed_config_guard.py -q      # -> 17 passed, 2 skipped
# 3) 四层阶梯：未测即 skip
PYTHONPATH=src python -m pytest tests/test_kovasznay_256_ladder.py -q  # -> 4 passed, 5 skipped
# 4) 三网格 summary 可从已提交 npz 逐叶复现（§3.1）
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py \
  --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --output /tmp/repro3 --require-grids 32,64,128
```

## 7. 误差判定（逐项，显式回答 <3%）

本单的"误差"分两类，分开回答：

| 项 | 实测 | 基准 | 是否 <3% |
|---|---|---|---|
| n=128 速度 L2 相对误差 | 2.262325e−04 | Kovasznay 1948 解析解 | **是**（0.0226%）|
| n=128 去均值压力 L2 相对误差 | 2.136311e−03 | 同上 | **是**（0.214%）|
| 观测阶 vs 名义二阶 | 1.9623 vs 2 | 二阶 | **是**（相对偏差 1.9%）|
| 三层再识别阶 vs 名义二阶 | 1.9194 vs 2 | 二阶 | **是**（相对偏差 4.0%（对 2 而言）→ 见下注）|
| 重跑 summary 与库中 summary 的差 | 0（逐叶值相等）| 自身 | **是**（恒等）|
| 四层阶 / 其标准误 / GCI | 无 | 无 | **无法回答——未测**（§5.1）|

注：判据本身是**收敛阶 ≥ 1.8**，"越接近 2 越好"不是判据；表中"相对 2 的偏差"仅作可读性
参考，不构成阈值。真正的判据结论：斜率 1.9623 ≥ 1.8，**PASS**，余量 9.0%，且与两段单段阶
（1.9370 / 1.9877）自洽。