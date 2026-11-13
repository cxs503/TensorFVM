# CASE-FVM-KOVASZNAY-ORDER — 32/64/128 收敛阶验证：机器可复算的证据管线（进行中，honest partial）

**日期**: 2026-11-13 · **席**: m17163 (tp-merge-2 generalist) · **仓**: TensorFVM
**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证）
**判据**: L2 收敛阶 ≥ 1.8（名义二阶），以 32/64/128 三套网格上 log-log 最小二乘斜率计量
**外部基准**: Kovasznay, L. I. G. (1948), *Laminar flow behind a two-dimensional grid*,
Proc. Cambridge Philos. Soc. 44:58–62, doi:10.1017/S0305004100023999

---

## 1. 为什么还要做（本单的增量在哪里）

m16173 板审已把该锚的 L4 判定卡在"证据可核性"上：

> `cases.json`, `summary.json`, figures **REFERENCED in report but MISSING**；
> 报告里的 96×96 误差数字"cannot be independently verified because cases.json is missing"。

即：不是数值结果不够好，而是**报告与原始数据之间没有一条机器可复算的链路**。本单补的就是这条链路，
并把判据从"报告正文里的表格"变成"读取原始场量、现场重算、再判"的可执行门。

## 2. 怎么做的

| 环节 | 工件 | 作用 |
|---|---|---|
| 求解 | `scripts/verify_kovasznay_order.py --n N` | 单网格一次求解（对称挤出 n×n×1），落 `cases.json` + `n<N>-fields.npz`（单元中心、体积、u、p、解析场、通量） |
| 判据汇总 | **`scripts/summarize_kovasznay_32_64_128.py`（本单新增）** | 从 `n<N>-fields.npz` **重算**速度/压力相对 L2 误差；与 `cases.json` 不符即硬报错（rtol=1e-10）→ 杜绝手抄；输出配对观测阶、最小二乘斜率+R²、`summary.json`、`RESULTS.md`、收敛/场量图 |
| 机器门 | **`tests/test_kovasznay_32_64_128_acceptance.py`（本单新增）** | 读已提交工件 → 重算误差 → 断言斜率与 verdict 标志，防止"手改 cases.json 让门变绿" |

误差分解口径（比"速度 L2"单值更可解释）：

- `velocity_full`：三分量速度相对 L2（相对解析速度范数）；
- `velocity_inplane`：仅 (u,v) 分量，解析解本身是二维的，这是**离散格式真实阶**的载体；
- `velocity_spanwise`：解析 w≡0，故相对范数无定义，改用 `sqrt(Σ V w² / Σ V |u_exact|²)`
  —— 即伪展向模态在总速度误差中的**份额**；
- 压力：固定规范与去均值规范两种（动量方程只定压力梯度，去均值才是有物理意义的比较量）。

## 3. 计算结果

### 3.1 n=32 级（本机完成，已提交 `runs-32-64-128/n32/`）

Re=40，κ=2π，λ=−0.963741，对称挤出 32×32×1=1024 单元，二阶对流，`coupled` + `reused_lu`，
Picard 松弛 0.7，tol=1e−8，outer 22 步收敛。

| 量 | 值 |
|---|---:|
| 速度相对 L2（全 3D） | 3.839693e−03 |
| 压力相对 L2（固定规范） | 7.812177e−02 |
| 压力相对 L2（去均值） | 2.895337e−02 |
| 最大离散散度 | 1.995e−10 |
| 全局动量不平衡 | 2.575e−10 |
| max\|w\|（伪展向） | 4.526e−03 |
| 解算耗时 | 101.8 s（1 线程，nice 19） |
| 采样偏差（点到单元平均） | 速度 6.55e−04 / 压力 1.22e−04 |

**交叉复现证据**：
- 与 `docs/case-kovasznay-order-order-verification/RESULTS_INPLANE.md` 的 n=32 记录相比：该文档速度 L2 = **3.83969e−03**，本次独立重跑 = **3.839693e−03**（相对差 1e−6 量级）；
- 与 `docs/verification-kovasznay-asymptotic/runs/level-n32-zero_gradient-reused_lu.json` 相比：该记录未存速度字段，压力(固定规范) L2 = 7.843493e−02，本次 = 7.812177e−02（相对差 4.0e−03）。压力差来自压力规范的冻结方式差异，量级一致。
→ 求解路径可复现，差异在可解释范围内。

### 3.2 4/8/16 管线自检（已提交 `selftest-4-8-16/`，用于证明判据管线本身可用）

| n | h | 单元 | 速度 L2(全) | 速度 L2(u,v) | w 份额 | 压力 L2(去均值) | 收敛 |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 4 | 0.25000 | 16 | 1.029423e−01 | 1.016212e−01 | 1.644e−02 | 4.177775e−01 | yes |
| 8 | 0.12500 | 64 | 3.710529e−02 | 3.620366e−02 | 8.130e−03 | 2.678226e−01 | yes |
| 16 | 0.06250 | 256 | 1.257478e−02 | 1.201574e−02 | 3.708e−03 | 9.732922e−02 | yes |

最小二乘斜率：全 3D 1.5166（R²=0.99971）、面内 1.5401、展向份额 1.0743、压力(去均值) 1.0509。
**这是前渐近区的诚实 FAIL**（4/8/16 远未进入二阶区，与既有 n=8 前渐近结论一致）；本组数字的用途是
验证管线（重算校验、斜率、图、markdown 全部由数据生成），**不作为锚点判据**。

### 3.3 32/64/128 判据：**未闭合（honest partial）**

n=64 本机在跑到第 4 个 outer 迭代时（约 2 min/迭代），其斜率尚不可得；n=128 完全未开展。
原因见 §4。**因此本单不声明锚点达标，`feature_ledger` 该锚应保持 open。**

## 4. 注意点（踩坑与算力现状——本单最大局限）

1. **算力被压满**：本机 `load15` 从 114 升到 206→374（nproc=128），单次 n=64 Picard 迭代从
   n=32 的 ~4.6 s 退化到 ~120 s。按此吞吐，64/128 三网格研究在本机不可行。
2. **集群队列堵塞**：集群作业 `cj1428`（本单递交，`stage_dir=本工作树`，`PYTHONPATH=$PWD/src`
   显式钉住本树源码）状态长期 `queued`，同期队列里 30+ 作业从 00:00 起一直未派发，而节点
   n315/n330 报 `load 38/37`、内存 946/980 GB 空闲 —— 调度器侧未派发，非资源不足。**这是平台侧
   待修的基础设施项**，本单无法从租户侧修复（不动 TensorPlatform）。
3. **editable 安装指向别的 worktree**：本机 `python -c "import tensorfvm"` 解析到
   `/root/tp-agent/work/m14716/src/tensorfvm`（他人分支）。所有本单运行均显式
   `PYTHONPATH=$PWD/src`，并在 `environment.json` 记录；集群作业同样显式钉住，避免"用别人的代码
   给自己的结果背书"。
4. **误差分解口径必须写清**：直接拿"全 3D 速度 L2"判阶会被 O(h) 伪展向模态拖到 ~1.68（既有
   16/32/64 结论），这是**离散化载体的性质**而不是格式退阶；判定必须同时给出面内阶与展向份额。
5. **压力规范**：固定规范的压力 L2 含一个未定的常数偏置，n≤16 时其斜率甚至 <1（4/8/16 自检：
   0.50），只有去均值规范才有阶的语义。

## 5. 误差判定（对判据逐条）

| 判据 | 阈值 | 实测 | 判定 |
|---|---:|---:|:--|
| 速度 L2 面内 log-log 斜率（32/64/128） | ≥1.8 | **未测得**（n=64 在跑、n=128 未跑） | **WITHHELD** |
| 压力 L2 去均值 log-log 斜率（32/64/128） | ≥1.8 | **未测得** | **WITHHELD** |
| 三网格全部数值收敛 | 全真 | n=32 真；64/128 未知 | **WITHHELD** |
| 单网格解的可复现性（n=32） | 相对差 ≪1% | 1e−6 量级（对比 RESULTS_INPLANE 3.83969e−03） | **PASS** |
| 判据管线可机器复算（4/8/16 自检） | 重算=cases.json | 通过（rtol 1e-10） | **PASS** |

**是否 <3%？** 本单不产生"是否 <3%"的判定——<3% 是对 96×96 级细网格的精度门槛，属另一条判据；
本单的判据是**收敛阶**（斜率），须显式声明："斜率未测得，故无法回答是否 ≥1.8"。

## 6. 交接（下一步，按成本排序）

1. `run_cluster --action ps --job_id cj1428` 看队列是否恢复；恢复后作业会依次跑 n=32/64/128
   （`stage_dir` 已钉本树），产出 `runs/kova-order/n{32,64,128}/`。
2. `python scripts/summarize_kovasznay_32_64_128.py --runs runs/kova-order --output <dir>
   --markdown docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128/RESULTS.md --figure .../convergence.png`
   → 得到三网格斜率与 PASS/FAIL。
3. `pytest tests/test_kovasznay_32_64_128_acceptance.py`（工件齐备后自动从 skip 变为实跑）。
4. 若 n=128 的 `coupled/reused_lu` 内存不可行，退路：三网格统一改用 `simplec` 分离式算法
   （保持 tol=1e−8、同一格式），并在报告中标注算法变更；**不可混用不同算法凑斜率**。

## 7. 交付物

- `scripts/summarize_kovasznay_32_64_128.py` — 重算校验 + 斜率 + 图/表/summary（新增）
- `tests/test_kovasznay_32_64_128_acceptance.py` — 证据绑定判据门（新增）
- `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128/n32/` — `cases.json`, `summary.json`, `n32-fields.npz`, `environment.json`
- `docs/CASE-FVM-KOVASZNAY-ORDER/selftest-4-8-16/` — 管线自检 `summary.json`, `RESULTS.md`, `conv.png`
- 本文件