# CASE-FVM-KOVASZNAY-ORDER — canonical runner 支持对称闭包 + 32/64 判据管线闭合（honest partial）

**日期**: 2026-11-13 · **席**: m17913 (tp-infra-2 infrastructure) · **仓**: TensorFVM
**锚点**: `CASE-FVM-KOVASZNAY-ORDER` —「L2 收敛阶 >= 1.8 on 32/64/128」
**基准**: Kovasznay (1948) Proc. Cambridge Philos. Soc. 44:58–62, doi:10.1017/s0305004100023999

## 1. 问题（开工时对表得到的缺口）

锚点要求 32/64/128 **三**套网格的 log-log 斜率。仓内现状（开工核对）：

| 项 | 状态 |
|---|---|
| `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n32,n64` | 有 `cases.json` + `n<N>-fields.npz`，闭包 `zero_gradient`，w 机器零 |
| 同目录 `n128/` | 只有 `environment.json` / `sources.json`，**无结果** |
| canonical runner `scripts/verify_kovasznay_order.py` | **不支持** `zero_gradient` 闭包（库侧 `kovasznay_solver(symmetry_closure=...)` 已有，脚本未暴露） |
| 证据门 `tests/test_kovasznay_32_64_128_acceptance.py` | 只有 `anchor`(free)/`selftest` 两集，**没有** zero-gradient 判据集 |

即：闭包修复版的历史数字**无法由 canonical runner 复现**（它们来自另一棵树的脚本），
因此无法当成"三网格判据"的证据链。本单补的正是这条链的**可复现性**。

## 2. 改了什么（最小、可验证）

1. `scripts/verify_kovasznay_order.py`（canonical 单网格 runner）
   - 新增 `--symmetry-closure {free,zero_gradient}`（默认 `free`，向后兼容历史调用）；
   - 闭包传入 `kovasznay_solver(..., symmetry_closure=...)`，并**写进 `cases.json`**，
     使混合闭包的网格序列不可能被无意汇总；
   - `sampling_bias()` 同步使用同一闭包；
   - 模块 docstring 记录两种闭包的物理差异（traction-free vs 真实对称条件）与后果。
2. `tests/test_kovasznay_32_64_128_acceptance.py`
   - 新增证据集 `anchor_zg`（`runs-32-64-128-zero_gradient`, grids 32/64/128），
     自动继承既有的「从原始场量重算」反手抄门、斜率一致门、verdict 反夸大 门;
   - 新增闭包一致性/机器零门：三级 `symmetry_closure` 必须全为 `zero_gradient`，
     `max|w| ≤ 1e-12` 且展向份额 ≤ 1e-12；
   - 新增全 3D 速度判据门（`velocity_full` 斜率 ≥ 1.8），不再需要"面内预算"辩护。

## 3. 验证

### 3.1 补丁复现性（本机 x86，n=32，零手抄对照）

用**打过补丁的 canonical runner** 重跑 n=32（`--symmetry-closure zero_gradient`），
与仓内已提交的 `runs-32-64-128-zero_gradient/n32/cases.json` 对照：

| 量 | 已提交 | 本次重跑 | 相对差 |
|---|---:|---:|---:|
| velocity_l2（全 3D） | 3.4356154111469706e−03 | 3.4356154111469706e−03 | 0 |
| pressure_l2（固定规范） | 7.843493416161752e−02 | 7.843493416161752e−02 | 0 |
| pressure_mean_free_l2 | 2.8851421855200135e−02 | 2.8851421855200135e−02 | 0 |
| max\|w\|（伪展向） | 1.6022438471154331e−16 | 1.6022438471154331e−16 | 0 |
| outer_iterations | 22 | 22 | — |

逐位一致（0 相对差），说明：补丁语义与产生该证据的脚本一致、求解路径确定性、
且跨架构（原证据为 loongarch64 集群，本次 x86 控制面）可复现。

### 3.2 32/64 两网格斜率（由 `n<N>-fields.npz` 重算，`summary-32-64/`）

由 `scripts/summarize_kovasznay_32_64_128.py --require-grids 32,64` 生成，
每个误差都在汇总时从原始场量重算并与 `cases.json` 对账（rtol 1e-10）：

| n | h | cells | vel L2 (全 3D) | vel L2 (面内) | w 份额 | p L2 (去均值) | 收敛 |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 32 | 0.03125 | 1024 | 3.435615e−03 | 3.435615e−03 | 6.779e−18 | 2.885142e−02 | yes |
| 64 | 0.01562 | 4096 | 8.972614e−04 | 8.972614e−04 | 1.119e−18 | 7.958339e−03 | yes |

| 量 | 斜率 | R² | 判定 (≥1.8) |
|---|---:|---:|:--|
| 速度 L2（全 3D） | **1.9370** | 1.000000 | PASS |
| 速度 L2（面内 u,v） | 1.9370 | 1.000000 | PASS |
| 压力 L2（去均值） | 1.8581 | 1.000000 | PASS |
| 压力 L2（固定规范） | 0.9857 | 1.000000 | FAIL（规范偏置，非格式精度量度） |

图：`runs-32-64-128-zero_gradient/summary-32-64/convergence.png`
（log-log 收敛曲线 + 误差预算 + n=64 速度云图 + 中心线剖面，全部由数据生成）。

## 4. 判据判定（honest partial）

| 判据 | 目标 | 实测 | 判定 |
|---|---:|---:|:--|
| 全 3D 速度 L2 斜率 **on 32/64/128** | ≥ 1.8 | **未测得**（n=128 无结果） | **WITHHELD** |
| 全 3D 速度 L2 斜率 on 32/64（可得的最细两网格） | ≥ 1.8 | 1.9370（R²=1.000000） | PASS（非锚点口径） |
| 面内速度 L2 斜率 on 32/64 | ≥ 1.8 | 1.9370 | PASS（非锚点口径） |
| 压力（去均值）斜率 on 32/64 | ≥ 1.8 | 1.8581 | PASS（非锚点口径） |
| 闭包一致 + 伪展向机器零（32/64） | 全 `zero_gradient`，max\|w\|≤1e−12 | 全 yes，1.602e−16 | PASS |
| `anchor_zg` 证据门（32/64/128） | — | **SKIP**（n=128 缺失，honest "not measured yet"） | 不通过也不冒充通过 |

**结论**：`feature_ledger` 该锚**保持 open**。本单不声明锚点达标；它声明的是
「判据管线现在可由 canonical runner 从零复现，且 32/64 两网格在物理闭包下已回到二阶区」。

**是否 <3%？** 本单判据是**收敛阶（斜率）**，不是精度阈值；斜率三网格值未测得，
故不能回答"是否 ≥1.8"。可回答的是 32/64 段：1.9370 ≥ 1.8（超出 7.6%）。

## 5. 注意点 / 局限

1. **n=128 未完成（本单最大缺口）**：本机 `load15` 开工 121、中途 319（nproc=128，他人作业多线程占满），
   nice 19 单线程下 n=128 在 ~7 分钟内仍未越过首次 Picard 迭代（初装 LU 占主导）；
   按容量纪律不再追加本机重算。集群侧 `cj1707`(n128)/`cj1708`(n64) 递交时**队列积压 288 单**，
   截至收尾仍 `queued`，与前单记录的调度器积压一致 —— 这是平台侧待修项，租户侧无法修复。
   → 接手方式：队列恢复后 `fetch cj1707`，把 `runs/kovasznay-zg/n128/` 三件套
   （`cases.json` / `n128-fields.npz` / `n128-*.json`）放到
   `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n128/`，
   再跑一次 `summarize_kovasznay_32_64_128.py --require-grids 32,64,128`，
   `anchor_zg` 证据门会自动从 SKIP 变成实跑（绿/诚实红）。
2. **`n128/` 残留的 `environment.json`/`sources.json`** 是前单未完成运行留下的，本单未删（不动他人证据），
   但也**不作判据**。
3. **口径**：`velocity_spanwise` 是"伪展向 RMS / 解析全速度范数"的**份额**（解析 w≡0，相对 L2 无定义）；
   zero-gradient 闭包下它降到 ~1e−18，故其 log-log 斜率没有物理意义，不得当判据。
4. **压力固定规范**斜率 0.9857 不是格式退阶：动量方程只定压力梯度，固定规范含随网格变化的常数偏置；
   有阶语义的是去均值规范。

## 6. 复现

```bash
# 单网格（本单新增闭包开关）
PYTHONPATH=src python scripts/verify_kovasznay_order.py --n 32 \
  --symmetry-closure zero_gradient --output runs/kovasznay-zg/n32

# 两网格汇总（应在 3 秒内完成，从 npz 重算全部误差）
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py \
  --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/summary-32-64 \
  --require-grids 32,64 --markdown .../RESULTS.md --figure .../convergence.png

# 证据门（n=128 缺失时 anchor/anchor_zg 如实 SKIP）
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py -q
```

## 7. 交付物

- `scripts/verify_kovasznay_order.py` — 新增 `--symmetry-closure`（改动 20 行）
- `tests/test_kovasznay_32_64_128_acceptance.py` — 新增 `anchor_zg` 集 + 闭包/机器零门 + 全 3D 速度判据门
- `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/summary-32-64/`
  — `summary.json`, `RESULTS.md`, `convergence.png`（程序化生成）
- 本文件