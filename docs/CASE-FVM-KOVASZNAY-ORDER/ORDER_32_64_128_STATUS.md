> **已结案（2026-11-14，m20033）**：n=128 层级已补齐，三网格 32/64/128 速度 L2 斜率
> **1.9623**（R²=0.999944）≥ 1.8，判据达成。下方 m18510 的"n=128 未落盘"状态已被
> `closeout-32-64-128.md` 取代；当时阻塞它的证据目录语义缺陷亦已修复
> （`NativeRunEvidence`，回归测试 `tests/test_native_run_evidence_resume.py`）。

# CASE-FVM-KOVASZNAY-ORDER — 32/64/128 收敛阶：混合配置门 + 既有 PASS 声明的诚实降级

**日期**: 2026-11-13 · **席**: m18510（selfstart, tp-infra-2）· **仓**: TensorFVM
**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证，ledger 状态 `open`）
**判据（ledger 原文）**: L2 收敛阶 ≥ 1.8（名义二阶），**以 32/64/128 三套网格上 log-log 最小二乘斜率计量**
**外部基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/s0305004100023999

---

## 1. 本单做了什么（与不做什么）

本单是**验证欠账清理**类：不重跑已闭合的算例，而是对既有结案声明做独立复核（adversarial review），
并把复核中发现的**一类真实失效模式**固化成机器门。

三件事：

1. **混合配置门（代码，本单新增）**：`scripts/summarize_kovasznay_32_64_128.py` 新增
   `validate_levels()`。此前该脚本只校验"层级齐全 + cases.json 可由 npz 复算"，
   **不校验各层是否用同一套离散配置**。而本仓同时存在两种挤出面闭合口径：
   `free`（历史牵引自由，留下 O(h) 伪展向速度）与 `zero_gradient`（物理对称，伪模态降到舍入级），
   两者伪展向预算相差 **13 个数量级**（n=64：2.49e−17 vs 9.41e−03）。若某层被就地重跑、
   序列里混进另一种闭合，最小二乘斜率仍会算出一个"好看的数字"，但它**不是任何格式的阶**。
   现在：`reynolds / symmetry_closure / convection_order / algorithm` 四项在序列内必须一致，
   不一致即 `SystemExit` 并打印冲突项与层级；一致则把该配置写进 `summary.json` 的
   `provenance.configuration`（可审计）。
2. **既有 PASS 声明的复核与降级（文档，本单结论）**：`docs/CASE-FVM-KOVASZNAY-ORDER/closure_report.md`
   记录的"✅ 通过"是在 **16/32/64** 上取得的（面内速度最小二乘斜率 1.808）。
   ledger 判据写的是 **32/64/128**，两者不是同一组证据；且 16/32/64 中 n=16 已在渐近区之外。
   本单据此判定该结案声明**对判据是高报（overstated）**，ledger 保持 `open` 是正确状态。
3. **跨文档数字不一致的复算确认（文档，本单结论）**：见 §4。

## 2. 支持证据（全部可从工作树内 npz 现场复算）

复算命令（本单实跑）：

```bash
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py \
  --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --output <out> --require-grids 32,64
```

| n | h | cells | vel L2 (full 3D) | vel L2 (u,v) | max\|w\| | p L2 (mean-free) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 3.125e−02 | 1024 | 3.435615e−03 | 3.435615e−03 | 1.60e−16 | 2.885142e−02 |
| 64 | 1.562e−02 | 4096 | 8.972614e−04 | 8.972614e−04 | 2.49e−17 | 7.958339e−03 |

`zero_gradient` 闭合，Re=40：**32→64 单段观测阶 = 1.93697**（全 3D 与面内一致，因伪展向模态已在舍入级），
去均值压力阶 1.8581，R²=1.000000。

**这仍然不是判据要的三网格斜率。** n=128 尚未落盘，故本单**不声明达标**，ledger 维持 `open`。

## 3. 未闭合部分与在飞作业

| 作业 | 内容 | 状态（本单收尾时） |
|---|---|---|
| cj1944 | `zero_gradient` n=128 → `runs-32-64-128-zero_gradient/n128` | 集群排队 |
| cj1945 | `free` n=64 → `runs-32-64-128/n64` | 集群排队 |
| cj1946 | `free` n=128 → `runs-32-64-128/n128` | 集群排队 |

控制面本机 load15 ≈ 118（nproc=128），按容量纪律未在本机起 n=128 重算；
三单全部递交集群（nodes 健康：n315/n330 均 ok，各 128 核）。

工件取回后闭合方式（两步，无需人工转录）：

```bash
# 1) 取回 cj1944 的 docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/n128/ 到工作树
# 2) 重算三网格斜率（门会自动变 green 或给出 honest FAIL）
PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py \
  --runs docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient \
  --markdown docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/RESULTS.md \
  --figure docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/order.png \
  --require-grids 32,64,128
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py -q
```

现门状态：`7 passed, 17 skipped`——skip 全部是"该层级不存在"，是诚实的 not-measured，不是 pass。

## 4. 注意点（复核发现的坑）

1. **16/32/64 ≠ 32/64/128**（见 §1.2）。历史结案报告的面内 1.808 与本次 32→64 的 1.937
   不是同一证据链，不可互相替代。
2. **`docs/case-kovasznay-order-order-verification/RESULTS_INPLANE.md` 的 n=32 面内列不可复算**：
   该文档记 3.72450e−03，而 `runs-32-64-128/n32/n32-fields.npz`（`free` 闭合）重算得
   **3.431614e−03**；同表的速度 L2 7.812177e−02/3.839693e−03 等值则与 npz 一致。
   即该表的"面内列"来自另一口径或另一套 run，**不能作为判据**。凡不能从 npz 复算的历史数字，
   本单一律不用（这与 m17163 的 `runs-32-64-128/README.md` §4.3 的发现一致，本单独立复现）。
3. **固定规范压力阶只有 0.986**：动量方程只定压力梯度，固定规范下压力含随网格变化的常数偏差，
   该阶不是格式精度的量度；有物理意义的是去均值规范（1.8581）。
4. **`free` 与 `zero_gradient` 不可混用**（§1.1）：本次新增门已覆盖，混合序列会硬报错。

## 5. 复现清单

| 工件 | 路径 |
|---|---|
| 混合配置门 | `scripts/summarize_kovasznay_32_64_128.py::validate_levels` |
| 门的单元测试 | `tests/test_summarize_kovasznay_mixed_config_guard.py`（6 passed） |
| 验收门（现 7 passed / 17 skipped） | `tests/test_kovasznay_32_64_128_acceptance.py` |
| 32/64 原始场与 summary | `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/{n32,n64,summary-32-64}` |