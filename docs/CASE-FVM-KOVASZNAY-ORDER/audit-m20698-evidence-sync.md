# CASE-FVM-KOVASZNAY-ORDER · 证据同步审计与修复（m20698）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证）
**判据**: 32/64/128 三网格全三维速度相对 L2 误差的 log-log 斜率 ≥ 1.8（名义二阶）
**基准**: Kovasznay 1948, *Proc. Camb. Phil. Soc.* 44:58–62, doi:10.1017/s0305004100023999
**席位 / 仓**: m20698 · TensorFVM · 分支 `agent/m20698` · 2026-11-14

## 0. 结论（disposition: request changes → 已修复，建议 ledger 转 verified）

锚点的**物理结论没有变**（斜率 1.9623，二阶成立），但本单开工时**验收门在主线是红的**，
且**负向对照的证据链在合流时被截断**。即：文档描述的"全绿"与合并后的树不自洽。
两处都已修复并加了防复发门。判定：**判据成立，证据链现已自洽且可复算**。

## 1. 发现（按严重度）

### D1（阻断·已修）committed `summary.json` 陈旧 → 验收门在主线红

`tests/test_kovasznay_32_64_128_acceptance.py::test_pair_orders_and_three_level_order_reproduce`
开工时 **FAILED**：

```
KeyError: 'pair_consistency'        tests/test_kovasznay_32_64_128_acceptance.py:225
```

根因：`scripts/summarize_kovasznay_32_64_128.py` 后来为三网格斜率增加了两个佐证块
（三网格 log-log 斜率无残差自由度，故需 `pair_consistency` 的逐段阶与
`three_level_estimates` 的 `f(h)=L+C h^p` 重辨识），但**入库的 `summary.json` 是旧版汇总器产物**，
既无 `pair_consistency` 也无 `three_level_estimates`。文档（`RESULT_32_64_128.md`、
`ORDER_32_64_128_STATUS.md`）却仍宣称验收门绿——**证据包过期，而不是求解器有问题**。

危害：任何"以文档为准"的读者都会得到错误结论；且验收门是唯一的守门人，门红即锚点实际不可交付。

### D2（阻断·已修）负向对照的证据与闸门在合流中丢失

- `runs-32-64-128-firstorder/` 在主线只剩 `n32`、`n128`，**`n64` 丢失**
  （该层在 `7bf86a2b`（分支 `agent/m20342`）中入库，合流时未带入）；
  而 `NEGATIVE_CONTROL.md`／`runs-32-64-128-firstorder/RESULTS.md` 仍引用 32/64/128 三段斜率
  1.2456、判别裕度 0.7167。
- 生成该对照的闸门 `scripts/kovasznay_order_sensitivity.py` 与其 pytest
  **同样不在主线**（只在未合分支上），即文档引用的可复算入口在主线不存在。

危害：负向对照（"该斜率不是对任何格式都报二阶"）在主线**不可复算**；缺层的表现是
`tests/test_kovasznay_order_sensitivity.py` **skip**（5 skipped），即缺陷以"诚实未测"的形式静默。

### D3（机制缺口·已补）没有"证据包与原始场同步"的门

D1、D2 都不是数值错误，而是**入库工件与其申明不符**且无门约束。这是可复发的一类缺陷。

## 2. 修复（最小改动）

| # | 改动 | 文件 |
|---|---|---|
| 1 | 用当前汇总器从 npz **重生成** `summary.json` / `RESULTS.md` / `order.png`（零手抄） | `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-zero_gradient/` |
| 2 | 从 `7bf86a2b` 恢复丢失的一阶层 **n=64** 全部原始工件（cases/fields/迭代史/trace） | `docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128-firstorder/n64/` |
| 3 | 从 `7bf86a2b` 取回负向对照闸门与其测试 | `scripts/kovasznay_order_sensitivity.py`、`tests/test_kovasznay_order_sensitivity.py` |
| 4 | 由闸门重生成对照工件（JSON/MD/PNG），数值与文档一致 | `docs/CASE-FVM-KOVASZNAY-ORDER/{order-sensitivity.json,NEGATIVE_CONTROL.md,order-sensitivity.png}` |
| 5 | 新增**证据同步门**：入库工件必须与当前脚本从 npz 现算结果**逐字节一致**，且"申明 32/64/128 就必须三层场都在" | `tests/test_kovasznay_committed_artifacts_fresh.py`（4 passed） |

## 3. 修复后实测（全部由脚本从 npz 现算）

**验收序列**（`zero_gradient`，Re=40，二阶对流，耦合）：

| n | h | vel L2（全 3D） | p L2（去均值） | max\|w\| | 收敛 |
|---:|---:|---:|---:|---:|:--|
| 32 | 0.031250 | 3.435615e−03 | 2.885142e−02 | 1.60e−16 | yes |
| 64 | 0.015625 | 8.972614e−04 | 7.958339e−03 | 2.49e−17 | yes |
| 128 | 0.007812 | 2.262325e−04 | 2.136311e−03 | 4.82e−17 | yes |

| 量 | 32→64 | 64→128 | 三段 LSQ 斜率 | R² | 判据 ≥1.8 |
|---|---:|---:|---:|---:|:--|
| 速度 L2（全 3D） | 1.93697 | 1.98772 | **1.9623** | 0.999944 | **PASS** |
| 压力 L2（去均值） | 1.85810 | 1.89735 | **1.8777** | 0.999964 | **PASS** |

**新增佐证（D1 修好后才可读）**：逐段阶 1.9370 / 1.9877，spread = 0.0508 ≤ 预声明带 0.20
（`both_above`=yes、`consistent`=yes）；三段 `f(h)=L+C h^p` 重辨识阶 1.9194（GCI 1.332），
与固定阶最小二乘的 1.9623 同侧相容；`verdict.corroborated = true`。

**负向对照**（唯一差别＝对流阶）：参考 1.9623 vs 一阶 upwind **1.2456**，判别裕度 **0.7167**（要求 ≥0.5）；
对照被 ≥1.8 门槛**拒绝**（≤1.5 上限亦满足）。六项闸门全 true。

## 4. 新门可证伪性（本单实做）

按"门必须能说不"的要求，两种真实失败态都注入验证过：

| 注入的失败态 | 结果 |
|---|---|
| 把入库 `summary.json` 换回旧的（无 `pair_consistency`） | `test_committed_summary_reproduces_byte_for_byte` **FAIL**；恢复后 PASS |
| 把一阶层 `n64/` 移走 | `test_control_series_has_every_level_it_reports` **FAIL**（而旧对照测试此时只是 skip）；恢复后 PASS |

即：新门把 D1/D2 这两类"文档与工件脱节"从**静默跳过／文档口径**变成**硬失败**。

## 5. 注意点 / 局限

1. **`directory` 字段记录传入路径**：汇总器把 `--runs` 原样写入 `summary.json`，故绝对路径会产生
   与入库版不同的字符串。同步门因此以仓库相对路径驱动脚本（与入库工件生产方式一致）。
   这是路径无关性上的小瑕疵，本单未改脚本（最小改动），已在此登记。
2. **本单未重跑任何求解**：全部为从入库 npz 的重算与工件再生；npz 与物理结论沿用既有运行
   （n=128 为集群 cj2598）。因此本单是"证据链修复 + 复核"，不是新的网格收敛研究。
3. **未变的诚实缺口**：`free` 展向闭合序列（`anchor`）仍只有 n=32，其 8 项断言继续 skip
   （honest not-measured）；该闭合不是本判据的适用口径（历史分析：全 3D 阶被 O(h) 伪展向模态压到 ~1.68）。
4. **既有无关红**：`tests/test_kovasznay_order_m17078.py::test_summary_exists` 因
   `runs/kovasznay-order-m17078/summary.json` 不在树内而失败——该路径在 HEAD 上同样缺失，
   与本单改动无关（历史 m17078 工件从未入库），本单未动它，如实登记为 unknown/未修。
5. **压力固定规范的阶（0.9772）不是精度量度**（动量方程只定压力梯度），沿用既有结论。

## 6. 检查项（本轮实际执行）

```
env PYTHONPATH=src python -m pytest \
  tests/test_kovasznay_32_64_128_acceptance.py \
  tests/test_kovasznay_order_sensitivity.py \
  tests/test_summarize_kovasznay_mixed_config_guard.py -q
# → 29 passed, 9 skipped   （修复前：1 failed, 17 passed, 9 skipped）

env PYTHONPATH=src python -m pytest tests/test_kovasznay_committed_artifacts_fresh.py -q
# → 4 passed

env PYTHONPATH=src python -m pytest \
  tests/test_kovasznay_order_asymptotic.py tests/test_kovasznay_order_m17078.py \
  tests/test_kovasznay_spanwise.py -q
# → 16 passed, 9 skipped, 1 failed（＝§5.4 既有无关红）
```

## 7. 误差判定（对解析基准逐项，显式回答）

| 关键量 | n=128 相对 L2 误差 | 三网格斜率 | 是否 ≥1.8 | 最细网格是否 <3% |
|---|---:|---:|:--:|:--:|
| 速度（全 3D） | 2.262e−04（0.023%） | 1.9623 | **是** | **是** |
| 压力（去均值） | 2.136e−03（0.214%） | 1.8777 | **是** | **是** |

## 8. 复现

```bash
D=docs/CASE-FVM-KOVASZNAY-ORDER
env PYTHONPATH=src python scripts/summarize_kovasznay_32_64_128.py --runs $D/runs-32-64-128-zero_gradient \
  --output $D/runs-32-64-128-zero_gradient --markdown $D/runs-32-64-128-zero_gradient/RESULTS.md \
  --figure $D/runs-32-64-128-zero_gradient/order.png --require-grids 32,64,128
env PYTHONPATH=src python scripts/kovasznay_order_sensitivity.py \
  --reference $D/runs-32-64-128-zero_gradient --control $D/runs-32-64-128-firstorder \
  --output $D/order-sensitivity.json --markdown $D/NEGATIVE_CONTROL.md --figure $D/order-sensitivity.png
env PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py \
  tests/test_kovasznay_order_sensitivity.py tests/test_kovasznay_committed_artifacts_fresh.py -q
```

## 9. 给下次自找活的线索

- 合流（merge-master）会**静默丢文件**：D2 的 n64 与两个脚本都是这样丢的。建议把这套
  "证据同步门"模式推广到其他算例池锚点（`selftest-4-8-16`、`runs-32-64-128` 等同类目录）
  ——新条目建议：`MECH-EVIDENCE-SYNC-GATE`（域=TensorFVM/验证域，
  验收：≥3 个算例池的入库 summary 均可从 npz 逐字节复算，且缺层即硬失败）。
- `test_kovasznay_order_m17078.py::test_summary_exists` 是长期无人清理的僵尸断言，
  宜单独一单改为 skip-if-absent 或补工件。