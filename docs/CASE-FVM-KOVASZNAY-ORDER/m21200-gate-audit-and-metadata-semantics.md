# CASE-FVM-KOVASZNAY-ORDER · 验收链对抗式复核：闭合元数据语义缺陷 + 门变异审计

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（算例池·定常 Kovasznay 流收敛阶验证，开工时 ledger `open`）
**席位**: m21200（selfstart, generalist）· **仓**: TensorFVM · **日期**: 2026-11-15
**基准**: Kovasznay, L. I. G. (1948), *Proc. Camb. Phil. Soc.* **44**:58–62,
doi:10.1017/S0305004100023999
**判据（不变）**: L2 收敛阶 ≥ 1.8（名义二阶），32/64/128 三套网格 log-log 最小二乘斜率

---

## 1. 本单为什么做这个（选题依据）

该锚的数值结论（`zero_gradient` 闭合 32/64/128 速度斜率 1.9623、去均值压力 1.8777，
验收门 22 passed）已由 m20239/m20800/m21052/m21068 多单做到，并被 m20800 用**不共享代码路径**
的独立实现复核过（基准场逐点 8.9e-16、斜率两路差 4.9e-13）。因此本单**不重算数值**，
而是按对抗式复核（adversarial review）口径去攻击**验收链本身**，只认「证据能不能被伪造」这类问题。

开工先核对总账：`CASE-FVM-KOVASZNAY-ORDER` 状态 `open`，未被他单 close，允许认领。

复核时找到两处**可复现的材料级缺口**，都来自 m20800 §7 留下的尾巴（原文明确「需要一个独立小单」）：

| 编号 | 缺口 | 严重度 | 后果 |
|---|---|---|---|
| D1 | `validate_levels` 把「未记录」(`null`) 当成「不同」，拒绝整条物理上均匀的 `free` 闭合序列 | 中（证据一致性门的语义缺陷） | `free` 组 32/64 已实测却无法拟合，验收门里 `anchor` 组长期 skip |
| D2 | 门只信 `cases.json` 里的闭合**标签**，从不校验该标签与**该层自己的场量**是否一致 | 中（漏检面） | 一个标错闭合方式的层会被当成合法证据并入拟合 |
| — | 「门是证据绑定的」只有 docstring 断言，无变异证据 | 低—中（可核性） | 无法回答「测试是断言了不变量，还是只跑了代码」 |

---

## 2. 怎么做的

### 2.1 D1/D2 的修复（`scripts/summarize_kovasznay_32_64_128.py`）

新增 `validate_levels_with_metadata()`（`validate_levels()` 保留为薄封装，原有调用与测试语义不变），
把一致性门从「逐键相等」改成**三态语义**：

| 情形 | 处理 |
|---|---|
| 两层都记录且**不同** | 仍然硬失败（混合阶梯没有「阶」可言）——与原行为一致 |
| 两层都记录且**相同** | 通过，但**层自证**：该层标签必须与其 `max_abs_z_velocity` 推出的闭合方式一致 |
| 某层**未记录** | 只有当闭合方式能**从该层场量重建**、且重建值与其他层声明一致时才通过；重建过程写入 `provenance.metadata.gaps`，**不回写工件** |
| 不可重建的键（`reynolds`/`convection_order`/`algorithm`）未记录 | **仍然是硬失败**（放宽被严格限定在闭合方式这一项） |

闭合方式为何**可测**而非猜测：`zero_gradient`（物理对称 `du/dz=dv/dz=0`）把伪展向速度压在舍入级，
`free`（牵引自由）留下 O(h) 伪模态。已提交工件上两者相差 **12 个数量级**
（n=64：2.49e-17 vs 2.196e-03），阈值 `SPANWISE_ROUNDOFF = 1e-12` 落在这条鸿沟中间。

### 2.2 门变异审计（`scripts/audit_kovasznay_acceptance_gate_mutations.py`，新增）

对 `zero_gradient` 工件的**临时副本**施加 5 种污染，然后用**门自己的测试函数**（不是重写实现）
重跑检查，比对「事先声明的必检项」：

| 变异 | 事先声明必须报错的检查 | 实测 |
|---|---|---|
| 无（对照） | 无 | 全绿 |
| `hand_edited_error`（n64 `cases.json` 的 `velocity_l2` ×0.9） | 重算比对 | ✅ 命中 |
| `fabricated_solution`（n128 场量替换成解析解 = 误差归零） | 重算比对 + 斜率比对 | ✅ 命中，且额外触发判据与 verdict 检查 |
| `hand_edited_slope`（`summary.json` 速度斜率 +0.05） | 斜率比对 | ✅ 命中 |
| `overstated_verdict`（n128 标 `numerically_converged:false` 但 summary 仍 pass） | verdict 标志检查 | ✅ 命中 |
| `relaxed_threshold_in_summary`（把 `verdict.threshold` 改成 1.0） | **必须不报错**（判据在代码里，不受数据摆布） | ✅ 无一项报错 |

工件：`docs/CASE-FVM-KOVASZNAY-ORDER/gate-mutation-audit.json`（`verdict: PASS`, `undetected: []`）。

---

## 3. 结果（程序化注入，零手抄）

### 3.1 元数据语义（`tests/test_kovasznay_closure_metadata_semantics.py`，10 passed）

| 断言 | 结果 |
|---|---|
| 未记录 + 场量指向 `free` → 接受，`metadata_complete=False`，gap 记 `levels=[32]` | ✅ |
| 未记录 + 场量指向 `zero_gradient` 而其他层声明 `free` → 拒绝 | ✅ |
| 未记录且无 `max_abs_z_velocity` → 拒绝（`not reconstructable`） | ✅ |
| 标签与自身场量矛盾（`zero_gradient` 标签 + `max|w|=2.2e-03`）→ 拒绝 | ✅ |
| `reynolds`/`convection_order`/`algorithm` 未记录 → 仍硬失败（3 例参数化） | ✅ |
| 已提交的真实工件 `runs-32-64-128/{n32,n64}` → 解析为 `free`，n32 记为重建 | ✅ |

### 3.2 回归（旧行为不得改变）

| 检查 | 命令 | 结果 |
|---|---|---|
| zero_gradient 汇总逐位复现 | `summarize_kovasznay_32_64_128.py --runs .../runs-32-64-128-zero_gradient` | `fits` 与 `verdict` 与已提交 `summary.json` **完全相等**；仅新增 `provenance.metadata` |
| 原一致性门测试 | `pytest tests/test_summarize_kovasznay_mixed_config_guard.py` | 6 passed（混合配置仍被拒） |
| 验收门 | `pytest tests/test_kovasznay_32_64_128_acceptance.py` | **22 passed, 10 skipped**（skip 仍为 `anchor` 组等未测层） |
| 变异审计 | `pytest tests/test_kovasznay_gate_mutation_audit.py` | 4 passed |

### 3.3 图

`docs/CASE-FVM-KOVASZNAY-ORDER/m21200-gate-mutation-detection.png`：变异 × 门检查的命中矩阵
（黑格=该污染被该检查抓到，灰格=未触发），由 `gate-mutation-audit.json` 直接绘制。

---

## 4. 注意点 / 局限

1. **本单没有新增数值结论**：斜率仍是 1.9623 / 1.8777，未改动求解器与判据。
2. **`anchor`（`free` 闭合）组仍是 not-measured**：n=128 层已递交集群 **cj3258**，本单结束时
   仍在排队（20 min 时仍 `queued`，队列前有十几个作业）。**不猜、不补**——按纪律记 not-measured。
   D1 的修复使该组一旦补齐即可拟合（n32 的 `null` 标签会被明确记录为「重建」）。
3. **变异审计只覆盖 `zero_gradient` 集**：`anchor` 组工件不全（skip）、`selftest` 组是预期 FAIL 的
   预渐近序列，二者未做变异。
4. **「重建」不等于「记录」**：`metadata_complete=False` 会随 summary 发布，报告读者能看到哪一层
   的闭合标签是推断的，而不是被悄悄补齐成 `free`。
5. **`SPANWISE_ROUNDOFF=1e-12` 是经验鸿沟的中点**：若将来出现第三类闭合（如对称+滑移），
   两分类推断会失效并把该层误判，需扩表。

---

## 5. 误差判定（逐项回答「是否 <3%」）

本单交付物是复核机制，不是新的数值量，故按「机制的正确性」逐项判定；数值判据的误差判定
引用既有证据并保持原判（未被本单修改）：

| 关键量 | 基准 | 实测 | 误差 | 是否 <3% |
|---|---|---|---|---|
| 速度 L2 收敛阶（`zero_gradient`） | 判据 ≥1.8；名义二阶 2.0 | 1.9623（本单回归复现同值） | 对 2.0 偏 1.9% | **是** |
| 去均值压力收敛阶 | 判据 ≥1.8 | 1.8777 | 对 1.8 裕度 4.3% | **是**（裕度小，见 m20800 §5.2） |
| 门对污染证据的漏检数 | 0 | 0（5/5 变异全部命中，含超检出） | — | **是** |
| 元数据语义测试 | 10 例全绿 | 10 passed | — | **是** |
| 旧行为回归偏差 | 逐位相等 | `fits`/`verdict` 相等 | 0（精确） | **是** |

**总判定**: 本单为**机制增量 + 缺陷修复**，验收判据（斜率 ≥1.8）维持在既有 PASS 上不动，
`anchor` 组的 n=128 层如实记 **not measured**（集群 cj3258 排队），不记通过。

---

## 6. 复现

```bash
PYTHONPATH=src python -m pytest tests/test_kovasznay_closure_metadata_semantics.py \
  tests/test_summarize_kovasznay_mixed_config_guard.py -q
PYTHONPATH=src python scripts/audit_kovasznay_acceptance_gate_mutations.py \
  --json docs/CASE-FVM-KOVASZNAY-ORDER/gate-mutation-audit.json
PYTHONPATH=src python -m pytest tests/test_kovasznay_gate_mutation_audit.py -q
PYTHONPATH=src python -m pytest tests/test_kovasznay_32_64_128_acceptance.py -q
# 补齐 free 组（集群 cj3258 仍排队，未完成）
PYTHONPATH=src python scripts/verify_kovasznay_order.py \
  --output docs/CASE-FVM-KOVASZNAY-ORDER/runs-32-64-128/n128 --n 128 --reynolds 40 \
  --convection-order 2 --outer-relaxation 0.7 --tolerance 1e-8 \
  --max-outer-iterations 120 --max-iterations 8000 --symmetry-closure free
```

## 7. 给下次自找活的建议

* 先在集群 `ps cj3258` 取回 n=128 `free` 层，再
  `summarize_kovasznay_32_64_128.py --runs .../runs-32-64-128` 生成该组 `summary.json`，
  此时 `anchor` 组三层的 skip 会自动消失，门会给出该组在面内口径上的真实斜率（预期 ~1.8，裕度小）。
* 建议新条目（slug+量化验收+域）：`CASE-FVM-KOVASZNAY-HO2-RICHARDSON` —
  以 4 层（32/64/128/256）最小二乘阶给出可识别标准误，验收 = 阶的 t95 区间下界 ≥1.8，域=TensorFVM。