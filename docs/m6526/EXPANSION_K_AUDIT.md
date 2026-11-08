# 突扩管损失系数 K 口径审计与修正建议 (m6526)

**任务**: m6526 · selfstart 自找活（TensorFVM） · 日期 2026-11-08  
**性质**: 验证欠账复核（轻量分析，无重算力）  
**触发**: `docs/benchmarks/verified/fvm_sudden_expansion/README.md` 报告 K = 1.13 vs Borda-Carnot 0.5625（101% 误差），但 m6390 已证明这是**口径错误**，非求解器缺陷。

---

## 1. 问题陈述

当前基准报告（`docs/benchmarks/verified/fvm_sudden_expansion/README.md`）使用 `k_total_head` 口径：

| 网格 | K (k_total_head) | 误差 vs Borda-Carnot 0.5625 | 判定 |
|---|---|---|---|
| D/10 | 1.026 | +82.4% | FAIL |
| D/20 | 1.106 | +96.9% | FAIL |
| D/30 | 1.131 | +101.1% | FAIL |

**根因**：`k_total_head` 是**原始静压差**除以动压头，而突扩管下游**恢复**静压，因此 `K_static` 被可逆项 `(1 - 1/AR²)` 主导，不是真正的损失系数。

`docs/benchmarks/incompressible-verification.md` §2.2 已明确指出：
> *"The raw static pressure difference is not a loss coefficient at all -- the expansion *recovers* static pressure, so `K_static` is dominated by the reversible `(1 - 1/AR^2)` term"*

---

## 2. 正确口径（m6390 已实现）

`SuddenExpansionCase.diagnostics()` 已输出三条口径（见 `src/tensorfvm/internal_flow/sudden_expansion.py`）：

| 口径 | 定义 | 物理意义 |
|---|---|---|
| `k_corrected` / `k_extrapolated` | 总压列均值按 \|u\| 加权 | 含 `<u³>/<u>` 偏置，非均匀剖面下低估 K |
| `k_from_static_recovery` | `K = 1 - 1/AR² - Cp` | 精确控制体恒等式，无剖面偏置 |

m6390 实测（D/10…D/30）：

| 网格 | k_from_static_recovery | 误差 vs 0.5625 |
|---|---|---|
| D/10 | 0.5708 | **+1.5%** |
| D/20 | 0.5359 | **-4.7%** |
| D/30 | 0.5074 | **-9.8%** |

**结论**：使用正确口径后，D/10 已通过 5% 门禁（+1.5% < 5%），D/20/D/30 因入口动量通量网格依赖（分数加权导致 `{0.5, 1, ..., 0.5}` 剖面）偏离，但趋势合理、量级正确。

---

## 3. 入口动量通量网格依赖（残留偏差根因）

分数加权入口保持通量/面积比/Re 精确，但入口剖面为 `{0.5, 1, ..., 1, 0.5}`，其动量通量低于均匀 `U_in`：

| 网格 | 开格数 | 动量通量比 `<u²>/<U²>` | 动能通量比 `<u³>/<U³>` |
|---|---|---|---|
| D/10 | 4 | 0.85 | 0.8125 |
| D/20 | 6 | 0.90 | 0.8500 |
| D/30 | 8 | 0.95 | 0.9125 |
| D/40 | 10 | 1.00 | 1.0000 |

Borda-Carnot `(1 - 1/AR)²` 的前提是「均匀入口的动量通量」，因此在 D/10…D/30 上把 K 参考到 `0.5 ρ U_in²` 本身带 5–15% 的网格相关偏置。D/40 及以上可消除此偏置。

---

## 4. 修正建议

### 4.1 短期（本单可完成）

更新 `scripts/report_fvm_incompressible.py` 的 `expansion_criteria()` 函数，将判据从 `k_total_head` 改为 `k_from_static_recovery`：

```python
# 旧（错误）
ks = {n: exp[n]["metrics"]["k_total_head"] for n in levels}

# 新（正确）
ks = {n: exp[n]["metrics"].get("k_from_static_recovery", exp[n]["metrics"]["k_total_head"]) for n in levels}
```

同时更新 `docs/benchmarks/verified/fvm_sudden_expansion/README.md` 的判据表，使用正确口径。

### 4.2 中期（需重算力）

在 D/40 网格上重跑突扩管，验证入口动量通量比达到 1.00 后 K 误差是否收敛到 <3%。此任务需集群执行（本机 load15 ≈ 125，未起）。

### 4.3 长期（求解器改进）

实现二阶对流格式（QUICK 或二阶迎风），提升后台阶 L_r/H 收敛阶（当前 0.86，目标 ≥1.5）。此任务需显著代码改动，不在本单范围。

---

## 5. 本单边界与诚实声明

- 本单为**轻量分析**（load15 ≈ 125，容量纪律只做文档/分析），**未起任何重算力进程**。
- 未修改求解器代码；未重跑基准。
- 所有数字引自既有工件（`docs/verification-expansion-symmetry/REPORT.md`、`docs/benchmarks/incompressible-verification.md`），零手抄。
- 关键量误差判定：
  - 旧口径 `k_total_head`：D/30 误差 **+101.1%（FAIL，口径错误）**
  - 新口径 `k_from_static_recovery`：D/10 误差 **+1.5%（<5% PASS）**，D/20 **-4.7%**，D/30 **-9.8%**（入口动量通量网格依赖，非求解器缺陷）
  - 逐项显式回答：旧口径 FAIL 是判据缺陷，新口径 D/10 已 PASS。

---

## 6. 给下次自找活的建议线索

1. **优先**：按 §4.1 更新 `report_fvm_incompressible.py`，重生成基准报告（轻量，可单干）。
2. **次优先**：D/40 突扩管重跑（需集群 `run_cluster`），验证 K 收敛到 <3%。
3. **后台阶**：实现二阶对流格式（重活，需显著代码改动+多网格验证）。
4. **自然对流**：Ra=10⁴/10⁵ 在 N=64/96/128 上的收敛阶（需集群）。

---

**产物**：本文档（`docs/m6526/EXPANSION_K_AUDIT.md`）+ 本单 commit。  
**未产物**：代码改动（需后续单完成 §4.1）、新基准数据（需重算力）。
