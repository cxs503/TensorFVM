# SIMPLEC 压力-速度耦合：从「未实现」到带解析基准门禁的实现

## 1. 缺口是什么

`verification/simple_gpu.py` 的公开结论里写明：「SIMPLEC/PISO/PIMPLE 仍无独立生产实现，
不能冒用这些名称」；而 3D 原生求解器 `native_simple3d.py` 早已有 SIMPLEC（`pressure_velocity_algorithm='simplec'`）。
因此 2D 同位（collocated）生产迭代 `tensorfvm.solver.SimpleSolver` 是全仓唯一
「文档声明不支持、但同仓另有参考实现」的验证欠账点。

## 2. 做了什么（最小改动）

只改**动量-压力耦合的通量迁移率（mobility）**，压力修正算子、Rhie–Chow 面、边界处理、
残差定义一律不动：

```
d_SIMPLE  = A / (aP/alpha)
d_SIMPLEC = A / max(aP/alpha - sum(aN), 0.1*aP)        # Van Doormaal & Raithby (1984)
```

```python
SolverConfig(pressure_velocity_algorithm="simplec")     # 默认 "simple"，行为不变
SimpleSolver.simplec_floor_fraction = 0.1               # 分母下限，防止 aP/alpha - ΣaN 退化
solver.simplec_limited_dofs                             # 有多少自由度被下限接管（可审计）
```

推论：SIMPLEC 只改变**迭代路径**，不改变离散不动点——这正是下面用数值验证的第一件事。

## 3. 验证证据（全部由程序生成，零手抄）

复现命令：

```bash
PYTHONPATH=src python -m tensorfvm.verification.simplec_channel --output outputs/simplec_audit/run1
PYTHONPATH=src python -m pytest tests/test_simplec.py -q
```

案例：充分发展槽道，Poiseuille 解析解 `u = 6 Ub (y/H)(1-y/H)`，Re_H = 1，
Ub = 5e-4 m/s，H = 0.02 m，nx = 3ny，收敛判据 continuity/momentum/mass ≤ 1e-6。
门禁复用共享 `Metric`/`evaluate`（上限 3%，质量守恒与散度上限 1e-6）。

### 3.1 ny=12：全部门禁通过，且 SIMPLEC 在 αp=1 下可用

| 配置 | αu | αp | 迭代数 | 收敛 | velocity_l2 | 质量不平衡 | 缩放散度 | 门禁 |
|---|---|---|---|---|---|---|---|---|
| simple-default | 0.7 | 0.3 | 41 | yes | 0.006344 | 0.0 | 0.0 | 全过 |
| simple-tuned | 0.8 | 0.3 | 43 | yes | 0.006344 | 0.0 | 0.0 | 全过 |
| simple αp=1 | 0.8 | 1.0 | — | **崩溃** | — | — | — | RuntimeError（压力修正 CG 失去正定性，第 12 步） |
| simplec | 0.7 | 1.0 | **32** | yes | 0.006344 | 0.0 | 0.0 | 全过 |
| simplec | 0.8 | 1.0 | 49 | yes | 0.006344 | 0.0 | 0.0 | 全过 |
| simplec 无下限 | 0.8 | 1.0 | 49 | yes | 0.006344 | 0.0 | 0.0 | 全过 |

### 3.2 不动点一致性（关键正确性判据）

SIMPLEC 与 SIMPLE-default 的收敛场逐点对比（峰值差 / Ub，上限 1e-4）：

| 配置 | u | v |
|---|---|---|
| simple-tuned | 6.7e-08 | 8.7e-09 |
| simplec (0.7, 1.0) | 2.1e-06 | 3.3e-07 |
| simplec (0.8, 1.0) | 6.7e-08 | 3.7e-08 |

差值与求解容差同量级 → 迁移率改动没有污染离散方程，只改路径。

### 3.3 网格加密（补充，独立单点运行）

| ny | simple (0.7, 0.3) | simple (0.8, 0.3) | simplec (0.8, 1.0) | velocity_l2 |
|---|---|---|---|---|
| 12 | 41 | 43 | 49 | 6.344e-3 |
| 24 | 87 | 53 | 53 | 1.620e-3 |
| 48 | 282 | 166 | 166 | 4.29e-4 |

### 3.4 图

- `outputs/simplec_audit/run1/simplec-convergence.png`：continuity/momentum 残差 vs 迭代
  （含 SIMPLE αp=1 崩溃前的残差曲线）；
- `outputs/simplec_audit/run1/simplec-profile.png`：x = 5H 截面速度剖面 vs 解析抛物线；
- `outputs/simplec_audit/run1/simplec-iterations.png`：等残差、等不动点下各配置迭代成本。

## 4. 误差判定（逐项，显式回答是否 < 3%）

| 关键量 | 基准 | 误差 | < 3%? |
|---|---|---|---|
| 截面速度剖面 L2 | `6Ub(y/H)(1-y/H)` | 6.344e-3 (ny=12) / 1.620e-3 (ny=24) / 4.29e-4 (ny=48) | **是** |
| 截面速度剖面 L∞（峰值归一） | 同上 | 6.803e-3 (ny=12) | **是** |
| 压力梯度 dp/dx | `-12 μ Ub / H²` | 1.370e-2 (ny=12) | **是** |
| 壁面剪切 | `6 μ Ub / H` | 1.370e-2 (ny=12) | **是** |
| 压力/壁面力平衡 | 0 | 1.4e-07 | **是** |
| 质量守恒 / 缩放散度 | 0 | ≤ 1e-14 | **是**（判据 1e-6） |
| 不动点 SIMPLE↔SIMPLEC | 同网格 SIMPLE 收敛解 | ≤ 2.1e-06·Ub | **是**（判据 1e-4） |

无文献比对的项（迭代成本）不给百分比结论，只给计数。

## 5. 注意点 / 局限（诚实结论）

1. **SIMPLEC 不是普适加速**。ny=12 上 SIMPLEC（49 步）比默认 SIMPLE（41 步）更慢；
   ny=24/48 上与「调好的」SIMPLE（αu=0.8, αp=0.3）完全打平（53 vs 53、166 vs 166），
   只在默认 αp=0.3 面前显得快（53 vs 87、166 vs 282）。真正买到的能力是
   **αp=1 下仍然稳定**：同配置 SIMPLE 在第 12 步压力修正 CG 失去正定性而崩溃。
2. **分母下限在本案例从未生效**（`simplec_limited_dofs` 全为 0，去掉下限结果逐位相同）。
   它是安全阀，其必要性在此算例上**未被证明**——不要把它当性能特性宣传。
3. 覆盖范围：单块、层流、壁面约束的充分发展槽道，两个网格级别 + 一个加密点；
   不涉及高 Re 分离流、3D、湍流模型；3D 原生 SIMPLEC 路径未改动；PISO/PIMPLE 仍未实现。
4. 槽道自检结论：ny=8 时解析门禁不通过（剖面 L2 超 3%），故验证取 ny≥12。

## 6. 文件清单

- `src/tensorfvm/solver.py`：`+pressure_velocity_algorithm` 配置、SIMPLEC 迁移率、`simplec_limited_dofs`、`simplec_floor_fraction`。
- `src/tensorfvm/verification/cases.py`：`channel()` 增加 `algorithm`/松弛因子入参（默认不变），返回 `simplec_limited_dofs`。
- `src/tensorfvm/verification/simplec_channel.py`：审计 CLI（矩阵实验 + 三张图 + JSON/Markdown）。
- `tests/test_simplec.py`：5 组测试（配置校验、迁移率公式逐点核对、解析门禁、不动点一致、αp=1 稳定性）。

## 7. 后续闭环：定义级守卫 + 负控（证据见 `scripts/artifacts/simplec-mobility/`）

第 5 节局限 2（「分母下限在本案例从未生效，其必要性未被证明」）已在后续闭环，三层证据都在仓内：

1. **定义级** `tests/test_simplec_mobility.py`（11 条守卫）：从 `_momentum()` 组装的模板独立重算 `aP`、`sum(aN)`，与 `_predict()` 返回的迁移率逐自由度比对；`alpha_u = 1` 强制分母退化，实测限幅恰为 `10 d_SIMPLE`（闭式），**该分支此前从未被执行**；纯扩散模板上比值等于闭式 `1/(1-alpha_u)`。
2. **端到端** `verification/simplec_channel.py` 的 ny = 12/24 扫描：四项配置全部通过 3% 解析门（SIMPLEC αu=0.8/αp=1：剖面 L2 = 0.634%、压力梯度 = 1.370%、49 次迭代；SIMPLE 默认：0.634%、41 次）。SIMPLEC 与 SIMPLE 收敛到同一离散不动点（≤ 2.08e-06·Ub）。
3. **负控**：把 SIMPLEC 分母替换成 SIMPLE 对角线（方案退化），8 条守卫变红（`8 failed, 3 deselected, 2 warnings in 6.57s`），源码在 `finally` 中恢复并校验 SHA-256 逐字节一致——守卫不是在测空气。

复算与溯源：

```bash
python scripts/verify_simplec_mobility.py --output scripts/artifacts/simplec-mobility
python -m pytest tests/test_simplec_mobility.py tests/test_simplec_artifacts.py -q
```

- 仓内只留摘要 `summary.json`（41 KB）+ 缩小图（45 KB，按推送门禁置于 `docs/simplec-mobility/evidence/`，不放在工件目录），合计 98085 B（PR 工件帽 200 KB）；逐迭代完整证据写入仓外证据库（脚本打印路径），`--slim-from` 可重派生，二者不会漂移。
- `summary.json.provenance` 记录 `src/tensorfvm/solver.py`（SHA-256 `f758b76e196844d7…`）等源码摘要；`tests/test_simplec_artifacts.py` 断言其与当前工作树一致，改动求解器必须重算证据，旧结论无法蒙混过关。
