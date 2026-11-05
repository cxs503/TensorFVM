# SIMPLEC 压力-速度耦合的验证闭环（定义级 + 端到端 + 负控）

## 1 案例介绍

二维同位（collocated）生产迭代 `src/tensorfvm/solver.py` 中，SIMPLEC 与 SIMPLE 的唯一物理差异是压力修正的**迁移率（mobility）**：

```
d_SIMPLE  = A / (aP/alpha_u)
d_SIMPLEC = A / max(aP/alpha_u - sum(aN), simplec_floor_fraction*aP)   # Van Doormaal & Raithby (1984)
```

验证载体是仓库自带的解析解算例 `verification/cases.py::channel`：二维充分发展层流槽道（Poiseuille），解析速度剖面 `u(y) = 6 Ub (y/H)(1 - y/H)`、压力梯度 `dp/dx = -12 mu Ub/H^2`、壁面剪应力 `6 mu Ub/H`；误差因此对齐闭式解。

| 参数 | 值 |
|---|---|
| 计算域 | L = 0.12 m, H = 0.02 m（nx = 3 ny） |
| 网格 | ny = 12（汇总）/ 12, 24（扫描） |
| 入流速度 Ub | 5.0e-04 m/s |
| 密度 / 雷诺数 | 1000 kg/m3 / Re_H = 1 |
| 收敛判据 | 残差 1e-06，最多 2000 次外迭代 |
| 验收门 | 与解析解相对误差 < 3% |

## 2 怎么做的

三层证据，缺一层不算闭环：

1. **定义级（本轮新增）** `tests/test_simplec_mobility.py`：从 `_momentum()` 组装的模板独立重算 `aP`、`sum(aN)`，与 `_predict()` 返回的迁移率逐自由度比对；用**纯扩散模板**（对流项→0）验证比值等于闭式 `1/(1-alpha_u)`；用 `alpha_u = 1` 强制分母退化，覆盖 `simplec_floor_fraction` 限幅分支（此前审计 `limited_dofs = 0`，该分支从未被执行）。
2. **端到端** `verification/simplec_channel.py`：四种配置 × 两套网格收敛到解析门，并记录 SIMPLE 在 `alpha_p = 1` 时的崩溃。
3. **负控（本轮新增）**：把 SIMPLEC 分母替换成 SIMPLE 的对角线（SIMPLEC 退化为 SIMPLE），守卫必须变红；源码在 `finally` 中原地恢复并用 SHA-256 校验逐字节一致。

工具链：PyTorch 2-D 同位离散 + pytest + matplotlib；复算：`python scripts/verify_simplec_mobility.py --output scripts/artifacts/simplec-mobility`。

## 3 计算结果

槽道算例（ny = 12）：

| configuration | alpha_u | alpha_p | iterations | converged | velocity_L2 vs analytic | mass imbalance | scaled divergence | gates |
|---|---:|---:|---:|---|---:|---:|---:|---|
| simple-default | 0.7 | 0.3 | 41 | yes | 0.634% | 3.47e-16 | 9.50e-15 | all pass |
| simple-tuned | 0.8 | 0.3 | 43 | yes | 0.634% | 3.47e-16 | 6.94e-15 | all pass |
| simplec-a07-p10 | 0.7 | 1.0 | 32 | yes | 0.634% | 3.47e-16 | 5.96e-15 | all pass |
| simplec-a08-p10 | 0.8 | 1.0 | 49 | yes | 0.634% | 0.00e+00 | 9.29e-15 | all pass |
| simplec-no-floor | 0.8 | 1.0 | 49 | yes | 0.634% | 0.00e+00 | 9.29e-15 | all pass |

固定点一致性（相对 `simple-default` 的峰值分量差 / Ub，限值 0.0001，判定 True）：

| 配置 | $\Delta u / U_b$ | $\Delta v / U_b$ |
|---|---:|---:|
| simple-tuned | 6.685e-08 | 8.666e-09 |
| simplec-a07-p10 | 2.080e-06 | 3.272e-07 |
| simplec-a08-p10 | 6.684e-08 | 3.695e-08 |
| simple-default | 0.000e+00 | 0.000e+00 |

- SIMPLE 在 `alpha_u = 0.7, alpha_p = 1` 下崩溃：`RuntimeError`（12 次迭代后）；SIMPLEC 同一 `alpha_p = 1` 收敛到 49 次迭代并通过全部解析门。
- 迭代成本：SIMPLEC(0.8, alpha_p=1) 49 次 vs SIMPLE 默认 41 次（-20% 更少）、SIMPLE 调参 43 次（-14%）。即**不是**普遍加速，换来的是 `alpha_p = 1` 的稳定性。
- 限幅分支：`simplec_limited_dofs = {"12/simple-default": 0, "12/simple-tuned": 0, "12/simplec-a07-p10": 0, "12/simplec-a08-p10": 0, "24/simple-default": 0, "24/simple-tuned": 0, "24/simplec-a07-p10": 0, "24/simplec-a08-p10": 0, "12/simplec-no-floor": 0}`，`floor_free_equivalent = True`（光滑槽道内限幅器不生效，其必要性由 `alpha_u = 1` 守卫单独证明）。

### 守卫与负控的实际执行记录

| 项 | 结果 |
|---|---|
| 守卫套件 `tests/test_simplec_mobility.py` | 11 passed, 2 warnings in 21.73s |
| 负控基线（未变异，子集） | 8 passed, 3 deselected, 2 warnings in 10.23s |
| 负控变异（SIMPLEC → SIMPLE） | 8 failed, 3 deselected, 2 warnings in 5.53s |
| 负控判定（变异后变红） | True |
| 变异后源码 SHA-256 恢复一致 | `f758b76e196844d7…` |

变异后变红的守卫：

- `test_diffusion_dominated_ratio_is_one_over_one_minus_alpha[0.7]`
- `test_diffusion_dominated_ratio_is_one_over_one_minus_alpha[0.8]`
- `test_floor_limiter_engages_and_stays_finite`
- `test_simplec_mobility_is_the_1984_denominator[u-0.7]`
- `test_simplec_mobility_is_the_1984_denominator[u-0.8]`
- `test_simplec_mobility_is_the_1984_denominator[v-0.7]`
- `test_simplec_mobility_is_the_1984_denominator[v-0.8]`
- `test_simplec_sustains_pressure_relaxation_one_where_simple_fails`

## 4 结果图

图件由 `verification/simplec_channel.py` 程序化生成：

![convergence](../../../docs/simplec-mobility/simplec-convergence.png) — 连续性/动量残差逐迭代对比：SIMPLE 默认 vs SIMPLEC，及 SIMPLE 在 `alpha_p = 1` 的崩塌（crimson）。

![profile](../../../docs/simplec-mobility/simplec-profile.png) — x = 5H 截面速度剖面 vs 解析 `6Ub(y/H)(1-y/H)`：两方案落在同一解析曲线上。

![iterations](../../../docs/simplec-mobility/simplec-iterations.png) — 等残差、等固定点下的迭代成本对比。

## 5 注意点 / 踩过的坑

1. **模板时效性**：`_predict()` 会推进速度场，动量模板的迎风系数依赖速度，定义级比对必须在 `_predict()` **之前**取模板，否则读到的是另一套系数（本轮首个失败即此：`alpha_u = 1` 下限幅分支偏差 3.2%）。
2. **限幅常量是自洽而非文献值**：`simplec_floor_fraction = 0.1` 无文献出处，是本实现的安全下限；守卫显式钉住该常量（`.1`），改动必须同时改测试。
3. **`alpha_u = 1` 是退化点**：此时 `aP/alpha - sum(aN)` 在纯扩散模板上恰为 0，SIMPLEC 的加速度修正理论发散；限幅把迁移率截到 `10 d_SIMPLE`。适用壁面/光滑区，湍流强对流单元仍待评估。
4. **适用范围**：单块层流（Re_H = 1）二维同位网格；不涉及三维 native SIMPLEC 与 PISO/PIMPLE。

## 6 误差判定（关键量 vs 基准，逐项）

| 关键量 | 基准 | 实测相对误差 | 是否 < 3% |
|---|---|---:|---|
| 截面速度 L2（SIMPLEC 0.8/1.0） | 解析 `6Ub(y/H)(1-y/H)` | 0.634% | 是 |
| 截面速度 L2（SIMPLE 0.7/0.3） | 同上 | 0.634% | 是 |
| 下游压力梯度 | 解析 `-12 mu Ub/H^2` | 1.370% | 是 |
| 全局质量不平衡 | 0（离散守恒） | 0.00e+00（绝对） | 是（< 1e-6） |
| 面散度（H/Ub 尺度） | 0 | 9.29e-15（绝对） | 是（< 1e-6） |
| SIMPLEC vs SIMPLE 固定点差 | 同一离散解 | ≤ 2.08e-06 × Ub | 是 |
| SIMPLEC 迁移率 vs 1984 定义 | `A/max(aP/alpha - sum(aN), 0.1aP)` | 逐自由度 rtol 1e-6 内一致 | 是（机器精度） |
| 纯扩散模板迁移率比 | 闭式 `1/(1-alpha_u)`（0.7→3.3333, 0.8→5.0） | 见守卫，rel 5e-3 内 | 是 |

结论：两个关键物理量（速度剖面、压力梯度）对解析解的误差均 **< 3%**；定义级误差在机器精度量级。限幅常量 0.1 的物理最优性未证明（只有自洽性证据），如实声明。

## 7 工件瘦身（合并门禁 200KB）

本目录工件已按门禁瘦身（原始体积见 `verification.json` 的 `artifact_slimming`），全保真度由 §2 命令重生成：

- JSON 去缩排（浮点 `repr` 全精度）；逐迭代 `history` 按 head 3 + 每 10 步 + tail 2 采样，每条带 `history_sampling`（`total_iterations` + 全轨迹 `sha256`），其余字段逐字节不变；
- PNG 改 128 色调色板，像素尺寸不变；
- 瘦身命令 `python scripts/slim_simplec_artifacts.py --output scripts/artifacts/simplec-mobility`，守卫 `tests/test_simplec_artifact_budget.py`。
