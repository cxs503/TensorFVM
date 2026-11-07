# 自动生成的判据与数字（machine-written）

本文件由 `scripts/run_resistance_chain.py` 从 `artifacts/resistance-chain/form_factor_chain.json` 生成，任何数字均未手工誊抄。重跑命令：`python3 scripts/run_resistance_chain.py`。

## C1 — 形式因子 (1+k) vs Holtrop

* 判定：**WITHHELD**
* Holtrop 回归值 (1+k) = 1.186736
* 5 % 闸门区间 = [1.127399, 1.246073]
* CFD 值：未提交（no resolved double-body RANS is affordable on this node (required ~2.9e+06 cells, predicted setup 62.94 h before the first iteration), and on the largest affordable mesh (2184 cells, spacing 0.16 m) the native SST path diverged (momentum residual 3.4e14 -> 5.0e72 over 40 outer iterations; see artifacts/resistance-chain/rans_attempt.json); an under-resolved stair-step mesh would produce a number with an unquantified bias, which is worse than withholding）
* 需要网格量级估计 = 2.946e+06 cells；外推 setup = 62.9 h，每 100 次外迭代 = 86.5 h

## 几何审计（先验证 Cb/L/B，再谈阻力）

| 量 | 目标 | 拟合值 | 相对误差 |
| --- | --- | --- | --- |
| Cb | 0.708000 | 0.708013 | +1.90e-05 |
| Cm | 0.985000 | 0.985000 | +8.88e-16 |
| Cwp | 0.808000 | 0.808029 | +3.59e-05 |
| LCB [%L] | -1.6880 | -1.6879 | +9.50e-05 |

* 解析湿表面积 = 8.6931 m²；Holtrop 回归值 = 8.7058 m²；差 -0.146 %

| 网格步长 [m] | cells | body cells | 体体积比 (2∇) | hull 面数 | hull 面积比 |
| --- | --- | --- | --- | --- | --- |
| 0.120 | 5308 | 628 | 0.9985 | 588 | 1.195 |
| 0.090 | 11284 | 1316 | 0.9981 | 960 | 1.202 |
| 0.060 | 26272 | 3392 | 0.9963 | 1852 | 1.228 |

* 几何审计结论：the mission particulars L=7 m, Cb=0.708, B=0.936 are not geometrically similar to KVLCC2 (L/B 7.48 vs 5.52, Cb 0.708 vs 0.8098); the form factor target therefore depends on which of the two conventions is meant
* 任务口径 L/B = 7.479（真船 5.517）；(1+k) 任务口径 = 1.186736，真船口径 = 1.359715

## C2 — ITTC-78 相关补贴 ΔCf

* 判定：**PASS**
* 与平台独立实现（`backend/physics/holtrop/friction.py::delta_cf_ittc78`）在 25 个 (ks, L) 点上最大相对偏差 = 0.0
* 逐 Re 表（8 点 log 分布）见 `artifacts/resistance-chain/friction_allowance.csv`，摘录：

| case | Re | ITTC-57 Cf | Schoenherr Cf | P-S Cf | 律间散布 | ΔCf(ITTC-78) | Holtrop Ca |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model | 1.000e+06 | 0.004687 | 0.004409 | 0.004471 | 6.15 % | 0.001382 | 0.000791 |
| model | 5.179e+07 | 0.002297 | 0.002277 | 0.002338 | 2.64 % | 0.001382 | 0.000791 |
| model | 1.000e+09 | 0.001531 | 0.001531 | 0.001571 | 2.59 % | 0.001382 | 0.000791 |
| ship | 1.000e+06 | 0.004687 | 0.004409 | 0.004471 | 6.15 % | 0.000176 | 0.000233 |
| ship | 5.179e+07 | 0.002297 | 0.002277 | 0.002338 | 2.64 % | 0.000176 | 0.000233 |
| ship | 1.000e+09 | 0.001531 | 0.001531 | 0.001571 | 2.59 % | 0.000176 | 0.000233 |

## 双口径裁决：补贴项只能加一次

| case | Re | (1+k)Cf | ΔCf(ITTC-78) | Holtrop Ca | 双计之和 | 双计/C_V | C_T(ITTC-78) | C_T(Ca) | 两口径相对差 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| model | 7.231e+06 | 0.003770 | 0.001382 | 0.000791 | 0.002173 | 57.6 % | 0.005152 | 0.004560 | +12.97 % |
| ship | 2.151e+09 | 0.001897 | 0.000176 | 0.000233 | 0.000408 | 21.5 % | 0.002072 | 0.002129 | -2.68 % |

## C3 — 统一口径后的 U_V（5 % 闸门）

| case | u_friction | u_form | u_allowance | u_cfd | U_V（统一后） | U_V（统一前，双计） | 通过 5 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| model_ittc78 | 0.55 % | 2.11 % | 0.31 % | 0.00 % | **2.20 %** | 24.45 % | 是 |
| model_holtrop_ca | 0.62 % | 2.39 % | 0.20 % | 0.00 % | **2.47 %** | 38.91 % | 是 |
| ship_ittc78 | 0.66 % | 2.64 % | 0.10 % | 0.00 % | **2.73 %** | 11.70 % | 是 |
| ship_holtrop_ca | 0.65 % | 2.57 % | 0.13 % | 0.00 % | **2.65 %** | 11.38 % | 是 |

闸门裁决力（能否区分统一口径与双计口径）：

| case | U_V 统一后（ITTC-78 口径） | U_V 统一前 | 闸门 | 拒绝双计 | 接受统一 |
| --- | --- | --- | --- | --- | --- |
| model | 2.20 % | 24.45 % | 5 % | True | True |
| ship | 2.73 % | 11.70 % | 5 % | True | True |

## 求解器代价实测（为何 C1 只能 WITHHELD）

| cells | setup [s] | 单次外迭代 [s] |
| --- | --- | --- |
| 128 | 9.6 | 0.08 |
| 432 | 33.4 | 0.09 |
| 1024 | 75.8 | 0.26 |
| 3456 | 263.4 | 2.50 |

* 拟合标度：setup ∝ N^1.00，迭代 ∝ N^1.06
* y+ = 100 所需首层网格高度 = 4.86 mm；对应近壁单元数 ≈ 7.365e+05
* 线性求解器：direct (SuperLU) is the only supported segregated method; pyamg is not installed, so no AMG fallback
* 结论：a resolved double-body RANS is outside the compute budget of this node; the form factor is therefore withheld rather than reported from an under-resolved mesh

## 判据汇总

| 判据 | 判定 | 说明 |
| --- | --- | --- |
| C1_form_factor_vs_holtrop_within_5pct | WITHHELD | no resolved double-body RANS is affordable on this node (required ~2.9e+06 cells, predicted setup 62.94 h before the first iteration), and on the largest affordable mesh (2184 cells, spacing 0.16 m) the native SST path diverged (momentum residual 3.4e14 -> 5.0e72 over 40 outer iterations; see artifacts/resistance-chain/rans_attempt.json); an under-resolved stair-step mesh would produce a number with an unquantified bias, which is worse than withholding |
| C2_delta_cf_vs_ittc78_within_3pct | PASS | the ITTC-78 allowance has no inherent Reynolds dependence (it scales with (ks/L)^(1/3)); the per-Re table carries it alongside the three friction laws |
| C3_U_V_within_5pct_after_unification | PASS |  |
| C4_pytest_cases | PASS |  |
| C5_figures | PASS |  |

