# XVAL-FVM-CONVERGENCE-DIAGNOSIS-V2 — ref vs got（程序生成，勿手改）

回归命令：`PYTHONPATH=src python docs/xval/convergence-diagnosis-v2/xval_driver.py --out docs/xval/convergence-diagnosis-v2`

- 生成时刻 `2026-11-13T05:15:01+00:00`；环境 {'python': '3.11.6', 'machine': 'loongarch64', 'numpy': '2.4.6'}
- ref = 独立实现 `gci_reference.py`（Roache 1994 / Celik et al. 2008 三网格 GCI，不 import 本仓任何模块）
- got = 本仓 `tensorfvm.verification.sod1978_convergence` 的相邻对速率与 log-log 最小二乘指数（v2 语义：对**误差**序列拟合）
- 预声明：p 相对差 ≤ 15%；导出量相对差 ≤ 15%；重叠三重组一致性 ≤ 15%；Fs=1.25

## A. 观测阶（ref vs got）

| sweep（网格） | 功能量 | 极限形式 | ref 收敛类 | p_ref | p_got(lsq) | p_got(相邻对) | Δlsq | Δsucc | lsq | succ | 单三重组 eq.17 比 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| main_100_200_400 `[100, 200, 400]` | shock_x | nonzero_limit | monotonic | 2.431 | 1.387 | 0.936 | 42.96% | 61.51% | ❌ | ❌ | 1.00 |
| main_100_200_400 `[100, 200, 400]` | contact_x | nonzero_limit | monotonic | 0.731 | 1.033 | 1.136 | 41.19% | 55.29% | ❌ | ❌ | 1.00 |
| main_100_200_400 `[100, 200, 400]` | l1_rho_full | zero_limit | monotonic | 1.068 | 0.936 | 0.894 | 12.36% | 16.24% | ✅ | ❌ | 0.54 |
| main_100_200_400 `[100, 200, 400]` | l1_rho_cellavg | zero_limit | monotonic | 0.817 | 0.894 | 0.917 | 9.35% | 12.15% | ✅ | ✅ | 0.53 |
| main_100_200_400 `[100, 200, 400]` | l1_rho_plateaus | zero_limit | monotonic | 3.002 | 2.872 | 2.773 | 4.32% | 7.61% | ✅ | ✅ | 0.15 |
| main_100_200_400 `[100, 200, 400]` | smooth_wave_l1 | zero_limit | monotonic | 2.143 | 2.114 | 2.096 | 1.36% | 2.21% | ✅ | ✅ | 0.23 |
| fine_200_400_800 `[200, 400, 800]` | shock_x | nonzero_limit | monotonic | 1.066 | 0.881 | 0.826 | 17.35% | 22.48% | ❌ | ❌ | 1.00 |
| fine_200_400_800 `[200, 400, 800]` | contact_x | nonzero_limit | monotonic | 0.787 | 1.422 | 1.708 | 80.72% | 117.09% | ❌ | ❌ | 1.00 |
| fine_200_400_800 `[200, 400, 800]` | l1_rho_full | zero_limit | monotonic | 0.939 | 0.876 | 0.857 | 6.69% | 8.66% | ✅ | ✅ | 0.55 |
| fine_200_400_800 `[200, 400, 800]` | l1_rho_cellavg | zero_limit | monotonic | 0.905 | 0.922 | 0.927 | 1.87% | 2.44% | ✅ | ✅ | 0.53 |
| fine_200_400_800 `[200, 400, 800]` | l1_rho_plateaus | zero_limit | monotonic | 4.749 | 1.563 | 0.353 | 67.08% | 92.56% | ❌ | ❌ | 0.78 |
| fine_200_400_800 `[200, 400, 800]` | smooth_wave_l1 | zero_limit | monotonic | 2.112 | 2.071 | 2.045 | 1.94% | 3.13% | ✅ | ✅ | 0.24 |
| combined_100_200_400_800 `[100, 200, 400, 800]` | shock_x | nonzero_limit | monotonic | 1.066 | 0.881 | 0.826 | 17.35% | 22.48% | ❌ | ❌ | 1.00 |
| combined_100_200_400_800 `[100, 200, 400, 800]` | contact_x | nonzero_limit | monotonic | 0.787 | 1.422 | 1.708 | 80.72% | 117.09% | ❌ | ❌ | 1.00 |
| combined_100_200_400_800 `[100, 200, 400, 800]` | l1_rho_full | zero_limit | monotonic | 0.939 | 0.876 | 0.857 | 6.69% | 8.66% | ✅ | ✅ | 0.55 |
| combined_100_200_400_800 `[100, 200, 400, 800]` | l1_rho_cellavg | zero_limit | monotonic | 0.905 | 0.922 | 0.927 | 1.87% | 2.44% | ✅ | ✅ | 0.53 |
| combined_100_200_400_800 `[100, 200, 400, 800]` | l1_rho_plateaus | zero_limit | monotonic | 4.749 | 1.563 | 0.353 | 67.08% | 92.56% | ❌ | ❌ | 0.78 |
| combined_100_200_400_800 `[100, 200, 400, 800]` | smooth_wave_l1 | zero_limit | monotonic | 2.112 | 2.071 | 2.045 | 1.94% | 3.13% | ✅ | ✅ | 0.24 |
| cfl0.2_100_200_400 `[100, 200, 400]` | shock_x | nonzero_limit | monotonic | 2.345 | 1.359 | 0.940 | 42.02% | 59.91% | ❌ | ❌ | 1.00 |
| cfl0.2_100_200_400 `[100, 200, 400]` | contact_x | nonzero_limit | monotonic | 1.240 | 2.918 | 4.145 | 135.36% | 234.35% | ❌ | ❌ | 1.00 |
| cfl0.2_100_200_400 `[100, 200, 400]` | l1_rho_full | zero_limit | monotonic | 1.114 | 0.942 | 0.887 | 15.47% | 20.35% | ❌ | ❌ | 0.54 |
| cfl0.2_100_200_400 `[100, 200, 400]` | l1_rho_cellavg | zero_limit | monotonic | 0.911 | 0.910 | 0.909 | 0.10% | 0.12% | ✅ | ✅ | 0.53 |
| cfl0.2_100_200_400 `[100, 200, 400]` | l1_rho_plateaus | zero_limit | monotonic | 2.709 | 3.182 | 3.561 | 17.49% | 31.47% | ❌ | ❌ | 0.08 |
| cfl0.2_100_200_400 `[100, 200, 400]` | smooth_wave_l1 | zero_limit | monotonic | 2.114 | 2.091 | 2.076 | 1.11% | 1.80% | ✅ | ✅ | 0.24 |
| order1_100_200_400 `[100, 200, 400]` | shock_x | nonzero_limit | monotonic | 0.359 | 0.765 | 0.869 | 113.18% | 142.31% | ❌ | ❌ | 1.00 |
| order1_100_200_400 `[100, 200, 400]` | contact_x | nonzero_limit | divergent | n/a | 0.540 | 0.663 | n/a | n/a | ❌ | ❌ | n/a |
| order1_100_200_400 `[100, 200, 400]` | l1_rho_full | zero_limit | monotonic | 0.556 | 0.634 | 0.651 | 14.03% | 17.06% | ✅ | ❌ | 0.64 |
| order1_100_200_400 `[100, 200, 400]` | l1_rho_cellavg | zero_limit | monotonic | 0.471 | 0.621 | 0.653 | 31.86% | 38.61% | ❌ | ❌ | 0.64 |
| order1_100_200_400 `[100, 200, 400]` | l1_rho_plateaus | zero_limit | monotonic | 1.670 | 1.485 | 1.398 | 11.07% | 16.31% | ✅ | ❌ | 0.38 |
| order1_100_200_400 `[100, 200, 400]` | smooth_wave_l1 | zero_limit | monotonic | 0.990 | 0.995 | 0.997 | 0.49% | 0.66% | ✅ | ✅ | 0.50 |

## B. 重叠三重组一致性（≥4 网格，非空转的渐近区检验）

| sweep | 功能量 | p(粗三重组) | p(细三重组) | 相对差 | 渐近区一致 |
|---|---|---|---|---|---|
| main_100_200_400 | shock_x | n/a | n/a | n/a | n/a |
| main_100_200_400 | contact_x | n/a | n/a | n/a | n/a |
| main_100_200_400 | l1_rho_full | n/a | n/a | n/a | n/a |
| main_100_200_400 | l1_rho_cellavg | n/a | n/a | n/a | n/a |
| main_100_200_400 | l1_rho_plateaus | n/a | n/a | n/a | n/a |
| main_100_200_400 | smooth_wave_l1 | n/a | n/a | n/a | n/a |
| fine_200_400_800 | shock_x | n/a | n/a | n/a | n/a |
| fine_200_400_800 | contact_x | n/a | n/a | n/a | n/a |
| fine_200_400_800 | l1_rho_full | n/a | n/a | n/a | n/a |
| fine_200_400_800 | l1_rho_cellavg | n/a | n/a | n/a | n/a |
| fine_200_400_800 | l1_rho_plateaus | n/a | n/a | n/a | n/a |
| fine_200_400_800 | smooth_wave_l1 | n/a | n/a | n/a | n/a |
| combined_100_200_400_800 | shock_x | 2.431 | 1.066 | 0.562 | ❌ |
| combined_100_200_400_800 | contact_x | 0.731 | 0.787 | 0.076 | ✅ |
| combined_100_200_400_800 | l1_rho_full | 1.068 | 0.939 | 0.121 | ✅ |
| combined_100_200_400_800 | l1_rho_cellavg | 0.817 | 0.905 | 0.107 | ✅ |
| combined_100_200_400_800 | l1_rho_plateaus | 3.002 | 4.749 | 0.582 | ❌ |
| combined_100_200_400_800 | smooth_wave_l1 | 2.143 | 2.112 | 0.015 | ✅ |
| cfl0.2_100_200_400 | shock_x | n/a | n/a | n/a | n/a |
| cfl0.2_100_200_400 | contact_x | n/a | n/a | n/a | n/a |
| cfl0.2_100_200_400 | l1_rho_full | n/a | n/a | n/a | n/a |
| cfl0.2_100_200_400 | l1_rho_cellavg | n/a | n/a | n/a | n/a |
| cfl0.2_100_200_400 | l1_rho_plateaus | n/a | n/a | n/a | n/a |
| cfl0.2_100_200_400 | smooth_wave_l1 | n/a | n/a | n/a | n/a |
| order1_100_200_400 | shock_x | n/a | n/a | n/a | n/a |
| order1_100_200_400 | contact_x | n/a | n/a | n/a | n/a |
| order1_100_200_400 | l1_rho_full | n/a | n/a | n/a | n/a |
| order1_100_200_400 | l1_rho_cellavg | n/a | n/a | n/a | n/a |
| order1_100_200_400 | l1_rho_plateaus | n/a | n/a | n/a | n/a |
| order1_100_200_400 | smooth_wave_l1 | n/a | n/a | n/a | n/a |

## C. 导出量（Richardson 外推 / GCI 带 / 对解析真值）

| sweep | 功能量 | 外推(ref) | 外推(用 got 的 p) | Δ | GCI(abs, ref) | GCI(abs, got p) | 细网格相对误差 | 外推相对误差 | GCI 覆盖真值 |
|---|---|---|---|---|---|---|---|---|---|
| main_100_200_400 | shock_x | 0.745493 | 0.745406 | 0.01% | 0.000063 | 0.000170 | 0.03% | 0.03% | ❌ |
| main_100_200_400 | contact_x | 0.630000 | 0.629871 | 0.02% | 0.000435 | 0.000275 | 0.03% | 0.02% | ✅ |
| fine_200_400_800 | shock_x | 0.745342 | 0.745313 | 0.00% | 0.000120 | 0.000156 | 0.02% | 0.01% | ✅ |
| fine_200_400_800 | contact_x | 0.629968 | 0.629864 | 0.02% | 0.000230 | 0.000099 | 0.01% | 0.02% | ✅ |
| combined_100_200_400_800 | shock_x | 0.745342 | 0.745313 | 0.00% | 0.000120 | 0.000156 | 0.02% | 0.01% | ✅ |
| combined_100_200_400_800 | contact_x | 0.629968 | 0.629864 | 0.02% | 0.000230 | 0.000099 | 0.01% | 0.02% | ✅ |
| cfl0.2_100_200_400 | shock_x | 0.745550 | 0.745434 | 0.02% | 0.000090 | 0.000235 | 0.04% | 0.03% | ❌ |
| cfl0.2_100_200_400 | contact_x | 0.630000 | 0.629865 | 0.02% | 0.000214 | 0.000044 | 0.00% | 0.02% | ✅ |
| order1_100_200_400 | shock_x | 0.742492 | 0.745036 | 0.34% | 0.005334 | 0.002154 | 0.20% | 0.38% | ✅ |
| order1_100_200_400 | contact_x | n/a | 0.630509 | n/a | n/a | 0.003740 | n/a | n/a | n/a |

零极限功能量（L1 范数类）只比观测阶：它们的相对 GCI 是消失量的百分比，单三重组 eq.17 比退化为 `r^-p`，两者都不承载信息（见 REPORT ⑤）。

## D. 已入库 v2 工件（`docs/repro-fvm-sod1978-convergence/study.json`）的重诊断

- 工件 sha256 `ee50854fa6248670…`，网格 `[50, 100, 200]`（r=2）
- published 激波位置 log-log 指数 = `-1.2494`；相邻对速率 = `1.8379`
- ref 对同一序列的收敛类 = `oscillatory`（拒绝给出阶：p=NaN）
- 判据翻转：`sublinear_convergence_observed` before=False → after=True
- contact 有效指数 `1.4695`（来源 loglog_lsq），shock 有效指数 `1.8379`（来源 successive_finest）
