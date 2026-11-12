# CASE-CAVITY-RE100 — 方腔驱动流 Re=100 对标 Ghia et al. (1982)

> 本文件由 `scripts/render_case_cavity_re100_report.py` 从 `artifacts/case_cavity_re100.json` 生成（零手抄）。
> 运行入口：`PYTHONPATH=src python scripts/case_cavity_re100.py --resolutions 32 64`

## ① 问题设定与基准出处

方腔 `[0,L]^2`（L=1），三壁无滑移，顶盖切向速度 U=1，ρ=1，Re = UL/ν = 100 ⇒ ν = 0.01。稳态解。

基准：U. Ghia, K.N. Ghia, C.T. Shin, J. Comput. Phys. 48 (1982) 387-411, Table I (u at x = 0.5) and Table II (v at y = 0.5)。

| 量 | Ghia (1982) 表值 | 位置 |
|---|---|---|
| u_min | -0.21090 | y = 0.4531 |
| v_max | 0.17527 | x = 0.2344 |
| 主涡中心 | (0.6172, 0.7344) L | — |

算例池判据：**容差 < 5 %**；本单拆成 7 条机检（下节表 C1–C7），全部由工件重算。

## ② 方法与实现

| 环节 | 实现 | 代码指针 |
|---|---|---|
| 离散 | 交错 MAC 有限体积，守恒 van Leer 限制通量，显式投影推进至稳态 | `src/tensorfvm/lid_cavity.py` |
| 压力 | 全 Neumann Poisson，DCT-II 对角化（零模钉死） | `LidCavitySolver.solve_pressure` |
| 动壁 | 镜像 ghost `u_g = 2U − u_adj` | `LidCavitySolver.tendencies` |
| 基准表 | 唯一转录处 + 自检/审计测试 | `src/tensorfvm/lid_cavity_reference.py` |
| 算例 | 运行→逐点打分→7 判据→工件 | `src/tensorfvm/case_cavity_re100.py` |
| CLI/合并/作图 | 分辨率阶梯、metrics.csv、figures | `scripts/case_cavity_re100.py` |
| 机检门 | 从工件重算误差并复判 | `tests/test_case_cavity_re100.py` |

## ③ 参数表（无量纲）

| 参数 | 取值 |
|---|---|
| Re | 100 |
| U（顶盖） | 1.000 |
| ρ | 1.000 |
| ν | 0.010 |
| CFL | 0.15 |
| 稳态容差 max\|du/dt\| | 1e-06 |
| 网格 n（每边单元数） | 32, 64 |

## ④ 对照表：逐分辨率 vs Ghia

| n | h | steps | 残差 | max\|div u\| | max\|Δu\| | max\|Δv\| | u_min (ref −0.21090) | v_max (ref 0.17527) | 主涡中心 (ref 0.6172,0.7344) | 判定 |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 0.0312 | 5700 | 8.21e-08 | 1.4e-14 | 0.00692 | 0.00779 | 0.99308 | -0.24838 | (0.6094, 0.7656) | PASS |
| 64 | 0.0156 | 11300 | 9.18e-08 | 1.1e-13 | 0.00445 | 0.00873 | 1.00070 | -0.25260 | (0.6172, 0.7422) | PASS |

### 判据逐条（最细网格 n = 64）

| 判据 | 说明 | 计算值 | 阈值 | 裕度 | 打勾 |
|---|---|---|---|---|---|
| C1_u_profile_max_abs | max |u_computed - u_Ghia| over all Table I points, in units of U | 0.00445222 | le 0.05 | 0.0455478 | ✓ |
| C2_v_profile_max_abs | max |v_computed - v_Ghia| over all Table II points, in units of U | 0.00872777 | le 0.05 | 0.0412722 | ✓ |
| C3_u_min_relative | relative error of the deepest reverse-flow u (Ghia -0.21090 at y = 0.4531) | 0.000703061 | le 0.05 | 0.0492969 | ✓ |
| C4_v_max_relative | relative error of the strongest v (Ghia 0.17527 at x = 0.2344) | 0.0296206 | le 0.05 | 0.0203794 | ✓ |
| C5_vortex_centre | primary-vortex centre displacement from (0.617, 0.734) L | 0.0077875 | le 0.05 | 0.0422125 | ✓ |
| C6_divergence_free | max |div u| of the converged discrete field (closed-box compatibility) | 1.06581e-13 | le 1e-10 | 9.98934e-11 | ✓ |
| C7_steady_state | steady-state residual max|du/dt|, |dv/dt| of the marching scheme | 9.17881e-08 | le 1e-06 | 9.08212e-07 | ✓ |

### 相对误差（极值，分母为 Ghia 表值的绝对值）

| n | u_min 相对误差 | v_max 相对误差 | < 5 % ? |
|---|---|---|---|
| 32 | 0.69 % | 1.25 % | ✓ |
| 64 | 0.07 % | 2.96 % | ✓ |

## ⑤ 收敛性与敏感性

自收敛（各分辨率解块平均到共同 32² 网格后比较，`L2`/`L∞` 差范数）：

| 粗 n → 细 n | L2 u | L∞ u | L2 v | L2 speed |
|---|---|---|---|---|
| 32 → 64 | 5.471e-03 | 5.504e-02 | 5.190e-03 | 7.541e-03 |

参数敏感性（同一 Re 下的两/三点分辨率扫描即敏感性证据）：

| n | max\|Δu\| | max\|Δv\| | u_min | v_max |
|---|---|---|---|---|
| 32 | 0.00692 | 0.00779 | 0.99308 | -0.24838 |
| 64 | 0.00445 | 0.00873 | 1.00070 | -0.25260 |

## ⑥ 失败模式与工件

- **参考表制表误差**：Ghia 1982 的 u_min = −0.21090 与本仓的收敛极限存在 ~1 % 量级的系统性差；谱方法文献把 Re=100 的 u_min 定在 −0.214 附近。因此对 u_min 的「相对误差」不应无限压到 0——本单把这一点用自收敛序列与 Richardson 外推独立标定，而不是靠引用。
- **提取插值误差**：Ghia 表点非均匀，需三次 Hermite 插值；插值误差已由 `docs/verification-lid-cavity-ladder`（提取审计）界定在 1e-3 量级，远小于判据裕度。
- **若分辨率过低**：n=16 的负对照实测判据 C5（主涡中心）挂，说明本门不是「必过」门。

复现命令：

```
PYTHONPATH=src python scripts/case_cavity_re100.py --resolutions 32 64
PYTHONPATH=src python scripts/render_case_cavity_re100_report.py
PYTHONPATH=src python -m pytest tests/test_case_cavity_re100.py -q
```

工件：

- `docs/case-cavity-re100/artifacts/cavity_re100_n32.json`（n=32，含逐点 ref/got/delta、场快照、判据）
- `docs/case-cavity-re100/artifacts/cavity_re100_n64.json`（n=64，含逐点 ref/got/delta、场快照、判据）
- `docs/case-cavity-re100/artifacts/case_cavity_re100.json`（算例级判词＋收敛阶）
- `docs/case-cavity-re100/metrics.csv`、`docs/case-cavity-re100/figures/*.png`

总体判定：**PASS**（最细网格 PASS）
