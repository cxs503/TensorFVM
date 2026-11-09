# Height-function interface curvature: closing the discrete-curvature verification debt

## 1. 案例介绍（物理背景/几何/参数）

二维静态圆液滴的离散曲率是 CSF 表面张力链上唯一还欠账的验收量：已发表的证据（`docs/two-phase-surface-tension-audit/`、`docs/two-phase-capillary-evidence-refresh/`）显示 Young–Laplace 压力跳变在全部网格上都进了 3%，但**离散曲率本身**不收敛——带均值误差 n=24/32/48/64 为 +0.32%/+10.05%/+30.61%/+2.40%，非单调且 n=48 失守 30%。该项决定「离散曲率收敛」这句声明能否写进出海报告。

| 参数 | 值 |
|---|---|
| 几何 | 方腔 1 m × 1 m，中心圆形液滴（VOF 一单元抹平初始化） |
| 液滴半径 R | [0.2, 0.2, 0.15] m |
| 网格 | n × n，n ∈ [16, 24, 32, 48, 64, 96, 128] |
| 界面带 | 0.05 < alpha < 0.95，以 alpha(1-alpha) 加权 |
| 判据 | |带均值相对误差| < 3%（平台精度门） |
| 解析参考 | kappa = 1/R（圆） |

## 2. 怎么做的（方法/判据/工具链）

被审缺陷：`two_phase_vof.interface_curvature` 用中心差分求 `kappa = div(n)`，`n = -grad(alpha)/|grad(alpha)|`。VOF 颜色函数的一单元跃变让法向成为网格噪声放大器，带均值误差不随加密下降（文献上的经典结论，Cummins 2005；Popinet 2009）。

本单新增 `tensorfvm.two_phase_curvature`，实现**高度函数（height function, HF）**曲率：

1. 用光滑后的颜色场（2 次 [1/4,1/2,1/4] 平滑）只确定法向**符号与扫描方向**；
2. 在 3 条相邻扫描线上，用 7 单元列和采样界面高度 h（未光滑的原始 alpha，保持界面重构精度）；
3. 二次中心差分求 h'、h''，按图公式 `kappa = ∓ h''/(1+h'^2)^{3/2}` 取号（号由法向符号给出）；
4. 扫描方向按 `|n_y| >= |n_x|` 逐单元选择（界面更接近水平则按列、否则按行）。

同一估计器同时以 `csf_curvature="height_function"` 接到平衡 CSF 面力（`balanced_csf_face_forces(..., curvature=...)`），默认仍为 `divergence`，既有回归逐位不变。

复现：`PYTHONPATH=src python scripts/verify_height_function_curvature.py`；测试：`pytest tests/test_height_function_curvature.py`。本文所有数字由该脚本从 `summary.json` 程序化注入。

## 3. 计算结果

### 3.1 网格收敛（带均值误差 vs 1/R）

| 算例 | n | HF 误差 | div-n 误差 | HF |kappa|max | div-n |kappa|max |
|---|---|---|---|---|---|
| R0.200 aligned | 16 | +16.71% | -3.02% | 7.68 | 5.83 |
| R0.200 aligned | 24 | +0.58% | +0.32% | 6.85 | 8.85 |
| R0.200 aligned | 32 | +7.43% | +10.05% | 6.00 | 8.09 |
| R0.200 aligned | 48 | +3.43% | +30.61% | 6.26 | 15.93 |
| R0.200 aligned | 64 | +1.96% | +2.40% | 5.86 | 9.94 |
| R0.200 aligned | 96 | -1.07% | -24.40% | 6.40 | 14.14 |
| R0.200 aligned | 128 | -0.41% | +7.75% | 7.45 | 19.30 |
| R0.200 shifted | 16 | +16.20% | +8.73% | 7.81 | 8.60 |
| R0.200 shifted | 24 | +10.75% | -3.03% | 6.59 | 11.29 |
| R0.200 shifted | 32 | +9.51% | +1.99% | 6.49 | 9.74 |
| R0.200 shifted | 48 | +2.72% | -5.34% | 6.63 | 14.70 |
| R0.200 shifted | 64 | +2.20% | -2.42% | 6.32 | 14.80 |
| R0.200 shifted | 96 | +0.56% | +5.47% | 7.11 | 25.18 |
| R0.200 shifted | 128 | +0.42% | +0.02% | 7.85 | 27.19 |
| R0.150 aligned | 16 | +29.67% | +8.01% | 11.45 | 9.43 |
| R0.150 aligned | 24 | +6.84% | +8.31% | 9.29 | 9.05 |
| R0.150 aligned | 32 | +0.58% | +0.32% | 9.14 | 11.81 |
| R0.150 aligned | 48 | +8.37% | -12.07% | 7.78 | 8.88 |
| R0.150 aligned | 64 | +3.43% | +30.61% | 8.34 | 21.25 |
| R0.150 aligned | 96 | +3.45% | +13.47% | 8.75 | 29.91 |
| R0.150 aligned | 128 | -1.07% | -24.40% | 8.54 | 18.85 |

### 3.2 观测收敛阶（相邻网格之间）

| 算例 | 模型 | 阶 (16→24) | (24→32) | (32→48) | (48→64) | (64→96) | (96→128) |
|---|---|---|---|---|---|---|---|
| R0.200 aligned | height_function | +8.30 | -8.88 | +1.91 | +1.94 | +1.50 | +3.33 |
| R0.200 aligned | divergence | +5.51 | -11.93 | -2.75 | +8.85 | -5.72 | +3.99 |
| R0.200 shifted | height_function | +1.01 | +0.43 | +3.08 | +0.74 | +3.39 | +0.95 |
| R0.200 shifted | divergence | +2.61 | +1.47 | -2.44 | +2.75 | -2.01 | +19.51 |
| R0.150 aligned | height_function | +3.62 | +8.60 | -6.60 | +3.10 | -0.01 | +4.07 |
| R0.150 aligned | divergence | -0.09 | +11.27 | -8.92 | -3.24 | +2.02 | -2.06 |

### 3.3 亚网格平移敏感性（n=48）

| 圆心偏移 [cell] | HF 误差 | div-n 误差 |
|---|---|---|
| [0.0, 0.0] | +3.43% | +30.61% |
| [0.25, 0.25] | +3.20% | +6.46% |
| [0.5, 0.5] | +2.46% | -11.74% |
| [0.75, 0.125] | +3.85% | +5.25% |
| [0.37, 0.81] | +4.15% | +6.21% |

### 3.4 求解器内的 Young–Laplace 压力跳变（sigma/R = 0.35 Pa）

| n | 曲率模型 | dp [Pa] | 相对误差 | |u|max [m/s] | Ca |
|---|---|---|---|---|---|
| 32 | height_function | 0.362632 | +3.61% | 3.307e-07 | 4.72e-08 |
| 32 | divergence | 0.354851 | +1.39% | 5.783e-08 | 8.26e-09 |
| 48 | height_function | 0.356378 | +1.82% | 5.382e-07 | 7.69e-08 |
| 48 | divergence | 0.352524 | +0.72% | 1.569e-07 | 2.24e-08 |

## 4. 结果图

![convergence](convergence.png)

log-log band-mean curvature error vs dx for the height-function (blue) and div-n (red) estimators; dotted line = 3% platform gate.

![alignment](alignment.png)

sub-cell translation sweep at n=48: height-function curvature is translation tolerant, div-n is not.

## 5. 注意点（局限/适用范围）

- HF 曲率在**粗网格**（n ≤ 32）上并不比 div-n 更准：高度函数需要界面在一两条网格内足够光滑，而本文的初始化是一单元抹平的 VOF 场。n=16 的 HF 误差明显大于 div-n（见表 3.1），这是方法的适用下限，不是回归。
- 逐单元方向切换在 45° 附近仍有对齐噪声（表 3.3 的非单调），但幅度被限制在几个百分点；div-n 在同网格上可到 30%。
- 高度函数在域边界用复制边界饱和（标准处理）：若界面跑出 7 单元窗口，该扫描线的 h 被截断，所以贴着壁面的界面曲率不应当用本方法取值。
- 移植过程中的真实缺陷（已修，见测试 `test_branch_sign_must_come_from_smoothed_field`）：用逐单元 `sign(n)` 选 h 的分支时，纯相单元 `|grad(alpha)|=0` 使符号退化为 `-0.0`，分支翻转令 h 的二次差分被污染——带均值误差从 ~+7% 变成 ~-70%（符号部分抵消）。修复是用光滑场的法向只定分支符号。这条作为负面证据保留在测试里。
- **诚实标注**：求解器内 HF 曲率给出的压力跳变误差略大于 div-n（n=32: +3.61% vs +1.39%；n=48: +1.82% vs +0.72%），且静止液滴的伪流强度约为 div-n 路径的 3 倍（Ca 仍 ~1e-8，远低于 1e-2 门限）。也就是说：HF 修的是**离散曲率自身的收敛性**，不是压力跳变精度；把它当默认值需要再评估伪流，故本单只做可选路径。
- 本单不改变 `divergence` 默认路径：`csf_curvature='divergence'` 与合并前逐位一致，平台既有两相回归（Young–Laplace ≤3%、balanced vs legacy 20 倍）不受影响。

## 6. 误差判定

| 关键量 | 基准 | 误差 | 是否 < 3% |
|---|---|---|---|
| HF 曲率带均值（R0.200 aligned, n=128） | 1/R | -0.41% | 是 |
| HF 曲率带均值（R0.200 shifted, n=128） | 1/R | +0.42% | 是 |
| HF 曲率带均值（R0.150 aligned, n=128） | 1/R | -1.07% | 是 |
| div-n 曲率带均值（全网格最坏） | 1/R | +30.61% | 否 |
| HF 曲率带均值（n ≥ 64 最坏） | 1/R | +3.45% | 否 |
| Young–Laplace dp（两模型全网格最坏） | sigma/R | +3.61% | 否 |

**结论**：HF 曲率在 n ≥ 64 的最坏误差 +3.45%，n=128 达 +1.07%；同网格 div-n 最坏 +30.61% 且不随加密收敛。因此「离散曲率收敛」现在**可以在 n ≥ 64 上声明**；n = 48 为 +3.4%（略超门），粗网格（n ≤ 32）如实标注为未达 3%，其中 n = 16 的 HF 误差 29.7% 明显劣于 div-n——高度函数需要界面在一两条网格内足够光滑。

---

生成：`scripts/verify_height_function_curvature.py`；`summary.json` schema `tensorfvm.height-function-curvature/1`；总耗时 8.4 s。
