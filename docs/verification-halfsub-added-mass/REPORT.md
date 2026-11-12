# TensorFVM · 半浸没球附加质量：奇对称（反对称）延拓法验证

本文件由 `summary.json` 程序化生成（无手抄数字）。复现：
`python -m tensorfvm.verification.halfsub_added_mass`。

## 1. 案例

半径 `a = 1` 的球，球心恰在静水面 `z = 0`（或然性无穷频率 / `Fn -> 0` 极限：
自由面即 Dirichlet 平面 `phi = 0`）。此时只有下半球是湿表面，
自由面条件通过**奇延拓** `Phi(x,y,-z) = -Phi(x,y,z)` 施加：延拓后的势在整个反射体外部调和、
在 `z = 0` 上恒零，因此下半空间的解就是原问题解，代价是上半球的 Neumann 数据不再刚体
（`:func:`tensorfvm.halfsub_added_mass.antisymmetric_normal_velocity`），
而附加质量只在下半球湿表面（真实湿面）积分。

| 量 | 参考值（rho=1, a=1） | 来源 |
|---|---|---|
| 垂荡 `M33` | 1.047198 | 闭式 `(1/3) pi rho a^3` = 0.5000 x 排水质量 |
| 横荡 `M11` | 0.572280 | 奇对称球谐级数（256 项），级数尾部漂移 3.7e-05 |
| 整球/双体值 | 2.094395 | `(2/3) pi rho a^3`（壁面镜像结果） |

## 2. 面元收敛（奇延拓 + 整球网格，Gauss 阶 3，湿面元占一半）

| n_theta x n_phi | 面元 | 湿面元 | M33 | vs 闭式 | M11 | vs 级数 |
|---|---|---|---|---|---|---|
| 12 x 6 | 72 | 36 | 0.906346 | -13.45% | 0.409255 | -28.49% |
| 24 x 12 | 288 | 144 | 1.017381 | -2.85% | 0.509507 | -10.97% |
| 48 x 24 | 1152 | 576 | 1.043289 | -0.37% | 0.546459 | -4.51% |
| 96 x 48 | 4608 | 2304 | 1.048030 | +0.08% | 0.560848 | -2.00% |

观测收敛阶：`p(M33) = 2.93`，`p(M11) = 1.28`。
Richardson 外推：`M33 -> 1.048745` (+0.148%)，`M11 -> 0.570903` (-0.241%)。

横荡阶低于垂荡是物理性质而非缺陷：Dirichlet 自由面与 Neumann 物面在水线相交，
解的局部行为是 `r^{1/2}` 型（水线奇点），面元法在能量范数上因此掉阶。

## 3. 结构恒等式与对照

* `M33(half) / M33(full sphere) = 0.500000000000`（奇延拓的垂荡数据恰是整球刚体垂荡数据，故必须严格等于 1/2）；
* 对照：整球（= 壁面/双体镜像）`M33 = 2.086577`，与真实自由面值之比 `1.9910`
  —— 用错镜像符号在同一个物体上高估 100%。

## 4. 判据

| id | 判据 | 值 | 阈值 | 结论 |
|---|---|---|---|---|
| c1_spectral_selfcheck | |heave series / (1/3 pi rho a^3) - 1| (validates the sway reference) | 1.17684e-14 | 1e-10 | PASS |
| c2_odd_extension_exact | |M33(half) / M33(full sphere) - 1/2| at the finest mesh | 1.83187e-15 | 1e-10 | PASS |
| c3_heave_finest | |M33 / ((1/3) pi rho a^3) - 1| at the finest mesh | 0.000794544 | 0.03 | PASS |
| c4_sway_finest | |M11 / M11(odd-harmonic series) - 1| at the finest mesh | 0.0199771 | 0.03 | PASS |
| c5_heave_richardson | |Richardson(M33) / exact - 1| | 0.00147766 | 0.01 | PASS |
| c6_sway_richardson | |Richardson(M11) / series - 1| | 0.00240744 | 0.01 | PASS |
| c7_decoupling | max off-diagonal |M_ij| / (rho a^3) at the finest mesh | 1.31839e-16 | 1e-10 | PASS |
| c8_surge_sway_symmetry | |M11 - M22| / M11 at the finest mesh | 0 | 1e-10 | PASS |
| c9_order_heave | observed order p of M33 (two finest intervals) | 2.93122 | 1 | PASS |
| c10_order_sway | observed order p of M11 (two finest intervals) | 1.28158 | 1 | PASS |
| c11_wall_control_factor | full-sphere (wall / double-body) M33 / free-surface M33 - 2 (control) | 0.00904757 | 0.05 | PASS |
| c12_sway_tail | relative drift of the sway series between the two largest truncations | 3.70792e-05 | 0.001 | PASS |

**all_pass = True**（耗时 181.8 s）

## 5. 注意点与适用范围

* **水线奇点决定收敛阶**：垂荡观测阶 `p = 2.93`、横荡 `p = 1.28`。
  横荡的 Neumann 数据在水线处非零，Dirichlet 自由面与 Neumann 物面相交产生 `r^(1/2)` 型局部奇点，
  面元法因此掉阶——这是物理性质，不是实现缺陷；判「精度」时必须用 Richardson 外推而非单网格。
* **奇延拓的适用条件**：本法的前提是物体能被水面镜像成自身（球心落在水面上）。
  一般船体（水线不是几何对称面）不能用奇延拓，必须用自由面镜像（`IMAGE_SPECS["free_surface"]`，
  水面镜像源强取反号）或自由面 Green 函数。
* **面元预算**：整球网格中只有一半面元是湿面元，因此相同面元数下本法的有效分辨率是半边网格的一半；
  这也是本单最细网格取 `96 x 48`（4608 面元）的原因。
* **级数参考的精度**：奇对称球谐级数在 rim 处数据有跳跃，部分和以 `1/n` 收敛；两档最大截断间漂移
  `3.7e-05`，故参考值 `M11 = 0.572280` 的不确定度 < 0.01%。
* **未覆盖**：长球/实船无解析锚（面元-级数互证需另立锚点）；绕水面内轴的横摇对球恒为零（D1 已证），
  故本锚族不能用来校核自由面镜像的转动分支；有限频率（带波浪、有航速）自由面未做。

## 6. 误差判定（基准逐项）

| 关键量 | 基准 | 面元值 | 相对误差 | <3% ? |
|---|---|---|---|---|
| 垂荡 `M33` | `(1/3) pi rho a^3 = 1.047198` | 1.048030 | +0.079% | 是 |
| 横荡 `M11` | 奇对称球谐级数 `0.572280` | 0.560848 | -1.998% | 是 |
| 垂荡（Richardson） | 同上 | 1.048745 | +0.148% | 是 |
| 横荡（Richardson） | 同上 | 0.570903 | -0.241% | 是 |
| 奇延拓恒等式 `M33(half)/M33(full)` | `0.5`（精确） | 0.500000000000 | -1.8e-15 | 机器精度 |
| 对照：整球（壁面镜像）`M33` | `2 x` 自由面值 | 2.086577 | `+99.1%` | 故意错 100%（对照基线） |

**结论：全部关键量误差 < 3%（最细网格 0.08% / 2.00%），
Richardson 外推后 < 0.3%。** 判据表全 PASS（`all_pass = True`）。
