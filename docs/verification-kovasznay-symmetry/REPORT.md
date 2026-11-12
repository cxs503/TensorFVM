# CASE-FVM-KOVASZNAY-ORDER —— 收敛阶 1.684 的根因：挤出面上的对称闭合缺失（已修复）

**锚点**: `CASE-FVM-KOVASZNAY-ORDER`（feature ledger, open）· 交付轮 m16130 FAIL 1.684、归属轮 m16396（账本 D10）、**本单 m16864 定位机制并给出修复**。

## 1. 案例介绍

Kovasznay (1948) 二维定常不可压精确解（`Re = 40`，`kappa = 2 pi`，`lambda = -0.963741`）：
`u = 1 - exp(lambda x) cos(2 pi y)`、`v = (lambda/2pi) exp(lambda x) sin(2 pi y)`、`p = (1 - exp(2 lambda x))/2`、**`w = 0` 恒等**。计算域 `[0,1]^3`，网格为对称挤出 `n x n x 1`（单层单元跨越挤出向），速度全域 Dirichlet 取精确面积均值，二阶对流，Picard 松弛 0.7，耦合直接解，收敛容差 1e-8。

**本单要回答的三个问题**：①交付轮 1.684 < 1.8 的格子差从哪来（D10 遗留）；②那 1.684 是求解器不一致，还是**算例边界条件少了一半**；③能否让该算例达到验收门 1.8。

## 2. 怎么做的

1. **逐项残差探针**（`src/tensorfvm/verification/kovasznay_symmetry.py::spanwise_drive_terms`）：
   取收敛态，把所有 `w` 自由度置零，重建冻结通量的动量算子，把挤出向动量行的残差按列块
   （`cell_u/cell_v/cell_w/face_u/face_v/face_w`）与算子家族（对流/粘性/对流源/压力）分解。
2. **解析态缺陷探针**（`analytic_defect`）：把解析离散状态（单元值、精确面均值、`w = 0`、
   精确压力）代入同一组挤出向动量行，看离散方程是否接受精确解。
3. **两种对称闭合对照**：`free` = 交付轮做法（把 `z` 平面的切向分量设成自由未知量）；
   `zero_gradient` = 真正的对称条件（同一批面交给求解器既有的 `zero_gradient_faces` 机制，
   面值绑定到 owner 单元值，离散法向导数恒为 0）。
4. **唯一性对照**：三个不同初值（含 ±0.1 随机扰动）解出的 `w` 必须一致——排除"零空间污染"假设。
5. 全部数字由本目录 `make_report.py` 从 `runs/kovasznay-symmetry/*/level.json` 程序化注入。

## 3. 计算结果

### 3.1 两种对称闭合的误差预算

| closure | n | cells | 收敛 | total L2 | in-plane L2 | spanwise L2 | max abs w | 时间 (s) |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| `free` | 6 | 36 | yes | 0.05634 | 0.05523 | 0.01114 | 0.02414 | 2.64 |
| `free` | 16 | 256 | yes | 0.01257 | 0.01202 | 0.003708 | 0.009414 | 21.2 |
| `free` | 32 | -- | NO | 0.00384 | 0.003432 | 0.001723 | 0.004526 | -- |
| `zero_gradient` | 6 | 36 | yes | 0.05513 | 0.05513 | 3.973e-17 | 1.747e-16 | 2.03 |
| `zero_gradient` | 16 | 256 | yes | 0.01203 | 0.01203 | 1.493e-17 | 1.784e-16 | 9.78 |
| `zero_gradient` | 32 | 1024 | yes | 0.003436 | 0.003436 | 6.779e-18 | 1.602e-16 | 49.1 |

### 3.2 观测阶（相邻网格对）

| closure | 网格对 | total | in-plane | spanwise | max abs w |
|---|---|---:|---:|---:|---:|
| `free` | 6 -> 16 | 1.53 | 1.56 | 1.12 | 0.96 |
| `free` | 16 -> 32 | 1.71 | 1.81 | 1.11 | 1.06 |
| `zero_gradient` | 6 -> 16 | 1.55 | 1.55 | 0.998 | -0.0215 |
| `zero_gradient` | 16 -> 32 | 1.81 | 1.81 | 1.14 | 0.155 |

### 3.3 挤出向动量行的驱动分解（`w` 置零后重建残差）

| closure | n | 残差 max | 对流(全块) | 对流源 | 压力 | 粘性同分量 | **粘性交叉分量** | 交叉/残差 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `free` | 6 | 0.000476 | 0 | 0 | 0 | 0 | **0.000476** | 1.000000 |
| `free` | 16 | 1.96e-05 | 0 | 0 | 0 | 0 | **1.96e-05** | 1.000000 |
| `zero_gradient` | 6 | 4.05e-19 | 0 | 0 | 0 | 0 | **4.05e-19** | 1.000000 |
| `zero_gradient` | 16 | 2.1e-19 | 0 | 0 | 0 | 0 | **2.1e-19** | 1.000000 |

### 3.4 解析态缺陷与挤出面切向偏移

| closure | n | 解析态挤出向行残差 | 挤出面 `du_t/dz`（两点） | 无量纲 | 面值-单元值 max |
|---|---:|---:|---:|---:|---:|
| `free` | 6 | 1.07e-18 | 0.3165 | 0.08797 | 0.1583 |
| `free` | 16 | 5.58e-19 | 0.1404 | 0.03596 | 0.07019 |
| `zero_gradient` | 6 | 1.07e-18 | 0 | 0 | 0 |
| `zero_gradient` | 16 | 5.58e-19 | 0 | 0 | 0 |

### 3.5 预声明判据

| 判据 | 内容 | 实测 | 目标 | 判定 |
|---|---|---:|---|---|
| C1 | zero-gradient symmetry closure pins the spanwise velocity to zero | 1.784e-16 | < 1e-10 | PASS |
| C2 | free tangential z-planes leave a spurious spanwise velocity | 0.02414 | > 1e-6 (negative control is not vacuous) | PASS |
| C3 | free planes settle on a tangential value that differs from their owner cell | 0.3165 | > 0 | PASS |
| C4 | spurious spanwise budget is first order in h | 1.122 | < 1.5 | PASS |
| C5 | the drive of the spanwise rows is the viscous cross-derivative block | 1 | > 0.99 of the residual | PASS |
| C6 | advection, pressure and same-component blocks contribute nothing | 0 | < 1e-9 of the residual | PASS |
| C7 | the assembled spanwise rows accept the analytic state (consistent equations) | 5.582e-19 | < 1e-9 | PASS |
| C8 | velocity L2 order with the symmetry closure meets the anchor gate | 1.808 | >= 1.8 | PASS |

### 3.6 唯一性对照（自由闭合，n=6）

| 初值扰动幅值 | max abs w |
|---|---:|
| 0.0 | 0.0241350829847 |
| 0.1 | 0.0241350833383 |
| -0.1 | 0.024135082791 |

三组初值解出的 `w` 向量最大逐点差 `1.59e-09`（相对 `6.6e-08`）——`w` 由离散方程唯一确定，排除零空间污染假设。

## 4. 结果图

`fig_kovasznay_symmetry.png`：左=三种预算随 h 的收敛（free 的 spanwise 预算是一阶，zero-gradient 直接压到舍入）；中=max |w|（free 一阶下降 vs zero-gradient 机器零）；右=挤出向动量行的驱动分解（唯一非零项是粘性交叉分量）。

## 5. 注意点

* **对称面不等于自由面**。`kovasznay_solver(..., symmetry_extrusion=True)` 只做了 `w = 0`，
  没有施加 `du_t/dz = 0`，于是 `z` 平面上的 `u, v` 成了"自由（traction-free）"未知量：
  实测它们与 owner 单元值差到 `max_abs_plane_offset`（见 3.4），两点法向导数因此看到
  `0.14–0.32` 的假剪切，粘性应力 `mu(du_i/dz + dw/dx_i)` 在面内面上出现 O(h) 源项。
* **不是方程不一致**：解析态代入同一批挤出向动量行，残差 `~1e-18`（3.4 列），
  说明离散方程本身接受精确解；问题在边值闭合，不在格式。
* **不是零空间污染**：三个初值（含 ±0.1 随机扰动）解出的 `w` 位级一致，
  所以 `w` 是被唯一确定的伪速度，归属轮"被离散耦合驱动"的判断成立，但驱动源在粘性交叉项。
* **对流/压力/同分量粘性贡献恒为 0**（3.3），因此任何"加强迎风/限制器"的路子都治不了它；
  修复必须落在对称面的闭合上。
* **判据 C2–C7 是负对照**（要求 `free` 闭合确实有病），C1/C8 是验收（要求修复确实有效）。
* 本单没有改 `kovasznay_solver` 的默认行为（保留交付轮数字可复现），只把闭合做成参数；
  `symmetry_closure` 默认 `free` 是为历史口径，**新算例应显式用 `zero_gradient`**。

## 6. 误差判定（关键量 vs 基准，逐项显式回答）

| 关键量 | 基准 | 实测 | 相对误差 | <3% ? |
|---|---|---:|---:|---|
| 速度场 total L2（zero-gradient，最粗 n=6） | Kovasznay 解析解 | 0.05513 | 5.51% | 否（粗网格离散误差，非模型误差；见下行收敛阶） |
| 速度场 total L2（zero-gradient，n=32） | 同上 | 0.003436 | 0.34% | 否（同上，二阶收敛会把误差带进 3%） |
| **收敛阶 16->32（zero-gradient）** | 理论二阶 / 验收门 1.8 | 1.808 | 距二阶 9.6% | 是（门 1.8 达标） |
| 挤出向速度 max abs w（zero-gradient，n=32） | 精确值 0 | 1.6e-16 | 舍入级 | 是 |

**结论一句话**：交付轮的 honest FAIL（1.684）根因是**算例的挤出面只施加了 `w = 0`、缺 `du_t/dz = 0`**；补上对称闭合后，挤出向预算降到机器零，速度 total L2 的 16->32 阶 = 1.808 ≥ 1.8，锚点验收条款达成。

