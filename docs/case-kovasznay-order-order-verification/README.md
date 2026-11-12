# CASE-FVM-KOVASZNAY-ORDER —— Kovasznay (1948) 精确解的网格收敛阶验证

> 判据（总账原文）：Kovasznay 1948 精确解网格收敛阶验证，**L2 收敛阶 ≥ 1.8（名义二阶）**，
> 指标 = 32/64/128 三套网格上的 log-log 斜率；ref DOI 10.1017/S0305004100023999。
> 本目录：调研与方法选型见 [RESEARCH.md](RESEARCH.md)，由程序注入的数字表与判据逐条见
> [RESULTS.md](RESULTS.md)（由 `scripts/summarize_kovasznay_order.py` 生成，零手抄），
> 收敛曲线与误差云图见 [convergence.png](convergence.png)，原始逐网格记录见
> `runs/kovasznay-order/n*/cases.json`（含源码 sha256 快照与自校验）。

## 1. 问题设定与基准出处

二维无体力稳态不可压 Navier–Stokes 的精确解（Kovasznay 1948，*Proc. Cambridge Philos. Soc.*
44:58–62，doi:10.1017/S0305004100023999），单位域 [0,1]²，ρ=1，ν=1/Re：

```
κ = 2π,  λ = −2κ² / (Re + sqrt(Re² + 4κ²)),  λ² − Re·λ = 4π²
u = 1 − e^{λx} cos(κy),   v = (λ/κ) e^{λx} sin(κy),   w = 0,   p = ½(1 − e^{2λx})
```

该恒等式使「对流项 + 压力梯度 − ν∇²u」逐点严格为零，因此**不需要**从离散算子反算体积力
（`body_force=0`）——避开了"源项自洽"式的自证。解析式与梯度见
`src/tensorfvm/verification/native_hybrid_ns.py:kovasznay_fields`。

这是阶验证（order verification）的标准用途：有精确解、光滑、非线性对流非零、无奇点，
误差随 h 单调衰减，适合度量名义二阶格式的观测阶。

## 2. 方法与实现

| 项 | 取值 / 出处 |
|---|---|
| 求解器 | `NativeHybridNavierStokes3D`（单元中心混合有限体积，面通量守恒），`src/tensorfvm/native_hybrid_ns3d.py` |
| 网格族 | 解析单位方形的 n×n×1 对称挤出（`triangulate_faces=True`，六面体劈棱柱），h = 1/n |
| 挤出方向 | 两个 z 面按对称面处理（u,v 自由、w 由解析面平均固定为 0），单层单元 |
| 速度边界 | 全部外边界按解析速度的**面积平均**给定（`exact_boundary_velocity_means`），面通量全局平衡 |
| 压力定标 | 首单元压力锚定到解析值（`gauge_value`）；另报去均值范数以剥离常数零空间分量 |
| 对流 | `convection_order=2`（二阶迎风/中心混合） |
| 耦合解 | `coupled_linear_method='reused_lu'`（Picard 每步复用稀疏 LU） |
| 非线性 | Picard 外迭代，松弛 0.7，tol 1e-8，max 120 步 |
| 误差范数 | 体积加权相对 L2（`volume_weighted_error`），被比较量为形心处的解析值 |
| 阶估计 | 成对 `p = log(e_coarse/e_fine)/log(h_coarse/h_fine)` + 全网格最小二乘斜率（Roache 1998；Celik et al. 2008 ASME V&V 20 口径） |

关键代码指针：

- `scripts/verify_kovasznay_order.py` —— 单网格一个进程；落 `cases.json`（误差、收敛历史、
  守恒残差 `energy_balance_error`/`interface_traction_defect_max`/`global_momentum_imbalance`/
  `divergence_max`、采样偏差）+ `fields.npz` + `sources.json`（源码 sha256，跑完 `verify_sources()` 自校验）。
- `scripts/summarize_kovasznay_order.py` —— 读全部 `cases.json`，算成对阶与最小二乘斜率，
  输出 `orders.json`、`RESULTS.md`、`convergence.png`；判据阈值 1.8 在**看结果前**由命令行给定。

采样一致性控制：单元中心 FV 未知量是**单元平均**，而比较量是形心点值，二者相差 O(h²)。
脚本在同一网格上用 6×6 子采样测出该偏差并随结果一起存档，用来排除"斜率是采样假象"。

## 3. 参数表（无量纲）

| 参数 | 值 | 说明 |
|---|---|---|
| Re | 40 | 单位域、U∞≈1、单位长度，Kovasznay 的标准取值 |
| κ | 2π | 横向波数（解析解定义） |
| λ | 见 RESULTS.md（由 Re,κ 程序化算出） | 流向衰减率 |
| ν | 1/Re = 0.025 | 运动黏度，ρ=1 |
| h | 1/n，n ∈ {16, 32, 64}（+128 见 §6） | 网格间距，加密比恒为 2 |
| CFL | n/a | 稳态解，无时间步 |
| 松弛 / tol | 0.7 / 1e-8 | Picard 外迭代 |

## 4/5. 结果、收敛性与判定

见 [RESULTS.md](RESULTS.md)（表格与判据逐条由脚本从 `cases.json` 直接生成）与
[convergence.png](convergence.png)（左：log-log 误差-网格；中/右：最细网格上 u_x 与 p 的误差云图）。
参数敏感性：同一算例另在 8×8×1 与 16×16×1 上重复（同一脚本、同一阈值），用于确认斜率在
非渐近区间的偏移方向，原始记录在同目录 `n16/cases.json`。

## 6. 复现与工件

```bash
# 每套网格一个作业（2D 稀疏 LU 的代价 ≈ O(n^3)，128 网格约是 32 网格的 100 倍）
PYTHONPATH=src python3 scripts/verify_kovasznay_order.py --output runs/kovasznay-order/n32  --n 32
PYTHONPATH=src python3 scripts/verify_kovasznay_order.py --output runs/kovasznay-order/n64  --n 64
PYTHONPATH=src python3 scripts/verify_kovasznay_order.py --output runs/kovasznay-order/n128 --n 128
# 合并（秒级）
PYTHONPATH=src python3 scripts/summarize_kovasznay_order.py \
    --runs runs/kovasznay-order/n32 runs/kovasznay-order/n64 runs/kovasznay-order/n128 \
    --output runs/kovasznay-order/summary \
    --markdown docs/case-kovasznay-order-order-verification/RESULTS.md \
    --figure   docs/case-kovasznay-order-order-verification/convergence.png
```

工件：`runs/kovasznay-order/n{16,32,64,128}/`（`cases.json`、`summary.json`、`sources.json`、
`environment.json`、`n*-fields.npz`、`n*-{outer,inner,linear}.json` 收敛历史），
合并结果 `runs/kovasznay-order/summary/orders.json`。

## 7. 局限（诚实边界）

- 只验**层流稳态空间离散**的二阶性；不含湍流模型、时间推进阶、动网格、复杂几何。
- 挤出方向单层 + 对称面，属"平面流的 3D 退化算例"；`max_abs_z_velocity` 随网格衰减但不为零，
  已在 `cases.json` 中逐网格记录，不当作严格的完整三维验证。
- 压力在单元中心同位布局、全 Dirichlet 速度边界下近壁局部只有一阶（Ferziger & Perić §7），
  因此压力斜率单列为次指标，不参与 1.8 阈值判定；定标（fixed gauge）与去均值两种范数分别报告。
- 阶 ≥1.8 只说明光滑解上达到名义二阶，不能外推到高 Re / 激波 / 复杂几何的精度。