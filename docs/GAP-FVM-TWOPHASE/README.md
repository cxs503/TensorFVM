# GAP-FVM-TWOPHASE -- VOF interface capture (Zalesak + dam break)

**Verdict: 8/10 gates PASS.**
The Zalesak criterion the gap anchor names (thresholded geometric error < 5 % at
the declared resolution) **passes**: 3.95 %
at N = 128 (observed order
1.23).  The dam-break comparison against
Martin & Moyce (1952) **fails honestly** (front
98.8 %, height
84.5 % against a 15 % gate); cause and
round-2 plan in section 5.  Every number below is rendered from
`artifacts/gap_twophase.json`
(3.11.6, torch 2.7.1).

## 1. 案例介绍（物理背景 / 几何 / 参数）

**Zalesak 旋转缺口盘**（Zalesak 1979, *J. Comput. Phys.* **31**, 335）—— 界面
捕获的标准形状精度算例：单位方域，圆心 (0.5, 0.75)、半径 0.15，竖直缺口宽
0.10 自圆盘上沿切至中心；刚体旋转场 u = -ω(y-1/2)、v = ω(x-1/2)，ω = 2π
（一圈 = 单位时间）。**一圈刚体旋转的精确解就是初始场**，因此几何误差可直接
对解析形状测量，不需要任何参考数值解。

**重力溃坝**（Martin & Moyce 1952, *Phil. Trans. R. Soc. A* **244**, 312）——
二维水柱瞬泄：半宽 a = 0.05715 m、高宽比 2（h0 = 0.1143 m），计算域
0.6 × 0.2 m，ρ_l/ρ_g = 1000，μ_l = 1e-3 Pa·s，g = 9.81 m/s²，自由面 void
模型；无量纲化 t* = t√(g/a)，x* = (x_f − a)/a，h* = h/h0（与实验曲线一致）。

溃坝网格参数（来自各 leg 的 `record`）：

| scheme | 网格 | dx [m] | dy [m] | 步数 |
|---|---|---|---|---|
| plic | 120x40 | 0.005 | 0.005 | 169 |
| plic | 240x80 | 0.0025 | 0.0025 | 343 |

Zalesak 网格：N = 64, 128
（CFL 0.4，一圈，步数见各 leg `steps`）。

## 2. 怎么做的（方法 / 流程 / 关键设置 / 工具链）

* **实现面（复用已登记的库，不另造轮子）**：几何 PLIC 重构与子网格通量
  (`tensorfvm.vof_plic`)、代数 CICSAM/HRIC 输运
  (`tensorfvm.twophase_bench.cicsam`)、形状度量（阈值化对称差 / marching-squares
  周长，`twophase_bench.metrics2d`）、溃坝求解器
  （`vof_phase0.solver.Phase0TwoPhaseSolver` + `PLICVOF`，变量密度投影 + void
  自由面）与 Martin & Moyce 数字化锚 (`vof_phase0.anchors`)。
* **新增面（本单交付）**：把散件组织成**可运行、机器判据**的验收面
  `tensorfvm.twophase_gap`。`run_zalesak_leg` / `run_dambreak_leg` 产出带
  `reference`/got/误差的 leg JSON；`evaluate_gates` 是只依赖 leg 工件的纯函数
  （报告数字不可能手抄）；`render_report` / `write_figures` 由 JSON/NPZ 渲染
  表格与图；`python -m tensorfvm.twophase_gap` 与 console script
  `tensorfvm-twophase-gap` 是一键复现入口。
* **集成调用**：生产求解器
  `Phase0TwoPhaseSolver.verify_interface_transport()` 直接调用 Zalesak leg，
  用**求解器实际装载的 VOF 方案**（PLIC 或代数）做运行前自检；由
  `tests/test_twophase_gap.py::test_solver_self_check_uses_installed_scheme` 覆盖。
* **度量定义**（避免"守恒即正确"的盲区）：
  e_geo = |{α>1/2} XOR {exact>1/2}| / |{exact>1/2}|；
  质量误差 = |A(T)−A(0)|/A(0)（A = Σα dx dy）；
  周长变化 = |P(T)−P(0)|/P(0)（α=1/2 等值线）。
* **判据先于结果**：`GATES`（Z1–Z5, D1–D5）硬编码在驱动里并写进 JSON；
  比较窗口 0.3 < t* ≤ 4.0、声明分辨率 N=128 均在求解前确定。
* **工具链**：PyTorch float64（CPU）+ NumPy + matplotlib；N=256 与 240×80
  leg 递交集群队列（loong64），其余本机执行。

## 3. 计算结果

### 3.1 判据表（机器裁定，来自 gap_twophase.json）

| id | case | metric | limit | got | verdict |
|---|---|---|---|---|---|
| Z1 | zalesak | geometric_error | <= 0.05 | 0.03955 | PASS |
| Z2 | zalesak | mass_drift_relative | <= 0.001 | 6.423e-16 | PASS |
| Z3 | zalesak | perimeter_error | <= 0.1 | 0.0643 | PASS |
| Z4 | zalesak | bound_violation | <= 1e-12 | 0 | PASS |
| Z5 | zalesak | convergence_order | >= 0.8 | 1.234 | PASS |
| D1 | dambreak | front_max_rel_error | <= 0.15 | 0.9882 | FAIL |
| D2 | dambreak | height_max_rel_error | <= 0.15 | 0.845 | FAIL |
| D3 | dambreak | volume_drift | <= 0.01 | 4.114e-16 | PASS |
| D4 | dambreak | bound_violation | <= 1e-10 | 0 | PASS |
| D5 | dambreak | control_max_abs_change | <= 1e-14 | 0 | PASS |

### 3.2 Zalesak 形状保真度（一圈刚体旋转）

| scheme | N | 几何误差 | 质量漂移 | 周长变化 | 峰值保持 | 解析周长 | 初始离散周长 |
|---|---|---|---|---|---|---|---|
| cicsam | 64 | 11.05 % | 5.42e-14 | 12.02 % | 1.0000 | 1.30426 | 1.34283 |
| cicsam | 128 | 4.38 % | 4.82e-16 | 7.04 % | 1.0000 | 1.30426 | 1.34819 |
| plic | 64 | 9.30 % | 8.76e-15 | 11.29 % | 1.0000 | 1.30426 | 1.34283 |
| plic | 128 | 3.95 % | 6.42e-16 | 6.43 % | 1.0000 | 1.30426 | 1.34819 |

解析参考：缺口盘面积 0.041251，周长
1.304261。几何误差收敛阶
**p = 1.23**
（N = 64 → 128：
9.30 % →
3.95 %）。N = 64 的
9.3 % **不满足** 5 %——这是分辨率
测量结果，也是判据必须绑定声明分辨率的原因。

### 3.3 溃坝 vs Martin & Moyce (1952)

| scheme | 网格 | front 误差 | height 误差 | 体积漂移 | 零重力控制 | 步数 |
|---|---|---|---|---|---|---|
| plic | 120x40 | 98.8 % | 84.5 % | 4.11e-16 | 0.0e+00 | 169 |
| plic | 240x80 | 84.8 % | 82.6 % | 1.31e-16 | 0.0e+00 | 343 |

网格加倍（120x40 → 240x80）把前锋误差从 98.8 % 降到 84.8 %，柱高误差 84.5 % → 82.6 %：误差随分辨率下降很慢，说明主因不是网格不足，而是方案本身（一阶动量对流 / void 自由面）对涌潮前锋的结构性滞后。

## 4. 结果图

* `artifacts/figures/fig1_zalesak_fields.png` —— 精确场 / PLIC N=128 捕获场 /
  差值场（缺口角点与 1/2 等值线钝化可见）。
* `artifacts/figures/fig2_zalesak_geometric_error.png` —— 几何误差 vs 分辨率
  （双对数）+ 5 % 判据线 + N=128 声明线。
* `artifacts/figures/fig3_dambreak_martin_moyce.png` —— 涌潮前锋 x*(t*) 与
  残余柱高 h*/h0 对实验点。

## 5. 注意点（坑 / 局限 / 适用范围）

1. **阈值化几何误差是分辨率函数，不是方法常数**：同一 PLIC 在 N=64 是
   9.30 %、N=128 是
   3.95 %。因此"< 5 %"必须连同声明
   分辨率一起写；早先的 campaign 正因如此把它排除在判据外（只判面积守恒 1 %）。
   本单把它作为 Z1 明确判据并给出收敛阶，填的就是这个空缺。
2. **守恒 ≠ 保形**：一圈内 α 的全局和漂移 ~1e-15（PLIC/CICSAM 都是机器精度），
   但形状误差分别是 3.95 % /
   4.38 %；只判质量守恒会把
   这个差别完全掩盖。
3. **溃坝对照是 honest FAIL**：数值涌潮前锋推进系统性偏慢（t*=3.0 处
   x*≈2.41 vs 实验 3.30，−27 %），早期 t*≈0.3 处相对误差最大。可归因三条：
   (a) 动量对流一阶迎风 → 前锋被数值扩散拉平，半网格深度判据滞后；
   (b) void 自由面模型（气相干跑、p=0 Dirichlet）低估早期塌落加速度；
   (c) 实验曲线是图上读点（±几 %），远不足以解释 27 %。
   仓库既有 `docs/two_phase_chain/report.json` 的同一对照同样 FAIL
   （front_max_rel_error 0.9882），因此这不是本单引入的回归，而是该能力面
   长期缺口——按两轮法留第 2 轮：先补界面钝化（PLIC + THINC/压缩混合）与二阶
   动量对流，再谈 15 % 判据。
4. **二维、层流、无表面张力**：本验收面不含 CSF 表面张力与湍流模型；Zalesak 是
   纯运动学算例，溃坝为层流 NS。结论只针对"界面捕获方案本身"，不能外推为
   工程两相流（晃荡/气泡）精度结论。
5. `interface_width_cells` 在二维场上是"某列内 10–90 % 连续段最长值"，圆周
   穿越的列会给出 12 之类的读数，**只作诊断、不参与判据**。

## 6. 误差判定（逐项 vs 基准，显式回答是否 < 3 %）

| 关键量 | 基准 | 误差 | 是否 < 3 % |
|---|---|---|---|
| Zalesak 几何误差（PLIC, N=128） | 解析形状（一圈后 = 初始场） | 3.95 % | **否**（判据 < 5 %，满足） |
| Zalesak 面积守恒（PLIC, N=128） | A(0) | 6.42e-16（相对） | 是 |
| Zalesak 周长（PLIC, N=128） | P(0) | 6.43 % | 否（判据 < 10 %，满足） |
| Zalesak 几何误差（N=64） | 解析形状 | 9.30 % | 否（未到声明分辨率） |
| 溃坝涌潮前锋 | Martin & Moyce (1952) | 98.8 % | 否（诚实 FAIL） |
| 溃坝残余柱高 | Martin & Moyce (1952) | 84.5 % | 否（诚实 FAIL） |
| 溃坝体积守恒 | V(0) | 4.1e-16（相对） | 是 |

## 7. 复现

```bash
# 全量验收面（本报告数字来源；本机约 3 分钟，N=256 与 240x80 leg 走集群）
python -m tensorfvm.twophase_gap --case all --out docs/GAP-FVM-TWOPHASE/artifacts \
    --cells 64 128 256 --schemes plic cicsam --dam-grids 120x40 240x80
# 只重渲染报告图（不重算）
python -m tensorfvm.twophase_gap --summary-only --out docs/GAP-FVM-TWOPHASE/artifacts
python docs/GAP-FVM-TWOPHASE/render_report.py
# 测试（含求解器集成自检）
python -m pytest tests/test_twophase_gap.py -q
```

工件：`docs/GAP-FVM-TWOPHASE/artifacts/` = `gap_twophase.json`（判据+数字）+
`leg_*.json`（逐 leg 记录含 reference 块）+ `fields_*.npz`（场/历史）+
`GAP-FVM-TWOPHASE.md`（自动渲染）+ `figures/*.png`。
