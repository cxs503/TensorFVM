# 阻力链成熟化：形式因子 CFD 提取与相关补贴双口径裁决（IN46-N01）

**一句话结论**：阻力链的**口径**已统一并可机器裁决（U_V 从双计的 11.7 %–24.5 % 降到 2.2 %–2.7 %，
5 % 闸门真正具备判别力），ITTC-78 相关补贴实现与平台独立实现逐点零偏差；
但**形式因子 (1+k) 的 CFD 数值没有提交**——本节点算力下无法完成"可信分辨率"的双体 RANS，
按纪律记为 **C1 = WITHHELD（honest FAIL）**，理由与实测代价见 §5 与自动生成的
[`RESULTS.md`](RESULTS.md)。

> 所有数字均在 [`RESULTS.md`](RESULTS.md)（由 `scripts/run_resistance_chain.py` 从
> `artifacts/resistance-chain/form_factor_chain.json` 自动生成，零手工誊抄）。

---

## ① 案例介绍（物理背景 / 几何 / 参数）

### 1.1 为什么是"双体 + 形式因子"

实船总阻力按 1978 性能委员会（ITTC-78）口径写成

```
C_T = (1 + k) C_F + ΔC_F          （R_T = R_F + R_Pv + ΔR，k = R_Pv / R_F）
```

形式因子 `(1+k)` 是**粘性压阻力与摩擦阻力之比**。把它从试验中分离出来需要做
"单航速多航态/低 Fn 外推"（Prohaska），而从数值上分离的唯一干净做法是**双体模型**：
把水面当成对称面，将船体关于设计水线镜像，自由面问题退化为封闭单相问题，此时

```
R_V = R_shear + R_pressure
C_F = R_shear / (q S) ,  C_Pv = R_pressure / (q S) ,  (1+k) = (C_F + C_Pv) / C_F
```

镜像使湿面积与受力同时加倍，故以**实船湿面积 S** 计算的无量纲系数不变
（Hughes 的形式因子假设：`k` 与 Re 基本无关，故模型尺度得到的 k 可搬到实船）。

### 1.2 两条并存的相关补贴口径（本节点要裁决的对象）

| 口径 | 公式 | 输入 | 适用尺度 |
| --- | --- | --- | --- |
| ITTC-78 粗糙度补贴 | `ΔC_F = [105 (k_s/L)^{1/3} − 0.64] × 10⁻³`（负值截零） | 表面等效砂粒粗糙度 `k_s` | **实船**（`k_s/L` 很小） |
| Holtrop-1984 相关补贴 | `C_A = 0.006(L+100)^{-0.16} − 0.00205 + 0.003√(L/7.5)·C_b⁴ c₂ (0.04 − c₄)` | 仅船型系数 | 实船（回归域内） |

两者**互为替代**，不是叠加项。历史链路把 `ΔC_F` 与 `C_A` 同时加进 `C_T`，等于把
相关补贴记了两遍——这就是本节点实测到的"口径未统一"缺陷（`RESULTS.md` 中
"双计/C_V"一行：模型尺度 57.7 %、实船尺度 21.5 %）。

### 1.3 基准船与几何（先验证 Cb/L/B）

任务给定基准：`L = 7.0 m, Cb = 0.708, Fn = 0.142`（本仓既有 "KVLCC2 模型尺度" 口径），
配套 `B = 0.936 m, T = 0.342 m, Cm = 0.985, Cwp = 0.808, LCB = −1.688 %L`。

工作区内**没有可信的 KVLCC2 型值表**（审计见 §5.1），因此船体由
`HullShape` 的参数化横剖面族给出，并用有界最小二乘把四个积分系数**拟合到主尺度系数**：
水线半宽 `w(ξ)` 分支控制 `Cwp`、舯剖面超椭圆族 `(m, g)` 控制 `Cm`、
随站位丰满度 `w^section_power` 控制 `Cb/Cwp` 之差，`LCB` 由纵向矩约束。
拟合残差全部 ≤ 4 × 10⁻⁵（见 `RESULTS.md` 几何审计表）；解析湿表面积与 Holtrop 回归式
相差 0.15 %（见 `RESULTS.md` 几何审计表），这是两条独立构造互相印证。

**几何审计（重要的物理提醒）**：真 KVLCC2 在任何尺度都是 `Cb = 0.8098`、`L/B = 5.52`；
任务口径的 `Cb = 0.708`、`L/B = 7.48` **与 KVLCC2 并非几何相似**。这一差别直接改变
Holtrop 形式因子：任务口径 (1+k) = 1.1867，真船口径 (1+k) = 1.3597（`RESULTS.md` 几何审计一节）。
**所以 "(1+k) 与 Holtrop 偏差 ≤ 5 %" 这句判据必须绑定口径**，否则基准本身就是可互换的两个数。

---

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

### 2.1 交付物结构（落库 TensorFVM）

| 文件 | 作用 |
| --- | --- |
| `src/tensorfvm/resistance_chain.py` | 摩擦线（ITTC-57 / Schoenherr / Prandtl-Schlichting）、ITTC-78 `ΔC_F`、Holtrop 形式因子/湿面积/`C_A`、双体阻力分解、统一链 `viscous_chain`、U_V 预算、模型—实船换算 |
| `src/tensorfvm/kvlcc2_double_body.py` | 参数化船体（系数拟合）、阶梯 Cartesian 双体网格、`y⁺` 首层高度、`NativeSimpleSSTRansSolver3D` 驱动、阻力符号约定 |
| `tests/test_resistance_chain.py`（24 例）、`tests/test_kvlcc2_double_body.py`（10 例） | 判据级单元测试 |
| `scripts/run_resistance_chain.py` | 一次跑出全部 json/csv/png + 自动写 `RESULTS.md` |

### 2.2 统一链的物理约束（代码级，而不是注释级）

`viscous_chain(principle, speed, regime, ks, ...)` 只接受 **一个** 补贴口径：

* `regime="ittc78_roughness"` **必须**给 `ks`；`regime="holtrop_ca"` **禁止**给 `ks`；
  两者同时给出直接 `ValueError` —— 把"双计"从"容易犯的错误"变成"跑不通的调用"；
* `regime="none"` 用于双体/光洁模型对照（`ΔC_F = 0`）；
* 结果对象记录 `one_plus_k_source`（`holtrop_regression` / `user_supplied`）与 `cf_law`，
  使"这条 `C_T` 是在哪个口径下算出来的"随数值一起流动，不再依赖口头约定。

### 2.3 验证不确定度 `U_V`（ITTC 7.5-02-02-02 风格，显式定义）

各分量取矩形（最大熵）界 `b`，折算标准不确定度 `u = b/√3`，相对 `C_T` 后平方和开根：

* `u_friction`：三条摩擦线在工况 Re 下的带宽半宽（真实散布，不是假设值）；
* `u_form`：CFD 形式因子与回归值之差（无 CFD 时用先验 5 % 带，并在结果里标 `prior_band`）；
* `u_allowance`：单口径的模型形带（2 %）；若恢复双计，该项变成 `(ΔC_F + C_A)/C_T`；
* `u_cfd`：CFD 网格离散不确定度（本节点 **withheld**，结果对象里显式标注）。

### 2.4 双体 RANS 路径（已实现，未在可信分辨率上运行）

1. `HullShape.coefficients()` 先验证几何（Cb/Cm/Cwp/LCB）→ 2. `build_double_body_mesh()`
生成**封闭双体**（船体 + 关于 z=0 的镜像，沿水线胶合）的阶梯 Cartesian 网格，
按 `y⁺ = 100` 反算首层高度；3. `NativeMesh3D.from_cells(..., triangulate_faces=True)`（SST
壁面距离算子要求三角面）；4. `NativeSimpleSSTRansSolver3D(wall_treatment="spalding",
wall_patch="hull")`；5. `solver.forces("hull")` → `drag_components()` → `form_factor_from_double_body()`。

关键设置（由代码显式固定、并由测试钉住）：hull 面必须作为 Dirichlet 零速面（SST 壁面模型的前置条件，
否则求解器直接抛错）；入口/远场给自由来流，出口给压力 Dirichlet；阻力符号取"体力对流体的作用力取负"，
并用 `drag_components` 的守卫拒绝混淆符号的输出。

### 2.5 工具链

`numpy 2.4` / `scipy 1.16`（`least_squares` 拟合、稀疏直接解）/ `matplotlib 3.10`（Agg）
/ TensorFVM native 三维求解器（纯 numpy/scipy，非 PyTorch 路径）。

复现：

```bash
python3 scripts/run_resistance_chain.py                                   # 全部产物 + RESULTS.md
python3 -m pytest tests/test_resistance_chain.py tests/test_kvlcc2_double_body.py -q   # 34 例
```

---

## ③ 计算结果

全部表格见自动生成的 [`RESULTS.md`](RESULTS.md)：几何审计表、逐 Re 摩擦线/补贴表
（8 点，1e6–1e9）、双口径对照表、U_V 预算表、求解器代价表、判据汇总表。
这里只给结论性事实（数值见上表）：

1. **ITTC-78 `ΔC_F` 与平台独立实现逐点零偏差**（25 个 (k_s, L) 点，最大相对偏差 0.0）→ C2 通过。
2. **`ΔC_F` 没有 Reynolds 依赖**（只随 `k_s/L` 走），逐 Re 表因此把它作为"同一列 + Re 行"呈现；
   模型尺度 `k_s/L` 大两个量级，直接套公式会给模型加到 ~1.4 × 10⁻³（≈ Cf 的 36 %）——
   **ITTC-78 补贴是实船口径，不能加在 7 m 模型上**，这是本节点发现的第二个口径陷阱。
3. **同一工况、两种口径给出的 `C_T` 差 13.0 %（模型）/**2.7 %（实船）；若两个都加，
   补贴项占 `C_V` 的 57.7 %（模型）/21.5 %（实船）。
4. **U_V**：统一口径后 2.20 %（模型 ITTC-78）、2.47 %（模型 Holtrop `C_A`）、
   2.73 %（实船 ITTC-78）、2.65 %（实船 Holtrop `C_A`）；统一前分别为 24.45 % / 11.70 %。
   → C3 通过，且闸门具备判别力（能拒掉双计口径、接受统一口径）。

---

## ④ 结果图

| 图 | 内容 | 说明 |
| --- | --- | --- |
| `artifacts/resistance-chain/fig_geometry.png` | 拟合水线、各站横剖面（含镜像全周）、`Cb/Cm/Cwp` 目标 vs 拟合柱状 | 几何先验证，再谈阻力 |
| `artifacts/resistance-chain/fig_friction_allowance.png` | 左：三条摩擦线 vs Re；右：`ΔC_F` 随 `k_s/L` 变化（log-log）并标出模型/实船工作点与 Holtrop `C_A` | 说明补贴是尺度相关口径 |
| `artifacts/resistance-chain/fig_dual_convention.png` | 模型/实船两种尺度下，单一 ITTC-78、单一 Holtrop、双计三种链路的 `C_T` 堆叠柱 | 双计把 `C_T` 抬到物理上不成立的水平 |
| `artifacts/resistance-chain/fig_uncertainty.png` | 左：`U_V` 分量堆叠；右：统一前后 `U_V` 与 5 % 闸门 | 闸门判别力 |
| `artifacts/resistance-chain/fig_feasibility.png` | 求解器 setup / 单次迭代代价 vs 网格量（log-log）+ 所需分辨率竖线 | C1 只能 WITHHELD 的量化理由 |

---

## ⑤ 注意点（踩坑、局限、适用范围）

### 5.1 审计发现（平台/仓库侧，只报告不改）

1. `ship-performance-platform/backend/data/geometry/ships/KVLCC2_Hull.stl` **不是真船几何**：
   文件仅 304 个三角形，包围盒 `320 × 72.5 × 20.8 m`（`y ∈ [0, 36.25]`、`z ∈ [−10.4, 10.4]`），
   与 KVLCC2 的 `B = 58 m`、`T = 20.8 m` 不符；同目录 `standard_models_enhanced.py` 声称
   `triangles: 140000`。据此不能做船体 CFD。
2. `tensorfvm/kvlcc2_geometry.py` 的 `Cb = 0.708` 与主尺度 `L/B = 7.48` 在几何上与 KVLCC2
   （`Cb = 0.8098, L/B = 5.52`）不自洽，其 `get_full_offsets()` 只是 `w(x)·shape_factor` 的经验拼装，
   且 `wetted_surface_approx = L(2T+B)√Cb` 只是量级估算。本节点因此改用"系数拟合船体"，
   并把这一不自洽量化成 (1+k) 的口径差（1.1867 vs 1.3597）。
3. 求解器能力边界（实测）：native 三维 SIMPLE 只支持 `direct`/`amg` 两种线性解法，
   而本机 `pyamg` 未安装 → 只能 SuperLU 直接解，`setup ∝ N^{1.6}`、单次外迭代 `∝ N^{1.55}`；
   SST 壁面距离算子要求 `triangulate_faces=True`；SST 壁面模型要求 hull 面为 Dirichlet 零速面。
4. 环境陷阱：`/tmp/timeit.py` 之类的陈旧文件会遮蔽标准库（脚本放在 `/tmp` 下会以
   `partially initialized module 'torch'` 的形式炸掉）；`site-packages` 里的 editable `.pth`
   指向多个历史工作树，跨仓 `import` 可能静默取到别的 worktree。脚本一律放在仓内运行。

### 5.2 方法论局限

* **阶梯网格的湿面积是放大的**：阶梯面按 `|n_x|+|n_y|+|n_z|` 计面积，实测 hull 面面积
  比解析值高 20 %–24 %（`RESULTS.md` 网格表 `hull_patch_area_ratio`），这会系统性抬高 `C_F`；
  同时体体积比仅 0.996–0.998（守恒尚可）。因此阶梯网格上的 `(1+k)` **不可用于 5 % 判据**。
* **分辨率需求**：`y⁺ = 100` 要求首层 ~4.8 mm，近壁单元 ~7.6 × 10⁵，整场 ~2.9 × 10⁶ cells；
  按实测标度外推 setup 62.9 h、每 100 次外迭代 86.5 h —— 超出本节点预算（`fig_feasibility.png`）。
* **可负担分辨率上 SST 路径发散了**（实测，原始日志 `artifacts/resistance-chain/rans_attempt.log`，
  机读摘要 `rans_attempt.json`）：2 184 cells（spacing 0.16 m，Re ≈ 6.9 × 10⁵）上 setup 124.4 s、
  40 次外迭代 25.0 s，动量残差从 3.4 × 10¹⁴ 单调涨到 5.0 × 10⁷²。说明即便放弃分辨率，
  阶梯网格 + Spalding 壁面处理的伪时间步/松弛设置本身还需要工程化（这是 TensorFVM 侧的一条欠账，
  不是本节点能用调参掩盖的）。因此 C1 同时受"算力"和"该路径未验证"两条限制。
  降低 Re（如 10 倍粘性、Re ≈ 7 × 10⁵）可把网格降到 ~10⁵ 量级，但那已不是基准 Re，
  且 Stern 分离对 Re 敏感，`k` 的 Re 无关假设需要单独验证，本节点未做。
* **U_V 的定义是"模型形不确定度"**：`u_cfd`（离散不确定度）在本节点 withheld，
  所以 C3 的 2.2 %–2.7 % 不包含 CFD 网格不确定度；一旦有可信 CFD，`U_V` 会上升。
* 摩擦线散布在模型 Re（7.2 × 10⁶）已达 6.1 %（ITTC-57 vs Schoenherr），
  这是**参考值本身的不可约不确定度**，不是本实现的误差；任何 <3 % 的判据若建立单条摩擦线上，
  必须先声明用的是哪条线。

### 5.3 适用范围

* `HullShape` 的系数拟合族适用于高块系数运输船（单桨、无球鼻/球鼻可后加），
  不适用于细长高速船（`Cp` 低、兴波主导）。
* Holtrop `(1+k)` 仅在其回归域（`Fn 0.10–0.45`、`Cm 0.55–0.998`）内有意义；
  `L/b` 偏离商船范围时只是外推，代码会把 `offset_consistency()` 一并输出以便判断。
* ITTC-78 `ΔC_F` 只在实船尺度、`k_s` 有实测/规范值时使用。

---

## ⑥ 误差判定（逐项，显式回答"是否 <3%"）

| 量 | 基准 | 结果 | 是否 <3 % |
| --- | --- | --- | --- |
| ITTC-78 `ΔC_F` 实现 | 平台独立实现 `physics.holtrop.friction.delta_cf_ittc78`（25 个 (k_s,L) 点） | 最大相对偏差 0.0 | **是（0.0 %）** |
| ITTC-57 `C_F` | 闭式 `0.075/(log₁₀Re−2)²`（Re = 10⁶ 时 = 4.6875 × 10⁻³） | 解析吻合到机器精度 | **是（0.0 %）** |
| Schoenherr `C_F` | 隐式方程残差 `0.242/√C_F − log₁₀(Re C_F)` | 残差 ≤ 1 × 10⁻¹² | **是（0.0 %）** |
| Schoenherr vs ITTC-57（模型 Re） | 两条历史关联式 | 6.1 %（参考带，非实现误差） | 否 —— 属参考值不可约散布，已在 `U_V` 中显式计入 |
| Holtrop `(1+k)`、湿面积、`C_A` | 平台权威实现 `physics.holtrop.mennen` | 1e−12 量级一致（KVLCC2 模型口径：1.186736 / 8.705806 m² / 7.9086e−4） | **是** |
| 船体几何系数 `Cb/Cm/Cwp/LCB` | 任务主尺度系数 | ≤ 4 × 10⁻⁵ | **是** |
| 解析湿面积 vs Holtrop 回归 | 两条独立构造 | 0.15 % | **是** |
| **CFD 形式因子 vs Holtrop (C1)** | Holtrop 1.186736（闸门 [1.1274, 1.2461]） | **未提交**：无可信分辨率的双体 RANS | **无法判定 → C1 = WITHHELD（honest FAIL）** |
| `U_V`（统一口径后） | 5 % 闸门 | 2.20 % / 2.47 % / 2.73 % / 2.65 % | **是** |
| `U_V`（统一前，双计） | 5 % 闸门 | 24.45 % / 11.70 % | 否（正是本节点要修掉的缺陷） |

**未完成部分（如实记录）**：形式因子 CFD 数值未产出；双体 RANS 的网格生成、边界条件、
阻力符号与分解链路已实现并通过 10 例测试，但在能给出 5 % 可信度的分辨率上跑不动。
因此"阻力链成熟化闭环"在**口径**维度闭合（C2/C3/C4/C5 通过），在**形式因子数值**维度
仍是欠账（C1）。

---

## 文件清单

```
src/tensorfvm/resistance_chain.py            统一阻力链（摩擦线/补贴/形式因子/U_V/换算）
src/tensorfvm/kvlcc2_double_body.py          双体几何 + 阶梯网格 + RANS 驱动 + 阻力分解
tests/test_resistance_chain.py               24 例
tests/test_kvlcc2_double_body.py             10 例
scripts/run_resistance_chain.py              全部产物 + 自动 RESULTS.md
docs/resistance-chain-form-factor/README.md  本文
docs/resistance-chain-form-factor/RESULTS.md 自动生成的数字与判据（零手抄）
artifacts/resistance-chain/form_factor_chain.json   全部数值与元数据
artifacts/resistance-chain/criteria.json            C1–C5 机器判据
artifacts/resistance-chain/friction_allowance.csv   逐 Re 摩擦线 + 补贴表（8 点 × 2 尺度）
docs/resistance-chain-form-factor/data/fig_*.png    5 张图（提交快照）
artifacts/resistance-chain/fig_*.png                同一批产物的运行目录（artifacts/ 不在本仓 sparse-checkout 锥内，故以 docs/ 下快照入库）
docs/resistance-chain-form-factor/data/rans_attempt.json/.log  双体 RANS 端到端尝试的实测证据（提交快照）
```