# TensorFVM CHT 共轭传热验证面：流-固热耦合 · Stefan 相变 · 冰-钢结构温度场 · 与 W4-N08 互裁

> 战役节点 **P295W1R2-N01**（TensorFVM）。全部数字由
> `PYTHONPATH=src python -m tensorfvm.cht.runner_phase --case all` 从工件
> `src/tensorfvm/verification/artifacts/cht-stefan/*.json` 程序化生成
> （表格见同目录 [`RESULTS.md`](RESULTS.md)，本文件只负责物理表述、方法与解读，不手抄数字）。
> 证据等级：**REAL**（全部温度/热流由离散导热方程积分产生，无查表、无硬编码温度）。

## 1. 案例介绍

极地/寒区船体上层建筑与甲板的结冰不是孤立的热现象：飞沫在冷结构表面冻结，
冻结潜热必须经由**冰层 - 接触面 - 钢质板架 - 舱内环境**这条串联热阻通道排走。
本节点在 TensorFVM 中建立三个相互独立的验证通道，并把这个通道与船舶平台的
W4-N08 积冰模块（`backend/api/marine_icing.py`，面元热平衡 + Polar Code 稳性）
接起来互裁：

| 通道 | 物理对象 | 判据 |
|---|---|---|
| C1 | 冰/钢两块**分别求解**的耦合界面（Dirichlet–Neumann 迭代 + 接触热阻） | 温度连续误差 ≤1 %，热流守恒残差 ≤1e-6，变异注入必红 |
| C2 | 一维 Stefan 相变（enthalpy-porosity，L=334 kJ/kg，ΔT_mush=0.5 K） | 相变前沿 vs Neumann 解析解 ≤3 % |
| C3 | 冰层 + 钢板的稳态温度场 | vs 一维级数解析解 ≤5 % |
| C4 | 与 W4-N08 同冰况互裁 | 表面热平衡复现（相对误差 ≤1e-4）+ 温度场 L2 ≤5 % |

材料与几何（设计算例，全部进入 `interface.json`/`ice_steel.json` 的 `design` 段）：

| 参数 | 数值 | 来源 |
|---|---|---|
| 冰导热系数 k_i | 2.2 W/(m·K) | 战役规格 |
| 钢导热系数 k_s | 50 W/(m·K) | 战役规格 |
| 接触热阻 R_c | 1e-4 m²K/W | 战役规格 |
| 冰层厚 h_i | 33.4 mm（= W4-N08 参考面元结冰率 5.572 mm/h × 6 h） | 由 W4-N08 工件反算 |
| 钢板厚 h_s | 10 mm | 设计假设 |
| 冰点 T_f | 271.35 K（−1.8 ℃，海水冰点） | W4-N08 |
| 舱内侧 T_back | 258.15 K（−15 ℃，冷舱） | 设计假设（见 §5 局限） |
| 冰/水物性 | ρ=917 kg/m³, c=2100 J/(kg·K), L=334 kJ/kg, T_m=273.15 K, ΔT_mush=0.5 K | 战役规格 |

W4-N08 冰况（`reference_panel`，从工件原样读出，含 sha256 溯源）：
T_a = −10 ℃、U = 10 m/s、T_sea = −1.8 ℃、RH = 0.8、飞沫通量 9.128 kg/(m²·h)、2 m × 2 m 壁面。

## 2. 方法与流程

### 2.1 C1：两域耦合 + 界面独立审计（`cht/ice_steel.py`）

冰域与钢域**各自**用单元中心有限体积离散并单独求解；二者只通过界面热流 q 和接触条件

    T_ice(Γ) − T_steel(Γ) = q · R_c

耦合（Dirichlet–Neumann 定常迭代，松弛 0.5，割线斜率）。界面连续性因此是**迭代挣来的**
不动点，而不是离散格式白送的：`interface_audit()` 用两侧最外两个单元中心的**单侧差分**
重新算出 q_ice、q_steel（不复用装配好的面导热系数），再外插界面温度，给出

* `flux_imbalance_rel = |q_ice − q_steel| / max(|q|)` ≤ 1e-6
* `temperature_continuity_rel = |ΔT_meas − q R_c| / ΔT_drive` ≤ 1e-2

变异注入（`TwoDomainCHT(mutation=...)`，三条都是真实回归形态）：`single_sweep`（耦合只扫一遍）、
`sign_flipped`（界面热流符号反号地加给钢域）、`steel_decoupled`（钢域按绝热处理）。
判据必须由绿转红，结果见 `interface.json → mutations`。

### 2.2 C2：enthalpy-porosity Stefan（`cht/phase_change.py`）

    ∂h/∂t = ∂/∂x(k ∂T/∂x),  h = c(T − T_sol) + L f_l(T),  f_l = (T − T_sol)/ΔT_mush

隐式 Picard（表观热容 c_eff = c + L/ΔT_mush 在糊状区），时间步之后**由守恒面热流更新焓**
（不由温度更新），因此离散能量账本到求解器容差闭合（记录在 `stefan.json → ledger`）。
注意：显式格式在此完全不可用——糊状区内 α_eff = k/(ρ c_eff) ≈ 3.6e-9 m²/s，
dx=0.1 mm 时步长上限 ~1e-9 s 量级，小时级积冰需 ~1e8 步；隐式格式在 dt=0.5 s 稳定。
前沿取 `f_l = 0.5` 交点（线性插值）；另给焓积分前沿作交叉检查。
对照解析解：一相 Neumann 相似解，λ e^{λ²} erf λ = St/√π，X(t) = 2λ√(αt)，St = 0.0629。

### 2.3 C3：冰/钢稳态场（`LayeredWall2D`）

同一网格内含冰层与钢层，界面面导热系数为两个半单元热阻 + R_c 的串联；
顶面/底面 Dirichlet。解析参考为独立实现的**一维级数热阻解**（`analytic_series_field`，
逐层线性、接触面处温度跳变 q·R_c）。二维求解器在侧向均匀时必须退化为一维（实测
~1e-12 K），侧向变化工况用于出图（fig2）。

### 2.4 C4：与 W4-N08 互裁（`cht/icing.py`）

本模块**独立重实现**（不 import 平台代码）W4-N08 的五项表面热平衡：对流（平板外流
Nusselt 层流/湍流转捩 + 自然对流立方和）、蒸发/升华（Chilton–Colburn 类比，h_m 由 h_c 导出）、
长波辐射（Idso–Jackson 晴空 + 云量）、飞沫显热（W c_pw (T_sea − T_s)）、冻结潜热（f W L_f），
并按 W4-N08 的湿/干生长判据分区。随后：

1. **复现检查**：同一冰况下我们的 q_conv / q_evap / q_lw / q_req / f / 结冰率 vs 工件数值；
2. **温度场互裁**：以 W4-N08 给出的表面温度与冰厚（同冰况导出）作为边界数据，CHT 有限体积场
   对照独立级数参考场，报 L2 相对误差；
3. **越出 W4-N08 的一步**：把 CHT 算出的壁面热阻放进串联，让**结构导热**参与表面热平衡，
   得到 conduction-aware 的 regime / 冻结份额 / 结冰率，并与 W4-N08 的绝热基底假设对比
   （这是模型差异，不是数值误差）。

## 3. 计算结果

所有数字见 [`RESULTS.md`](RESULTS.md)（由工件生成）。摘要：

* **C1**：温度连续 3.5e-14（限值 1e-2）、界面热流残差 1.5e-12（限值 1e-6）、与级数解的 q 相对差 1.4e-13；
  三种变异全部变红（`single_sweep` 靠温度连续性 0.88 变红，`sign_flipped`/`steel_decoupled`
  靠热流残差 2.0 / 1.0 变红）。
* **C2**：前沿相对误差 300 s 1.10 %、600 s 0.53 %、1200 s 0.21 %（判据 3 %）；
  能量账本不平衡 2.6e-15；温度剖面最大偏差 0.087 K（驱动温差 10 K 的 0.9 %）；
  nx 200→800 收敛 2.51 %→0.66 %。
* **C3**：L2 相对误差 ~1e-14（判据 5 %），最大绝对误差 ~1e-12 K；级数解 q = 852.6 W/m²，
  接触面温度跳变 0.0853 K = q R_c（逐字核对）。
* **C4**：五项通量与 W4-N08 工件相对误差 ≤1.7e-7；f 6.1e-8；结冰率 4.0e-8；温度场 L2 ~9e-15（判据 5 %）；
  conduction-aware 互裁：冷舱 −15 ℃ 时 W4-N08 的 wet/f=0.543 翻转为 **dry/f=1**，
  结冰率 5.57 → **10.26 mm/h**；扫掠显示 −1.8 ℃（与海水连通、近似绝热基底）时两路一致。

图：fig1 Stefan 前沿/剖面 vs Neumann；fig2 冰/钢二维温度场云图 + 剖面 vs 解析；
fig3 界面审计与变异对照（对数坐标）；fig4 表面热平衡复现 + 冻结份额的基底温度扫掠。

## 4. 误差判定（逐项回答「是否 <3%」）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| Stefan 前沿 X(t) | Neumann 相似解 | 1.10 / 0.53 / 0.21 % | **是** |
| Stefan 温度剖面 | Neumann 相似解 | ≤0.087 K（0.9 % of ΔT） | **是** |
| 冰/钢稳态场 | 一维级数解析 | ~1e-14（数值零） | **是** |
| 界面温度连续 | q R_c 精确跳变 | 3.5e-14 | **是** |
| 表面热平衡（C4） | W4-N08 工件 | ≤1.7e-7 | **是** |
| 冻结份额/结冰率（C4） | W4-N08 工件 | 6.1e-8 / 4.0e-8 | **是** |
| 温度场（C4） | 独立级数参考场 | ~9e-15 | **是** |
| 界面热流守恒 | 两侧独立重构 | 1.5e-12 | **是**（限值 1e-6） |

**注意 C3/C4 的 L2 是「数值零」**：两条路线解的是同一个线性边值问题，因此这一项验证的是
**实现正确性（FV vs 闭式）**，而不是表面闭合模型是否正确——后者由 C4 的通量复现项
（对 W4-N08 工件 ≤1.7e-7）承担。两者合起来才构成完整的互裁链，本文件不把前者当卖点。

## 5. 注意点 / 局限 / 适用范围

1. **湿/干生长假设是本节点最大的模型不确定性**：W4-N08 只做表面热平衡、假定基底绝热
   （q_cond = 0）。一旦把 CHT 的壁面热阻串进平衡，冷舱会提供额外排热，冻结份额上升甚至
   wet→dry 翻转（RESULTS 表）。方向明确：对**冷舱/与海水隔热的舱壁**，W4-N08 的结冰率
   是**偏小（不保守）**；对与海水温度连通的湿表面（T_back ≈ T_f），两者一致。
   建议后续把舱内侧边界条件（内舱空气温度 + 内表面对流系数）做成 W4-N08 的输入项。
2. **C3/C4 的 L2 判据在同类线性问题上必然接近机器精度**，能抓住实现错误（网格、接触热阻、
   BC 装配），抓不住模型错误；任何声称「L2 <5 % ⇒ 物理正确」的表述都不成立。
3. **相变模型是纯导热 + 焓法**：不求解糊状区流动（Voller 的 Darcy 孔隙动量汇在本问题中
   无流体域，未启用），也未含盐分（海水冻结区间被压缩成 0.5 K 糊状带）。真实海冰糊状区
   可达 ~1 K 且含卤水，本文的 ΔT_mush 是规格值。
4. **ΔT_mush 敏感性非单调**（0.5/0.2/0.05 K 对应 1.10/1.44/1.98 %）：主控误差是初始液相
   过热（T_init = T_liquidus）与 t→0 的时间分辨率，糊状带宽不是主控项；因此这里不宣称
   「ΔT 收敛阶」，只报「全部 ≤2 %，判据 3 % 满足」。
5. **单位陷阱（踩过）**：`icing.py` 的表面热平衡以 ℃ 计，CHT 场量以 K 计。混用会让
   湿/干判据错到「regime=none、q_cond = −16.8 kW/m²」；测试里显式写了
   `T_BACK − 273.15` 的换算并注释。
6. **W4-N08 数据只读入不手抄**：`src/tensorfvm/data/marine_icing_w4n08/reference_panel.json`
   由 `scripts/extract_w4n08_reference.py` 从平台工件提取，文件内带源路径与源 sha256；
   `load_w4n08_reference()` 返回时再附本地文件 sha256。
7. **凝结/升华与辐射的线性化**：C4 的 conduction-aware 求解用二分求根（单调分支），
   未做辐射—蒸发的 Newton 耦合；在 −40…0 ℃ 范围内二分到 1e-10 ℃，足够。

## 6. 复现方式

```bash
PYTHONPATH=src python -m tensorfvm.cht.runner_phase --case all   # 工件 + 图 + RESULTS.md
PYTHONPATH=src python -m pytest tests/test_cht_phase_change.py -q # C1–C5 判据（约 20 s）
python scripts/extract_w4n08_reference.py --source <W4-N08 validation.json>  # 重新提取互裁数据
```

产物：
* `src/tensorfvm/verification/artifacts/cht-stefan/{interface,stefan,ice_steel,w4n08_crosscheck,acceptance}.json`
* `docs/cht-stefan-ice-steel/figures/fig{1..4}*.png`、`RESULTS.md`（自动生成）
* 测试：`tests/test_cht_phase_change.py`（15 个用例，C1–C5）