# TensorFVM 共轭传热验证面：流-固热耦合 + Stefan 相变 + 冰-钢结构温度场 + 与 W4-N08 互裁

> 战役节点 P295W1R2-N01（TensorFVM，分支 agent/m10735）。证据等级 **REAL**：
> 全部温度/热流由离散导热方程积分产生，无查表、无硬编码温度。
> 本报告数字全部来自工件 `src/tensorfvm/verification/artifacts/cht-stefan/*.json`
> 与自动生成表 `docs/cht-stefan-ice-steel/RESULTS.md`（`python -m tensorfvm.cht.runner_phase --case all`）。

## ① 案例介绍（物理背景 / 几何 / 参数）

寒区船舶上层建筑与甲板的积冰是**热平衡问题**：飞沫在冷结构表面冻结，冻结潜热必经
"冰层 → 接触面 → 钢板 → 舱内环境" 串联热阻排走。本节点把这条通道建成三个可互裁的验证面：

| 通道 | 对象 | 判据 |
|---|---|---|
| C1 | 冰/钢**两域分别求解**的耦合界面（Dirichlet–Neumann + 接触热阻 R_c） | 温度连续 ≤1 %、热流残差 ≤1e-6、变异必红 |
| C2 | 一维 Stefan 相变（enthalpy-porosity，L=334 kJ/kg，ΔT_mush=0.5 K） | 前沿 vs Neumann ≤3 % |
| C3 | 冰层+钢板稳态温度场 | vs 一维级数解析 ≤5 % |
| C4 | 与 W4-N08 同冰况互裁 | 表面热平衡复现 ≤1e-4 + 温度场 L2 ≤5 % |

参数表（全部落在 `interface.json` / `ice_steel.json` 的 `design` 段）：冰 k=2.2 W/(m·K)、
ρ=917 kg/m³、c=2100 J/(kg·K)、L=334 kJ/kg、T_m=273.15 K、ΔT_mush=0.5 K；钢 k=50 W/(m·K)；
接触热阻 R_c=1e-4 m²K/W；冰厚 33.4 mm（= W4-N08 参考面元结冰率 5.572 mm/h × 6 h）、钢板 10 mm；
冰点 T_f=271.35 K（−1.8 ℃）；舱内侧 T_back=258.15 K（−15 ℃，冷舱，设计假设）。
W4-N08 冰况（从平台工件原样读出，含 sha256 溯源）：T_a=−10 ℃、U=10 m/s、T_sea=−1.8 ℃、
RH=0.8、飞沫通量 9.128 kg/(m²·h)、2 m×2 m 壁面 → 模块给出 wet、f=0.543297、5.572267 mm/h。

## ② 怎么做的（方法 / 流程 / 关键设置 / 工具链）

1. **C1 两域耦合 + 独立审计**（`cht/ice_steel.py`）：冰域与钢域各自单元中心有限体积离散、单独求解，
   仅通过界面热流 q 与接触条件 T_ice−T_steel = q·R_c 耦合（Dirichlet–Neumann 定常迭代，松弛 0.5 +
   割线斜率）。界面连续性因此是迭代不动点而非格式白送。`interface_audit()` 用两侧最外两个单元中心的
   **单侧差分**重新算 q_ice/q_steel（不复用装配好的面导热系数），再外插界面温度 → 温度连续误差与
   热流守恒残差。三种真实回归形态的变异注入：`single_sweep`（耦合只扫一遍）、`sign_flipped`
   （界面热流反号加给钢域）、`steel_decoupled`（钢域绝热）。
2. **C2 enthalpy-porosity Stefan**（`cht/phase_change.py`）：∂h/∂t = ∂/∂x(k·∂T/∂x)，
   h = c(T−T_sol) + L·f_l，糊状区 f_l=(T−T_sol)/ΔT_mush；隐式 Picard（表观热容 c+L/ΔT_mush），
   时间步之后**由守恒面热流更新焓**（非由温度），故离散能量账本到求解器容差闭合。
   显式格式不可用：糊状区 α_eff≈3.6e-9 m²/s，小时级积冰需 ~1e8 步；隐式 dt=0.5 s 稳定。
   前沿取 f_l=0.5 交点；对照一相 Neumann 相似解（St=0.06287、λ=0.17549、X=2λ√(αt)）。
3. **C3 冰/钢稳态场**（`LayeredWall2D`）：同一网格含冰层与钢层，界面面导热系数 = 两个半单元热阻 +
   R_c 串联；顶/底 Dirichlet；参考解为独立实现的**一维级数热阻解**（逐层线性 + q·R_c 跳变）。
   二维求解器在侧向均匀时退化为一维（实测 ~1e-12 K）。
4. **C4 互裁**（`cht/icing.py`）：独立重实现（不 import 平台代码）W4-N08 五项表面热平衡——对流
   （平板层流/湍流转捩 Nusselt + 自然对流立方和）、蒸发/升华（Chilton–Colburn，h_m 由 h_c 导出）、
   长波（Idso–Jackson + 云量）、飞沫显热、冻结潜热，并按 W4-N08 湿/干判据分区。然后
   (a) 复现其 q_conv/q_evap/q_lw/q_req/f/结冰率；(b) 以其表面温度与冰厚为边界数据，CHT 有限体积场
   对独立级数参考场报 L2；(c) 再把 CHT 壁面热阻放进串联，做 conduction-aware 的 regime/f/结冰率，
   与 W4-N08 的绝热基底假设对比（模型差异，非数值误差）。
5. 工具链：numpy/scipy（banded 与稀疏直接解）、matplotlib(Agg) 出图、pytest 判据、ruff 过 lint 门。

## ③ 计算结果（程序化注入，零手抄）

| 判据 | 阈值 | 实测 | 结论 |
|---|---|---|---|
| C1 界面温度连续（相对驱动温差） | ≤1e-2 | **3.508e-14** | 通过 |
| C1 界面热流守恒残差 | ≤1e-6 | **1.529e-12** | 通过 |
| C1 变异全部变红 | true | **true**（single_sweep 连续性 0.883；sign_flipped 热流 2.0；steel_decoupled 热流 1.0） | 通过 |
| C2 Stefan 前沿 vs Neumann | ≤3 % | **1.10 / 0.53 / 0.21 %** @300/600/1200 s | 通过 |
| C2 能量账本不平衡 | ≤1e-9 | **2.576e-15** | 通过 |
| C3 冰/钢温度场 L2 | ≤5 % | **1.99e-14**（最大绝对误差 1.6e-12 K） | 通过 |
| C4 五项通量复现 W4-N08 | ≤1e-4 | **≤1.72e-7**（q_conv 218.9928、q_evap 142.5713、q_lw 98.1371、q_req 459.7012 W/m²） | 通过 |
| C4 冻结份额 / 结冰率复现 | ≤1e-4 | **6.11e-8 / 4.01e-8**（f=0.543297、5.572267 mm/h） | 通过 |
| C4 温度场 L2（CHT vs 独立参考场） | ≤5 % | **8.99e-15** | 通过 |

补充：Stefan 网格/时间收敛 nx 200→800、dt 1.0→0.5 s：前沿误差 2.51 %→0.66 %（单调）；
ΔT_mush 0.5/0.2/0.05 K 对应 1.10/1.44/1.98 %（非单调，主控误差是初始液相过热与 t→0 分辨率）。
级数解 q=852.613 W/m²，接触面跳变 0.0853 K = q·R_c（逐字核对）。
C4 conduction-aware 扫掠：背面 −1.8 ℃（近似绝热基底）时与 W4-N08 完全一致（f=0.543、5.57 mm/h）；
背面 −10 ℃ 起 regime 由 wet 翻为 dry、f→1、结冰率 →10.26 mm/h；−15 ℃ 时 q_cond=602.2 W/m²、
表面温度 −5.67 ℃。方向明确：**冷舱工况下 W4-N08 的绝热基底假设使结冰率偏小（不保守）**。

## ④ 结果图

fig1 Stefan 前沿/剖面 vs Neumann；fig2 冰/钢二维温度场云图 + 剖面 vs 解析；fig3 界面审计与变异对照
（对数坐标）；fig4 表面热平衡复现 + 冻结份额随背面温度扫掠（figures 内嵌）。

## ⑤ 注意点（踩过的坑 / 局限 / 适用范围）

1. **湿/干生长是最大模型不确定性**：W4-N08 只做表面热平衡、假定基底绝热（q_cond=0）。把 CHT 壁面
   热阻串进平衡后，冷舱排热使冻结份额上升甚至 wet→dry 翻转。对冷舱/隔热舱壁，W4-N08 结冰率
   **偏小（不保守）**；对与海水温度连通的湿表面两者一致。建议把舱内侧边界条件（内舱空气温度 +
   内表面对流系数）纳入 W4-N08 输入。
2. **C3/C4 的 L2 判据在同类线性问题上必然是机器精度**：它验证实现正确性（网格/接触热阻/BC 装配），
   **不能**用来证明表面闭合模型正确——后者由 C4 对工件的通量复现项（≤1.7e-7）承担。
   "L2<5% ⇒ 物理正确" 不成立。
3. **相变模型为纯导热 + 焓法**：无糊状区流动（Voller 的 Darcy 孔隙动量汇无流体域可作用，未启用），
   未含盐分（海水冻结区间压成 0.5 K 糊状带；真实海冰糊状区更宽且含卤水）。
4. **单位陷阱（真踩到）**：icing 闭合以 ℃ 计、CHT 场量以 K 计；混用会让湿/干判据错到 regime=none、
   q_cond=−16.8 kW/m²；测试显式写 `T_BACK − 273.15` 并注释。
5. **W4-N08 数据只读入不手抄**：`data/marine_icing_w4n08/reference_panel.json` 由
   `scripts/extract_w4n08_reference.py` 从平台工件提取，带源路径 + 源 sha256；加载时再附本地 sha256。
6. **C4 表面闭合用二分**（单调分支），未做辐射—蒸发 Newton 耦合；−40…0 ℃ 内二分到 1e-10 ℃ 足够。
7. 适用范围：薄冰层（≤数十 mm）、准稳态结构导热、层流/转捩平板外流；不覆盖飞溅区两相流动、
   盐水糊状区、结构瞬态热惯性（钢板热时间常数 ≈7.8 s ≪ 6 h，本文按准稳态处理）。

## ⑥ 误差判定（逐项回答是否 <3 %）

| 关键量 | 基准 | 误差 | <3 % ? |
|---|---|---|---|
| Stefan 前沿 X(t) | Neumann 相似解 | 1.10 / 0.53 / 0.21 % | **是** |
| Stefan 温度剖面 | Neumann 相似解 | ≤0.087 K（0.9 % of ΔT） | **是** |
| 冰/钢稳态场 | 一维级数解析 | ~1e-14 | **是**（数值零，见局限 2） |
| 界面温度连续 | q·R_c 精确跳变 | 3.5e-14 | **是** |
| 界面热流守恒 | 两侧独立重构 | 1.5e-12 | **是**（限值 1e-6） |
| W4-N08 通量 / f / 结冰率 | 平台工件 | ≤1.7e-7 / 6.1e-8 / 4.0e-8 | **是** |
| C4 温度场 | 独立参考场 | ~9e-15 | **是** |

**诚实声明**：全部判据通过，但通过的性质不同——C1/C2 是对解析解的**真实验证**；C3/C4 的场 L2 是
**一致性检查**（同一线性问题两条实现）；C4 的物理价值在通量复现与 conduction-aware 差异
（wet→dry 翻转、结冰率 5.57→10.26 mm/h），该差异是模型缺陷发现，不是数值误差。
无 MOCK：无查表温度、无硬编码温度；C1 变异测试证明审计能抓住真实回归。

产物：`src/tensorfvm/cht/{phase_change,ice_steel,icing,runner_phase}.py`、
`tests/test_cht_phase_change.py`（15 用例）、`src/tensorfvm/verification/artifacts/cht-stefan/*.json`、
`docs/cht-stefan-ice-steel/{README,RESULTS}.md` + 4 图、`scripts/extract_w4n08_reference.py`。
复现：`PYTHONPATH=src python -m tensorfvm.cht.runner_phase --case all`；
`PYTHONPATH=src python -m pytest tests/test_cht_phase_change.py -q`（15 passed, 9.3 s）。