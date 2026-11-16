# TensorFVM 验证欠账账本（VERIFICATION DEBT LEDGER）

- 建账: m13807（selfstart 自找活，2026-11-11）
- 性质: 求解器仓选题锚（backlog 类）。本文件是 TensorFVM 侧的缺口账，
  后续 selfstart 单选题**先读本账**，认领条目后在 goal 首行标注【锚点=SLUG】。
- 纪律: 每条欠账带量化验收；done 条目移入 §4 不得重做；honest FAIL 不得覆盖。

## 1. 建账依据（本单扫描证据，全部只读）

| 来源 | 证据 |
|---|---|
| 平台报告 13305 | 潜在流附加质量对 Lamb 椭球精确解验证（平移项 <3% 全 PASS）；§6 明示两项未做：转动附加惯量 M_44..M_66、自由面/双体镜像未对解析锚验证 |
| 平台报告 13277 | 自然对流 Ra=1e5/1e6 网格阶梯，3% 判据 N=64 闭合；§5 明示超范围项：Ra~1e8 振荡起始、3D 腔 |
| 平台报告 13233 | 两相 VOF 验证面（溃坝 Ritter + Zalesak 旋转 + 质量守恒门）；§5 记录有限水柱 Ritter 适用域陷阱 |
| 平台报告 13285 | 验证索引机器门：死链 2→0、无活测试 PASS 条目 7→0、无状态缺口 5→0 |
| 功能点总账 | `feature_ledger` 56 条 planned/in_progress；FVM 相关：GAP-FVM-TWOPHASE（planned）、KPI-QC-RECHECK（planned） |
| 提案池 | /srv/tensor/agent/proposals/ 现存 p9087…p97 等 20 条，无 FVM 附加质量/转动惯量在飞单 |
| 仓内测试面 | tests/ 已有 ~30 个 two_phase/vof 测试文件（GAP-FVM-TWOPHASE 实际进度远超总账 planned 状态，总账疑似滞后） |

## 2. 欠账条目（按价值/可单干性排序）

### D1 转动附加惯量 M_44..M_66（最高优先，可单干）
- 来源: 报告 13305 §6 明示 follow-up；`maneuvering/dynamics.py` 的 roll/pitch/yaw
  质量矩阵需要 J_z 等转动附加项（现仅平移 added_mass 六自由度占位）。
- 做法: 复用 `panel_potential.solve_sources` 源分布机制，边界条件改为
  `V·n = (Ω×r)·n`（单位角速度驱动），对表面压力矩积分得附加转动惯量。
- 解析锚（Lamb, Hydrodynamics）:
  - 球（半径 a, ρ=1）: 绕任一轴 I_add = (5/8)·π·a^5；
  - 长球 a/b=2（绕对称轴/赤道轴）: 用 Lamb 椭球公式（与 13305 平移项同族）。
- 量化验收: 最细面元级误差 <3%；Richardson 外推 <1%；矩阵对称/去耦 <1e-10；
  观测收敛阶 p≥1；论文级报告（六要素）。
- 建议总账条目: `GAP-FVM-ADDED-MASS-ROT`（13305 已建议 GAP-FVM-ADDED-MASS，
  本条为其转动扩展；若总账已收 GAP-FVM-ADDED-MASS 则并入）。


> **D1 已结（m14707, 2026-11-11）——且原锚点被证伪。** 原文提出的球体锚点
> `I_add = (5/8) pi rho a^5` 不是理想流体附加转动惯量：球面上 `r x n = 0`，
> 故转动 Neumann 数据 `(Omega x r) . n = 0`，外域问题只有 `phi = 0`，
> 流体动能与压力扰动同时为零 → **球绕任意直径的附加转动惯量恰好为 0**
> （Fossen：球的附加质量矩阵 = diag(m_a, m_a, m_a, 0, 0, 0)）。
> 若按原锚点验收，正确实现会被判 100% 错。已改为下列正确锚点族并交付：
> 球的平行轴锚 `I_oo = m_a d^2`（d=1.5a，Richardson 后 0.22%）、
> 回转体横摇 `I_44 ≡ 0`（5e-34）、球绕直径 `≡0`（面元残差 1.1e-5）。
> 证据：`docs/verification-rotational-added-mass/{REPORT.md,summary.json}`、
> `tests/test_rotational_added_mass.py`（7 测全绿）、
> `src/tensorfvm/panel_added_mass.py::added_mass_matrix_6dof`。
> 明示未完成：长球横摇/艏摇块无解析式，只做了结构性验证 → 见 D6/D7。

### D2 自由面/双体镜像附加质量验证
- 来源: 报告 13305 §6（`panel_potential` 已有 single_hull/twin_hull 镜像，未对解析锚验证）。
- 量化验收: 半浸没球（双体镜像）平移附加质量 vs 镜像解析解 <3%；
  水平面 φ=0 边界残差 <1e-8。

> **D2 已结（m15998, 2026-11-12）——判据兑现，且 D2 原文的第二句被证据改写。**
> 新增 `src/tensorfvm/verification/image_added_mass.py`（验证驱动，六项判据 C1–C6）
> + `tests/test_image_added_mass.py`（15 测全绿）+ `docs/verification-image-added-mass/`。
> 解析锚（闭式，先冻结后跑）：切平面**平行**方向取"双体=刚体平移"，半体
> `m_full/2`（球 `pi rho a^3/3`，椭球用 Lamb `k_axial/k_transv`）；切平面**法向**
> 取"双体=拉伸运动 `w=U|cos theta|`"，闭式 `m_stretch = 4 pi rho a^3 sum c_n^2/((2n+1)(n+1))`，
> 半体 `m_stretch/2 = 1.7403`（比平行向大 66%，"半刚体附加质量"的取巧做法低 40%）；
> 四分之一体 `k_axial rho V/4`。实测（最细网格 32x24, 768 面元）：球平行向
> 0.14%/0.25%（Richardson 后 0.26%/0.07%，观测阶 1.89/1.66）、法向 0.50%、
> 椭球 a/b=2,4 最差 0.60%、四分之一体最差 1.36% —— **全部 <3%**。
> 镜像面刚性壁残差 `max|u_z|/U = 0.0`（浮点零），而壁面势 `max|phi|/(U a) = 0.51`
> —— 故 `IMAGE_SPECS` 的镜像全是**同号（Neumann，静水/零频刚性壁）**，
> D2 原文要求的 "水平面 `phi=0` 残差 <1e-8" 描述的是**未实现**的 ω→∞ 自由面模型
> → 记为 NOT_SUPPORTED（实现它需要在 `influence_matrix`/`source_potential` 加逐镜像符号），
> 不再当作验收句。
> 顺带修掉一个真缺陷：`influence_matrix` 过去**无条件**把"自面板"项清零（那只是
> 场点=面板形心时的平直面板主值），对任意场点集（壁面探针/势函数评估）会白丢
> 每场点一个真实面板贡献（壁面残差修前 7.8e-3，修后 0.0）；已加
> `field_is_panel` 判定并有回归测。
> 明示未做：镜像下的转动块；半椭球法向（无闭式 → WITHHELD，面板值只报不判）。

### D3 总账 GAP-FVM-TWOPHASE 状态复核（轻量，半天）
- 现象: 总账 planned，但仓内 two_phase/vof 测试面已很厚（test_two_phase_*、
  test_vof_* 共 30+ 文件，含 P400W2-N01 验证面 13233）。
- 做法: 逐条对总账验收句（VOF/界面捕获验证）核实现状 → 要么升账 done+证据链，
  要么列出剩余缺口清单。产出=总账更新建议 + 证据指针。

### D4 KPI-QC-RECHECK 的 FVM 份额（可单干）
- 来源: 总账 KPI-QC-RECHECK（已合并功能 100% 独立参考复核）。
- 做法: 抽 2-3 个已合并 FVM 功能（如阻力链 KCS/KVLCC2 正典值 13281、
  自然对流 13277）做独立复算，误差逐项 <3% 判定。

### D5 自然对流 Ra~1e8 振荡起始 / 3D 腔（重算力，需拆单）
- 来源: 报告 13277 §5。超出单干范围 → 若认领走 POST missions 派单（runner=native）。


### D6 长球横摇/艏摇附加惯量的解析锚（D1 的 honest 未完成部分）
- 现象: `I_55` a/b=2 收敛到 `I_55/(rho V a^2)=0.0584`，两级网格不确定度 1.0%，
  但**没有解析式可对标**；观测阶估计被纬度分辨率噪声污染（诊断值 5.57 不可信）。
- 做法: 均匀面积/等弧长网格 + 椭球调和函数闭式（`phi ∝ x y ∫_lambda^inf ...`），
  或等价的椭球转动势闭式，得 `I_55(a,b)` 解析值再判 <3%。
- 量化验收: 解析 vs 面元 <3%；观测阶 p≥1；a/b=2,4,8 三点一致。
- 现状: 已有的结构性验证（轴对性 <1e-15、互易 <1e-16、近球 (a^2-b^2)^2 消失、
  细长极限单调趋向 strip 值 0.2）留作回归护栏。
> **D6 已结（m14810, 2026-11-11）——解析锚已建立并交付，判据全 PASS。** 新增
> `src/tensorfvm/ellipsoid_added_inertia.py`：三轴椭球绕主轴转动附加惯量的**闭式解**
> （椭球坐标二次 Lamé 势 `phi = C2 x y FF(lambda)`，振幅由 Neumann 数据定出，
> 表面积分闭式 `I = (2 pi/15) a^3 b^3 c`）；唯一数值成分为一维积分 `FF(0)`（quadrature
> 1e-13，另有 prolate 闭式独立互证 1e-15）。验证驱动
> `python -m tensorfvm.verification.ellipsoid_rotational_reference`（9/9 判据 PASS）：
> 闭式势的**数值法向导数**复现转动数据残差 1.76e-10（不依赖推导自身代数）；调和性
> 4.9e-8；四极子衰减 8e-4；细长极限 a/b=100 → 0.1995（strip 1/5，误差 0.05%）；
> 近球 (a^2-b^2)^2 律 1.23%；**面元 vs 闭式 a/b=2/4/8 最细网格误差 2.39%/1.86%/0.42%
> —— 全 <3%，D1 的「解析 <3%」判据兑现**。证据
> `docs/verification-ellipsoid-rotational-reference/{REPORT.md,summary.json,fig_rotational_reference.png}`、
> `tests/test_ellipsoid_rotational_reference.py`（15 测全绿）。D7（接进
> `maneuvering.dynamics.J_z`）仍未做。


### D7 转动附加质量接进 maneuvering.dynamics 的 J_z（能力落地）
- 现象: `maneuvering/dynamics.py` 的 J_z 仍用细长体公式；本单提供了
  `rotational_added_inertia(panels)`，但尚未接线，也没量细长体公式的高估幅度。
- 做法: 用面元 `I_44/I_55/I_66` 替换细长体 J_z 或给出修正系数，并做操舵响应对比。
- 量化验收: 同一船的 J_z 两法差异报告 + 回转直径/初转期变化 <15% 才可换默认值。


### D11 CASE-FVM-PERIODIC-HILLS 第二回合（m16503, 2026-11-12）—— 四个装配缺陷已修，锚点仍 OPEN
- 起因: `docs/case-periodic-hills/README.md` round-1 honest FAIL，其诊断（"momentum 对角 ~265x 过刚"）不完整。
- 交付: `src/tensorfvm/periodic_hill.py` 五处修复 + `src/tensorfvm/verification/periodic_hill_round2.py`
  （8 条预声明判据 C1–C8）+ `tests/test_periodic_hill_round2.py`（8 测，连同旧 7 测共 15 绿）
  + `docs/case-periodic-hills/{ROUND2.md,round2_summary.json,fig_round2.png}`。
- 四个真缺陷: ①`diag_mom += coef`（(nx,) 广播进 (ny,nx)）→ 对角 56208 vs off-diag 892（16x64）；
  ②`c_n[0]`/`c_n[-1]` 从未赋值 → 壁面无粘项恒 0，**无壁面摩擦**（round-1 栓塞流 u_max/u_b=1.0031 的真因）；
  ③压力算子 off-diag 除体积、diag 不除 → 同一行两半差一个 Vol；④周期接缝未取最小像（8.4375 vs 0.5625，接缝扩散弱 15x）。
- 关键数字: 平面槽道 vs 解析 Poiseuille —— u_max 相对误差 **2.57e-4**、u_max/u_b 1.34717 vs 解析 1.34748
  （**注意 ERCOFTAC bulk 只积 [H,3.035H] 带，1.5 是满高平均，用 1.5 判会判错正确实现**）、
  G 0.011690 vs 2νu_b/H²=0.011686；flux 级质量守恒 3.3e-16（round-1 山丘首步 2.33）。
- 反例/踩坑: ①奇偶(棋盘)模对"面平均散度"和"宽 LSQ 梯度"都**精确不可见**（0.0/0.0）→
  `div(grad)` 型 Poisson 有半个谱的零空间，Rhie-Chow 是必需而非可选；②把 Rhie-Chow 项做成隐式会
  退化（1D 下 M+N(G)=O(h³)，前导项相消）；③修好对角而不补壁面项只会得到"更刚的栓塞流"。
- 未完成(honest FAIL): Re=2800 场解仍不收敛——48x24 第 293 步、64x32 第 156 步发散（越细越早，
  dt 0.02/0.01/0.005 与 blend 1.0/0.75/0.5/0.0 全试过，非 CFL）；故 x_r/H、x_s/H 不报。
  剩余症结=地形贴体网格的非正交延迟修正（或 MAC 交错），已列为 round-3 首项。

## 3. 给下次自找活的建议线索
1. 首选 D1：单干、解析锚明确、直接喂 maneuvering.dynamics 的 J_z 缺口，
   且总账无对应锚（新缺口，finish 时申报 NEW 条目）。
2. D3 是纯账目活，任何席位可领，半天内可交。
3. 选题前必跑总账 SQL（owner 2026-11-09 令）：
   `SELECT slug,title,acceptance,status FROM feature_ledger WHERE status IN('planned','in_progress')`。
4. 本仓无 backlog_*/expand_audit_* 文件，本文件即首个缺口账；后续欠账 append 到 §2，
   完成移 §4（append-only，不删旧条）。

## 4. 已完成（done，不得重做）
### D6 长球横摇/艏摇附加惯量的解析锚 —— 已交付（m14810, 2026-11-11）
- 实现: `ellipsoid_added_inertia.{rotational_added_inertia, harmonic_integral,
  harmonic_integral_spheroid, surface_product_moment, rotation_potential,
  neumann_data, ellipsoidal_coordinate}`（闭式解，无面元依赖）。
- 验证: `python -m tensorfvm.verification.ellipsoid_rotational_reference`；9/9 判据 PASS；
  产物 `docs/verification-ellipsoid-rotational-reference/{REPORT.md,summary.json,png}`。
- 关键数字: 闭式 vs 面元 a/b=2/4/8 → 2.39%/1.86%/0.42%（最细网格 2176 面元）；
  三点外推（仅用面元序列）复现闭式值 0.28%/0.20%/0.69%；细长极限 a/b=100 → 0.1995
  （strip 1/5）；球与长球横摇严格 0。
- 反例/踩坑: ①`FF(lambda)` 的调和结构靠恒等式 `D = |grad F|^2/4`，符号取错
  （`d lambda/dn` 应为 `+4/|grad F|`）会把细长极限算成 1/15 而非 1/5——用 strip 极限
  与面元双向校核才发现；②`quad` 截断到有限上界会低估 `FF(0)` 达 31%（尾部为 t^-7，
  局部误差估计看不见），必须积分到 inf；③转动势是**二阶**调和（`phi ~ r^-3`），
  衰减判据应是 phi(2r)/phi(r)=1/8，不是偶极的 1/2。


### D1 转动附加惯量 M_44..M_66 —— 已交付（m14707, 2026-11-11）
- 实现: `panel_added_mass.added_mass_matrix_6dof / rotational_added_inertia /
  surface_generators / sphere_offset_axis_added_inertia / sphere_panels_offset`；
  `panel_potential.solve_sources(..., normal_velocity=...)`（非均匀 onset 正态速度入口，
  均匀来流路径行为不变）。
- 验证: `python -m tensorfvm.verification.rotational_added_mass`；
  6/6 判据 PASS；产物 `docs/verification-rotational-added-mass/{REPORT.md,summary.json}`。
- 关键数字: 球绕直径 1.14e-5（相对 m_a a^2，理论 0）；长球横摇 5.1e-34（理论 0）；
  偏置轴球 `m_a d^2` 面元 1.46% / Richardson 0.22%；长球横摇块两级网格 1.0%；
  细长极限 a/b=2/4/8 → 0.0565/0.1227/0.1688（strip 0.2）。
- 反例资产: 原 D1 锚点 `(5/8) pi rho a^5` 已证伪并入档（球转动附加惯量恒为 0）。
> **D2 已结（m15999, 2026-11-12）——双体镜像附加质量对精确镜像锚结算，7/7 判据 PASS。**
> 关键结论：**镜像装配在机器精度上是精确的**——半体（开镜像）解与整体（无镜像）解是同一组方程，
> 配对残差 `|2 M_half - M_full|/M_full = 4.7e-14`（a/b=1/2/4 全族）；水线 Neumann 残差恒为 0
> （成对消去）；双体片体 `twin_hull` 与独立两体解相对差 3.6e-15，片体耦合按 `s^-3` 衰减
> （拟合斜率 −3.001）。**解析锚**：被壁面切开的半球/半长球绕**平行于壁面**的平移
> `M_half = ½ M_full` 精确成立，面元 vs Lamb 闭式误差 0.07 %/0.66 %（a/b=1）、
> 0.48 %/1.00 %（a/b=2）、0.46 %/0.22 %（a/b=4），全 <3 %。
> **honest WITHHELD**：法向平移 `M_zz` 无锚——壁面镜像反射法向速度，重组体上下半朝相反方向法向运动，
> 不是整机体刚性平移；naive「½ 整机体」锚与正确值差 102 %（C5 量化），故只报值不判精度。
> 证据 `docs/verification-mirror-added-mass/{REPORT.md,summary.json,fig_mirror_added_mass.png}`、
> `tests/test_mirror_added_mass.py`（9 测全绿）、`src/tensorfvm/verification/mirror_added_mass.py`。
> 附带发现：`panel_potential.influence_matrix` 的平面对角「主值置零」按**索引**实施，
> 仅在场点=同一面板集质心时正确；用任意独立场点求值时每点会静默丢掉一个面元贡献（本单实测偏差 1e-3），
> 已在验证脚本里改用自带核并留档（库函数本身未改，避免动已发布的数字）。
### D2 自由面/双体镜像附加质量验证 —— 已交付（m15999, 2026-11-12）
- 实现: `verification/mirror_added_mass.{hemi_spheroid_panels, waterplane_residual,
  half_body_added_mass, pairing, twin_hull_demihull, two_body_demihull, run}`；
  测试 `tests/test_mirror_added_mass.py`（9 测）。
- 验证: `python -m tensorfvm.verification.mirror_added_mass`；7/7 判据 PASS（60 s）。
- 关键数字: 镜像配对残差 4.7e-14；水线 Neumann 残差 0.0；twin_hull vs 两体解 3.6e-15；
  片体衰减斜率 −3.001；面元 vs ½ Lamb 误差 0.07–1.00 %（<3 %）。
- 未完成/受限: 法向平移 M_zz 无解析锚 → WITHHELD（*1/2 整机体*错 102 %）。
- 踩坑: ①配对必须单元对齐（半体取 n_theta/2），否则 2% 的离散差会被误读成镜像错误；
  ②`influence_matrix` 的按索引主值置零对任意场点不成立（偏差 1e-3，会伪装成边界条件失效）。
> **D7 已结（m14901 + m15859 前序，本单 m16116 核对）——本条目未及时注销，差点被重做。**
> 面元 J_z 已交付：`maneuvering/added_inertia_panel.py`（KVLCC2 封闭面元 + 盖面零通量闭合，
> 几何门 Cb/Cwp 通过）→ `J_z(∞) = 4978.2 kg m²` vs 细长体 `13411.1`（**细长体高估 169%**，
> 比值 0.371，幂律拟合 q = 0.67）；`dynamics.ManoeuvreModel(..., j_z_override=...)` 已接线
> （默认 None 保持细长体）；35° 回转 / 10-10 蛇形逐项对比最大变化 10.93%（< 15% 验收门满足，
> 但**仍不改默认值**，理由见报告）。证据 `docs/maneuvering-added-inertia/{REPORT.md,summary.json,fig*.png}`、
> `docs/verification-added-inertia-panel/`、`tests/test_maneuvering_added_inertia.py`。
> 教训：本条 DONE 状态滞后 ≈ 一天，扫描期差点重开同一活——完成条目必须当天 append 注销。
## 4b. 追加欠账（append-only）
### D8 表面辐射（视图因子 / 灰体辐射腔 / 天穹长波）验证面 —— 已交付（m16116, 2026-11-12）
- 缺口来源: `docs/verification-catalog.md` 第 10.1 节 gap 5「Radiation: No heat transfer
  verification」长期 `[[gap-status: open]]`；仓内只有 `cht/ice_conjugate` 的天穹长波标量项，
  无任何面-面辐射能力与验证。
- 交付: `src/tensorfvm/radiation.py`（二维线段视图因子 = Hottel 交叉线法闭式；同心圆对
  闭式 `F12 = 1, F21 = r1/r2, F22 = 1 - r1/r2`；灰体辐射腔辐射度解法，含再辐射绝热面；
  天穹等效温度/天空压降诊断）+ `src/tensorfvm/verification/radiation.py`（11 条预声明判据）
  + `tests/test_radiation.py`（13 测全绿）+ `docs/verification-radiation/{REPORT.md,summary.json,fig_radiation.png}`。
- 量化结果: 交叉线法 vs 闭式 0.0e+00（平行板 5 组、垂直共边 4 组）；方腔 `F_opp = √2−1`、
  `F_adj = 1−√2/2` 精确，行和 1（1e-14）；五边形腔 vs 独立 Gauss-Legendre 核积分 2.55e-05；
  灰体双面腔 vs 闭式 3.7e-16 + 能量闭合 3.7e-16；再辐射黑板 `T2⁴ = (T1⁴+T3⁴)/2` 精确；
  结冰长波项 `dq/dT = 4εσT³` 7.4e-08（FD 底噪）。
- 反例/踩坑（已写进模块 docstring）: ①圆环（双连通）域**不能**用交叉线法拼——内圈行的有符号
  贡献互相抵消为 0（应为 1），任何走向约定都救不回来，同心圆必须走闭式；②片段走向必须统一
  「辐射域在左手侧」，取绝对值会让凹边界重复计数（实测行和 2.0 而非 1）；③天穹关联式之间
  −20 °C 时互差 30%（+10 °C 仅 1%）→ 结冰长波项带模型形式不确定度，记为 WITHHELD。
- 明示未做: 三维视图因子、遮挡（部分阻挡）扣除、参与介质/谱线、有任何辐射实验数据的对标。

### D9 物种输运 / 混合分数 / Burke–Schumann 验证面 —— 已交付（m16264, 2026-11-12）
- 缺口来源: `docs/verification-catalog.md` §10.1 gap 4「Combustion: No species transport
  verification」长期 `[[gap-status: open]]`；仓内 `compressible` 的 "species" 只是双材料编号，
  没有任何质量分数输运能力。
- 交付: `src/tensorfvm/species.py`（Fick + 零和通量修正；一维有限体积稳态/瞬态输运；Bilger `Z`；
  Burke–Schumann 闭式解含焓一致温度）+ `src/tensorfvm/verification/species_transport.py`
  （24 条预声明判据）+ `tests/test_species_transport.py`（23 测全绿）+
  `docs/verification-species-transport/{REPORT.md,summary.json,fig_species_transport.png}`；
  catalog 已关闭 gap 4，并附加 §16 条目。
- 关键数字: 瞬态 vs erfc 2.8e-06（阶 2.00）；稳态对流-扩散 central 1.1e-04（阶 2.00）/upwind
  8.7e-03（阶 0.98）；应变混合层 vs erf 5.9e-05（阶 1.93）；火焰面驻定 3.5e-06；
  `Z_st=0.05496` vs 文献 0.055 = 0.067%；`Z` 反应不变性 5.6e-17；ΣY 漂移 2.1e-15。
- 反例/踩坑: ①单标量开 zero-sum 修正会把通量乘 `(1-Y)`（误差 1.1e-5→0.99），库已改为 `ValueError`；
  ②一维守恒形式漏横向对流项 → rel-L2 1.4e+08（数值爆炸，不是几十 %）；
  ③`CNUB` 少乘 1e3 会让"火焰"只有 300 K（J/mol ÷ g/mol = kJ/kg）；④扁平键 `Y_O` 是氧化剂物种
  而非元素 O，误用会得到 14.5% 的假失配 → 现返回按物种名索引的 `Y` 视图。
- 明示未做（WITHHELD）: 有限速率 Arrhenius 化学与详细机理；`GAP-FVM-COMBUSTION-SPECIES`
  提案的层流火焰速度 vs Cantera 外部锚（本环境 pip 索引无 cantera → 装不上）；`T_ad` 与文献
  2226 K 的对比（常数 cp + 无离解，实测 2589 K，不作判据）。
- 顺带修的一处陈旧索引: catalog gap 5（Radiation）早已由 m16116 交付却仍标 open，
  本单补上 `closed-by`（gate 只查死链，查不出"陈旧 open"）。

### D10 CASE-FVM-KOVASZNAY-ORDER 的 FAIL 归属（m16396, 2026-11-12）—— 已归属，锚点仍 OPEN
- 起因: 交付轮 m16130 用**单一速度 L2 范数**判 1.684 < 1.8 → honest FAIL；m16173 交差审计指出
  原始 cases.json 缺失、无法独立复算。
- 交付: `scripts/audit_kovasznay_order_components.py`（误差预算三分解）+ `tests/test_kovasznay_order_budget.py`
  + `docs/verification-kovasznay-order-attribution/{REPORT.md,summary.json,fig_order_budget.png,make_report.py}`
  + `runs/kovasznay-order/`（n=16/32 复现产物）。
- 关键数字: 交付轮阶梯**逐位复现**（n16 1.2575e-02 / n32 3.8397e-03）；同一解拆范数后
  **面内 (u,v) 阶 1.808（过 1.8 门）**、**挤出向 w 阶 1.106**（w 精确值恒 0）、total 1.711；
  w 误差平方份额 8.7%(n16) → 20.1%(n32)（网格越细越主导）；勾股闭合 1.1e-16。
- 结构探针(n=8): 挤出向 cell-w 自由度行和 0.30–0.40、零行 0 → **不是无质量自由度**，
  是真被离散耦合驱动的 O(h) 伪速度。
- 未完成: 未定位 z 动量里产生该 O(h) 驱动力的具体项（未改求解器）；`planar` 无挤出对照
  在 9600×9600 耦合直接解上超时 → "去掉挤出向即二阶"记 WITHHELD；n=64 未跑。
- 给下次: 在 `native_face_mean3d` / thin-layer viscous 两侧拆项做 n=8 的逐项残差探针，
  定位后或改离散、或在验证口径里把 w 预算单列（本单已提供分解工具）。

### D12 浅水变床面（bed + well-balanced）能力缺失 —— 已交付（m16596, 2026-11-12）
- 缺口来源: `shallow_water/swe1d.py` docstring 明写 "flat bed here"；仓内无任何
  `[[gap-status: open]]` 覆盖它（catalog 全部已闭合），属总账外新缺口（NEW）。
- 交付: `src/tensorfvm/shallow_water/bed.py`（Audusse 2004 静水重构 + HLL + MUSCL + SSP-RK3；
  `well_balanced=False` 保留朴素中心源项作负对照）+ `src/tensorfvm/verification/shallow_water_well_balanced.py`
  （19 条预声明判据 + 3 条舍入复判）+ `tests/test_shallow_water_well_balanced.py`（11 测全绿）
  + `docs/verification-swe-well-balanced/{REPORT.md,summary.json,criteria.md,fig_well_balanced.png}`。
- 关键数字: 静止湖（光滑/台阶/干湿）max|u| = 5.0e-14 / **0.0** / 2.6e-14 m/s，η 漂移 ≤5.6e-15，
  体积漂移 ≤6.3e-15；**负对照**（朴素中心源项）在台阶床上 max|u| = 3.94e-01 m/s（超门 39 倍）⇒ 判据非空；
  z=0 与已发布平床求解器**逐位同解**（0.0/0.0，276 步）；常床面偏移 1.8e-15/9.7e-15。
- 附带缺陷修复: `ShallowWater1D.cfl_number()` 原返回 `max_wave_speed()*dx/dx` = 波速（m/s，量纲错），
  现返回真实 Courant 数（`dt_last` 记账），回归测 `test_cfl_number_is_a_dimensionless_courant_number`。
- 反例/踩坑: ①"常床面不改变解"不能写成 `==0`——`η-z` 往返有 eps·|h| 舍入，实测 1.8e-15（判 FAIL 后
  以推导的舍入门 1e-11 复判）；②静水重构下动量通量**刻意不守恒**（压力修正两侧不同），只有体积守恒；
  ③干胞里每级会有舍入级 hu 被清零（`dry_flips` 1.2e5 次），是簿记量不是失稳。
- 明示未做（WITHHELD）: 摩阻/科氏力、2-D、Goutal–Maurel 跨临界堰流半解析锚、变床面光滑解的空间收敛阶。

## 4c. 追加欠账（append-only，2026-11-12）

### D13 CASE-FVM-PERIODIC-HILLS 第三回合（m16672, 2026-11-12）—— 梯度算子不一致（缺陷 vi/vii）已修，Re=2800 仍 honest FAIL
- 交付: `periodic_hill.compact_gradient` 两处修复 + `src/tensorfvm/verification/periodic_hill_round3.py`
  （6 条预声明判据 C1–C6，全 PASS）+ `tests/test_periodic_hill_round3.py`（7 测，连同 round-2/round-1 共 22 绿）
  + `docs/case-periodic-hills/{ROUND3.md,round3_summary.json,fig_round3.png}`。
- 缺陷 vi（南面索引错位，round-2 未发现）: eta 面项写成 `c_n[j] n[j] (phi[j+1]-phi[j])`，行 0 用壁面系数乘内部跳量、
  行 ny-1 丢掉南面 → 算子在任何网格上都不收敛（实测 18.8/19.2/19.3 % 于 32x16/64x32/128x64，比值 0.973）；
  改为按**层号**配对、两个 Neumann 壁层既不进 b 也不进归一化矩阵。
- 缺陷 vii（两侧面族都无非正交修正）: `|beta/alpha|` 最大 0.843（eta）、`|beta|/dn_e` 0.843（东面），
  两点法一阶偏差在丘区是 O(1) 相对误差；加标准显式延迟修正 `g_n = jump/alpha - (beta/alpha)(grad phi).t`。
- 陷阱: 延迟修正**不能**从自身输出迭代（|beta/alpha|~1 时非压缩，实测误差 1.9e-1→2.1e-1→3.0e-1），
  round-2 计划的“两次 sweep”写法会失败；必须用宽 LSQ 梯度作切向估计。
- 关键数字: 修正后梯度误差 2.49/0.87/0.30 %（比值 2.86/2.85 → p=1.51，一阶）；最细网格增益 63x；
  平面槽道 `u_max/u_b` 1.34547 vs 解析 1.34637（0.067 %，<3 %）；`max|div_cell|` 4.0e-16。
- 未完成(honest FAIL): Re=2800 48x24 第 500 步 max|u|=1.86、第 1000 步 NaN（64x32 500 步内 NaN）；
  `x_s/H`、`x_r/H` 继续 withheld。round-2 的“第 293 步发散”与本回合不可直接比较（旧算子带 O(1) 梯度误差=数值阻尼）。
- 下一回合首项（round-4）: 交错（MAC）布置使单元速度=面速度，或对压力梯度修正做限制器；不要再用 `div(grad)` 型一致化。
### D9 Hysing 腿修复 —— 已处置（m16730, 2026-11-12），形状门仍未结
- 结论: **"气泡下沉"归因被证伪**。三条缺陷全是测量/几何，不是物理：
  ①`bubble_metrics()` 内部已取反、调用点又传 `1-alpha` → 量的是液相（液相 c=5.645 vs 气相 0.638，
  与旧 FAIL 记录的 `c=5.621` 对上）；②`nx = cells_per_diameter`（应 `dx = d/cpd`）；
  ③`start_y = 0.5 = r` 使气泡底缘贴壁（Hysing 为 1 d）。
  附带：子格面积分数初值（初体积偏差 3.45%→0.96%）、marching-squares 周长（数字圆盘 c 0.638→0.993）。
- 修后 cpd=8: 峰值 U_T 0.2451 vs 0.2342 = **+4.6 %**；y_c(3)=1.5225 vs 1.984（−23.3%）；
  c_min 1.0026 vs 0.4644（**FAIL，分辨率受限**）；最差单步体积漂移 7.2e-6（门 1e-6，FAIL）。
- 证据: `docs/GAP-FVM-TWOPHASE/HYSING-LEG-REPAIR.md`、`docs/GAP-FVM-TWOPHASE/artifacts/leg_hysing/`、
  `docs/GAP-FVM-TWOPHASE/artifacts/fig_hysing_leg.png`、`tests/test_hysing_gate.py`(26 passed)。
- 遗留: B1 度量错配（ref 是全程最大、门用末窗均值）→ 只加记 `peak_velocity`，不改门；
  形状门需 cpd≥32（显式气相黏性 dt∝h²，cpd=32 ≈10 h）→ 记分辨率受限；`DECLARED_CPD` 由 64 改 16。

### D14 脉动（Womersley）非定常验证面 —— 已交付（m16821, 2026-11-13）
- 缺口来源：`PeriodicChannelSolver` 的**定常** Poiseuille 已对解析抛物线验证，但**非定常**行为
  （时间精度/相位关系/振荡驱动下的壁面切应力）全无验证面；catalog 无对应条目，属总账外新缺口（NEW）。
- 交付：`PeriodicChannelSolver.step(body_force_m_s2=None)`（逐位置时变体积力；`None` 与旧常量路径
  逐位同解，A5 + 回归测把关）+ `src/tensorfvm/verification/pulsatile_channel.py`（16 条预声明判据）
  + `tests/test_pulsatile_channel.py`（12 测全绿，53 s）+ `docs/verification-pulsatile-channel/`
  `{REPORT.md,criteria.md,summary.json,fig_pulsatile_channel.png}`；catalog 新增 §17。
- 关键数字：闭式解数值残差 2.7e-9（数值微分，不依赖推导）；`omega->0` 归抛物线 1.1e-9；
  Stokes 层衰减律斜率 `slope*delta` = −0.9959/−0.9993（W=40/80）；生产求解器 vs 闭式剖面
  **1.54 %/0.61 %/0.39 %**（W=2/5/10，ny=32，dt/T=1/200，8 相位，按周期包络归一）；流量幅值
  0.33 %、相位差 0.11 deg；壁面切应力幅值 1.65 %（ny=64 三点单边）；格式自洽一阶壁面通量 8.2e-4；
  时间收敛阶 **0.875**（一阶，理论自洽）；ny/2→ny 误差 7.6e-3→6.1e-3；`max|v| = 0`、`max|div| = 0`；
  负对照（振荡 vs 定常剖面）差 134 %。定常回归 ny=32/64 误差 1.95e-3/4.88e-4（O(dy^2)）。
- 反例/踩坑：①采样窗口含 `t+T` 重复端点 → 单频 DFT 幅值低 7.6 %（假 FAIL），去重后 0.33 %；
  ②按逐相位范数归一化在过零点放大成 5.6 % 假误差 → 改按周期包络归一；③收敛阶公式分母写反会得
  p=−0.87（假 FAIL）；④`omega H^2/nu=1e-6` 时 `1−cosh/cosh` 相消，舍入地板 1e-10，判据阈值应为 1e-6。
- 明示未做：壁面振荡驱动（Stokes 第二问题，求解器壁面为齐次无滑移 → NOT_SUPPORTED）；
  二阶时间格式；湍流/可压缩/可变形槽道；实验数据对标（层流振荡槽道无更好基准，闭式解即精确解）。

### D10 续（m16864, 2026-11-13）—— 机制已定位且可修：挤出面缺对称闭合（不 是方程不一致）
- **结论**：交付轮 1.684 的根因是**算例边值只做了一半**。`kovasznay_solver(..., symmetry_extrusion=True)`
  只把 `w` 钉在 `z` 平面上，切向 `u,v` 被设成自由未知量（= traction-free），而对称面要求
  `du_t/dz = 0`（Neumann）。实测自由面值与 owner 单元值差到 0.07（n=16），两点法向导数因此看到
  0.14 的假剪切，粘性交叉项 `mu(du_i/dz + dw/dx_i)` 在面内面上给出 O(h) 源项 → 一阶伪 `w`。
- **逐项证据**（`spanwise_drive_terms`，收敛态把 `w` 置零后重建）：对流 0、对流源 0、压力 0、
  同分量粘性 0，**全部残差 = 粘性交叉分量**（交叉/残差 = 1.000000，n=6/16 两级）；
  解析态代入同一批挤出向行残差 1e-18 ⇒ 方程本身是自洽的，问题在边值闭合。
- **修复**：同一批面交给既有 `zero_gradient_faces` 机制（`symmetry_closure='zero_gradient'`）。
  修复后 n=16/32：`max|w|` 1.8e-16/1.6e-16（机器零），`spanwise_l2` 1.5e-17/6.8e-18，
  in-plane 预算不变（1.2033e-2 / 3.4356e-3，相对差 0.15%），**速度 total L2 阶 16→32 = 1.808 ≥ 1.8**。
- **负对照**：`free` 闭合 n=16 复现交付轮逐位数字（1.257478e-02 / 1.201574e-02 / 3.707731e-03），
  唯一性对照三初值（含 ±0.1 随机扰动）解出的 `w` 位级一致 ⇒ 排除"零空间污染"。
- **产物**：`src/tensorfvm/verification/kovasznay_symmetry.py`（C1–C8 预声明判据，全 PASS）、
  `tests/test_kovasznay_symmetry.py`（10 测全绿，19 s）、`docs/verification-kovasznay-symmetry/`
  `{REPORT.md,summary.json,fig_kovasznay_symmetry.png,make_report.py}`、`runs/kovasznay-symmetry/`；
  `kovasznay_solver` 新增 `symmetry_closure`（默认 `'free'` 保历史口径，**新算例应用
  `'zero_gradient'`**）。
- **账本动作建议**：`CASE-FVM-KOVASZNAY-ORDER` 的验收条款（速度 L2 阶 ≥1.8）在正确闭合下达成
  → 建议升 `done`，并把"挤出/对称面必须用 zero-gradient 闭合"写进算例规格；仓内默认值未改，
  故任何仍用 `'free'` 的旧结果都保留可复现。D10 原文"未定位 z 动量里产生该 O(h) 驱动力的具体项"至此结清。
- **明示未做**：n=64 三级阶；`planar`（无挤出全 3D 立方）对照仍未跑（9600² 耦合直接解超时问题不变，
  本单用 1e-18 的解析态缺陷替代该对照的作用）；未把二阶壁面闭合（`NativeViscous3D.wall_closure=
  'three_point'`）接到 hybrid 粘性路径（本单证伪了"壁面闭合是元凶"——元凶是 `z` 平面切向自由度）。

### D15 Stokes 第二问题（壁面振荡驱动）—— 已交付（m16924, 2026-11-12）
- 来源: D14 明示 NOT_SUPPORTED 项（`PeriodicChannelSolver` 壁面原为齐次 no-slip，无法提壁面驱动）。
- 能力: `step(wall_velocity_m_s=...)` / `set_wall_velocity()` —— 非齐次 no-slip（half-cell ghost
  `2u_w-u_0`、quadratic ghost `-2u_0+u_1/3+(8/3)u_w`），壁面贡献为隐式壁行常源 `coef·u_w/dy²`；
  `wall_shear`/`dissipation` 同口径减 `u_w`。默认 (0,0) 逐位复现历史齐次壁结果。
- 验证: `src/tensorfvm/verification/stokes_second_problem.py`（19 条预声明判据 **19/19 PASS**, 39.9 s）
  + `tests/test_stokes_second_problem.py` + `docs/verification-stokes-second/{REPORT.md,criteria.md,summary.json,fig_stokes_second.png}`。
- 关键数字（ny=64, dt/T=1/200, W=H/δ=2/5/10）: 剖面 L2 相对闭式 **0.735%/0.682%/0.760%**；
  bulk 幅值 0.067%、相位 0.385°；壁面切应力三点估计 2.64%（方案自洽通量 4.6e-3%）；
  非定常动量收支残差 **3.8e-15**（ρU0Hω 归一）；时间阶 1.025、壁法向阶 0.662（quadratic 闭合）；
  定常运动壁 = Couette 1.2e-11（两种闭合）；负对照（振荡 vs 定常壁）差 282%。
- 附带缺陷修复（性能）: 每步三对角求解 Python Thomas → LAPACK 批解（n>256 回退），
  单次 11.7 ms → 0.94 ms（n=64），单步 40 ms → 5.6 ms；等价性专测 1.4e-16；
  `test_channel_periodic.py` 43 s→23 s、`test_pulsatile_channel.py` 53 s→21 s。
- 踩坑: ①离散动量收支必须**逐步**记账且含**两侧**壁面通量（漏上壁得 9.3% 假残差，W 越大越小=漏项特征）；
  ②启动瞬变慢（νπ²/H²），5 时间常数≈93 s，W=10 需 ~30 周期；收敛研究改用闭式初值；
  ③`half-cell` 一阶壁面闭合 + 时间误差同量级会把空间阶压到 0.134（假 FAIL），需 quadratic + dt/T=1/1000；
  ④`μ∂u/∂y` 与教科书「+45°」差 180°（后者是平板受的力），判据按平板反力口径写。
- 明示未做: 壁面法向运动/行波壁（`v` 钉零，NOT_SUPPORTED）；多频/随机驱动；二阶时间格式；
  可压缩/湍流/变物性；实验对标（层流振荡通道的闭式解即精确解，无更优基准）。
- 建议总账条目: `CASE-FVM-STOKES-SECOND`（算例池），验收=本文件 19 判据。
## 4d. 追加欠账（append-only，2026-11-13）
### D16 2-D 浅水（SWE-2D）能力与 Thacker (1981) 精确解验证面 —— 已交付（m16944），长时程仍 honest FAIL
- 缺口来源：`shallow_water/` 只有 1-D（`swe1d`/`bed`）；2-D 算子与 2-D 精确解验证面全缺，总账无条目。
- 交付：`src/tensorfvm/shallow_water/{swe2d.py,thacker.py}`（Audusse 静水重构 + 三分量 HLL + SSP-RK3，
  闭式解 + `check_thacker_solution` PDE 残差认证）、`verification/thacker_basin.py`（C1–C9 预声明判据）、
  `scripts/run_thacker_basin.py`、`tests/test_shallow_water_2d.py`（11 测，1 xfail）、
  `docs/case-fvm-thacker-basin/{criteria.md,REPORT.md,artifacts/*}`。
- 关键数字（`--periods 0.5`、n=160、窗口 0 ≤ t ≤ 0.5T）：静湖 `max|u| = 5.7e-15`；闭式解 PDE 残差
  `7.75e-8`；水量 vs 解析 `πa²H/2` 差 `0.0038 %`、漂移 `1.1e-14`；中心水深史 L2 **0.726 %**；
  岸线平均 **2.354 %**；对称性缺陷 **0.000 %**；负对照（中心源项）静湖爆炸到 `5.0e10 m/s`。
  C7 **honest FAIL**：中心误差阶 0.98，岸线误差饱和（阶 0.43）。
- 两个真实缺陷（实测、已修）：①HLL 必须吃星态深度，否则静湖假质量通量 `0.15 m²/s`；
  ②跨湿/干界面必须关掉 `eta` 重构，否则静湖 `max|dh| = 1.3`（水漏干滩）。
- 下一回合首项：**长时程动量注入**——满 2 周期时 n=40 第 200 步出现 `18.2 m/s` 非物理速度
  （同刻物理上界 ≈9），CFL 步长塌缩；需正性保持（positivity-preserving）通量限制或干湿翻转时的
  动量重投影，再谈 2 周期精度；次项：岸线一阶饱和（干湿重构加二阶/界面追踪），转置对照口径未定位。

### D9 三网格 log--log 斜率 / 单元中心 FV 观测阶验证面 —— 已交付（m17696, 2026-11-13）
- 缺口来源: 仓内观测阶全部由"两点、比率 2 硬编码"的散装估计器给出
  （`case_cavity_re100.observed_order`、`benchmark_poiseuille_channel.observed_order`、
  `verification.core.observed_orders` …），两点零自由度、无不确定度、无法识别非单调/前渐近；
  Richardson/GCI 家族（`verification.richardson`/`multi_grid`）只覆盖"功能量差"路线；
  **log--log 斜率路线**与**单元中心 FV 的单元平均口径**从未被验证过。
- 交付: `src/tensorfvm/verification/loglog_order.py`（`loglog_order` 三网格 LSQ + 标准误 +
  命名守卫；`cell_error_norms` 三种范数口径；闭式制造解/单元平均；2 阶与两种 4 阶面导数模板；
  `truncation_ladder` / `solution_error_ladder` / `stretched_weights` / `run` / `figure` / `main`）
  + `tests/test_loglog_order.py`（29 测全绿）+ `python -m tensorfvm.verification.loglog_order`
  + `docs/verification-three-grid-loglog-cellcentre/{REPORT.md,summary.json,make_report.py,fig_loglog_order.png}`。
- 12 条预声明判据全 PASS。关键数字: 2 阶解误差阶 2.004（标准误 1.3e-3）；单元平均专用 4 阶
  模板 3.908 / 截断 3.882；**路径 B**（4 阶解对格心点值参考）读成 1.884；**路径 C**（点值 4 阶
  模板喂单元平均）读成 2.042 —— 即 4 阶能力被两种不同机制各自掩蔽到 2 阶；2 阶截断常数比
  单元平均/点值 = 1.9970（解析 2）；1 阶负对照 0.921；与 `richardson_gci` 互校差 3.0%；
  对单元数拟合得 p/d = 1.000（2 维）；拉伸网格加权 vs 非加权 |Δp| = 7.96e-4（量级比 1.219）。
- 反例/踩坑: ①点值 4 阶中点模板 `(1,-27,27,-1)/24` 对**点值**是 4 阶，对**单元平均**只有 2 阶
  （带 h²/12·u''' 不消），单元平均专用模板是 `(1,-15,15,-1)/12`（Laplacian 即经典
  `(-1,16,-30,16,-1)/(12h²)`）——模板必须与数据种类配对；②纯 Laplacian 有常数零空间，
  误差只定义到差一个常数，本单用 κ>0 消除而不是平移场去凑；③三网格 log--log 的
  `asymptotic` 标志必须报（相邻两点斜率离散度 > 25% 时是"限定阶"而非 FAIL/合格）。
- 建议总账条目: `XVAL-FVM-THREE-GRID-LOGLOG-SLOPE-CELLCENTRE-FV-OR`（本单锚点，本次交付即其证据）。
- 下一步线索: 非线性（限制器退化到 1 阶）、边界掉阶（近壁一阶 BC 压整体阶到 1）、
  非结构网格重建误差三项各自需要独立阶梯，均未覆盖。

## 4e. 追加欠账（append-only，2026-11-13，PR#545 重开）

### D17 等效沙粒粗糙度 k_s（粗糙壁 RANS）验证面 —— 已交付（m16801 → m18360 重开落库），高 k_s^+ 段 honest FAIL
- 缺口来源: `rans_validation` 只支持水力光滑壁（`wall_values` 给光滑 ω_w），任何 k_s/粗糙度
  能力在仓内**完全缺失**（`grep -i rough` 只命中 `biomimetic_riblet` 的关系式库，无壁面条件）；
  而 KMM180 只锚定光滑壁 → 工程阻力（涂装/污损/ plating roughness）无入口。总账外新缺口。
- 交付: `src/tensorfvm/rans_validation/roughness.py`（Nikuradse 差分锚 + Wilcox(1988) `S_R` +
  壁值封顶 + 穿越点二分 + 粗糙对数律拟合）+ `channel.solve_channel(..., k_s=)` 最小挂钩
  （默认 None 逐位同旧解）+ `src/tensorfvm/verification/rough_wall_channel.py`（预声明判据 R1–R10、
  机器判级、`__main__` 入口写 `summary.json`）+ `tests/test_rough_wall_roughness.py`
  （合成/契约/登记/**工件漂移门**四层，26 测）+ `docs/verification-rough-wall-rans/{REPORT.md,
  summary.json,fig_roughness_function.png,make_report.py}`。
- 关键数字（Re_τ=20000, n=512, y₁⁺=0.40, κ 用各模型自含值）: k-ω 的 ΔU_b⁺ 对 Nikuradse
  差分锚误差 k_s⁺=30/70/100 → **−0.130/−0.166/−0.236 w.u.（≤3.2% of u⁺）**，300/1000 →
  −0.609/−0.978（FAIL）；粗糙对数层斜率 −2.608 vs −2.306（**13.1% FAIL**）；截距差
  B_rough−B_smooth = 3.064 vs 3.30（k_s⁺=100）；平移 y 无关性极差 0.018 w.u.；Re 独立性
  |ΔU_b⁺(Re2000)−ΔU_b⁺(Re20000)| = 0.086 w.u.；光滑极限逐位为 0.0。
- 反例/踩坑: ①封顶必须 `min(ω_smooth, ω_rough)`——`ω_rough ∝ S_R` 随 k_s⁺ **下降**，取 `max`
  会让粗糙度永远被光滑值盖住；`y₁⁺=10` 实测 `ω_smooth=3.2e3 < ω_rough=4.0e3`，旧写法会
  **静默返回光滑解**，现改为抛 `ValueError`（失败即抛）；②网格是标定的一部分：`y₁⁺=3` 同一
  k_s⁺=100 只给 −5.55（欠 1.73 w.u.），低 Re 处理本来就要求 `y₁⁺ ≲ 1`；③SST 上迁移 Wilcox `S_R`
  **不合规**（k_s⁺≥70 起饱和，斜率 −0.757 vs −2.450，69% 偏差）→ 只登记不给"SST+k_s 已验证"；
  k_s⁺≥300 的高估经带内漂移检验（0.03 w.u.）确认不是外区污染，是 `S_R` 高段偏陡的模型行为。
  ④**工件不得只靠外部脚本复现**：旧报告写 `/tmp/run_full.py`，仓库内无法重跑 → 本单补
  `-m tensorfvm.verification.rough_wall_channel` 入口 + 重评漂移门；在 main `879c7da0` 重跑全量
  后 `summary.json`/`fig_roughness_function.png` **逐字节不变**（重放确定性，1 min 50 s）。
- 明示未做（WITHHELD）: k-ε 壁面函数粗糙化（`u_wall` 与 ε_w 需一致化，现抛 NotImplementedError）、
  过渡区（k_s⁺<30）Hama 型过渡函数、方向性粗糙（riblet/砂纹）、粗糙层内分辨。
- 给下次: 建议总账新条目 `FEAT-FVM-ROUGH-WALL-KS`（验收=本账 R1–R10，工程可用域 k_s⁺≤100）；
  后续首项=k-ε 粗糙壁面函数（工业船体常用模型），次项=Eça/Hoekstra `k_s` 型 ω 壁条件替代 `S_R`
  以修高 k_s⁺ 段。

## 4f. 追加欠账（append-only，2026-11-14）

### D10 续（m20758）—— 四层阶估计器已落地自测；n=256 实算 not measured（交付通道丢工件）
- 起因: `CLOSURE-32-64-128.md` §4.5 自记欠账——三层插值掉 `f(h)=L+C h^p`，`order_stderr`
  不可识别（dof=0）。交付: `scripts/summarize_kovasznay_32_64_128.py` 新增 `order_uncertainty()`
  （整梯阶 + stderr + dof + 线性化/t95 区间 + GCI + 滑窗斜率，散度 > `PAIR_ORDER_BAND=0.20` 即拒绝）；
  `tests/test_kovasznay_256_ladder.py`（4 条合成阶梯单测常跑 + n=256 工件门，缺失即 skip）；
  `docs/CASE-FVM-KOVASZNAY-ORDER/ORDER-32-64-128-256-ATTEMPT.md`、`make_order_uncertainty_report.py`。
- 关键结论: 三层向后兼容**逐叶值零差异**（判据 1.9623 一位未动）；纯 `h^2` 合成阶梯阶误差 <1e-3；
  纯 `h^1` 阶梯过不了 1.8 门（旗标非硬编码）；三层时 `order_stderr` 如实 None。
- **未完成（honest）**: `cj2912`（n=256, 8 核, n330）rc=0 但取回目录为空、无作业日志；同次 fetch 里
  既有 n32/n64/n128 也缺失 → "节点早退"与"交付过滤器丢工件"不可判别。四层阶/不确定度 not measured。
- 给下次: ①同一作业内把 `cases.json` + ≤4096 单元降采样 CSV 写进 `docs/**.md` 纯文本；
  ②若仍 rc=0 且空，先跑 n=256 `--max-outer-iterations 1` 内存探针；③n=512 待 n=256 落地后再议。

### D12 CASE-FVM-CHANNEL-KMM180 的 y+<30 精确渐近律验证（m20892, 2026-11-13，selfstart）
- 缺口: 锚点只判 `y+>30`，但它的误差预算把 SST/k-ω 的 **109–113 %** 壁单位亏空归因于
  `y+<30`（band `buffer_origin`），而该区**从未对任何精确律验证过**——归因是"闭合模型
  的账"，却可能其实是动量积分的数值缺陷。
- 交付: `src/tensorfvm/verification/kmm180_sublayer.py`（解析锚 `u+ = y+ - y+²/(2Re_tau)`，
  预声明判据 C1–C5 + C4b）+ `tests/test_channel_kmm180_sublayer.py`（16 测全绿，含
  篡改负对照）+ `docs/case-fvm-channel-kmm180-sublayer/{REPORT.md,summary.json,profiles.csv,3 png}`。
- 关键数字（n=64/128/256/512, Re_tau=180）: C1 `y+≤1.5, n≥128` 最差偏差 **0.0030 %**
  （sst）/ 0.0036 %（komega）；C2 观测阶（全网格拟合）**1.57/1.50**，干净网格对 64→256
  局部阶 **2.38/2.29**（二阶）；C3 `y+ = 2` 残差随加密不再下降（256→512 变化 9.96 %/
  2.75 %），即 **O(y+⁴) 湍流项的物理地板** 0.0179 %/0.0222 %，对应 `∫₀²ν_t⁺ds ≈ 3.6/4.4e-4`；
  C5 壁函数装配（Spalding 根 vs 质量通量积分导数）**5.5e-8 %**。
- **honest FAIL（不掩盖）**: C4（发布 u+ 必须等于自身 shear 积分，全网格 ≤0.05 %）在
  n=64 为 **0.086 %** → FAIL；n≥128 为 0.031/0.015/0.008 %（C4b 通过，C4 的 FAIL 留在表里）。
- 踩坑（两个静默单位陷阱，都伪装成求解器缺陷）: `_sublayer_mass_flux` 的自变量是
  **外单位** `y/δ` 且积分 `∫u+ d(y/δ)`——传壁单位 30 会积到通道外（报 3.6e4 % 缺口），
  对 `y+` 而非 `y/δ` 求导会留下 `Re_tau` 因子（报 103 % 缺口）；两者都不抛错。
- 未做: k-ε 壁函数分支没有解析底层（C1–C4 不适用）；`y+≳5` 之上同一比较会变成判模型
  而非判求解器，未越界使用。

### D14 CASE-FVM-PERIODIC-HILLS 第十二回合（m20920, 2026-11-14）—— SST 输运方程三处离散缺陷已修，锚点仍 OPEN
- 起因: round 11 以 `x_r/H = 2.05` vs 5.34 收束，并把结论记成「SST 在 Re=2800 过扩散」。
  rounds 10-11 只对**代数闭包**做过验证，从未检查 `k`/`omega` 的**输运离散**。
- 交付: `src/tensorfvm/periodic_hill_sst.py` 三处修复
  （① 扩散系数多除一次 `sigma`，omega 的有效涡扩散因此是应有值的 2 倍；
   ② 文档写明但从未实施的 `omega` 壁面 Dirichlet `60 nu/(beta1 d^2)`，实际传的是 0；
   ③ 内部面系数南北不对称，SST 算子改为双面平均，体积加权对称性入测；
   附: 改用闭包的 F1 混合 `gamma`/`sigma_k`/`sigma_omega`，omega 产生项改写成
   `gamma*(P_k/nu_t)` 消除 `nu_t->0` 的 1/1e-12 假值）
  + `tests/test_periodic_hill_sst_transport_coefficients.py`（5 测全绿，
  独立参考算子/闭式/精确再现，非自证）+ `docs/case-periodic-hills/ROUND12.md`
  + `scripts/round12_analysis.py`。
- 撤回: `ROUND11.md` §6 的「SST 过扩散」**模型类结论**不成立——它测的是一个解了
  错误方程的求解器；改为 WITHHELD 直至修复后重跑出数。
- 待办（本单未完成的部分，交下一单）: cj3059 的 96x48 时间平均重跑（提交时仍在跑）
  → 读 `docs/case-periodic-hills/round11_sst_r12-96x48.json`，用
  `scripts/round12_analysis.py` 打分，判 `x_r/H` 是否进 5.34±10%。
  若仍不达标，按「先分离两个修复的贡献」做消融（需要给 `SSTKOmegaConfig` 加
  `omega_wall` 开关）。
- 已知未修（明示）: 父求解器 `periodic_hill.py` 的动量扩散算子仍是「南面用 P 元胞系数」，
  同样的不对称在这里改造会牵动 rounds 4-11 全部已发布的动量数字，故只记录不改。
- 环境坑（重要）: site-packages 的 editable `.pth` 指向 `/srv/tensor/agent/work/m20057/src`，
  裸 `python -m pytest` 会导到别人的工作树（本单实测新测试曾因此报 AttributeError）。
  必须 `PYTHONPATH=<本工作树>/src`，集群用 `stage_dir` 而不是 `ref=main`。

### D15 CASE-FVM-PERIODIC-HILLS 第十三回合（m20987, 2026-11-15）—— 动量/SA 输运的 eta 面配对缺陷已修，round-12 待办已闭环，锚点仍 OPEN
- 起因: round 12 收尾时 (a) 修好闭包的 96x48 重跑 cj3059 仍在跑（判据 WITHHELD），
  (b) 明示未修「父求解器动量扩散算子南面用 P 元胞系数」。
- 交付 A（闭环 round-12 待办）: cj3059 succeeded，取回 `round11_sst_r12-96x48.{json,npz}`
  并用 `scripts/round12_analysis.py` 打分：`x_r/H = 3.144`（vs 5.41 → 41.9%）、
  `y/H=0.5` 0/7 站点进 10%、`max nu_t/nu = 297`、`x_r/H` 平稳性 spread_rel 2.5e-5。
  即：round 12 的两处输运修复把 2.046→3.144（+54%），缺陷真实且实质；但均值仍过扩散，
  锚点验收仍 FAIL，SST 模型类结论继续 WITHHELD（求解器仍有实测离散缺陷）。
- 交付 B（修缺陷）: 新增 `periodic_hill.eta_face_viscosity`（一个 eta 面一个 μ，两侧共享；
  壁面层用相邻元胞值），`_assemble_momentum` 与 `SpalartAllmaras._diffusion_operator`
  改用它（SST 输运已在 round 12 修过）。
  实测（32x16，`nu_t/nu` 至 300）: 旧算子与面循环定义矩阵有 960 项不同（最大 0.944）；
  离散耗散恒等式 `Σ_P Vol_P φ_P(Lφ)_P = -Σ_f c_f μ_f (φ_N-φ_P)²` 被破坏
  **3.65%（ν_t/ν=100 时 3.53%，=0 时 0）**，符号随 `dμ/dη` 翻转
  （增厚处欠耗散 +3.5%，变薄处过耗散 -3.2%）。不是「产生能量」，是耗散量错。
- 测试: `tests/test_periodic_hill_eta_face_viscosity.py` 8 测全绿（独立面循环 1e-14、
  耗散恒等式 1e-10、旧规则负对照用解析 `max c_n*0.5|Δμ|/Vol` 1e-12、均匀 μ 逐位兼容、
  SA 算子同规则）；回归 `test_periodic_hill{,_round2,_round3}` 22 passed、
  round-12 SST 系数测试 5 passed（须 `PYTHONPATH=<worktree>/src`）。
- 待办（交下一单）→ **已闭环（m24004, 2026-11-16）**: cj3108 `succeeded`（n315），
  工件 `round11_sst_r13-96x48.{json,npz}` 自作业 tarball 取回，`scripts/round13_analysis.py`
  打分（未重跑、未重数字化）：`x_r/H = 3.767`（vs 5.41 → 30.4%；vs 5.34 → 29.5%）、
  `y/H=0.5` 0/7 站点进 10%、`max nu_t/nu = 202.5`、平稳性 spread_rel 2.1e-6、gate FAIL。
  判读：动量算子修复把 3.144→3.767（+19.8%），该缺陷解释 41.9% 缺口中的 11.5 点；
  剩余 30% 为 SST@96x48 的模型类/分辨率问题（算子已是单面通量散度，1e-14/1e-10 验证），
  SST 模型类结论继续 WITHHELD，下一测量=固定算子下的网格阶梯。
- **数字时效性**: 本修复改变 rounds 7–12 全部含 ν_t 的动量结果（rounds 4–6 常黏度门是
  逐位不变的）。旧数字保留原样记录，不回溯改写。
- 产物: `docs/case-periodic-hills/{ROUND13.md,round13_operator_defect.json,fig_round13_eta_face_operator.png,
  round11_sst_r13-96x48.json,round11_sst_r13-96x48.npz,round13_comparison.json,
  fig_round13_profile_y05.png,fig_round13_series.png}`
  + `scripts/round13_analysis.py`。

### D16 CASE-FVM-CHANNEL-KMM180 第三条路线（三维 LES）两条 xfail 的根因是 6 处装配缺陷（m21024, 2026-11-15，selfstart）
- 缺口: `channel_les3d` 留下两条 xfail（层流抛物不复现、60 步冒烟不绿），并写明「Re_tau=180 无任何 u+ 读数」。
- 交付: `src/tensorfvm/channel_les3d.py` 六处修复（① `dy_center` 行数/中心距；② `v` 梯度改在插值后的
  单元中心求；③ v 向 y 扩散的通量间距 `dy` 与控制体 `dy_cc` 分离（旧代码用 `dy_cc` 当相邻 v 面间距且
  行数崩）；④ v 的横向扩散系数插值到 y 面；⑤ `smagorinsky_viscosity` 的 `delta`/`y_plus` 列广播；
  ⑥ `wall_normal` 通量分母 `dy`→`dy_cc`（层流稳态从 3.9 % 停滞转为收敛））+ 新增
  `reichardt_profile()` 初值律、`diffusion_number()`、`convective_cfl()`、`initialise(profile=...)`
  + `tests/test_channel_les3d.py` 15 测全绿（原 2 条 xfail 已删除，断言改为可复现的真值）
  + `docs/case-fvm-channel-kmm180-les3d/{REPORT-M21024.md,m21024/*}`。
- 关键数字: 层流稳态 0.97 %（`dt=1.5e-3`，旧测试 `dt=5e-3` 越界 4.7/2 倍 → 发散 1e129，被误记为"未收敛"）；
  壁剪 3.8e-4；内部截断阶 1.99，墙单元残差 O(1) 不随 h 降（半单元导纳 u(yc0)/yc0 的固有缺陷）；
  Reichardt 初值 vs LM2015：对数窗 2.18 %。
- **honest BLOCKED（锚点仍 not_run）**: 产线 64x64x32（y1+=0.51）在 Re_tau=180 仍发散；功率迭代
  `Cs=0` 0.9996/步 vs `Cs=0.1` **1617/步** ⇒ 失稳来自 `nu_t` 反馈（首超限在壁面行，nut≈3nu），
  `dt` 降到 2.5e-4（超保守 4 倍）仍 ~250 步发散 ⇒ 不是步长问题。
- 给下次: ① y 向扩散隐式化（Thomas/(x,z)，系数随 nu_eff 逐步入），同时换二阶单侧壁梯度消 O(h) 地板；
  ② 壁面 nu_t 硬约束并复测放大率 <1；③ 通过后 T=80 自旋 + T=40 统计窗，块平均 ≥20 样本，
  用 `kmm1987.log_region_deviation(high_fraction=0.2)` 出锚点读数。

## 4g. 追加欠账（append-only，2026-11-15）
### D18 XVAL-FVM-M18737-ROUND2 补轮：加密**被定级解**（certified-<1% 的定量裁决）—— m21306 selfstart
- 缺口来源：`docs/XVAL-FVM-M18737-ROUND2/CERTIFIED-REF-CLOSURE.md` 的唯一绑定门是**被定级解**自身
  L1 = 1.7685 % > 1 %；上一轮只加密了参考解，被定级解停在 nx=3200，"加密被定级解是否/要多贵才能 <1%" 无账。
- 交付：`src/tensorfvm/verification/wc_refine.py`（守恒限制度量 `restrict`/`cell_average_l1`、旧插值口径
  `pointwise_l1`、误差模型 `observed_order`/`forecast_error`/`nx_for_error`、残差归属 `residual_density`/
  `region_shares`、总审 `audit`）+ `tests/test_wc_refine.py`（20 测全绿，6 s）+ `docs/XVAL-FVM-M18737-ROUND2/`
  `{PREDECLARATION-refine.md,REFINEMENT-CLOSURE.md,refinement_figures.py,refinement_report.py,summary.json,figures/}`。
- 冻结预测（测量前写死，`PREDECLARATION-refine.md`）：用 e(1600)=3.2513 %、e(3200)=1.7685 %
  定阶 p=0.8785，外推 e(6400)=0.962 %、e(12800)=0.523 %；判据 F1(区间)/F2(阶 0.70–1.20)/F3(裁决)。
- 已实测结论：①**度量假设 M1 成立**——守恒限制 vs 旧插值口径差 0.0069 pp（≪0.05），
  1.776 % 不是插值假象，裁决不随口径翻转；②参考自认证 L1 0.5437 %/峰值 0.0230 %（门 1 仍过）；
  ③L1 残差**前沿局域化**：x∈[0.40,0.55] 平滑平台只占 0.05 %，
  x∈[0.72,1.00] 占 62.31 % ⇒ 加密/高阶是对症手段；
  ④峰值度量非单调（1600 档 0.1247 % < 3200 档 0.7604 %），判据以 L1 为主。
- 未完成（明示）：nx=6400/12800 被定级解**作业 cj3325 仍在跑**（28 min 未出工件；nx=1600 单档已 >120 s），
  故 F1/F2/F3 的数值裁决 **not measured**，冻结预测仍在账上待验；另：CFL/时间推进敏感性、高阶格式
  （WENO5 本体）对照未做。重跑：`wc_ladder.py --plan 6400:hllc,12800:hllc --outdir runs/wc-refine`，
  工件回填 `docs/woodward-colella-double-blast/ladder/data/ladder_profile_nx<6400|12800>_hllc.npz` 后
  重跑 `refinement_figures.py` + `refinement_report.py` 即自动改判，无需改代码。

## 4h. 追加欠账（append-only，2026-11-15）
### D19 CASE-FVM-PERIODIC-HILLS 第十七回合（m21402, selfstart）—— 再附长度的**度量算子**自身有缺陷：UNDEFINED 回到实测，锚点仍 OPEN(honest FAIL)
- 起因: `ROUND16.md` 把 SA-96x48 与 SST-patankar 两档的 `x_r/H` 记为 UNDEFINED（是 WITHHELD），
  并建议做网格收敛阶梯去「看 UNDEFINED 会不会随加密消失」。
- 只读审计发现（无新算力）: `r16_patankar_96x48.json` 里 `time_averaged.crossings = [7.6816]`
  而同一 artefact 的 `x_r_over_H = null` —— **交点被记录过，数值被丢掉**。
  根因是 round-9 `zero_crossings_periodic` 的两个性质：
  ① 只识别 `tau<0 -> tau>=0`（升交点=再附），**`+ -> -`（分离）从不识别**，
     所以返回的 `separation` 只是「1.9 以下的升交点」，一条气泡/周期时必然 NaN；
  ② 硬窗 `[1.9, 7.1]` 之外的实测交点直接返回 NaN，与「没有交点」不可区分。
  该算子被 `periodic_hill_round11.py` 的时均与分块打分使用 ⇒ 缺陷进入已发布数字。
- 交付: 新算子 `src/tensorfvm/verification/wall_shear_bubbles.py`
  （`signed_crossings` 有向交点/亚网格插值、`bubble_intervals` 气泡配对+接缝展开+逆向面积、
  `separation_reattachment` 主气泡按 `∫|tau|` 取最大、窗口只作**报告过滤**
  `status="out_of_window"` 且保留数值、退化输入 `all_reversed`/`no_recirculation`/`not_finite`、
  保留 legacy 的 `crossings`/`sign_changes` 键义以免下游静默重解释、`tanh_shear_profile` 认证剖面）
  + 生产路径 `periodic_hill_round11.py` 改用它（`_REATTACHMENT_WINDOW=(1.9,7.1)` 保留为过滤器）
  + `tests/test_wall_shear_bubbles.py` 21 测全绿（解析零点 1.4e-7=0.001·dx²、逆向面积 vs 独立求积 6.9e-11、
  窗内与 legacy 1e-12 一致、**负对照**复现 legacy 对同一剖面返回 NaN、跨缝 8.2→9.3 长度 1.1、双气泡主选、
  退化输入、3 档存盘 npz 集成）
  + `scripts/round17_{rescore,report,measure}.py` + `docs/case-periodic-hills/{ROUND17.md,round17_rescore.json,fig_round17_shear_profiles.png,fig_round17_xr_comparison.png}`。
- 关键数字（参考值取权威口径 `periodic_hill_reference.X_R_OVER_H=5.41`，非 5.34）:
  重打分三档存盘剖面 —— SA 7.651（legacy NaN，+41.4%）、SST-r11 2.046（+62.2%）、SST-r12 3.144（+41.9%）；
  从交点列表恢复 —— patankar 7.682（legacy NaN，+42.0%）、newton2 3.767（+30.4%）。
  `x_s/H` 在**全部三档存盘剖面** legacy 都是 NaN（结构性不可测），新算子给 0.397/0.641/0.480；
  Re=2800 的 `x_s/H` 基准在权威口径里是 **WITHHELD**，故只报观测不判误差。
- 两条结论性改动:
  ① **round-16 的「网格阶梯看 UNDEFINED 是否消失」前提被否证** —— null 是报告窗造的，
     7.65/7.68 一直在账上；阶梯仍是真欠账，但问题改述为「+42% 是否随 h 下降」。
  ② **模型类结论翻转**：rounds 11–15 的「SST 在 Re=2800 过扩散」是在错误 sink 线性化
     （`S_P=2βω`，无补偿项 ⇒ ν_t 高 2 倍）下测的；模型一致 Patankar 系数下 ν_t 由 202→49、
     气泡由 3.77→7.68，误差从 −30% 翻成 +42% ⇒ 正确表述是「2-D URANS+SST 在分离剪切层
     **混合不足**（气泡过长）」，round-11 的模型类结论撤回。
- artefact 契约缺陷（已记）: `r16_*.json` 只存升交点列表、未存 `mean_tau_b`，patankar 档 `x_s/H`
  事后不可恢复；`round17_measure.py` 用 `save_fields=True` 保留剖面（作业 cj3409）。
- 验收判定: **FAIL（honest）**，锚点保持 OPEN。本轮增量是把 WITHHELD 变成可测的 REJECTED，
  不是把 FAIL 翻成 PASS；两个 <3% 项都是算子自身的离散恒等式，不代表物理精度。
- 未完成（明示，交下一单）: ① `cj3409`（`scripts/round17_measure.py`，96x48/20000 步、`save_fields=True`
  保留时均剪切剖面，用于补 patankar 档的 `x_s/H` 与表 3）本单收尾时仍在**排队**（递交 05:11，队列在飞 53 个作业），
  未出数，故报告表 3 与 platform report `id 19343` 均未引用其数值；出数后只需重跑
  `scripts/round17_report.py` 即自动把表 3 换成实测（脚本按 `round17_measure_96x48.json` 存在与否分支）。
  ② `cj3413`（全 periodic_hill 相关测试，本地被 120 s 单命令闸门拦下、必须上队列）同样在排队；
  本单本地已实测的关键测试为 `tests/test_wall_shear_bubbles.py` 21 passed、
  连同 round9 的 20 + round11 的 13 共 **54 passed**（`PYTHONPATH=<本工作树>/src`）。
- 给下次: ① 96x48→144x72→192x96 只跑 patankar 测「+42% 是否随 h 降」；② 单独追 `x_s/H` 偏后
  （0.40–0.64，但无权威基准，先补基准再判）；③ 给 round-11 写出端加「必存时均剪切」断言。

## 4i. 追加欠账（append-only，2026-11-16）
### D20 CASE-FVM-CHANNEL-KMM180 三维 LES 路线：隐式壁法向扩散 + SSP-RK3 已交付，两个失稳诊断被否证（m21536，selfstart）
- 缺口: D16 记录产线 64x64x32 在 Re_tau=180 数步内发散，且归因为「nu_t 反馈」与「显式 y 扩散刚性」。
- 交付: `src/tensorfvm/channel_les3d.py`（① `implicit_wall_normal` 三对角隐式壁法向扩散，矩阵由 `diffusion()`
  同一表达式导出、算符恒等钉到 1e-11；② `time_scheme='rk3'` SSP-RK3；③ 壁面距离 `y+=min(y,2-y)Re_tau`
  ——旧定义把上壁读成 y+=360，van Driest 阻尼与 Reichardt 初值在上半槽道此前是错的；④ `wall_shear('second_order')`
  + 采样/统计双读数）+ `tests/test_channel_les3d_implicit.py` 9 测 + `docs/case-fvm-channel-kmm180-les3d-stability/`。
- **否证**: Cs=0 同样发散（131 步）与 Cs=0.1（142 步）同量级 ⇒「nu_t 反馈」不是主因；隐式化扩散后仍发散 ⇒
  「仅扩散刚性」不成立；euler(187) vs rk3(142) 同量级 ⇒ 也不是积分器阶数。发散步数随 dt 近似反比（142→74），
  即每步放大 `|G|=1+O(dt)`，降 dt 只推迟不消除。
- 未做/给下次: 逐行/逐模态增长率扫描定位被放大模态（下一步具体动作）；产线统计 T_spin/T_stat 与 u+ 读数仍 not_run。

## 4j. 追加欠账（append-only，2026-11-16）
### D21 XVAL-FVM-M18737-ROUND2 第五轮：换度量（输运范数 W1）把 certified-<1% 变成可证明——但「多得一阶」的假设被否证（m21711，selfstart）
- 缺口来源：`docs/XVAL-FVM-M18737-ROUND2/R4-REFERENCE-UNCERTAINTY-CLOSURE.md` 裁决 `upper_bound_only`
  （逐点 L1 0.8931 % + 参考侧 `U_ref` 0.7181 % = 1.6112 % > 1 %），并给出 P5：逐点 `<1 %` 上界需
  参考分辨率 `nx_ref ≈ 1.33e5`。逐点 L1 的残差由激波抹平带主导（观测阶 ~0.83），加密参考极贵。
- 本单做法（**无新算力**，全部由已存 `ladder/data/ladder_profile_*.npz` 重新积分）：
  把度量换成**输运范数**（negative-norm s=−1 ≡ 归一化累积质量分布的 W1 距离）
  `d_W1 = 100·∫|D|/(mass·L)`，`D(x)=∫_{0.4}^{x}(ρ_c−ρ_f)`；参考侧误差棒同口径重算；
  诚实上界 = 点估计 + `eps_ref`。判据 P1–P9 在计算前冻结（`PREDECLARATION-w1-norm.md`）。
- 交付：`docs/XVAL-FVM-M18737-ROUND2/wc_w1_norm.py`（度量 + 冻结判据 + 残差定位 + **注入误差探针**）
  + `wc_w1_norm_figures.py`/`figures/fig_w1_norm.png` + `render_w1_doc.py`（零手抄渲染）
  + `W1-NORM-CERTIFICATION.md` + `tests/test_compressible_wc_w1_norm.py`（9 测全绿，5 s）。
- 结果数字（程序注入）：`d_W1`@6400 = 0.009781 % of span（≈0.63 个粗格），参考自认证 6400 vs 12800
  `d_W1` = 0.005753 %，`eps_ref(W1)` = 0.005854 % ⇒ 诚实上界 **0.015635 % ≪ 1 %**（P4 PASS，余量 64×）；
  同批剖面的逐点 L1 仍是 0.9004 %（P5 PASS：弱范数不是免费午餐）。
  度量有效性：刚体平移/定点搬质量/1% 全局幅值错的实测-闭式比 **1.0000 / 1.0000 / 1.0001**；
  负控制（数字化已发表曲线）落在本单上界 4.53× 之上。
- **诚实 FAIL**：P1「积分后多得一阶」被证伪（p_W1 = 0.987 vs p_L1 = 0.833，收益仅 0.15；
  残差是对撞区 x≈0.765 的**净质量通量缺陷**，按 O(h) 收缩，W1 的优势是常数 ~100× 而非阶）；
  P7 方向鲁棒性 1/6 对越界（nx=1600 达 0.46，nx≥3200 ≤ 0.12）。冻结判据 7/9 PASS。
- 未收口（明示）：逐点 L1 的 `<1 %` 上界**没有**被本单改变（仍需 `nx_ref≈1.33e5`）；
  `d_W1` 对「抹平宽度」天然不敏感，不能替代逐点判据。下一手可检验预测：若残差确为净通量缺陷，
  `max|D|` 应精确按 h¹ 收缩（实测 0.004509→0.002352→0.001078，比值 1.917/2.181）；
  WENO5@25600 回填后若比值显著偏离 2，需重新归因。

### D22 XVAL-FVM-M18737-ROUND2 第六轮：参考族「阶」门的仪器验证——WENO7 探针被证伪（m21843，selfstart）
- 缺口来源：`CERTIFIED-REF-CLOSURE.md`（1.776 % honest FAIL）与 `REF-ATTRIBUTION.md` 的同族地板 0.548 %
  只能说明「参考族还在动」，不能说明「参考族极限离被定级解多远」；R3/R4/R5 三处把下一手写成
  「换重建阶（WENO7）比继续均匀加密更值得」。而 `xcode/wc_weno7.py` 此前**从未在 blast 数据上跑过**。
- 本单做法：先冻结判据（`PREDECLARATION-r6.md` + `predeclaration_r6.json`，Q1–Q6，两条带披露的 AMENDMENT：
  Q3 的 `U_ref=e/(2^p−1)` 定义修正、生产者加发散守门）→ 写生产者 `xcode/wc_weno7_ladder.py`
  （发散档 status=diverged 且不落剖面）→ 3200/6400 档递交调度（`cj3733`）→ 本机只做单核冒烟扫描
  （200/400/600/800/1600）→ `ref_order_r6.py` 程序化对账/作图/渲染。
- **否定结论（可复用）**：该 WENO7 探针**不能**当高阶参考——① nx=200 与 nx=800 直接发散（NaN），
  而 400/600/1600 能跑（非单调稳健性）；② 能跑档位峰值**越过**收敛峰：@600 6.6431（+2.8 %）、
  @1600 6.7114（+3.9 %）vs WENO5@12800 收敛 6.4595，而同档 WENO5 只有 −0.19 %。⇒「提高重建阶即得
  更可信参考」在本实现上不成立；修好边界/正性处理前 WENO7 只能诊断、不得定级。
- 流程缺陷修复：旧流程会把含 NaN 的剖面当参考写出（下游 L1 变 NaN 而不报错）→ 生产者现在
  记 `status="diverged"` 并删除剖面，加载器只能报 not_measured。
- 新测数（程序注入）：同网格跨阶差 L1(WENO7@1600 vs WENO5@1600) = 见 `summary_r6.json`（<1 %）。
- **未收口（明示）**：Q1–Q6 全部 `not_measured`——WENO7@3200/6400 的调度作业 `cj3733` 在交付时仍排队
  （集群 95 个在飞作业），命中 0/6，**不得**读作 PASS。certified 裁决不推进（仍 False，1.776 %）。
- 给下次：① 先修 `wc_weno7.py` 的边界/正性（反射壁 ghost + 界面声速对负压态的兜底），并用熵波阶测试
  之外的**含激波收敛性**测试锁住（当前 6 阶熵波测试完全看不出 NaN/越限）；② 有了可信 WENO7 后再回填
  Q1–Q6；③ 若 WENO7 修不好，改用异族代码（CLAWPACK/pyclaw 已有 `xcode` 对照）做参考——那是**真正**
  的独立代码而非同族换阶。

## 4k. 追加欠账（append-only，2026-11-15）
### D23 CASE-FVM-PERIODIC-HILLS 第二十一回合（m21912, selfstart）—— 闭合无关姊妹算例落地：`x_r/H` 的两个口径差 50 %，`Re=1400` 发稳定位为步长相关数值失稳，锚点仍 OPEN(honest FAIL)
- 缺口来源：round 20（m21737）递交的 `cj3663/cj3664/cj3665`（laminar Re=700/1400/2800）当时全部**排队未出数**，
  故该轮裁决 WITHHELD、`attribution()` 无输入。本单三作业全部 succeeded 并已 fetch 入库。
- 交付：`docs/case-periodic-hills/round20_laminar_Re{700,1400,2800}_96x48.{json,npz}`（0.6 MB，未重跑）
  + `src/tensorfvm/verification/periodic_hill_field_metrics.py`（`backflow_extent` 独立速度场度量 / `definition_spread` 两口径对账）
  + `scripts/round21_analysis.py`（表+两图，零手抄）+ `scripts/round21_re1400_probe.py`（发散定位探针）
  + `tests/test_periodic_hill_field_metrics.py`（17 passed, 1 skipped，5 s）
  + `docs/case-periodic-hills/{ROUND21.md, round21_table.json, round21_re1400_probe.json, round21_re1400_probe_dt001.json, fig_round21_ladder.png, fig_round21_re1400.png}`；
  **试图**重跑 `round20_analysis.py` 以把 round-20 图上三颗「queued 空菱形」换成实测点：该脚本在 artefact 存在后
  **崩溃**（`AttributeError: 'str' object has no attribute 'get'`，`sister_table`→`score_case` 路径把
  `payload["closure"]`（round-20 里是字符串）当字典取 `.get("name")`）⇒ `round20_table.json`/`fig_round20_ladder.png`
  **未刷新**，round-20 的图仍是 queued 版本。此缺陷为「有 artefact 才触发」，此前一直被 skip 掩盖，记为新欠账。
- 关键数字（程序注入）：闭合无关 `x_r/H`（时均剪切零点）= **8.078 @Re700（+54.2 % vs 5.24）**、**8.264 @Re2800（+52.8 % vs 5.41）**、
  Re=1400 **DIVERGED @step 3013**；逐样本集合平均 3.993 / 3.502（两档均 `definition_sensitive`，差 50.6 % / 57.6 %）。
  第二条独立通路（`mean_u<0` 逆流端点）8.016 / 8.203，与剪切值差 −0.78 % / −0.74 % ⇒ 度量链路自洽（**非**精度证据）。
  闭合的贡献：8.264 → SST 7.600/7.682，只占「需要修正量」的 **20–23 %**，其余 ~77–80 % 不能由这类闭合解释。
- 两条结论性改动：
  ① **`Re=1400` 发散是数值的**：本机探针 `dt=0.02` 在 step 2969 前 `max|u|≤2.85`、`u_b=1.000`、CFL 0.64（<1），
     随后 <100 步 `max|u|` 冲到 12.63 并溢出；`dt=0.01` 同长度窗口（t=60）不发散（`max|u|=1.786`）⇒ 步长相关、且与 Re **非单调**（700/2800 收敛）。
  ② **发布口径的定常性证据缺失**：artefact 的 `stationarity.ok` 由逐样本旧算子算得，而该序列与末帧快照逆流端点
     （5.766 vs 集合平均 3.993）自相矛盾 ⇒ 它**不能**为发布值 8.078 提供定常性证明（新欠账）。
- 归因：预声明规则 `attribution()` 输出 **`inconclusive`**（52.8 % 未超过姊妹最差 54.2 %；规则要求「超过」才判 framework_attributable）。
  数字指向框架/二维/口径而非闭合，但**不篡改规则、不升级结论**；下一轮先预声明修正规则（合法判据：闭合无关误差随 Re 平坦，
  实测 54.2 %@700 vs 52.8 %@2800）。
- 验收判定：**FAIL（honest）**，锚点保持 OPEN。三项可测量：+54.2 %、+52.8 %、发散；无一项进 10 %，更无 <3 %。
- 给下次：① 写出端按块存时均剪切剖面，补发布值 8.078 的块间稳定性（同时可解两口径差 50 % 的瞬时零点结构）；
  ② `dt=0.01` 复跑 Re=1400 补阶梯的洞（约 2× 成本）；③ `gate` 段仍按旧值 5.34 打分，与 `lower_re` 的 5.41/5.24 并存，引用时勿混。

## 4l. 追加欠账（append-only, m22013 selfstart）
### D24 CASE-FVM-CAVITY-RE1000 归因审计——「1.5 % 属于 Ghia」被证伪为不可裁决，锚点门仍绿
- 缺口来源：`COMPREHENSIVE_REPORT.md` §3.2 / `SPECTRAL.md` §1 把「三码互吻合」当作「残差属于参考表」的证据；
  该推理在 GCI 口径下不成立（残差落在求解器自身不确定度带内）。
- 交付：`src/tensorfvm/cavity_attribution.py`（`gci` / `extremum_floor` / `contraction_ratios` /
  `is_asymptotic` / `attribution_verdict`；裁决可为 `within_own_uncertainty` / `reference_attributable` /
  `unresolved`）+ `tests/test_cavity_attribution.py`（17 passed, 13.7 s）+
  `docs/CASE-FVM-CAVITY-RE1000/ATTRIBUTION_AUDIT.md` + `audit_attribution.py`
  + `runs/attribution_audit.json` + `figures/fig_attribution_audit.png`；§3.2 就地加更正块（原句保留，不覆盖）。
- 关键数（程序注入）：u_min 残差 0.630 %、v_max 0.786 %，表值在 GCI 带内位置 0.43 / 0.60；
  跨码**原始**最细网格差 0.855 %（引用值 0.259 % 为两外推之差）；谱阶梯自身增量比 0.042/0.023
  vs 外推假定 0.312；表极值地板 u_min ≤0.724 %、v_max ≤0.083 %；Re=1000 谱解 `lid_velocity_max`=1.074、
  `wall_slip_defect`=3.1e-4（未收敛），Re=100 谱阶梯已收敛（N48−N64 = 1.3e-8）。
- **未收口（明示）**：第四级网格 n=192（cj3849 递交于本单，交付时未取回 ⇒ 记 not_run）；
  异族第三方代码对照未做；Ghia 原文网格收敛数据无法取得 ⇒ **不得**把「不可裁决」读成「Ghia 有错」或「求解器有错」。
- 给下次：① 取回 cj3849 后把 `informative` 变为 True，重算 GCI 带（预期 p 仍 ≈2，带收窄）；
  ② 若要真正裁决，需异族代码（CLAWPACK/FEniCS）而不是同族换阶；③ `extremum_floor` 可直接复用于
  Re=400 锚点与任何「表 + 极值判据」的算例（地板 > 容差即该判据不可判级）。

## 4f. 追加欠账（append-only，D16 长时程窗口推进）

### D16-b Thacker 盆 2 周期窗口：度量协议 + 动量注入归因（honest FAIL，未修）
- 缺口来源：D16 首项"长时程动量注入"只有一句现象（18.2 m/s、步长塌缩），无度量协议：
  不知差多少、非物理速度在哪里、压制速度能否救精度。
- 交付：`src/tensorfvm/verification/thacker_longrun.py`（2 周期窗口度量 + 归因 + 候选疗法排除，
  含精确速度包络 `|beta|s`、峰值单元状态、精确岸线外水量、薄水膜裁剪变体）、
  `tests/test_thacker_longrun.py`（4 测全绿）、`docs/case-fvm-thacker-2period/{REPORT.md,metrics.json,fig_thacker_2period.png}`。
- 关键数字（n=40，2 周期）：中心水深 L2 **34.22 %**（容差 3 %）、`max|u|` **71.16 m/s** 于
  $h/H=1.2\times10^{-3}$、$r=1.062$ m 的薄膜（精确岸线 0.760 m，该点精确速度 1.6e-15）；
  精确岸线外水量 **11.77 %** 而体积漂移仅 1.7e-14。
- 已排除的疗法：薄水膜动量清零（thin=1e-2/3e-2）把 `max|u|` 压到 24.2/13.7，但中心 L2 仅降到
  33.5/30.6 %、岸线外水量 11.4/10.3 % ⇒ **化妆非治疗**，不写入求解器默认路径。
- 未做（not_run）：n=80/160 的 2 周期（本机门禁拦，需 `run_compute`）；逐项动量账分解未落代码。
- 建议总账条目：`XVAL-FVM-THACKER-2PERIOD-MOMENTUM`，验收＝中心水深 L2 ≤3 % ∧ 全程
  `max|u|` ≤1.2×精确包络 ∧ 岸线外水量 ≤0.5 % ∧ 体积漂移 ≤1e-9。

## 4m. 追加欠账（append-only, m22380 selfstart）
### D25 CASE-FVM-PERIODIC-HILLS 第二十四回合（m22380）—— 壁面法向导数阶数审计：`x_r/H` 的后处理贡献 ≤3.64 %，honest FAIL 不是后处理造成的；量级结论必须升到 ≥2 阶
- 缺口来源：round 17 认证了零点**算子**、round 21 用 `mean_u` 独立对账、round 22 钉死气泡对象，
  但三者共用的输入 `tau_b = du_t/dn`（求解器两点单侧差分）**从未被审计**。round 23 的「给下次」第 3 条即此。
- 交付：`src/tensorfvm/verification/periodic_hill_wall_shear_order.py`（1/2/3 阶单侧壁面导数，同一几何/切向投影/零点算子；
  `wall_gradient`/`wall_shear_profile`/`crossing_scale_invariance`/`order_sensitivity`）
  + `tests/test_periodic_hill_wall_shear_order.py`（10 passed；连同 field_metrics/bubble_object/reference/round23 共 50 passed, 1 skipped，5.5 s）
  + `scripts/round24_wall_shear_order.py`（读盘审计 8 个可用 artefact + 3 图，零重跑）
  + `docs/case-periodic-hills/{ROUND24.md, round24_wall_shear_order.json, fig_round24_tau_order_profiles.png,
  fig_round24_xr_postproc_band.png, fig_round24_bias_field.png}`（正文数字全部由脚本注入）。
- 关键数字（程序注入）：解析标定（平壁 Poiseuille，精确 `du/dn`=42.504）1 阶相对误差 **恰等于 `s_0/h`**
  （ny=12/24/48 → 4.167/2.083/1.042 %，比值恒 1.000000），2/3 阶到机器精度 ⇒ 阶数是量出来的。
  锚点 `round20_laminar_Re2800_96x48`（冠后气泡 vs Breuer 5.41）：`x_r/H` = **1.254 / 1.230 / 1.224**（1/2/3 阶），
  后处理带 **2.37 %**，误差由 −76.82 % → **−77.37 %**（−0.55 pp）。全库 8 例后处理带 **0.52–3.64 %**，
  收敛比 `(o1−o2)/(o2−o3)` = **1.76–6.91**（全部 >1 ⇒ 三条阶梯同向收敛）。1 阶偏差符号**不一致**（3/8 例 1 阶 > 3 阶）
  ⇒ 不许乘性「修正」，只能报带。量级偏差局部最大 **33.5 %** of `max|tau_3|`（mean 2.6–5.9 %）。
- 两条结论性改动：① 锚点 honest FAIL **不能**归因于后处理（后处理带比 FAIL 余量小两个量级）；
  ② 任何摩擦速度/阻力/壁面切应力**量值**结论必须用 ≥2 阶壁面导数（零点位置类结论可沿用求解器两点差分）。
- 明确未做：只审计**时均**场的阶数敏感性（逐样本零点需写出端按块存剪切剖面，仍欠账）；
  eta 线斜交误差（`|beta/alpha|` 近壁 O(0.6)）未做显式修正 ⇒ 报出的是后处理不确定度**下界**；
  收敛比不支持 Richardson 外推，未外推真值。
- 给下次：① 用 `mesh.beta_n` 做斜交一阶修正，把「下界」变真值估计；
  ② SST 的 `u_tau`/壁面律诊断（round 15）改用 `wall_shear(order=2)`；③ 求解器默认路径本轮**未改**（保持 round 17–23 可比性）。

## 4n. 追加欠账（append-only, m22529 selfstart）
### D26 XVAL-FVM-M18737-ROUND2 第 12 轮：nx=6400 的「certified <1 %」被自己的误差棒否证（点估计 0.893 %，诚实上界 1.601–1.611 %）
- 缺口来源：`CERTIFIED-NX6400-CLOSURE.md`（11-16 合并）用**两级** WENO5 自认证（6400 vs 12800 <1 %）
  作 gate 1，随即把 `L1(MUSCL@6400, WENO5@12800)=0.893 %` 判为 `certified=True`。两级门只有距离、
  没有观测阶 ⇒ 没有误差棒，`0.893 %` 是 `|num−ref|` 的点估计，而 `R4/R11` 早已把参考侧误差定价
  （B0 上界 1.6112 %、B3 3.0474 %）。文档链因此自相矛盾：一处说「certified 达成」，一处说「<1 % 不可证明」。
- 交付：`docs/woodward-colella-double-blast/xval/wc_certified_ref.py` 新增 `reference_error_bar()`
  （三级 WENO5 阶梯 → `p_ref`、`eps_ref`），并把 `certified` 拆为
  `certified_point_estimate` / `certified_provable` / `certified`（后者要求 `L1+eps_ref<1 %`）；
  `tests/test_compressible_wc_certified_ref.py` 9 项全绿（新增「certified 必须过误差棒」与
  「docs 线性插值 vs 包守恒限制两实现口径对账」）；`docs/XVAL-FVM-M18737-ROUND2/
  {certified_nx6400_errata.py, CERTIFIED-NX6400-ERRATA.md, certified_nx6400_errata.json,
  figures/fig_errata_{ladder,budget,selfconv}.png}`；`CERTIFIED-NX6400-CLOSURE.md` 末尾追加勘误块。
- 关键数字（程序注入，零手抄）：`p_ref`=0.8327；`eps_ref`=0.7076 %（docs）/ 0.7181 %（包，差 1.5 %）；
  诚实上界 **1.601–1.611 %**；族内 Richardson（无参考）**1.523 %**；峰值差 0.0411 %；窗质量 0.008 %；
  族内观测阶 0.8053→0.7657（一阶渐近，与 R6 一致）。按实测阶外推，<1 % 需 `nx≈12800`（预测 0.895 %，**未测量**）。
- 归因：不是求解器退步，而是**两级自认证不足以支撑 <1 %**：比较两个各自带 ~1.5 % 误差的数值解，
  差值 0.893 % 小于任一方的自身误差（误差部分抵消）。族内路线（完全不碰参考）独立给 1.523 %，
  两路同量级 ⇒ 结论不是度量伪影。
- 验收判定：锚点总账 `<15 %` **PASS**（最保守上界 1.611 %），`<3 %` PASS；`<1 %` **honest FAIL**
  （点估计进线、可证明上界越线）。锚点保持 OPEN。
- 给下次：① 真正收口 `<1 %` 只能加细被定级解（`MUSCL/HLLC@12800`，单核 ~180 min 量级，
  需分段重启或超单作业墙钟），或按 R4 的 `nx_ref≈1.33e5` 路线；② 度量实现口径差（1.5 %）应统一为
  单一实现，否则所有上界都要带这个系统项；③ 任何「certified」判据今后必须显式包含 `eps_ref` 项
  （已由本单的布尔语义强制）。

## 4n. 追加欠账（append-only, m22630 selfstart）
### D27 CASE-FVM-CAVITY-RE1000 第四级网格（cj3849 取回）—— 表值吻合度不是精度的证明；「加密会更准」被证伪，锚点门仍绿
- 缺口来源：`ATTRIBUTION_AUDIT.md` 与 `SPECTRAL_HIORDER.md` 都留了一条 not_run：第四级 n=192 已递交
  （`cj3849`）但交稿时未取回；三级阶梯（32/64/128，加密比 2）只给出**一个**收缩比，因此
  「表值落在 GCI 带内」无法检验自己的前提。
- 交付：取回 `cj3849`（node n315, rc=0）的 `field_n192.npz` + JSON 落库
  `docs/CASE-FVM-CAVITY-RE1000/runs/refine/`；新模块 `src/tensorfvm/cavity_refinement.py`
  （`field_functionals`/`profile_errors`/`pair_order`/`textbook_order`/`ladder_verdict`）；
  驱动 `docs/CASE-FVM-CAVITY-RE1000/grid_ladder_4level.py`（零重跑，四级原始 MAC 面量全部独立重算，
  与存档 JSON 对账 1e-12）；`tests/test_cavity_grid_ladder.py`（16 passed）；
  `docs/CASE-FVM-CAVITY-RE1000/GRID-LADDER-4LEVEL.md` + `runs/grid_ladder_4level.json`
  + `figures/fig_ladder4_two_references.png`（表内数字全部脚本注入）。
- 关键数（程序注入）：n=192 判级网格上登记门 C1..C7 全 PASS（余量由 2.7× 降到
  2.60×）；对 Ghia 的误差 u_min 0.647% →
  1.155%（**变差 1.79×**），
  v_max 0.779% → 1.276%（1.64×）；
  对独立谱极限（SPECTRAL_HIORDER 四阶外推 −0.388603/0.377101）则单调下降
  0.832% → 0.332%；
  128→192 成对阶 p = 2.270（u_min）/
  2.052（v_max），与均匀段教科书阶
  2.080 一致（未见角点奇性降阶）；
  离散带 GCI：u_min 1.324% → 0.416%，
  v_max 1.266% → 0.472%。
- **结论**：n=128 的「与 Ghia 最吻合」是误差抵消，不是更准；残差（≈1.2–1.3%）> 自身离散带
  （≈0.42%），且加密使表值吻合度变差、使与独立极限的吻合度变好
  ⇒ 判决 `reference_offset_dominates`（方向性证据从「两码一致」升级为「四网格 + 双参考」）。
  仍不主张 Ghia 表值错在哪一条（仓内无第三方核对副本）。
- 明示未做（not_run）：第五级 n≥320（成本 O(n³) 量级）；第三方发布高精度参考（Botella & Peyret 等）
  无核对副本；本单未重跑求解器（只重算度量）。
- 建议总账条目：`XVAL-FVM-CAVITY-THIRD-REFERENCE`，验收＝引入一份**异族第三方**存档参考（含其自身
  不确定度）后，`u_min/v_max/vortex` 三方互证误差 ≤0.5%，并写明该参考的网格/阶数与其 GCI。

## 4o. 追加欠账（append-only, m22685 selfstart）
### D28 CASE-FVM-PERIODIC-HILLS 第二十六回合——归档里 6/14 是**全零 dump**；层流解只有一个再循环单元，`x_r/H` 口径带宽 129.6 %（判 `definition_sensitive`），锚点仍 OPEN(honest FAIL)
- 缺口来源：round 23「给下次」第 1 条（SST 加密网格看收敛）与 round 25 遗留（闭式无关 `Re=2800` 网格阶梯）都指向
  「加密档的场」，但**没有任何门检查过归档里到底有没有场**；另外 round 17/21/22/24 四轮审计的都是壁剪这条**一维**口径，
  从没人问过「被配对成气泡的那段壁剪区间，是否对应场里一个闭合再循环单元」。
- 交付：`src/tensorfvm/verification/periodic_hill_archive_audit.py`（`artifact_health` / `recirculation_cells` /
  `wall_spans` / `bubble_object_audit` / `definition_routes` / `definition_band` / `ladder_summary`）
  + `tests/test_periodic_hill_archive_audit.py`（19 passed，4.5 s；含合成夹具：两单元 / 抬离单元 / 跨接缝单元 / 非有限 / 缺键）
  + `scripts/round26_archive_audit.py`（零重跑，读 14 个 `.npz`）+ `docs/case-periodic-hills/
  {round26_archive_audit.json, ROUND26.md（脚本生成）, fig_round26_archive_health.png,
  fig_round26_bubble_objects.png, fig_round26_definition_band.png, fig_round26_cells_laminar.png}`。
- 关键数字（程序注入）：14 归档 **8 可用 / 6 全零**（`round11_sst_r18_120x60/144x72`、`round20_laminar_Re1400_96x48`、
  `round20_laminar_Re2800_144x72/192x96/240x120`，速度/时均/壁剪三者全零）；零剪切的 `mean_tau_b` 让认证算子返回
  `status="no_recirculation"`、`x_r/H=nan` —— 与「流动完全附着」不可区分，故必须先过完整性门。
  **两条网格阶梯判决均为 `insufficient_rungs`**（SST r18 可用 2/4，唯一可用对 72→96 变化 1.07 %；层流可用 1/4）。
  层流锚点：连通再循环单元 **1 个**，壁剪口径给出 **2 个气泡**（first 1.254 / largest 8.264），场侧单元末贴壁点 8.258
  ⇒ `object_incomplete`（同一单元在 1.25 处只是**抬离壁面**，抬离带内 `max|τ|` 仅剖面的 2.15 %），
  口径带宽 **129.6 %**（上限 8.264 / 下限 1.254 / 参考 5.41）⇒ `definition_sensitive`；阈值扫描
  `u<0 … u<-0.05` 全部 1 单元（结论不依赖阈值）。SST r18(7.682)/SA round9(7.651) 为单单元、`object_supported`、
  带宽 0.1 %。**无任何口径 <3 % 或 <10 %**，锚点继续 honest FAIL。
- 归因结论：① round 23/25 的遗留**不是算力欠账而是数据欠账**（细档没有场，重跑并过门后才可能可答）；
  ② 层流「−76.8 %」是 `wall_shear:first` 口径的产物，同解在 largest/场侧口径下是 **+52.8 %**，引用必须带口径；
  ③ 排序口径不变：三种口径下 SST 都最接近参考 ⇒ 「最优闭包=SST」不是口径伪影。
- 明确未做（not_run）：未重跑任何求解器（本项目定位为只读审计）；`144x72/192x96` 的稳定 `dt` 重跑未做；
  完整性门只看 `mean_u`（归档无 `samples`/收敛记录，「发散/被杀/写盘失败」无法区分）。
- 建议总账条目：`CASE-FVM-PERIODIC-HILLS-LADDER-DATA`，验收＝「按 round-23 稳定 dt（144×72→0.00665、
  192×96→0.0025）重跑并把**非零**场入档后，`artifact_health` 全绿且可用档 ≥3，`ladder_summary` 由
  `insufficient_rungs` 变为 `measured`，并给出相邻档 `x_r/H` 相对变化与对 5.41 的误差」；所属域=TensorFVM/验证。

## 4p. 追加欠账（append-only, m22755 selfstart / verifier 席）
### D29 CASE-FVM-DAMBREAK-STOKER 的 oracle 独立性与仪器标定（锚点条目已 PASS，本单补欠账而非重做）
- 缺口来源：池条目 `docs/oss_solutions/CASE-FVM-DAMBREAK-STOKER.json` 的每个数字都对着
  `shallow_water.exact` 打分，而该模块**既是判据来源又是参考实现**，其自检
  `check_stoker_solution` 用的还是自己那套公式 ⇒ 闭式的系统性错误会同时通过两边；
  且 `front_fan` 类估计器继承精确稀疏波形状，误差量级是**仪器精度**不是求解器精度。
- 交付：`src/tensorfvm/verification/dambreak_independent_oracle.py`（独立 oracle：
  自相似解 + Rankine-Hugoniot，中间态用二分法解单标量方程，不 import `exact`；
  独立仪器：整场 L1、半跃变水跃定位器、测量式平台水平、水位交点波前、prominence 极值计数；
  仪器标定：注入位移/平台缩放/wiggle）+ `scripts/run_dambreak_independent_verifier.py`
  + `tests/test_dambreak_independent_oracle.py`（13 测全绿，14.3 s）
  + `docs/case-fvm-dambreak-stoker-verifier/{REPORT.md,metrics.json,fig_verifier_*.png}`。
- 关键数字（程序注入）：independent vs 仓内 oracle `t*=2` 全剖面最大差 `2.2e-16`、中间态
  `h2/u2/s/step_ratio` 相对差 `0`；跳跃残差 质量 `0`、动量 `2.24e-16`、稀疏波不变量 `0`。
  `n=1600` 独立口径：干床深度 L1 `0.077 %`、水位波前（h=0.05h0）`0.081 %`、薄膜湿边
  `2.89 %`；湿床 L1 `0.043 %`、水跃位置 `0.021 %`、水位阶跃 `1.3e-6`。
  仪器标定：注入 ±0.05/±0.10 m 水跃位移复原相对误差 ≤`1.24e-5`；平台缩放 ±1 % 复原
  ≤`4.2e-8`（=2 % 时 `3.3e-7`，含重新定位）；单调门盲区 k=4 `1.41e-4 h0`、k=8
  `3.70e-4 h0`；负对照深度 L1 `12.0 %` 被正确判错。
- 判定：锚点验收（`x_f`、水位阶跃、单调无伪振荡，<5 %）在独立 oracle + 独立仪器下
  **复现 PASS**，四项判据最大误差 `0.081 %`（比 5 % 低两个量级）。
- 三条非阻塞发现（引用必须带口径）：F1 `front_fan=1.7e-4` 是形状先验估计器口径，同解在
  水位交接口径下为 `8.1e-4`、薄膜湿边口径下 `2.89 %`；F2 单调门是**阈值命题**（prominence
  >1e-4 h0 时未见极值，>1e-6 h0 时 n=1600 湿床仍有 2 个、平台 overshoot `+5.5e-6 h0`）；
  F3 阶跃误差与窗口定义有关（本单 `1.3e-6` vs 池内 `5.76e-7`），1e-6 级差别不是物理精度比较。
- 明示未做（not_run）：异族代码（CLAWPACK/FEniCS）交叉；`t*>2` 长时程；二维/斜置溃坝。
- 建议总账动作：`CASE-FVM-DAMBREAK-STOKER` 证据链已加一层独立复核 → 可升 `verified`；
  若要求更强的独立性，另立 `XVAL-FVM-DAMBREAK-THIRD-CODE`（验收＝异族代码同解 L1 ≤5 %）。

## 4q. 追加欠账（append-only，m22872 重开 PR#806 第 3 轮）

### D10 续（m22872）—— 三网格标量已搬上文本通道；n=256 仍 honest WITHHELD
- 起因: `ORDER-32-64-128-256-ATTEMPT.md` §5.1 处方（同作业内把 cases.json + 降采样 CSV 写进
  `docs/**.md`），以及 `cj2912`/`cj3258` 两次 `rc=0` 但取回目录为空的事实——证据只存 npz 是单点故障。
- 交付: `scripts/write_kovasznay_evidence_packs.py`（从已提交 npz 渲染文本 pack，不重解，`--check`
  复核 sha256/行）；`runs-32-64-128-zero_gradient/n{32,64,128}/n*-evidence.md`（三份，约 1.9 MB）；
  `tests/test_kovasznay_ladder_text_channel.py`（16 测：行逐字一致 + sha256 绑定 + 文本重算 L2 +
  纯文本斜率 ≥1.8 且与 summary 差 ≤1 % + 重渲染逐列不变）；
  `docs/CASE-FVM-KOVASZNAY-ORDER/TEXT-CHANNEL-32-64-128.md`。
- **顺带修的真缺陷**: `evidence_pack.lattice_indices` 把 `i%stride==0` 掩码套在单格展向轴上，
  展向索引是浮点噪声（n=32/64/128 → 15/31/63），`stride>1` 时掩码整体为空 →
  `ValueError: downsample selected no cells`，**最细层级根本写不出文本 pack**（n=128 需 stride 2、
  n=256 需 stride 4）。修法：单格轴不抽稀 + 用轴内排序位置取代绝对索引；回归测已加。
- 关键数字: 文本通道复现 vel L2 n=32/64/128 偏差 0/0/0.594 %，去均值压力 n=128 偏差 2.741 %；
  纯文本重建的三网格斜率 **1.9581 vs 已提交 1.9623（0.22 %）**，仍过 1.8 门。
- **未完成（honest）**: n=256 仍未测（本轮 `cj4441` 递交后长时间排队，未在时限内落盘）→
  四层阶与 `order_stderr` 继续 `None`，不报任何四层数字；建议下次以"文本优先"方式重投，
  并先跑 `--max-outer-iterations 1` 内存探针。

## 4r. 追加欠账（append-only，m22998 selfstart / verifier 席）

### D30 CASE-FVM-DAMBREAK-STOKER 的数值配置包络——CFL/阶数稳健，**深度下限是隐藏的自变量**
- 缺口来源：池条目只声明 `n = 1600`，把 `cfl = 0.4 / order = 2 / dry_depth = 1e-12` 当默认值；
  `shallow_water.robustness` 扫的是物理不变性（平移/边界/域长/相似缩比），**`dry_depth` 这个绝对长度
  从未被变化过**，而求解器在五处（`set_state`/`primitive`/`_face_states`/`_hll_flux.dry_pair`/`_sanitise`）
  用它定义干床前沿的被测速度场与通量——即 D29 速度腿 open FAIL 的成因之一。
- 交付：`src/tensorfvm/verification/dambreak_config_envelope.py`（三轴扫描 + 硬失效门 + 包络证书 +
  `floor_diagnostics`/`analytic_tip_length`/`hidden_mass_instrument_error`）
  + `scripts/run_dambreak_config_envelope.py`（artifacts 全部程序注入，~60 s 单核）
  + `tests/test_dambreak_config_envelope.py`（14 测全绿，9.7 s）
  + `docs/case-fvm-dambreak-stoker-config-envelope/{REPORT.md, SUMMARY.md, metrics.json,
  proposed_pool_amendment.json, fig_*.png×4}`；
  `dam_break.run_case` 仅新增 `dry_depth` 透传（默认 = 求解器默认，池数字逐位不变，测试钉住）。
- 关键数字：CFL ∈ {0.1…0.95} **8/8 全绿**（干床 `front_fan` 3.958e-4→1.714e-4，0.1↔0.4 只差 1.2 % ⇒ 池点读数不受 Δt 污染；
  0.6→0.95 改善 2.3×，故度量在池带外 CFL 依赖）；阶数 1 与 2 **都全绿**（一阶 `depth_l1` 差 5.0×，5 % 门是二阶误差的 13 倍
  ⇒ 池门分辨不出格式阶）；深度下限轴 **首破 `front_wetted<5%` 于 `dry_depth = 1e-6`**（n=1600 上 6.64 %，n=800 上 8.28 %），
  认证带 `[1e-15, 1e-9]` 对加密稳健；出厂 `1e-12` 只剩 1.7× 余量；`1e-2` 时 headline `front_fan` 吃掉 92 % 预算（4.58e-2）、
  `5e-2` 时读数无定义（硬失效）。湿床对该轴免疫（h ≥ 0.5，`floored_cells = 0`）。
  下限**改动力学而非只改读数**：湿边后退 0.194 m（floor 1e-9）远大于几何薄层 `3t√(g·floor) = 1.9e-4 m`（~1000×），
  扣除本征后退 0.031 m 后，下限驱动部分对几何薄层长度的 log–log 斜率 ≈ 0.35（经验律，一条网格族）。
  仪器有效域：藏水量读数在薄层 ≳ 若干格时才可信（`floor ≥ 1e-3` → 误差 3.6 %；`1e-2` → 2.5e-5；`1e-6`（薄层 < 1 格）→ 100 %）。
  负对照（下限设到解析水深之上）被正确判错：干床 NaN 读数 + 3 伪极值 + L1 16.1 %，湿床水跃 13.0 % + 3 伪极值。
- 明示未做（not_run）：二维/斜置溃坝的同一轴；`t* > 2` 与下限的耦合；异族代码交叉；
  湿床 `h1 < floor` 退化区（本单湿床下限全部 ≪ h1，仅负对照越界）。
- 建议总账动作：把 `proposed_pool_amendment.json` 的 `configuration` + `certified_band` 合入池条目；
  `CASE-FVM-DAMBREAK-STOKER` 可升 `verified`（理由已从「数字都通过」变为「数字在其声明的配置带内通过」）。

## 4s. 追加欠账（append-only，m23069 selfstart / verifier 席）

### D31 CASE-FVM-DAMBREAK-STOKER 的池条目溯源审计 —— 数字全部可复现，但证据链**未冻结**，且 main 的 staleness 普查已过期
- 结论（好消息，先说）：**池条目的全部数字能用当前源码逐位复现**——最差判据相对差 `0.0`，
  步数 1513 / 918 完全一致；条目与生成器**逐字节相同**（14411 B），无手改、无单位滑移；
  六门判据值全部可由工件独立重派生（drift 0）；`accepted_grid_min = 400` 由 primary 判据唯一决定
  （只用 primary 口径也是 400），故 `front_wetted<5%` 这个被条目自标为 diagnostic 的量**没有**
  污染接受网格。7 项溯源检查 5 PASS / 2 FAIL，审计器 5/5 注入探针全部报警（非空转）。
- **FAIL-1（可修，机制级）**：`docs/shallow-water-dam-break/` 没有 `manifest.json`，
  该算例**完全不在** staleness 冻结网里（53 例中无 dam-break）。含义：「生产源码改过、
  工件没重跑」这类漂移不会被 `tests/test_source_staleness.py` 抓到，本次只是运气好（P1 证明当前无漂移）。
  建议：为 `docs/shallow-water-dam-break/` 补 `manifest.json`（impl_files 至少含
  `shallow_water/dam_break.py`、`swe1d.py`、`exact.py`、`scripts/run_shallow_water_dam_break.py`），
  量化验收：`audit()` 的 `cases` +1、该例 `stale=False`、篡改 `dam_break.py` 后该例 `stale=True`。
- **FAIL-2（可修，下一轮自动闭环）**：池条目仍缺 `dry_depth`。存档 `metrics.json` 的
  `discretisation` 是 `run_case` 加入该键**之前**生成的（源码现已记录）。已把
  `x_min/x_max/x_dam/dry_depth` 全部接进 `case_pool_dambreak.build_entry` 并重生成条目，
  其中前三项已落地，`dry_depth` 待工件重生成后自动补齐；
  量化验收：条目每个 case 的 `REQUIRED_INPUTS` 12 项全非空。
- **顺带修 main 红测**：`test_source_staleness.py::test_live_tree_ledger_reproduces_and_ratchet_is_respected`
  开工时失败（冻结普查 fresh=1276/changed=101 vs 实测 1253/124，23 条绑定源码被近期合并改动），
  但 `unregistered_stale == []`（无算例新变味，棘轮未破）→ 按 `--update-ledger` 复位，19/19 通过。
- 证据：`docs/CASE-FVM-DAMBREAK-STOKER-PROVENANCE/{REPORT.md,summary.json,SUMMARY.md,fig_provenance.png}`、
  `scripts/audit_dambreak_pool_provenance.py`、`tests/test_dambreak_pool_provenance.py`。
- 教训：**锚点"物理论证厚"不等于"证据链可引用"**。下次自找活优先问「这条证据链谁能独立重跑」，
  而不是继续在同一锚点上加第 16 个物理判据。

## 4t. 追加欠账（append-only，m23193 selfstart / verifier 席）

### D32 CASE-FVM-DAMBREAK-STOKER 进入冻结网（D31 的 FAIL-1/FAIL-2 闭环），并查出目录 §19.1 与自身工件不一致
- 缺口来源：D31 自己的建议——`docs/shallow-water-dam-break/` 无 `manifest.json`，溃坝算例是唯一不在
  `tests/test_source_staleness.py` 冻结网里的验证算例；池条目缺 `dry_depth`，12 个输入只记 11 个。
- 交付：`scripts/register_dam_break_staleness.py`（`--write/--check/--json`，digest 现算）+
  `docs/shallow-water-dam-break/manifest.json` 与 `source-snapshot/`（6 个生产文件逐字节冻结，156 KB）+
  重跑驱动后重生成的 `metrics.json`（新增 `dry_depth`）/`case_pool_entry.json`/目录 §19.1 +
  `case_pool_dambreak._failing_cell_sentence`（结论句改为派生，删掉手抄的「n=200/400 (6.61 %)」）+
  `tests/test_dambreak_staleness_registration.py`（13 测全绿，6.9 s：注册/棘轮/派生/快照完整性/漂移可见
  + 三个负对照：篡改快照副本、缺文件、活树模式改源码，各自必须 stale）+
  `docs/CASE-FVM-DAMBREAK-STOKER-STALENESS-NET/{REPORT.md,summary.json,fig_staleness_closure.png,make_report.py}`。
- 关键数字：audit cases 53→54、entries 1378→1384、fresh 1253→1259、**stale 18→18（零增账）**、
  该算例 `frozen=True / stale=False / 6-6 fresh`；`audit_dambreak_pool_provenance.py` 的 `P5_case_is_frozen`
  由 false 翻 true（7 项检查 6 PASS + 唯一的 `P5_entry_is_generated` 因本单重生成条目而为 true）；
  池条目 `dry_depth` 12/12 输入；重跑后判据**逐位不变**（除新增键与 wall_time/meta）。
- 附带发现（同族欠账）：目录 §19.1 在 main 上就印着与 `convergence.json` 不符的序列误差（n=100/200/400/800
  印 4.695/6.613/6.613/1.398 %，工件是 0.958/0.639/0.680/0.532 %），据此写下的「PASS at n≥800、
  n=200/400 miss series」也是错的（`accepted_grid_min = 400`）。已重新生成并让结论句从 `honest_fail` 派生。
- 明示未做（not_run）：二维/斜置溃坝的同一冻结；`hires.json`/`bore_subcell.json` 未重跑；
- 建议：把 `scripts/audit_dambreak_pool_provenance.py` P1（当前源码 vs 存档，~5 s）接进 CI——冻结只保
  「可重导出」，不保「源码漂移」，两条腿才闭环；另建议另立 `XVAL-FVM-DAMBREAK-THIRD-CODE`
  （验收＝异族代码同解 L1 ≤5 %），这是 D29 以来唯一始终 not_run 的独立性轴。

## 4l. 追加欠账（append-only，2026-11-16）
### D24 XVAL-FVM-M18737-ROUND2 第七轮：E2（max|D| 的 h^1 收缩）在 25600 档被**否证**——参考族自差比 h^1 收缩更慢且趋势变差；并发现 R5 预注册的类别误差（m23342，selfstart）
- 缺口来源：`W1-NORM-CERTIFICATION.md`「给下一手」第 3 条（= D21 的可检验预测，判据 E2 在
  `PREDECLARATION-r7.md`/`PREDECLARATION-r10.md` 冻结：`p_D = log2(max|D|(6400→12800)/max|D|(12800→25600)) ∈ [1.85,2.15]`）。
  R7/R10 两轮 E2 均 `not_measured`（当时 25600 档不在库）；25600 档此后以 CFL 0.8 变体回填（R15 已用并过 E1r6400 门），
  但 E2 本身从未被测量。
- 本单做法（**零新算力**，全部由已存 npz 重新积分；判据先冻结于 `PREDECLARATION-r16.md`）：
  在**同仪器**（CFL 0.8）参考阶梯 6400/12800/25600 上测 E2；度量一律调用 `wc_w1_norm` 同一实现。
- 交付：`docs/XVAL-FVM-M18737-ROUND2/{PREDECLARATION-r16.md, wc_w1_r16_flux_defect.py, r16_flux_defect.json,
  render_r16_doc.py, R16-FLUX-DEFECT-CLOSURE.md, figures/fig_r16_flux_defect.png}`
  + `tests/test_wc_r16_flux_defect.py`（11 测全绿，4.4 s）。
- **否定结论（可复用）**：E2 **FALSIFIED**——`max|D|(6400→12800)=0.000724`、`max|D|(12800→25600)=0.000474`
  ⇒ `p_D = 0.6116`（带 [1.85,2.15] 之外）；参考族比值趋势 **1.852→1.565→1.528**（逐档变小，越细收缩越慢）。
  即参考族（WENO5）自差**不是**纯 O(h) 量，与 R6（p≈0.8 非渐近）、R14 B3 honest FAIL 一致：
  参考族误差棒不能用 h^1 外推低估。定位不变：x_peak=0.7648（对撞区，E2c PASS）、
  rel_net_mass=2.46e-6（非守恒缺陷，E2d PASS）。
- **预注册类别误差（本单发现）**：R5 引用的 0.004509→0.002352→0.001078（比值 1.917/2.181）是**被定级族**
  （hllc vs weno5@12800）的 max|D|；而 E2 判据定义在**参考族**自差上，其先验 `≈0.000539(=0.001078/2)`
  误用了被定级族的 0.001078（参考族 6400→12800 实测 0.000720/0.000724）。无论按哪个口径，
  实测 0.000474 都不在 h^1 预测（0.000360 或 0.000539）附近。
- **未收口（明示）**：certified 裁决不改变（仍 `CERTIFIED_UNDER_1PCT_AT_NX12800 | HONEST_UNDER_1PCT_B0_B1_B2_B4`，
  总账 <15 % PASS）；「慢收缩成分」的阵面归属未分解。
- 给下次：① 逐阵面分解 (12800→25600) 的 D(x)（接触间断最可疑——无耗散机制）；② 25600 CFL 0.4 档回填后
  分离空间/时间误差；③ 可信参考走异族代码（CLAWPACK/pyclaw，D22 ③）。
