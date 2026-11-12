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
| 提案池 | /root/tp-agent/proposals/ 现存 p9087…p97 等 20 条，无 FVM 附加质量/转动惯量在飞单 |
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

### D14 等效沙粒粗糙度 k_s（粗糙壁 RANS）验证面 —— 已交付（m16801, 2026-11-13），高 k_s^+ 段 honest FAIL
- 缺口来源: `rans_validation` 只支持水力光滑壁（`wall_values` 给光滑 ω_w），任何 k_s/粗糙度
  能力在仓内**完全缺失**（`grep -i rough` 只命中 `biomimetic_riblet` 的相关式库，无壁面条件）；
  而 KMM180 只锚定了光滑壁 → 工程阻力（涂装/污损/ plating roughness）无入口。总账外新缺口。
- 交付: `src/tensorfvm/rans_validation/roughness.py`（Nikuradse 差分锚 + Wilcox(1988) `S_R` +
  壁值封顶 + 穿越点二分 + 粗糙对数律拟合）+ `channel.solve_channel(..., k_s=)` 最小挂钩
  （默认 None 逐位同旧解）+ `src/tensorfvm/verification/rough_wall_channel.py`（预声明判据 R1–R10，
  机器判级，`summary.json`）+ `tests/test_rough_wall_roughness.py`（合成/契约/登记三层）+
  `docs/verification-rough-wall-rans/{REPORT.md,summary.json,fig_roughness_function.png,make_report.py}`。
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
- 明示未做（WITHHELD）: k-ε 壁面函数粗糙化（`u_wall` 与 ε_w 需一致化，现抛 NotImplementedError）、
  过渡区（k_s⁺<30）Hama 型过渡函数、方向性粗糙（riblet/砂纹）、粗糙层内分辨。
- 给下次: 建议总账新条目 `FEAT-FVM-ROUGH-WALL-KS`（验收=本文 R1–R10，工程可用域 k_s⁺≤100）；
  后续首项=k-ε 粗糙壁面函数（工业船体常用模型），次项=Eça/Hoekstra `k_s` 型 ω 壁条件替代 `S_R` 以修高 k_s⁺ 段。
