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
