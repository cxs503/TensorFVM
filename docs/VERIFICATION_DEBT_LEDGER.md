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

### D2 自由面/双体镜像附加质量验证
- 来源: 报告 13305 §6（`panel_potential` 已有 single_hull/twin_hull 镜像，未对解析锚验证）。
- 量化验收: 半浸没球（双体镜像）平移附加质量 vs 镜像解析解 <3%；
  水平面 φ=0 边界残差 <1e-8。

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

## 3. 给下次自找活的建议线索
1. 首选 D1：单干、解析锚明确、直接喂 maneuvering.dynamics 的 J_z 缺口，
   且总账无对应锚（新缺口，finish 时申报 NEW 条目）。
2. D3 是纯账目活，任何席位可领，半天内可交。
3. 选题前必跑总账 SQL（owner 2026-11-09 令）：
   `SELECT slug,title,acceptance,status FROM feature_ledger WHERE status IN('planned','in_progress')`。
4. 本仓无 backlog_*/expand_audit_* 文件，本文件即首个缺口账；后续欠账 append 到 §2，
   完成移 §4（append-only，不删旧条）。

## 4. 已完成（done，不得重做）
（空——本账首建）
