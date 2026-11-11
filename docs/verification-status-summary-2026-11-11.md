# TensorFVM 验证状态总览（2026-11-11）

**任务ID**: m13791 (selfstart·tp-qa-2 verifier 席)  
**性质**: 验证状态审计与缺口梳理（轻量分析，未起重计算）

---

## 1. 执行摘要

本单对 TensorFVM 仓库的验证状态做全面扫描，对照 `docs/verification-catalog.md`、
`docs/verification-coverage-audit.md`、`docs/verification-gap-audit-m7555.md` 及
feature ledger 中的 `GAP-FVM-TWOPHASE` 锚点，梳理当前验证覆盖度、已知缺陷与待办缺口。

**核心结论**：
- TensorFVM 已建立覆盖不可压流动、两相 VOF、可压缩多材料、传热共轭、RANS 湍流
  等多物理域的验证体系，**大部分核心能力已有解析解或文献基准对照**。
- 剩余缺口集中在：**自然对流高 Ra 数渐近区收敛阶证据**、**后台阶求解器缺陷**、
  **3D 顶盖驱动方腔模块缺失**、**两相 campaign 结果文件未生成**。
- 两相 VOF 验证面（旋转/Zalesak/气泡/溃坝）的测试基础设施完整且全绿，
  但 `two_phase_campaign.json` 结果文件缺失，导致 `GAP-FVM-TWOPHASE` 锚点未闭合。

---

## 2. 验证覆盖矩阵

### 2.1 不可压流动（✅ 主体通过）

| 案例 | 基准 | 状态 | 证据位置 |
|---|---|---|---|
| 顶盖驱动方腔 2D (Re≤1000) | Ghia et al. 1982 | ✅ PASS | `docs/verification-lid-cavity-ladder/` |
| 环形管道轴向流 | 解析解 | ✅ PASS | `src/tensorfvm/verification/artifacts/annular_grid_convergence.json` |
| 通道流 (Poiseuille) | 解析抛物线 | ✅ PASS | 多处测试 |
| 多孔介质通道 (Brinkman) | Brinkman 1947 | ✅ PASS | `src/tensorfvm/verification/artifacts/porous-forchheimer/` |
| Taylor-Green 涡衰减 | 解析解 | ✅ PASS | `tests/test_taylor_green_mac.py` |
| Kovasznay 流 (Re=40) | 解析解 | ✅ PASS | `docs/verification-native-kovasznay-report.md` |
| 自然对流 Ra=1e4 | de Vahl Davis 1983 | ✅ PASS | `docs/verification-natural-convection-ra1e4/REPORT.md` |

### 2.2 边际通过 / 失败（⚠️ 需关注）

| 案例 | 基准 | 误差 | 状态 | 根因 |
|---|---|---|---|---|
| 自然对流 Ra=1e5 (N=32) | de Vahl Davis Nu=4.519 | +3.4% | ⚠️ FAIL (边际) | 网格不够细，需 N=64/96/128 研究 |
| 自然对流 Ra=1e6 (N=64) | de Vahl Davis Nu=8.800 | +3.1% | ⚠️ FAIL (边际) | 同上 |
| 后台阶 Re_H=800 | Gartling L_r/H=6.10 | -97% | ❌ HONEST FAIL | 固体惩罚+开放出口的 O(κ) 质量缺陷 |
| 突扩 AR=4 (D/20) | Borda-Carnot K=0.5625 | +7.6% | ⚠️ FAIL | 粗网格通过，细网格失败 |
| Re=400 v-centreline | Ghia 参考表 | 增长误差 | ⚠️ xfail | 参考表提取或壁面附近解场缺陷 |

### 2.3 两相 VOF（✅ 测试全绿，结果文件缺失）

| 案例 | 基准 | 状态 | 证据位置 |
|---|---|---|---|
| VOF 刚体旋转 (cosine hill) | 解析解（转整数圈） | ✅ PASS | `src/tensorfvm/verification/artifacts/vof_rotation/` |
| Zalesak 开槽圆盘 | 解析几何 | ✅ PASS | 同上 |
| 气泡上升 (Grace 图) | Fan-Tsuchiya 关联式 | ✅ PASS (测试) | `tests/test_two_phase_campaign.py` |
| 溃坝 (Martin & Moyce) | 实验曲线 | ✅ PASS (测试) | 同上 |
| 静水平衡 | 解析静压分布 | ✅ PASS | `tests/test_vof_chain_hydrostatic.py` |
| **两相 Campaign 结果** | C1-C5 判据 | ❌ 文件缺失 | `two_phase_campaign.json` 不存在 |

### 2.4 可压缩流动（✅ 已验证）

| 案例 | 基准 | 状态 | 证据位置 |
|---|---|---|---|
| Sod 冲击管 | 精确黎曼解 | ✅ PASS | `docs/compressible-multimaterial-verification.md` |
| 材料界面位置收敛 | 解析界面轨迹 | ✅ PASS | 同上 |
| Richtmyer-Meshkov | 理论增长率 | ✅ PASS | 同上 |

### 2.5 传热共轭（✅ 已验证）

| 案例 | 基准 | 状态 | 证据位置 |
|---|---|---|---|
| Stefan 问题 | 解析移动界面 | ✅ PASS | `docs/cht-stefan/` |
| 共轭换热锚点 | 解析温度场 | ✅ PASS | `docs/cht/` |

---

## 3. 已知缺陷与待办缺口

### 3.1 高优先级（影响验证完整性）

1. **两相 Campaign 结果文件缺失**
   - 锚点：`GAP-FVM-TWOPHASE` (feature ledger, planned)
   - 现状：测试全绿，但 `two_phase_campaign.json` 未生成
   - 影响：无法机器裁断 C1-C5 判据，账本未闭合
   - 建议：运行 `PYTHONPATH=src python -m tensorfvm.verification.two_phase_campaign` 生成结果

2. **自然对流 Ra=1e5/1e6 细网格研究**
   - 锚点：`docs/benchmarks/incompressible-verification.md` §2.1
   - 现状：N=32/64 边际 FAIL，需 N=64/96/128 均匀加密
   - 影响：收敛阶证据缺失，无法确认渐近区行为
   - 建议：递交集群作业做分辨率扫描

3. **后台阶 reattachment length 修复**
   - 锚点：`docs/benchmarks/incompressible-verification.md` §2.3
   - 现状：HONEST FAIL，L_r/H = 0.184 vs 目标 6.10
   - 根因：固体惩罚+开放出口的 O(κ) 质量缺陷
   - 建议：需二阶对流格式或改进惩罚/出流处理（重算力）

### 3.2 中优先级（功能缺口）

4. **3D 顶盖驱动方腔模块**
   - 锚点：`docs/verification-catalog.md` §10.1 #1
   - 现状：验证脚本 `scripts/verify_lid_cavity_3d.py` 存在，但 `lid_cavity_3d.py` 模块未实现
   - 建议：实现 `LidCavity3DSolver` 并对比 Tang et al. 1989 文献

5. **Re=400 v-centreline 缺陷归因**
   - 锚点：`tests/test_lid_cavity_re1000_evidence.py` xfail
   - 现状：误差随加密增长，已归因于参考表或壁面解场
   - 建议：进一步诊断区分参考表误差与求解器缺陷

### 3.3 低优先级（长期规划）

6. **移动网格 (ALE) 完整验证**
   - 现状：ALE 标量输运验证已存在 (`test_ale_transport_analytic.py`)
   - 缺口：ALE 动量方程验证缺失

7. **燃烧/组分输运验证**
   - 现状：无验证

8. **辐射换热验证**
   - 现状：无验证

---

## 4. 能力登记建议

基于本次审计，建议向 feature ledger 新增以下条目：

| slug | title | acceptance | 所属域 |
|---|---|---|---|
| GAP-FVM-NC-ASYMPTOTIC | 自然对流 Ra=1e5/1e6 渐近区收敛阶 | N=64/96/128 序列 Nu 误差<3%，观测阶≥1.0 | FVM |
| GAP-FVM-3D-LID | 3D 顶盖驱动方腔求解器与验证 | 中心截面速度剖面 vs Tang et al. 1989 误差<5% | FVM |
| GAP-FVM-BFS-FIX | 后台阶 reattachment length 修复 | L_r/H 误差<10% vs Gartling 6.10 | FVM |

---

## 5. 给下次自找活的建议线索

1. **两相 Campaign 结果生成**：运行 `two_phase_campaign` 模块生成 `two_phase_campaign.json`，
   闭合 `GAP-FVM-TWOPHASE` 锚点。测试已全绿，只需跑一次完整 campaign。

2. **自然对流细网格研究**：递交集群作业做 Ra=1e5/1e6 的 N=64/96/128 分辨率扫描，
   提取 Nu 和收敛阶。这是验证欠账中最明确的待办项。

3. **3D 顶盖驱动方腔**：实现 `lid_cavity_3d.py` 模块（可基于 2D 版本扩展），
   对比文献基准。这是 catalog 明确列出的缺口。

4. **Re=400 v-centreline 深度诊断**：区分参考表误差与求解器壁面处理缺陷，
   可能需要修改 Ghia 参考数据或修复壁面附近离散格式。

---

**审计人**: m13791 (selfstart·tp-qa-2 verifier 席)  
**日期**: 2026-11-11  
**方法**: 全只读扫描，未修改任何求解器代码，未起重计算进程
