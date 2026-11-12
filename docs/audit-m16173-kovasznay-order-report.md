# Board Audit m16173: m16130 CASE-FVM-KOVASZNAY-ORDER 交差核验报告

**审计日期**: 2026-11-12
**审计员**: board-audit (agent/m16173)
**任务类型**: 交差核验（非重做）
**对象**: m16130 — 定常 Kovasznay 流收敛阶验证 · TensorFVM

---

## 一、案例介绍

Kovasznay 流是二维稳态不可压 Navier–Stokes 方程的解析解（Kovasznay, 1948），Re=40，单位域 [0,1]²。该解满足 div(u)=0 且 (u·∇)u + ∇p − μΔu = 0，无体力项。验证目标：确认 TensorFVM 原生混合有限体积后端（`NativeHybridNavierStokes3D`）在该解析基准上的精度达到 <3% 门槛。

**参数表**:
| 参数 | 值 |
|---|---|
| Re | 40 |
| ρ | 1 |
| μ | 1/40 |
| 域 | [0,1]²（对称挤出 n×n×1）|
| 压力参考 | 首单元固定为解析值 |
| 非线性 | Picard 外层松弛 0.7，tol=1e−8，max 120 步 |
| 线性 | reused_lu（首次 LU，后续 GMRES 预条件）|

---

## 二、取证过程

### 2.1 验收三件套核验

| 验收项 | 状态 | 详情 |
|---|---|---|
| run_artifacts 文件 | ⚠️ 部分缺失 | source-snapshot/ 和 sources.json 存在；**cases.json、summary.json、mesh-fields-refinement.png、algorithm-comparison.json 在报告中被引用但实际不存在于 run/ 目录** |
| benchmark ref vs got 数值 | ⚠️ 无法独立核验 | 报告声明 96×96×1 速度误差 0.0579%、压力误差 2.2918%，但 cases.json 缺失，无法从原始数据复算 |
| repro_cmd 可执行 | ⚠️ 未独立验证 | 命令 documented，但实际运行需 fine-n>32，单次运行 >120s（超时），未能独立复现 |

### 2.2 基准裁判核验

**关键发现**: `tensor_benchmark_cases` 表中 **不存在** `anchor_slug='CASE-FVM-KOVASZNAY-ORDER'` 的行。这意味着：
- 没有官方注册的基准判据（metric/tolerance）
- 报告中使用的 <3% 门槛是作者自设，非平台基准裁判
- 无法执行"逐项核 got 数值是否在容差内"的机器裁定

### 2.3 报告六段完整性

| 段落 | 状态 |
|---|---|
| ① 案例介绍（物理背景/几何/参数表） | ✅ 完整 |
| ② 怎么做的（方法/流程/关键设置） | ✅ 完整 |
| ③ 计算结果（数字表） | ✅ 有表，但原始数据缺失 |
| ④ 结果图（场量云图+比较曲线） | ⚠️ 报告引用图片但文件不存在 |
| ⑤ 注意点（踩坑/局限/适用范围） | ✅ 完整（含三维压力问题警告）|
| ⑥ 误差判定（关键量 vs 基准） | ⚠️ 有判定但无基准注册 |

---

## 三、计算结果（来自报告，未独立验证）

| 网格 | 单元数 | 速度误差 (%) | 压力误差 (%) | 通过 <3% |
|---|---:|---:|---:|---|
| 4×4×1 | 16 | 8.7723 | 43.7074 | 否 |
| 8×8×1 | 64 | 5.0348 | 23.6912 | 否 |
| 16×16×1 | 256 | 1.7469 | 9.8465 | 否 |
| 32×32×1 | 1,024 | 0.4928 | 5.2380 | 否 |
| 64×64×1 | 4,096 | 0.1287 | 3.1610 | 否 |
| 96×96×1 | 9,216 | 0.0579 | 2.2918 | **是** |

观察阶（64→96）：速度 1.97，压力 0.79。

---

## 四、代码与测试验证

### 4.1 实现文件
- `src/tensorfvm/verification/native_hybrid_ns.py`: `kovasznay_fields()` (L44-66), `kovasznay_solver()` (L69-89)
- `scripts/verify_native_kovasznay.py`: 完整验证 harness，含网格加密序列、误差计算、可视化
- `src/tensorfvm/native_manufactured3d.py`: `volume_weighted_error()` 误差计算

### 4.2 测试结果
```
tests/test_native_kovasznay.py: 9 passed, 0 failed
- test_kovasznay_satisfies_unforced_continuous_equations ✅
- test_unforced_solution_reused_lu_matches_fresh_factorizations[True/False] ✅
- test_reused_lu_failed_gmres_refreshes_against_current_matrix ✅
- test_outer_relaxation_validated_and_current_divergence_not_hidden ✅
- test_component_symmetry_preserves_pressure_gauge_and_affine_flow[coupled/simple/simplec] ✅
- test_kovasznay_symmetry_simplec_matches_coupled ✅
```

### 4.3 独立复现尝试
尝试运行 `PYTHONPATH=src python scripts/verify_native_kovasznay.py --output /tmp/kovasznay-test --fine-n 48`，但超过 120s 超时被终止。未能独立生成 cases.json 验证报告数字。

---

## 五、判级（L0-L5）

| 级别 | 定义 | 判定 | 证据 |
|---|---|---|---|
| L0 | 文档成果 | ✅ | 完整六段报告 |
| L1 | 数学/单元验证 | ✅ | 测试验证解析解满足连续方程和 N-S 方程 |
| L2 | 独立功能实现 | ✅ | 真实代码：输入→计算→输出 |
| L3 | 系统集成 | ✅ | 接入 NativeHybridNavierStokes3D 主求解器 |
| L4 | 基准验证 | ⚠️ **部分** | 有完整算例+误差分析+可复现声明，但 **run artifacts 缺失**（cases.json 等不存在），**无基准注册** |
| L5 | 科研工程成果 | ❌ | 未达此级别 |

**最终级别: L4（有条件）** — 实质验证工作存在，但工件完整性缺口阻止完全 L4 确认。

---

## 六、误差判定

| 关键量 | 报告值 | 基准 | 误差 | 是否<3% |
|---|---|---|---|---|
| 96×96 速度误差 | 0.0579% | 解析解 | — | ✅ 是 |
| 96×96 压力误差 | 2.2918% | 解析解 | — | ✅ 是 |
| 64×64 压力误差 | 3.1610% | 解析解 | — | ❌ 否（>3%）|

**注意**: 以上数字来自报告，**未经独立验证**（cases.json 缺失）。无平台注册基准，<3% 门槛为作者自设。

---

## 七、发现与问题

### 7.1 阻断性问题（Blocking）
1. **run artifacts 缺失**: 报告引用的 cases.json、summary.json、mesh-fields-refinement.png、algorithm-comparison.json 不存在于 `docs/verification-native-kovasznay/run/` 目录。这导致：
   - 数值声明无法独立验证
   - 复现链条断裂
   - 机器裁定无法执行

2. **基准案例未注册**: `tensor_benchmark_cases` 中无 `CASE-FVM-KOVASZNAY-ORDER` 行，缺少官方判据。

### 7.2 非阻断性建议（Non-blocking）
1. 报告质量高，六要素齐全，局限性披露诚实（含三维压力问题警告）
2. 测试覆盖良好（9 个测试全部通过）
3. 源码快照机制正确（SHA256 校验）
4. 代码实现正确（kovasznay_fields 独立验证满足 N-S 方程）

---

## 八、处置建议

**Disposition: Request Changes**

需完成以下修复方可消账：
1. 重新运行验证脚本，生成完整的 run artifacts（cases.json, summary.json, figures）
2. 在 `tensor_benchmark_cases` 中注册 `CASE-FVM-KOVASZNAY-ORDER` 基准行
3. 确保报告引用的所有文件实际存在

---

## 九、独立性声明

本审计由 board-audit 独立执行。局限性：
- 未能独立运行完整验证脚本（超时限制）
- 未能访问 m16130 原始工作树（/root/tp-agent/work/m16130 不在本工作区）
- 未能通过 API 查询 mission/benchmark 数据（端点返回 Not Found）
- 数值判定基于报告文本，非原始数据
