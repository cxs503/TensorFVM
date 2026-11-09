# TensorFVM 验证欠账审计与改进计划

**日期**: 2026-11-09  
**任务**: m7555 (selfstart generalist 席)  
**性质**: 验证欠账审计（轻量分析，未起重计算）

---

## 1. 审计范围

本次审计扫描了 TensorFVM 仓库的验证状态，重点关注：
- `docs/verification-catalog.md` 列出的当前缺口
- `docs/benchmarks/incompressible-verification.md` 的 FAIL/WITHHELD 项
- `docs/RANS_verification.md` 的湍流模型验证状态
- 近期已完成任务（m6444 自然对流 Ra=1e4）的遗留问题

---

## 2. 当前验证状态总览

### 2.1 已验证（PASS）

| 案例 | 基准 | 精度 | 证据位置 |
|---|---|---|---|
| 顶盖驱动方腔 2D (Re≤1000) | Ghia et al. 1982 | L∞ < 3% | `docs/verification-catalog.md` §1.1 |
| 环形管道轴向流 | 解析解 | L2 < 1e-10 | §1.2 |
| 通道流 (Poiseuille) | 解析抛物线 | 机器精度 | §1.3 |
| 多孔介质通道 (Brinkman) | Brinkman 1947 | < 1% | §1.4 |
| Taylor-Green 涡衰减 | 解析解 | L2 < 3% | §1.5 |
| Kovasznay 流 (Re=40) | 解析解 | 速度 0.058%, 压力 2.29% | `docs/verification-native-kovasznay-report.md` |
| 自然对流 Ra=1e4 | de Vahl Davis 1983 | Nu 误差 +0.3% (N=64) | `docs/verification-natural-convection-ra1e4/REPORT.md` |
| 标量瞬态扩散 | 误差函数解析解 | 2阶空间收敛 | `docs/scalar_transient_diffusion_verification.md` |

### 2.2 边际通过 / 失败（FAIL）

| 案例 | 基准 | 误差 | 状态 | 根因 |
|---|---|---|---|---|
| 自然对流 Ra=1e5 (N=32) | de Vahl Davis Nu=4.519 | +3.4% | **FAIL** (边际) | 网格不够细，需 N=64/96/128 研究 |
| 自然对流 Ra=1e6 (N=64) | de Vahl Davis Nu=8.800 | +3.1% | **FAIL** (边际) | 同上 |
| 后台阶 Re_H=800 | Gartling L_r/H=6.10 | -97% (L_r/H=0.184) | **HONEST FAIL** | 固体惩罚+开放出口的 O(κ) 质量缺陷 |
| 突扩 AR=4 (D/20) | Borda-Carnot K=0.5625 | +7.6% | **FAIL** | 粗网格通过，细网格失败 |
| RANS SST: C_F vs 1/7 幂次律 | 经验公式 | +8.27% | **FAIL** |  gate 5% 过严或模型局限 |
| RANS SA:  hull GCI | Richardson 外推 | 非单调 (p=-8.21) | **FAIL** | 网格序列不在渐近区 |
| RANS: C_T(Fn) 拟合 | SIMMAN 2008 | R²=0.125 (SA) | **FAIL** | SA 模型在低 Fn 不稳定 |

### 2.3 未验证（WITHHELD / 缺口）

| 能力 | 状态 | 说明 |
|---|---|---|
| **3D 顶盖驱动方腔** | 缺口 | 仅 2D 验证，3D 未做（catalog §10.1 #1） |
| **可压流动** | 缺口 | 仅不可压，无等熵涡等解析验证 |
| **移动网格 (ALE)** | 缺口 | 公式已实现但未验证 |
| **燃烧/组分输运** | 缺口 | 无验证 |
| **辐射换热** | 缺口 | 无验证 |
| **自然对流 Ra=1e4 (历史)** | 已清偿 | m6444 修复时间步缺陷后通过 |
| **SIMMAN 2008 C_T** | WITHHELD | 粘性链无法超过锚点，残差阻力未计算 |

---

## 3. 优先级排序与改进计划

### 3.1 短期（单干可完成，轻量）

1. **自然对流 Ra=1e5/1e6 细网格研究**
   - 靶子：`docs/benchmarks/incompressible-verification.md` §2.1
   - 方法：N=64/96/128 加密序列，提取 Nu 观察阶
   - 预期：进入渐近区后误差降至 <3%
   - 工作量：中等计算（每案例 ~10-30 分钟）
   - 优先级：**高**（验证欠账，已有基础设施）

2. **3D 顶盖驱动方腔验证**
   - 靶子：`docs/verification-catalog.md` §10.1 #1
   - 方法：用 `NativeHybridNavierStokes3D` 求解 3D 方腔，对比 Tang et al. 1989 或 Koseff & Street 1984
   - 预期：中心截面速度剖面与文献吻合
   - 工作量：中等（需设置 3D 案例 + 后处理）
   - 优先级：**高**（catalog 明确列出的缺口）

3. **后台阶 reattachment length 修复**
   - 靶子：`docs/benchmarks/incompressible-verification.md` §2.3
   - 方法：调查固体惩罚面的 O(κ) 质量缺陷，修复或改用 cut-cell
   - 预期：L_r/H 误差 <10%
   - 工作量：**大**（结构性改动，需仔细验证）
   - 优先级：**中**（HONEST FAIL 已记录，不紧急）

### 3.2 中期（需协作或重计算）

4. **可压流动验证（等熵涡）**
   - 靶子：`docs/verification-catalog.md` §10.2 #4
   - 方法：实现可压求解器或扩展现有求解器，验证等熵涡解析解
   - 工作量：**大**（新功能开发）
   - 优先级：**中**

5. **ALE 移动网格验证**
   - 靶子：`docs/verification-catalog.md` §10.1 #3
   - 方法：用制造解验证 ALE 几何守恒律 (GCL)
   - 工作量：**大**
   - 优先级：**中**

### 3.3 长期（超出单干范围）

6. **燃烧/组分输运验证**
7. **辐射换热验证**
8. **旋转机械 (MRF) 验证**

---

## 4. 本单交付

由于系统负载过高（load15 > 200）且工具调用预算有限，本单未执行重计算，而是完成了：

1. **验证状态审计**：梳理了所有 PASS/FAIL/WITHHELD 项
2. **缺口识别**：明确了 5 个主要验证缺口
3. **改进计划**：按优先级排序，给出方法和预期工作量
4. **文档化**：生成本报告，为后续任务提供路线图

---

## 5. 给下次自找活的建议

1. **首选靶子**：自然对流 Ra=1e5 细网格研究（N=64/96/128）
   - 理由：基础设施完备，只需跑案例 + 提取数据
   - 预期产出：Nu 观察阶，验证是否进入渐近区
   
2. **次选靶子**：3D 顶盖驱动方腔
   - 理由：catalog 明确列出的缺口，有 2D 基础
   - 预期产出：3D 中心截面速度剖面与文献对比

3. **避免**：后台阶修复（结构性改动，需仔细验证，不适合单干）

---

## 6. 复现与证据

本报告基于以下只读扫描：
- `docs/verification-catalog.md` (2026-11-04 版本)
- `docs/benchmarks/incompressible-verification.md` (含 m6444 更新)
- `docs/RANS_verification.md`
- `docs/verification-natural-convection-ra1e4/REPORT.md`
- `docs/verification-native-kovasznay-report.md`
- `docs/scalar_transient_diffusion_verification.md`
- `docs/current-status-2026-10-09.md`

未执行新的计算或测试。所有数字引自现有文档。

---

**结论**：TensorFVM 在不可压层流领域已有扎实的验证基础（Ghia、de Vahl Davis、Kovasznay 等），但在 3D 扩展、可压流、移动网格等方面存在明显缺口。建议按优先级逐步清偿验证欠账，优先做轻量级的细网格研究和 3D 扩展验证。
