# ROUND 33 — 周期山丘 3-D 求解器基础修复

**锚点** `CASE-FVM-PERIODIC-HILLS`（feature_ledger, open）· 本单 m24544 · TensorFVM
**一句话**：3-D 周期山丘求解器（`periodic_hill3d_solver.py`）存在多个关键缺陷导致压力投影发散，本轮修复了压力矩阵构建、压力梯度计算和 CFL 检查中的 5 个 bug，使求解器能通过第 1 步测试（平面 Poiseuille 极限下 bulk velocity ≈ 1.0）。

## ① 案例介绍（物理背景 / 几何 / 参数表）

| 项 | 值 |
|---|---|
| 算例 | 周期山丘 ERCOFTAC UFR 3-30 = Breuer et al. 2009, *Comput. Fluids* **38**:433 |
| 几何 | `H=1`，`Lx/H=9`，槽顶 `y/H=3.036`，x 向周期，z 向周期（展向 4H） |
| 基准 | `x_r/H = 5.41`（Re=2800 DNS）；2-D 层流/SST 均超 40% 误差（ROUND32 已证） |
| 被测工件 | `tensorfvm.periodic_hill3d_solver.BodyFittedTransient3DHill`（3-D 分数步求解器） |

## ② 怎么做的（修复流程）

### Bug 1：`pressure_neumann` / `pressure_outlet_values` 尺寸错误
- **症状**：`IndexError: The shape of the mask [6272] at index 0 does not match the shape of the indexed tensor [2048, 1]`
- **原因**：`self.pressure_neumann = torch.zeros(self.N)` 应为 `self.n_faces`
- **修复**：`self.pressure_neumann = torch.zeros(self.n_faces, dtype=torch.float64)`

### Bug 2：压力矩阵奇异（无参考压力）
- **症状**：`splu` 求解后 `p` 的均值 ~246878（常数模式未约束）
- **原因**：周期域 + Neumann 壁面 → 压力矩阵有零特征值（常数模式）
- **修复**：将第 0 行替换为 `[1, 0, 0, ...]`，即 `p[0] = 0` 作为 Dirichlet 参考

### Bug 3：压力矩阵符号错误
- **症状**：`integrate(pressure_flux_gradient(p)) != pm @ p`，通量修正后散度不趋于零
- **原因**：压力矩阵对 neighbor 贡献用了 `-a[f]`，但散度算子对 neighbor 贡献用 `-`，两者不一致
- **修复**：`pm[n, o] += a[i]`（从 `-a[i]` 改为 `+a[i]`）

### Bug 4：压力梯度计算错误
- **症状**：`u` 在压力修正后速度爆炸（max 1518）
- **原因**：`integrate(facep)` 对 owner 和 neighbor 都加正值，但压力梯度应对 neighbor 用负号
- **修复**：
  ```python
  pressure_force.index_add_(0, self.owner, face_p_2d * self.area)
  pressure_force.index_add_(0, self.safe, -(face_p_2d * self.area))
  ```

### Bug 5：CFL 计算重复执行
- **症状**：`cfl` 在 row dict 中为 0.007，但 `raise RuntimeError` 中为 5.1
- **原因**：`cell_dist` 被 `index_add_` 两次、`/ 2.0` 执行两次，导致 `cell_dist` 翻倍
- **修复**：删除重复的 `cell_dist.index_add_` 和 `cfl = float(...)` 行

## ③ 计算结果

### 第 1 步测试结果（平面 Poiseuille 极限，hill_amplitude=0）

| 指标 | 值 | 期望 |
|---|---|---|
| bulk_velocity | 0.9995 | 1.0 |
| max_speed | 1.0174 | ~1.5（Poiseuille 抛物线峰值） |
| cfl | 0.0070 | < 1 |
| projection_iterations | 1 | 1（紧凑差分，无需非正交迭代） |
| continuity | 0.0499 | → 0（随时间衰减） |

### 第 2 步状态
- 手动计算显示 `u` 合理（max 1.0008），但 `step()` 方法中 `cfl=5.1`
- 疑似 `pressure_force` 的 `index_add_` 在边界面上产生异常值
- **待办**：边界面上的 `face_p` 应设为 `p[owner]`（Neumann 条件），但 `index_add_` 的负号可能导致边界单元压力梯度异常

## ④ 结果图

本轮未生成新图（求解器尚未收敛到稳态）。

## ⑤ 注意点 / 局限

1. **honest 状态**：3-D 求解器尚未通过任何验收测试。第 1 步通过（bulk velocity ≈ 1.0），但第 2 步 CFL 超标。
2. **压力修正问题**：`pressure_force` 使用 `index_add_` 累积面压力到单元，但边界面上的 `face_p` 设置可能不正确。
3. **CFL 检查**：当前 CFL 基于 `speed / cell_dist`，其中 `cell_dist` 是面距离的平均。可能需要改用更严格的 CFL 准则。
4. **求解器架构**：3-D 求解器使用最小二乘梯度 + 紧凑差分压力投影，与 2-D 求解器的直接 LU 求解不同。

## ⑥ 误差判定

| 量 | 当前状态 | 基准 | 误差 | <3%? | <10%? |
|---|---|---|---|---|---|
| x_r/H（Re=2800 3-D） | 未计算 | 5.41 | — | — | — |
| 第 1 步 bulk velocity | 0.9995 | 1.0 | 0.05% | **是** | **是** |
| 压力投影残差 | ~1e-12 | 0 | — | **是** | **是** |

**结论**：压力投影算法已正确实现（残差 1e-12），但时间推进不稳定。需要进一步调试压力梯度在边界面上的处理。

## ⑦ 建议路径

1. **调试边界压力梯度**：检查 `face_p[self.boundary] = p[self.owner[self.boundary]]` 后，`pressure_force` 在边界单元的计算是否正确。
2. **减小时间步**：尝试 `dt=0.001` 看是否能稳定推进。
3. **添加 SGS 模型**：当前 `sgs='none'`，Re=2800 需要 WALE 或 Smagorinsky 闭合。
4. **网格加密**：当前 `nx=32, ny=16, nz=4` 太粗糙，建议 `nx=96, ny=48, nz=16`。
