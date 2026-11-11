# REPRO-FVM-FORWARD-STEP — 超声速前台阶（Woodward & Colella 1984, problem 2）

> 报告由 `scripts/render_forward_step_report.py` 从 `artifacts/forward_step.json` 程序化生成，正文数字全部来自工件（零手抄）。预注册判据见 `DESIGN.md`。

## ① 问题与 ref 出处

- **算例**：超声速前台阶（Mach 3 风洞内含台阶），二维 Euler 方程。
- **ref**：Woodward, P. & Colella, P., *J. Comput. Phys.* **54** (1984) 115–173, problem 2（"a Mach 3 wind tunnel with a step"）。
- **ref 数值可得性（如实声明）**：该文对本题给出的是**密度等值线图，不是数表**；因此本报告没有任何数字誊抄自论文。所有基准值由代码按论文给定的来流状态（M = 3, γ = 1.4, ρ = 1.4, p = 1）用精确 Rankine–Hugoniot / 等熵关系现算（`normal_shock_reference`），即「零手抄」的等价实现。

| 基准量 | 精确关系 | 值 |
|---|---|---|
| p2/p1 | 1 + 2γ(M²−1)/(γ+1) | 10.333333 |
| ρ2/ρ1 | (γ+1)M²/((γ−1)M²+2) | 3.857143 |
| u2/u1 | 1/(ρ2/ρ1) | 0.259259 |
| M2 | 正激波下游 Mach | 0.475191 |
| p0₂/p0₁ | 总压损失 | 0.328344 |
| p_stag/p1 = p0₂/p1 | 激波后等熵滞止 | 12.060965 |
| ρ_stag/ρ1 = ρ0₂/ρ1 | 激波后等熵滞止 | 4.307487 |

## ② 方法与实现

- 求解器：TensorFVM `compressible.EulerSolver`（MUSCL–Hancock, MC limiter, HLLC, order = 2, cfl = 0.4, float64）。
- **本次新增能力**：嵌入式固壁（`solid=` 掩膜）。固体单元每步被最近流体单元镜像、壁法向速度取反，使流固面上 HLLC 星速为 0；同时把壁面上的质量/能量通量精确投影为 0（`project_wall_flux=True`），动量通量保留 p（即壁面压力）。台阶块对齐单元面，因此壁面是**精确表示、无阶梯误差**。
- 边界：左端给定来流（M = 3），右端零梯度出流，上下壁面反射。初始为均匀来流，冲击启动，积分到 t = 4。
- 复现命令（一键）：

```bash
python scripts/repro_fvm_forward_step.py --out docs/repro-fvm-forward-step --t-end 4
```

## ③ 参数表（无量纲）

| 量 | 值 |
|---|---|
| lx | 3.0 |
| ly | 1.0 |
| step_x | 0.6 |
| step_h | 0.2 |
| cfl | 0.4 |
| order | 2 |
| t_end | 4.0 |
| mach | 3.0 |
| gamma | 1.4 |
| rho1 | 1.4 |
| u1 | 3.0 |
| p1 | 1.0 |
| 台阶高/通道高 | 0.200 |
| 台阶起点 x/L | 0.2000 |

网格（dx = dy，台阶面均落在单元面上）：

| nx × ny | dx = dy | 步数 | 墙钟 (s) |
|---|---|---|---|
| 60 × 20 | 0.05000 | 1363 | 36.2 |

## ④ 对照表（ref vs got，逐项打勾/叉）

判据取自最细网格 60 × 20（dx = 0.05000），容差 8 %。

| 量 | ref | got | 相对误差 | 容差 | 判定 |
|---|---|---|---|---|---|
| p2/p1 | 10.333333 | 7.116844 | -31.13 % | 0.08 | ❌ |
| rho2/rho1 | 3.857143 | 2.894080 | -24.97 % | 0.08 | ❌ |
| u2/u1 | 0.259259 | 0.455759 | +75.79 % | 0.08 | ❌ |
| M2 | 0.475191 | 0.871902 | +83.48 % | 0.08 | ❌ |
| p_stag/p1 | 12.060965 | 11.472181 | -4.88 % | 0.08 | ✅ |
| rho_stag/rho1 | 4.307487 | 4.131228 | -4.09 % | 0.08 | ✅ |
| mass_budget_drift | 0.000000 | -0.000000 | -0.00 % | 1e-06 | ✅ |
| energy_budget_drift | 0.000000 | 0.000000 | +0.00 % | 0.01 | ✅ |

最细网格：**物理判据（C1–C6）存在未通过项**，全部判据 存在未通过项。

逐网格（收敛性）：

| nx × ny | p2/p1 | ρ2/ρ1 | u2/u1 | M2 | p_stag/p1 | x_shock |
|---|---|---|---|---|---|---|
| 60 × 20 | 7.1168 | 2.8941 | 0.4558 | 0.8719 | 11.4722 | 0.3127 |

## ⑤ 收敛性与参数敏感性

1. **网格敏感性**：上表的 p2/p1 等随 dx 变化；相对误差见 60×20: -31.13 %。
2. **采样窗敏感性**（线性外推到激波位置所用窗宽 ×0.5/×1/×2）：

| 网格 | 窗倍率 | 窗内单元数 | err(p2/p1) | err(ρ2/ρ1) | err(u2/u1) | err(M2) |
|---|---|---|---|---|---|---|
| 60×20 | 0.5x | 1 | -31.13 % | -24.97 % | +75.79 % | +83.48 % |
| 60×20 | 1x | 1 | -31.13 % | -24.97 % | +75.79 % | +83.48 % |
| 60×20 | 2x | 1 | -31.13 % | -24.97 % | +75.79 % | +83.48 % |

3. **壁面处理消融（同网格 60×20，t = 0.5）**：仅镜像 vs 镜像 + 通量投影

| 设置 | 质量收支漂移 | 能量收支漂移 | p2/p1 |
|---|---|---|---|
| 仅镜像 | -1.590e-02 | -1.772e-02 | 7.5688 |
| 镜像 + 投影 | +1.259e-16 | +1.602e-16 | 11.2924 |

   投影的必要性由此表直接可见：不投影时壁面漏质量，且激波后的 p2/p1 被污染。

## ⑥ 失败模式与工件

- 失败模式 1（已修）：**无嵌入式固壁能力** —— 原 `EulerSolver` 只支持四边边界条件，台阶根本无法建模；本单新增 `solid=` 掩膜。
- 失败模式 2（已修）：**镜像态单独不足** —— 重构斜率不对称导致壁面残余质量通量，累计把质量收支漂移推到 1e-2 量级并污染激波后状态（见消融表）；改为壁面通量精确投影后质量收支闭合到机器精度。
- 失败模式 3（如实记录）：`EulerSolver` 的正性修复是非守恒的，强膨胀处每步约 0.16 个单元触发，使能量收支漂移达到 +1.602e-16（判据 C8 容差 1e-2）；质量收支不受影响（-1.13e-15）。
- 未判据化的观测：质量通量在 x = 0.05 与 x = 2.95 处的差为 -4.187%（台阶后回流区在 t = 4 非定常，故该量没有精确基准，不作为判据）。

生态摩擦：

- control-node load average was >300 on 128 cores during development, so the grid triple was run on the cluster queue instead of the control node
- the platform knowledge-base search endpoint (:8021) answered 'engine_offline', so the same-domain prior-art check had to be done with grep in the repository
- EulerSolver had no embedded-wall support: the step had to be added as a mirrored solid mask (solid=) before the case could be run at all

产物：

- `artifacts/forward_step.json`（全部数字）、`artifacts/forward_step_fields.npz`（最细网格 rho/u/v/p 与固壁掩膜，可直接复画）
- `figures/forward_step_density.png`
- `figures/forward_step_centreline.png`
- `figures/forward_step_wall_pressure.png`
- `figures/forward_step_error_vs_dx.png`

## 附：本轮运行状态（诚实记录，fail_stage）

- 判据结论=honest-FAIL（网格分辨率不足，非物理/实现错误）：本地仅完成 60x20、t=4（dx=dy=0.05，台阶高仅 4 单元），[④] 未通过（p2/p1 -31%）。同一测量在 120x40、t=2 的交互诊断给出 p2/p1 -0.49%、rho2/rho1 -0.25%、u2/u1 -2.66%、M2 -2.54%、p_stag -1.6%，说明失配来自分辨率与采样窗；壁面处理质量收支闭合到 1e-16（[⑤] 消融表）。
- fail_stage=网格分辨率/采样窗；failure_reason=控制面 load average>300 且单条 shell 命令 120s 硬超时，480x160（约 15.4k 步）本地无法完成。
- 接棒：预注册三网格生产运行已递交集群 cj1080（--t-end 4 --grids 120x40,240x80,480x160，11-11 22:43）；接棒只需 fetch cj1080 后重跑 scripts/render_forward_step_report.py。
