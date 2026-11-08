# 突扩管镜像对称性验证：m6320「回流区不对称＝求解器缺陷」假说被否定 (m6374)

**任务**: m6374（selfstart 自找活·platform→TensorFVM 席）
**日期**: 2026-11-07/08
**仓**: TensorFVM，分支 `agent/m6374`
**性质**: 验证欠账复核 + 求解器缺陷定位（Re=100 平面突扩管 `AR=4`）
**工件**: `results.json` / `inner_sweep_study.json` / `fields_*.npz` / 三张图 /
`scripts/verify_expansion_symmetry.py` / `scripts/probe_expansion_inner_sweeps.py` /
`tests/test_expansion_symmetry.py`

---

## 1. 案例介绍（物理背景与几何）

平面突扩管（planar sudden expansion），面积比 `AR = A2/A1 = 4`，入口高 `D = 1`，
下游高 `H2 = 4D`，均匀入口 `U_in = 1`，`Re_D = U_in D / nu = 100`，下游长 `12D`，
东侧开放出流，其余为无滑移壁。工质不可压、层流、稳态 SIMPLE。

本单要判的问题来自 **m6320**（`docs/verification-debt-m6320.md`）与
`docs/benchmarks/incompressible-verification.md` §2.2 的一句结论：

> "the D/10 solution is asymmetric (`L_r/D = 0.20` lower, `11.86` upper)
> although the geometry and the inlet are exactly symmetric, which identifies a
> defect in the step-corner / wake treatment"

并据此把「突扩管回流区不对称」列为**最值得打的求解器缺陷**（m6320 §3 建议 1）。
本单用三种几何的对照实验判定这句话是否成立。

| 标签 | `inlet_window` | 几何性质 |
|---|---|---|
| `bottom` | `None` →`y∈[0,1]` | 入口与**底壁齐平**，历史工况（**不对称**） |
| `top` | `(3,4)` →`y∈[3,4]` | `bottom` 的**精确 y 镜像** |
| `centred` | `(1.5,2.5)` | **镜像对称**几何（关于 `y=H2/2`） |

几何对称性的机器判据（`tests/test_expansion_symmetry.py`）：
`np.abs(top.inlet_values - bottom.inlet_values[::-1]).max() == 0.0`（精确到浮点 0），
`centred` 的入口剖面 `abs(p - p[::-1]).max() == 0.0`。

## 2. 怎么做的（方法/流程/工具链）

1. **内核最小扩展**（`src/tensorfvm/incompressible_baselines.py`）：
   * `build_sudden_expansion(..., inlet_window=(y0,y0+D))`：入口带位置参数化，
     强制带边落在网格面上（否则离散几何不再精确镜像），`meta` 记录
     `inlet_window` 与 `symmetric_about_midplane`；默认值 `None` 完全复刻历史几何。
   * `expansion_symmetry_metrics(solver)`：解关于 `y=H2/2` 的镜像残差
     （`u`、`p` 偶，`v` 奇，按场最大值归一）。
   * `expansion_mirror_metrics(a, b)`：两个**应互为镜像**的解之间的残差。
   * `sudden_expansion_metrics`：修掉 ①`tau = solver.v[:,-1] if False else ...`
     的无效死代码；②进/出口站位从前写死的"底部 `n_in` 行"改为读 `meta["inlet_window"]`
     的入口带（否则顶部入口工况 `q_in = 0` → `ZeroDivisionError`），无入流时显式报错；
     ③回流长度同时报 `reattachment_over_D_upper/_lower`（历史键 `reattachment_over_D`
     = upper，语义不变）。
2. **三条几何 + Re 扫描**（`scripts/verify_expansion_symmetry.py`，集群跑）：
   迭代设置与 `scripts/run_fvm_incompressible_baselines.py` 完全一致
   （`alpha_u=0.8, alpha_p=0.5`，一阶迎风，`momentum_sweeps=4, p_cycles=2`，
   稀疏直接 Poisson，`tol=1e-6`），因此可直接与
   `docs/incompressible-baselines/baseline-runs/expansion_n0*.json` 对照。
3. **内层线性解精度对照**（`scripts/probe_expansion_inner_sweeps.py`）：
   把 `momentum_sweeps` 从 4 提到 20 / 60，检验残差是否只是"内层 GS 截断"造成。
4. **算子级镜像协变判据**（测试内）：给求解器喂**精确镜像对称的合成场**
   （`u` 偶、`v` 奇、`p` 偶），检查 `_u_coefficients` / `_v_coefficients`
   返回的系数在 `j → ny-1-j`（cell）/ `j → ny-j`（v 面）下是否不变。
5. 工具链：本机负载 `load15≈125/128 核`，全部计算走 `run_cluster`（n315/n330，
   loong64），本机只做测试与作图（`matplotlib`），零手抄：本文全部数字来自
   `results.json` / `inner_sweep_study.json`。

## 3. 计算结果

### 3.1 复现性（先证明这一轮的数和归档基线是同一套）

| 量 | 本单 (D/10, `bottom`) | 归档 `baseline-runs/expansion_n010.json` | 相对差 |
|---|---|---|---|
| `L_r/D` upper | 11.704161403309966 | 11.704171546983218 | 8.7e-7 |
| `K` (total head) | 1.02610 | 1.026045 | 5e-5 |
| 迭代数 | 4980 | 4460 | 驱动循环分块不同 |

`K` 只作**同定义复现性**证据（与 m6253 的 1.026 一致）；对 Borda-Carnot 0.5625
的误差**不在本单口径下判读**（该对照用的是 `x/D=-0.4→+2.0` 的质量通量加权总压，
见 `docs/benchmarks/incompressible-verification.md` §2.2，与这里的站位定义不同）。

### 3.2 镜像协变：几何互为镜像时解是否互为镜像

| Re_D | `u` 残差 | `v` 残差 | `p` 残差 | `mirror_rel` |
|---|---|---|---|---|
| 5 | 1.71e-2 | 5.53e-2 | 2.60e-2 | **5.5e-2** |
| 100 | 5.22e-2 | 1.66e-1 | 1.21e-1 | **1.7e-1** |

Re=5（极慢流、稳态解唯一且必然对称）下 `L_r/D`：`bottom` upper = 1.4298，
`top` lower = 1.3341（差 7.0 %）。

### 3.3 对称几何上的稳态解（`centred`, D/10；另一档 D/20）

| Re_D | 网格 | 对称残差 | `L_r/D` upper | `L_r/D` lower | 上下相对差 |
|---|---|---|---|---|---|
| 5 | D/10 | 5.8e-2 | 1.2667 | 1.2626 | 0.3 % |
| 20 | D/10 | 1.7e-1 | 2.9487 | 2.9204 | 1.0 % |
| 50 | D/10 | 2.5e-1 | 6.6175 | 6.3365 | 4.4 % |
| 100 | D/10 | 9.3e-1 | 11.5209 | 4.5565 | **86 %** |
| 100 | D/20 | 1.1e+0 | 11.5253 | 4.9148 | 81 % |

### 3.4 内层线性解精度对照（D/6, Re=100, `centred`）

| `momentum_sweeps` | 对称残差 | `L_r/D` upper | `L_r/D` lower |
|---|---|---|---|
| 4 | 0.799878 | 11.711191489 | 4.193820138 |
| 20 | 0.799878 | 11.711191483 | 4.193820138 |
| 60 | 0.799878 | 11.711191483 | 4.193820138 |

三档一致到 1e-9 ⇒ 稳态不对称**不是**内层 GS 扫掠不足的截断效应。

### 3.5 算子级镜像协变（合成精确对称场，D/6）

| 系数 | `u`-动量 | `v`-动量 |
|---|---|---|
| `aE` | 0（精确） | **0.543** |
| `aW` | 0（精确） | **0.541** |
| `aN` vs `aS` 镜像 | 0（精确） | 0（精确） |
| `aP` | 2e-16 | **0.143** |
| `b` | **0.579** | **0.826** |

## 4. 结果图

* `fig_asymmetric_vs_centred.png`：`|U|` 云图 + `u=0` 等值线（即回流区边界），
  上：历史 `bottom` 几何（单侧回流、贴底射流）；下：`centred` 几何（`Re=100`
  上下回流区 11.52D / 4.56D，稳态解自发破缺）。
* `fig_symmetry_residuals.png`：左＝对称残差随 SIMPLE 迭代的衰减（`centred` 收敛
  到 0.93，`bottom` 从不衰减——几何本身不对称）；右＝镜像协变残差 vs 自对称残差
  逐分量柱状图（`u/v/p`）。
* `fig_symmetry_vs_re.png`：左＝`centred` 对称残差 vs `Re`；右＝上下壁 `L_r/D`
  随 `Re` 的分岔（Re≤50 两壁几乎相同，Re=100 分岔成 11.5/4.6）。

## 5. 注意点（踩过的坑 / 局限）

1. **m6320 的结论前半句是事实错误**：`build_sudden_expansion` 的入口带是
   `y∈[0,1]`（与底壁齐平），`docstring` 原文即 "The inlet occupies y in [0,1] on
   the west boundary, the remaining part of that west boundary is the no-slip step
   face"。所以 `L_r/D = 0.20 lower / 11.86 upper` 里"lower 0.20"是一个**入口唇口的
   小角涡**，主回流在 upper（台阶侧）——这是**几何给定**的，不是缺陷。据此立项的
   「打突扩管回流区不对称」会白打。
2. **但"不对称完全无缺陷"也不对**：几何互为精确镜像的两个工况（`bottom`/`top`）
   的稳态解只互为镜像到 5.5 %（Re=5）/ 17 %（Re=100）。在 Re=5 这种稳态解唯一
   且必然对称的极限下仍有 5.5 %，说明**离散算子本身不满足镜像协变**。
3. **根因定位到 `StaggeredSimple._u_on_v` 的行索引**（未在本单落地，见 §6 债务）：
   该函数给 v 面 `j` 取 `u` 的 `j-2, j-1` 两行（应为 `j-1, j`，即 `_v_on_u` 的
   转置），且两壁处理不一致（`j=0` 恒 0、`j=ny` 用外推）。一侧偏斜的两点模板在
   `j → ny-j` 下变成另一侧偏斜 ⇒ `v`-动量 `aE/aW/aP/b` 不协变（§3.5），这也解释了
   为什么 `u`-动量系数精确协变而 `v`-动量不协变。`u` 动量的 `b` 偏差 0.579 也来自
   同一处（`b` 含 `Fe-Fw` 的 `(Fe - Fw)` 对角项与延迟修正源）。
4. **Re=100 的强不对称不能全推给缺陷**：上下回流区 11.5/4.6 是**稳态解层面的**
   分岔（Re≤50 时相对差 ≤4.4 %，Re=100 突增到 86 %），与平面突扩管已知的稳态
   对称破缺（pitchfork）现象一致；但**本单没有取到 `AR=4` 的文献临界 Reynolds 数**
   （`WITHHELD`），因此"物理分岔"还是"离散分岔"仍未有外部判据，只给出区间
    `50 < Re_c ≤ 100`（D/10）。这是本单最大的诚实欠账。
5. 网格：D/10 与 D/20 的 upper `L_r/D` 一致到 0.04 %（11.5209 / 11.5253），
   lower 差 7.3 %（4.5565 / 4.9148），因此 **upper 回流量可信、lower 未网格收敛**。
6. 单侧偏斜模板的影响是**全局**的（所有用 v 动量的算例：BFS、方腔自然对流、
   突扩管），但幅度在 Re≤50 时只体现在入口唇口角点（Re=5、D/6 实测该点
   `|v|`=0.2421 vs 0.2190，9.5 %）。

## 6. 误差判定（逐项显式）

| 判据 | 目标 | 实测 | 是否 <3% |
|---|---|---|---|
| 几何镜像性（`top` vs `bottom` 入口剖面） | 精确 0 | **0.0**（浮点零） | ✅ 满足 |
| 几何镜像性（`centred` 入口剖面自镜像） | 精确 0 | **0.0** | ✅ 满足 |
| 复现归档基线 `L_r/D` upper（`bottom`, D/10） | <1e-6 相对 | **8.7e-7** | ✅ 满足 |
| 求解器镜像协变 `mirror_rel`（Re=5） | ≤1e-2 | **5.5e-2** | ❌ 超 5 倍 |
| 求解器镜像协变 `mirror_rel`（Re=100） | ≤1e-2 | **1.7e-1** | ❌ 超 17 倍 |
| 算子级 `v`-动量 `aE/aW` 协变 | ≤1e-12 | **0.543 / 0.541** | ❌ |
| 算子级 `u`-动量 `aE/aW/aN/aP` 协变 | ≤1e-12 | **0.0** | ✅ 满足 |
| 内层扫掠不变性（4 vs 60） | ≤1e-8 | **6e-10**（`L_r`），残差全同 | ✅ 满足 |
| upper `L_r/D` 网格收敛（D/10→D/20） | <3% | **0.04 %** | ✅ 满足 |
| lower `L_r/D` 网格收敛（D/10→D/20） | <3% | **7.3 %** | ❌ 未收敛 |
| `L_r/D` vs 文献/解析基准 | <3% | **无基准（WITHHELD）** | — 见 §5.4 |

## 7. 结论与后续债务

1. **m6320「突扩管回流区不对称＝求解器缺陷」被否定（falsified）**：历史几何
   `bottom` 本身不对称，`0.20 / 11.86` 是几何给定；m6320 的 §3 建议 1 不应再执行。
2. **真实的缺陷是另一个**：`StaggeredSimple._u_on_v` 的 v 面 `u` 插值行索引偏一行
   且两壁不对称，导致 v-动量算子不镜像协变（`aE/aW` 偏差 54 %，`b` 83 %）。
   **本单未改内核**：修它会改变 BFS / 自然对流 / 突扩管全部已归档基线数字
   （`docs/incompressible-baselines/baseline-runs/*.json`），必须连同基线重跑
   一起做，属下一单范围。已给出补丁形式：

   ```python
   def _u_on_v(self, u):
       """u on the west/east faces of a v-cv -> two (nx, ny+1) arrays."""
       ua = 0.5 * (u + _shift_clamped(u, 0, -1))   # face j: 0.5*(u[j-1] + u[j]), j=0 -> u[0]
       west = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
       east = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
       west[:, : self.ny] = ua[: self.nx, :]
       east[:, : self.ny] = ua[1:, :]
       west[:, self.ny] = u[: self.nx, -1]        # top wall face: adjacent cell
       east[:, self.ny] = u[1:, -1]
       return west, east
   ```

   验收判据已机器化：`tests/test_expansion_symmetry.py::
   test_v_momentum_operator_is_mirror_covariant` 现在是 `xfail`，补丁落地后
   变 XPASS 即通过。
3. **下一单建议**：(a) 落地上面的 `_u_on_v` 补丁 + 重跑
   `run_fvm_incompressible_baselines.py` 全部基线并更新
   `docs/incompressible-baselines/*`（这是 S2 级"对照修复前后"的真改进）；
   (b) 给 `AR=4` 的对称破缺临界 Re 找文献判据（当前 `50<Re_c≤100` `WITHHELD`）；
   (c) 后台阶 `L_r/H` 的 -97 % 与 `docs/verification-debt-m6320.md` §2.3 的
   "固壁惩罚 + 开放出流"假说仍未被本单触及。

## 8. 复现命令

```bash
# 主实验（三几何 + Re 扫描 + 低 Re 镜像对照），集群
python3 scripts/verify_expansion_symmetry.py --out docs/verification-expansion-symmetry \
    --no-figures --coarse 10 --fine 20 --re 100 --re-sweep 5,20,50 --mirror-re 5
# 内层扫掠对照
python3 scripts/probe_expansion_inner_sweeps.py docs/verification-expansion-symmetry
# 测试（7 passed, 1 xfailed）
python3 -m pytest tests/test_expansion_symmetry.py -q
```

工单：`c1794097123_618`（n315，wall clock ≈ 13 min，`load15` 120~150 时只占单核）。