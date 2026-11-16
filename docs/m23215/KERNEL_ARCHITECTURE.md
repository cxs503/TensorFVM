# TensorFVM 数值内核架构（CAP-REG-FVM, m23215）

> 全部数据由 `scripts/m23215_fvm_audit.py` 对工作树做 AST 扫描得到（`docs/m23215/fvm_audit.json` 为机读版），
> 本文件只解释这些数意味着什么。方法是：包内 import 关系解析到**模块**级（相对 import 归一到 `tensorfvm.*`），
> 再把模块聚合到子包；复制变体用「函数体 AST 归一化哈希」判重（改名的复制仍算复制，改了数字的不算）。

## 1. 规模与形状

- 模块 586（顶平铺 196，58134 行；子包 24）
- 函数 6528 / 类 443；总行 198293
- **唯一一处 `Protocol`/`ABC` 接口定义**：`mesh_api.py:Mesh2D`（仅 2D）——全仓没有第 2 个抽象接口
- 定义 `class.step()` 的模块 **39** 个：39 套各自的时间推进，没有共享的 time-integration 层
- 后端分裂：import torch 的模块 192 个、import numpy 413 个、import scipy 135 个（同一物理量的两条张量/数组语义并存）
- 顶平铺模块中**没有任何模块 import 它**的有 74 个（垂直案例代码）

## 2. 子包级 import 依赖图

边 = `src → dst` 的 import 次数（只计包内 import）。`<top>` 表示顶平铺模块。

```mermaid
graph LR
  _top-->|10|verification
  _top-->|3|rans_validation
  _top-->|2|inc_bench
  _top-->|2|rans_ship_resistance
  _top-->|2|wave_tank_td
  _top-->|1|cht
  _top-->|1|ice
  _top-->|1|wave_tank
  adjoint_shape-->|1|_top
  cavitation-->|1|_top
  cht-->|1|_top
  compressible-->|3|_top
  compressible-->|1|verification
  ice-->|2|_top
  immersed-->|2|_top
  immersed-->|1|verification
  inc_bench-->|2|_top
  internal_flow-->|3|inc_bench
  internal_flow-->|1|_top
  maneuvering-->|7|_top
  maneuvering-->|3|rans_ship_resistance
  rans_ship_resistance-->|4|_top
  rans_ship_resistance-->|1|verification
  rans_validation-->|3|_top
  shallow_water-->|1|_top
  ship_freesurface-->|3|_top
  thermal-->|1|_top
  thermal-->|1|internal_flow
  two_phase-->|5|vof_phase0
  two_phase-->|3|_top
  two_phase-->|1|cht
  two_phase-->|1|vof
  twophase_bench-->|4|_top
  twophase_bench-->|2|verification
  twophase_gap-->|3|_top
  twophase_gap-->|3|vof_phase0
  twophase_gap-->|1|two_phase
  twophase_gap-->|1|twophase_bench
  verification-->|200|_top
  verification-->|16|rans_validation
  verification-->|16|shallow_water
  verification-->|11|compressible
  verification-->|4|immersed
  verification-->|4|vof_verification
  verification-->|3|maneuvering
  verification-->|3|vof_phase0
  verification-->|1|cht
  verification-->|1|internal_flow
  verification-->|1|rans_ship_resistance
  vof-->|3|_top
  vof_phase0-->|2|_top
  vof_phase0-->|1|twophase_gap
  vof_threecase-->|3|_top
  vof_threecase-->|3|vof_phase0
  vof_threecase-->|1|twophase_bench
  vof_verification-->|3|_top
  vof_verification-->|2|vof_phase0
  wave_tank-->|2|_top
  wave_tank_td-->|6|wave_tank
  wave_tank_td-->|1|_top
```

| 子包 | 模块 | 行 | 包内 import | 出边（→目标: 次数） |
|---|---|---|---|---|
| `<top>` | 196 | 58134 | 233 | cht:1, ice:1, inc_bench:2, rans_ship_resistance:2, rans_validation:3, verification:10, wave_tank:1, wave_tank_td:2 |
| `adjoint_shape` | 11 | 3428 | 18 | <top>:1 |
| `cavitation` | 8 | 2329 | 9 | <top>:1 |
| `cht` | 13 | 4959 | 11 | <top>:1 |
| `compressible` | 20 | 7809 | 35 | <top>:3, verification:1 |
| `ice` | 7 | 2125 | 11 | <top>:2 |
| `immersed` | 6 | 1681 | 5 | <top>:2, verification:1 |
| `inc_bench` | 5 | 1476 | 5 | <top>:2 |
| `internal_flow` | 21 | 6043 | 38 | <top>:1, inc_bench:3 |
| `maneuvering` | 10 | 2626 | 16 | <top>:7, rans_ship_resistance:3 |
| `rans_ship_resistance` | 6 | 1728 | 5 | <top>:4, verification:1 |
| `rans_validation` | 7 | 1216 | 9 | <top>:3 |
| `shallow_water` | 19 | 11071 | 35 | <top>:1 |
| `ship_freesurface` | 8 | 2644 | 10 | <top>:3 |
| `thermal` | 6 | 1792 | 6 | <top>:1, internal_flow:1 |
| `two_phase` | 8 | 3023 | 8 | <top>:3, cht:1, vof:1, vof_phase0:5 |
| `twophase_bench` | 6 | 1479 | 7 | <top>:4, verification:2 |
| `twophase_gap` | 8 | 2450 | 7 | <top>:3, two_phase:1, twophase_bench:1, vof_phase0:3 |
| `verification` | 188 | 72419 | 165 | <top>:200, cht:1, compressible:11, immersed:4, internal_flow:1, maneuvering:3, rans_ship_resistance:1, rans_validation:16, shallow_water:16, vof_phase0:3, vof_verification:4 |
| `vof` | 5 | 1930 | 5 | <top>:3 |
| `vof_phase0` | 6 | 1487 | 4 | <top>:2, twophase_gap:1 |
| `vof_threecase` | 5 | 1191 | 3 | <top>:3, twophase_bench:1, vof_phase0:3 |
| `vof_verification` | 6 | 1254 | 5 | <top>:3, vof_phase0:2 |
| `wave_tank` | 5 | 1475 | 5 | <top>:2 |
| `wave_tank_td` | 6 | 2524 | 6 | <top>:1, wave_tank:6 |

**结构性结论**：`<top>` 是事实上的内核层（顶平铺被大量子包 import），子包之间几乎只在同族内向依赖；
没有一条从子包指回统一 mesh/flux/BC/linear/time 层的边，因为**那一层不存在**。

## 3. 复制变体族谱：证实与修正

派单背景称「vof×4、twophase×3、wave_tank×2 平行实现」。实测只支持一半：

| 族 | 子包（实测模块数） | 行 | 函数体精确重复 | 判定 |
|---|---|---|---|---|
| two_phase | 22（two_phase, twophase_bench, twophase_gap） | 6952 | 2 | — |
| vof | 22（vof, vof_phase0, vof_threecase, vof_verification） | 5862 | 2 | — |
| wave_tank | 11（wave_tank, wave_tank_td） | 3999 | 1 | — |

### 3.1 vof 族（4 个子包）——**修正**：不是 4 份复制，是 1 份内核 + 3 层下游

- 精确重复的函数体仅 **2** 处：
  - `dx@vof_phase0/dam_break.py` ≡ `dx@vof_threecase/dambreak.py`（同一 `dx/dy` 网格辅助函数，复制但无害）
  - `dy@vof_phase0/dam_break.py` ≡ `dy@vof_threecase/dambreak.py`（同一 `dx/dy` 网格辅助函数，复制但无害）
- 真正的下游依赖（实测 import 边）：`vof_threecase/dambreak.py → vof_phase0/dam_break.py, vof_phase0/anchors.py`；
  `vof_verification/cases.py → vof_plic.py, vof_phase0/advection.py`；`vof/ice_water.py → vof_plic.py`。
- 结论：`vof_phase0` 是事实内核（dam-break/VOF 推进），`vof` 是 THINC + 冰水应用，`vof_threecase`/`vof_verification` 是两代**验证战役**。
  四代「平行」应为「**四代串行 + 2 处小函数复制**」——重复的是战役脚本与指标（`metrics`/`campaign`），不是求解器。

### 3.2 twophase 族（3 个子包）——**部分证实**

- 精确重复 2 处（`dx/dz` 网格步长、`column_depth` 两处实现）。
- 跨族边：`twophase_gap/dambreak_gate.py → two_phase/vof.py, vof_phase0/dam_break.py`；`twophase_bench/* → vof_plic.py`；`two_phase/chain.py → vof_phase0/anchors.py`。
- 结论：`two_phase` 是生产多相（漂移流/空化输运/Stefan），`twophase_bench` 与 `twophase_gap` 是**两代基准与误差闸门**；
  三支共享 `vof_phase0` 的 dam-break 内核，共享方式是 import 而非复制。

### 3.3 wave_tank 族（2 个子包）——**证伪「平行」**

- 精确重复 1 处（`area` ≡ `wetted_area`，面积积分）。
- 依赖方向单向且明确：`wave_tank_td/{cases,potential2d,seakeeping,sigma_tank,wave_target} → wave_tank/theory.py`（5 条边）。
- 结论：`wave_tank` = 造波理论 + 几何/求解库，`wave_tank_td` = 时域耐波上层。**不是两支平行实现**，是「库 + 消费者」。

### 3.4 结论修正表

| 派单描述 | 实测 | 裁定 |
|---|---|---|
| vof 四代平行实现各 5-6 py | 5/6/5/6 模块，精确重复 2 函数体，存在 5 条下游 import 边 | 模块数证实，**「平行」证伪** |
| twophase 三支 | 8/6/8 模块，精确重复 2 函数体 | 模块数证实，来源是三份**战役/闸门**代码 |
| wave_tank 两支 | 5/6 模块，5 条单向 import | **证伪**：库 + 消费者 |

## 4. 共享件清单（实测复用关系）

派单点名的 `native_linear.py` **在工作树中不存在**；实际的线性/共享件是下表这些。

| 共享件 | 被 src 内 import 次数 | 真实使用者（样本） | 判据 |
|---|---|---|---|
| `solver.py` | 32 | `annular.py`, `benchmark.py`, `benchmark_30p30n.py` |  |
| `two_phase_solver.py` | 13 | `ship_freesurface/cfd_verification.py`, `ship_freesurface/verification_runner.py`, `twophase_gap/hysing_gate.py` |  |
| `sparse_simple.py` | 8 | `conservative_pressure.py`, `coupled_coarse.py`, `curved_channel.py` |  |
| `body_fitted.py` | 7 | `annular.py`, `body_fitted3d.py`, `multi_element_least_squares.py` |  |
| `backend_registry.py` | 6 | `annular.py`, `body_fitted.py`, `curved_channel.py` |  |
| `external_linear.py` | 5 | `external_cached.py`, `external_coupled.py`, `external_high_order.py` |  |
| `mesh_api.py` | 3 | `annular.py`, `backend_registry.py`, `curved_channel.py` |  |
| `annular.py` | 6 | `verification/annular.py`, `verification/annular_convergence.py`, `verification/annular_order.py` |  |
| `ellipsoid_added_inertia.py` | 3 | `verification/added_mass_parallel_axis.py`, `verification/ellipsoid_rotational_reference.py`, `verification/rotational_added_mass.py` |  |
| `radiation.py` | 3 | `verification/radiation.py`, `verification/radiation3d.py`, `verification/radiation_occlusion.py` |  |
| `vof_plic.py` | 11 | `two_phase_solver.py`, `twophase_bench/cicsam.py`, `twophase_bench/ritter.py` |  |
| `convergence_xval.py` | 2 | `convergence_diagnosis.py`, `verification/convergence_xval_round2.py` |  |
| `native_hybrid_rans3d.py` | 3 | `verification/native_developed_channel.py`, `verification/native_hybrid_rans.py`, `verification/native_sst_wall.py` |  |
| `native_block_solver3d.py` | 1 | `native_hybrid_stokes3d.py` | 需可选依赖 pyamg（`tensorfvm[amg]`），本环境不可执行 |
| `pmm_solver.py` | 0 | — | **src 内零引用**（仅 `tests/test_pmm_solver.py`） |
| `overset_flux.py` | 0 | — | **src 内零引用**（仅 `tests/test_overset_flux.py`） |
| `native_linear_diagnostics.py` | 1 | `native_simple3d.py` | 诊断件，src 内有引用 |

- **真实内核层**：`solver.py`（2D SIMPLE）、`two_phase_solver.py`（多相）、`sparse_simple.py`/`external_linear.py`（线性后端）、`body_fitted.py`/`mesh_api.py`/`backend_registry.py`（网格）、`vof_plic.py`（自由面几何）。
- **孤儿共享件**：`overset_flux.py`、`pmm_solver.py` 没有任何 `src` 消费者——它们只能通过测试调用，属于「有实现、无接线」（registry 里 `wired=False` 的机械来源）。

## 5. 五接口缺口表（判据：存在可被多方共用的抽象 + ≥2 个实现方 + 有 dispatch）

| 接口 | 现状（实测证据） | 实现份数 | 缺口 | registry 中的替代条目 |
|---|---|---|---|---|
| **mesh** | 仅 `mesh_api.py:Mesh2D` 一个 `Protocol`（2D）；3D 各有各的网格类（`unstructured_fvm3d`/`openfoam_polymesh`/`overset`/`body_fitted3d`/`native_mesh3d`…） | ≥5 | 无 3D 网格契约、无统一面几何/拓扑查询 | `fvm.mesh.contract2d`, `fvm.mesh.unstructured3d`, `fvm.mesh.openfoam_polymesh`, `fvm.mesh.overset` |
| **flux** | 只在域内出现：`vof_plic`/`vof_cicsam`/`vof/thinc`（VOF 面通量）、`compressible/reconstruct`（可压缩斜率限制）、`overset_flux`（唯一接口通用件，但 src 内零引用） | 4 | 无 `flux(face, state) → 通量` 的统一抽象；各族自己造面插值 | `fvm.flux.overset_transfer`, `fvm.freesurface.cicsam`, `fvm.compressible.slope_limiter` |
| **BC** | 9 个 boundary 相关模块（`free_surface_bc`, `native_boundary_constraints3d`, `native_boundary_pressure3d`, `external_open_boundary`, `*_boundary_audit`…），各自内联写边界 | 9 | 无 BC 类型注册表/无 slip-壁面-open-出流统一抽象 | 无（未登记 → 缺口如实保留） |
| **linear** | 5 个线性后端：`sparse_simple`(scipy)、`external_linear`(splu)、`native_block_solver3d`(pyamg 块预条件)、`overset_poisson`、`native_linear_diagnostics`(诊断)；调用侧直接依赖具体类 | 5 | 无 `solve(A, b) → x` 后端协议，无 fallback 链 | `fvm.linear.*`（5 项） |
| **time** | 39 个模块定义 `class.step()`，各 solver 自带时间推进（Crank–Nicolson/PISO/显式 SSP…） | 39 | 无共享时间积分层，无统一 CFL/子步/事件接口 | 无（未登记 → 缺口如实保留） |

**缺口合计**：5 个接口中 **0 个**已统一；`mesh` 只有 2D 骨架，`flux` 只有 1 个无人引用的通用件。这就是重构的靶子。

## 6. 其他机械发现（诚实清单）

- **同名遮蔽模块 1 处**：`wave_tank.py` 与 `wave_tank/__init__.py` 都解析为 `tensorfvm.wave_tank`；包目录胜出，`wave_tank.py` 是**不可 import 的死代码**。
- **STUB 探针**：`1` 个函数体为空/`raise NotImplementedError`，其中唯一命中是 `ship_freesurface/geometry.py::half_breadth` 的**抽象基类方法**（`raise NotImplementedError` 供子类实现）——即内核层**没有真实空壳**。
- **MOCK 探针**：20 个模块含随机数调用（`uses_random`），抽样核对全部是**检验/扰动**用途（对称性检验、限幅器随机场、DNS 初场、标定 study），未发现用随机数冒充求解结果。
- **未登记家底**：586 模块中 586 - 46 未登记；它们既没有能力声明也没有误差带，按本单判据不能进 mature。

## 7. 复现

```bash
python scripts/m23215_fvm_audit.py --json docs/m23215/fvm_audit.json   # 本文所有数字
python src/tensorfvm/registry.py --static                              # 46 项能力（免 torch）
python scripts/m23215_apply_capability_decorators.py --check           # wired/tested 的机械推导
```
