# TensorFVM 求解器与能力清单（CAP-REG-FVM, m23215）

> 本文件由 `src/tensorfvm/registry.py` 的静态 AST dump 与 `scripts/m23215_fvm_audit.py` 的机械审计**程序化生成**；
> 表中每个数字都可用文末命令重算，不存在手抄。冻结判据：`mature` 仅当 `verified and wired`。

## 0. 结论摘要（全部为实测值）

- 注册能力 **46** 项：mature **3** / developing **43** / planned **0**；wired 35、tested 41、verified 4、带 endpoint 3。
- 注册表覆盖 **24** 个子包中的 **23** 个 + 顶平铺模块，指针对工作树全部可解析（error 级 problem = 0）。
- 仓库规模：**586** 个模块 / **198293** 行 / 6528 函数 / 443 类；其中被测试点名的模块 477，被 docs 点名的 79。
- 诚实缺口：586 个模块里只登记了 46 项（7.8%），其余未登记的模块**没有能力声明**——本单不替它们背书。

## 1. 证据分级与成熟度判据

| 本文件等级 | 判据 | registry 字段 |
|---|---|---|
| **A** | 有实测误差带，且 `max_error_pct` 由 `scripts/m23215_error_bands.py` 现算、逐位比对 | `verified=True` + `evidence.grade ∈ {ANALYTIC, XVALUE, CONSERVATION}` + `evidence.max_error_pct` |
| **B** | 只有测试绑定（有 `test_path` 且该文件定义 `test_*`），没有机器核验的误差带 | `tested=True, verified=False`，`grade=MECHANICS` |
| **C** | 仅声明：无测试绑定、无误差带 | `tested=False`，`grade=NONE` |

`mature` 的门：`wired and verified`（服务端同款反虚标规则在注册时即抛 `MaturityDisciplineError`）。
`wired` 的机械判据是「`src/tensorfvm` 内存在**别的**模块 import 它」，不是「`__init__` 导出了它」；`verified` 的机械判据是「误差带文件里有这个 id 的数，且与 registry 里的数逐位相等」（`tests/test_capability_registry_fvm.py::test_registry_error_bands_equal_the_measured_bands`）。

## 2. 能力清单（46 项，按 id 排序）

| 能力 id | kind | maturity | 等级 | max_err% | wired | tested | 模块:入口 | test_path |
|---|---|---|---|---|---|---|---|---|
| `fvm.cavitation.inception` | cavitation | developing | B | — | Y | Y | `tensorfvm.cavitation.inception:cavitation_number` | `tests/test_cavitation_propeller_oracle.py` |
| `fvm.compressible.euler` | compressible | developing | B | — | Y | Y | `tensorfvm.compressible.solver:EulerSolver` | `tests/test_compressible_isentropic_vortex.py` |
| `fvm.compressible.slope_limiter` | compressible | developing | B | — | Y | Y | `tensorfvm.compressible.reconstruct:minmod` | `tests/test_sod1978_convergence.py` |
| `fvm.core.annular_axial` | solver | mature | A | 0.2813 | Y | Y | `tensorfvm.annular:solve` | `tests/test_annular_grid_convergence.py` |
| `fvm.core.ins2d_collocated` | solver | developing | B | — | Y | Y | `tensorfvm.solver:SimpleSolver` | `tests/test_backend_registry.py` |
| `fvm.core.ins3d_cylinder` | solver | developing | B | — | Y | Y | `tensorfvm.solver3d:Cylinder3DSolver` | `tests/test_body_fitted3d.py` |
| `fvm.differentiable.shape_adjoint` | differentiable | developing | B | — | Y | Y | `tensorfvm.adjoint_shape.adjoint:adjoint_gradient` | `tests/test_adjoint_shape_sensitivity.py` |
| `fvm.flux.overset_transfer` | flux | developing | A | 0 | n | Y | `tensorfvm.overset_flux:PlanarFluxTransfer` | `tests/test_overset_flux.py` |
| `fvm.freesurface.algebraic_vof` | free_surface | developing | B | — | Y | Y | `tensorfvm.vof_phase0.advection:AlgebraicVOF` | `tests/test_vof_phase0_advection.py` |
| `fvm.freesurface.cicsam` | free_surface | developing | B | — | Y | Y | `tensorfvm.vof_cicsam:CICSAMVOF` | `tests/test_vof_cicsam.py` |
| `fvm.freesurface.fct` | free_surface | developing | B | — | Y | Y | `tensorfvm.vof_fct:LimitedVOFTransport` | `tests/test_vof_rotation_verification.py` |
| `fvm.freesurface.plic_geometry` | free_surface | mature | A | 0 | Y | Y | `tensorfvm.vof_plic:reconstruct` | `tests/test_vof_plic_interface.py` |
| `fvm.freesurface.thinc` | free_surface | developing | C | — | Y | n | `tensorfvm.vof.thinc:thinc_flux` | — |
| `fvm.freesurface.threecase_campaign` | free_surface | developing | B | — | n | Y | `tensorfvm.vof_threecase.campaign:zalesak_sweep` | `tests/test_vof_threecase.py` |
| `fvm.ice.load_path_closure` | ice | developing | B | — | Y | Y | `tensorfvm.ice.benchmark:row_g4_load_path_closure` | `tests/test_ice_guard_pipeline.py` |
| `fvm.immersed.control_box` | module | developing | B | — | Y | Y | `tensorfvm.immersed.forces:ControlBox` | `tests/test_cylinder_re100_case.py` |
| `fvm.inc_bench.armaly_experiment` | verification | developing | B | — | n | Y | `tensorfvm.inc_bench.armaly:experiment_point` | `tests/test_armaly1983_reproduction.py` |
| `fvm.internal_flow.backward_step` | solver | developing | B | — | Y | Y | `tensorfvm.internal_flow.backward_step:BackwardStepCase` | `tests/test_internal_flow_benchmarks.py` |
| `fvm.linear.block_amg3d` | linear_solver | developing | B | — | Y | Y | `tensorfvm.native_block_solver3d:NativeHybridBlockSolver3D` | `tests/test_native_block_solver3d.py` |
| `fvm.linear.diagnostics` | linear_solver | mature | A | 0 | Y | Y | `tensorfvm.native_linear_diagnostics:linear_system_diagnostics` | `tests/test_native_linear_diagnostics.py` |
| `fvm.linear.overset_poisson` | linear_solver | developing | B | — | Y | Y | `tensorfvm.overset_poisson:solve_overset_poisson` | `tests/test_overset_poisson.py` |
| `fvm.linear.sparse_body_fitted` | linear_solver | developing | B | — | Y | Y | `tensorfvm.sparse_simple:SparseBodyFittedSolver` | `tests/test_sparse_simple.py` |
| `fvm.linear.splu_direct` | linear_solver | developing | B | — | Y | Y | `tensorfvm.external_linear:ExternalFlowSolver` | `tests/test_external_flow.py` |
| `fvm.maneuvering.dynamics` | maneuvering | developing | C | — | Y | n | `tensorfvm.maneuvering.dynamics:ManoeuvreModel` | — |
| `fvm.mesh.backend_registry` | mesh | developing | B | — | Y | Y | `tensorfvm.backend_registry:build_mesh` | `tests/test_backend_registry.py` |
| `fvm.mesh.contract2d` | mesh | developing | B | — | Y | Y | `tensorfvm.mesh_api:validate_mesh2d` | `tests/test_backend_registry.py` |
| `fvm.mesh.openfoam_polymesh` | mesh | developing | B | — | n | Y | `tensorfvm.openfoam_polymesh:PolyMesh` | `tests/test_openfoam_polymesh.py` |
| `fvm.mesh.overset` | mesh | developing | B | — | Y | Y | `tensorfvm.overset:OversetGrid` | `tests/test_multi_solid_overset.py` |
| `fvm.mesh.unstructured3d` | mesh | developing | B | — | n | Y | `tensorfvm.unstructured_fvm3d:UnstructuredFVM3D` | `tests/test_unstr3d_convection_45deg.py` |
| `fvm.multiphase.bench_campaign` | multiphase | developing | C | — | n | n | `tensorfvm.twophase_bench.campaign:run_campaign` | — |
| `fvm.multiphase.dambreak_gate` | verification | developing | B | — | Y | Y | `tensorfvm.twophase_gap.dambreak_gate:decompose_dambreak_error` | `tests/test_dambreak_error_decomposition.py` |
| `fvm.multiphase.drift_flux` | multiphase | developing | B | — | Y | Y | `tensorfvm.two_phase.drift_flux:wallis_drift_velocity` | `tests/test_two_phase_drift_flux.py` |
| `fvm.multiphase.rayleigh_plesset` | multiphase | developing | B | — | Y | Y | `tensorfvm.two_phase.cavitation_transport:plesset_solution` | `tests/test_two_phase_cavitation.py` |
| `fvm.multiphase.solver` | multiphase | developing | B | — | Y | Y | `tensorfvm.two_phase_solver:TwoPhaseSolver` | `tests/test_two_phase_solver_verification.py` |
| `fvm.multiphase.stefan` | multiphase | developing | B | — | Y | Y | `tensorfvm.two_phase.stefan:stefan_lambda` | `tests/test_stefan_convergence_verification.py` |
| `fvm.pmm.section_ns2d` | solver | developing | B | — | n | Y | `tensorfvm.pmm_solver:SectionNS2D` | `tests/test_pmm_solver.py` |
| `fvm.resistance.speed_sweep` | module | developing | B | — | Y | Y | `tensorfvm.rans_ship_resistance.speed_sweep:speed_points_kn` | `tests/test_gci_inventory.py` |
| `fvm.seakeeping.strip_theory` | seakeeping | developing | C | — | n | n | `tensorfvm.wave_tank_td.seakeeping:hydrostatic_stiffness` | — |
| `fvm.seakeeping.wave_theory` | seakeeping | developing | B | — | Y | Y | `tensorfvm.wave_tank.theory:dispersion_k` | `tests/test_p190_n04_wave_tank_metrology.py` |
| `fvm.shallow_water.ritter_exact` | verification | developing | B | — | Y | Y | `tensorfvm.shallow_water.exact:ritter_state` | `tests/test_dambreak_config_envelope.py` |
| `fvm.ship_freesurface.wave_elevation` | seakeeping | developing | B | — | n | Y | `tensorfvm.ship_freesurface.cfd_verification:compute_wave_elevation_from_alpha` | `tests/test_ship_freesurface_cfd.py` |
| `fvm.thermal.conjugate_energy` | thermal | developing | B | — | Y | Y | `tensorfvm.cht.energy:EnergyLedger` | `tests/test_cht_conjugate_anchors.py` |
| `fvm.thermal.rayleigh_benard` | thermal | developing | C | — | n | n | `tensorfvm.thermal.rayleigh_benard:RayleighBenardCase` | — |
| `fvm.turbulence.hybrid_rans3d` | turbulence | developing | B | — | Y | Y | `tensorfvm.native_hybrid_rans3d:NativeHybridRans3D` | `tests/test_native_hybrid_rans3d.py` |
| `fvm.turbulence.validation_matrix` | turbulence | developing | B | — | n | Y | `tensorfvm.rans_validation.matrix:solve_case` | `tests/test_rans_validation_sensitivity.py` |
| `fvm.verification.gci` | verification | developing | B | — | Y | Y | `tensorfvm.vof_verification.metrics:gci` | `tests/test_vof_cicsam.py` |

### 2.1 A 级（有实测误差带，4 项）

| 能力 | 参考解 | max_err% | 说明 |
|---|---|---|---|
| `fvm.core.annular_axial` | closed-form annular axial Poiseuille profile | 0.2813 |  |
| `fvm.flux.overset_transfer` | exact conservation of the face-integrated flux across a non-matching interface | 0 |  |
| `fvm.freesurface.plic_geometry` | algebraic identity A(n, c(alpha)) == alpha (exact cut area) | 0 |  |
| `fvm.linear.diagnostics` | componentwise backward error of a direct solve (exact residual) | 0 |  |

## 3. 子包覆盖（实测：模块数/行数/内部 import/已注册能力数）

| 子包 | 模块 | 行 | 包内 import | 跨包 import | 已注册能力 | 代表能力 id |
|---|---|---|---|---|---|---|
| `adjoint_shape` | 11 | 3428 | 18 | 1 | 1 | `fvm.differentiable.shape_adjoint` |
| `cavitation` | 8 | 2329 | 9 | 1 | 1 | `fvm.cavitation.inception` |
| `cht` | 13 | 4959 | 11 | 1 | 1 | `fvm.thermal.conjugate_energy` |
| `compressible` | 20 | 7809 | 35 | 4 | 2 | `fvm.compressible.euler` |
| `ice` | 7 | 2125 | 11 | 2 | 1 | `fvm.ice.load_path_closure` |
| `immersed` | 6 | 1681 | 5 | 3 | 1 | `fvm.immersed.control_box` |
| `inc_bench` | 5 | 1476 | 5 | 2 | 1 | `fvm.inc_bench.armaly_experiment` |
| `internal_flow` | 21 | 6043 | 38 | 4 | 1 | `fvm.internal_flow.backward_step` |
| `maneuvering` | 10 | 2626 | 16 | 10 | 1 | `fvm.maneuvering.dynamics` |
| `rans_ship_resistance` | 6 | 1728 | 5 | 5 | 1 | `fvm.resistance.speed_sweep` |
| `rans_validation` | 7 | 1216 | 9 | 3 | 1 | `fvm.turbulence.validation_matrix` |
| `shallow_water` | 19 | 11071 | 35 | 1 | 1 | `fvm.shallow_water.ritter_exact` |
| `ship_freesurface` | 8 | 2644 | 10 | 3 | 1 | `fvm.ship_freesurface.wave_elevation` |
| `thermal` | 6 | 1792 | 6 | 2 | 1 | `fvm.thermal.rayleigh_benard` |
| `two_phase` | 8 | 3023 | 8 | 10 | 3 | `fvm.multiphase.drift_flux` |
| `twophase_bench` | 6 | 1479 | 7 | 6 | 1 | `fvm.multiphase.bench_campaign` |
| `twophase_gap` | 8 | 2450 | 7 | 8 | 1 | `fvm.multiphase.dambreak_gate` |
| `verification` | 188 | 72419 | 165 | 260 | 0 | — |
| `vof` | 5 | 1930 | 5 | 3 | 1 | `fvm.freesurface.thinc` |
| `vof_phase0` | 6 | 1487 | 4 | 3 | 1 | `fvm.freesurface.algebraic_vof` |
| `vof_threecase` | 5 | 1191 | 3 | 7 | 1 | `fvm.freesurface.threecase_campaign` |
| `vof_verification` | 6 | 1254 | 5 | 5 | 1 | `fvm.verification.gci` |
| `wave_tank` | 5 | 1475 | 5 | 2 | 1 | `fvm.seakeeping.wave_theory` |
| `wave_tank_td` | 6 | 2524 | 6 | 7 | 1 | `fvm.seakeeping.strip_theory` |
| `<top>`（顶平铺） | 196 | 58134 | 233 | 22 | 20 | — |

## 4. 顶平铺关键模块（按被 import 次数，实测）

| 模块 | 被 src 内模块 import 次数 | 状态 |
|---|---|---|
| `registry.py` | 46 | 未登记 |
| `solver.py` | 32 | 已登记 |
| `verification/core.py` | 24 | 未登记 |
| `native_mesh3d.py` | 18 | 未登记 |
| `rans_validation/channel.py` | 16 | 未登记 |
| `shallow_water/exact.py` | 15 | 已登记 |
| `shallow_water/swe1d.py` | 14 | 未登记 |
| `panel_added_mass.py` | 13 | 未登记 |
| `shallow_water/dam_break.py` | 13 | 未登记 |
| `two_phase_solver.py` | 13 | 已登记 |
| `verification/kmm1987.py` | 13 | 未登记 |
| `native_hybrid_ns3d.py` | 12 | 未登记 |

## 5. 怎么验证的

```bash
python src/tensorfvm/registry.py --static --strict-exit      # 46 caps, 0 error
python src/tensorfvm/registry.py --static --parity           # static↔import 逐字段一致
PYTHONPATH=src python scripts/m23215_error_bands.py          # A 级 4 条误差带现算
python scripts/m23215_fvm_audit.py --check                   # 装饰器↔注册表对账
python -m pytest tests/test_capability_registry_fvm.py -q     # 17 项契约测试
```

