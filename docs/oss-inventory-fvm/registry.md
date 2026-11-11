# TensorFVM 开源 FVM 技术盘点：仓级注册表与分级（registry + tier）

> 程序化生成，请勿手改：`python scripts/oss_inventory_fvm/build_registry.py` → `python scripts/oss_inventory_fvm/render_report.py`。
> 生成时间 2026-11-11T12:23:26+00:00；目标仓 TensorFVM；mission 14887。

## 0. 证据口径（读表前必读）

| 通道 | 含义 | 可信度 |
|---|---|---|
| `metadata.source = github_api` | 本轮实测：stars / SPDX licence / 最近 push / 语言，逐仓记录 slug 与抓取时间 | **实测** |
| `metadata.source = upstream_not_github` | 项目主仓不在 GitHub（自建 GitLab / darcs / SourceForge / NASA 发布），本轮**无法**测量元数据 | 未测（字段留空，不得引用为现状） |
| `metadata.source = commercial` | 闭源商业软件，仅作验收基线 | 不可引用 |
| `curation.*` | 专家标注：技术族/离散方式/能力标签/分级/缺口映射 | **标注**（本轮未逐仓读源码复核） |

计数：30 个项目，T1=8 / T2=17 / T3=5，元数据实测 18 个，缺口锚点 8 个。

## 1. 分级定义

| Tier | 含义 |
|---|---|
| T1 | absorb now: directly closes a P0/P1 TensorFVM gap; licence allows reading and re-implementing; upstream implementation is well tested |
| T2 | reference design: same paradigm, read for algorithm/validation design and cross-check numbers, not scheduled for porting |
| T3 | context only: non-FVM paradigm, closed source, or licence forbids code reuse; kept for the state of the art and benchmark provenance |

## 2. 全量注册表

| 项目 | id | Tier | 技术族 | 离散方式 | stars | licence | 最近 push | 缺口映射 |
|---|---|---|---|---|---|---|---|---|
| AMReX | `amrex` | T1 | block-structured AMR | block-structured FVM with embedded boundaries | 831 | NOASSERTION | 2026-11-11 | GAP-FVM-AMR-QUADTREE, GAP-FVM-MPI-DECOMP |
| Basilisk | `basilisk` | T1 | adaptive quadtree/octree | quadtree/octree cell-centred FVM with wavelet adaptivity | n/m | n/m | n/m | GAP-FVM-AMR-QUADTREE |
| Code_Saturne | `code-saturne` | T1 | general unstructured | unstructured co-located FVM, polyhedral cells | 312 | GPL-2.0 | 2026-11-10 | GAP-FVM-MPI-DECOMP, GAP-FVM-AMG-KRYLOV |
| hypre | `hypre` | T1 | linear algebra | n/a (algebraic multigrid) | 876 | NOASSERTION | 2026-11-11 | GAP-FVM-AMG-KRYLOV |
| OpenFOAM (ESI / OpenCFD distribution) | `openfoam-esi` | T1 | OpenFOAM lineage | unstructured collocated FVM | n/m | n/m | n/m | GAP-FVM-MRF-SLIDING-AMI, GAP-FVM-IMPLICIT-DUAL-TIME, GAP-FVM-LOWMACH-PRECOND |
| OpenFOAM (Foundation, OpenFOAM-dev) | `openfoam-foundation` | T1 | OpenFOAM lineage | unstructured collocated FVM, polyhedral cells | 2302 | NOASSERTION | 2026-11-11 | GAP-FVM-MRF-SLIDING-AMI, GAP-FVM-IMPLICIT-DUAL-TIME, GAP-FVM-MPI-DECOMP, GAP-FVM-AMR-QUADTREE |
| PETSc (incl. hypre bindings) | `petsc` | T1 | linear algebra | n/a (Krylov/AMG library) | 550 | NOASSERTION | 2026-11-10 | GAP-FVM-AMG-KRYLOV |
| SU2 | `su2` | T1 | compressible density-based | unstructured vertex-centred FVM | 1856 | NOASSERTION | 2026-11-10 | GAP-FVM-IMPLICIT-DUAL-TIME, GAP-FVM-LOWMACH-PRECOND, GAP-FVM-AMG-KRYLOV |
| CFL3D | `cfl3d` | T2 | compressible density-based | structured cell-centred FVM | 350 | Apache-2.0 | 2022-08-12 | GAP-FVM-IMPLICIT-DUAL-TIME |
| Chombo | `chombo` | T2 | block-structured AMR | block-structured AMR (FVM + FD) | n/m | n/m | n/m | GAP-FVM-AMG-KRYLOV |
| Clawpack (incl. AMRClaw, PyClaw) | `clawpack` | T2 | hyperbolic finite volume | finite-volume wave-propagation, logically rectangular AMR | 210 | BSD-3-Clause | 2026-11-05 | GAP-FVM-AMR-QUADTREE |
| FiPy | `fipy` | T2 | python prototyping | generic cell-centred FVM on unstructured meshes | 649 | NOASSERTION | 2026-11-02 | — |
| foam-extend | `foam-extend` | T2 | OpenFOAM lineage | unstructured collocated FVM | 66 | GPL-3.0 | 2022-09-16 | GAP-FVM-IMPLICIT-DUAL-TIME |
| Gerris | `gerris` | T2 | adaptive quadtree/octree | adaptive quadtree FVM, VOF | 1 | n/a |  | GAP-FVM-AMR-QUADTREE |
| Gmsh | `gmsh` | T2 | meshing | n/a (mesh generator) | 135 | NOASSERTION | 2026-11-11 | — |
| INCflo | `incflo` | T2 | block-structured AMR | incompressible FVM with embedded boundaries | closed | closed | n/m | GAP-FVM-AMR-QUADTREE |
| MFIX / MFiX-Exa | `mfix` | T2 | multiphase | unstructured FVM two-fluid + DEM | n/m | n/m | n/m | — |
| OpenQBMM | `openqbmm` | T2 | multiphase | OpenFOAM-based FVM population balance | 113 | NOASSERTION | 2026-09-17 | — |
| PeleC | `pelec` | T2 | block-structured AMR | block-structured compressible FVM + AMR | closed | closed | n/m | GAP-FVM-AMR-QUADTREE, GAP-FVM-COMBUSTION-SPECIES |
| PeleLMeX | `pelelmex` | T2 | block-structured AMR | low-Mach variable-density FVM + AMR | closed | closed | n/m | GAP-FVM-LOWMACH-PRECOND, GAP-FVM-COMBUSTION-SPECIES |
| preCICE | `precice` | T2 | coupling infrastructure | n/a (partitioned coupling library) | 992 | LGPL-3.0 | 2026-11-09 | GAP-FVM-MPI-DECOMP |
| PyFoam | `pyfoam` | T2 | tooling | n/a (case/run tooling) | n/m | n/m | n/m | — |
| REEF3D | `reef3d` | T2 | free-surface marine | structured Cartesian FVM, level-set free surface | 136 | GPL-3.0 | 2026-11-11 | — |
| solids4foam | `solids4foam` | T2 | OpenFOAM lineage | unstructured FVM solid mechanics | 118 | GPL-3.0 | 2026-11-10 | — |
| Truchas | `truchas` | T2 | multiphase | unstructured FVM multiphase + heat transfer | 11 | NOASSERTION | 2026-11-02 | GAP-FVM-RADIATION-P1 |
| Cart3D | `cart3d` | T3 | compressible density-based | Cartesian cut-cell FVM with AMR | n/m | n/m | n/m | GAP-FVM-AMR-QUADTREE |
| Ansys Fluent | `fluent` | T3 | commercial | unstructured FVM | closed | closed | n/m | — |
| FUN3D | `fun3d` | T3 | compressible density-based | unstructured vertex-centred FVM | n/m | n/m | n/m | GAP-FVM-IMPLICIT-DUAL-TIME |
| Nektar++ | `nektar` | T3 | high-order (non-FVM) | spectral/hp discontinuous Galerkin | n/m | n/m | n/m | — |
| PyFR | `pyfr` | T3 | high-order (non-FVM) | flux reconstruction | 585 | NOASSERTION | 2026-11-09 | — |

## 3. 分级理由与可吸收内容（Tier 1 全列）

### amrex — AMReX

- 来源：AMReX-Codes/amrex（metadata `github_api`）
- 分级理由：canonical blueprint for the AMR gap: level/flux regridding, refluxing and time subcycling contracts
- 可吸收：refluxing (flux register) conservation across coarse-fine interfaces + regridding/clustering strategy
- 许可风险：BSD-3-Clause (compatible)
- 缺口映射：GAP-FVM-AMR-QUADTREE, GAP-FVM-MPI-DECOMP

### basilisk — Basilisk

- 来源：http://basilisk.fr/（metadata `upstream_not_github`）
- 分级理由：the cheapest route to conservative adaptive VOF: wavelet error estimator + balanced quadtree refinement
- 可吸收：wavelet-based refinement criterion + VOF flux conservation across level boundaries
- 许可风险：GPL-3.0 (read-only)
- 缺口映射：GAP-FVM-AMR-QUADTREE

### code-saturne — Code_Saturne

- 来源：code-saturne/code_saturne（metadata `github_api`）
- 分级理由：industrial-grade unstructured FVM with published verification book; the model for our mesh-quality and linear-solver acceptance gates
- 可吸收：gradient/limiter treatment on highly non-orthogonal polyhedra
- 许可风险：GPL-2.0 (read-only)
- 缺口映射：GAP-FVM-MPI-DECOMP, GAP-FVM-AMG-KRYLOV

### hypre — hypre

- 来源：hypre-space/hypre（metadata `github_api`）
- 分级理由：BoomerAMG is the de-facto AMG for unstructured FVM; a binding is the minimal-change fix for our gap
- 可吸收：boomeramg parameter set validated on our Poisson cases
- 许可风险：Apache-2.0/MIT (compatible)
- 缺口映射：GAP-FVM-AMG-KRYLOV

### openfoam-esi — OpenFOAM (ESI / OpenCFD distribution)

- 来源：https://develop.openfoam.com/（metadata `upstream_not_github`）
- 分级理由：same paradigm; ESI branch carries the overset and compressible-VOF machinery TensorFVM still lacks at production quality
- 可吸收：overset interpolation weights + compressibleInterFoam pressure equation with phase-change source
- 许可风险：GPL-3.0 (read-only)
- 缺口映射：GAP-FVM-MRF-SLIDING-AMI, GAP-FVM-IMPLICIT-DUAL-TIME, GAP-FVM-LOWMACH-PRECOND

### openfoam-foundation — OpenFOAM (Foundation, OpenFOAM-dev)

- 来源：OpenFOAM/OpenFOAM-dev（metadata `github_api`）
- 分级理由：closest paradigm match to TensorFVM (collocated FVM, VOF, compressible) and the richest set of the missing P0/P1 mechanisms (AMI/MRF, dynamic AMR, implicit steady)
- 可吸收：MULES algebraic VOF limiter + AMI/GGI sliding interface topology + dynamicRefineFvMesh hanging-node flux correction
- 许可风险：GPL-3.0: read-only, re-implement from the papers, do not copy code
- 缺口映射：GAP-FVM-MRF-SLIDING-AMI, GAP-FVM-IMPLICIT-DUAL-TIME, GAP-FVM-MPI-DECOMP, GAP-FVM-AMR-QUADTREE

### petsc — PETSc (incl. hypre bindings)

- 来源：petsc/petsc（metadata `github_api`）
- 分级理由：do not re-implement AMG: the P1 linear-solver gap is closed faster by binding PETSc/hypre than by writing our own multigrid hierarchy
- 可吸收：GMRES/BiCGStab + BoomerAMG via a thin Python binding
- 许可风险：BSD-2-Clause (compatible)
- 缺口映射：GAP-FVM-AMG-KRYLOV

### su2 — SU2

- 来源：su2code/SU2（metadata `github_api`）
- 分级理由：the reference open implementation of the two P0 compressible gaps (implicit dual time, low-Mach preconditioning) with published verification cases
- 可吸收：Weiss-Smith/Turkel preconditioning matrix + LU-SGS linearisation + discrete adjoint boundary conditions
- 许可风险：LGPL-2.1 (read-only; re-implementation safest)
- 缺口映射：GAP-FVM-IMPLICIT-DUAL-TIME, GAP-FVM-LOWMACH-PRECOND, GAP-FVM-AMG-KRYLOV

## 4. 未实测元数据的项目（诚实清单）

| 项目 | 主仓 | 说明 |
|---|---|---|
| OpenFOAM (ESI / OpenCFD distribution) | https://develop.openfoam.com/ | ESI distributes via its own GitLab (develop.openfoam.com); no canonical GitHub repository exists |
| Basilisk | http://basilisk.fr/ | distributed by darcs from basilisk.fr; no GitHub repository, so metadata cannot be measured here |
| FUN3D | https://fun3d.larc.nasa.gov/ | released through the NASA software catalogue under a restricted agreement; no public repository |
| Cart3D | https://www.nas.nasa.gov/publications/software/docs/cart3d/ | restricted NASA release; no public repository |
| Chombo | https://commons.lbl.gov/display/chombo/Chombo+-+Software+for+Adaptive+Solutions+of+Partial+Differential+Equations | hosted by LBNL (SourceForge mirror); no verified GitHub repository |
| MFIX / MFiX-Exa | https://mfix.netl.doe.gov/ | NETL hosts MFIX on its own GitLab instance; no verified GitHub repository |
| PyFoam | https://openfoamwiki.net/index.php/Contrib/PyFoam | hosted on SourceForge; no verified GitHub repository |
| Nektar++ | https://www.nektar.info/ | official hosting is a GitLab instance; no verified GitHub repository |
| Ansys Fluent | — | closed source, no repository |

> 上述项目的能力标注同样只是标注；若纳入采纳路线，第一步是拉取源码做一次源读审计（本波未做）。

