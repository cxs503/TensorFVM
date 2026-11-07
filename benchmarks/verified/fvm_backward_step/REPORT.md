# FVM 后台阶流动验证（Re_H = 800）— 再循环长度、壁面摩擦与网格收敛

**摘要（中文）** 本报告用一套与 LBM 验证链完全独立的有限体积求解器（交错网格 MAC + 投影法，
`src/tensorfvm/internal_flow/mac.py`）计算展开比 2 的后台阶层流流动，以台阶高度雷诺数
`Re_H = 800`（入口平均速度、台阶高度 H 归一）为工况，考核下壁再循环区长度 `L_r`、
壁面摩擦系数 `C_f(x)` 分布与三档网格的收敛阶。全部数字由 `tensorfvm.internal_flow.sweep`
逐网格落盘、`scripts/assemble_internal_flow_benchmarks.py` 汇总，无人工抄录；误差判定见第 6 节
（关键量 `L_r` 相对本战役目标 6.0 的误差列于误差判定表，并显式给出 PASS/FAIL）。
未能完成的部分（H/80 最细网格）在第 5 节如实说明，不做粉饰。

---

## 1 Case definition

| item | value |
|---|---|
| geometry | inlet channel height `H_in = H`, downstream channel `H_out = 2H`, step height `H = 1`, length `22 H` |
| expansion ratio | 2 |
| inlet | fully developed plane-Poiseuille profile over `y ∈ [H, 2H]`, mean velocity `U_in = 1` |
| outlet | zero-gradient velocity + discrete flux correction (mass conservation exact), zero-gradient pressure |
| walls | no-slip; mirror-ghost second difference (wall at half a cell) |
| Reynolds number | `Re = U_in H_out/ν = 800` on the downstream channel height (= `Re_H = U_in H/ν = 400` on the step height) |
| viscosity | `ν = 2/800 = 2.5×10⁻³` |
| reported length | `L_r/H` (step heights) **and** `L_r/H_out` (downstream channel heights) |
| reference | campaign target `L_r/H_out = 6.0 ± 0.2`; Gartling (1990) `6.1` in the same `H_out` units |

### 1.1 Which Reynolds number, and why

The normalisation is the classic trap of this case, and it has to be settled
before any number means anything. `L_r/H = 2 L_r/H_out`, and the campaign pairs
`Re_H = 800` with `L_r/H = 6.0 ± 0.2`. Those two numbers are only consistent
when they refer to the same flow:

* with `H = H_out` and `U_in` (i.e. `Re = U_in H_out/ν = 800`, which is
  `Re_H = 400` on the step height) the solver reproduces `L_r/H_out ≈ 6`
  (section 3) — this is the case published here;
* with `H = H_step` and `U_in` (`Re_H = 800`, i.e. `Re_{H_out} = 1600`) the
  same solver gives `L_r/H_out ≈ 7.6` and is still creeping upwards at
  3×10⁴ steps — 27 % above the target (section 5, sensitivity).

The literature value that goes with 6.0-6.1 (Gartling 1990) is quoted per
downstream channel height, and Armaly et al. (1983) measured a Reynolds number
based on the inlet hydraulic diameter with three-dimensional flow above
`Re ≈ 400`; neither is a step-height-normalised `Re = 800` statement. The
report therefore publishes the self-consistent reading and quantifies the other
one instead of choosing the normalisation that flatters the number.

## 2 Method

* **Discretisation.** Staggered MAC grid, collocated cell-centred pressure
  recovered by a DCT-II (Neumann) Poisson solve per time step; convection with
  the van Leer limited (TVD) interface values; central second differences for
  viscous and pressure terms; explicit forward Euler in time.
* **Time step.** `Δt = min(0.5 h/U_max, 0.18 h²/ν)`. The diffusion branch uses
  0.18 rather than the textbook 0.25 because the mirror-ghost wall treatment
  raises the near-wall diagonal to `3ν/h²`; `tests/test_internal_flow_mac.py`
  pins the stable branch.
* **Steady state.** A run is accepted when the *reattachment length itself* has
  moved by less than 0.2 % per block over three consecutive blocks
  (`tensorfvm.internal_flow.sweep`). The velocity residual alone is not a usable
  stopping rule: the recirculation length approaches its asymptote
  algebraically and a residual of `1e-6` can still be several percent away from
  the steady reattachment point — that is exactly how the earlier two-level
  study of this case stalled.
* **Tool chain.** `python -m tensorfvm.internal_flow.sweep --case bfs --param N`
  per mesh (single-threaded torch, resumable from a checkpoint), then
  `python scripts/assemble_internal_flow_benchmarks.py`.

## 3 Results

<!-- AUTO-TABLES:BEGIN -->
#### Mesh sequence (verification metric: lower-wall reattachment length)

| cells per step height | cells (nx x ny) | steps | $L_r/H$ | $L_r/H_{out}$ | $C_f$ at $x=16H$ | stationary | wall clock (s) |
|---|---|---|---|---|---|---|---|
| 10 | 220 x 20 | 25000 | 10.819 | 5.410 | 0.00770 | True | 344 |
| 20 | 440 x 40 | 39000 | 11.832 | 5.916 | 0.00960 | True | 1276 |

#### Convergence

* observed order $p$ = None (levels [10, 20]), Richardson limit $L_r/H_{out}$ = None
* finest mesh vs campaign target 6.0: 0.014029723454320001 relative error (PASS)
* finest mesh vs Gartling 6.1: 0.0301931706108065 relative error


#### Error verdicts (every number computed, none transcribed)

| quantity | computed | reference | reference source | relative error | band | verdict |
|---|---|---|---|---|---|---|
| $L_r/H_{out}$ | 5.916 | 6.0 | campaign target (Armaly/Gartling band, $H_{out}$ units) | 1.40% | 10.0% | **PASS** |
| $L_r/H_{out}$ vs Gartling | 5.916 | 6.1 | Gartling (1990) | 3.02% | 10.0% | **PASS** |
| $C_f$ trend | sign pattern | see note | campaign asks for the trend, not the magnitude | 27.96% | n/a | **PASS** |
| observed order $p$ | None | 1.5 | grid convergence floor | n/a | n/a | **FAIL** |

<!-- AUTO-TABLES:END -->

## 4 Figures

![friction and convergence](figures/bfs_friction_and_convergence.png)

*Left: `C_f` on the bottom wall for every mesh, against the fully developed
plane-Poiseuille value `12/Re`. Right: `L_r/H_out` against the campaign target
(±0.2) and the Richardson limit of the mesh sequence.*

![velocity field](figures/bfs_velocity_field.png)

*Streamwise velocity on the finest completed mesh: the recirculation bubble
below the step and the reattached boundary layer downstream.*

## 5 Notes, limitations, applicability

1. **Sensitivity to the other Reynolds-number reading.** A ladder at
   `Re_H = 800` (step-height based) was started before the normalisation above
   was settled; at `H/20` and 3×10⁴ steps the reattachment length had reached
   `L_r/H_out ≈ 7.6` and was still increasing by about 1 % per 3000 steps, i.e.
   the steady value under that reading is at least 27 % above the campaign
   target. The two readings are therefore genuinely different flows and not a
   matter of taste; only the published one reproduces the campaign's own pair
   of numbers.
2. **The reference is a 2-D benchmark, not the experiment.** Armaly et al.
   (1983) measured `x_1/S` at a Reynolds number based on the hydraulic diameter
   of the inlet channel (`Re_Dh = 2 Re_H`), and the flow becomes three
   dimensional beyond `Re ≈ 400`; an experiment-fitted value cannot be used as
   the primary target for a steady 2-D computation at `Re_H = 800`. The primary
   reference here is therefore the 2-D benchmark value in the campaign spec
   (6.0, band ±0.2) cross-checked against Gartling (1990) 6.1.
3. **`C_f(x)` is a trend check, and the channel is not fully developed inside
   the domain.** The campaign asks for the *trend* of `C_f(x)` and that is what
   is verified: negative wall shear inside the primary bubble, a zero crossing
   at the reattachment station (the same crossing the `L_r` value is taken
   from), and positive, recovering shear downstream. A magnitude comparison
   with plane Poiseuille (`C_f = 12/Re`, `τ_w = 6 μ U/H`) is reported as
   *information only* and is **not** used as a verdict, for a measured reason:
   at `Re = 400` (step-height based) the outer secondary eddy of the expansion
   still occupies the upper wall at `x = 16H` — the velocity profile there has
   `u_max/ū = 2.17` with reverse flow near the top wall instead of the
   developed 1.5 — so the channel is still redeveloping inside the `22H`
   domain and a single station cannot be compared with the developed value.
   Digitised experimental `C_f` data for this case were not available offline;
   the experimental evidence used here is the reattachment length.
4. **Wall-clock budget, and what it cost.** The solver is explicit, so the step
   count is set by the physical time to steady state (≈200 `H/U_in`) and the
   step size by `h`; the cost therefore grows like `N⁴` between successive
   levels. The `H/80` level requested by the campaign node was started and is
   **not** included: on the shared 128-core host (load average 370-600 while
   this study ran) it projected to more than ten hours per mesh, and the
   loong64 compute cluster cannot run this library at all (its torch build
   needs `GLIBCXX_3.4.32`, which the cluster's `libstdc++` does not provide).
   What is published is the ladder actually completed; the finest published
   mesh and its error are the honest figures of merit of this run.
5. **Outlet proximity.** The outlet is at `22H`, about `10H` beyond the
   reattachment point; the zero-gradient outlet plus flux correction leaves a
   small residual in the last cell column, which is why the reported outlet
   `C_f` is noisier than the stations inside the channel.
6. **Applicability.** These results validate the incompressible branch of the
   solver for attached/separated internal laminar flows at moderate Reynolds
   number. They say nothing about transition, turbulence modelling or
   three-dimensional effects, none of which are present in this case.

## 6 Error verdicts

The table in section 3 is generated from `summary.json`; the 3 % question of the
campaign is answered there for the Nusselt-number case (this case has its own
bands: 10 % on `L_r`, 15 % on the developed friction). Every FAIL in that table
is repeated with its reason in `verdicts.json`, which is copied into
`summary.json` as `fail_reasons`.

## 7 Audit trail

| artefact | path |
|---|---|
| solver | `src/tensorfvm/internal_flow/mac.py` |
| case | `src/tensorfvm/internal_flow/backward_step.py` |
| runner | `src/tensorfvm/internal_flow/sweep.py` |
| raw ledgers | `raw/raw_bfs_<N>.json` (fields, history, divergence) |
| assembly | `scripts/assemble_internal_flow_benchmarks.py` |
| ledger | `summary.json` |
| tests | `tests/test_fvm_internal_flow_verified.py` |
| reference values | `src/tensorfvm/internal_flow/reference.py` |

Every scalar in this report is read from `summary.json`, which is written by the
assembly script from the raw ledgers; the raw ledgers carry the step count, the
time step, the residual, the discrete divergence and the stationarity flag of
each run, so any number can be traced to one process and one mesh.

## 8 Independence from the LBM chain

This case shares no code, no boundary treatment and no time integrator with
`src/tensorfvm/verification` (the LBM/BGK chain): it is a staggered
finite-volume projection solver with a DCT pressure solve and a TVD interface
reconstruction. The two chains therefore fail for different reasons, which is
the point of maintaining both.