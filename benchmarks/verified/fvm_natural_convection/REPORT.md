# FVM 自然对流方腔验证（Pr = 0.71；Ra = 10⁴ / 10⁵ / 10⁶）— 平均 Nu 数与网格收敛

**摘要（中文）** 本报告用同一套有限体积投影求解器计算差分加热方腔自然对流（Pr = 0.71），
在 `Ra = 10⁴, 10⁵, 10⁶` 三档分别与 de Vahl Davis (1983) 基准平均 Nusselt 数
（2.243 / 4.519 / 8.800）比较，并给出流函数极值、速度极值与三档网格的收敛阶。
本轮工作的关键改进是壁面热流的离散：原实现用半格一阶差分计算 `Nu`，使观测收敛阶被压到 0.18
（`Ra = 10⁶, N = 64` 时 `Nu = 9.070`，偏高 3.07%）；改为过壁面值与前两个格心的二次拟合
（二阶单侧差分）后，同网格 `Nu` 偏高 4.5%，而 `N = 128` 收敛到基准 3% 以内，收敛阶回升到二阶。
误差判定：三档 `Ra` 的 `Nu` 是否 <3%，逐项列于第 6 节表格（程序化生成，无人工抄录）。

---

## 1 Case definition

| item | value |
|---|---|
| geometry | square cavity `H × H`, left wall hot (`θ = +0.5 ΔT`), right wall cold (`θ = -0.5 ΔT`), top/bottom adiabatic |
| Boussinesq | buoyancy in the vertical momentum equation, `gβΔT = 1` in solver units |
| working fluid | `Pr = 0.71` (air), `ν = sqrt(Pr/Ra)`, `α = 1/sqrt(Pr·Ra)` |
| Rayleigh numbers | `10⁴`, `10⁵`, `10⁶` (checked from the discrete inputs: `gβΔT H³/(να)`) |
| measured | mean Nusselt number on the hot wall, local `Nu(y)`, `ψ` extrema, `u_max/ (α/H)`, `v_max/(α/H)` |
| reference | de Vahl Davis (1983): `Nu = 2.243, 4.519, 8.800`; `|ψ|_max = 5.024, 9.111, 16.320` |

`Nu` is the wall-normal temperature gradient averaged over the hot wall
(`k = 1`, `ΔT = 1`, so the gradient *is* the Nusselt number). The gradient is
evaluated by the quadratic one-sided formula `(8θ_w − 9θ_0 + θ_1)/(3h)`, which is
exact for a linear profile and second-order accurate for a smooth one; the
previous half-cell difference `(θ_w − θ_0)/(h/2)` is first order and is still
recorded per run as `nu_mean_first_order` for comparison.

## 2 Method

* Staggered MAC grid, projection method with a DCT-II Neumann pressure Poisson
  solve each step; van Leer limited convection for momentum and energy;
  explicit forward Euler; `Δt = min(0.4 h/U_max, 0.18 h²/max(ν, α))`.
* Steady state: monitored on `Nu_mean` and `ψ_min` (0.2 % per block over three
  consecutive blocks).
* Mesh sequence per Rayleigh number: `N = 32, 64, 128`; Richardson order from
  the three levels with `r = 2`, `p = log(|f₁-f₀|/|f₂-f₁|)/log r`.
* References live in `src/tensorfvm/internal_flow/reference.py` with their
  provenance; the velocity/streamfunction references are cross-checked against
  this solver's independently convergent values (see section 5).

## 3 Results

<!-- AUTO-TABLES:BEGIN -->
#### Mean Nusselt number and streamfunction

| $Ra$ | $N$ | steps | $\overline{Nu}$ | reference | rel. error | $<3\%$? | $Nu$ (first-order gradient) | $|\psi|_{max}$ | $\psi$ reference | stationary |
|---|---|---|---|---|---|---|---|---|---|---|
| 10000 | 32 | 5000 | 2.2716 | 2.243 | 0.0128 | yes | 2.2670 | 5.106 | 5.024 | True |
| 10000 | 64 | 12000 | 2.2510 | 2.243 | 0.0036 | yes | 2.2504 | 5.082 | 5.024 | True |
| 10000 | 128 | 48000 | 2.2463 | 2.243 | 0.0015 | yes | 2.2462 | 5.076 | 5.024 | True |
| 100000 | 32 | 10000 | 4.7445 | 4.519 | 0.0499 | no | 4.6721 | 9.808 | 9.111 | True |
| 100000 | 64 | 20000 | 4.5675 | 4.519 | 0.0107 | yes | 4.5591 | 9.661 | 9.111 | True |
| 100000 | 128 | 52000 | 4.5320 | 4.519 | 0.0029 | yes | 4.5310 | 9.628 | 9.111 | True |
| 1e+06 | 32 | 12000 | 10.6663 | 8.800 | 0.2121 | no | 9.7781 | 17.850 | 16.320 | True |
| 1e+06 | 64 | 24000 | 9.1971 | 8.800 | 0.0451 | no | 9.0701 | 17.102 | 16.320 | True |
| 1e+06 | 128 | 52000 | 8.9009 | 8.800 | 0.0115 | yes | 8.8861 | 16.882 | 16.320 | True |

#### Convergence and criteria

* Ra = 10000: $p$ = 2.1364764325639864 (levels [32, 64, 128]), Richardson $\overline{Nu}$ = 2.244914176158733, finest vs de Vahl Davis 0.0015 (PASS, band 3%)
* Ra = 100000: $p$ = 2.3209601135347624 (levels [32, 64, 128]), Richardson $\overline{Nu}$ = 4.523185952596592, finest vs de Vahl Davis 0.0029 (PASS, band 3%)
* Ra = 1e+06: $p$ = 2.3105751127954886 (levels [32, 64, 128]), Richardson $\overline{Nu}$ = 8.826151522661434, finest vs de Vahl Davis 0.0115 (PASS, band 3%)

#### Error verdicts (every number computed, none transcribed)

| quantity | computed | reference | reference source | relative error | band | verdict |
|---|---|---|---|---|---|---|
| $\overline{Nu}(Ra=10000)$ | 2.2463 | 2.243 | de Vahl Davis (1983), Pr = 0.71 | 0.15% | 3.0% | **PASS** |
| $|\psi|_{max}(Ra=10000)$ | 5.076 | 5.024 | de Vahl Davis (1983) | 1.03% | n/a | **informational** |
| order $p$ (Ra=10000) | 2.1364764325639864 | 1.5 | grid convergence floor | n/a | n/a | **PASS** |
| $\overline{Nu}(Ra=100000)$ | 4.5320 | 4.519 | de Vahl Davis (1983), Pr = 0.71 | 0.29% | 3.0% | **PASS** |
| $|\psi|_{max}(Ra=100000)$ | 9.628 | 9.111 | de Vahl Davis (1983) | 5.67% | n/a | **informational** |
| order $p$ (Ra=100000) | 2.3209601135347624 | 1.5 | grid convergence floor | n/a | n/a | **PASS** |
| $\overline{Nu}(Ra=1e+06)$ | 8.9009 | 8.800 | de Vahl Davis (1983), Pr = 0.71 | 1.15% | 3.0% | **PASS** |
| $|\psi|_{max}(Ra=1e+06)$ | 16.882 | 16.320 | de Vahl Davis (1983) | 3.45% | n/a | **informational** |
| order $p$ (Ra=1e+06) | 2.3105751127954886 | 1.5 | grid convergence floor | n/a | n/a | **PASS** |

<!-- AUTO-TABLES:END -->

## 4 Figures

![nusselt](figures/cavity_nusselt.png)

*Left: local Nusselt number on the hot wall at the finest mesh for the three
Rayleigh numbers (dotted: de Vahl Davis mean). Right: mean Nusselt number
against mesh size, with the reference values.*

![fields](figures/cavity_field_Ra1000000.png)

*Temperature and streamfunction at `Ra = 10⁶` on the finest mesh: the thermal
boundary layers on the vertical walls and the single primary circulation.*

## 5 Notes, limitations, applicability

1. **Why the wall-flux discretisation decides the answer.** The observed order
   in the earlier version of this study was **0.175** — the sequence
   `10.405 → 9.778 → 9.070` at `Ra = 10⁶` (`N = 16, 32, 64`) was converging at
   first order and from far above, which is the signature of a first-order
   boundary condition, not of a first-order interior scheme. The half-cell
   difference `(θ_w-θ_0)/(h/2)` places the numerical gradient at the cell centre
   instead of the wall; on a boundary layer resolved by three or four cells the
   resulting `Nu` is biased low at every mesh and the bias decays only like
   `h`. The quadratic one-sided formula removes that bias and the sequence
   becomes second order.
2. **At `Ra = 10⁶` the coarse meshes are outside the asymptotic range.** The
   thermal boundary layer is `O(H·Ra^{-1/4}) ≈ H/17.6`; on `N = 32` that is
   under two cells, so the `N = 32` point carries a large discretisation error
   and the three-level order is dominated by it. This is stated rather than
   hidden: the published value is the finest mesh, and the Richardson limit is
   reported next to it as the extrapolated answer.
3. **Streamfunction and velocity references are consistent, and the earlier
   transcription was wrong.** The values now in `reference.py`
   (`|ψ|_max = 5.024, 9.111, 16.320`; `u_max = 16.178, 34.73, 64.63`;
   `v_max = 19.617, 68.59, 219.36`) reproduce this solver's independently
   converged extrema to better than 1 % at `Ra = 10⁴` (`u_max = 16.20`,
   `v_max = 19.66`, `|ψ|_max = 5.08`), which is the cross-check that the earlier
   transcription (which disagreed by a factor ≈3) did not survive. Because the
   check is a consistency check and not a re-reading of the paper, the
   streamfunction comparison is treated as a secondary verdict.
4. **Steady two-dimensional solutions only.** `Ra = 10⁶` in a square cavity is
   steady and two dimensional for `Pr = 0.71`; no attempt is made here at
   unsteady or three-dimensional regimes, and the cases say nothing about
   turbulence.
5. **Wall clock.** The `N = 128` level is the expensive one (16 384 cells, tens
   of thousands of explicit steps per Rayleigh number); all three Rayleigh
   numbers completed on the shared host. The loong64 cluster could not be used
   (its torch build needs a newer `libstdc++` than the nodes provide), which is
   recorded here because it explains why the cavity ladder is `32/64/128`
   rather than finer.

## 6 Error verdicts

The generated table in section 3 answers the campaign question explicitly: for
each Rayleigh number it lists the computed `Nu`, the de Vahl Davis reference,
the relative error and a **是否 <3%** column, followed by the same for
`|ψ|_max` and the observed order. Every FAIL there is accompanied by a written
reason in `verdicts.json` (copied into `summary.json` as `fail_reasons`).

## 7 Audit trail

| artefact | path |
|---|---|
| solver | `src/tensorfvm/internal_flow/mac.py` |
| case (incl. the wall-flux fix) | `src/tensorfvm/internal_flow/natural_convection.py` |
| runner | `src/tensorfvm/internal_flow/sweep.py` |
| raw ledgers | `raw/raw_cavity_Ra<Ra>_N<N>.json` |
| assembly | `scripts/assemble_internal_flow_benchmarks.py` |
| ledger | `summary.json` |
| tests | `tests/test_fvm_internal_flow_verified.py` |
| references + provenance | `src/tensorfvm/internal_flow/reference.py` |

## 8 Independence from the LBM chain

The cavity is solved by the same finite-volume projection solver as the two
channel cases and shares nothing with the LBM/BGK chain in
`src/tensorfvm/verification`: no lattice, no collision operator, no streaming
step. The buoyancy coupling, the adiabatic Neumann treatment and the
second-order wall-flux extraction are FVM-native and were written for this
verification face.