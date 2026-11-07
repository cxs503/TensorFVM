# Incompressible FVM verification surface: backward-facing step, sudden expansion, natural-convection cavity

Independent verification of the TensorFVM finite-volume kernel on three classical incompressible benchmarks.  The solver path used here (`tensorfvm.inc_bench`) shares no mesh, projection or pressure code with the LBM verification path or with the lid-cavity DCT solver, so the evidence is independent rather than a re-reading of existing evidence.

## Criteria overview

| criterion | requirement | value | reference | error | verdict |
|---|---|---|---|---|---|
| bfs.reattachment.coarse | |L_r/S - 6.0| <= 0.2 (2-D benchmark) | 10.69646 | 6.00000 | 4.6965 | FAIL |
| bfs.friction_far | downstream |C_f| within 15 % of the analytic plane-channel value | -0.01009 | 0.03000 | 1.3362 | FAIL |
| bfs.friction_zero_crossing | C_f(x) changes sign exactly at the reattachment point | 10.69646 | 10.69646 | 0.0000 | PASS |
| expansion.area_ratio.coarse | stair-step geometry reproduces A_out/A_in = 4 within 0.2 % | 4.00000 | 4.00000 | 0.0000 | PASS |
| expansion.borda_carnot_sign | measured loss exceeds Borda-Carnot (viscous excess at Re_D = 100) | 0.79187 | 0.56250 | 0.2294 | PASS |
| expansion.mass_balance | inlet and outlet mass flux agree within 1e-4 | 0.00000 | 0.00000 | 0.0000 | PASS |
| cavity.nusselt.10000 | |Nu - Nu_ref| / Nu_ref <= 15% (target 3%) | 2.28819 | 2.24300 | 0.0201 | PASS |
| cavity.psi_max.10000 | psi_max within 15 % of de Vahl Davis | 5.15627 | 5.07100 | 0.0168 | PASS |
| cavity.energy_balance.10000 | hot-wall and cold-wall Nusselt numbers agree within 1 % | 0.00481 | 0.00000 | 0.0048 | PASS |
| cavity.nusselt.100000 | |Nu - Nu_ref| / Nu_ref <= 15% (target 3%) | 4.64118 | 4.51900 | 0.0270 | PASS |
| cavity.psi_max.100000 | psi_max within 15 % of de Vahl Davis | 10.03026 | 9.11100 | 0.1009 | PASS |
| cavity.energy_balance.100000 | hot-wall and cold-wall Nusselt numbers agree within 1 % | 0.00121 | 0.00000 | 0.0012 | PASS |
| cavity.nusselt.1e+06 | |Nu - Nu_ref| / Nu_ref <= 15% (target 3%) | 0.00000 | 8.80000 | 1.0000 | FAIL |
| cavity.psi_max.1e+06 | psi_max within 15 % of de Vahl Davis | diverged | 16.32000 | diverged | FAIL |
| cavity.energy_balance.1e+06 | hot-wall and cold-wall Nusselt numbers agree within 1 % | diverged | 0.00000 | diverged | FAIL |

## 1. Backward-facing step (Armaly geometry, expansion ratio 1.9423)

| grid | nx | ny | cells | ER (mask) | L_r/S | min C_f | C_f far | C_f analytic | rel. err. | iterations |
|---|---|---|---|---|---|---|---|---|---|---|
| coarse | 243 | 33 | 8019 | 1.94118 | 10.696 | -0.03105 | -0.01009 | 0.03000 | +133.62% | 1300 |

Richardson extrapolation of L_r/S: **n/a** (observed order n/a; three finite grid values required).

## 2. Planar sudden expansion (AR = 4, Re_D = 100)

| grid | cells | L_r/h_in | K | K_Borda-Carnot | K - K_BC | mass imbalance | iterations |
|---|---|---|---|---|---|---|---|
| coarse | 17408 | 2.104 | 0.7919 | 0.5625 | 0.2294 | 2.87e-07 | 1300 |

## 3. Differentially heated square cavity (Pr = 0.71)

| Ra | cells | Nu | Nu_ref | err | psi_max | psi_max_ref | err | u_max | err | energy balance |
|---|---|---|---|---|---|---|---|---|---|---|
| 1e+04 | 1024 | 2.288 | 2.243 | +2.01% | 5.156 | 5.071 | +1.68% | 16.32 | +0.87% | +0.481% |
| 1e+05 | 2304 | 4.641 | 4.519 | +2.70% | 10.030 | 9.111 | +10.09% | 43.71 | +25.86% | +0.121% |
| 1e+06 | 4096 | 0.000 | 8.800 | +100.00% | diverged | 16.320 | diverged | diverged | diverged | diverged |


## Artifact snapshot

The JSON/metrics/PNG evidence for exactly the numbers above is frozen under
`docs/inc-bench/artifacts/` (the pipeline writes to `artifacts/inc_bench/`, which
is outside this repository's sparse-checkout definition).

Result summary, stated without varnish:

* natural-convection cavity: Ra = 1e4 (32^2) mean Nu +2.0 % and Ra = 1e5 (48^2)
  +2.7 % versus de Vahl Davis (1983), energy budget closed to 0.5 % / 0.1 %;
  Ra = 1e6 **diverged** and is reported as a FAIL;
* backward-facing step: the coarse grid did not reach a steady state
  (max continuity defect 0.37, velocity change 1e-1 per iteration) and gives
  L_r/S = 10.7 against the 6.0 +/- 0.2 benchmark, with C_f still negative at
  x = 10 S; this is an **honest FAIL**, cause = under-converged SIMPLE on this
  case, not a modelled physics limit;
* sudden expansion: coarse grid K = 0.79 versus Borda-Carnot 0.5625 (the expected
  positive viscous excess at Re_D = 100), mass balance 2.9e-7, L_r/h_in = 2.10
  with the literature value WITHHELD; only one of the three grid levels completed.
