# Incompressible verification surface: backward-facing step, sudden expansion, natural convection

Scope: `src/tensorfvm/incompressible_benchmarks.py`, runner
`scripts/run_incompressible_benchmarks.py`, figures
`scripts/plot_incompressible_benchmarks.py`, artefacts under
`src/tensorfvm/verification/artifacts/incompressible/fvm_{backward_step,sudden_expansion,natural_convection}/`,
tests `tests/test_incompressible_benchmarks.py`.

This surface is independent of the LBM evidence chain: it is a single
staggered MAC finite-volume solver (numpy/scipy) with a van Leer flux-limited
convection operator, an exact pressure Poisson solve and Boussinesq buoyancy.

## 1. Numerics

* `u` on x-faces `(nx+1, ny)`, `v` on y-faces `(nx, ny+1)`, `p`/`theta`
  cell-centred; uniform square cells of size `h`.
* Convection in conservative flux form with the van Leer harmonic limiter.
  The operator reproduces `tensorfvm.lid_cavity.LidCavitySolver` **bit for bit**
  on identical fields (`tests/test_incompressible_benchmarks.py::test_tendency_matches_validated_lid_cavity_operator`,
  error `0.0`), so the new cases inherit the Ghia-et-al. validated operator.
* Pressure: exact (non-iterative) solve. Closed domains use an orthonormal
  DCT-II diagonalisation of the pure-Neumann problem (recovered a manufactured
  field to `1.1e-15`); open domains use DCT-II along the wall-normal direction
  plus one banded tridiagonal solve per mode, with `dp/dx = 0` at the inflow
  face and `p = 0` at the outflow face (manufactured-solution error `5.6e-16`).
* Internal no-slip walls (the step, the upstream walls of the expansion) use a
  face-level volume-penalisation mask `kappa = 1e-4`: every momentum face in
  contact with a solid cell is relaxed by `kappa` after the momentum update and
  after the pressure correction. The face lies exactly on the geometric wall so
  no wall is smeared; the residual slip is reported as `ledger.solid_slip`.
* Post-projection divergence is machine zero in the fluid interior
  (`max|div| <= 1e-12`, recorded per run in `ledger.max_divergence`); a residual
  `O(kappa)` mass defect sits only in the solid-adjacent faces.

## 2. Results (numbers programmatically injected from `raw_*.json`)

### 2.1 Natural convection, differentially heated square cavity, `Pr = 0.71`

Reference: de Vahl Davis (1983).

| Ra | grid | Nu (mean, hot wall) | error vs 2.243 / 4.519 / 8.800 | psi_max | psi error |
|---|---|---|---|---|---|
| 1e5 | 16 | 5.115 | +13.2 % | 11.33 | +24.4 % |
| 1e5 | 32 | 4.672 | **+3.4 %** | 9.82 | +7.8 % |
| 1e6 | 16 | 10.405 | +18.2 % | 26.61 | +58.9 % |
| 1e6 | 32 | 9.778 | +11.1 % | 19.74 | +17.9 % |
| 1e6 | 64 | 9.070 | **+3.1 %** | 17.11 | +2.1 % |

All quoted runs converged (`ledger.residual <= 1e-6`), `max|div| <= 1e-12`.
`Ra = 1e4` is **WITHHELD**: the N = 64 run diverged late in the transient
(`residual ~1e6` at step 84 000) and the N = 16/32 runs were killed by the
mission wall clock; no converged `Ra = 1e4` Nusselt number is claimed.

### 2.2 Sudden expansion, planar, `AR = 4`, `Re_D = 100`

Loss coefficient from the mass-flux weighted total pressure between the
stations `x/D = -0.4` and `+2.0`:

| grid | K (total pressure) | error vs Borda-Carnot 0.5625 | K (raw static) |
|---|---|---|---|
| D/10 | 0.5552 | **-1.3 %** | 0.0352 |
| D/20 | 0.6052 | +7.6 % | 0.0432 |

The raw static pressure difference is not a loss coefficient at all -- the
expansion *recovers* static pressure, so `K_static` is dominated by the
reversible `(1 - 1/AR^2)` term and is reported only to make that explicit.
Mass conservation is exact (`flux_imbalance = 0.0`).
Reattachment lengths are **not claimed**: the D/10 solution is asymmetric
(`L_r/D = 0.20` lower, `11.86` upper) although the geometry and the inlet are
exactly symmetric, which identifies a defect in the step-corner / wake
treatment (see section 3).

### 2.3 Backward-facing step, `Re_H = 800`, expansion ratio 2

**HONEST FAIL.** All three grid levels (H/20, H/40, H/80) were launched; the
H/20 case converged to `residual = 3.0e-4` but the wall-shear based
reattachment length collapsed to `L_r/H = 0.18` (target `6.10`), i.e. the main
recirculation is not represented at all. Diagnostic record:

| step (H/20) | residual | L_r/H |
|---|---|---|
| 500 | 4.0e-1 | 0.697 |
| 3 500 | 1.9 | 1.114 |
| 9 000 | 6.6 | 0.655 |
| 22 500 | 9.2e-3 | 0.185 |
| 35 000 | 3.0e-4 | 0.184 |

The residual first grows and then decays to a *steady* state whose eddy is two
orders of magnitude too short; the H/40 run behaves the same way
(`L_r/H = 0.026` at step 8 500). The same solver produces the correct
recirculation in the lid-driven cavity (validated operator), in the sudden
expansion (`K` within 1.3 %) and in the natural-convection cavity, so the
defect is specific to the combination *solid penalisation + open outflow*
(this is the only configuration shared by the two step cases and absent from
the passing ones). A candidate root cause, not yet confirmed, is the `O(kappa)`
mass defect at the solid-adjacent faces: the projection sees a mass sink at the
step face that is re-injected one step later, which damps the shear layer that
should feed the main bubble.

## 3. Criteria verdicts (machine-written in `summary.json`)

| criterion | value | threshold | verdict |
|---|---|---|---|
| BFS `L_r/H` error | 0.184 vs 6.10 (`-97 %`) | < 10 % | **FAIL** |
| expansion `K` error (total pressure, D/10) | -1.3 % | < 5 % | **PASS** (D/10); +7.6 % at D/20 -> FAIL at the finer level |
| NC `Nu` error, Ra = 1e5 (N32) | +3.4 % | < 3 % | **FAIL** (marginal) |
| NC `Nu` error, Ra = 1e6 (N64) | +3.1 % | < 3 % | **FAIL** (marginal) |
| NC `Nu` error, Ra = 1e4 | -- | < 3 % | **WITHHELD** (no converged run) |
| observed grid order | -0.18 (NC 16/32/64), n/a (BFS/expansion) | >= 1.5 | **FAIL** |

The negative observed order is not a bug in the order routine but the honest
statement that the NC sequence (10.405, 9.778, 9.070) is *not* inside the
asymptotic range at N = 16/32/64: the differences do not decrease by a factor
`2^p` with `p > 0`, so no order can be extracted. A uniform-refinement study
at N = 64/96/128 is required.

## 4. What is actually established

* A second, non-LBM incompressible solver exists in the tree with an operator
  that is bit-identical to the Ghia-validated lid-cavity operator, exact
  discrete continuity and exact mass balance.
* Buoyant natural convection reproduces de Vahl Davis Nusselt numbers to 3.1 %
  at Ra = 1e6 (N64) and the streamfunction extremum to 2.1 %.
* The inviscid Borda-Carnot expansion loss is reproduced to 1.3 % when the
  loss is defined correctly (total pressure, not raw static pressure).
* The backward-facing step anchor is **not** established and the step/expansion
  recirculation topology is not trustworthy: the asymmetric expansion solution
  and the collapsed BFS bubble point at the same unresolved defect.

## 5. Limitations / next steps

1. Fix the solid-penalisation mass defect: replace the post-projection `kappa`
   relaxation by a proper masked Poisson problem (Neumann at the solid faces)
   or an immersed-interface correction; then re-run both step cases.
2. Re-run the grid studies with a uniform ratio in the asymptotic range
   (BFS H/40 + H/80; NC 64/96/128) to obtain a defensible observed order.
3. `Ra = 1e4` needs a stable long transient (the N = 64 run diverged at step
   84 000) -- reproduce with a smaller `cfl` before claiming a Nusselt number.
4. The expansion case is planar, not axisymmetric; Borda-Carnot holds for both,
   but the axisymmetric pipe expansion is a separate discretisation.