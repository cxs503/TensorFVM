# fvm_natural_convection — differentially heated square cavity (FVM baseline)

**Status: PASS on the mean-Nusselt criterion at all three Rayleigh numbers; the
streamfunction amplitude is only converged at Ra = 1e4.**

| item | value |
|---|---|
| geometry | unit square, hot west wall (T=1), cold east wall (T=0), adiabatic north/south |
| physics | steady Boussinesq, Pr = 0.71, `nu = sqrt(Pr/Ra)`, `alpha = nu/Pr`, `g*beta = 1` |
| grids | 32² / 64² / 96² (uniform) |
| criterion | `Nu` within 3 % of the contract value (2.243 / 4.519 / 8.800; de Vahl Davis 1983: 2.238 / 4.509 / 8.817) |
| solver | staggered SIMPLE, implicit upwind + frozen van Leer defect correction, multigrid pressure correction |
| code | `src/tensorfvm/staggered_simple.py`, `src/tensorfvm/incompressible_baselines.py` |
| driver | `scripts/run_fvm_incompressible_baselines.py --only cavity_n096_ra1e06` |

Results (programmatically generated in `results.json`, tables in
`docs/incompressible-baselines/RESULTS.md`):

| Ra | finest Nu | error vs contract | `psi/alpha` | error vs de Vahl Davis |
|---|---|---|---|---|
| 1e4 | 2.2062 (n=64) | 1.64 % | 5.12 | 0.98 % |
| 1e5 | 4.4363 (n=64) | 1.83 % | 10.33 | 13.4 % |
| 1e6 | 8.7352 (n=96) | 0.74 % | 18.62 | 14.1 % |

Honest notes
- `Nu` is second-order accurate and confirms the kernel; the deviation from the
  contract value is a discretisation effect that decreases with refinement
  (Ra=1e6: 20.9 % at n=32 -> 1.46 % at n=64 -> 0.74 % at n=96).
- `psi_max` is *not* converged at Ra >= 1e5: it is a cumulative integral of `u`
  from the wall, so it accumulates the near-wall velocity error that `Nu`
  (a wall derivative) does not see. Reported as a measured deficiency.
- The Nusselt post-processing uses a second-order one-sided derivative with the
  wall node at 0 and cell centres at h/2, 3h/2: `f'(0) = (-8/3 f0 + 3 f1 - 1/3 f2)/h`.
  An earlier wrong stencil (nodes h/2, 2h) biased the conduction limit by +8.3 %.
