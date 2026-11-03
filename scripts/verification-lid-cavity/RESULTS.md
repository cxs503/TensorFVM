# Lid-driven cavity (Ghia et al. 1982) verification — TensorFVM

Generated 2026-11-03 21:55 by `scripts/verify_lid_cavity_ghia.py`; all numbers below are read back from `summary.json`.

## 1. Case

Unit square cavity `[0,L]^2`, no-slip on `x=0`, `x=L`, `y=0`, lid `u=U` on `y=L`; `Re = U L / nu`, `U = L = rho = 1`. Benchmark: U. Ghia, K.N. Ghia, C.T. Shin, *J. Comput. Phys.* 48 (1982) 387–411, Tables I–II.

## 2. Method

Staggered MAC finite volume, conservative van Leer-limited face fluxes, explicit projection with a pure-Neumann pressure Poisson problem (DCT-II diagonalisation), marched to steady state (`cfl=0.15`, tolerance 1e-07).

## 3. Results

| Re | n | steps | converged | max abs err u | max abs err v | u_min (ref) | v_max (ref) | vortex centre (ref) | max div |
|---|---|---|---|---|---|---|---|---|---|
| 100 | 32 | 5750 | yes | 0.0069 | 0.0078 | -0.21087 (-0.21090) | 0.17607 (0.17527) | (0.609,0.766) (0.617,0.734) | 1.4e-14 |

## 4. Error judgement (is each key quantity within 3 %?)

* Re=100, n=32: pointwise max error 0.0069 U (0.69 % of U) → <3 %; u_min 0.016 % → <3 %; v_max 0.456 % → <3 %

## 5. Caveats

* `cfl=0.4` is unstable against the moving-lid shear layer (checkerboard flip in the lid-adjacent cells at n=64, Re=100: max|du| = 1.04/step, |u|max = 1.27 > U); `cfl=0.15` is the shipped default. The steady state does not depend on it, only the cost.
* Re = 100 on 64^2 reached max|du| = 0.0045 U in 9500 steps (steady-state rate 6e-8, from the
  run log); that row is not in `summary.json` because the production ladder was still
  running when this note was written, so it is quoted from the log, not from JSON.
* Re = 1000 needs far more time steps than the 12 000-step production cap used here; those rows are honest "not converged to tolerance" runs and their error is indicative only.
* Wall vorticity is first-order (one-sided replication) and is not used for any metric.
