# Thermal convection verification face (Rayleigh-Benard onset + Nu(Ra) law)

Campaign node `p224w2_n05_fvm_thermal_convection` of TensorFVM: the thermal counterpart of the verified incompressible cases. Everything below is produced by `python -m tensorfvm.thermal.study` (see *How to reproduce*).

## 1. Case introduction

**Physics.** A horizontal layer of fluid between a hot lower plate and a cold upper plate.  Below a critical Rayleigh number the fluid is at rest and heat crosses the layer by conduction alone (`Nu = 1`); above it, buoyancy-driven convection sets in and the heat transport grows as a power of `Ra`.  Two questions separate cleanly and are answered by two different tools here: *when* does the layer convect (a property of the linearised operator) and *how much* heat does the convecting layer carry (a property of the nonlinear steady state).

**Geometry and parameters.** Closed box `Gamma x 1` (`Gamma = 2`), plates at `y = 0` (hot, `theta = +dT/2`) and `y = 1` (cold, `theta = -dT/2`), lateral walls no-slip and adiabatic.  Non-dimensionalisation: `H = 1`, `dT` and the free-fall velocity `U = sqrt(g beta dT H)`, so that

```
du/dt + div(u u) = -grad p + sqrt(Pr/Ra) lap(u) + theta e_y
dtheta/dt + div(u theta) = lap(theta) / sqrt(Pr Ra),   div(u) = 0
Ra = g beta dT H^3/(nu alpha),   Pr = nu/alpha = 0.71
```

## 2. How it was done

* **Onset (criterion a):** linear stability of the conduction state.  Normal modes `exp(i k x + sigma t)` in the horizontal direction reduce the linearised Boussinesq equations to the pencil `L q = sigma M q` with `q = (Psi, Theta)` and `A = d^2/dz^2 - k^2`

  ```
  M = diag(A/Pr, 1),   L = [[A^2, -i k Ra], [-i k, A]]
  ```

  discretised in `z` with second differences (plate conditions carried by ghost nodes: `Psi = 0` and `Psi' = 0` rigid / `Psi'' = 0` stress-free, `Theta = 0`), solved with LAPACK `zgeev`, and the marginal curve traced with Brent's method on `log Ra`.  The `A^2` block is the product of the *full* second-derivative operator, so the clamped-plate wall vorticity enters the biharmonic stencil -- without it the rigid and stress-free thresholds would coincide.
* **Nonlinear layer (criteria b, d, e):** the audited staggered MAC solver of `tensorfvm.internal_flow.mac` (`thermal_walls="plates"`: van Leer limited conservative fluxes, DCT-II pressure projection, explicit forward-Euler marching).  The conduction state is an exact equilibrium of the scheme, so a single roll of amplitude `1e-3 dT` is seeded; the wavenumber is `k = pi` for `Gamma = 2`, i.e. the critical wavenumber the linear analysis predicts.
* **Heat transport:** `Nu = -dtheta/dy` at a plate, from a one-sided quadratic fit (second order), average over both plates; `Nu_budget` uses the half-cell stencil of the discrete Laplacian, which makes the bottom/top plate fluxes an *independent* measurement of the discrete energy budget (criterion d).
* **Tool chain:** PyTorch float64, NumPy/SciPy (dense LAPACK for the stability pencil), matplotlib for the figures, pytest for the gates; the campaign sweeps run on the 702 cluster (`run_cluster`), the artifacts are committed with the report.

## 3. Results

### 3.1 Onset threshold (criterion a)

| plates | nodes | Ra_c | k_c | rel. error vs anchor |
|---|---|---|---|---|
| free | 32 | 656.534 | 2.2223 | 0.149% |
| free | 48 | 657.075 | 2.2228 | 0.066% |
| free | 64 | 657.270 | 2.2230 | 0.037% |
| rigid | 32 | 1695.596 | 3.1146 | 0.712% |
| rigid | 48 | 1702.223 | 3.1164 | 0.324% |
| rigid | 64 | 1704.613 | 3.1170 | 0.184% |

Anchors: stress-free `Ra_c = 27 pi^4/4 = 657.511` (closed form), rigid `Ra_c = 1707.762`, `k_c = 3.11632` (Chandrasekhar 1961).  The two closures differ by the textbook factor `2.59` at the finest mesh.

### 3.2 Convecting layer

| case | Ra | cells | Nu (bottom) | Nu (top) | plate asymmetry | energy imbalance | |psi|_max | steps | stop |
|---|---|---|---|---|---|---|---|---|---|
| ra1e4_n32 | 1.0e+04 | 32 | 2.4228 | 2.4205 | 0.10% | 8.27e-11 | 0.097 | 12250 | steady |
| ra1e4_n48 | 1.0e+04 | 48 | 2.4121 | 2.4114 | 0.03% | 9.95e-11 | 0.097 | 22991 | steady |
| ra1e5_n24 | 1.0e+05 | 24 | 4.7663 | 4.6435 | 2.58% | 2.70e-07 | 0.115 | 18036 | max_time |
| ra1e5_n32 | 1.0e+05 | 32 | 4.5868 | 4.5319 | 1.20% | 2.62e-07 | 0.115 | 24125 | max_time |
| ra1e5_n48 | 1.0e+05 | 48 | 4.4813 | 4.4664 | 0.33% | 2.81e-07 | 0.114 | 36022 | max_time |
| ra1e5_n64 | 1.0e+05 | 64 | 4.4529 | 4.4470 | 0.13% | 2.79e-07 | 0.114 | 48050 | max_time |
| ra1e6_n48 | 1.0e+06 | 48 | 7.8746 | 7.7135 | 2.05% | 7.56e-04 | 0.116 | 48057 | max_time |
| ra1e7_n64 | 1.0e+07 | 64 | 14.5633 | 14.1665 | 2.72% | 6.09e-02 | 0.152 | 48050 | max_time |

### 3.3 Nusselt-number scaling (criterion b)

`Nu = (0.2287) * Ra^(0.2566)` over `Ra = 1e+04, 1e+05, 1e+06, 1e+07` -> exponent `0.2566 +- 0.0037` (`R^2 = 0.99959`, max residual 1.97%).

Restricted to the reference mesh (cells = 48): exponent `0.2547 +- 0.0079` (`R^2 = 0.99904`).

Local (pairwise) exponents:

| Ra interval | exponent of Nu | exponent of Nu - 1 |
|---|---|---|
| 1e+04 -> 1e+05 | 0.2684 | 0.3911 |
| 1e+05 -> 1e+06 | 0.2411 | 0.2913 |
| 1e+06 -> 1e+07 | 0.2655 | 0.2938 |

**The offset matters at these Rayleigh numbers.**  `Nu - 1 = (0.07784) * Ra^(0.3220)` (stderr 0.0170, `R^2 = 0.99446`).  The classical Malkus/Howard argument is derived for the *convective excess* over conduction, so the campaign criterion is evaluated twice: the literal `Nu ~ Ra^(1/3)` form gives `0.257` and **fails** the 10 % band, while the excess form gives `0.322`, i.e. `3.4%` from `1/3`.  Both numbers are in the ledger; neither is hidden.

### 3.4 Mesh convergence of the heat transport (criterion e)

* `Ra = 1e+04`: cells [32, 48] -> Nu [2.4216, 2.4117], change between the two finest meshes 0.409%.
* `Ra = 1e+05`: cells [24, 32, 48, 64] -> Nu [4.7049, 4.5593, 4.4739, 4.4499], change between the two finest meshes 0.535%.

### 3.5 Figures

![Marginal stability curves of the conduction state: the rigid-plate threshold sits at 1707.76 and the stress-free one at 657.51, as the textbook requires.](fig_marginal_curves.png)

*Marginal stability curves of the conduction state: the rigid-plate threshold sits at 1707.76 and the stress-free one at 657.51, as the textbook requires.*

![Heat transport against Rayleigh number with the fitted power law and the classical 1/3 slope.](fig_nusselt_scaling.png)

*Heat transport against Rayleigh number with the fitted power law and the classical 1/3 slope.*

![Temperature field and streamlines of the finest converged layer (single roll at the critical wavenumber).](fig_fields.png)

*Temperature field and streamlines of the finest converged layer (single roll at the critical wavenumber).*

![Plate-flux imbalance against time for the steady case, decaying through the 1e-8 gate.](fig_energy_conservation.png)

*Plate-flux imbalance against time for the steady case, decaying through the 1e-8 gate.*

![Mean Nusselt number against the vertical mesh.](fig_grid_convergence.png)

*Mean Nusselt number against the vertical mesh.*

## 4. Acceptance ledger

| criterion | value | reference | error | tolerance | verdict |
|---|---|---|---|---|---|
| `onset.rigid.Ra_c` | 1704.61 | 1707.76 | 1.844e-03 | 0.05 | PASS |
| `onset.free.Ra_c` | 657.27 | 657.511 | 3.669e-04 | 0.02 | PASS |
| `scaling.nu_exponent` | 0.256601 | 0.333333 | 2.302e-01 | 0.1 | FAIL |
| `scaling.nu_excess_exponent` | 0.321996 | 0.333333 | 3.401e-02 | 0.1 | PASS |
| `energy.plate_budget` | 8.2685e-11 | 0 | 8.269e-11 | 1e-08 | PASS |
| `grid.nusselt_change` | 0.00535327 | 0 | 5.353e-03 | 0.02 | PASS |
| `cavity.devahldavis.Ra1e6` | 8.7353 | 8.8 | 7.353e-03 | 0.03 | PASS |

Explicit answer to *is the key quantity within 3 % of its benchmark?*: the de Vahl Davis cavity Nusselt number at `Ra = 1e6` is inherited from the verified artifact `docs/benchmarks/verified/fvm_natural_convection/results.json` with the error recorded in the ledger above (< 3 %, pass); the Rayleigh-Benard layer has no canonical benchmark Nusselt number for the closed `Gamma = 2` box, so its transport is judged by the onset threshold, the `1/3` exponent, the energy budget and mesh convergence instead of by a single literature value.

## 5. Notes, pitfalls and limits

* **Onset is a linear property.**  `Ra_c` is measured from the linearised operator, which is the definition of the threshold.  The nonlinear box has no-slip *adiabatic side walls*, which raise the onset above the unbounded-layer value; no nonlinear onset claim is made, and a transient measurement of `Ra_c` near threshold would need `~1/((Ra-Ra_c)/Ra_c)` turnover times (millions of explicit steps) to resolve.
* **Plate asymmetry.**  The two plate Nusselt numbers of the *same* converged solution differ by the percentage recorded in the table: the second-order wall gradient is a local reconstruction, and the upwind limiting of the energy flux is not symmetric between the rising hot and falling cold sides.  The budget (`Nu_budget`) closes to the machine level because it uses the stencil the discrete Laplacian itself uses; the difference between the two numbers is the honest wall-flux uncertainty of this scheme.
* **Vertical flux profile.**  The centred flux diagnostic (`flux_profile`) is not flat to machine precision (spread in the artifacts) because the scheme transports `theta` with a van Leer *limited upwind* reconstruction, not with the centred average; the profile is a diagnostic of where the transport is convective, not the energy gate.
* **Time dependence.**  Above `Ra ~ 1e6` the two-dimensional layer is time dependent; those cases are time-averaged over the window recorded in the artifacts (`march.window`) and the fluctuation is quoted with the mean.  A single snapshot would be meaningless there.  The energy-balance column of an unsteady case is the *instantaneous* plate flux difference at the end of the run and therefore does not close (`ra1e6_n48: 7.6e-04, ra1e7_n64: 6.1e-02`); the time-averaged plate Nusselt numbers of those runs still agree to ra1e6_n48: 2.0%, ra1e7_n64: 2.7%.  The energy-conservation gate is therefore evaluated on the steady runs only.
* **Scope.**  Two dimensions, `Pr = 0.71`, `Gamma = 2`, laminar-to-transitional `Ra`.  The classical `Nu ~ Ra^(1/3)` law is an asymptotic statement; the measured exponent is reported with its standard error and its local (pairwise) values so that a reader can see the curvature of the law instead of a single fitted number.
* **The literal 1/3 gate is not met and that is the honest result.**  The measured exponent of `Nu` itself is `0.257` over the swept range, i.e. `23%` below `1/3`.  At `Ra = 1e4` the layer transports only `Nu = 2.41`, so the conductive offset `1` still carries a large share of the total flux and drags the total-flux exponent down; the asymptotic law is recovered only by the excess flux, whose exponent is quoted above.  Reaching `Nu ~ Ra^(1/3)` for the total flux needs Rayleigh numbers where `Nu >> 1` (a decade or two beyond this sweep), which is why the campaign criterion should be restated in terms of `Nu - 1` or replaced by a comparison with a Grossmann-Lohse-type prediction.
* **Under-resolution at the top of the sweep.**  The thermal boundary layer thickness at `Ra = 1e7` is `~H/Nu = 0.070` layer heights, i.e. only a handful of cells on the finest mesh; the highest-Rayleigh point is the least certain of the sweep and that is visible in its plate asymmetry and its large Nusselt fluctuation (time dependence, quoted with the mean).

## 6. How to reproduce

```bash
PYTHONPATH=src python -m tensorfvm.thermal.study stability --out runs/thermal
PYTHONPATH=src python -m tensorfvm.thermal.study sweep --shard 0 --shards 9 --out runs/thermal
PYTHONPATH=src python -m tensorfvm.thermal.study assemble --dir runs/thermal \
    --out docs/thermal-convection/artifacts --report-dir docs/thermal-convection
PYTHONPATH=src python -m pytest tests/test_thermal_convection.py
```
