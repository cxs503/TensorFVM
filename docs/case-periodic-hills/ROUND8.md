# CASE-FVM-PERIODIC-HILLS — round 8: production limit tuning and equilibration diagnosis

- **Mission**: 18492 (selfstart, TensorFVM) · branch `agent/m18492` · anchor `CASE-FVM-PERIODIC-HILLS`
- **Driver**: `PYTHONPATH=src python scripts/round8_production_limit_sweep.py`
- **Machine**: loong64 128-core, load15 ≈ 85–95 (moderate computation only)

## 1. What round 7 left behind

Round 7 added a production limiter (`SpalartAllmarasConfig.production_limit`) that bounds the SA production term by `C × destruction`. The default value `C = 10` (from Allmaras et al. 2012) was found to be **too aggressive**: it reduced `ν_t/ν` to ~0.2, far below the expected ~20 for a turbulent channel at Re=2800.

Round 7's open questions:
1. What production_limit value gives `ν_t/ν` in the target range [15, 30]?
2. Does the limiter improve the reattachment length `x_r/H` toward the reference 5.34?
3. Has the flow equilibrated, or is `ν_t` still growing?

## 2. Method

### 2.1 Parameter sweep

Swept `production_limit` values `[None, 10, 50, 100, 200]` on the 48×24 mesh at Re=2800, running 500 steps (t=10) per case. Results:

| production_limit | x_r/H | max ν_t/ν | error vs 5.34 |
|---|---|---|---|
| None | 2.621 | 4.03 | 50.9% |
| 10 | 2.621 | 0.20 | 50.9% |
| 50 | 2.621 | 0.52 | 50.9% |
| 100 | 2.621 | 0.71 | 50.9% |
| 200 | 2.621 | 1.91 | 50.9% |

**Finding**: At 500 steps, the flow has not developed enough to show differentiation between limiter values. The reattachment length is the same (2.621) for all cases, and `ν_t/ν` is still low (max 4.03 even without limiting).

### 2.2 Time evolution (no limiting)

Ran the unlimited case to 3000 steps (t=60) to track equilibration:

| steps | t | x_r/H | max ν_t/ν |
|---|---|---|---|
| 100 | 2 | nan | 0.21 |
| 200 | 4 | nan | 0.20 |
| 500 | 10 | 2.621 | 4.03 |
| 1000 | 20 | 4.074 | 16.17 |
| 1500 | 30 | 6.093 | 21.07 |
| 2000 | 40 | nan | 25.99 |
| 3000 | 60 | 5.561 | 36.38 |

**Key findings**:
1. `ν_t/ν` is **still growing** at t=60 (36.38 and rising), confirming the round 6 diagnosis that the closure has not equilibrated.
2. The reattachment length **oscillates** as the flow develops: 2.6 → 4.1 → 6.1 → nan → 5.6. This is characteristic of an unsteady separated flow that has not reached a statistically stationary state.
3. At t=60, `x_r/H = 5.561` is within 4.1% of the reference 5.34 — the closest any run has come — but `ν_t/ν = 36.38` is still 80% above the target of ~20.

### 2.3 Production limit comparison at 1000 steps

| production_limit | x_r/H | error | max ν_t/ν |
|---|---|---|---|
| None | 4.074 | 23.7% | 16.17 |
| 50 | 4.014 | 24.8% | 4.33 |

At 1000 steps, the production limiter at `C=50` reduces `ν_t/ν` from 16.17 to 4.33, but the reattachment length is essentially unchanged (4.07 vs 4.01). This suggests that at this stage of development, the reattachment length is not sensitive to the eddy viscosity level — the flow structure is still evolving.

## 3. Interpretation

### 3.1 The equilibration problem

The SA closure on the periodic hill at Re=2800 **does not equilibrate within t=60**. The eddy viscosity grows monotonically (0.2 → 36 over 3000 steps), and the reattachment length oscillates. This is consistent with the round 6 finding that the closure is over-diffusive, but the root cause is deeper:

1. **2D URANS limitation**: The periodic hill at Re=2800 is a fundamentally 3D turbulent flow. A 2D URANS simulation with the SA model cannot capture the spanwise instabilities and 3D coherent structures that govern the separation bubble dynamics. The eddy viscosity grows to compensate for the missing 3D physics, but never reaches a stationary state.

2. **Mesh resolution**: The 48×24 mesh has a first-cell height of ~0.04H at the hill crest, giving `y+ ≈ 10–20` (estimated from the wall shear). This is marginal for a wall-resolved RANS simulation, which typically requires `y+ < 1`. The coarse near-wall resolution may be contributing to the over-diffusion.

3. **Production limiter tuning**: The standard value `C = 10` is too aggressive for this case. Values in the range `C = 50–200` reduce `ν_t/ν` but do not improve the reattachment length at early times. The limiter may be more effective at later times when the flow is more developed.

### 3.2 Honest verdict

**Gate: FAIL** (unchanged from rounds 1–7).

| metric | best value (t=60, no limiting) | reference | error | within 10%? |
|---|---|---|---|---|
| x_r/H | 5.561 | 5.34 | 4.1% | **YES** |
| max ν_t/ν | 36.38 | ~20 | 82% | **NO** |
| u/U_b at y/H=0.5 | not measured | Breuer 2009 | — | **NOT RUN** |

The reattachment length at t=60 is within 10% of the reference, but the eddy viscosity is still growing and 82% above the expected value. The velocity profile has not been measured. The honest verdict is that the case **has not converged** and the single data point at t=60 is not representative of a stationary state.

## 4. Recommendations for round 9

1. **Longer runs are essential**: The flow needs at least t=100–200 to equilibrate. This requires 5000–10000 steps on the 48×24 mesh, which is beyond the scope of a single-agent mission under load constraints. Recommend submitting to the cluster queue (`run_cluster action=submit`).

2. **Mesh refinement study**: Run with `ny = 48` and `ny = 64` to assess the effect of first-cell height on the eddy viscosity profile and reattachment length.

3. **3D extension**: The fundamental limitation is 2D URANS. A 3D simulation (even with coarse spanwise resolution) would capture the spanwise instabilities and likely produce a more realistic separation bubble. This is a major development effort.

4. **Alternative closure**: The SST k-ω model is known to perform better for separated flows than the SA model. Implementing SST would be a significant addition to TensorFVM.

5. **Statistics collection**: The `Statistics` class needs to be extended to collect Reynolds stresses and mean velocity profiles at the Breuer 2009 stations (x/H = 0.05, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0).

## 5. Artefacts

- `scripts/round8_production_limit_sweep.py`: sweep script
- `docs/case-periodic-hills/round8_sweep.json`: raw sweep results
- `docs/case-periodic-hills/ROUND8.md`: this document

## 6. Reproduction

```bash
# Lightweight sweep (500 steps, 5 cases)
PYTHONPATH=src python scripts/round8_production_limit_sweep.py

# Time evolution (no limiting, 3000 steps)
PYTHONPATH=src python -c "
from tensorfvm.periodic_hill import PeriodicHillConfig, PeriodicHillSolver
from tensorfvm.periodic_hill_turbulence import SpalartAllmaras, SpalartAllmarasConfig
import math

config = PeriodicHillConfig(nx=48, ny=24, reynolds=2800.0, time_step_s=0.02)
solver = PeriodicHillSolver(config)
solver.turbulence = SpalartAllmaras(solver, SpalartAllmarasConfig(production_limit=None))

for step in [100, 200, 500, 1000, 1500, 2000, 3000]:
    while solver.turbulence.steps < step:
        solver.advance(1)
    sep, rea = solver.reattachment()
    nu_t = solver.turbulence.viscosity().max() / solver.nu
    xr = f'{rea:.3f}' if not math.isnan(rea) else 'nan'
    print(f'{step:5d} | x_r/H={xr:>7} | max nu_t/nu={nu_t:.2f}')
"
```
