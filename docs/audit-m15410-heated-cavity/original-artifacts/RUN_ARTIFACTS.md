# de Vahl Davis (1983) Heated Cavity Benchmark — Run Artifacts

## Benchmark Definition
- **Case**: Natural convection in a differentially heated square cavity
- **Reference**: de Vahl Davis, G. (1983), "Natural convection of air in a square cavity: a bench mark numerical solution", Int. J. Numer. Methods Fluids 3, 249-264.
- **Parameters**: Pr = 0.71, Ra ∈ {10³, 10⁴, 10⁵, 10⁶}
- **Geometry**: Unit square [0,1]², hot left wall (T=+0.5), cold right wall (T=-0.5), adiabatic top/bottom
- **Solver**: TensorFVM StaggeredMAC (van Leer limited conservative fluxes, DCT-II projection, explicit marching)
- **Mesh**: n×n uniform cells, n ∈ {32, 64, 128}, with grid sequencing

## Acceptance Results (finest mesh n=128)

| Ra | Nu_ref | Nu_got | Nu_err% | u_max_ref | u_max_got | u_err% | v_max_ref | v_max_got | v_err% | Status |
|---|---|---|---|---|---|---|---|---|---|---|
| 10³ | 1.118 | 1.1179 | 0.01% | 3.649 | 3.647 | 0.06% | 3.697 | 3.694 | 0.07% | PASS |
| 10⁴ | 2.243 | 2.2463 | 0.15% | 16.178 | 16.194 | 0.10% | 19.617 | 19.637 | 0.10% | PASS |
| 10⁵ | 4.519 | 4.5320 | 0.29% | 34.73 | 34.976 | 0.71% | 68.59 | 68.658 | 0.10% | PASS |
| 10⁶ | 8.800 | 8.9009 | 1.15% | 64.63 | 65.337 | 1.09% | 219.36 | 220.707 | 0.62% | PASS |

**All cases pass the <5% tolerance criterion.**

## Artifacts
- `summary.json` — Full benchmark ledger with all mesh results, convergence history, and acceptance checks
- `case_Ra*.json` — Individual case results with per-mesh diagnostics
- `nu_comparison.png` — Bar chart comparing Nu values
- `nu_convergence.png` — Mesh convergence plot showing error vs resolution
- `nu_profile_Ra1e5.png` — Local Nusselt number profile on hot wall
- `velocity_comparison.png` — Velocity extrema comparison

## Reproduction
```bash
bash scripts/repro_heated_cavity.sh
```

## Bug Fix
Fixed incorrect velocity reference values for Ra=1e4 in `src/tensorfvm/internal_flow/reference.py`:
- Was: u_max=4.535, v_max=6.776 (incorrect transcription)
- Fixed: u_max=16.178, v_max=19.617 (correct per de Vahl Davis Table 3)
Also corrected Nu reference for Ra=1e3 from 1.334 to 1.118.
