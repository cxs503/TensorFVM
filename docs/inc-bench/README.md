# Independent incompressible verification surface (`tensorfvm.inc_bench`)

Three classical incompressible benchmarks solved by a dedicated, self-contained
steady finite-volume kernel, so that the FVM side of the ecosystem has its own
verification evidence instead of borrowing the LBM or lid-cavity evidence.

| case | configuration | criterion |
|---|---|---|
| backward-facing step | Armaly geometry, expansion ratio 1.9423, `Re_Dh = 800` (inlet hydraulic diameter) | reattachment `L_r/S = 6.0 ± 0.2`, wall `C_f(x)` |
| planar sudden expansion | area ratio 4, `Re_D = 100` | pressure-loss coefficient `K` vs Borda–Carnot `(1-1/AR)²`, reattachment band |
| differentially heated cavity | `Ra = 10⁴/10⁵/10⁶`, `Pr = 0.71` | mean `Nu` and `ψ_max` vs de Vahl Davis (1983) |

## What the kernel is

`tensorfvm/inc_bench/solver.py` implements a steady staggered-MAC finite-volume
solver with SIMPLE pressure–velocity coupling:

* uniform Cartesian grid, `u` on vertical faces, `v` on horizontal faces, `p` and
  `T` cell centred; a boolean `fluid` mask makes any cell a stair-step solid;
* convection: first-order upwind inside the implicit operator plus a conservative
  bounded deferred correction (hybrid blend that decays with the local face
  Péclet number, i.e. second order where the grid resolves the shear and pure
  upwind where it does not);
* diffusion: central, with the wall distance honoured on closed faces (half a
  cell transverse, one cell along the face normal), which is what makes the wall
  shear and hence `C_f` meaningful;
* pressure correction: Jacobi-preconditioned CG on the SPD correction system,
  warm-started from the previous iterate;
* optional Boussinesq buoyancy and a steady energy equation for the cavity
  (`ν = √(Pr/Ra)`, `α = ν/Pr`, `gβΔT = L = 1`, free-fall scaling).

This kernel is deliberately **not** shared with `tensorfvm.lid_cavity` (DCT
projection, closed box) or `tensorfvm.solver` (cylinder SIMPLE), so a failure or a
success in one path says nothing about the other.

## Conventions (they matter more than the numbers)

* **Reynolds number.** `Re = U_mean D_h / ν` with `D_h = 2 H_in` for the step
  (Armaly's definition), `D = 2 h_in` for the planar expansion.
* **Reattachment length.** First zero crossing of the wall friction downstream of
  the separation point, linearly interpolated between cells.  It is therefore
  only as sharp as the near-wall tangential velocity, which is second order in
  the wall distance.
* **Friction coefficient.** `C_f = τ_w / (½ ρ U_ref²)` with `τ_w` from the
  half-cell wall shear and `U_ref` the inlet mean velocity.
* **Loss coefficient.** `K = (p_in - p_out + ½ρ(U_in² - U_out²)) / (½ρU_in²)`,
  with `p` area-averaged over the inlet and outlet planes.
* **Cavity scaling.** Fields are reported in the free-fall scaling used
  internally, and additionally rescaled by `α` to compare with the de Vahl Davis
  table (`ψ α⁻¹`, `u α⁻¹`).

## Reproducing

```bash
python scripts/run_inc_bench.py run --case backward_facing_step --level coarse
python scripts/run_inc_bench.py run --case natural_cavity --level medium --rayleigh 1e5
python scripts/run_inc_bench.py summary
python scripts/inc_bench_report.py     # markdown tables from summary.json
python scripts/inc_bench_figures.py    # PNG set
```

Outputs land in `artifacts/inc_bench/` (`<case>_<level>[_ra<Ra>].json` plus
`summary.json`), and the report tables are rendered from `summary.json`, so no
number is transcribed by hand.

## Limitations (stated up front)

* Steady, two-dimensional, laminar.  No turbulence model, no unsteadiness.
* Stair-step solid boundaries: the step lands exactly on a grid line (that is
  checked in the tests), but curved walls would be first-order accurate.
* First-order upwind is what the implicit operator uses; the deferred correction
  restores most but not all of the second-order accuracy, so the effective order
  measured on the grid triple is reported rather than assumed.
* The `Ra = 10⁶` cavity is at the low-resolution end of what is publishable; the
  resolution used is reported with each number.
* The digitised Armaly `C_f(x)` table and a citable `L_r` for the planar AR = 4
  expansion at `Re_D = 100` were not available offline: those comparisons are
  graded `WITHHELD` in `reference.py` and are *not* used as pass criteria.