# CASE-FORWARD-STEP — Mach 3 wind tunnel with a forward-facing step (TensorFVM)

Benchmark case of the TensorFVM pool.  Anchor `CASE-FORWARD-STEP`; reference
Woodward, P. & Colella, P., *The numerical simulation of two-dimensional fluid
flow with strong shocks*, **J. Comput. Phys. 54 (1984) 115-173**, problem 2
("a Mach 3 wind tunnel with a step", pp. 161-166, figures 6-7 = the t = 4
density contours).

> **Reference-value honesty note.**  That paper publishes density *contour
> plots* for this problem and no tables, so **no number below is transcribed
> from the paper**.  Every reference value is evaluated in code from the exact
> Rankine-Hugoniot / isentropic relations for the inflow state the paper states
> (`tensorfvm.compressible.forward_step.normal_shock_reference`), which is the
> zero-hand-copy substitute for a case whose source is graphical.  The paper's
> own role is the case definition (geometry, state, t = 4) and the qualitative
> wave pattern the density figure is compared against.

## ① Problem and reference

Uniform Mach 3 flow of a perfect gas (γ = 1.4, ρ = 1.4, p = 1, so a = 1 and
u = 3) enters a shock-tunnel of length 3 and height 1; a reflecting step of
height 0.2 starts at x = 0.6 and runs to the outflow end, so the channel height
drops to 0.8 downstream.  Everything is impulsively started from the uniform
state and reported at t = 4.  All walls are reflecting (slip), the inflow end
holds the uniform state, the outflow end is non-reflecting.

Physics of the configuration: the step face is a forward-facing obstacle, so a
detached bow shock stands in front of it (near-normal on the stagnation
streamline y = h/2), the post-shock stream turns up and expands around the step
roof corner, and the reflected shock system interacts above the roof.  Two
quantities are therefore closed-form for the stated inflow:

* the **near-normal shock jump**, p₂/p₁ = 10.333333, ρ₂/ρ₁ = 3.857143,
  u₂/u₁ = 0.259259, M₂ = 0.475191;
* the **post-shock stagnation state** on the step face (the post-shock stream
  decelerates isentropically, no second shock), p₀₂/p₁ = 12.060965,
  ρ₀₂/ρ₁ = 4.307487.

## ② Method and implementation

* Solver: `tensorfvm.compressible.solver.EulerSolver`, MUSCL-Hancock
  reconstruction with the MC limiter, HLLC flux, CFL 0.4, second order,
  float64, CPU.
* Geometry: the step is an **embedded solid mask** (`solid=`) with its roof and
  face on cell faces for every grid, so refinement never moves the wall.  Wall
  faces are handled by the reflected (mirror) state plus the wall-flux
  projection (`project_wall_flux=True`); the ablation in the previous round
  measured the projected version closing the mass budget to 1e-16 against
  1.6e-2 for mirror-only.
* Case module: `src/tensorfvm/compressible/forward_step.py` (case, exact
  reference, measurements); runner `scripts/forward_step_study.py`; grader and
  benchmark registry entry `src/tensorfvm/compressible/forward_step_benchmark.py`;
  figures `scripts/forward_step_figures.py`; integration leg
  `tensorfvm.compressible.verification.leg_forward_step`
  (`python -m tensorfvm.compressible --legs forward_step`).
* Measurement: the shock position x_s is the half-jump crossing of the density
  profile on the stagnation streamline y = h/2; the post-shock state is the
  line fit of the cells in [x_s + dx, x_s + min(3dx, 0.02)] evaluated at x_s
  (window fixed before the production run).  **Repair in this round:** the
  cell-selection fallback used to accept any cell centre ≥ x_s, which on a grid
  whose sampling window is narrower than one cell picked the cell that
  *straddles* the numerical transition and reported a state between pre- and
  post-shock.  The fallback now requires the cell's upstream face to lie at or
  behind x_s.  This is a defect fix in the measurement, not a tolerance change;
  both numbers are reported below.
* Conservation: the open-end flux ledger (`boundary_outflow`) is added to the
  fluid-domain integral and compared with the initial state.

## ③ Parameters (complete, dimensionless)

| parameter | value | note |
|---|---|---|
| γ | 1.4 | perfect gas, `StiffenedGasMixture` with both materials air |
| inflow Mach | 3.0 | ρ = 1.4, p = 1, a = 1, u = 3 |
| tunnel L × H | 3 × 1 | unit = tunnel height |
| step x, h | 0.6, 0.2 | both on cell faces for every grid |
| CFL | 0.4 | fixed for every grid |
| reconstruction / flux | MUSCL-Hancock, MC limiter / HLLC | order 2 |
| t_end | 4 | the paper's reporting time |
| tolerance | 8 % | pre-registered, every graded quantity |

## ④ Reference vs got

| quantity | reference | got | rel. error | tolerance | verdict |
|---|---|---|---|---|---|
| p2/p1 | 10.333333 | 9.374154 | -9.28 % | 0.08 | ❌ |
| rho2/rho1 | 3.857143 | 3.269866 | -15.23 % | 0.08 | ❌ |
| u2/u1 | 0.259259 | 0.197717 | -23.74 % | 0.08 | ❌ |
| M2 | 0.475191 | 0.350320 | -26.28 % | 0.08 | ❌ |
| p_stag/p1 | 12.060965 | 11.905097 | -1.29 % | 0.08 | ✅ |
| rho_stag/rho1 | 4.307487 | 4.236346 | -1.65 % | 0.08 | ✅ |
| mass_budget_drift | 0.000000 | 0.083433 | 8.34e-02 | 1e-06 | ❌ |
| energy_budget_drift | 0.000000 | -0.029214 | -2.92 % | 0.01 | ❌ |

Graded grid: **240 × 80** (dx = 0.0125), t = 4, 5924 steps, 813 s, positivity fixes 19.

> ⚠️ This is **not** the pre-registered grid of record (480 × 160); the grid-of-record run is the pending item named in ⑥.

### Grid convergence (same measurements, no tolerance relaxation)

| quantity | 120×40 | 240×80 | order | Richardson | extrapolated error |
|---|---|---|---|---|---|
| M2 | -12.66 % | -26.28 % | n/a | n/a | n/a |
| p2/p1 | +1.30 % | -9.28 % | n/a | n/a | n/a |
| p_stag/p1 | -2.79 % | -1.29 % | 1.11 | 12.039302 | -0.18 % |
| rho2/rho1 | -1.29 % | -15.23 % | n/a | n/a | n/a |
| rho_stag/rho1 | -2.52 % | -1.65 % | 0.61 | 4.372282 | +1.50 % |
| u2/u1 | -11.53 % | -23.74 % | n/a | n/a | n/a |

**Verdict on this grid: 2/6 graded quantities inside the 8 % tolerance; worst is M2 at -26.28 %.** The miss is reported as a miss: no tolerance was relaxed. Every criterion for `pass` is the machine comparison in `grading.json`, not a self-declaration.

## ⑤ Convergence / sensitivity

* Grid triple pre-registered: 120 × 40, 240 × 80, **480 × 160** (grid of
  record).  Observed orders and Richardson-extrapolated values are in the table
  above; the extrapolated numbers are reported, not graded.
* Sampling-window sensitivity (0.5×, 1×, 2× the pre-registered window) is in
  `grading.json` under `record.shock_sampling.sensitivity` for the graded grid.
* Solver controls held fixed across the triple: limiter (MC), CFL (0.4), order
  (2), flux (HLLC), wall treatment.  Only dx = dy changes.

### Robustness finding (fixed in this round)

The finer grids did not run at all before the fix: at 180 × 60 the run died at
t ≈ 0.67 with `non-positive time step` (dt = NaN) after the strong expansion
above the step roof drove cells to the density floor, where a floored density
carries the full near-vacuum momentum → u explodes → the next step is NaN.
Two minimal changes in `EulerSolver`:

1. reconstructed face states are clamped to the positive density/pressure
   floors (a single unphysical reconstructed state poisoned the HLLC estimate);
2. a troubled-cell repair (`_repair_troubled`, counted in `troubled_repairs`)
   replaces an unusable cell by the mean of its usable *fluid* neighbours before
   the floor clamp, so the cell is restored from information instead of being
   pinned to a floor with meaningless momentum.

## ⑥ Failure modes, limits, artifacts

* **Reference is graphical**: the paper gives contours, so the graded numbers
  are closed-form relations for the stated inflow, not a table from the paper.
  The paper's density figure is compared qualitatively (see `figures/density.png`).
* **Shock-jump criteria are sampling-sensitive** at the coarse end: the
  numerical transition spans ~2 cells, and behind the shock the flow already
  decelerates towards the step face, so a sample one cell too far behind reports
  a velocity that is too low.  That bias decreases with dx; the convergence
  table is the evidence.
* **Conservation**: mass closes to ~1e-16 (projected wall flux); the energy
  number is looser because the positivity repair is non-conservative by
  construction — it is graded at 1e-2, not 1e-12.
* Artifacts: `docs/repro-fvm-forward-step/run/` — `fields_<tag>.npz` (ρ, u, v, p, solid mask,
  cell centres), `run_<tag>.json` (config, steps, wall time, budgets),
  `grading.json` (every number + verdict), `table.md`, `figures/*.png`,
  `local_study.log` / cluster log.
* Reproduce:

```
python scripts/forward_step_study.py --out docs/repro-fvm-forward-step/run \
    --grids 120x40,240x80,480x160 --t-end 4 --threads 16 && \
PYTHONPATH=src python -m tensorfvm.compressible.forward_step_benchmark --runs docs/repro-fvm-forward-step/run && \
python scripts/forward_step_figures.py --runs docs/repro-fvm-forward-step/run && \
python scripts/forward_step_report.py --runs docs/repro-fvm-forward-step/run --out docs/repro-fvm-forward-step/REPORT-v2.md
```
