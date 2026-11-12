# CASE-FVM-PERIODIC-HILLS — round 5: a Spalart--Allmaras closure, and why it does not close the anchor

- Mission 16880 (selfstart, TensorFVM) · branch `agent/m16880` · anchor `CASE-FVM-PERIODIC-HILLS` (feature ledger, `open`)
- Driver: `PYTHONPATH=src python -m tensorfvm.verification.periodic_hill_round5 docs/case-periodic-hills [--quick]`
- Time trace (the decisive artefact): `PYTHONPATH=src python scripts/round5_hill_trace.py` → `docs/case-periodic-hills/round5_hill_trace.json`
- Tests: `pytest tests/test_periodic_hill_round5.py -q` → **15 passed**; rounds 1--4 still green (`tests/test_periodic_hill.py`, `test_periodic_hill_round3.py`, … )

## 1. Where round 5 starts

Round 4 stabilised the Re = 2800 run and measured `x_r/H = 8.137` against Breuer et al. (2009)'s 5.34,
with a 48x24 → 96x48 change of **0.025 %**: the bubble was grid-converged and 52 % too long. That
established a *model-class* diagnosis — a 2D **laminar** Navier–Stokes solver cannot reproduce a turbulent
reference — and left two candidate cures: a turbulence closure (new capability) or a staggered layout
(structural, round-5 item in `ROUND4.md` §5). Round 5 implements the first and grades it against
pre-declared criteria, including the honest possibility that it does not help.

## 2. What was built

**`src/tensorfvm/periodic_hill_turbulence.py` (new, 300 lines).** Spalart & Allmaras (1992, *AIAA J.*
32(7):1598, doi:10.2514/3.12149), original form (no trip, no `ft2`): `nu_t = nu~ fv1`,
`S~ = |omega| + nu~ fv2/(kappa^2 d^2)`, `r`, `g`, `fw`, `cw1 = cb1/kappa^2 + (1+cb2)/sigma`.
Discretisation on the terrain-following mesh: the solver's own conservative face fluxes with first-order
upwind, variable-coefficient diffusion `(nu + nu~)` face-averaged, both walls Dirichlet `nu~ = 0`, the
destruction term linearised onto the diagonal (its near-wall time scale `d^2/nu~` is far shorter than
`dt` once `nu~ >> nu`). The wall distance is an **exact point-to-segment** distance to the hill contour
(plus the flat top wall) with the periodic `+-L` images included.

**`PeriodicHillSolver._assemble_momentum(mu_cell)` / `set_eddy_viscosity(nu_t)` (refactor).** The momentum
operator now takes a per-cell total viscosity; `turbulence=None` keeps the round-4 laminar path.
`advance()` advances the closure at the end of every fractional step.

**One defect found by the new gate, in the new code, before it could be trusted:** the west-face
coefficient used the **east** face metric (`c_e[j]` instead of `c_e[j, im]`). C1 caught it — the matrix
differed from an independently assembled round-4 operator by `2.8e-2` relative. Fixed; now `1.4e-14`
absolute (`4.0e-13` scaled by `dt/nu`).

## 3. Criteria (pre-declared in the driver docstring, machine-evaluated)

| criterion | measured | verdict |
|---|---|---|
| C1 uniform `mu` == round-4 operator, and the round-4 channel gate reproduces its published number | matrix `3.98e-13` (scaled); `u_max/u_b = 1.3458891` vs published `1.34589` (drift `8.8e-7`) | **PASS** |
| C2 wall distance exact: plane channel == vertical distance; on the hill flank strictly shorter | plane error `0.0` (bit-identical); hill `d/d_vertical` min `0.7606`, 16.7 % of cells ≥5 % shorter | **PASS** |
| C3 `nu~ = 0` is an exact solution (no spurious production) | `max|nu~| = 0.0` after 50 steps, `nu_t = 0` | **PASS** |
| C4 closure reproduces the log law in the plane channel (`30 <= y+ <= 0.3 h+`) | **not converged**: `u_tau = 4.00` ⇒ `Re_tau = 1.7e4` on a `Re_h = 8500` channel, i.e. the averaging window sits inside the impulsive start | **WITHHELD** |
| C5 negative control: laminar channel deviates from the log law more than the SA run | not evaluable while C4 is withheld (ratio `0.016` is computed against C4's invalid number) | **WITHHELD** |
| C6 Re = 2800, 48x24, `dt = 0.01` finite with the closure active, `nu_t >= 0`, `u_b` imposed | finite (trace to `t = 35`); `max nu_t/nu = 13.45` (trace: 21.1 at `t = 35`); `min nu_t = 8.3e-7`; `|u_b - 1| = 1.0e-10`; `max|div flux| = 6.4e-14` | **PASS** |
| C7 mean `x_r/H`, `x_s/H` within 10 % of Breuer 2009 (5.34, 0.227) | quick window: `x_r/H = 3.07` (**−42 %**), `x_s/H = 0.808` (+256 %); long trace: `x_r/H` **sweeps 2.16 → 7.41** while `nu_t/nu` still grows (2.1 → 21.1) | **FAIL (honest)** |

## 4. Results

### 4.1 The closure does not equilibrate inside the affordable window

`docs/case-periodic-hills/round5_hill_trace.json` (48x24, `dt = 0.01`, closure attached after 500 laminar
steps):

| t | max\|u\| | max nu_t/nu | x_s/H | x_r/H | G |
|---|---|---|---|---|---|
| 7.5 | 1.640 | 2.12 | 1.03 | 2.16 | 3.51e-03 |
| 12.5 | 1.510 | 9.41 | 0.84 | 2.72 | 3.65e-03 |
| 20.0 | 1.389 | 15.54 | 0.84 | 3.84 | 4.13e-03 |
| 27.5 | 1.293 | 18.79 | 0.84 | 5.16 | 4.73e-03 |
| 32.5 | 1.236 | 20.30 | 0.84 | 7.41 | 5.48e-03 |
| 35.0 | 1.233 | 21.10 | 0.84 | 7.41 | 6.04e-03 |

The eddy viscosity is still growing at the end of the run (21 nu and rising), the body force needed to
hold `u_b = 1` rises with it (3.5e-03 → 6.0e-03, i.e. the flow is being progressively "thickened"), and
`x_r/H` is a monotone function of `nu_t`: it **crosses the reference 5.34 at t ≈ 28 only on its way past
it**. Any single number quoted from this run is a property of the window, not of the solution — which is
exactly the trap round 4 recorded for the instantaneous laminar trace. The separation point, by contrast,
settles quickly at `x_s/H = 0.84` (0.808 in the time-mean of the quick run) against the reference 0.227:
the closure delays separation by a factor 3.7, and that is a stable, converged disagreement.

### 4.2 What this falsifies

The hypothesis implicit in `ROUND4.md` §5 ("a turbulence closure is the route to the Breuer numbers") is
**not supported by these measurements as a one-step cure**: a standard SA closure on this mesh produces
a growing, unconverged eddy viscosity and a bubble that is either too short or too long depending on when
you look. The remaining suspects, in order of evidence strength:

1. **Production is over-estimated on this mesh.** `|omega|` comes from the wide least-squares gradient on
   a 48x24 terrain-following mesh and the SA convection is first-order upwind; both bias the production
   term up in a separated shear layer, and `nu_t` then grows instead of equilibrating.
2. **No transition treatment.** The model is fully turbulent from `nu~ = 3 nu`; on a `Re_h = 8500` case the
   laminar part of the hill boundary layer is being filled with eddy viscosity too early, which is
   consistent with separation moving downstream (`x_s/H` 0.23 → 0.84) rather than upstream.
3. **The wall-normal resolution** (24 rows over 3.04 H, first cell `d/H = 0.037`) is coarse for a
   low-Re wall-bounded RANS model even though it was adequate for the laminar runs.

## 5. Reproduction

```bash
PYTHONPATH=src python -m tensorfvm.verification.periodic_hill_round5 docs/case-periodic-hills --quick
PYTHONPATH=src python scripts/round5_hill_trace.py        # the time trace table of §4.1
pytest tests/test_periodic_hill_round5.py -q             # 15 tests
```

`round5_summary.json` is the **quick-mode** summary (`--quick`: 600 + 600 steps on the hill, 300 + 400 on
the channel). Its C1--C3, C6 numbers are window-independent (operator algebra, geometry, invariance); its
C4/C5 are marked withheld above and its C7 is the short-window value quoted in the table.

## 6. Honest ledger

- **Delivered**: SA closure module, per-cell-viscosity momentum operator with a round-off-exact
  regression against the round-4 operator, exact wall-distance geometry, 15 tests, time-trace artefact.
- **Defect found and fixed**: the west-face metric bug in the new assembly (caught by C1 before any
  physics was published).
- **Honest FAIL**: C7 (and the anchor stays `open`); C4/C5 are **withheld, not passed** — the channel
  validation needs `>= 2e4` steps at `ny = 64` (`t >= 200`) and was not run inside this mission's budget.
- **Scope honesty**: everything here is 2D; the round-4 collocated divergence issue (`max|div u_cell|`
  O(1e-1)) is untouched, the staggered (MAC) layout is still the structural cure.
- **Next round (round 6) first item**: re-run the channel gate to convergence (`t >= 200`) *before* any
  further hill claim; then either (a) damp the SA production (limit `S~`, or use the `f_t2`/`r`-based
  correction), or (b) switch the closure convection to second order and re-check whether `nu_t`
  equilibrates. Do not tune the closure against Breuer's number.
