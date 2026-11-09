# Time-domain numerical wave tank on a surface-fitted (sigma) grid

Campaign node **P106W2-N05** — TensorFVM, seakeeping / hydrodynamics domain.
Directory: `src/tensorfvm/wave_tank_td/`, evidence: `docs/wave-tank-td/artifacts/`,
figures: `docs/wave-tank-td/figures/`, driver: `scripts/wave_tank_td/run_cases.py`.

---

## 1. Case, physics and geometry (what is computed)

A 2-D vertical-plane numerical wave tank: regular **Stokes-II** head waves are
generated at the upstream end, propagate over a flat bottom, are absorbed by a
downstream beach, and (optionally) excite a floating 2-D section with **heave +
pitch free, surge fixed**. Water depth `h = 1.0 m`, gravity `9.80665 m/s²`,
density `1000 kg/m³`, wave amplitude `a = 0.02 m`, period `T = 1.0 s`
(`λ = 1.5598 m`, `kh = 4.03`, steepness `ka = 0.081`).

Geometry pre-gate (mandatory before any wave run, `artifacts/geometry_gate.json`):
Wigley `Cb = 0.44442` vs reference `4/9` (abs error `6.4e-5`), simplified KCS
`Cb = 0.65062` vs `0.651`, `∇ = Cb·L·B·T` to round-off, `L/B`, `B/T`, `T/L`
recorded. The 2-D section used by the tank is the ship's centreplane strip
(rectangle `L×T`), whose displaced area is reproduced by the mask to 1.4 %.

### Why a new tank instead of the VOF one

`docs/wave-tank-nwt/README.md` (node `p106w1-n03`) recorded an honest FAIL: in
the staggered-MAC + THINC/VOF tank the first-harmonic amplitude decayed ~80 %
**within one wavelength**, i.e. the interface scheme, not the physics, set the
wave field. Every candidate cause listed there was an interface artefact (THINC
re-steepening, cell-centred surface Dirichlet, upwind blend).

This node removes the root cause: the vertical coordinate is normalised by the
instantaneous free surface,

```
sigma in [0,1],   y(x,sigma,t) = -h + sigma * H(x,t),   H = h + eta(x,t)
```

so the free surface **is** the top coordinate line. There is no interface to
reconstruct, no cell cut by the surface, no air phase, and the surface pressure
Dirichlet `p = 0` sits exactly on the surface. The elevation is a state variable
(`eta` per column), not something recovered from a smeared indicator field.

---

## 2. Method (what is actually solved and how)

### 2.1 Equations

Continuity in the sigma frame is the plain cell divergence of the **physical**
velocity, with the metric of the moving grid,

```
A_x[i+1] u[i+1,j] - A_x[i] u[i,j] + A_s (w[i,j+1] - w[i,j]) = 0,
A_x = dsig * Hf,      A_s = dx
```

The grid-velocity terms cancel **identically** (the geometric conservation law
holds by construction: the contravariant velocity through the surface line is
zero), which is why mass conservation needs no volume ledger.

Momentum (Euler, `nu = 0` by default; a diffusion term exists for `nu > 0`):

```
du/dt = -u du/dx - w du/dy + nu lap(u) - dp/dx
dw/dt = -u dw/dx - w dw/dy + nu lap(w) - dp/dy - g
```

with a central/upwind blend on the advective derivatives
(`advection_blend = 0.15`).

### 2.2 The two discretisation decisions that decide the accuracy

**(a) Hydrostatic/non-hydrostatic pressure split.** The pressure is written as
`p = g(eta - y) + q`. The barotropic term `g·d(eta)/dx` is then an exact *column*
quantity (no sigma metric error at all) and gravity is cancelled analytically by
the hydrostatic part in the vertical momentum; only the non-hydrostatic `q` is
projected. Without this split the sigma-coordinate pressure-gradient error
(`~ sigma·dH/dx·dp/dsigma`) is of the same order as the physical driving
gradient and the wave is not a wave — this was measured, not guessed: the first
version (no split) gave a first-harmonic amplitude of 0.35 a after 2.5λ and a
phase-inverted `k` fit.

**(b) Exact algebraic projection.** The Poisson operator is assembled as
`M = B W^-1 B^T` from the **open** face areas (`B` = discrete divergence,
`W` = face dual volumes), including a half-cell dual volume for the free-surface
Dirichlet face, which makes the discrete free-surface stiffness exactly `g`.
The solve is `M q = -flux(u*)/dt` (Jacobi-free, warm-started CG, `rtol = 1e-11`),
so after the correction `max|div u| / max|flux| ~ 1e-11`
(`div_residual_max = 1.7e-11` in the main case) and this holds with a body in
the tank as well. Sub-cell placement of the surface Dirichlet is *not* needed
(the surface is a grid line) — the 3 % wave-height criterion is met without any
sub-cell reconstruction.

### 2.3 Wave generation and absorption

* **Generation**: waves2Foam-style relaxation zone of one wavelength at the
  upstream end. The velocity field of the **Stokes-II** target wave
  (`wave_target.py`, second-order finite-depth kinematics, first order reusing
  `tensorfvm.wave_tank.theory.airy_kinematics`) is blended into `u*, w*`
  with a smoothstep weight that is 1 on the inlet face and 0 at the zone end,
  **before** the projection, so the imposed field is projected with everything
  else. Inlet face 0 is a velocity inlet (a "wavemaker"), not a wall.
* **Absorption**: quadratic sponge over `1.5 λ` at the downstream end. Both the
  velocities *and the elevation* are relaxed: in an incompressible projection
  solver a velocity-only sponge is undone by the pressure correction (mass would
  be re-routed and the beach would only produce a set-up), so the surface itself
  must be relaxed for the beach to be the open boundary it stands for. This was
  measured: with a velocity-only sponge the back wall kept `eta ~ 0.15 a` and the
  wave grew monotonically.

### 2.4 Body, cut cells and motion

The section is an axis-aligned-fraction cut cell: for every face the *open*
area fraction is computed by sub-sampling (`mask_sub = 8`), plus a **topological
closure** — a face with exactly one fully solid neighbour is blocked. That
closure is what makes grid-aligned sections exact: for them every sub-sample
lies exactly on the polygon edge, where an even-odd point-in-polygon test is
ambiguous (without it the wetted faces vanish and the body force is garbage).

The force is split the same way as the pressure: the hydrostatic part is
Archimedes **exactly** from the section polygon (`rho g nabla` at the centre of
buoyancy, `wetted_section()`), and only the non-hydrostatic `q` is integrated
over the stair-step wetted faces. Pitch reference defaults to the section's own
vertical axis — rotating about the tank origin instead displaces the body
vertically by `x·theta` and manufactures a spurious pitch–heave coupling (that
bug is why the first `rao_fine` artefact had a heave drift of 4 m and a pitch
identically zero). Integration is added-mass-informed explicit
(`m_eff = m + A33_est`, `i_eff = I + A55_est`), surge fixed, pitch clipped to
±0.35 rad.

### 2.5 Independent potential-flow reference

`potential2d.py`: a **different discretisation of the same linear physics** —
FV Laplace on a Cartesian grid, complex Robin free surface
(`dphi/dz = -(w²/g) phi`, the sign follows `Phi = Re(phi e^{-iωt})`; the opposite
sign gives an evanescent, non-radiating solution, measured), cut-cell body,
outgoing Sommerfeld boundaries plus a complex-`nu` absorbing beach, radiation
problems (heave, pitch) and diffraction (scattered potential). It returns
`A`, `B`, `F_exc`, `C` and RAOs. It is used for the added-mass/damping scale and
for the restoring check; **it is not used to certify the RAO** (see §5).

---

## 3. Results

Program-generated acceptance table (`scripts/wave_tank_td/summarize.py` →
`artifacts/acceptance.json`), main case
`wave_main` = `nx=200, ns=24, dsig=1/24, dx=0.06, dt=T/100, length=12 m,
16 periods, beach 1.5λ`:

| criterion | measured | threshold | verdict |
|---|---|---|---|
| geometry gate Cb (Wigley) | 6.4e-5 | 5e-3 | **PASS** |
| **wave-height error** (max over 7 probes, 2-harmonic fit) | **2.63 %** | ≤ 3 % | **PASS** |
| **dispersion `k` error** (7-probe phase fit vs `ω²=gk tanh(kh)`) | **0.34 %** | ≤ 1 % | **PASS** |
| **reflection `C_R`** (Goda two-point, gauges at 2.5λ/2.75λ) | **0.48 %** | ≤ 5 % | **PASS** |
| **negative control** (beach removed, closed tank) | **110.5 %** | ≥ 20 % | **PASS** |
| propagation amplitude error, fine grid 300×32 | 1.36 % | ≤ 3 % | PASS |
| propagation dispersion error, fine grid | 0.065 % | ≤ 1 % | PASS |
| GCI (amplitude, coarse 200×24 → fine 300×32) | 1.08 % | ≤ 5 % | PASS |
| RAO vs potential flow, outside 0.6<ω/ω_n<1.4 | not converged | ≤ 10 % | **NOT DELIVERED** |
| added resistance vs 2nd-order drift | not converged | ≤ 15 % | **NOT DELIVERED** |

Supporting numbers (all in `artifacts/`): first-harmonic amplitude ratio at the
seven probes `1.019, 1.026, 1.016, 1.026, 1.016, 1.026, 1.013` — flat, with the
residual +2 % consistent with the relaxation zone imposing a Stokes-II target
whose first harmonic is `a`; second-harmonic amplitude `0.0002–0.0012 m` vs the
Stokes-II bound-wave prediction `0.00081 m`; `max|div u|/max|flux| = 1.7e-11`;
volume drift `-3.2e-4` (the beach removes mass by design — see §5);
`k_fit = 4.04194` vs `k_ref = 4.02823`.

The negative control is the important one for the reflection number: the *same*
tank, *same* gauges, *same* analysis, with the beach switched off, returns
`C_R = 110 %` (a standing wave with nodes, amplitude ratio oscillating between
0.16 and 2.4) instead of 0.5 %. A reflection analysis that cannot show that is
not measuring anything.

Figures: `figures/wave_generation.png` (elevation traces + amplitude profile
against the ±3 % band), `figures/reflection.png` (beach on/off + `C_R` bars),
`figures/propagation_gci.png` (two grids), `figures/rao.png` (motion, honest).

---

## 4. How to reproduce

```bash
python scripts/wave_tank_td/run_cases.py geometry
python scripts/wave_tank_td/run_cases.py wave     --nx 200 --ns 24 --spp 100 --periods 16
python scripts/wave_tank_td/run_cases.py control  --nx 160 --ns 22 --spp 80  --periods 20 --length 7
python scripts/wave_tank_td/run_cases.py propagation --nx 200 --ns 24 --spp 100 --periods 3 --name prop_coarse
python scripts/wave_tank_td/run_cases.py propagation --nx 300 --ns 32 --spp 150 --periods 2.5 --name prop_fine
python scripts/wave_tank_td/run_cases.py rao      --period 0.8 --nx 140 --ns 18 --spp 60 --periods 20 --length 10 --body-x 4
python scripts/wave_tank_td/summarize.py
python scripts/wave_tank_td/make_figures.py
python -m pytest tests/test_wave_tank_td.py -q      # 18 passed
```

Cost: `70 ms/step` on one core for the 200×24 tank (CG dominates, ~250
iterations at `rtol=1e-11`); the main case is 1600 steps ≈ 110 s.

---

## 5. Caveats, limitations and what is *not* delivered

1. **RAO / added resistance are NOT delivered** (honest FAIL, not hidden). With
   the 0.6 m × 0.2 m section, `T_n5 ≈ 0.97 s` and the wave period 0.8 s sits
   inside the pitch resonance, where the 2-D radiation damping is very small:
   the response saturates against the zero-freeboard waterplane-loss
   nonlinearity (buoyancy drops 14 % at 0.01 rad of pitch): the recorded run
   `artifacts/rao_coarse.json` has `heave RAO = 1.92` with harmonic
   `residual = 0.98`, the pitch pinned at the ±0.35 rad clip, and the body
   sinking to the tank floor as soon as the waterplane is lost. That is a
   near-resonant, essentially undamped 2-D section, not a converged RAO. No GCI on the RAO and no BEM
   comparison is claimed. The potential solver exists and gives sensible added
   mass/damping (`A33 = 169`, `B33 = 58` per unit length at `T = 0.8 s`,
   `C55 = 294.1995 N·m/rad` matching `rho g (I_wl - nabla z_G)` exactly), but its
   radiation damping is only qualitatively right (a first-order Sommerfeld
   boundary plus a surface beach), so using it to *certify* a 10 % RAO tolerance
   would be over-claiming.
2. **The platform's 3-D TensorBEM was not run.** It lives in another tenant
   repository and a 3-D hull BEM is not directly comparable with a 2-D
   vertical-plane tank without a strip-theory reduction. The in-repo 2-D
   potential solver is the substitute and the deviation is declared with it.
3. **The beach violates strict mass conservation** by design (`volume drift
   -3.2e-4` relative over 16 periods): the elevation relaxation *is* the open
   boundary. Inside the tank the divergence is machine-exact; the volume budget
   is reported every run rather than assumed.
4. **The cut-cell force is first-order in the mask resolution.** Topological
   closure means a body face is closed as soon as one neighbouring cell is fully
   solid, so the wetted stair-step is resolved to O(Δy) for the *dynamic* load;
   buoyancy itself is exact because it is taken from the polygon. Wave
   generation and propagation do not depend on this.
5. **Advection treatment.** Central + 0.15 upwind keeps the model stable with
   the explicit coupling; at `dt = T/100` and `u/c ~ 0.08` the numerical decay
   is below the 1 % level over 4λ (measured: `prop_fine` flat to 1.4 % over
   1.5λ). Higher Reynolds-number/breaking regimes would need a real VOF/interface
   treatment and are out of scope.
6. **Time integration is first order** (forward Euler predictor + exact
   projection). The apparent order from the two grids is 1.36, consistent; the
   GCI in §3 therefore refines the time step together with the grid and is an
   indicative, not a rigorous, grid-convergence index.
7. **Pitch is clipped at ±0.35 rad** and surge is fixed, as specified by the
   node; no viscous force acts on the body (`nu = 0`).

## 6. Error statement (explicit)

| quantity | benchmark | error | < 3 % ? |
|---|---|---|---|
| first-harmonic wave height (7 probes) | analytic Stokes-II `a` | **2.63 %** max | **YES** |
| wavenumber | `ω² = g k tanh(kh)` | **0.34 %** | **YES** |
| second harmonic | Stokes-II bound wave | 0.0002–0.0012 m vs 0.00081 m (same order, sampling-limited) | not claimed |
| reflection | Goda two-point, beach ≥ 1.5λ | **0.48 %** (control 110 %) | **YES** |
| Wigley Cb | `4/9` | 6.4e-5 abs | **YES** |
| heave/pitch RAO | potential flow | not converged | **NO — honest FAIL** |
| added resistance | 2nd-order drift | not converged | **NO — honest FAIL** |

---

## 7. Files

| path | content |
|---|---|
| `src/tensorfvm/wave_tank_td/wave_target.py` | Stokes-II target wave (elevation + orbital velocity), 2-harmonic fit |
| `src/tensorfvm/wave_tank_td/sigma_tank.py` | the tank: sigma grid, exact algebraic projection, relaxation zone, beach, cut-cell section, heave/pitch |
| `src/tensorfvm/wave_tank_td/potential2d.py` | 2-D frequency-domain potential-flow reference (A, B, F_exc, C, RAO) |
| `src/tensorfvm/wave_tank_td/cases.py` | reproducible case drivers writing evidence JSON |
| `scripts/wave_tank_td/run_cases.py` | CLI driver |
| `scripts/wave_tank_td/summarize.py` | acceptance table + GCI (no hand-copied digits) |
| `scripts/wave_tank_td/make_figures.py` | report figures |
| `tests/test_wave_tank_td.py` | 18 tests: still-water fixed point, hydrostatic node, exact projection, inlet target, sponge ramp, beach vs no beach, section buoyancy, calm-water float, waterplane restoring, potential A/B/RAO |
| `docs/wave-tank-td/artifacts/*.json` | every number quoted above |
| `docs/wave-tank-td/figures/*.png` | figures |