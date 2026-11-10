# Time-domain wave tank, second campaign node: floating-section seakeeping evidence

Campaign node **p106w2-n06** — TensorFVM, seakeeping / hydrodynamics.
Code: `src/tensorfvm/wave_tank_td/` (`seakeeping.py` is new in this node).
Evidence: `docs/wave-tank-td-seakeeping/artifacts/` · figures:
`docs/wave-tank-td-seakeeping/figures/` · drivers:
`scripts/wave_tank_td/{run_seakeeping,sweep_seakeeping,gate_table}.py`.

**Verdict in one line.** The tank-quality gates this node was asked to prove are
measured and reported (geometry gate, wave height, dispersion, reflection, and a
negative control that proves the reflection gate is not vacuous); the
**floating-section gates (RAO against potential flow, added resistance against
the BEM second-order drift, two-grid GCI) are an honest FAIL** with a quantified
root cause — the immersed-body coupling of this tank had no body-velocity
boundary condition, and the cut-cell force feedback it does have is not
convergent. Sections 4.3–4.5 give the numbers, not excuses.

---

## 1. Case, geometry and parameters

| quantity | value |
|---|---|
| water depth `h` | 1.0 m |
| gravity / density | 9.80665 m/s² / 1000 kg/m³ |
| incident wave (designated run) | Stokes-II, `a = 0.02 m`, `T = 1.0 s`, `λ = 1.5598 m`, `kh = 4.028`, `ka = 0.0806` |
| tank length | 12.0 m (14.0 m for the exploratory sweeps) |
| grid (designated) | `nx × ns = 200 × 22`, `dt = T/90` |
| relaxation zone | 1.0 λ (waves2Foam-style smoothstep velocity target) |
| absorbing beach | 1.5 λ, relaxation rate 6.0 s⁻¹ |
| floating section | box `L = 1.2 m`, `T = 0.2 m`, freeboard `0.05 m`, mass `ρLT = 240 kg/m` |

Geometry pre-gate (mandatory before any wave run,
`artifacts/geometry_gate.json`): Wigley `Cb = 0.444416` vs reference `4/9`
(abs error `6.4e-5`), simplified KCS `Cb = 0.650618` vs `0.651` (abs error
`5.9e-4`), `∇ = Cb·L·B·T` to round-off, `L/B`, `B/T`, `T/L` recorded. For the
2-D strip the gate is the **hydrostatic stiffness**, see §3.1.

### 1.1 The section must have a freeboard (a geometry gate, not a detail)

The previous node used a section whose deck sat exactly on the still-water line.
That body is *neutrally stable in heave*: sinking it cannot increase the
displaced area, so the Archimedes force stops growing and nothing pushes back.
`potential_energy("heave")` measures exactly that: below the waterline the
curvature of the hydrostatic potential energy is `< 2 %` of `ρg B` (test
`test_zero_freeboard_section_has_no_heave_restoring_once_submerged`), while the
linear waterplane formula happily reports `ρg B`. With a 5 cm freeboard the
curvature is `0.98–1.00 ρg B` on both sides. All sections here carry a
freeboard, and the gate records `GM = 0.575 m > 0`.

## 2. Method

### 2.1 What the tank solves (unchanged from node n05)

A σ-grid (surface-fitted) incompressible projection solver: the free surface *is*
the top coordinate line, `y = -h + σ·H(x,t)`, `p = 0` on it, exact algebraic
projection from the open face areas (`max|div u|` at round-off), advection blend
0.15, relaxation-zone generation, and a sponge that relaxes the *surface* (a
velocity-only sponge is undone by the projection). See
`docs/wave-tank-td/README.md` §2 for the derivation.

### 2.2 What this node adds

1. **`seakeeping.py` — the measurement layer.** Hydrostatic stiffness from the
   second derivative of the hydrostatic potential energy; Goda two-point
   incident/reflected split; three-probe wavenumber inversion; the 2-D far-field
   drift force from the radiation-stress balance
   `F_x = (ρg/2)(a_i² + a_r² − a_t²)(2n − 1/2)` with `n = c_g/c`; the
   Gerritsma–Meijers energy bookkeeping reported *with its factor* so the
   energy/force confusion cannot slip in; two-grid GCI; and a frequency-domain
   2-DOF (heave/pitch) reference solver that takes its coefficients from any
   potential-flow source, so `potential2d` and TensorBEM are adjudicated by the
   same routine.
2. **Body dynamics repaired** (defect fixes, §3.2): pitch rotated about the
   section reference instead of the tank origin, consistent 2×2 mass matrix with
   the CG offset, radiation damping/added-mass inputs, and prescribed
   (`fixed` / `forced`) motion modes for diffraction and forced-oscillation runs.
3. **Contour pressure integrator** replacing the cut-cell face-fraction force,
   which is discontinuous in the body position.
4. **Moving-body flux term** in the continuity equation — the tank previously
   had *no* body-velocity boundary condition at all (§4.3).

### 2.3 Tool chain

`python scripts/wave_tank_td/run_seakeeping.py` (one case) →
`artifacts/*.json` → `python scripts/wave_tank_td/gate_table.py`
(table + criteria + figures, numbers read from artifacts, none typed).

## 3. Verified results

### 3.1 Hydrostatics: three independent routes agree to `1e-9`

`K33 = ρg B_wl`, `K55 = ρg ∇ GM` with `GM = I_wl/∇ − (z_G − z_B)`; the numeric
Hessian of `U(heave, pitch) = m g z_G − ρg ∫_submerged z dA` evaluated on the
*rotated wetted polygon*; and `potential2d.hydrostatic_restoring`.

| route | `K33` [N/m] | `K55` [N m/rad] |
|---|---|---|
| analytic `ρg B`, `ρg∇GM` | 11767.98 | 1353.318 |
| energy Hessian (numeric) | 11767.98 | 1353.317 |
| `potential2d` (after the fix) | 11767.98 | 1353.318 |

Relative differences `8.4e-10` (`K33`) and `6.1e-09` (`K55`)
(`artifacts/geometry_gate.json`, figure `figures/hydrostatics.png`).

### 3.2 Defects found and fixed (each with the measurement that exposed it)

| # | defect | evidence it was real | fix |
|---|---|---|---|
| 1 | body polygon rotated about the **tank origin**, not the section reference | a 0.02 rad pitch of a section at `x₀ = 4 m` displaced it vertically by `x₀θ = 8 cm`; measured heave ran to `-0.6 m` and pitch pinned at the ±0.35 rad clip in a calm run | rotate about `ref` (`body_polygon`) |
| 2 | `potential2d.hydrostatic_restoring` dropped the buoyancy-centre term: `ρg(I_wl − ∇z_G)` | for the 0.6×0.2 m box it returns `264.8` vs the exact `147.1 N m/rad` (factor 1.8); TensorBEM's documented `C[4,4] = ρgI + ρg∇(z_B − z_G)` and the energy Hessian both give 147.1 | `ρg(I_wl + ∇(z_B − z_G))` + regression test |
| 3 | cut-cell body force is discontinuous in the body position | in a wave run the heave force jumped `412 → 23018 N/m` inside one period at fixed geometry; forces of ±38 N/m on a 118 N/m buoyancy | contour pressure integrator (`body_force_interpolated`), statics agreement with the cut-cell integral to `2e-3` |
| 4 | heave/pitch mass matrix decoupled although the CG sits 7.5 cm under the reference | natural frequencies off by 5–10 % against the frequency-domain reference | 2×2 consistent mass matrix |
| 5 | **no body-velocity boundary condition** — the section was a static obstacle in the mask | a prescribed 1 cm heave oscillation produced exactly **zero** fluid force and zero divergence | moving-body flux `A_blocked v_body·n` added to the continuity statement |
| 6 | sections with zero freeboard are heave-neutrally stable | see §1.1 | freeboard + geometry gate |

### 3.3 Tank-quality gates (measured, designated run `wave_T1.00_p16`)

| criterion | measured | threshold | verdict |
|---|---|---|---|
| geometry gate (Wigley `Cb`, KCS `Cb`, `∇`, `GM > 0`, `K33/K55`) | `6.4e-5`, `5.9e-4`, `0`, `0.575`, `1e-9` | `5e-3` | **PASS** |
| wave-height error, max over 7 probes | **3.048 %** | ≤ 3 % | **FAIL (marginal)** |
| dispersion `k` from three probes vs `ω² = gk tanh(kh)` | **0.395 %** (`k = 4.0442` vs `4.0282`) | ≤ 1 % | **PASS** |
| reflection `C_R`, Goda two-point | **0.470 %** | ≤ 5 % | **PASS** |
| negative control (beach removed) | **110.51 %** | ≥ 20 % | **PASS** |

Time-step sensitivity of the wave-height gate, same grid (`200 × 22`, 16
periods): `dt = T/90 → 3.05 %`, `dt = T/80 → 3.34 %`, and the previous node
recorded `2.63 %` at `dt = T/100`. The 8-period run of the same grid gives
`30.0 %`, i.e. the field needs ~14 periods to reach the steady partial standing
wave — a *transient* limitation, not a dissipation one. The gate is therefore
**not met at the delivered time step**: reporting 3.05 % as a pass would have
required rounding, and the honest statement is one number above the line.

The negative control is the important one: with the beach removed the same
measurement returns `C_R = 110.51 %`, so the reflection gate can fail — it is
not a vacuous inequality.

### 3.4 2-D far-field drift force and the TensorBEM convention

`far_field_drift_2d` reduces exactly to `ρg a²/4` (deep-water radiation stress)
and to `(ρg/4)(a_i² + a_r² − a_t²)` — the coefficient TensorBEM documents for
`tensorbem.drift_force.two_d_far_field_drift`. Finite depth enters through
`2n − 1/2` (`= 0.5054` at `kh = 4.03`). The factor between the *energy content*
`(ρg/2)(a_i² + a_r² − a_t²)` and the *force* is `1/(2n − 1/2) ≈ 1.98` in deep
water; both are returned by `gerritsma_meijers_drift` so the two conventions
cannot be confused in a later node. Tests:
`test_deep_water_drift_reduces_to_radiation_stress`,
`test_finite_depth_factor_and_energy_identity`.

## 4. What did not work (honest FAIL)

### 4.1 Free 2-DOF motion diverges

With the defects of §3.2 fixed, the free-body run still diverges: on a
`140 × 18` grid at `T = 0.9 s` the heave went from `-4e-4 m` to `0.29 m` and the
pitch to `NaN` inside 1.4 s, with force excursions of `412 → 23018 N/m` (cut-cell
integrator) and, after the contour fix, a steady *growth* of the velocity at the
body (`max|u|`: `0.13 → 1.8 m/s` in 0.5 s). The added-mass ratio for this section
is `A33/m ≈ 2.7`, i.e. the explicit staggered coupling is in the regime where a
lagged added-mass force amplifies instead of damping.

### 4.2 Forced oscillation: the radiation force exists now, but it is not converged

After adding the moving-body flux, a 1 cm forced heave of a *submerged*
`0.6 × 0.2 m` strip at `T = 0.9 s` gives

| grid | `A33` [kg/m] | `B33` [N s/m²] | harmonic residual |
|---|---|---|---|
| `180 × 22` | 36.8 | 37.0 | 0.11 |
| `240 × 28` | −0.6 | 11.2 | 0.94 |

The expected added mass of that strip is `O(10²)` kg/m (a circular cylinder of the
same width gives `ρπa² ≈ 283`), and the sign change between grids says the
cut-cell representation of the body surface is not capturing the displaced mass
consistently — the force is being read from a staircase surface whose effective
area depends on `dx`. Before the flux term the same run returned *exactly zero*
force on both grids, so the change is a real step, not a converged result.

### 4.3 Why the RAO/added-resistance gates are not claimed

Both need a stable, converged floating-body run; §4.1 and §4.2 say there is
none. In particular there is **no** two-grid GCI on a body-coupled quantity that
would be meaningful while `A33` changes sign between the grids, and no BEM
adjudication of the added resistance for the *surface-piercing* section at all:
TensorBEM documents its 2-D diffraction/radiation solver as valid for a
**submerged closed body** (`solve_diffraction_indirect` docstring), and a
floating section needs the free-surface intersection, which that solver does not
model. The measurement chain that the next node needs is in place and unit
tested; the body coupling is not.

### 4.4 Limits that apply to everything above

* 2-D vertical plane, constant depth, no current, no viscosity beyond a small
  `nu` (0 by default) — the tank is inviscid apart from numerical dissipation;
* linear-ish waves: Stokes-II target, `ka ≤ 0.081`;
* the free surface over the body footprint has no Dirichlet row (the deck is dry),
  which is physical but leaves the pressure in that column constrained only by
  its neighbours;
* wave propagation in this tank is strongly resolution dependent: at 18 points
  per wavelength (`T = 0.9 s`, `dx = 0.07 m`) the amplitude collapses by 77 % over
  6–12 periods, while at 22 points per wavelength (`T = 1.0 s`) it is within 3 %.
  Do not run this tank below ~22 points per wavelength.

## 5. Error table versus the node criteria

| criterion (node) | required | measured | margin |
|---|---|---|---|
| wave height | ≤ 3 % | 3.048 % | **+0.048 pt (FAIL)** |
| dispersion `k` | ≤ 1 % | 0.395 % | 0.61 pt |
| reflection `C_R` | ≤ 5 % | 0.470 % | 4.53 pt |
| negative control `C_R` | ≥ 20 % | 110.51 % | 90.5 pt |
| RAO vs potential flow, out of band | ≤ 10 % | not delivered | — |
| resonance damping sensitivity + GCI | ≤ 5 % | not delivered | — |
| added resistance vs BEM drift | ≤ 15 % | not delivered | — |

`pass_all = false` in `artifacts/summary.json` (`failed` lists the six items
above, three of them the honest FAILs of §4).

## 6. Reproduce

```bash
# geometry + hydrostatic gate
python scripts/wave_tank_td/run_seakeeping.py --geometry-gate \
    --outdir docs/wave-tank-td-seakeeping/artifacts
# designated wave/reflection run (200x22, 16 periods, dt = T/90)
python - <<'PY'
import sys; sys.path.insert(0, "src")
from tensorfvm.wave_tank_td import cases as wt
wt.run_wave_case(period=1.0, nx=200, ns=22, length=12.0, periods=16.0,
                 steps_per_period=90, name="wave_T1.00_p16",
                 artifact_dir="docs/wave-tank-td-seakeeping/artifacts")
PY
# negative control (beach removed)
python - <<'PY'
import sys; sys.path.insert(0, "src")
from tensorfvm.wave_tank_td import cases as wt
wt.run_reflection_control(period=1.0, nx=160, ns=22, length=7.0, periods=20.0,
                          steps_per_period=80, name="control_no_beach_T1.00",
                          artifact_dir="docs/wave-tank-td-seakeeping/artifacts")
PY
# table + criteria + figures
python scripts/wave_tank_td/gate_table.py \
    --artifacts docs/wave-tank-td-seakeeping/artifacts \
    --figures docs/wave-tank-td-seakeeping/figures
# unit tests
python -m pytest tests/test_wave_tank_td_seakeeping.py tests/test_wave_tank_td.py -q
```

## 7. Next node (concrete)

1. Face-level body velocity with an **implicit** added-mass treatment (solve the
   pressure with the body acceleration as an unknown, or sub-iterate the FSI
   2–3 times per step) — this is the single blocking item for every failed gate.
2. Replace the staircase mask force with a **ghost-fluid / body-fitted** surface
   quadrature so `A33` stops depending on `dx`.
3. Then re-run §3.3 at `dt = T/100` (wave-height gate is 0.05 pt off) and the
   RAO sweep with the reference in `seakeeping.reference_rao`.