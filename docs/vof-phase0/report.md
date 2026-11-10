# TensorFVM two-phase Phase 0 — VOF free-surface capture + dam-break anchor

**Node** `p233w1-N03` · **Repo** TensorFVM · **Branch** `agent/m9234` · **Package** `tensorfvm.vof_phase0`
**Evidence class** REAL (algorithm + measured error) with one honest FAIL (AC2)
**Artifacts** `docs/vof-phase0/artifacts/phase0_report.json`, `docs/vof-phase0/artifacts/figures/*.png`

---

## 1. Case and scope

TensorFVM had two-phase pieces (`two_phase_vof`, `vof_fct`, `vof_plic`, `two_phase_solver`,
`verification/dam_break.py`) but no *acceptance layer*: no machine-checkable gate that
ties the volume-fraction transport to a reference experiment, no mass ledger bound, no
density-ratio stability envelope. This node builds that layer (Phase 0) as a
self-contained package.

The physical anchor is the classical dam break (Martin & Moyce 1952): a rectangular
column of water at rest on a rigid bed is released and the gravity-driven surge is
tracked. It is the canonical VOF verification case because the free surface is
transient, strongly curved and topologically simple, and because an independent
experimental front trajectory and an exact shallow-water solution both exist.

| Quantity | Value | Note |
|---|---|---|
| column half-width `a` | 0.05715 m | Martin & Moyce apparatus |
| column height `h0` | 0.11430 m | aspect `h0/a = 2` (reference curve) |
| domain | 0.60 m × 0.20 m | side walls, free-slip bed, open (atmospheric) top |
| liquid / gas density | 1000 / 1 kg m⁻³ | ratio 1000 : 1 |
| liquid / gas viscosity | 1.0e-3 / 1.0e-5 Pa s | water / air |
| gravity | 9.81 m s⁻² | |
| surface tension | 0 | not resolved at these grids |
| grids | 60×20, 120×40, 240×80 | refinement ratio 2 |
| time stepping | explicit, CFL 0.4 on `max(|u|,|v|,√(g h))` | 83 / 168 / 332 steps to `t* = 4` |
| dimensionless scales | `t* = t√(g/a)`, `x* = (x_f − a)/a` | front *displacement* |

Module layout (new, `src/tensorfvm/vof_phase0/`):

* `advection.py` — algebraic VOF transport, flux-corrected (Zalesak 1979), dimensionally
  split, exactly conservative;
* `solver.py` — variable-density MAC projection solver with two surface models
  (`void` free-surface and fully coupled `two_phase`);
* `anchors.py` — digitised Martin & Moyce surge/height curves and the Ritter (1892)
  analytic dam-break solution;
* `dam_break.py` — coupled case driver, front/height extraction, mass ledger;
* `scripts/run_vof_phase0.py` — the acceptance campaign (gates + report JSON + figures).

## 2. What was done

**Interface transport.** The volume fraction is advanced in conservative flux form
with a two-step FCT scheme: a donor-cell flux, an acceptor (Lax-Wendroff) candidate and
a Zalesak limiter on the anti-diffusive part whose capacities are taken from the
*low-order updated* state. Every face flux is single-valued, so the update telescopes:
the liquid volume is conserved to round-off, which is what AC4 requires. Three
implementation traps were measured and fixed (see §5).

**Flow solver.** Explicit momentum predictor (first-order upwind convection, harmonic
viscosity blend, gravity, free-slip walls) plus a variable-density projection
`div((1/ρ)_f grad φ) = div(u*)/dt`, Jacobi-preconditioned CG (rtol 1e-9; measured 205
mean iterations at 240×80). The gas is treated as an atmospheric **void**: the pressure
problem is solved on the liquid cells only, with `p = 0` on every face that touches the
void. The reason is measured, not stylistic — see the equilibrium test in §3.

**Verification.** `pytest tests/test_vof_phase0_advection.py tests/test_vof_phase0_dam_break.py`
→ **27 passed** (15 advection, 12 dam-break/anchors), covering 1D uniform advection
(periodic), 1D reversal, 2D translation, solid-body rotation, shear flow, sweep-order
consistency, boundedness, exact conservation, the CFL guard, the hydrostatic rest state,
the 1000:1 stability envelope and the reference-curve scalings.

## 3. Results

All numbers below come from `docs/vof-phase0/artifacts/phase0_report.json` (written by the
campaign, no hand transcription).

### 3.1 Gate summary

| Gate | Criterion | Measured | Verdict |
|---|---|---|---|
| **AC1** | ≥ 10 VOF pytest cases (1D/2D advection, rotation, shear) | 15 advection + 12 anchor/solver tests, all passing | **PASS** |
| **AC2** | front error ≤ 10 % vs Martin & Moyce at ≥ 3 times | −60.5 % (t*=1), −31.0 % (t*=2), −11.7 % (t*=3) | **FAIL (honest)** |
| **AC3** | ρ ratio 1000:1 stable ≥ 1000 steps | 1000 steps, finite, α ∈ [0, 1.00002], volume residual 2.6e-16 | **PASS** |
| **AC4** | total volume residual ≤ 1e-6 m² | 8.7e-19 m² (worst over all grids: 2.6e-18 m²) | **PASS** |
| **AC5** | 3-grid GCI study, error monotonically decreasing | −51.5 % → −42.6 % → −31.0 % (monotone); Richardson order **negative** | **PASS with caveat** |
| **AC6** | paper-quality report with field + comparison figures | 4 figures emitted | **PASS** |

### 3.2 Surge front versus the experimental anchor (AC2, finer grid 240×80)

| t* | x* sim | x* Martin & Moyce | rel. error | Ritter limit x* |
|---|---|---|---|---|
| 1.00 | 0.553 | 1.400 | −60.5 % | 2.83 |
| 2.00 | 1.690 | 2.450 | −31.0 % | 5.66 |
| 2.99 | 2.915 | 3.300 | −11.7 % | 8.47 |

The residual column height at `t* = 1` is **+61.9 %** above the digitised curve.
Figure `front_comparison.png` shows the trajectory against both anchors; the simulation
sits between the experiment and the shallow-water limit everywhere, i.e. the *ordering*
of the physics is right (viscous 2D collapse is slower than the inviscid Ritter
rarefaction) but the absolute surge speed is too low.

### 3.3 Grid convergence (AC5)

| grid | dx [mm] | front error at t* = 2 | x* at t* = 2 |
|---|---|---|---|
| 60×20 | 10.0 | −51.5 % | 1.187 |
| 120×40 | 5.0 | −42.6 % | 1.406 |
| 240×80 | 2.5 | −31.0 % | 1.690 |

The error against the experiment decreases monotonically, so the stated AC5 criterion is
met. **But** the self-convergence is not in the asymptotic range: the coarse→medium
change (0.219) is *smaller* than medium→fine (0.284), giving a negative Richardson order.
The extrapolated limit is therefore not meaningful; what the table does say is that the
front is still moving toward the experimental curve at 2.5 mm cells, i.e. a large part of
the AC2 deficit at `t* ≥ 2` is discretisation, while the early-time deficit is not.

### 3.4 Conservation, stability, and model selection

| Check | Result |
|---|---|
| liquid volume residual, 240×80, `t* = 4` | **8.7e-19 m²** (relative 1.3e-16) |
| liquid-cell divergence after projection (120×40) | max **1.4e-5 s⁻¹**, mean 9.8e-6 s⁻¹ (CG rtol 1e-9) |
| 1000:1 fully coupled projection, 1000 steps | finite, α ∈ [0, 1.00002], volume residual 2.6e-16, 17.7 s |
| stratified pool at rest, `void` model, 200 steps | spurious velocity **1.3e-14 m/s** (exact equilibrium) |
| stratified pool at rest, `two_phase` model, 200 steps | **diverges** (|u| 1.6e77, α up to 3.27) |

The last two rows are the reason the accuracy gates run on the void model: the fully
coupled variable-density projection with a 1000:1 jump inside one elliptic problem is
numerically unstable even for a pool at rest, whereas the void formulation reproduces the
rest state to machine precision. The 1000:1 ratio is still exercised (AC3) with the
coupled model, and it is stable on the *transient* dam break for ≥ 1000 steps.

### 3.5 Figures

1. `figures/free_surface_field.png` — liquid volume fraction at the end of the run.
2. `figures/front_comparison.png` — surge front: simulation vs experiment vs Ritter.
3. `figures/grid_convergence.png` — front error versus cell size (three grids).
4. `figures/mass_ledger.png` — absolute liquid-volume residual over the run.

## 4. Error judgement (key quantities vs baseline, explicit)

| Quantity | Baseline | Measured error | < 3 % ? |
|---|---|---|---|
| liquid volume (every step, every grid) | exact conservation | ≤ 1.3e-16 relative | **yes** |
| stratified pool spurious velocity | 0 (exact rest) | 1.3e-14 m/s | **yes** |
| 1D VOF advection, one period (L1 / wave measure) | exact translation | 3.5 % | **no** (marginal) |
| 2D rotating disk, one revolution (L1 / disk area) | exact rotation | 64 % at 64² (peak α 0.88) | **no** |
| surge front at t* = 2 | Martin & Moyce | **31 %** | **no** |
| surge front at t* = 3 | Martin & Moyce | 11.7 % | **no** |
| residual column height at t* = 1 | Martin & Moyce | 61.9 % | **no** |
| 1000:1 stability (1000 steps) | no divergence | no divergence, bounded α | **yes** (qualitative) |

**Verdict: AC2 is an honest FAIL.** The package meets the conservation, boundedness and
stability gates outright and meets the grid-convergence criterion as stated, but it does
**not** reproduce the measured surge within 10 % at three times.

## 5. Traps, defects and limitations

**Fixed during this work (all measured, not guessed).**

1. *Flux clipping destroys conservation.* The production `vof_scheme="plic"` path loses
   **3.2 %** of the liquid column on the 120×40 dam break because the geometric sweep is
   followed by a `clip` that discards the overshoot. The Phase-0 transport never clips:
   the limiter acts on the anti-diffusive flux, before the update.
2. *Limiter capacities must use the low-order updated state.* Taking the Zalesak
   capacities from the pre-update field overstates the headroom by exactly the donor
   flux and lets the scheme overshoot: measured −1.9e-2 / +1.9e-2 on a 1D square wave.
   Fixed → bounded to 2e-18 and L1 error 0.0106.
3. *MUSCL face values must not be clipped into the donor/acceptor bracket alone.* That
   clip forbids steepening a smeared front. Measured on the rotating disk (64², one
   revolution): peak α 0.95 (MUSCL + limiter) vs 0.50 (pure acceptor/Lax-Wendroff
   candidate), which over-diffuses a sharp interface.
4. *The momentum predictor must carry `-∇p/ρ`.* Without it the hydrostatic rest state is
   not a fixed point and the surge start is delayed by ~0.5 dimensionless time units
   (measured: the whole field free-falls at `g` and the front never picks up).
5. *Face inverse density.* Arithmetic mean of `1/ρ` on an interface face (`0.5` for a
   1000:1 jump) gives a 1e4 m s⁻² launch and CFL 9.0 on the first step; the harmonic
   face value `2/(ρ₁+ρ₂)` is the consistent one.

**Pre-existing defect found in the production solver (reported, not patched).**
`TwoPhaseSolver._projection_step` integrates the vertical momentum on
`v[1:-1, 1:-1]`, i.e. on columns `1..nx-2` only, while `_velocity_correction` and the
Poisson operator use *all* columns. The two wall-adjacent columns therefore get the
pressure correction without the predictor (no pressure gradient, gravity, diffusion or
convection). On the dam break this freezes the column at the upstream wall: measured wall
depth **0.98 h0** while the neighbouring column falls to 0.05 h0, i.e. the column cannot
drain. Extending the block to all columns fixes the drainage (wall depth 0.83 → 0.31 h0
over t* = 0.5 → 3), **but** it fails four existing verified tests
(`test_two_phase_projection::test_csf_acceleration_*`,
`test_bubble_rise::test_advective_accel_is_zero_for_uniform_flow`,
`test_bubble_rise::test_hydrostatic_pool_at_rest`), so it was reverted: a shared-module
behaviour change of this size needs its own node and its own regression baseline. The
Phase-0 package is therefore self-contained and does not touch the shared solver.

**Pre-existing verification debt confirmed on `main` (b29a0e15).**
`tests/test_dam_break_verification.py::test_surge_front_against_martin_moyce` fails with
`surge_front_mean_error = 0.2504 ≥ 0.15`, and
`::test_column_height_against_martin_moyce` fails because the column height does not
decrease at all. Both are the signature of the frozen-wall-column defect above, plus a
normalisation error in that module (it compares `x_f/a` with a curve digitised as
`(x_f − a)/a`, a spurious +1/a offset at early times).

**Limitations of this Phase-0 result (scope statement).**

* The surge is too slow at early times by a factor the grid study does **not** remove
  (−60 % at `t* = 1` on all three grids, versus −31 % at `t* = 2` on the finest). The
  near-wall drainage and the thin leading tongue are the weak points; both are
  single-cell phenomena at 2.5–10 mm cells, and the surface pressure condition is applied
  at the void cell centre rather than at the reconstructed interface position.
* Bed friction is free-slip; the experiment has a rigid frictional bed. This does not
  explain a *lag* (a free-slip bed should make the surge faster, and the simulation does
  overtake the digitised curve beyond `t* ≈ 4`), but it does mean the late-time agreement
  cannot be read as physical agreement.
* The Martin & Moyce curve is a digitisation (few-percent reading error); the 10 % gate
  is wider than the digitisation but the reference is not a precision dataset. The
  residual-height curve in particular is inconsistent with a free-fall collapse at
  `t* = 0.5` and is therefore reported as informational only, never as a gate.
* Single-phase (void) gas; no surface tension, no compressibility, 2D only.
* The 1000:1 stability evidence (AC3) is a *stability* statement, not an accuracy
  statement: the coupled two-phase projection is not used for the quantitative front
  comparison because it fails the exact rest-state test.

## 6. Reproduction

```bash
pytest tests/test_vof_phase0_advection.py tests/test_vof_phase0_dam_break.py -q   # AC1
python scripts/run_vof_phase0.py --out docs/vof-phase0/artifacts                      # AC2–AC6
```

Runtime: ~4 min single core for the full campaign (240×80 grid is 200 s of it);
`--quick` runs a 3-grid smoke study in ~30 s.