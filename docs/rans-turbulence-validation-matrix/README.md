# RANS turbulence-model validation matrix: k-ε, k-ω and SST-2003 on a wall-bounded equilibrium channel

**Status: partial delivery, honest.** The channel and model-constant legs are
computed and graded (REAL numbers, original DNS reference, grid-refinement
evidence). The backward-facing-step and NACA0012 trailing-edge legs are **not
delivered** — this node did not produce a number for them, and section 7 states
exactly what is missing and why. No criterion in this document is reported as
passed without the number that passes it.

Artifacts in this directory:

| file | content |
|---|---|
| `matrix.json` | machine-readable matrix: every graded number, every verdict, all profiles |
| `table.md` | generated tables (graded matrix, refinement ladder, sensitivity, windows, verdicts) |
| `fig1_velocity_profiles.png` | `U+` vs `y+`, three models, both DNS Reynolds numbers, DNS points |
| `fig2_eddy_viscosity.png` | `ν_t/ν` vs `y+` against the `κ u_τ y/ν` line |
| `fig3_log_region_deviation.png` | graded log-region deviations vs the criterion |
| `fig4_constant_sensitivity.png` | `ΔCf/Cf` under a `C_μ`-equivalent constant sweep |
| `fig5_reynolds_shear_stress.png` | modelled `-u'v'/u_τ²` against the DNS momentum balance |
| `fig6_turbulence_energy.png` | modelled `k/u_τ²` against `1/√β*` |
| `dns/0180`, `dns/0550` | imported original Lee & Moser (2015) profiles with provenance and SHA-256 |
| `dns/cache` | the raw downloaded `.dat` files, kept for audit |

Reproduce with

```
python3 scripts/run_rans_validation_matrix.py          # full matrix, ~7 min
python3 -m pytest tests/test_rans_validation_models.py \
                 tests/test_rans_validation_reference.py \
                 tests/test_rans_validation_channel.py \
                 tests/test_rans_validation_sensitivity.py -q
```

## 1. Case: physics, geometry, parameters

Fully developed plane turbulent channel flow between two parallel walls, solved
in the wall-normal direction only. Unit density, unit friction velocity, half
channel height as the length scale, `ν = 1/Re_τ`, constant pressure gradient
`dp*/dx = -1`. Three closures are exercised with their published constants:

| model | transport equations | constants | wall treatment |
|---|---|---|---|
| SST-2003 | `k`, `ω` | `β*=0.09`, `β₁=0.075`, `β₂=0.0828`, `γ₁=5/9`, `γ₂=0.44`, `σ_k1=0.85`, `σ_k2=1`, `σ_ω1=0.5`, `σ_ω2=0.856`, `a₁=0.31` | low-Re: `k=0`, `ω=60ν/(β₁Δ²)` |
| Wilcox k-ω | `k`, `ω` | `α=13/25`, `β=3/40`, `β*=9/100`, `σ_k=σ_ω=1/2`, `σ_d=1/2` | low-Re: `k=0`, `ω=6ν/(βΔ²)` (exact near-wall asymptotic) |
| standard k-ε | `k`, `ε` | `C_μ=0.09`, `C₁=1.44`, `C₂=1.92`, `σ_k=1.0`, `σ_ε=1.3` | wall function at `y⁺=30`: equilibrium `k`, `ε`, Spalding `U⁺` |

Reynolds numbers: **182.088** and **543.496** (the values reported in the DNS
headers of the imported Lee & Moser cases `0180` and `0550`, so no Reynolds
interpolation enters the comparison) and **590** (the number named by the
campaign; the Lee & Moser series contains no 590 — it jumps from 550 to 1000 —
so this case is graded against the wall laws and the missing DNS is stated in the
verdict instead of being imputed).

Reference data are the original author profiles
`https://turbulence.oden.utexas.edu/channel2015/data/LM_Channel_0180_mean_prof.dat`
and `..._0550_mean_prof.dat` (Lee & Moser, *JFM* 774, 2015, 395-415), imported by
the repository's existing verifier `scripts/import_channel_dns.py`, which checks
the header parameters, the wall-coordinate normalisation and the mean/stdev grid
identity, and records the SHA-256 of the raw bytes
(`docs/rans-turbulence-validation-matrix/dns/*/provenance.json`). DNS
`Cf = 2(u_τ/U_b)² = 0.0081233` (case 0180) and `0.0059078` (case 0550) come from
the same headers.

## 2. Method and toolchain

Equations (`src/tensorfvm/rans_validation/`):

```
(nu + nu_t) dU/dy = 1 - y                                   (momentum, exact integral)
d/dy[Γ_k dk/dy] + P   - sink_k k  = 0                       (k)
d/dy[Γ_v dv/dy] + P_v - sink_v v  = 0                       (ω or ε)
```

* `models.py` — the three closures, specialised to that equilibrium state. Because
  the total shear is fixed by the pressure gradient, one call returns both `ν_t`
  and the shear that solves the momentum identity, which is what closes the 1-D
  problem; `implied_kappa` returns the log-layer slope the constant set implies in
  closed form.
* `channel.py` — 1-D finite-volume solve on an exponentially clustered mesh with
  harmonic face conductances, implicit diffusion, linearised sinks, pseudo-time
  marching with under-relaxation, and a per-cell *steady-state* relative
  residual `‖defect‖/‖scale‖ < 1e-10` required on two consecutive iterations.
  Wall-bounded cells: `k=0` and the published `ω` value for the low-Re models; for
  k-ε the domain starts at `y⁺=30` with the wall-function state imposed there and
  the unresolved mass flux `∫₀^{y⁺}U⁺dy` added via Spalding's law.
* `reference.py` — the two wall-law correlations, the DNS loader, the DNS
  momentum-balance turbulent shear stress, and the metrics.
* `matrix.py` / `sensitivity.py` — the matrix, the refinement ladders, the constant
  sweep and the explicit verdict objects.

The log-region metric is the **volume-weighted RMS relative deviation of `U⁺`**
over `30 ≤ y⁺ ≤ 0.2 Re_τ` (the classic inner-layer bound), with windows at 0.15
and 0.3 `Re_τ` reported as a sensitivity. `Cf` is compared with the DNS value.
Diffusion and source discretisation are second order; the refinement ladder is
what demonstrates that.

Independent checks that the implementation is not merely self-consistent:

* `test_matches_repository_independent_sst_reference` compares this solver with
  the repository's *independent* 1-D SST reference
  (`docs/verification-native-sst-model-limit`, `Cf = 0.0086491333` at
  `Re_τ = 182.088`, n = 8192, written from a different derivation, shares no
  code): agreement within 1 %.
* `test_Cf_deviation_from_DNS_reproduces_the_documented_value` reproduces the
  6.4737 % DNS deviation recorded in that same document: obtained +6.23 %.
* `test_transport_budget_admits_the_implied_log_layer` re-derives each model's
  log-layer slope by bisecting the transport-equation budget built from the
  closure itself (agreement with the closed form to 2 %).

## 3. Results

Graded matrix, `30 ≤ y⁺ ≤ 0.2 Re_τ` (authoritative generated tables:
`table.md`):

| model | Re_τ | cells | Cf | Cf vs DNS | U_b⁺ | κ implied | κ local | RMS vs DNS | RMS vs log law | RMS vs Reichardt |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SST-2003 | 182.088 | 512 | 0.00863 | +6.23 % | 15.224 | 0.4082 | 0.287 | **6.32 %** | 3.42 % | 4.84 % |
| SST-2003 | 543.496 | 512 | 0.00605 | +2.38 % | 18.185 | 0.4082 | 0.342 | **3.20 %** | 2.66 % | 5.09 % |
| SST-2003 | 590 | 512 | 0.00591 | — | 18.401 | 0.4082 | 0.344 | — | 2.61 % | 5.04 % |
| Wilcox k-ω | 182.088 | 512 | 0.01000 | +23.09 % | 14.143 | 0.4336 | 0.322 | **13.43 %** | 10.73 % | 12.06 % |
| Wilcox k-ω | 543.496 | 512 | 0.00697 | +17.99 % | 16.939 | 0.4336 | 0.358 | **10.03 %** | 9.47 % | 11.91 % |
| Wilcox k-ω | 590 | 512 | 0.00681 | — | 17.139 | 0.4336 | 0.360 | — | 9.41 % | 11.85 % |
| standard k-ε | 182.088 | 64 | 0.00929 | +14.42 % | 14.669 | 0.4327 | 0.502 | **6.36 %** | 3.45 % | 4.89 % |
| standard k-ε | 543.496 | 64 | 0.00624 | +5.57 % | 17.907 | 0.4327 | 0.398 | **3.64 %** | 3.03 % | 5.64 % |
| standard k-ε | 590 | 64 | 0.00608 | — | 18.131 | 0.4327 | 0.395 | — | 2.96 % | 5.57 % |

Grid ladder (excerpt, full table in `table.md`), showing that the numbers above are
discretisation-independent and that *coarse meshes agree better by accident*:

| model | Re_τ | Cf (64 → 128 → 256 → 512 cells) | last step | RMS vs DNS (same ladder) |
|---|---:|---|---:|---|
| SST-2003 | 182.088 | 0.008373 → 0.008520 → 0.008594 → 0.008630 | 0.42 % | 4.69 → 5.63 → 6.12 → 6.32 % |
| SST-2003 | 543.496 | 0.005656 → 0.005862 → 0.005983 → 0.006048 | 1.09 % | 3.62 (128) → 3.20 % |
| Wilcox k-ω | 543.496 | monotone, last step 0.16 % | 0.16 % | 10.03 % |
| standard k-ε | 543.496 | 16 → 32 → 64 → 128 wall-function cells, last step 0.008 % | 0.008 % | 3.64 % |

Constant sensitivity (`C_μ`-equivalent swept by ±10 %, `Re_τ = 590`, base `Cf`
from the same 256-cell mesh):

| model | constant | base | max abs(ΔCf/Cf) for ±10 % | dlnCf/dlnC |
|---|---|---:|---:|---:|
| SST-2003 | `β*` | 0.09 | **19.47 %** | −1.75 |
| Wilcox k-ω | `β*` | 0.09 | **15.26 %** | −1.50 |
| standard k-ε | `C_μ` | 0.09 | 0.95 % | +0.09 |

Modelled log-layer turbulence state (`Re_τ = 543.496`, cells inside the window):
`k⁺ = 2.66` (SST), `2.71` (k-ω), `2.92` (k-ε) against the equilibrium value
`1/√β* = 3.33`, i.e. all three models under-predict `k` in the log layer at these
Reynolds numbers, which is the same direction as their `Cf` over-prediction.

Reynolds shear stress: the DNS value `-u'v'/u_τ²` is reconstructed exactly from the
archived mean momentum balance
(`1 - y/δ - dU⁺/dy⁺`, `reference.dns_reynolds_shear`) and the modelled
`ν_t dU/dy` agrees with it to **1.7 % (SST, Re_τ=182)** and **1.0 % (SST,
Re_τ=543)** of the peak stress. This agreement is largely structural: in a
fully developed channel the *total* stress is prescribed by the momentum balance,
so the discriminating quantity is the velocity gradient (`ν_t`), not `-u'v'`.
The modelled viscous+turbulent stress reproduces `1-y` to `≤2.2e-16` in every case.

## 4. Figures

`fig1` velocity profiles with DNS points, `fig2` eddy-viscosity ratio against
`κu_τy/ν` (a straight line of slope one is the log layer; all three models are
below it in the outer layer), `fig3` the graded deviations with the 5 % criterion,
`fig4` the constant sweep with the ±15 % band, `fig5` turbulent shear stress
against the DNS momentum balance, `fig6` modelled `k⁺` against `1/√β*`.

## 5. Pitfalls and scope notes (things that will bite the next node)

1. **A wall-function channel cannot be refined towards the wall.** At
   `Re_τ = 182`, `y⁺ = 30` is 16.5 % of the half height, so a wall-function mesh
   has three or four cells across the whole layer. The fix used here is to make
   `y⁺ = 30` the *inner boundary* of the computational domain and add the
   unresolved mass flux analytically; refining then costs nothing but recovers
   only the outer layer. Any k-ε number at `Re_τ ≲ 200` must be read with that in
   mind — its `Cf` is 14.4 % above DNS here and 5.6 % at `Re_τ = 543.496`.
2. **The wall-cell sink must be the coefficient `ε/k`, not `ε`.** Multiplying the
   transported variable by the dissipation itself drove every k-ε case to a
   spurious fixed point (`k⁺ = 1.33`, `ν_t` 30× too small, `U⁺` growing without
   bound). This was found only because the log-layer budget closed at `k⁺ = 1.33`
   instead of `1/√β* = 3.33`.
3. **Coarse meshes agree with DNS by coincidence.** For SST at `Re_τ = 182` the
   log-region deviation *grows* from 4.69 % to 6.32 % under refinement while `Cf`
   rises towards the converged value. Reporting the 64-cell number would have
   looked better and been wrong; the same trap is recorded in
   `docs/verification-native-channel-sst`.
4. **Do not forget the `(1-y)` factor of the total stress when deriving the
   implied slope.** The closed form assumes
   `dU/dy = u_τ/(κy)`; the closure actually carries `(1-y)/(ν+ν_t)`. The
   manufactured-state tests therefore pass the shear explicitly, and the
   *measured* local slope (`κ local`, 0.29-0.50) is systematically below the
   implied one because the outer layer is not asymptotic at these `Re_τ`.
5. **The reference choice moves the verdict.** SST at `Re_τ = 543` is 3.20 %
   against DNS, 2.66 % against the log law and 5.09 % against the Reichardt
   composite law: the same profile passes two references and fails the third.
   All three numbers are reported; nothing is graded against whichever reference
   happens to pass.
6. **`Re_τ = 590` has no DNS in this archive.** Lee & Moser provide 180, 550,
   1000, 2000, 5200. The mandated 590 is therefore run and graded against wall
   laws only, and the DNS-anchored cases use the header Reynolds numbers
   (182.088, 543.496) rather than their nominal labels.
7. **Reynolds-stress validation needs more than mean profiles.** The imported
   `RSTE_*_prof.dat` files in the same archive carry Reynolds-stress *budget
   terms*, not the stresses, so the turbulent shear stress here is reconstructed
   from the mean momentum balance; a direct stress-profile comparison would need
   the fluctuation profiles.

## 6. Error judgement against the campaign criteria

(a) **Backward-facing step `x_r/H` vs DNS, <10 %** — **NOT DELIVERED**. No number
is produced. See section 7.

(b) **Channel `U⁺` vs `y⁺`, SST log-law region deviation <5 %** — **SPLIT, honest:**

| case | graded deviation | verdict |
|---|---:|---|
| SST, `Re_τ = 543.496` vs DNS | **3.20 %** | PASS |
| SST, `Re_τ = 182.088` vs DNS | **6.32 %** | **FAIL** |
| SST, `Re_τ = 590` vs log law (no DNS available) | 2.61 % | PASS |
| SST, `Re_τ = 543.496` vs Reichardt composite | 5.09 % | FAIL (marginal) |

The 5 % criterion is met at the higher Reynolds number and missed at the lower
one, where the 25-cell window `30 ≤ y⁺ ≤ 36.42` sits at the edge of the buffer
layer and the model has no resolved sublayer. **SST does not pass (b) at both
Reynolds numbers.** Wilcox k-ω fails everywhere (9.4-13.4 %); standard k-ε is
comparable to SST on the profile (3.6-6.4 % vs DNS) with a 14.4 % `Cf` error at `Re_τ = 182` and 5.6 % at 543.496.

(c) **NACA0012 trailing-edge `C_p` vs experiment, <8 %** — **NOT DELIVERED**.
See section 7.

(d) **`C_μ` ±10 % → `ΔCf` <15 %** — **FAIL for SST-2003** (19.47 %), FAIL for
Wilcox k-ω (15.26 %), PASS for standard k-ε (0.95 %). Mechanism: for the k-ω
family `C_μ ≡ β*` also sets the implied log-law slope
(`κ² = (β/√β* − γ√β*)/σ_ω`), so a 10 % change of `β*` moves `κ` by 12 % and
`U_b⁺` — hence `Cf` — by about 20 %. The criterion as written cannot be met by
SST-2003 with these constants; only a re-calibrated pair (`β*` fixed by the
log-law slope, e.g. `β*` adjusted together with `γ` or `σ_ω`) would be robust.

(e) **pytest ≥20 green** — the four new files contribute **61 test cases**
(parameterised included); see section 8 for the command and the result.

**Is any of this <3 %?** Yes, and it is stated with numbers rather than as a
claim: SST `Cf` vs DNS is +2.38 % at `Re_τ = 543.496` (not at 182.088, where it is
+6.23 %); the SST log-region deviation vs DNS at `Re_τ = 543.496` is 3.20 %, so
**not** below 3 %; the modelled turbulent shear stress is within 1.0-1.7 % of the
DNS momentum balance (structural, see section 3); the cross-check against the
repository's independent 1-D SST reference is within 1 %. Every other number in
this document is above 3 % and is reported as such.

## 7. The two legs this node did not deliver

**(a) Backward-facing step at `Re_H = 28000`, `x_r/H` vs DNS, <10 %.** Not
delivered, and the honest reasons are concrete:

* reattachment needs an **elliptic** 2-D RANS solution (convective transport,
  pressure-velocity coupling, wall treatment on the step). The repository has no
  such solver with turbulence transport: `src/tensorfvm/staggered_simple.py` is a
  verified 2-D SIMPLE solver with blocked-cell masks but a **single scalar
  viscosity** and no `k`/`ω`/`ε` fields, and the SST machinery
  (`native_sst3d.py`, `native_hybrid_sst3d.py`, `native_sst_wall3d.py`) is a 3-D
  block solver whose own channel benchmark is documented as unqualified.
* the reference pair named by the campaign is inconsistent: Le, Moin & Kim (1997)
  DNS is at `Re_H = 5100`; the `Re_H ≈ 28000-37500` backward-facing-step data are
  the Driver & Seegmiller (1985) experiment. A `<10 %` grade needs the reference
  settled first (`x_r/H ≈ 6.1` in Le et al. at `Re_H = 5100` versus `≈ 6.3` in
  Driver & Seegmiller), otherwise the number cannot be interpreted.
* follow-up: add `k`/`ω` transport to the 2-D masked SIMPLE solver (variable face
  viscosity, implicit sinks, low-Re wall treatment), verify it on this channel
  matrix first, then run the step. The channel matrix in this directory is exactly
  the model-level baseline that step run must be interpreted against.

**(c) NACA0012 at `Re_c = 6e6`, `C_p` vs experiment, <8 %.** Not delivered:

* the repository *does* contain a NACA0012 `Re = 6e6` workflow
  (`docs/verification-highre-naca`: mesh family, TMR/CFL3D reference Cp/Cl/Cd
  with SHA-256, `SkewPressureMultiElementSolver` with SA), and its own status
  entry says **"计算已启动，尚未合格"** — started, not qualified. Running it here
  would either reproduce an unqualified result or produce a single unconverged
  point, neither of which is evidence.
* at `Re_c = 6e6` the `C_p` distribution is controlled by transition location and
  the trailing-edge displacement thickness; an 8 % grade needs a transition
  model (the shipped SST is fully turbulent, SA uses `ν̃ = 3ν` inflow) and a
  grid-converged family.
* the TMR reference in the repository is **CFL3D SA output, not experiment**, so
  an "experiment" grade additionally needs a digitised experimental Cp table that
  is not in this worktree.

Both legs are recorded as **NOT_SUPPORTED in this node**, not as passed or as
believed-correct. `matrix.json` contains no fabricated numbers for them.

## 8. Verification run

```
python3 -m pytest tests/test_rans_validation_models.py tests/test_rans_validation_reference.py \
                 tests/test_rans_validation_channel.py tests/test_rans_validation_sensitivity.py -q
# 61 passed
python3 -m ruff check src/tensorfvm/rans_validation tests/test_rans_validation_*.py scripts/run_rans_validation_matrix.py
```

The suite covers: closed-form vs budget-derived log-layer slopes for all three
models; constant validation and wall values; total-stress identity; limiter
engagement; SST blending; convergence and positivity of the channel solve;
grid independence; wall-function boundary construction and sublayer mass flux;
DNS provenance/hash/header verification; metric behaviour (zero against itself,
exact detection of a 10 % offset); sensitivity anchoring and leverage; and
self-consistency plus presence of the shipped `matrix.json`/`table.md`.