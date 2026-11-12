# CASE-FVM-PERIODIC-HILLS — round 2: the four assembly defects, and where the case still dies

- Mission: 16503 (selfstart, models seat), branch `agent/m16503`, repo TensorFVM
- Anchor: `CASE-FVM-PERIODIC-HILLS` (feature ledger, status `open`)
- Supersedes the diagnosis in `docs/case-periodic-hills/README.md` (round 1, mission 16139)
- Machine artefacts: `docs/case-periodic-hills/round2_summary.json`,
  `docs/case-periodic-hills/fig_round2.png`
- Driver: `python -m tensorfvm.verification.periodic_hill_round2` (8 criteria, ~2 min)
- Tests: `pytest tests/test_periodic_hill_round2.py tests/test_periodic_hill.py -q` → **15 passed**

## 1. Case, geometry and parameters (unchanged from round 1)

Periodic hills, ERCOFTAC UFR 3-30 / Breuer et al. (2009): a channel periodic in `x`,
bounded below by the six-cubic hill contour (`y/H = 1` at the crest, flat between
`x/H = 1.929` and `7.071`, period `Lx = 9H`) and above by a flat wall at
`H_top = 3.036H`. The streamwise body force `G` is adjusted every step so that the
ERCOFTAC bulk velocity `u_b = 1/2.035 ∫_H^{3.035H} u dy` is held at `U_b`, hence
`Re = U_b H/ν` is a grid-independent input. Acceptance metrics: separation and
reattachment length `x_s/H`, `x_r/H` and mean profiles `u(y)/U_b`.

| parameter | value |
|---|---|
| `nx × ny` | 16×64 (gate), 32×16, 48×24, 64×32 (reach probe) |
| `Re` | 100 (gate + reach), 2800 (reach) |
| `ν` | `U_b H / Re` |
| `dt` | 0.05 (gate), 0.01 (reach) |
| convection | blended central/upwind, `β = 0.75` |
| body-force controller | `G += 0.5 (U_b − u_b)/dt` (relax 0.2 in the reach probe) |

## 2. What was done

Round 1 blamed "the assembled implicit momentum matrix has `diag(M) ~ 564` where
`1/dt + ν Σ(|A_f|/d_n)/Vol` should be `~2.1` — an assembly bug in `_build`". That
observation was real but the diagnosis was incomplete: re-deriving every operator of
`PeriodicHillSolver._build` from the finite-volume identities produced **four**
independent defects, three of which change the leading-order physics.

**(i) Diagonal accumulators broadcast over every row.**
`diag_mom` and `diag_prs` are `(ny, nx)` arrays but were incremented with row
coefficients of shape `(nx,)` inside the `for j in range(ny)` loop, i.e.
`diag_mom += coef` added row `j`'s coefficients to *all* rows. Measured on the
16×64 round-1 operator: `|diag| = 56208` against an off-diagonal row sum of `892`
(≈`ny`× too stiff), and the LU solve returned ~`1e-3` of the right-hand side.
Fixed by writing `diag_mom[j] = ...` per row.

**(ii) The two wall faces had no Dirichlet coefficient.**
`c_n` was allocated as zeros and only `c_n[1:-1]` was filled, yet the wall no-slip
terms are `c_n[0]/Vol` and `c_n[-1]/Vol`. With both exactly zero the momentum
operator had **no wall friction at all** — the channel free-slipped. Measured: round 1
produced a plug flow with `u_max/u_b = 1.0031` and `G = 539.8` (analytic `G = 0.0117`).
This, not the diagonal scale, is what made round 1's plane-channel gate fail.
Fixed by filling `c_n = |A_n|/d_n` on all `ny+1` levels (the wall rows then carry
`|A|/(h/2)`, twice the interior coefficient of the same face).

**(iii) The pressure operator mixed two normalisations.**
The Poisson right-hand side is a face *flux* (`Σ_f A_f u_f`), so the operator must be
`Σ_f |A_f|/d_n (p_P − p_N)` with **no** volume division; round 1 divided the
off-diagonals by the cell volume but not the diagonal — the two halves of the same row
differed by a factor `Vol`. Fixed by building a separate off-diagonal matrix with the
bare face coefficients, mirrored with a positive diagonal, symmetric, and regularised
on the constant null space by `1e-10·mean(diag)` instead of pinning a row (a pinned row
leaves that cell's continuity unsatisfied — exactly what the mass gate measures).

**(iv) The periodic seam was not unwrapped.**
`dn_e` and the least-squares gradient used `xc[0] − xc[nx−1] = −(L − dx)` for the wrap
face, i.e. the long way round the domain: `8.4375` instead of `0.5625` on a 16-cell
mesh. The diffusion coefficient of the seam face was therefore **15× too weak** and the
pressure-gradient stencil was one-sided there. Fixed with the minimum image
(`dx − L·round(dx/L)`) in all three places.

**(v) The projection could not conserve mass, and needed Rhie–Chow.**
Round 1 corrected only the cell velocities with the wide least-squares gradient. Two
facts make that impossible: (a) the face-averaged divergence is blind to the odd-even
mode (`div` of a checkerboard velocity field = `0.0` exactly), and (b) the wide gradient
is blind to it as well, so `div(grad)` is singular on half the spectrum; and the LSQ
gradient is not the adjoint of the face-averaged divergence, so the correction injects
energy. Measured consequence in round 1: `max|div u| = 2.33` at the **first** hill step.
Round 2 therefore (1) keeps the compact face-coefficient Laplacian as the implicit
operator, (2) adds the standard **explicit** Rhie–Chow term
`dt[½((∇p)_P+(∇p)_N)·n − (p_N−p_P)/d_n]` to the flux that feeds the Poisson right-hand
side, (3) corrects the **face fluxes** as well as the cell velocities, and (4) carries
those projected fluxes into the next step's convection. Making the interpolated
gradient implicit as well was measured to be degenerate: both terms are `O(h)φ''` with
equal magnitude and opposite sign (`M + N(G) = O(h³)` in 1-D), so the operator
cancels at leading order.

## 3. Results (programmatic, `round2_summary.json`)

| id | criterion | verdict | value |
|---|---|---|---|
| C1 | momentum operator is an M-matrix row by row (`|diag| = Σ|off|`) | **PASS** | `1.4e-14` |
| C2 | wall faces carry the no-slip Dirichlet coefficient | **PASS** | `31.246` = `|A_n|/d_n`, `d_n = h/2` |
| C3 | pressure operator symmetric, sum-zero rows, no volume scaling | **PASS** | row-sum `4.4e-10` vs `1e-10·\|diag\|` |
| C4 | east-face distance uses the minimum image across the seam | **PASS** | `0.0` |
| C5 | plane channel reproduces analytic Poiseuille (`<1e-3` on `u_max`) | **PASS** | `2.57e-4` |
| C5b | `u_max/u_b` matches the analytic band value (`<1e-3`) | **PASS** | `−8.5e-5` |
| C6 | projection conserves mass at the flux level (`<1e-8`) | **PASS** | `3.3e-16` |
| C7 | odd-even mode in the collocated null space (documented) | **PASS** | flat: `0, 0, 0`; hill div `3.80` |
| C8 | hill case reach (report only) | **REPORT** | see below |

Plane-channel gate, 16×64, `Re = 100`, `dt = 0.05`, 6000 steps (13 s):

| quantity | round 1 | round 2 | analytic |
|---|---|---|---|
| `u_max / u_b` | 1.0031 (plug flow) | **1.34717** | 1.34748 |
| profile error on `u_max` | 100 % (no wall shear) | **2.57e-4** | 0 |
| `G` at steady state | 539.8 | **0.01169** | `2νu_b/H²` = 0.01169 |
| `max|div u|` | 2.33 (hill, first step) | **3.3e-16** (flux level) | 0 |

Note on the anchor: `u_max/u_b = 1.5` is the value for the mean over the **full**
channel height. The ERCOFTAC bulk velocity of this case averages only the band
`[H, 3.035H]` above the crest, for which the analytic ratio is
`1.34748` — computed from the solver's own bulk functional applied to the analytic
profile (`poiseuille_band_ratio`), zero hand-copying. Judging the code against `1.5`
would have failed a correct implementation.

**C8 — the hill case is still not runnable (honest FAIL).**
Reach before `max(|u|,|v|) > 1e3` or non-finite, `dt = 0.01`, 600 steps requested:

| grid | Re = 100 | Re = 2800 |
|---|---|---|
| 32×16 | no blow-up in 600 steps | no blow-up in 600 steps |
| 48×24 | no blow-up in 600 steps | **blows up at step 293** |
| 64×32 | no blow-up in 600 steps | **blows up at step 156** |

The failure is resolution-dependent and Re-dependent and is **not** a CFL limit: with
the same code the hill case diverges at every `dt` tested (0.02, 0.01, 0.005) and at
every convection blend (1.0, 0.75, 0.5, 0.0), while the same settings are stable and
exact for the x-uniform channel. No `x_r/H` is quoted: the fields are not converged and
no statistics window exists.

## 4. Figures

`docs/case-periodic-hills/fig_round2.png`
(a) plane-channel velocity profile vs the analytic Poiseuille solution, with the
round-1 plug-flow answer `u/u_b ≈ 1.003` marked as a band;
(b) the four operator defects, round 1 vs round 2 (symlog): momentum diagonal 63× too
stiff, wall coefficient `0 → 31.246`, seam face distance `15 dx → dx`;
(c) `max(|u|,|v|)` for the hill case: Re = 100 stays bounded on every grid, Re = 2800
departs at step 293 (48×24) and 156 (64×32), i.e. the finer the grid the earlier the
departure — the signature of a scheme that is not consistent, not of one that is merely
under-resolved.

## 5. Notes, limitations, traps

1. **The round-1 diagnosis was wrong in its conclusion, right in its symptom.** The
   stiff diagonal was real (defect i) but it *hid* the missing wall friction: because
   the diagonal was ~`ny`× too large, the solve returned ~`1e-3` of the response and the
   absence of wall shear was invisible. Fixing only the diagonal (round 1's plan step 1)
   turns the plug flow into a *stiff* plug flow, not into Poiseuille flow.
2. **A "consistent" `div(grad)` Poisson does not exist on this mesh.** The face-averaged
   divergence of a checkerboard velocity is exactly `0.0` (both meshes), and the
   checkerboard is also invisible to the wide LSQ gradient on the orthogonal mesh — so
   the null space is half the spectrum. Rhie–Chow (or staggering) is mandatory; any
   "make the gradient the adjoint of the divergence" shortcut is singular. The slanted
   hill mesh leaks a partial response (`div = 3.80`) but that is an artefact of
   non-orthogonality, not a substitute.
3. **Making the Rhie–Chow term implicit degenerates the operator** (`M + N(G) = O(h³)`
   in 1-D), so only the compact difference may sit on the left-hand side.
4. **Still open (withheld, not claimed):** `x_s/H`, `x_r/H` at Re = 2800 against
   Breuer et al. 2009 (`x_r/H = 5.34`, `x_s/H = 0.227` via Volpiani et al. 2021 §III —
   intermediate source, the primary paper is paywalled and was not retrievable).
5. **Cell-level divergence is a diagnostic, not a constraint.** The cell-average
   divergence remains `O(1)` for the hill case even when the flux-level divergence is
   `1e-9`, because the velocity corrector uses a least-squares fit over four faces while
   the flux correction uses the exact per-face normal derivative. On this
   non-orthogonal (terrain-following) mesh the two differ at first order; the standard
   cure is the non-orthogonal **deferred correction** of the face gradient, which is the
   first item of round 3.
6. **Controller gain.** The body force is updated with gain `relax/dt`, i.e. the
   velocity is pushed by `relax·(U_b − u_b)` per step; with `relax = 0.5`, `dt = 0.01`
   the first step kicks `G` to 50 and `u_max` to 2.3 `U_b`. The reach probe therefore
   uses `relax = 0.2`. This is a tuning knob, not a case-definition change, and it is
   recorded because a large initial kick can be mistaken for an instability.

## 6. Error verdict

| quantity | reference | measured | error | < 3 % ? |
|---|---|---|---|---|
| plane-channel profile, `u_max` | analytic Poiseuille | `2.57e-4` rel. | 0.026 % | **yes** |
| plane-channel `u_max/u_b` | analytic band value 1.34748 | 1.34717 | 0.023 % | **yes** |
| plane-channel steady `G` | `2νu_b/H²` = 0.011686 | 0.011690 | 0.03 % | **yes** |
| flux-level mass conservation | 0 | `3.3e-16` | — | **yes** |
| `x_r/H`, `x_s/H` at Re = 2800 | Breuer 2009 (5.34, 0.227) | **not produced** | — | **no — honest FAIL** |

The three analytic anchors are closed-form, so the 3 % bar is met with three to four
orders of margin. The case's own acceptance metrics are **not** delivered: the field
solution does not survive 300 steps at `Re = 2800`, so there is nothing to compare.
That is the honest state after round 2 — four real defects removed, the round-1 gate
closed at `2.6e-4`, and the remaining blocker localised to the non-orthogonal
treatment of the pressure correction.

## 7. Round-3 plan (concrete, one item per file)

1. `periodic_hill.compact_gradient`: add the non-orthogonal **deferred correction**
   (two sweeps: `(φ_N−φ_P − (∇φ)_prev·t (t·d))/(n·d)`), and switch the cell-velocity
   corrector to it; gate on `max|div_cell| < 1e-6·U_b/H` for the flat channel and
   `< 1e-2` for a steady hill solution at `Re = 100`.
2. Re-run the reach probe: require 5000 steps at `Re = 2800`, 64×32 without blow-up
   before any `x_r/H` is quoted.
3. Grid convergence `96×48 / 144×72 / 192×96`, time-averaged over ≥ 20 flow-through
   times; then `x_r/H` vs 5.34 and profiles at `x/H = 0.5, 2, 4, 6`.
4. Pre-registered expectation, unchanged from round 1: a 2-D computation without a
   turbulence model will overshoot `x_r/H` by more than 10 %; if so the verdict is FAIL
   with the 2-D-vs-turbulent gap quantified, and the next step is a closure, not
   tolerance tuning.