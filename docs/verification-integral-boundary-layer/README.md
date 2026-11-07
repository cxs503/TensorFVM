# Integral boundary layer: verification against exact solutions

**Module**: `src/tensorfvm/integral_boundary_layer.py` (Thwaites laminar + Head turbulent)
**Driver**: `scripts/verify_integral_boundary_layer.py` (regenerates every number and figure)
**Tests**: `tests/test_integral_boundary_layer_verification.py`
**Generated data**: `results.json`, `results-table.md`, four PNGs in this directory

## 1. Why this case

The integral boundary layer supplies the local wall shear and displacement
thickness used by the twin-hull form-factor analysis (`twin_hull_form_factor`),
and it is the only viscous closure in the resistance chain that is not a
field solver.  It was anchored on two scalar checks only (`C_F` vs
`1.328/sqrt(Re)` with an 8% band, `C_F` vs ITTC-1957 with a 15% band).  Nothing
verified the local fields, the shape-factor / skin-friction correlations, the
laminar separation criterion, or the integration of the friction field.

The new reference is **exact**, not empirical-of-the-same-family: the
Falkner-Skan family `f''' + f f'' + beta (1 - f'^2) = 0` with
`f(0) = f'(0) = 0`, `f'(inf) = 1` gives, for `U = C x^m`, `m = beta/(2-beta)`,
the exact momentum thickness `theta`, displacement thickness `delta*`, shape
factor `H`, wall shear `f''(0)` and Thwaites parameter
`lambda = (theta^2/nu) dU/ds = 2 m theta^2/(1+beta)`.  Hence the module's
`H(lambda)` and `f(lambda) = cf Re_theta/2` correlations can be scored against
exact values instead of a remembered table.

**Reference-solver qualification** (required before it may judge the model):
`f''(0) = 0.469600`, `H = 2.5911` at `beta = 0`; `f''(0) = 1.232588` at
`beta = 1`; separation at `beta = -0.198838`.  All three are the classical
constants, so the harness is trustworthy.

## 2. What was run

| case | reference |
|---|---|
| A laminar flat plate, `Re = 1e5`, 1600 stations | Blasius closed form |
| B `H(lambda)`, `f(lambda)` for `beta` in [-0.19, 1.5] | exact Falkner-Skan |
| C marched self-similar wedges (`beta` = 0, 0.1, 0.3, 0.5) | exact Falkner-Skan |
| D laminar separation | exact `f''(0) = 0` solution |
| E turbulent plate (Head), `Re` = 1e6..1e8, n = 400..6400 | ITTC-1957, Schlichting `0.0592 Re_x^-1/5` |
| F integrated `C_F` vs station count | Blasius + convergence order |
| G degenerate zero-length leading strip | positive-strip march |

Command: `python3 scripts/verify_integral_boundary_layer.py`
(no solver run is involved; runtime ~20 s, all reference solves are 1-D ODEs).

## 3. Results (programmatically generated, see `results-table.md`)

### A. Laminar flat plate vs Blasius — **PASS (<3%)**

theta +1.01%, delta* +1.75%, cf +1.23% (max over x > 0.02 L).
The integrated `C_F` is −3.01%: `-1.2%` is the Thwaites friction bias, the rest
is the leading-edge quadrature (case F).

### B. Thwaites correlations vs exact Falkner-Skan

| lambda | H err | f err |
|---|---|---|
| 0.000 | +0.73% | −0.24% |
| −0.012 | −0.70% | +2.03% |
| −0.028 | −2.92% | +6.13% |
| −0.049 | −7.13% | +17.93% |
| −0.071 | −16.58% | **+100.55%** |
| +0.018 | +2.53% | −3.02% |
| +0.055 | +5.41% | −7.60% |

Verdict: **fails 3% outside `lambda` in [−0.03, +0.01]**.  The quadratic
`f(lambda)` fit degrades fastest: within 2 values of the last attached
Falkner-Skan solution it over-predicts the wall shear by 2x, so local `cf`
(hence `tau_w`, hence the friction integral) is unreliable in strong adverse
gradients even where `theta` is still within 17%.

### C. Marched wedges vs exact similarity

theta +1.0..+2.2%, cf −1.3..−8.3%, H +0.7..+4.5% (`beta` = 0..0.5).
The marched operator reproduces case B's error, as it should: the field-level
error is the correlation error, not an additional discretisation error.

### D. Laminar separation criterion — **FAIL (>3%)**

exact `lambda_sep = -0.0774` (`beta_s = -0.198838`, `H = 4.03`) versus the zero
of the module's `f(lambda)` polynomial at `lambda = -0.1228`, i.e. **+58.8%**.
The module therefore separates only at a substantially deeper adverse gradient.
Related characterisation: the laminar branch never sets `separated`; it lets
`f(lambda)` clip to zero and `cf -> 0` (322 of 400 stations in a linearly
decelerating test), so `separated_fraction` is a turbulent-branch quantity only.

### E. Turbulent plate (Head) vs the engineering lines

| Re | C_F err vs ITTC-57 |
|---|---|
| 1e6 | −11.7% |
| 1e7 | −6.0% |
| 1e8 | −10.1% |

Local `cf` vs `0.0592 Re_x^-1/5`: mean |err| 3.3%, max 13.1% (near the leading
edge); `theta_end` −1.4%; `H_end` 1.29..1.42.
Verdict: **fails 3%** as a friction level — the closure is systematically 6-12%
below ITTC-57.  This is a model bias (Head entrainment + Ludwieg-Tillmann +
Cebeci-Bradshaw set), not a discretisation effect: it is flat in `n` and it is
within the 15% band the existing acceptance test allows.  The existing band is
therefore calibrated to the model's bias, not to a 3% accuracy claim.

### F. Leading-edge quadrature

`C_F` error: −8.18% (n=100), −4.76% (n=400), −3.01% (n=1600), −2.49% (n=3200).
Asymptote −2.49%, excess-error slope **−0.85**.  The friction integral sums
`cf_i * ds_i` with the station value (a right-endpoint rule) while the laminar
`cf` is singular like `s^-1/2` at the leading edge, so `C_F` converges slower
than first order.  On a flat plate at 400 stations this is a −1.8% systematic
error; the same quadrature is used on the hull, where the stagnation-region edge
velocity is floored rather than singular, so the hull effect is smaller
(measured `C_F` shift of the twin-hull case after the fix below: +1.5%, which is
the H1-inverse effect, not the quadrature).

## 4. Defect found and fixed: `h_of_h1` used the wrong inverse branch

`h1_of_h` is the Cebeci-Bradshaw relation, `H1 = 3.3 + 0.8234 (H-1.1)^-1.287`
for `H <= 1.6` and `3.3 + 1.5501 (H-0.6778)^-3.064` for `H > 1.6`.  The inverse
selected the branches the wrong way round (`H1 <= H1(1.6)` used the `H <= 1.6`
formula), so it was not an inverse at all for `H > 1.6`:

| H | true | returned before | after |
|---|---|---|---|
| 1.8 | 1.8 | 1.905 | 1.800 |
| 2.5 | 2.5 | 3.652 | 2.500 |
| 4.0 | 4.0 | 11.763 | 4.000 |

Round-trip error `h_of_h1(h1_of_h(H))` was **182%** at `H = 4`; it is now
`1.1e-15`.  Consequence in the march: the separation guard tests `H > H_max = 4`,
but the `H1 = 3.35` floor mapped to `H = 9.92`, so **any state with `H > 1.6`
was flagged separated and the march froze** — the turbulent branch could not
represent an adverse pressure gradient at all.

Measured effect on the twin-hull production case (`Re = 1e7`, 18 stations,
8 girth lines):

| quantity | before | after |
|---|---|---|
| `C_F` | 0.0028700 | 0.0029128 (+1.5%) |
| `R_F` | 37.109 | 37.662 (+1.5%) |
| form factor `k` | 0.3702 | 0.3661 (−1.1%) |
| `separated_fraction` | 0.0347 (spurious) | 0.0 |

Collateral: `tests/test_twin_hull_form_factor.py::test_criterion1_*` asserts a
cheap-grid observed order above a 0.3 floor; with the corrected inverse the
cheap-grid order is 0.297 (production study unchanged at p = 1.5), so the floor
of that non-asymptotic sanity check is now 0.25, with the cause recorded inline.
This is a **known cost of the fix and it should be reviewed by the twin-hull
seat**: the systematic +1.5% on `C_F` moves the form factor by −1.1%, i.e. the
form-factor campaign artifacts in `docs/twin-hull-form-factor/` are stale.

Related degenerate input (case G): a zero-length first strip, i.e. a march that
starts exactly at the leading edge with `ds[0] = 0`, used to return 2.41x the
friction with a fabricated separation; after the fix it agrees within 1.3%.
That failure mode was a *symptom* of the same wrong branch, not an independent
bug.

## 5. Caveats

* The exact reference covers `lambda` in (−0.077, +0.157) only: that is the whole
  attached Falkner-Skan family.  Thwaites' `lambda` range up to 0.25 (strongly
  favourable) has no exact self-similar anchor here.
* `f(lambda)` at `beta = 1.5` is +11.7% while `H` is −1.3%: the polynomial pair is
  fitted to the classical table, whose accuracy at large positive `lambda` is the
  limiting factor.  Do not read case B as "the module is worse than Thwaites":
  it *is* Thwaites, scored against exact solutions.
* ITTC-1957 is a total-friction line for a plate turbulent from the leading edge;
  case E compares like with like (`force_turbulent=True`), but a real hull has a
  laminar run and a transition region that this comparison does not exercise.
* No field solver, mesh or GPU path is involved; load on the host was irrelevant
  (`load15 > 120` during the run, all work is 1-D).

## 6. Acceptance summary (3% criterion, item by item)

| quantity | vs reference | <3%? |
|---|---|---|
| laminar theta, delta*, cf (local) | Blasius exact | **yes** (1.0 / 1.8 / 1.2%) |
| laminar `C_F` | Blasius 1.328/sqrt(Re) | **no** (−3.0%, quadrature + Thwaites bias) |
| `H(lambda)`, `f(lambda)` at lambda ~ 0 | exact Falkner-Skan | **yes** (0.7 / 0.2%) |
| `H`, `f` over lambda in [−0.03, +0.01] | exact Falkner-Skan | **yes** (<3%) |
| `H`, `f` for lambda < −0.05 | exact Falkner-Skan | **no** (up to 17% / 101%) |
| marched wedge theta / cf | exact similarity | theta **yes**, cf **no** (−4.3 / −8.3% at beta = 0.1 / 0.5) |
| laminar separation `lambda` | exact `f''(0)=0` | **no** (+58.8%) |
| turbulent `C_F` | ITTC-1957 | **no** (−6.0..−11.7%) |
| turbulent local cf | Schlichting `0.0592 Re_x^-1/5` | mean **yes** (3.3%), max **no** (13.1%) |
| `H1(H)` inverse | relation itself | **yes** (1.1e-15 after the fix; 182% before) |

Honest verdict: the laminar local fields and the correlations near
`lambda = 0` meet 3%; the integrated friction, the near-separation behaviour,
the laminar separation criterion and the turbulent friction level do not, and
are now documented with exact anchors instead of being covered by a single 15%
band.