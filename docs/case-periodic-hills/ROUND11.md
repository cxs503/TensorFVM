# CASE-FVM-PERIODIC-HILLS — round 11: the SST branch, time-averaged at Re_b = 2800

- Mission 20762 (selfstart, verifier seat) · branch `agent/m20762` · repo TensorFVM
- Anchor: `CASE-FVM-PERIODIC-HILLS` (算例池·周期山丘流分离与再附长度)
- Acceptance: `x_r/H` and `u/U_b` at `y/H = 0.5` vs Breuer et al. (2009), `< 10 %`
- Reference (unchanged, single-sourced in round 6): `REFERENCE["x_r_over_H"] = 5.34`,
  table `BREUER_PROFILE_Y05`
- **Verdict: FAIL, and the round-10 hypothesis is falsified.** Numbers below.

## 1. What was open

| round | what it established | what it left open |
|---|---|---|
| 9 | the **SA** time mean at 96x48 has *no* reattachment (`x_r/H` undefined, 0/7 profile stations, `nu_t/nu = 65.5`) | is the failure a closure-class problem? |
| 10 | the SST `k-omega` closure is *correctly implemented* (algebraic identity, positivity, held `u_b`), and on 48x24 / 300 steps gives `x_r/H = 2.22`, `nu_t/nu = 7.5` — "the right direction" | round 10 §6 item 1: **a longer SST run at the full `Re = 2800`**, time-averaged, on the same budget round 9 used for SA |

Round 10's own words: *"the full Re = 2800 time-averaged acceptance is not run in
this round — it is a compute question, not a model-class one."*  This round is
that compute, and the answer is **no**: the SST branch does not approach the gate
when it is actually run to statistical stationarity.

## 2. Method (what is new, what is deliberately reused)

New module `src/tensorfvm/verification/periodic_hill_round11.py`:

| piece | what it does |
|---|---|
| `build_solver(closure, ...)` | closure selector: `"sst"` → `SSTKOmega` (round 10), `"sa"` → `SpalartAllmaras` (rounds 5–9); a typo raises instead of silently producing an artefact labelled with the wrong closure |
| `run_time_averaged_case(...)` | the round-9 loop (advance, sample, `Statistics`, per-block wall shear) with the closure selectable and a divergence guard (a blow-up is recorded as `diverged_at_step` and still written as an honest FAIL) |
| `gate_summary(payload)` | re-derives the acceptance verdict from a payload's own numbers, with explicit failing reasons; `None`/nan `x_r/H` **fails**, it is never skipped |
| `closure_comparison(payloads)` | cross-closure table; consumes round-9 SA artefacts and round-11 SST artefacts in the same schema |
| `strict_finite` | strict-JSON artefacts (no NaN/Inf), so `NaN != NaN` cannot make a verdict look "changed" |

Reused unchanged — this is what makes the SST verdict comparable to round 9:
every *metric* is round 9's (`time_averaged_reattachment`, `horizontal_profile`,
`compare_profile`, `stationarity_report`, `averaging_window`) and every
*reference* number is round 6's.  No tolerance convention, interpolation rule or
reference value was touched.

Numerics: cell-centred FV, terrain-following mesh (`y = y_b(x) + eta(3.036 - y_b)`),
explicit blended convection (`blend = 0.75`), implicit diffusion, periodic pressure
Poisson with wall `p_eta = 0`, ERCOFTAC bulk-velocity controller (`u_b = 1`
enforced to 1e-6, so `Re` is a grid-independent input), `dt = 0.02`,
`Re_b = u_b h / nu = 2800`, sampling every 20 steps after `t = 160`, window
`t = 160…400` (601 samples).

SST settings are round 10's defaults (`a1 = 0.31`, `beta* = 0.09`,
`sigma_k = 0.85`, `sigma_w = 0.5`, `gamma = 5/9`, freestream `I = 0.05`,
`nu_t/nu` cap `1e4`, no production limiter, first-order transport upwind).

Run: cluster job `cj2913`, shard 1 (96x48), 20 000 steps, wall clock 2 056 s;
shards 0 (48x24) and 2 (144x72) were still in flight when this was written
→ reported as **`unknown / not_run`**, not as passed and not as failed by analogy.

## 3. Result — the time-averaged gate

96x48, `t = 160…400`, 601 samples, SST:

| metric | round 11 SST time mean | round 9 SA time mean (same mesh/budget) | Breuer 2009 | SST error | `< 10 %`? |
|---|---|---|---|---|---|
| `x_r/H` (from the mean wall shear) | **2.046** | *undefined* (0/601 samples close the bubble) | 5.34 | **61.7 %** | **NO** |
| `x_s/H` | *undefined* (no separation crossing in the mean) | *undefined* | 0.227 | — | NO |
| `u/U_b`, `x/H = 1.5` | 0.0372 | −0.131 | 0.05 | 0.0128 (abs band) | YES |
| `u/U_b`, `x/H = 2.0` | 0.175 | −0.030 | 0.25 | 30.0 % | NO |
| `u/U_b`, `x/H = 3.0` | 0.371 | 0.060 | 0.65 | 43.0 % | NO |
| `u/U_b`, `x/H = 4.0` | 0.514 | 0.151 | 0.85 | 39.6 % | NO |
| `u/U_b`, `x/H = 5.0` | 0.595 | 0.233 | 0.95 | 37.4 % | NO |
| `u/U_b`, `x/H = 6.0` | 0.618 | 0.282 | 1.00 | 38.2 % | NO |
| `u/U_b`, `x/H = 7.0` | 0.570 | 0.295 | 1.02 | 44.1 % | NO |
| profile stations inside 10 % | **1 / 7** | 0 / 7 | — | — | **NO** |
| `max nu_t/nu` (window mean) | **342.5** | 65.5 | ~26 (channel, round 7) | 13x | NO |
| block spread of `x_r/H` | `6.9e-08` | undefined | — | — | stationary ✓ |
| `x_r/H` coverage | 601/601 samples | 0/601 | — | — | — |
| **gate** | **FAIL** | FAIL | | | |

## 4. The substantive finding: the SST branch is *frozen*, not turbulent

The stationarity diagnostics are the interesting part of this table.  Over the
whole 240-time-unit window, both `x_r/H` and `max nu_t/nu` are constant to
**`7e-8` relative** (block means `2.04562579 / 2.04562565 / 2.04562565 /
2.04562565`; `nu_t/nu = 342.475` in every block).  A separated flow at
`Re = 2800` is *unsteady*; a metric that is constant to machine-level precision
over 240 time units is not "well converged statistics", it is a **steady state
that the time stepping has locked into**, `d/dt -> 0`.

The physical reading is consistent and quantitative: with `nu_t/nu = 342`, the
effective Reynolds number of the mean flow is `Re / (1 + nu_t/nu) = 2800 / 343 ~ 8`.
At `Re_eff ~ 8` the hill bubble is short and steady — which is exactly what is
measured (`x_r/H = 2.05`, a reattaching but far too short bubble, and a
`y/H = 0.5` profile that is everywhere 30–44 % low).  The SA branch behaves the
same way at a lower diffusion level (`nu_t/nu = 65`, `Re_eff ~ 42`) and never
reattaches at all.  Both branches are therefore on the *same monotone curve*:
more closure diffusion → shorter bubble, from "never closes" (SA, `nu_t/nu = 65`)
through `x_r/H = 2.05` (SST, `nu_t/nu = 342`).  Neither is near 5.34.

**Round 10's hypothesis is falsified.** Round 10 saw `x_r/H = 2.22`,
`nu_t/nu = 7.5` after 300 steps on 48x24 and concluded that "the next step toward
the 10 % gate is a *longer* SST run at the full `Re = 2800`, not a re-tuning of
the SA branch".  Running it longer does not move `x_r/H` toward the reference:
between `t = 160` and `t = 400` it is pinned at `2.046` while `nu_t/nu` grows to
`342`.  The 48x24 / 300-step number was a **transient of the closure's own
spin-up** (freestream seed `nu_t/nu = 3`), not evidence about the converged
state.  The missing ingredient is therefore in the closure *physics/numerics*
(production–destruction balance, wall treatment of `omega`, transport upwinding),
not in run length.

## 5. Figures

| file | content |
|---|---|
| `fig_round11_profile_y05.png` | time-averaged `u/U_b` on `y/H = 0.5`, SST vs SA vs the Breuer table and its ±10 % band |
| `fig_round11_xr_timeseries.png` | instantaneous `x_r/H(t)` (top) and `max nu_t/nu(t)` (bottom, log) for both closures at 96x48; the SST curves are flat lines — the frozen state |
| `fig_round11_nu_levels.png` | window-mean `max nu_t/nu` per closure vs the round-7 turbulent-channel level (26.1) |
| `fig_round11_field_sst_96x48.png` | time-averaged `u/U_b` field, SST 96x48, hill body drawn |
| `fig_round11_field_sa_96x48.png` | same for the SA baseline (round 9 artefact) |

## 6. Error judgement (explicit)

| quantity | value | reference | error | `< 3 %`? |
|---|---|---|---|---|
| `x_r/H` | 2.046 | 5.34 | **61.7 %** | NO |
| `u/U_b` at `y/H = 0.5`, 6 of 7 admissible stations | 0.175 … 0.618 | 0.25 … 1.02 | **30–44 %** | NO |
| `u/U_b` at `x/H = 1.5` | 0.0372 | 0.05 | 0.0128 absolute | NO (abs band 10 %) |
| `nu_t/nu` level | 342.5 | ~26 (channel) | 13x | NO |

**Acceptance gate: FAIL — 61.7 % on the primary metric, 1/7 profile stations
inside 10 %.**  This is an honest FAIL, not a tolerance question: no reading of
`5.34 ± 10 %` reaches 2.046.

What *is* verified to `< 3 %` in this round (the parts that make the FAIL
trustworthy):

| check | result | `< 3 %`? |
|---|---|---|
| round-11 loop reproduces the round-9 driver on identical input (SA, 16x8, 60 steps): `x_r/H`, `x_s/H`, profile, samples, coverage | bit-identical (`rtol 1e-12`) | YES |
| round-11 gate re-derives round 9's *recorded* SA verdict from `round9_96x48.json` | identical (`pass=False`, 0/7, coverage 0.0) | YES |
| bulk velocity held at `u_b = 1` (controller) | `1e-6` per sample | YES |
| artefacts are strict JSON (no NaN/Inf), non-finite → `null` | enforced by test | YES |
| gate can only pass with `x_r/H`, all profile stations *and* a stationary window; band is inclusive at the 10 % edge | unit-tested incl. boundary | YES |
| diverged run still writes an analysable artefact (`diverged_at_step`) | unit-tested | YES |

## 7. Pitfalls / limitations

1. **A stationary diagnostic is not a converged solution.** `stationarity_report`
   says `ok: True` for the SST run, and it is *right* — the statistics are
   stationary.  But a steady, frozen field satisfies that test trivially.  The
   discriminating check is the *level*: `nu_t/nu = 342` is 13x the independent
   channel level (round 7), so "stationary" here means "stationary and far too
   diffusive".  Any future gate must keep both.
2. **Short-budget closure spin-up can look like progress.** Round 10's 48x24 /
   300-step `x_r/H = 2.22` and this round's converged `x_r/H = 2.046` agree to
   8 %, which is *not* corroboration — the closure starts from a freestream seed
   and the first thing it does is lock onto that steady state.
3. `x_s/H` is undefined in the mean for both closures: the mean wall shear is
   already positive at the first admissible station downstream of the crest, so
   there is no separation crossing to find.  Reported as `null`, not
   interpolated.
4. 20 of the `y/H = 0.5` sampling columns are inside the hill body
   (`admissible_stations`, round 6).  They are dropped, never extrapolated.
5. Shards 0 (48x24) and 2 (144x72) of `cj2913` had not finished
   (`unknown / not_run`); only 96x48 is tabulated.  The 144x72 shard is the
   mesh-refinement question and is *not* answered here.
6. 2-D URANS vs 3-D DNS/LES: the reference flow is genuinely 3-D.  That ceiling
   was already noted in round 9 and is not changed by this round — but it is not
   the binding constraint either, because the SST solution is not even unsteady.

## 8. Reproduce

```bash
# unit tests (12, ~7 s)
PYTHONPATH=src python -m pytest tests/test_periodic_hill_round11.py -q
# periodic-hill regression (32, ~40 s)
PYTHONPATH=src python -m pytest tests/test_periodic_hill.py \
    tests/test_periodic_hill_round9.py tests/test_periodic_hill_round10.py -q
# the run (cluster; ~2056 s for 96x48 on one core)
PYTHONPATH=src python scripts/round11_sst_long_average.py --closure sst --case-index 1 --case-count 3
# cross-closure table + figures from the committed artefacts
PYTHONPATH=src python scripts/round11_analysis.py
```

## 9. Next (scoped, from the diagnosis — not another run of the same thing)

1. **Balance the SST source terms in this configuration**: with the flow frozen,
   read `production / k` vs `beta* omega` per cell from `SSTKOmega.last` /
   `source_terms()`.  A steady state at `nu_t/nu = 342` on a 96x48 mesh implies
   production is not being limited where the strain is largest (the
   `min(P_k, 10 beta* k omega)` limiter of SST-2003 is available through
   `production_limit` but defaults to `None`, i.e. **off** in round 10/11 — this
   is the first thing to test).
2. **Wall `omega` treatment**: the round-10 closure uses the algebraic
   `omega_wall ~ 60 nu / (beta1 d1^2)`-family value only implicitly; check the
   near-wall cell value against it.  An `omega` that is too low at the wall
   inflates `nu_t` exactly as observed.
3. Only after 1–2: the 144x72 mesh (shard 2 of `cj2913`) becomes informative; on
   the current closure it would refine a wrong state.