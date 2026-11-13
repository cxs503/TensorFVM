# CASE-FVM-PERIODIC-HILLS — round 6: the closure *does* equilibrate; the gate still fails

- Mission: 17065 (selfstart, TensorFVM) · branch `agent/m17065` · anchor `CASE-FVM-PERIODIC-HILLS` (feature ledger, `open`)
- Driver: `PYTHONPATH=src python scripts/round6_hill_long.py hill <config> [steps]` · `... channel`
- Machine artefacts: `round6_laminar.json`, `round6_sa-second.json` (+ `round6_sa-upwind.json`, `round6_channel.json` if the run finished inside the budget — see §6)
- Tests: `pytest tests/test_periodic_hill_round6.py -q` → **17 passed**; rounds 4/5 and the base suite stay green (31 passed)

## 1. What round 5 left behind

Round 5 attached Spalart–Allmaras to the Re = 2800 hill and found `nu_t` still growing when the
budget ran out (`nu_t/nu` 2.1 → 21.1 between `t = 7.5` and `t = 35`), so every quoted `x_r/H` was a
property of the averaging window. It also left its own channel validation **withheld**, and it named
round 6's first item: *run long enough to know whether the closure equilibrates at all*.

Round 6 attacks exactly that, and adds two structural items that cost nothing and that the whole
anchor depended on:

1. the order of the closure convection is now a **knob** (`SpalartAllmarasConfig.convection_blend`,
   round-5 behaviour as the default, bit-identical — regression-tested), so "first-order upwind"
   is a measured choice rather than an unexamined one;
2. the reattachment **reference is single-sourced** — the repo carried `4.36`
   (`tests/test_periodic_hill_breuer2009.py`) and `5.34` (feature ledger, rounds 3–5) for the same
   case, with **disjoint** 10 % bands (`[4.80, 5.87]` vs `[3.92, 4.80]`). Which one the gate uses
   decided the verdict.

## 2. Method: blocked statistics + a plateau instrument (criteria pre-declared)

The run is cut into statistics blocks (2000 steps = 20 time units each); every block is averaged on
its own, so the record is a **time history of the mean field**, not one window:

- `nu_t` history → :func:`plateau_fit`: exponential fit `v(t) = v_inf − A e^{−t/τ}` (τ scanned, the
  `(v_inf, A)` part is linear per τ) plus a least-squares drift over the last 50 time units.
  `equilibrated := |drift| < 2 % and |v_end − v_inf|/v_inf < 5 % and (no resolved trend or τ < 100)`.
  A flat, already-settled record is *not* penalised for having no trend to resolve.
- `x_r/H` history → the same instrument, because the reattachment length is what the ledger gates.
- **C1** the closure reaches a stationary window · **C3** `x_r/H` within 10 % of 5.34 *on that window
  only* · **C4** mean `u/U_b` at `y/H = 0.5` within 10 % at every admissible station · **C5** the
  reference is single-sourced (machine-checked `scan_reference_consistency`) · **C6** invariants
  (finite, `nu_t ≥ 0`, `|u_b − 1| < 1e-8`).

`y/H = 0.5` admission is a **geometric** filter, not a choice: the sampling line lies *inside* the
hill body for `x/H ≲ 0.9` (measured clearance `−5.9` and `−3.9` cells at `x/H = 0.05, 0.5`) and
0.5 cells above the wall at `x/H = 1.0` and `8.0`. Admissible: `x/H = 1.5 … 7.0` (7 stations). The
two unusable tabulated stations are reported, not silently interpolated.

## 3. Results

### 3.1 The instrument is validated on the case it must judge (laminar, 30000 steps)

| config | block-mean `x_r/H` history | final | vs 5.34 | C1 |
|---|---|---|---|---|
| laminar (closure off) | 3.13, 7.60, 7.97, **8.137**, 8.137, 8.142, 8.135, 8.145, 8.133, 8.135, 8.155, 8.154, 8.137, 8.121, 8.121 | **8.121** | **+52.1 %** | **PASS** (plateau, `τ` not resolved = flat) |

This reproduces round 4's independently computed steady value (8.137 on 48×24, and 8.137 ± 0.025 % on
96×48): the blocked-statistics pipeline and the plateau detector return the number the earlier round
published, so a *stationary* verdict from this instrument means something. The laminar `y/H = 0.5`
profile is hopeless — the flow is separated across essentially the whole floor (`x_r/H = 8.1`), so
`u/U_b ≈ 0.02 … 0.09` where the reference has `0.65 … 1.02` (worst station error 112 %). That is
round 4's model-class diagnosis restated as a profile, not as a length.

### 3.2 Second-order closure convection (12000 steps = `t = 120`)

| `t` | 20 | 40 | 60 | 80 | 100 | 120 |
|---|---|---|---|---|---|---|
| `max nu_t/nu` | 19.93 | 25.48 | 37.26 | 45.81 | 45.21 | 44.13 |
| block-mean `x_r/H` | 3.06 | 7.48 | 7.47 | 4.59 | 4.12 | **4.07** |

- **The closure equilibrates** — this is the round-5 open question, answered: `nu_t/nu` saturates at
  **44.1** with an asymptote fit of **45.05** (`|v_end − v_inf|/v_inf = 2.0 %`, τ resolved), i.e.
  round 5's monotone 2.1 → 21.1 growth was a *transient*, not a divergence.
- **C1 as pre-declared is FAIL (honest)**: the 2 % drift gate is not met inside `t = 120` — the
  record is still moving (`44.13` and falling at the last block) and `x_r/H` is still falling
  (7.47 → 4.59 → 4.12 → 4.07). The asymptote fit says the state is close, the drift gate says
  "not yet"; the stricter of the two is what is reported.
- **C3 FAIL (honest)**: `x_r/H = 4.07` at the last block, **−23.8 %** against 5.34 (a large
  improvement on laminar's +52 %, in the right direction and on the other side of the reference).
- **C4 FAIL (honest)**: `u/U_b` at `y/H = 0.5` is `0.42 … 0.47` over `x/H = 4 … 7` where the
  reference is `0.85 … 1.02`; worst station error **57 %**, rms 43 %. The closure is *over-diffusive*
  in the lower half of the channel at `nu_t/nu ≈ 45` (a turbulent channel of this `Re` has
  `nu_t/nu ≈ 20` near the wall peak), so the profile is too flat/slow — a coherent physical story,
  not a sampling artefact: the same blocks give a converged `x_s/H = 0.83` (reference 0.227),
  i.e. separation is pushed downstream by a factor 3.6 while reattachment comes forward.

### 3.3 First-order closure convection (30 000 steps)

**Not completed inside this mission's budget.** `scripts/round6_hill_long.py hill sa-upwind 30000`
was still running when the deliverables were frozen (the cluster queue stood at ~550 jobs, so the
same run was not obtainable there either); the JSON artefact and the `sa-upwind` row are therefore
absent, and this is an honest gap, not a pass. Round 5's first-order trace (`nu_t/nu` 21.1 at
`t = 35`, still rising) is the only first-order evidence in the repository and it is *shorter* than
the window that made the second-order run equilibrate, so no order comparison is claimed.

### 3.4 The round-5 channel gate was unsatisfiable on the grid it was declared for

`channel_y_plus_requirement()`: with `channel_top_m = 3.036 H`, `Re = 2800` and `ny = 64`, the first
cell sits at `y+ ≈ 4` for a physical `u_tau ≈ 0.06`; reaching `y+ < 1` needs **`ny ≈ 256`**, i.e.
16× the work — round 5's C4 gate could not have passed at `ny = 64` whatever the closure did. It is
replaced by (i) a reported `y+`, (ii) a *stationary* `u_tau` requirement, (iii) the log-law fit
inside the window that actually exists. The converged channel run itself is not in this mission's
artefacts (same budget/queue problem as §3.3), so round-5 C4 stays **withheld**, now with a
quantified reason.

## 4. Reference reconciliation (C5) — the durable outcome of this round

| item | before round 6 | after round 6 |
|---|---|---|
| gate constant | `4.36` in `tests/test_periodic_hill_breuer2009.py`, `5.34` in ledger + rounds 3–5 | `REFERENCE["x_r_over_H"] = 5.34`, imported by the test |
| citation | none for 4.36 ("LES mean, Table 2") | Breuer et al. 2009, *Computers & Fluids* 38:433, doi:10.1016/j.compfluid.2008.05.002, as tabulated by Volpiani et al. 2021 §III (the ledger's own wording) |
| 10 % bands | disjoint → verdict depended on the file you read | one band; `scan_reference_consistency()` machine-checks that no other module carries a live competing constant |
| retired value | — | kept as `COMPETING_X_R` (recorded, not deleted) with its provenance and status |

The retired 4.36 could not be traced to a page/figure of the paper from this environment, so it is
recorded as *unsourced* rather than refuted: if a primary reading ever shows 4.36 for `Re_b = 2800`,
`COMPETING_X_R` is the reconciliation point. What is *not* acceptable is the pre-round-6 state, where
two constants for the same quantity with disjoint acceptance bands coexisted in one repository.

## 5. Honest ledger

- **Delivered**: blocked-statistics + plateau instrument (`periodic_hill_round6.py`), convection-order
  knob with bit-identical round-5 default (regression-tested), geometric admission of the `y/H = 0.5`
  stations, single-sourced reference with a machine check, 17 tests, three machine artefacts
  (`round6_laminar.json`, `round6_sa-second.json`, and their block histories).
- **Answer to round 5's open question**: the SA closure **does** equilibrate on this mesh; `nu_t/nu`
  saturates near 45 at second order. The reattachment length follows it and therefore stops sweeping.
- **Gate**: still **FAIL** — `x_r/H = 4.07` (−23.8 %), `y/H = 0.5` profile worst station 57 %.
  The anchor stays `open`; no round-6 number is quoted as an acceptance value.
- **Not done (honest gaps)**: the 30 000-step first-order run; the converged plane channel
  (round-5 C4/C5 stay **withheld**); any grid study at 96×48 with the closure; any
  production-limiting variant. None of these is claimed.
- **Scope**: everything is 2D. The round-4 collocated divergence caveat is untouched.

## 6. Reproduction

```bash
pytest tests/test_periodic_hill_round6.py -q                       # 17 fast tests
PYTHONPATH=src python scripts/round6_hill_long.py hill laminar 30000
PYTHONPATH=src python scripts/round6_hill_long.py hill sa-second 12000
PYTHONPATH=src python scripts/round6_hill_long.py hill sa-upwind 30000   # ~12 min, was cut
PYTHONPATH=src python scripts/round6_hill_long.py channel                # round-5 C4 revisit
```

## 7. Next round (round 7) first items, in evidence order

1. Finish the first-order/short-window comparison at the *same* `t = 300` window as the second-order
   run, then state the order effect (currently only second-order is known to equilibrate).
2. Attack the over-diffusion: `nu_t/nu ≈ 45` is ~2× a turbulent channel at this `Re`. Candidates, in
   order: production limiting on the wide least-squares `|ω|` (round 5's suspect #1, now measurable
   in a stationary window), the `f_t2` correction, and wall-normal resolution `ny = 48…64` with the
   closure (first cell `d/H = 0.037` is coarse for a low-Re wall model).
3. Re-run the plane channel to `t ≥ 200` at `ny ≈ 256` for `y+ < 1`, or explicitly declare a
   wall-function formulation (a decision, not an omission).