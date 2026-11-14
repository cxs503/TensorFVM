# CASE-FVM-PERIODIC-HILLS — round 9: the time-averaged gate (and why it settles the case)

- Mission 19916 (selfstart, platform seat) · branch `agent/m19916` · repo TensorFVM
- Anchor: `CASE-FVM-PERIODIC-HILLS` (算例池·周期山丘流分离与再附长度)
- Acceptance: `x_r/H` and `u/U_b` at `y/H = 0.5` vs Breuer et al. (2009), `< 10 %`
- Reference (single-sourced, unchanged): `REFERENCE["x_r_over_H"] = 5.34`,
  `x_s/H = 0.227`, table `BREUER_PROFILE_Y05`, all in
  `src/tensorfvm/verification/periodic_hill_round6.py`
- Driver: `scripts/round9_long_time_average.py` (one configuration per process)
- Analysis: `scripts/round9_analysis.py` → `round9_summary.json` + 5 figures
- Machine artefacts: `round9_<tag>.json` / `.npz` (means field, wall shear, series)

## 1. What rounds 1–8 left open

Rounds 1–8 each produced an *instantaneous* verdict, and every one of them ended
with the same recommendation as item #1: run long enough that the metrics can be
**time-averaged**, because the quantities the ledger compares are means in
Breuer et al. (2009), not snapshots. Round 8's own summary:

| quantity | round 8 observation |
|---|---|
| `x_r/H` at `t = 60` | 5.561 (**4.1 %**, "close") |
| `max nu_t/nu` at `t = 60` | 36.4 and **still rising** (round 8 §2.2) |
| instantaneous `x_r/H` history | 2.62 → 4.07 → 6.09 → nan → 5.56 |
| `u/U_b` at `y/H = 0.5` | **not measured** |
| statistics at Breuer stations | **not implemented** |

So the open question was narrow and answerable with compute: *does the flow reach
a statistically stationary state, and what do the two acceptance metrics equal in
the time mean?* That is what this round measures. Nothing else was changed — no
tolerance tuning, no new closure, no reference edits.

## 2. Method (what was added, and what it is pinned to)

New module `src/tensorfvm/verification/periodic_hill_round9.py`
(20 tests in `tests/test_periodic_hill_round9.py`, all green locally):

| piece | what it does | independent pin |
|---|---|---|
| `zero_crossings_periodic` | separation/reattachment from a wall-shear profile on the periodic grid | reproduces `PeriodicHillSolver.reattachment` to `1e-9` on the same field (`test_zero_crossings_matches_solver_reattachment`) |
| `time_averaged_reattachment` | `x_s/H`, `x_r/H` from the **time-averaged** wall shear | synthetic step-shear with known zeros |
| `horizontal_profile` | `u/U_b` on the line `y/H = 0.5`; columns inside the hill are *dropped*, not extrapolated | analytic Poiseuille on the hill-free mesh (`< 4e-3`, second-order interpolation error); crest columns rejected on the real geometry |
| `block_means`, `stationarity_report`, `averaging_window` | contiguous-block spread + drift of the sampling window; returns the numbers, not just a verdict | constant series accepted, drifting series rejected, non-finite samples rejected |
| `compare_profile` | per-station error vs the Breuer table | delegates to round 6's `profile_error` (same tolerance convention: relative where `|ref| ≥ 0.1`, absolute below — a relative test against `u/U_b = -0.02` is meaningless) |

`Statistics` (merged class) was extended to accumulate the kinematic **wall
shear** on both walls (`mean_wall_shear()`); the `mean()` contract is unchanged.

Numerics per run (unchanged from round 6/8 unless stated): cell-centred FV,
explicit blended convection (`blend = 0.75`), implicit diffusion, periodic
pressure Poisson with wall `p_η = 0`, ERCOFTAC bulk-velocity controller,
Spalart–Allmaras closure, `dt = 0.02`, `Re_b = 2800`, sampling every 20 steps,
`x_r/H` per snapshot plus block-wise `x_r/H` recomputed from the **block-averaged
wall shear** (the metric that matters — instantaneous crossings are ill-defined).

## 3. Results

### 3.1 Completed run: 96×48, 20 000 steps (`t = 400`), burn-in `t = 160`

601 samples, wall clock 1083 s (cluster job `cj2427`, shard 1).

| metric | time mean (`t = 160…400`) | Breuer 2009 | error | `< 10 %`? |
|---|---|---|---|---|
| `x_r/H` (from mean wall shear) | **no crossing** (`None`/nan) | 5.34 | — | **NO** |
| snapshots with a reattachment crossing | **0 / 601 (0.0 %)** | — | — | — |
| `x_s/H` | no crossing | 0.227 | — | NO |
| `u/U_b` at `y/H = 0.5`, `x/H = 1.5` | −0.131 | 0.05 | 0.181 (abs) | NO |
| `x/H = 2` | −0.030 | 0.25 | 112 % | NO |
| `x/H = 3` | 0.060 | 0.65 | 91 % | NO |
| `x/H = 4` | 0.151 | 0.85 | 82 % | NO |
| `x/H = 5` | 0.233 | 0.95 | 76 % | NO |
| `x/H = 6` | 0.282 | 1.00 | 72 % | NO |
| `x/H = 7` | 0.295 | 1.02 | 71 % | NO |
| `max nu_t/nu` (window mean) | 65.5 | ~26 (turbulent channel, round 7) | ≈ 2.5× | NO |
| block-mean `x_r/H` spread | undefined (0/4 blocks have a crossing) | — | — | NO |
| **gate** | **FAIL** (`x_r` undefined, profile 0/7 stations, window not stationary) | | | |

### 3.2 Reading

1. **The time mean has no recirculation closure at all.** Instantaneously the
   bubble sometimes closes (round 8: `x_r/H` 4–6 at single times); averaged over
   601 samples on 96×48 the bottom-wall shear never returns to positive between
   the crest and `x/H = 7`. The near-10 % agreement quoted in round 8 was a
   *transient*, not a converged state — this round settles that question in the
   negative, which is exactly what round 8 asked for.
2. **The closure level is the cause, not the mesh.** `ν_t/ν = 65` is 2.5× the
   turbulent-channel level measured independently in round 7 (26.1), and the mean
   profile at `y/H = 0.5` is 70–112 % low everywhere downstream. Over-diffusion
   flattens the shear layer until the bubble cannot close. This is consistent
   with rounds 5–7 (production limiting changed `ν_t` by 4× but moved `x_r/H` by
   < 2 %, i.e. the level is set by the destruction/`d` scaling, not by the
   production cap).
3. **2-D URANS at `Re = 2800` is the ceiling.** The reference is 3-D DNS/LES of a
   genuinely 3-D flow; a 2-D mean cannot reproduce the spanwise transport that
   feeds the recovery region.
4. **The gate is not "close" under any reading**: 0/7 profile stations inside
   10 %, and the primary metric does not exist. No tolerance interpretation of
   `5.34 ± 10 %` can turn this into a pass.

### 3.3 Status of the other meshes (honest)

`48×24` and `144×72` were submitted as shards 0 and 2 of the same job (`cj2427`)
and were **still running** when this report was written (36 min in, cluster
queue depth ~20–36 jobs). Only the 96×48 shard had completed and been fetched, so
only it is tabulated. The missing shards are a *coverage* gap, not a claim: the
verdict for them is `unknown / not_run`, **not** "passed" and not "failed by
analogy". The driver writes one artefact per shard, so completing them is a
re-run of the same job, not new development.

## 4. Figures

| file | content |
|---|---|
| `fig_round9_xr_timeseries.png` | (a) instantaneous `x_r/H(t)` with the time mean dotted and the `5.34 ± 10 %` band; (b) `max ν_t/ν(t)` on a log axis over the same window |
| `fig_round9_field_mean.png` | time-averaged `u/U_b` field with the hill body — the stalled, non-reattaching mean state |
| `fig_round9_profile_y05.png` | time-averaged `u/U_b` at `y/H = 0.5` vs the Breuer table and its ±10 % band |
| `fig_round9_vertical_profiles.png` | time-averaged vertical profiles at `x/H = 0.5, 1, 2, 3, 4, 6` |
| `fig_round9_stationarity.png` | block means of `x_r/H` over the averaging window (all bars absent: 0/4 blocks have a crossing) |

## 5. Pitfalls / limitations (found the hard way)

1. **Instantaneous crossings are not a metric.** 6 of 10 snapshots in earlier
   rounds had `x_r = nan`; block-averaging the wall shear first and searching for
   the crossing afterwards is both more robust and the physically correct object
   (the reference number is a mean).
2. `admissible_stations` (round 6) rejects `x/H ∈ {0.05, 0.5, 1.0, 8.0}` at
   `y/H = 0.5` because the sampling line is inside the hill body there. Those
   stations are reported as *skipped*, never interpolated.
3. The `y/H = 0.5` metric needs the **absolute** band wherever the reference is
   near zero; two rows (`x/H = 1.5`) are decided that way. This convention is
   round 6's, not a new one.
4. The averaging window is 240 time units, i.e. long compared with the bubble
   dynamics seen in round 8, but the diagnostic that would prove convergence — a
   stationary `ν_t` — is still not reached (`ν_t/ν` plateaus at 65 but the
   profile keeps being wrong), so "non-stationary" here means "non-stationary and
   too diffusive".
5. Local capacity rule: the 96×48 run is 1083 s on the cluster; lengthening it to
   `t = 800`, or the 144×72 case, cannot be done under the 120 s local shell cap.

## 6. Error judgement (explicit)

| key quantity | computed (time mean) | reference | error | `< 3 %`? | `< 10 %`? |
|---|---|---|---|---|---|
| `x_r/H` | undefined (0/4 blocks, 0/601 snapshots) | 5.34 | not computable | **NO** | **NO** |
| `u/U_b` @ `x/H = 2` | −0.030 | 0.25 | 112 % | **NO** | **NO** |
| `u/U_b` @ `x/H = 4` | 0.151 | 0.85 | 82 % | **NO** | **NO** |
| `u/U_b` @ `x/H = 6` | 0.282 | 1.00 | 72 % | **NO** | **NO** |
| `u/U_b` @ `x/H = 7` | 0.295 | 1.02 | 71 % | **NO** | **NO** |
| `max ν_t/ν` (vs channel level 26.1) | 65.5 | 26.1 | 151 % | **NO** | **NO** |

**Verdict: honest FAIL.** Not one acceptance quantity is within 10 %, the primary
one does not exist in the mean field, and the cause is identified (closure level
2.5× too high in 2-D URANS), not papered over. `agent/m19916` must not be read as
a pass for `CASE-FVM-PERIODIC-HILLS`.

## 7. What would actually close this case (next round, scoped)

1. **A different closure**, not more tuning: SST `k-ω` (round 8 §4 item 4) or a
   Reynolds-stress transport model. The SA branch on this geometry is now
   demonstrably saturated at `ν_t/ν ≈ 65` with the bubble open.
2. **3-D, even coarse**: 8–16 spanwise cells would let the mean field develop the
   spanwise transport that the reference LES resolves; this is a solver feature
   (`PeriodicHillSolver` is 2-D by construction) and therefore a new mission, not
   a re-run.
3. **Finish the two missing meshes** of `cj2427` (48×24, 144×72) — pure compute,
   no code change; expected to confirm, not change, the verdict.
4. **`x_s/H` needs a wall-shear definition of separation**: the merged search
   only sees `tau<0 → tau≥0`; at this closure level separation is *not* resolved
   as a crossing either, so `x_s` should be reported as undefined rather than
   `nan`-adjacent. (Documented here, not changed, to keep this round minimal.)

## 8. Reproduce

```bash
# 1. 96x48, t = 400, sampled from t = 160 (1083 s on one cluster core)
PYTHONPATH=src python scripts/round9_long_time_average.py \
    --tag 96x48 --nx 96 --ny 48 --steps 20000 --burn-in 8000 --sample-every 20
# 2. figures + machine summary from the artefacts
PYTHONPATH=src python scripts/round9_analysis.py
# 3. post-processing contract (fast, 17 s)
PYTHONPATH=src python -m pytest tests/test_periodic_hill_round9.py -q
```