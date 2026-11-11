# Reproduction of Armaly *et al.* (1983) backward-facing step flow with TensorFVM

**External anchor.** Armaly, B. F., Durst, F., Pereira, J. C. F. & Schoenung, B. (1983),
*Experimental and theoretical investigation of backward-facing step flow*,
J. Fluid Mech. **127**, 473-496.

**Mission.** m15213 (`REPRO-FVM-ARMALY1983`, round 1). Reproduce two published
quantities with the existing TensorFVM capability, tolerance as predeclared:

| # | item | tolerance | verdict (this report) |
|---|---|---|---|
| 1 | lower-wall reattachment length `L_r/S` at `Re_Dh = 100` and `Re_Dh = 389` vs experiment | < 15 % | **PASS** (11.8 % / 10.3 %) |
| 2 | outlet velocity profile fully developed (parabola deviation) | < 5 % | **PASS at Re = 100** (0.10 %), **FAIL at Re = 389** (10.9 %) with the 12 S outlet; cause = domain length, not solver (see §6) |

---

## 1. Case, geometry and parameters

The reproduction uses the Armaly channel *as built by the existing case factory*
`tensorfvm.inc_bench.cases.backward_step` (step height `S = 1`, inlet channel
`H_in = 1.061 S`, downstream channel `H_out = 2.0625 S`, expansion ratio
`H_out/H_in = 1.9412` versus the paper's 1.9423, **0.058 % geometry error** — the
stair-step mesh reproduces the expansion ratio, it is not assumed), parabolic
inlet with mean velocity `U_in = 1`, no slip on both horizontal walls, zero-gradient
outlet, uniform Cartesian grid.

| parameter | value |
|---|---|
| fluid | incompressible, laminar, constant property |
| `Re_Dh = U_mean · D_h / ν`, `D_h = 2 H_in` (Armaly's basis) | 100 and 389 |
| viscosity `ν = 2 H_in / Re_Dh` | 0.02125 / 0.005462 |
| grid (coarse level) | 243 × 33 cells, 7203 fluid cells, `Δ = S/16` |
| downstream length | 12 S (reattachment) |
| inlet | fully developed parabola over the inlet channel, zero over the step face |
| walls | no slip, `y = 0` (bubble wall) and `y = H_out` |
| outlet | zero-gradient velocity, `p = 0` at the last column |
| solver | `tensorfvm.inc_bench.solver.SteadyFlow` — staggered MAC SIMPLE, first-order
upwind implicit + bounded deferred correction, CG pressure correction |
| solver controls | `α_u = 0.7`, `α_p = 0.3`, inner CG 300 it / 1e-8, outer tol 1e-6 |

`Re_Dh = 389` is the last tabulated Armaly point below the onset of spanwise
(three-dimensional) structure at `Re ≈ 400`, i.e. the highest Reynolds number at
which a two-dimensional solver may legitimately be compared with the experiment.

**Experimental anchors** (`tensorfvm.inc_bench.armaly.ARMALY_1983_EXPERIMENT`,
grade REAL, ±0.2 S digitisation band): `L_r/S = 3.0` at `Re_Dh = 100`,
`L_r/S = 7.9` at `Re_Dh = 389`. They are *not* interpolated — a Reynolds number
without an anchor raises, so a fit can never masquerade as a measurement.

## 2. How it was done

1. `src/tensorfvm/inc_bench/armaly.py` (new, 300 lines) adds only the analysis
   surfaces the reproduction needs — the experimental anchors, the downstream
   development metric, the field snapshot and a one-call run record — and
   drives the **existing** steady SIMPLE kernel; no solver physics was
   re-implemented or forked.
2. `L_r` = first zero crossing of the lower-wall friction `C_f(x)`, linearly
   interpolated between faces, measured from the step plane and normalised by `S`.
3. Development metric: at a column the cell-centred `u(y)` is compared with
   plane Poiseuille flow **carrying the same mass flux** through that column,
   `u_par = 6 U_b η(1-η)`, `η = y/H_wet`. The gate is `max|u-u_par| / max u_par`;
   normalising with the column's own flux means a bulk-velocity mismatch cannot
   be hidden behind a rescaling.
4. Runs: `scripts/repro_armaly1983_bfs.py` (CLI, one JSON per point +
   `armaly1983_matrix.json`), figures `scripts/plot_armaly1983.py`, tests
   `tests/test_armaly1983_reproduction.py` (5 tests, 17 s).
5. Reproduce:

```bash
PYTHONPATH=src python scripts/repro_armaly1983_bfs.py \
    --re 100 389 --levels coarse --downstream 12 \
    --pressure-iterations 300 --threads 4 --out runs/armaly_local
python scripts/plot_armaly1983.py --runs runs/armaly_local \
    --out docs/repro-armaly1983-bfs/figures
python -m pytest tests/test_armaly1983_reproduction.py -q
```

Cluster submission of the refinement/development matrix (levels `coarse`/`medium`,
downstream 12 S and 40 S) is staged in `scripts/_cluster_armaly.sh` and was
queued as job `cj1017` (7 shards); its results are **not** part of this report.

## 3. Results

| `Re_Dh` | grid | iterations (converged) | max \|div u\|·Δ | `L_r/S` computed | experiment | rel. error | gate 15 % |
|---|---|---|---|---|---|---|---|
| 100 | 243×33 (S/16) | 730 | 7.2e-6 | **2.645** | 3.0 ± 0.2 | **11.84 %** | PASS |
| 389 | 243×33 (S/16) | 1550 | 1.5e-3 | **7.086** | 7.9 ± 0.2 | **10.31 %** | PASS |

| `Re_Dh` | outlet at `x = 12 S` | `U_b` | peak `u` | `max\|u-u_par\|/max u_par` | L2 | gate 5 % |
|---|---|---|---|---|---|---|
| 100 | 243×33 | 0.5160 | 0.7741 | **0.098 %** | 0.081 % | PASS |
| 389 | 243×33 | 0.5160 | 0.7741 | **10.91 %** | 8.81 % | **FAIL** |

Development along the channel (max deviation at 0.5 / 0.75 / 1.0 of the domain):

| `Re_Dh` | `x = 6 S` | `x = 9 S` | `x = 12 S` |
|---|---|---|---|
| 100 | 3.28 % | 0.40 % | 0.098 % |
| 389 | 47.5 % | 22.5 % | 10.9 % |

Mass conservation after convergence: `max|div u|·Δ ≤ 1.5e-3` (normalised by the
inlet flux) at `Re = 389` and `7.2e-6` at `Re = 100`. `C_f,min` inside the bubble:
`-5.3e-2` (`Re = 100`) and `-2.35e-2` (`Re = 389`), i.e. the bubble is resolved on
both grids and the zero crossing is a genuine sign change, not an extrapolation.

## 4. Figures

* `figures/fig_reattachment.png` — `L_r/S` versus `Re_Dh`: computed points against
  the two experimental anchors with their ±0.2 S band, errors annotated.
* `figures/fig_profile.png` — outlet `u(y)` against the same-flux Poiseuille
  parabola, per Reynolds number.
* `figures/fig_development.png` — the parabola deviation versus `x/S`, with the
  5 % gate drawn: it shows *why* `Re = 389` fails (the profile is still 11 % away
  at 12 S) and that the decay is monotone.
* `figures/fig_field.png` — streamwise velocity contour with streamlines of the
  converged field (the separation bubble and its reattachment are visible).
* `figures/fig_friction.png` — lower-wall `C_f(x)` with the reattachment star.

## 5. Points to watch (pitfalls found while doing this)

1. **The inner pressure solve must actually converge.** With `pressure_iterations = 60`
   (a 4× speed-up attempt) the coupled iteration *diverges* at `Re = 100`, coarse
   grid: `max|u|` reached `1e132` within 1500 iterations. With 300 inner
   iterations / `1e-8` the same case converges monotonically in 730 iterations.
   Any "fast" configuration of this kernel must be re-validated for stability,
   not just for wall-clock time.
2. **Convergence, not resolution, was the earlier failure mode.** The archived
   `docs/inc-bench/report.md` backward-step entry (`L_r/S = 10.7`, "did not reach a
   steady state") is reproduced here as a *non-converged* artefact: with the
   iteration run to `Δu < 1e-6` the same coarse grid gives 2.645 at `Re = 100`. No
   grid refinement can repair an unconverged SIMPLE iterate.
3. **The reference table in `inc_bench/reference.py` mixes two normalisations.**
   `ARMALY_1983_REATTACHMENT` is documented as "`L_r` in step heights" but its
   numbers (3.9, 4.9, 5.7, 6.1, … and 6.1 at `Re_Dh = 800`) are downstream-channel
   heights (`L_r/H_out`), i.e. a factor 2 convention error, and 3.9 at `Re = 100`
   contradicts the experiment's 3.0. This reproduction therefore did **not** use
   that table; it added an explicit, cited experimental anchor table instead. The
   old table is left untouched (out of scope for this mission) but should be
   corrected or graded WITHHELD.
4. **The 2-D/3-D boundary matters.** Above `Re_Dh ≈ 400` the Armaly experiment is
   three-dimensional, so a 2-D comparison there measures the solver against a
   different flow; `Re = 389` is chosen deliberately as the last 2-D point.
5. **Domain length is part of the acceptance criterion, not a detail.** The
   entrance length of the downstream channel is `L_e ≈ 0.05 Re_{H_out} H_out`
   with `Re_{H_out} = Re_Dh/2`: about 5 S at `Re = 100` and 19 S at `Re = 389`.
   The 12 S domain is therefore *inside* the developing region at `Re = 389` —
   which is exactly what the 10.9 % deviation measures.
6. **Anchor provenance is second-hand.** The two experimental values are the ones
   the 2-D benchmark literature quotes for this experiment; they were not
   re-digitised from the published figure in this offline environment, hence the
   ±0.2 S band. With an 11 % / 10 % computed error and a 7 % digitisation band the
   reattachment verdict is not marginal, but it is not a sub-3 % claim either.

## 6. Error verdict, item by item

| key quantity | computed | reference | error | < 3 % ? |
|---|---|---|---|---|
| `L_r/S`, `Re_Dh = 100` | 2.645 | 3.0 (exp., Armaly 1983) | **+11.84 %** | **No** — passes the 15 % gate |
| `L_r/S`, `Re_Dh = 389` | 7.086 | 7.9 (exp., Armaly 1983) | **-10.31 %** | **No** — passes the 15 % gate |
| outlet parabola deviation, `Re = 100` | 0.098 % | 0 (fully developed) | +0.098 pp | Yes (of the profile itself) |
| outlet parabola deviation, `Re = 389` | 10.91 % | 0 | +10.91 pp | No — **fails the 5 % gate**, cause: 12 S < `L_e ≈ 19 S` |
| expansion ratio of the mesh | 1.9412 | 1.9423 | 0.058 % | Yes |
| mass conservation `max|div u|·Δ`, `Re = 389` | 1.5e-3 | 0 | — | No (flux-normalised residual; acceptable for a converged SIMPLE run) |

There is **no 3 % claim in this report**: the mission tolerance was 15 % for the
reattachment length and 5 % for the profile. Both grids used here are the
*coarsest* level of the existing case factory and the discretisation is first-order
upwind at the wall; a sub-3 % comparison would need the `medium`/`fine` levels,
which are staged on the cluster (`cj1017`) and are not reported here.

Honest statement of what is **not** delivered: (a) grid-refinement evidence
(`coarse` only), (b) the 40 S-outlet run that would decide whether the `Re = 389`
profile gate is reachable, (c) `C_f(x)` comparison against the experimental curve
(no offline digitisation — graded WITHHELD), (d) `Re ≥ 400` behaviour (3-D
experiment, out of scope of a 2-D solver).

## 7. Ecosystem friction (TensorFVM, this mission)

| stage | blocker | evidence | suggestion |
|---|---|---|---|
| solver stability | inner pressure CG budget is a hidden stability parameter: `pressure_iterations=60` diverges (`1e132`) while 300 converges | pilot log, coarse `Re = 100`, `high_order=True` | make the inner tolerance adaptive and expose the achieved inner residual; document the stability floor in `SolverOptions` |
| performance | 0.37 s/iteration at 8 k cells, 1–3.5 s at 34 k cells (pressure CG dominates: 300 → 60 iterations is 4× cheaper per iteration) | `runs/armaly_local/*.json` `seconds`/`iterations` | replace the Jacobi-CG inner solve with a multigrid or AMG preconditioner; 40 S domains are otherwise a 2 h-scale run |
| reference data defect | `reference.ARMALY_1983_REATTACHMENT` is in `L_r/H_out` units while documented and consumed as step heights (factor 2), and its `Re = 100` value (3.9) disagrees with the experiment (3.0) | `src/tensorfvm/inc_bench/reference.py` docstring vs table | re-grade the table WITHHELD or re-express it in step heights with per-point provenance |
| API gap | no experimental anchor object and no downstream-development metric existed; neither `L_r` nor the parabola gate had a citable reference set for `Re = 100`/`389` | had to add `inc_bench/armaly.py` for this mission | fold the anchor table into `reference.py` and keep one convention (step heights) repo-wide |
| workflow | the control plane caps a shell call at 120 s while a single converged anchor run needs 275–577 s; the cluster queue was 231 jobs deep, so the two headline runs had to be run locally | `cj1017` still `queued` after 5 min; local runs 274 s / 577 s | submit early and keep a small default matrix in the repo so a reproduction never blocks on queue depth |
| artifact policy | `artifacts/inc_bench/` sits outside the sparse checkout, so pipeline output is lost | `docs/inc-bench/report.md`, "Artifact snapshot" | keep reproduction output under `docs/<topic>/artifacts/` (as done here) |

## 8. Artifacts and reproduction command

```
src/tensorfvm/inc_bench/armaly.py             analysis surface + experimental anchors
scripts/repro_armaly1983_bfs.py               run matrix -> JSON (one file per point)
scripts/_cluster_armaly.sh                    cluster shard launcher (7 production points)
scripts/plot_armaly1983.py                    the five figures
tests/test_armaly1983_reproduction.py         5 tests (anchors, no interpolation, metric, record)
docs/repro-armaly1983-bfs/artifacts/*.json    per-run records (fields + numbers + solver options)
docs/repro-armaly1983-bfs/figures/*.png       figures quoted in §4
```

Every number in §3 is read out of those JSON records; nothing was transcribed by
hand from the solver output, and nothing was copied from the paper except the two
experimental anchors listed with their provenance and band.