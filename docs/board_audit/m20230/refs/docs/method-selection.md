# CASE-CAVITY-RE1000-FVM — round 2: survey, method selection, migration plan

Mission m15458 (round 2 of the two-round protocol for anchor `CASE-CAVITY-RE1000-FVM`).
Round 1 (m15369) ended `budget_exhausted` with no converged row: it spent its 80
tool calls writing two runner scripts and never got a number out. This note is the
round-2 step-①②③ deliverable: what is out there, what is the standard way, what we
adopt here and why.

---

## 1. What the anchor actually demands

| item | value | source |
|---|---|---|
| case | 2-D lid-driven cavity, unit square, one moving wall | Ghia, Ghia & Shin (1982) §2 |
| Re | 1000 (= U·L/ν with U = 1, L = 1 ⇒ ν = 1e-3) | Ghia et al. (1982) Table II |
| reference | u along x = 0.5 (Table I), v along y = 0.5 (Table II) | Ghia et al. (1982) |
| tolerance | < 5 % on the key quantities (case-pool ledger) | `tensor_benchmark_cases` row `CASE-CAVITY-RE1000-FVM` |
| extra evidence required | ≥ 2 resolutions / 2 parameter points, convergence | case-pool report spec §⑤ |

The reference numbers are **not** transcribed into any report by hand: the committed
copy lives in `src/tensorfvm/lid_cavity_reference.py` (byte-for-byte transcription of
Tables I–II, with a `self_check()` and a dispute register) and every table in the
deliverable is injected from `artifacts/CASE-CAVITY-RE1000-FVM/summary.json`, which
is written by the runner from the same module.

## 2. Survey ① — open-source implementations (what the cache actually offers)

`SELECT repo,cache_nodes FROM oss_inventory WHERE cache_nodes IS NOT NULL` is not
readable by this agent's role (`oss_inventory`/`oss_solutions` are `permission denied`;
only `\d` catalogue access works). The node-side cache that *is* reachable from this
container is `/root/oss-cache`, and it holds three repositories:

| repo | family | useful for a Re = 1000 cavity? |
|---|---|---|
| `Maarten-vd-Sande_lbm` | LBM (D2Q9) | no — different discretisation, no FVM projection solver |
| `pylbm` | LBM | no |
| `hschwane/GraSPH` | SPH | no |

So the horizontal survey returns **nothing reusable for a collocated/staggered FVM
cavity**. The cluster-side cache (`~/oss-cache/<owner>__<repo>`, 30 repos) is the
place where an FVM counterpart would live; a grep job was queued (`cj1123`, `queued`,
324 jobs ahead at submission) and had not been dispatched inside this mission's wall
clock — recorded as a scope limitation, not glossed over (§8). The method survey
therefore rests on the literature (§3) plus this repo's own machinery, which is what
the case-pool rule asks for anyway ("跑不动=生态病记 friction，不另起炉灶").

## 3. Survey ② — method genealogy for this benchmark

| method | how it handles Re = 1000 | cost to reach a steady state | verdict for us |
|---|---|---|---|
| vorticity–streamfunction + coupled strongly implicit multigrid (**Ghia et al. 1982** themselves; 129² and 257² grids) | removes the pressure problem, solves the biharmonic-ish system implicitly | a few multigrid V-cycles per Re | would need a new solver; the repo's asset is a *pressure-projection* FV solver, and the reference values must stay comparable, not the method |
| SIMPLE / SIMPLEC (Patankar & Spalding 1972; Perić) with a pressure-correction outer loop | implicit momentum, under-relaxed | O(10³) outer iterations, nearly grid-independent | robust and standard, but a full rewrite of the time loop |
| pseudo-transient continuation with explicit projection (what TensorFVM has) | explicit CFL ≈ 0.15·h/U | steps ∝ n (information must cross the box at U) ⇒ ~10⁵ steps at n = 128 | **adopted**, with the cost cut by continuation (§4) |
| Reynolds/grid continuation (Erturk 2008, *Int. J. Numer. Methods Fluids* **48** 747–774, uses continuation to Re = 21 000; standard in pseudo-time codes) | start from a nearby converged state | same steady state, a fraction of the steps | **adopted** — and *checked*, see §5 |
| full-approximation-scheme multigrid acceleration | asymptotically grid-independent convergence | best known | out of scope this round (needs a coarse-grid operator for the staggered projection scheme); recorded as the next lever in §7 |

## 4. Selection and why

Adopted for round 2:

1. **`LidCavitySolver` as-is** (staggered MAC, conservative van Leer face fluxes,
   explicit projection, DCT-II pure-Neumann pressure solve). No solver change: the
   round-1 debt was *not* an algorithmic defect, it was "no converged Re = 1000 row at
   more than one resolution".
2. **Grid continuation** `n = 32 → 64 → 128`: converge coarse, bilinearly prolong the
   cell-centred velocity onto the fine grid (face fields rebuilt by averaging, wall
   faces zeroed), keep marching. The steady state of the fine *discrete* system is
   independent of the initial guess, so this is a cost optimisation only.
3. **Continuation is verified, not trusted**: `n = 64` is also run **cold**, and the
   two converged states are compared field-wise (`max|Δu|`, `max|Δv|`). A continuation
   bug shows up there as a non-zero difference.
4. **Per-case checkpoints** (`fields/Re1000_n*.pt`, every 20 000 steps + at the end of
   each case): round 1 lost its whole budget to a run whose intermediate state was
   never written.
5. **Two resolutions** (n = 64, n = 128) both converged to `max|Δu|/Δt ≤ 1e-7`, plus
   n = 32 as a documented under-resolved control.

Kept from the shipped configuration on purpose (so the numbers stay comparable with the
committed ladder): `cfl = 0.15`, `tolerance = 1e-7` measured every 250 steps,
`max_steps = 2.5e5`, float64, single process.

## 5. What is new relative to `docs/verification-lid-cavity-ladder/`

The ladder contains exactly one Re = 1000 row (n = 64, 2.80 %/2.34 %) and states in its
own §5.2 that the n = 128 attempt was **not converged** when it was stopped (change rate
8.3e-6 at step 66 750 against a cap of 2e5) and was therefore excluded. Round 2 fills
that hole: n = 128 converged, with a fresh n = 64 row produced by the same runner so the
two rows share one code path, one timestamp and one tolerance.

## 6. Migration plan (how this becomes a reusable capability)

| step | artefact | status |
|---|---|---|
| runner with continuation + checkpoints | `scripts/run_cavity_re1000_m15458.py` | done |
| machine-readable evidence | `artifacts/CASE-CAVITY-RE1000-FVM/summary.json` | done |
| 5 % ledger verdict per resolution | `judge_5pct()` in the runner ⇒ `verdict_5pct` | done |
| regression test that re-derives the verdict from the committed JSON | `tests/test_case_cavity_re1000_m15458.py` | done |
| report (six sections) | `docs/CASE-CAVITY-RE1000-FVM/report.md` | done |
| `oss_solutions` row | `artifacts/CASE-CAVITY-RE1000-FVM/oss_solutions_insert.sql` | emitted (the table is not writable by this role; the platform applies it) |

## 7. Levers left for round 3 (measured statements, not promises)

* **SSP-RK2/RK3 instead of forward Euler** would lift the stable Courant number from
  0.15 towards 0.4–0.8 (the ladder documents forward Euler going unstable at 0.4 on the
  lid shear layer); at 2 stages this is a 1.5–2.5× saving.
* **FAS multigrid in pseudo-time** is the only lever that makes the step count
  grid-independent; it needs a coarse-grid correction for the staggered projection pair.
* **Reynolds continuation** 400 → 1000 would save the spin-up from rest entirely (the
  Re = 400 converged field is already in the committed ladder).

## 8. Limitations recorded up front

* The cluster-side open-source cache and `oss_inventory`/`oss_solutions` are not
  readable from this container (permission denied); the horizontal survey is therefore
  limited to the three local repos listed in §2.
* The `cj1123` cluster job (the same runner, loong64 nodes) stayed `queued` for the
  whole mission; the results below were produced on the container CPU instead. That is
  the documented fallback, and it is stated in the report rather than hidden.
* Steady state only. Ghia's Re = 1000 solution is steady, so this is not a restriction
  for the anchor, but nothing here says anything about unsteady cavity flow.