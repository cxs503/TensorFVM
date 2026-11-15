# Board audit — m21018 `GAP-DIFFLBM-EXT-LIT-CYL` (TensorFVM, cylinder Re = 100)

Auditor: board-audit · audit mission **m21105** · subject mission **m21018**
(`agent/m21018`, merged as **PR #719**, merge-master commit `cda4e7f9`, subject
commit `69311e24`) · date 2026-11-15 · verdict **L2, honest FAIL, benchmark
`pending`** · disposition **request changes (rework), GAP stays open**.

This is a *closure check*, not a re-do: the subject's own numbers, artifacts and
tests were re-examined and, where cheap, re-executed. The referee is
`tensor_benchmark_cases`, not the delivered report.

---

## 1. What the mission delivered (physical problem & geometry)

Two-dimensional incompressible flow past an unbounded circular cylinder at
Re = U D/ν = 100 (laminar von Kármán shedding) — JOUR-DIFFLBM literature anchor
3 (FVM side). Reference quantities and bands, as registered both in
`docs/case-cylinder-re100/reference/reference.json` and in the platform table:

| metric | reference | band | source |
|---|---|---|---|
| C_d,mean | 1.340 | ±0.07 (5 %) | Braza et al., JFM 165 (1986) 79–130, doi:10.1017/S0022112086003014 |
| St | 0.164 | ±0.005 (3 %) | Schäfer & Turek, NNFM 52 (1996), doi:10.1007/978-3-322-89849-4_39 |
| C_l amplitude | 0.32 | informational | Braza 1986 |

Domain 10 D upstream / 20 D downstream / ±10 D laterally (blockage 5 %), D = 1,
U = 1, ν = 0.01, Δt = 0.02, t_end = 120 tU/D.

## 2. Method as delivered (and what the audit actually checked)

* `tensorfvm.immersed` masked-MAC fractional step (pre-existing, mission m9040)
  with a staircase cylinder mask from `immersed/geometry.circular_cylinder`.
* **New in m21018**: a control-volume momentum budget for the hydrodynamic force
  (`src/tensorfvm/immersed/forces.py`: `ControlBox`, `budget_terms`,
  `ForceRecorder`) and gated statistics (`verification/cylinder_re100.py:164`
  `shedding_statistics`: half-amplitude gate, zero-crossing St + independent FFT
  cross-check).
* Case harness + CLI `python -m tensorfvm.verification.cylinder_re100`.
* Grid ladder 12 / 20 / 32 cells per D (360×240, 600×400, 960×640), 6001 steps
  each, rendered into `docs/case-cylinder-re100/{README.md,ladder.json}`.

Audit checks actually performed (log: `repro_log.md`): CLI `--help` (R1);
end-to-end run at 8 cells/D, 1200 steps, 24.4 s (R2); bit-for-bit match of the
subject 32 cells/D first force sample, `1552.09585999` (R3); divergence forensics
on the delivered raw history (R4); the two test modules re-run — 16 passed in
24.4 s (R5); the referee table queried directly in postgres (R6).

## 3. Results — the referee table decides

`SELECT case_name,metric,tolerance … WHERE anchor_slug='GAP-DIFFLBM-EXT-LIT-CYL'`
→ `cd_mean 5 %`, `st 3 %`, repo TensorFVM, **status = pending** (never passed).

| grid (cells/D) | steps | C_d,mean | err vs 1.34 | St | err vs 0.164 | C_l amp | referee |
|---|---|---|---|---|---|---|---|
| 12 | 6001 | 1.5193 | **+13.38 %** (>5 %) | 0.1441 | **−12.11 %** (>3 %) | 0.113 | FAIL |
| 20 | 6001 | 1.5093 | **+12.63 %** (>5 %) | 0.1551 | **−5.44 %** (>3 %) | 0.201 | FAIL |
| 32 | 6001 | **nan** | – | not reported | – | nan | **run diverged at step ≈15** |

Successive grid differences as delivered: 12→20 `ΔC_d = −0.67 %`, `ΔSt =
+7.05 %`; 20→32 **NaN** — the ladder therefore contains *no* mesh-convergence
evidence, and the reported "finest grid verdict" is not a verdict about physics.

Figures: `figures/fig1_cpd32_divergence.png` (force budget blowing up),
`figures/fig2_ladder_errors.png` (errors vs the registered bands),
`figures/fig3_cpd20_signal.png` (the 20 cells/D shedding record the numbers come
from — this part is physical: 10 clean periods, zero-crossing and FFT St agree
to 2.6e-4).

## 4. Findings, severity-ranked

**F1 — blocking (correctness of the conclusion): a numerical blow-up is reported
as a physics gate.** `verification/cylinder_re100.py:200`
(`if not np.isfinite(amp_half) or amp_half < cl_gate`) funnels a non-finite lift
record into the same message as a quiet cylinder: `no vortex shedding: lift
half-amplitude nan < gate 0.05`. In the delivered 32 cells/D history the drag
budget grows `1.3e26 → 1.8e171 → 6.8e309 → 6.3e315 → nan` between `t = 0.20` and
`t = 0.32` (5986/6001 rows non-finite). `docs/case-cylinder-re100/README.md`,
`ladder.json` and the per-grid `report.md` all present that as
`GATED: no vortex shedding`, and the ladder prints `20 -> 32 … +nan %` as if it
were a grid difference. Consequence: the divergence is invisible to a reader who
does not open the CSV, and ~2639 s of cluster time were spent stepping a NaN
field instead of failing fast.

**F2 — blocking (capability): the anchor is not established.** Both registered
metrics miss tolerance on every grid that shed (best: C_d +12.63 % vs 5 %,
St −5.44 % vs 3 %); C_l amplitude is −37 % low, consistent with a staircase body
at 12–20 cells/D. `tensor_benchmark_cases.status` is still `pending`. Staircase
immersed at these resolutions is simply not competitive with body-fitted
literature values; the gap must either buy resolution (≥40 cells/D, likely
expensive) or move the production path to the body-fitted backend.

**F3 — major (reproducibility): the run artifacts were never committed.**
`runs/` and `artifacts/` are untracked in the subject worktree (`git status` →
`?? runs/ ?? artifacts/`), so `git ls-tree -r HEAD | grep '^runs/'` is empty:
the raw force histories that back the ladder exist only on a worktree that will
be pruned. Only the derived `docs/case-cylinder-re100/{README.md,ladder.json}`
made it into git. Remedy applied by this audit: the compact referee payloads are
copied here with a sha256 manifest (`artifacts/manifest.json`).

**F4 — major (test gap): the failure mode is untested.** The 16 tests assert good
invariants (force control-volume independence, mask-area convergence, St
recovery from a synthetic signal, St withheld when nothing sheds) but none feeds
a non-finite history, which is the only failure that occurred in production.
A regression test with a NaN tail would have caught F1.

**F5 — minor (report shape): the per-grid report has 5 sections, not 6.** It
carries problem/anchor/results enthus/gates/figures and prints explicit FAIL
flags, but has no limitations section and no explicit "is the error < 3 %?"
sentence — the honest-FAIL information is present, the reflexive summary is not.

**F6 — informational (integration): no platform path.** `backend_registry.py`
has no immersed entry and no `/calculations/*` route exposes the backend; the new
modules are called only by the case harness and the tests. This is what keeps the
delivery at L2, and it is the cheapest thing to fix next (wire `ForceRecorder`
into the production immersed solver's diagnostics and expose the case through the
platform benchmark route).

**Overstated claims:** none material — the delivery says FAIL and reports the
numbers straight. The one softened claim is the label `GATED: no vortex shedding`
for a diverged run (F1); it is a code defect rather than a dressed-up number.

## 5. Points to watch

* St from zero crossings and from the FFT agree to 2.6e-4 at 20 cells/D — the
  frequency estimator is sound; the error is in the force level, not the
  statistics.
* The statistics window (last 10 shedding periods) is legitimate; the shedding
  detection gate is a genuine improvement over the predecessor module (m20955),
  which read St = 0.267 out of 1e-7 noise. The fix needed is a *non-finite*
  branch, not removing the gate.
* Domain blockage (5 %) and the staircase mask both bias C_d upward; a
  body-fitted cross-check at 20 D × 20 D would separate the two error sources.
* Local re-runs of the 32 cells/D grid are not affordable in the audit sandbox
  (>100 s/step observed); the divergence stands on the delivered artifact plus
  the bit-for-bit agreement of step 1.

## 6. Error decision (explicit)

| quantity | got (best grid, 20 cells/D) | reference | error | <3 %? |
|---|---|---|---|---|
| C_d,mean | 1.5093 | 1.3400 | **+12.63 %** (band 5 %) | **no** |
| St | 0.1551 | 0.1640 | **−5.44 %** (band 3 %) | **no** |
| C_l amplitude | 0.2005 | 0.3200 | −37.3 % | no (informational) |

The 3 % question is answered **no** for both registered metrics. Ladder
convergence cannot be assessed because the finest grid diverged.

## 7. Grading and disposition

* **L2 — independent functional implementation.** Real code, input → compute →
  output, entry point executed by the audit, non-empty run artifacts, invariant
  tests. Evidence pointers landed in `tensor_maturity_verdicts`
  (`scope='m21018 GAP-DIFFLBM-EXT-LIT-CYL'`, `mission_id=21105`,
  `verdict_by='board-audit'`), `maturity_gate.judge` replayed by the auditor
  before landing → `gate_level=2`, no missing keys.
* **Not L3**: the new force/statistics modules have no main-solver or platform
  caller (F6).
* **Not L4**: both registered metrics are outside tolerance and there is no
  convergence evidence (F2, plus the diverged finest grid).
* **Disposition: request changes.** Keep `GAP-DIFFLBM-EXT-LIT-CYL` open. Rework
  order of value: (1) split `diverged` from `no shedding` in the gate and fail
  fast on non-finite fields (F1); (2) commit the raw histories or make the ladder
  write them under `docs/` (F3); (3) NaN regression test (F4); (4) resolution or
  body-fitted comparison to get inside 5 %/3 % (F2).

## Files

```
docs/audit/board-audit-m21018/
├── README.md              this report
├── repro_log.md           R1-R7 commands, outputs, what was NOT run
├── findings.json          machine-readable, severity-ranked
├── verdict.json           the landed evidence payload
├── land_verdict.py        idempotent INSERT + maturity_gate replay (--judge)
├── collect_evidence.py    copies the raw artifacts + sha256 manifest
├── make_figures.py        figures 1-3 (all numbers read from artifacts)
├── artifacts/             cpd12/20/32 result|benchmark|errors|report + raw histories
└── figures/               fig1-3 png
```

## Independence limitations

The audit is run by the same platform that produced the delivery; it did not
re-derive the 12/20 cells/D runs from scratch (the 12 cells/D rerun was submitted
as compute job **cj3192** and was still queued at hand-off — recorded as *not
run*, not as passed). Cross-checks that *were* independent: referee-table query,
test rerun, 8 cells/D end-to-end run, step-1 bit-for-bit match, CSV forensics.