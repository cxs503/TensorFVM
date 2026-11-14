# Board cross-audit — m20059 / CASE-FVM-CONVDIFF-PHASE (TensorFVM)

**Audit mission:** m20109 · **Audited mission:** m20059 (TensorFVM, status `succeeded`)
**Audited commit:** `f2ba8c48` "TensorFVM: convection-diffusion phase-accuracy case
(CASE-FVM-CONVDIFF-PHASE)" (branch `agent/m20059`, **not yet merged to `main`**)
**Audit verdict row:** `tensor_maturity_verdicts` — repo `TensorFVM`,
scope `m20059 CASE-FVM-CONVDIFF-PHASE`, mission_id 20109, verdict_by `board-audit`.
**Declared level: L4** (benchmark verification). **L5 withheld** — the reference is a
textbook closed form evaluated in code, not an experiment or a printed reference table.

This is verification, not re-do. No prose from the audited `REPORT.md` is used as
evidence: every number below is recomputed from the audited bytes, or from an
independent re-implementation of the measurement layer
(`scripts/audit/m20109_convdiff_phase_audit.py`).

---

## 1. What the delivered case is

A one-mode linear convection–diffusion problem `phi_t + c phi_s = D phi_ss` on a
closed annular ring (64 cells around, radius 1, thickness/span 0.05), initial field
`sin(2*pi*i/64)`, marched with the **shipped** kernel
`tensorfvm.moving_mesh.ale_scalar_step` (explicit first-order upwind + point-implicit
diffusion) on the **shipped** `StructuredCurvilinearMesh` with
`periodic=(True, False, False)`. The case measures the **phase error** of the scheme
against the analytic travelling wave `exp(-D k^2 t) sin(k(s - c t))`, plus three
sensitivity axes (Courant, grid, diffusion number).

## 2. Checks performed (all reproducible from this work tree)

| stage | command | result |
|---|---|---|
| inventory | `python scripts/audit/m20109_convdiff_phase_audit.py --inventory` | 11/11 delivered files present, sha256 of every audit byte-copy == audited tree, git blobs recorded (`artifacts/board_audit_m20109/inventory.json`) |
| referee | `--referee` | `tensor_benchmark_cases` rows for the anchor = **0 (EMPTY)**; tolerance in force = mission goal **<3 %**, ref = analytic sine (`referee.json`) |
| tests | `PYTHONPATH=src python -m pytest tests/test_convdiff_phase.py -q` (in a `git archive f2ba8c48` checkout) | **13 passed in 5.82 s** |
| independent recompute | `--recompute` | own ring builder, own `np.linalg.lstsq` phase fit, instrumented kernel wrapper: **200/200** calls to `ale_scalar_step`; per-step lag `1.479386e-05 rad` vs artifact `1.479386e-05 rad` (rel. diff **0.0**), amplitude `0.8346557` vs `0.8346557` (rel. diff 1.1e-15) (`recompute.json`) |
| report | `--report` | 6/6 required sections present; headline numbers quoted in §4 re-derived from `summary.json` (`lag/step` quoted = artifact, **match**); `REPORT.md` regenerates **byte-identically** from `run_case.py`, i.e. the report is program-generated, not hand-edited (`report.json`) |
| repro | `PYTHONPATH=src python docs/CASE-FVM-CONVDIFF-PHASE/run_case.py` then `--compare-repro` | `summary.json` **leaf-identical, n_diffs = 0** (`repro.json`) |

### Numerical verdict item by item (referee = mission-goal tolerance <3 %)

| quantity | reference | got (audit) | error | verdict |
|---|---|---|---|---|
| phase lag / step, N=64, C=1/4, 200 steps | leading-order upwind `C theta^3 (1-C)(1-2C)/6` = 1.478494e-05 rad | 1.479386e-05 rad | 0.06 % | **<3 % PASS** |
| phase error vs analytic wave (as reported, index reference `C*steps`) | 0 | 0.06028 % | — | **<3 % PASS** |
| phase error vs analytic wave (geometric reference `C*steps/(ds/(V/A))`, audit) | 0 | 0.03945 % | — | **<3 % PASS** |
| order of the per-step lag in `dx` (16/32/64/128) | 3 | 3.0398 (audit ladder), 3.0044 (artifact) | 1.3 % | **PASS** |
| phase exact at C=1/2 and C=1 | 0 | 0 (machine zero, both sides) | — | **PASS** |
| amplitude vs modified equation `exp(-0.5 C (1-C) theta^2 n)` | 0.83472 | 0.8346557 | 0.0018 % | **<3 % PASS** |
| amplitude vs pure analytic sine (`D=0`, amplitude must be 1) | 1 | 0.83466 | **16.53 %** | **FAIL — honest, expected for first-order upwind** |
| diffusive branch, a=1e-2 vs continuum analytic | 0 | 24.75 % | **FAIL — documented limitation** (zero-gradient patch closure dilutes the wave speed by `1/(1+w)`), *not* claimed as certified | |

The audited report declares the two FAILs itself (§4 and §6, criteria C6) — that is
consistent with the numbers and is the honest-FAIL discipline working.

## 3. Findings (severity-ranked)

**F1 — medium — no registered referee row.** `tensor_benchmark_cases` has **0** rows
for `anchor_slug='CASE-FVM-CONVDIFF-PHASE'`. The bench-side judge therefore cannot
grade this case at all; grading rests on the mission-goal text (<3 %, ref = analytic
sine). *Repro:* `psql -c "SELECT ... FROM tensor_benchmark_cases WHERE
anchor_slug='CASE-FVM-CONVDIFF-PHASE'"` → 0 rows (`artifacts/board_audit_m20109/
referee.json`). *Remediation:* register one row (metric list + tolerance `<3 %`) so
later audits are machine-judged. Non-blocking for the numbers, which were recomputed
independently here.

**F2 — medium — case not reachable from the case pool.** The module is not registered
in `src/tensorfvm/verification/__main__.py`, nor anywhere else in `main`
(`grep -rn CONVDIFF src scripts` → only the module itself). The only consumers are its
own driver and `tests/test_convdiff_phase.py`. So the "算例池" integration stops at the
branch: the platform cannot enumerate or re-run the case without knowing this path.
*Repro:* `recompute.json`/`report.json` plus the grep above.

**F3 — low/medium — the reported phase error is inflated ~1.5x by a reference-convention
asymmetry.** `phase_metrics` takes the exact advance as `C*steps` **cells**, while the
continuum wave advances `c*dt/ds = C*steps/(ds/(V/A))` cells with `ds/(V/A)=1.0002083`;
the artifact's own criterion C4 *does* use the physical `c*ds` (that is why its
amplitude deviation is 9.7e-5 while the index-consistent value is 1.8e-5). Audit
recompute with the geometric reference gives **0.03945 %** instead of 0.06028 %.
The direction is conservative, so the `<3 %` conclusion is unaffected, but the headline
error is not the physical one. *Repro:* `recompute.json → artifact_vs_audit`
(`lag_relative_audit_index` vs `lag_relative_audit_geometric`).

**F4 — low/medium — REPORT §5 states something its own table contradicts.**
`run_case.py:395` hardcodes "the phase lag is independent of the diffusion number
(a = 0 ... 1e-1 changes the lag by less than 1.4e+02 deg here, i.e. below the measured
lag itself ...)", immediately followed by "But the *phase* is **not** independent of
diffusion on this mesh". The §4 table shows `lag_deg` = 0.1695 → 9.098 → 69.61 →
−144.40: a 500x change, 2000x the a=0 lag. The second sentence is right; the first is
machine-inserted boilerplate left un-updated while the numbers are programmatic.
No PASS/FAIL depends on it, but a skimming reader is misled in exactly the dimension
the case is about.

**F5 — low — a probe value is mislabelled as "measured C".** REPORT §2/§3 quote
"measured `C` = 10.202297" — that is `dt*speed*A/V` for the *unit-probe*
(`speed=dt=1`), i.e. `A/V`, not the run's Courant number; the run and the sweep table
use `C = 0.25`. A Courant number of 10.2 for an explicit upwind scheme is physically
absurd and the label invites exactly that misreading.

**F6 — informational — report six-section completeness and programmatic numbers hold.**
All six required sections exist; §4 rows re-derive from `summary.json`; `REPORT.md`
regenerates byte-identically. The one weakness is prose (F4), not numbers.

**F7 — informational — scope limits, correctly declared.** Only the `a = 0` branch is
certified to <3 %; the diffusive branch is a *characterisation* of the shipped
zero-gradient patch closure (measured `w` = 0.031/0.308/3.08 at a = 1e-3/1e-2/1e-1),
and amplitude accuracy is not certified at all (16.5 % is a scheme property, not a bug).
The case therefore certifies **phase/dispersion**, which is what its title claims.

## 4. Checks not run (unknown ≠ passed)

* GitHub CI on this repo — no `.github/workflows` in the audited tree; the CI signal
  for this branch is **not_run**.
* Full-repo `pytest` (only `tests/test_convdiff_phase.py` was executed; the rest of the
  suite is out of scope for a case audit and the heaviest item here is 6 s).
* Cluster (702 loong64) submission — unnecessary: the whole case runs in seconds.
* Third-party CFD cross-check — not applicable: the reference is analytic.
* The audited branch is **not merged** into `/root/tensor-ecosystem/TensorFVM`, so
  "does it still pass on `main`" is unverified (merge-train item, not an audit defect).

## 5. Independence limitations

The audit reuses the *production* mesh class and kernel (that is the object under
test), but re-implements the geometry node array, the march loop, and the phase
estimator independently (`np.linalg.lstsq` instead of the case's closed-form
projection), and recomputes everything from a fresh `git archive f2ba8c48` checkout in
`/tmp`, not from the audited work tree. Byte copies of the audited artifacts are kept in
`docs/board_audit_m20109/audited/` with sha256 in `inventory.json`, so the verdict can be
re-checked after the audited work tree disappears.

## 6. Disposition

**Approve with notes — L4 confirmed (benchmark verification), L5 withheld.**

Reasoning: real production code path (200/200 instrumented kernel calls), analytic
reference certified in code (PDE residual 8.5e-6), order 3.04 reproduced, two honest
FAILs declared rather than hidden, byte-identical repro, 13 passing gates, six-section
report whose numbers re-derive from artifacts. The blocking-level defects are the two
registry gaps (F1, F2) and two prose defects (F4, F5); none of them changes a number.
Recommended remediation, in order: register the bench row (F1), hook the case into the
case-pool CLI (F2), fix the §5 boilerplate (F4) and relabel the probe `C` (F5), and
decide whether the phase reference should carry the geometric factor (F3).

Figures: `docs/board_audit_m20109/figures/audit_fig1_phase_and_order.png` (Courant sweep
audit vs artifact vs leading-order law; grid-order ladder) and
`audit_fig2_error_budget.png` (error budget against the 3 % tolerance in force).