# PR #82 (`agent/m3240`) merge into `main` — conflict resolution, loss audit, test evidence

Unit m3948 · repo **TensorFVM** · branch `agent/m3948` → pushed back to `agent/m3240`
Base `main` = `db97955` ("Merge #78 (m3149) via merge-governor"), PR head = `99f8e6e`
Merge base (true three-way base) = `0e9bd51` ("Merge #49 (m2766) via merge-governor")

## 1. What the PR and the base actually are

PR #82 is a *reference-data* claim, not a solver feature. Unit m3240 proposed that the
lid-driven cavity benchmark table of Ghia, Ghia & Shin (1982) is mis-transcribed at one
point — the 10 %-of-height v-centreline extraction at Re = 400, x = 0.9063 — and that
the published value must be replaced:

| item | value | provenance |
|---|---|---|
| geometry | lid-driven cavity, unit square, lid u = 1, no-slip walls | `lid_cavity.py` |
| Re | 400 (also 100 / 1000 in the ladder) | `summary.json["meta"]` |
| grids | n = 32 / 64 / 128 (cell-centred, uniform) | `summary.json["meta"]["grids"]` |
| printed entry v(Re=400, x=0.9063) | **-0.23827** | Ghia et al. (1982), Table II |
| proposed entry | **-0.38760** | m3240, `CORRECTIONS` register, `applied=False` |
| solver Richardson limit at that point | -0.38989 (observed order 2.00) | `DISPUTES` / ladder bundle |
| PR files touched | 7 (2 code/test, 1 bundle, 4 new artefacts) | `git diff 0e9bd51 99f8e6e` |

`main` had meanwhile taken the *opposite* invariant (unit m3099): a published transcription
is never overwritten; the disagreement is registered with evidence and stays visible as a
strict xfail. That collision — not a textual accident — is what made PR #82 dirty.

## 2. How the merge was done

1. `git fetch --deepen=200 --filter=blob:none origin main agent/m3240` — the local clone had
   been shallow, which is why an earlier attempt reported "unrelated histories".
2. `git merge-base origin/main origin/agent/m3240` → `0e9bd51`; `main` is 19 commits ahead of
   it, the PR 1 commit ahead. Not an unrelated history: 2 genuine content conflicts plus 1
   silent-data-overwrite hazard.
3. Three-way merge with **semantic union**: both readings are kept, made *separable* and each
   one reproducible, instead of picking a winner per hunk.
4. Every file touched by either parent is then re-checked by a mechanical loss audit
   (`scripts/m3948_merge_loss_audit.py`), because git's auto-merge silently accepted one
   generated file (see §2.3).

### 2.1 Conflict list and per-file resolution

| file | side(s) | resolution |
|---|---|---|
| `src/tensorfvm/lid_cavity_reference.py` | both | `V_CENTRELINE` keeps **-0.23827** (the print). m3240's proposal is preserved verbatim as a new `CORRECTIONS` register (`published`, `proposed`, `basis`, `status="proposed"`, `applied=False`) reachable through `table(400, "v", corrected=True)`, `corrected_table()`, `applied_corrections()`. main's `REFERENCE` / `DISPUTES` / `smoothness_audit()` block is kept verbatim and cross-references the proposal (`proposed_correction`, `proposed_correction_register`). m3240's "Correction history" docstring is kept as a *history / proposal register* that states what actually holds. Only cosmetic loss: m3240's `KeyError` rewording (main's variant prints `Re=` too and is a superset). |
| `tests/test_lid_cavity_re1000_evidence.py` | both | **both tests kept.** main's strict xfail `test_re400_v_centreline_discrepancy_is_still_present_and_unexplained` stays (against the print the debt is real: 0.115 / 0.142 / 0.149). m3240's claim is kept as `test_re400_v_centreline_discrepancy_resolved_by_the_proposed_correction`, re-derived from the committed *published-scoring* bundle against `table(400, "v", corrected=True)`, and strengthened: it also pins the printed value, the "only x = 0.9063 differs" set, the 10× shrink and the n = 32 > 3 % control (that grid does not resolve the lid shear layer). |
| `docs/verification-lid-cavity-ladder/summary.json` | **PR only** (silent hazard) | No text conflict: `main` did not touch it, so git would have overwritten it silently. m3240 had re-scored the *shared* ladder bundle against the corrected entry (99 290 lines changed), which would have made main's m3099 tests score against a table main does not accept. The shared bundle is restored to published-transcription scoring byte-for-byte, and m3240's regeneration is preserved byte-for-byte at `docs/verification-lid-cavity-correction/summary.json` (sha256 equality asserted by the audit). |
| `docs/verification-lid-cavity-correction/{report.md, *.png}` | PR only | added; kept as written, plus a §7 "Merge integration" note recording what stayed, what moved and what is *not* claimed. |

### 2.2 The loss audit (`scripts/m3948_merge_loss_audit.py`)

For **every** file either parent touched (48 files), compare the merged tree against both
parents as a line multiset (whitespace-only lines ignored, since git resolves them with their
block):

```
preserved(side)  ⇔  multiset(merged) − multiset(base) ⊇ multiset(side) − multiset(base)
```

Every residual drop must be covered by an explicit `ADJUDICATIONS` rule whose `must_contain`
needles are *found in the merged file* — a documented rewrite is only accepted if the merged
file really contains the claimed replacement. Moved content is covered by `RELOCATIONS`,
which demands byte equality (sha256) between the old and the new path.

Result: `failures=0`, `ok=True` (`docs/m3948/merge-loss-audit.json`). main's 28 added paths and
each PR path are `preserved` (10 rewritten lines in the reference module and 15 in the test are
the adjudicated re-housings above; 83 724 lines of the regenerated bundle are
`preserved_by_relocation`, sha256 `…` match). **No content silently disappeared.**

## 3. Results

### 3.1 Test matrix — merged tree vs pristine `main`

Same suites, same interpreter, `PYTHONPATH=src`; baseline extracted with `git archive db97955`
so it is untouched by the merge (`docs/m3948/tests.json`).

| suite | merged (`3386a61`+deliverables) | `main` @ `db97955` | delta |
|---|---|---|---|
| `test_ghia_reference_consistency.py` + `test_lid_cavity_re1000_evidence.py` | **19 passed**, 1 skipped, 1 xfailed (8.8 s) | 18 passed, 1 skipped, 1 xfailed (10.6 s) | **+1 pass** (the preserved claim test) |
| `test_lid_cavity_extraction.py` + `test_wall_functions.py` + `test_gci_band_coverage.py` + `test_wall_cell_gradient_attribution.py` | **47 passed** (44.7 s) | 47 passed (19.6 s) | 0 |
| `test_analytic_reference_residuals.py` + `test_porous_forchheimer_reference.py` | **21 passed** (91.7 s) | 21 passed (60.3 s) | 0 |
| `test_lid_cavity.py` (live smoke run) | **9 passed** (74.9 s) | 9 passed (45.8 s) | 0 |
| **total** | **96 passed, 0 failed, 1 skipped, 1 xfailed** | 95 passed, 0 failed, 1 skipped, 1 xfailed | **+1, no regression** |

Not claimed: `tests/test_native_gradient_jump_verification.py` (its whole import closure is
byte-identical to `main` in the merged tree, so its verdict cannot differ; the 105 s attempt was
reaped by the harness after 4 green tests) and the full-suite run (one attempt was reaped at
62 %; the box sits at load ≈ 760 with other agents' suites). Both statements are repeated in the
machine-readable `tests.json` as `not_run` / `timeout` — never as pass.

### 3.2 Numbers re-derived from the committed bundles

`scripts/m3948_merge_verification_figure.py` recomputes them from the two committed bundles; it
also asserts that the bundle's stored `profiles.reference` equals the sampled Ghia table, which
is what proves the *restored* ladder bundle really scores against the print:

| n | max abs error vs **printed** table | max abs error vs **proposed** table | bundle `max_abs_error` | max divergence |
|---|---|---|---|---|
| 32 | 0.11457 | 0.03604 | 0.11457 | 7.2e-15 |
| 64 | 0.14237 | **0.00738** | 0.14237 | 3.2e-14 |
| 128 | 0.14931 | **0.00431** | 0.14931 | 2.5e-13 |

The bundle field matches the printed-scoring recomputation exactly, and the error *grows* under
refinement against the print (0.115 → 0.142 → 0.149) while it *shrinks* under refinement against
the proposal (19.3× at n = 64, 34.6× at n = 128) — the two readings are cleanly separated, which
is exactly what the merge had to preserve.

## 4. Figure

`docs/m3948/outputs/pr82-merge-verification.png`

* left — vorticity of the committed Re = 400, n = 64 field (the field the dispute is about);
* middle — Re = 400 v-centreline: printed table, proposed table, solver profiles n = 32/64/128;
* right — pointwise |error| against both references on a log scale with the 3 % limit.

## 5. Notes, traps, limits

* The dirt was **not** an unrelated history but a shallow local clone (`refs` stopped at
  `db97955`); `git fetch --deepen` resolved the phantom problem. Costly trap: never conclude
  "unrelated histories" before deepening.
* The dangerous conflict was the *silent* one: a generated bundle touched by one side only.
  Text-level conflicts are loud; regenerated artefacts are not. Any merge containing a
  regenerated JSON artefact needs an explicit scoring decision, not git's.
* Convention trap found while writing the figure: the committed Ghia tables run **y/L = 1 → 0**
  (descending abscissa). `numpy.interp` on a descending axis silently returns garbage
  (0.446 instead of 0.142 max error); the script now sorts explicitly and asserts the stored
  reference. Anyone re-scoring these bundles should do the same.
* Harness trap: a shell call is reaped after ~120 s (`setsid`/`nohup` do not survive), so long
  suites must be split; on a load-760 box timings are not comparable across trees.
* Limits: only the affected surface was re-run (4 suites, 96 tests); no full-suite claim.
  The proposal is *conditional*: if `-0.38760` is wrong, the 0.74 % / 0.43 % figures are void,
  which is why the published-scoring gate still fails and stays an xfail.

## 6. Error judgement against the baselines

| key quantity | value | baseline / reference | relative error | < 3 %? |
|---|---|---|---|---|
| merged test tally | 96 passed / 0 failed | main: 95 passed / 0 failed | — | **yes** (no regression, +1) |
| bundle `max_abs_error` (n = 64) vs recomputed | 0.14237 | 0.14237 | **0.0 %** | **yes** |
| bundle `max_abs_error` (n = 32 / 128) vs recomputed | 0.11457 / 0.14931 | 0.11457 / 0.14931 | **0.0 %** | **yes** |
| v-centreline max error, printed scoring, n = 64 | 0.14237 | 3 % acceptance band (0.03) | **374 %** | **no** — unchanged, still a strict xfail |
| v-centreline max error, proposed scoring, n = 64 | 0.00738 | 0.03 | 0.74 % of the band | yes, **but only if the proposal holds** |
| v-centreline max error, proposed scoring, n = 128 | 0.00431 | 0.03 | 0.43 % of the band | yes, same caveat |
| v-centreline max error, proposed scoring, n = 32 | 0.03604 | 0.03 | **120 %** | **no** — n = 32 does not resolve the lid shear layer |
| max divergence, all rows | ≤ 2.5e-13 | divergence-free requirement | — | **yes** |

Honest status: the merge is conflict-free, loss-audited and green on the affected surface, but
**the scientific question is not settled** — the printed-table gate still fails (374 % of the
band) and the proposal is registered as evidence, not as a correction. Whether PR #82 lands is
decided by `merge_governor`'s pr-gate after the branch push, not by this document.

## 7. Reproduction

```
python3 scripts/m3948_merge_loss_audit.py            # -> docs/m3948/merge-loss-audit.json
python3 scripts/m3948_merge_verification_figure.py   # -> docs/m3948/outputs/pr82-merge-verification.png
python3 scripts/m3948_test_matrix.py --tree . --label merged --suites ref,ladder-surface
PYTHONPATH=src python3 -m pytest tests/test_lid_cavity_re1000_evidence.py -q   # the merged claim test
```