# PR #82 (agent/m3240) — merge resolution report

Unit m3910 · TensorFVM · branch `agent/m3240` · base `main` @ `db97955`

## 1. Why the PR was dirty

`git merge-base origin/main origin/agent/m3240` initially failed with
*"refusing to merge unrelated histories"*: the local clone is **shallow**
(`.git/shallow` listed `db97955` and `e3ac284` as boundaries), so `main`'s
history stopped at its own tip.  `git fetch --deepen=200 --filter=blob:none
origin main agent/m3240` removed every boundary, after which the true merge base
appeared:

| ref | commit | note |
|---|---|---|
| merge-base | `0e9bd51` | "Merge #49 (m2766) via merge-governor" |
| `main` | `db97955` | 19 commits after the base (incl. m3099's Re=400 attribution) |
| `agent/m3240` | `99f8e6e` | 1 commit after the base ("matured mission deliverables") |

So the dirt was **not** an unrelated history, it was a stale local clone plus a
genuine content conflict in two files (below).  A stale local ref
`refs/heads/agent/m3240 = 5b0ad9d` (a local-only merge built on an old `main`,
never pushed) was discarded in favour of the GitHub head `99f8e6e`.

## 2. Conflict list and how each file was resolved

Both conflicting files carried the *same scientific collision*: m3240 proposed
that the Ghia (1982) Table II entry `v(Re=400, x=0.9063) = -0.23827` is a
transcription error and rewrote it to `-0.38760`; `main` (unit m3099) had, in the
meantime, attributed the same point and adopted the opposite invariant — *the
published number is never overwritten; the mismatch is registered with evidence
and stays visible as a strict xfail*.

Resolution rule used for both files: **semantic union, with the two readings kept
separable** — the published transcription stays byte-for-byte as printed, and
m3240's proposal is preserved as a first-class, reproducible register entry with
an opt-in scoring path.  Nothing was dropped silently; the one deliberate
*non*-application (overwriting the published value) is re-housed and documented
in the code, in `docs/verification-lid-cavity-correction/report.md` §7 and here.

### 2.1 `src/tensorfvm/lid_cavity_reference.py` (content conflict)

| side | change |
|---|---|
| m3240 | `V_CENTRELINE[400.0]`: `-0.23827 → -0.38760`; "+ Correction history" docstring block; `KeyError` message rewording; trailing-newline fix |
| main | appended `REFERENCE`, `DISPUTES` register (`computed_n`, `observed_order`, Richardson limit, interpolation bound, evidence, adjudication), `smoothness_audit()` |

Resolution:

* `V_CENTRELINE` keeps **`-0.23827`** (the print).  Rationale: m3240's proposed
  value *is* the solver's Richardson limit (-0.38989 rounded), i.e. it is fitted
  to the solver and therefore cannot adjudicate the reference; overwriting the
  table with it would erase the only independent evidence the verification has.
* m3240's proposal is preserved verbatim in a new `CORRECTIONS` register
  (`published`, `proposed`, `proposed_by`, `basis`, `status="proposed"`,
  `applied=False`, `adjudication`) and reachable through
  `table(400.0, "v", corrected=True)`, `corrected_table()`, `applied_corrections()`.
* m3240's "+ Correction history" docstring block is kept as
  "Correction history / proposal register", rewritten to say what actually holds
  (proposal kept, not written into the table, debt still real against the print).
* `main`'s `DISPUTES` record gains `proposed_correction: -0.38760` and
  `proposed_correction_register: "CORRECTIONS (opt-in, never applied)"`, so the
  register cross-references the proposal it refuses to apply.
* `main`'s `REFERENCE` / `DISPUTES` / `smoothness_audit()` block kept verbatim;
  the `KeyError` message keeps main's more informative `Re=` form (m3240's
  rewording was cosmetic); the EOF newline fix from m3240 is kept.

### 2.2 `tests/test_lid_cavity_re1000_evidence.py` (content conflict)

| side | change |
|---|---|
| m3240 | deleted the strict-xfail test, added `test_re400_v_centreline_discrepancy_resolved` (asserts `v.max_abs_error < 0.03` at n=64/128 from the bundle) |
| main | kept the strict xfail with an updated attribution reason (extraction exonerated, defect localised beside the right wall) |

Resolution: **both tests kept**.
`test_re400_v_centreline_discrepancy_is_still_present_and_unexplained` stays
(strict xfail, main's reason text) because against the published table the debt is
still there (0.115/0.142/0.149).  m3240's claim is kept as
`test_re400_v_centreline_discrepancy_resolved_by_the_proposed_correction`, which
re-scores the committed ladder's computed profiles against
`table(400.0, "v", corrected=True)` and additionally asserts that the published
value is untouched, that only x = 0.9063 differs, and that n = 32 stays above 3 %
(that grid does not resolve the lid shear layer).  The module docstring bullet is
rewritten to describe exactly this split.

### 2.3 `docs/verification-lid-cavity-ladder/summary.json` (silent data overwrite, not a text conflict)

m3240 re-scored the shared ladder bundle against the corrected entry (99 290
inserted / 99 290 deleted lines; `v.max_abs_error` at Re=400 n=64: 0.14237 →
0.00738).  `main` did not touch this file, so git auto-merged it — i.e. it would
have been a *silent* semantic overwrite.  Unit m3099's tests re-derive the dispute
record from this bundle, so it had to score against the print:

* the shared bundle was restored to the published-transcription scoring
  (`git checkout <merge-base> -- docs/verification-lid-cavity-ladder/summary.json`);
* m3240's corrected-scoring bundle is preserved verbatim as
  `docs/verification-lid-cavity-correction/summary.json` (2.2 MB), which is what
  §3 ("new" column) of that report was computed from, and which the new test
  re-derives its numbers from.

### 2.4 `docs/verification-lid-cavity-correction/report.md` (no conflict, annotated)

m3240's report is kept as written; a §7 "Merge integration" section was appended
recording the above (what stayed, what moved, what is still not claimed).

## 3. Verification

Commands (`PYTHONPATH=src`), on the merged tree and on `main` @ `db97955`
(extracted with `git archive` so the baseline is pristine):

```
python -m pytest tests/test_lid_cavity.py tests/test_lid_cavity_extraction.py \
    tests/test_ghia_reference_consistency.py tests/test_lid_cavity_re1000_evidence.py \
    tests/test_wall_functions.py -q
```

| tree | result |
|---|---|
| merged (this branch) | **62 passed, 1 skipped, 1 xfailed** (48.7 s) |
| `main` baseline | **61 passed, 1 skipped, 1 xfailed** (105.6 s) |

No regression; the single extra pass is m3240's preserved claim test.  Reference
level checks: `smoothness_audit()["ok"] is True` with `400/v/0.9063` (and its
neighbour) flagged, `self_check()["ok"] is True`, published `-0.23827`,
corrected `-0.38760`, `CORRECTIONS[...]["applied"] is False`.

A full-suite run was started on both trees but the box is shared and the baseline
run was still in the slow `tests/test_native_hybrid_*` block when this report was
written; it is **not** claimed here.  The affected surface is the four files
above, all covered by the table.

Figure: `docs/m3910/outputs/pr82-merge-re400-v-centreline.png` (left: published
vs proposed entry and the n=32/64/128 profiles; right: pointwise |error| against
both references with the 3 % limit), generated by
`scripts/m3910_merge_pr82_figure.py` from the committed bundles only.

## 4. Numbers

| quantity | value | source |
|---|---|---|
| published entry v(Re=400, x=0.9063) | **-0.23827** | Ghia et al. (1982) Table II, unchanged |
| m3240 proposed entry | **-0.38760** | `CORRECTIONS`, `applied=False` |
| solver Richardson limit at that point | -0.38989 (order 2.00) | `DISPUTES` / ladder bundle |
| max abs error, published scoring, n=64/128 | **0.14237 / 0.14931** | ladder bundle (restored) |
| max abs error, corrected scoring, n=64/128 | **0.00738 / 0.00431** | correction bundle |
| max abs error, corrected scoring, n=32 | 0.03604 | correction bundle (still > 3 %) |
| divergence, all rows | < 1e-11 | bundle |

The corrected-scoring improvement (19.3× at n=64, 34.6× at n=128) is *conditional*
on the proposal being right; it is therefore reported as a proposal, not as a
pass, and the default gate still fails.