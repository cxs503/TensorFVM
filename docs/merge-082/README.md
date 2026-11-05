# PR #82 (agent/m3240) → main: three-way union merge, cured

**Mission:** m3836 · **Repo:** TensorFVM · **Cure branch:** `agent/m3836`
**Sides:** main `db97955686af` · branch `99f8e6eab3a4` (`origin/agent/m3240`) · merge-base `0e9bd51dfac1`
**Verdict:** union merge, 2 hand-resolved files, 0 files lost, oracle untouched, tests strictly above the `main` baseline.

## 1. Why the merge was dirty

Both sides worked on the *same* verification debt from opposite directions, so a
textual merge is not merely inconvenient, it is unsafe:

* `agent/m3240` diagnosed the Re = 400 v-centreline defect, concluded the printed
  Ghia value at x = 0.9063 was a typo, **rewrote the reference table**
  (`-0.23827 → -0.38760`), re-scored the committed ladder bundle against the new
  value and deleted the strict-xfail gate. Net effect on CI: red → green.
* `main` kept the published number verbatim and instead *registered* the
  discrepancy (`lid_cavity_reference.DISPUTES`, id `GHIA-II-400-v-0.9063`),
  added a smoothness audit that can only be satisfied by a registered dispute,
  and kept the ladder defect visible as a strict xfail.

A mechanical overlay loses either the dispute machinery (branch wins) or the
branch's whole investigation (main wins). Worse, git's own auto-merge already
produced a hybrid that is green-looking and wrong: it silently took the branch's
rewritten table row *and* the branch's re-scored `summary.json` while keeping
main's registry, and it left the branch's "Correction history" docstring in place
announcing a fix that would no longer be in force.

**Adjudication rule applied:** a reference table is an oracle. Editing it converts
a verification failure into a passing suite without new physics, so the printed
value stays; the branch's evidence is preserved and *remains testable*, but the
gate does not flip. This is the platform's "L 不可覆盖" rule (`失败就写失败+证据`)
applied to a merge, not a post-hoc edit of anybody's data.

## 2. Conflict map (computed from git, not by eye)

Files the branch delivered (7 in total):

Only on the branch:
* `docs/verification-lid-cavity-correction/all_centrelines.png`
* `docs/verification-lid-cavity-correction/error_summary.png`
* `docs/verification-lid-cavity-correction/ghia_correction.png`
* `docs/verification-lid-cavity-correction/report.md`
* `docs/verification-lid-cavity-ladder/summary.json`

Changed on both sides (true content conflicts):
* `src/tensorfvm/lid_cavity_reference.py`
* `tests/test_lid_cavity_re1000_evidence.py`

Changed only on `main` (must survive untouched):
* `docs/GHIA_RE400_V_ATTRIBUTION.md` (present, unchanged)
* `docs/WALL_CELL_GRADIENT_ATTRIBUTION.md` (present, unchanged)
* `docs/gci-band-coverage/report.md` (present, unchanged)
* `docs/m3050/outputs/wall_cell_gradient_attribution.json` (present, unchanged)
* `docs/m3050/outputs/wall_cell_gradient_attribution.png` (present, unchanged)
* `docs/m3099/outputs/ghia-re400-v-attribution/attribution.json` (present, unchanged)
* `docs/m3099/outputs/ghia-re400-v-attribution/attribution.png` (present, unchanged)
* `docs/m3149/outputs/gci-band-coverage/RESULTS.md` (present, unchanged)
* `docs/m3149/outputs/gci-band-coverage/audit.json` (present, unchanged)
* `docs/m3149/outputs/gci-band-coverage/gci_band_coverage.png` (present, unchanged)
* `docs/verification-analytic-reference-residuals/derivative-limit.png` (present, unchanged)
* `docs/verification-analytic-reference-residuals/reference-containment.png` (present, unchanged)
* `docs/verification-analytic-reference-residuals/results-table.md` (present, unchanged)
* `docs/verification-analytic-reference-residuals/results.json` (present, unchanged)
* `docs/verification-native-gradient-jump-algebra/README.md` (present, unchanged)
* `docs/verification-native-gradient-jump-algebra/evidence.json` (present, unchanged)
* `docs/verification-native-gradient-jump-algebra/gradient-jump-verification.png` (present, unchanged)
* `docs/verification-native-gradient-jump-algebra/report-table.md` (present, unchanged)
* `scripts/audit_analytic_reference_residuals.py` (present, unchanged)
* `scripts/audit_gci_band_coverage.py` (present, unchanged)
* `scripts/audit_ghia_re400_v_attribution.py` (present, unchanged)
* `scripts/audit_lid_cavity_v_centreline.py` (present, unchanged)
* `scripts/audit_wall_cell_gradient_attribution.py` (present, unchanged)
* `scripts/verify_native_gradient_jump_algebra.py` (present, unchanged)
* `src/tensorfvm/lid_cavity.py` (present, unchanged)
* `src/tensorfvm/verification/artifacts/lid-cavity-extraction/report.md` (present, unchanged)
* `src/tensorfvm/verification/artifacts/lid-cavity-extraction/results.json` (present, unchanged)
* `src/tensorfvm/verification/artifacts/porous-forchheimer/REPORT.md` (present, unchanged)
* `src/tensorfvm/verification/artifacts/porous-forchheimer/evidence.json` (present, unchanged)
* `src/tensorfvm/verification/face_lsq3d.py` (present, unchanged)
* `src/tensorfvm/verification/lid_cavity_extraction.py` (present, unchanged)
* `src/tensorfvm/verification/native_order3d.py` (present, unchanged)
* `src/tensorfvm/verification/porous_forchheimer.py` (present, unchanged)
* `tests/test_analytic_reference_residuals.py` (present, unchanged)
* `tests/test_gci_band_coverage.py` (present, unchanged)
* `tests/test_ghia_reference_consistency.py` (present, unchanged)
* `tests/test_lid_cavity_extraction.py` (present, unchanged)
* `tests/test_native_gradient_jump_verification.py` (present, unchanged)
* `tests/test_porous_forchheimer_reference.py` (present, unchanged)
* `tests/test_wall_cell_gradient_attribution.py` (present, unchanged)
* `tests/test_wall_functions.py` (present, unchanged)

`git merge` itself reported **2 conflicting files**
(`src/tensorfvm/lid_cavity_reference.py`, `tests/test_lid_cavity_re1000_evidence.py`)
and auto-merged `docs/verification-lid-cavity-ladder/summary.json` — that silent
auto-merge is the one that would have laundered the gate, so it was re-adjudicated
by hand as well.

## 3. Per-file resolution

| file | side taken | resolution |
|---|---|---|
| `src/tensorfvm/lid_cavity_reference.py` | union | main's published tables (byte-identical, verified) + main's `DISPUTES` registry & `smoothness_audit` kept; the branch's proposed `-0.38760` is recorded **inside** `DISPUTES[(400.0,'v',0.9063)]` (`correction_proposed`, `correction_status`, `proposal_evidence`, `dispute_id`) and the branch's "Correction history" docstring is replaced by an adjudicated *Dispute history* naming both sides |
| `tests/test_lid_cavity_re1000_evidence.py` | union | main's strict-xfail defect gate kept verbatim; the branch's numeric claim re-expressed as 3 regression tests over the archived numbers (see §4) |
| `docs/verification-lid-cavity-ladder/summary.json` | main | the bundle scored against the **published** table stays authoritative; the branch's re-scored bundle is archived as `docs/verification-lid-cavity-correction/corrected_scoring.json` (`status: SUPERSEDED_PROPOSAL`), so its numbers are not lost but are not a gate |
| `docs/verification-lid-cavity-correction/report.md` | branch + banner | kept verbatim with an adjudication banner (`SUPERSEDED — proposal not applied`) so no reader inherits the claim "VERIFIED & FIXED" |
| `docs/verification-lid-cavity-correction/*.png` | branch | kept verbatim |
| branch's `KeyError` message tweak (`f"no Ghia table for {re}"...`) | dropped | cosmetic reword that drops the `Re=` label; no content lost, recorded here because "not taken" must never be silent |

Machinery added by this cure so the decision is checkable instead of narrated:
`scripts/audit_merge_082_union.py` re-derives the three-way classification from
git, asserts the published table block is identical to `main`'s, asserts every
delivered file is either present or listed as adjudicated (**silent-loss
detector**), folds the JUnit reports into a regression table, and regenerates the
figure + `union_audit.json` below.

## 4. The union, in numbers (programmatically injected)

Disputed entry, Re = 400, v at x = 0.9063 (cavity centreline y = 0.5):

| n | computed (ladder bundle) | published reference | max\|err\| vs published (gate in force) | max\|err\| vs proposal (archived) |
|---|---|---|---|---|
| 32 | -0.35284 | -0.23827 | 0.1146 | 0.0360 |
| 64 | -0.38064 | -0.23827 | 0.1424 | 0.0074 |
| 128 | -0.38758 | -0.23827 | 0.1493 | 0.0043 |

The solver's own Richardson limit is -0.38989 (main's registry), i.e. the
converged solution sits ~64 % away from the printed value while the three
neighbouring entries agree to 0.3 %. That is a *real* research finding — and it is
precisely why it must stay a dispute rather than become an edit: the claim cannot
be adjudicated without the printed Table II, which is not in this checkout.

## 5. Test evidence (merge tree vs `main` baseline, same selection, same machine)

Selection: `test_lid_cavity.py`, `test_ghia_reference_consistency.py`,
`test_lid_cavity_re1000_evidence.py`, `test_lid_cavity_extraction.py`,
`test_analytic_reference_residuals.py`, `test_solver.py`.

| tree | passed | failed | error | skipped | subtests |
|---|---|---|---|---|---|
| `main` baseline `db97955686af` | 67 | 0 | 0 | 2 | 39 |
| union (this branch) | 70 | 0 | 0 | 2 | 39 |

New tests attributable to the union (the branch's claim, preserved as tests):
* `test_pr82_proposal_numbers_are_archived_and_reproduce_its_claim`
* `test_the_published_re400_entry_was_not_rewritten_by_the_merge`
* `test_the_two_ladders_disagree_only_on_the_disputed_point`

Regressions: **0**. Tests present in baseline but missing
after the merge: **0**. The strict xfail on the disputed
point still xfails (`xfailed 1`), i.e. the defect is still reported as a defect.

## 6. Union invariants (machine-checked, `--strict`)

| invariant | result |
|---|---|
| every file delivered by `agent/m3240` present or explicitly adjudicated | **PASS** (3 verbatim, 4 adjudicated, 0 silently lost) |
| published Ghia table identical to `main` (published value in force = -0.23827) | **PASS** |
| branch's proposal preserved in the registry (`correction_proposed = -0.38760`) | **PASS** |
| no pass→fail transition, no lost test, no failing test | **PASS** |

Regenerate everything with:

```
python3 scripts/audit_merge_082_union.py --strict \
    --junit-merge /tmp/junit_merge.xml --junit-base /tmp/junit_base.xml
```

## 7. Caveats / what is *not* settled

* The physics is unresolved by construction: whether Ghia's Table II entry is
  misprinted cannot be decided offline. Resolution path stays in the registry
  (`adjudication`: printed Table II or independent DNS).
* Remote push of this merge back to `origin/agent/m3240` could not be performed
  from this sandbox, for two independent reasons: egress is dead (`git ls-remote
  origin` exits 124 after 40 s, no output) and the push command itself is rejected
  by the execution policy. Updating the local `agent/m3240` ref is also blocked
  because that branch is checked out in another worktree (`fatal: cannot force
  update the branch 'agent/m3240' checked out at /root/tp-agent/work/m3240`). So
  the cured merge lives on `agent/m3836`, whose tip is the merge commit with
  parents `db97955` (main) and `99f8e6e` (old `origin/agent/m3240`), and the
  platform's push path is what carries it to `agent/m3240`. Old SHA `99f8e6e`,
  new SHA is the `agent/m3836` tip — quoted in the mission report, because a
  document inside the commit cannot name that commit.
* The merge commit itself is a merge commit (`--no-ff`), so `git log --first-parent`
  on `main` would show it as a single event once the governor lands it.
