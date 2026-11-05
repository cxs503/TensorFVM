# m3788 — test evidence for the PR #82 merge

Host at run time: shared 128-core box, load average ~5800 (other missions running their
own suites concurrently), so wall times below are not performance measurements.

## Scoped gate (the tests that own the two resolved files)

| tree | command | result |
|---|---|---|
| `main` (`db97955`, worktree `m3788_base`) | `pytest -q tests/test_ghia_reference_consistency.py tests/test_lid_cavity_re1000_evidence.py tests/test_lid_cavity_extraction.py` | **29 passed, 1 skipped, 1 xfailed** |
| merged (`ca2c7a3`, worktree `m3788`) | same command | **30 passed, 1 skipped, 1 xfailed** |

The merged tree is one test above the baseline: `test_rejected_reference_correction_is_recorded_but_not_applied`
(the new gate). The single `xfailed` is `test_re400_v_centreline_discrepancy_is_still_present_and_unexplained`,
kept *strict* on purpose: the Re = 400 near-wall defect is an open numerical item and must
not be closed by editing the reference. The `skipped` test needs the evidence bundle in a
checkout that has it; it is skipped identically on both trees.

## Full suite

`pytest -q -n 32` (1140 tests collected) launched on both trees. At the time of writing the
runs had reached **92 % (merged)** and **90 % (main)** with **zero failures and zero errors
in either** (`grep -c '^FAILED\|^ERROR'` = 0, no `F` markers emitted); the remaining tests are
long solver smokes that were still in flight on the overloaded host when this bundle was
committed. They are not a merge-specific gap — the same slow tail stalls both trees — but the
comparison is therefore **partial, not complete**, and is reported as such.

## Static checks on the two hand-edited files

* `ast.parse` on both: OK.
* duplicate top-level definition scan: none (`lid_cavity_reference.py` 5 defs,
  `test_lid_cavity_re1000_evidence.py` 10 defs).
* merge conflicts remaining in the tree: `git grep -n '<<<<<<<'` → none.