# Verification reproducibility from a clean sparse checkout

EDIT_WHITELIST: `scripts/run_openfoam_parallel.py`, `tests/evidence_support.py`,
`tests/test_suboff_validation.py`, `tests/test_evidence_checkout_policy.py`,
`tests/test_openfoam_parallel_runner.py`, `tests/test_openfoam_parallel_runner.py`,
`docs/`.

## 1. Why this mattered

The repository is normally exercised from a **sparse cone** (`src/ tests/ scripts/`
plus two top-level files). Its own policy file
(`tests/evidence_support.py`) already states the rule: a tracked artefact that is
absent from a checkout is a *checkout* condition, so the owning test must `skip`
with an actionable reason — never fail, and never silently pass. A full test run
from such a cone reported **7 failed, 750 passed, 44 skipped, 6 xfailed**. All
seven red tests were reproducibility defects, not physics defects: the contracts
they guard were intact and unmodified.

## 2. Root causes found

| # | Failing tests | Root cause | Class |
|---|---|---|---|
| F1 | `tests/test_openfoam_parallel_runner.py` (4) | The test runs `scripts/run_openfoam_parallel.py` as a child process. The script imported `tensorfvm` unconditionally; the child does not inherit pytest's `pythonpath=src`, and the distribution is not installed in the dev image, so the runner died with `ModuleNotFoundError: No module named 'tensorfvm'` *before* the checkpoint audit it is supposed to perform. | Packaging idiom violation: 140 scripts import `tensorfvm` without a `src/` bootstrap; the other audit scripts already use `sys.path.insert(0, .../'src')`. |
| F2 | `tests/test_suboff_validation.py` (3) | The DARPA SUBOFF AFF-1 validation spec is tracked at **`benchmarks/suboff_aff1_validation.json`** — outside `docs/` — and the test read it with a bare `Path.read_text`. `published()` only resolves `docs/` evidence, so the spec could not be resolved from a sparse cone at all: `FileNotFoundError`. | Evidence-resolution gap: a tracked spec existed in the object store while the guard had no way to reach it. |

F2 is the more interesting one: skipping would have been *correct but weaker*. The
spec is tracked in git, so the assertion can still be executed — the bytes are the
tracked ones, not a substitute.

## 3. Fixes (minimal)

1. `scripts/run_openfoam_parallel.py`: resolve the package from the sibling `src/`
   tree exactly like the other audit scripts (`sys.path.insert(0, .../'src')`),
   with a comment stating why. The runner now works from any source checkout, and
   its `ModuleNotFoundError` can no longer masquerade as a checkpoint failure.
2. `tests/evidence_support.py`: new `tracked(relative)` resolver. Resolution order
   = worktree → `TENSORFVM_EVIDENCE_ROOT` materialisation → `HEAD` blob extracted
   into a temporary cache → `pytest.skip` with an actionable reason. It rejects
   absolute paths and `..` escapes (`ValueError`).
3. `tests/test_suboff_validation.py`: read the spec through `tracked(...)`; the
   three metric tests now actually run instead of failing/skipping.
4. New regression tests pin the invariants rather than the incident:
   `test_runner_imports_the_package_from_a_source_checkout` (child process with
   `PYTHONPATH`/`PYTHONHOME` stripped, asserts exit 0), plus five policy tests for
   `tracked` (worktree preference, evidence-root redirect, blob recovery, skip
   reason, path-escape rejection).

No solver source file, tolerance, gate or numerical path was touched.

## 4. Verification

```
python -m pytest tests/test_suboff_validation.py \
                 tests/test_openfoam_parallel_runner.py \
                 tests/test_evidence_checkout_policy.py -q
# before: 7 failed, 1 passed
# after : 17 passed  (24.6 s)

python -m pytest tests/test_lbm_reference_audit.py tests/test_periodic_mac.py \
                 tests/test_suboff_geometry.py -q
# 8 passed, 9 skipped   (existing published() consumers unchanged: skips preserved)
```

Reproducibility gate: **red tests 7 → 0, with no test converted to a skip and no
assertion relaxed** (the 3 SUBOFF tests now execute against the tracked spec).

## 5. Accuracy statement (vs 3 % criterion)

No numerical result changed, so the 3 % accuracy criterion is not applicable to
this work; the paper-facing numbers are unchanged. The tested quantities in the
SUBOFF spec are reproduced exactly: with the spec's own
`rho U^2 L^2` scale, the dimensional force `87.4 N` maps to
`Cd = 0.09312081297064881` with relative error `< 1e-14` (machine precision), and
the gate flip is exercised in the other direction (`1.03x` force →
`nominal_error_below_3_percent == False`). In other words the 3 % decision boundary
is asserted, not asserted-away.

## 6. Limitations / follow-ups

* 140 scripts import `tensorfvm` without the `src/` bootstrap. Only one of them is
  executed as a child process by the test suite, so only that one is *proven* red;
  the rest are latent and are not fixed here (blanket edits would collide with
  in-flight branches). A cheap future gate: import each `scripts/*.py` in a
  subprocess with a scrubbed environment.
* `docs/` is absent from the sparse cone by design, so `published()`-guarded tests
  skip (9 skips above). Making the cone materialise `docs/` is a workflow change,
  not a test change.
* The blob fallback needs the `.git` object store; from an installed wheel it skips
  with a reason instead of failing (by construction, pinned by a test).