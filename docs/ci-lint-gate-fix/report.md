# PR #104 — why `lint (advisory)` was red, and what was fixed

Mission m4184 · branch base `agent/m3588` (PR #104 head `a4948d1`) · ruff 0.15.0

## 1. Symptom and root cause

PR #104 added three Python-producing artefacts (`scripts/audit_two_phase_surface_tension_fix.py`,
`scripts/audit_two_phase_surface_tension_defect.py`, docs + JSON). CI reported
`lint (advisory) = failure`, which blocks the merge gate even though the job is documented as
advisory. Two independent causes:

1. **The PR's own new files were dirty.** Both scripts imported `math` without ever using it
   (`F401`) and were not `ruff format` clean (`uv`-style layout). Reproduced locally, see
   `artifacts/ci-lint-gate-fix/ruff-before.txt` (excerpt: `F401 math imported but unused`,
   `2 files would be reformatted`).
2. **The advisory flag sat on the wrong level.** `.github/workflows/ci.yml` had
   `continue-on-error: true` on the *job* `lint (advisory)`. A job-level flag stops the
   *workflow run* from failing, but the job — and therefore its check run — is still reported
   as **failed**. Any gate that looks at check-run conclusions (as pr-gate does) sees a red
   check on every single pull request, regardless of what the PR touched. That is why the
   signal was permanently red and could never be "fixed" by cleaning up code alone.

The pre-existing tree debt is large and unrelated to PR #104: 9822 violations
(`E702` 7601, `E701` 1852, `F401` 98, …) and 554 unformatted files, see
`artifacts/ci-lint-gate-fix/ruff-statistics.json`. Auditing that backlog is not a lint-red fix
and is deliberately left alone here.

## 2. Changes (4 files, all inside the PR's blast radius)

| file | change |
|---|---|
| `scripts/audit_two_phase_surface_tension_defect.py` | drop unused `import math`; `ruff format` |
| `scripts/audit_two_phase_surface_tension_fix.py` | drop unused `import math`; `ruff format` |
| `.github/workflows/ci.yml` | `continue-on-error` moved from the job to the two full-tree steps; added a **blocking ratchet** that lints only *newly added* `*.py` under `src/ tests/ scripts/`; added a debt summary step (`$GITHUB_STEP_SUMMARY`) |
| `pyproject.toml` | dev extra `ruff>=0.6` → `ruff>=0.15,<0.16` so local and CI formatting agree (a drifting formatter would make the ratchet flaky) |

Design of the ratchet: new files are fully authored by the PR, so they can be held to the strict
bar without inheriting legacy debt; *edits* to legacy files stay advisory. The backlog can then
only shrink. Base is taken from `github.event.pull_request.base.sha`, falling back to
`git merge-base origin/main HEAD` for push events (`fetch-depth: 0`).

## 3. Evidence (all commands re-runnable)

```
$ ruff check scripts/audit_two_phase_surface_tension_{defect,fix}.py
All checks passed!                                   # exit 0
$ ruff format --check scripts/audit_two_phase_surface_tension_{defect,fix}.py
2 files already formatted                            # exit 0
```

Negative control — the same two files at the pre-fix commit must still fail, proving the ratchet
is not a no-op: `ruff check` → `Found 2 errors`, exit 1; `ruff format --check` →
`2 files would be reformatted`, exit 1 (`ruff-before.txt`).

Ratchet simulation (exactly the CI step body, run locally):
`added py files = scripts/audit_two_phase_surface_tension_defect.py,
scripts/audit_two_phase_surface_tension_fix.py` → both steps exit 0 (`ruff-after.txt`).

Behaviour preservation of the de-imported/reformatted scripts:
`python scripts/audit_two_phase_surface_tension_defect.py` regenerates
`docs/two-phase-surface-tension-fix/outputs/{defect_summary.json,divergence_history.png}`
**byte-identical** to the committed copies (`git status` shows no modification under `docs/`),
i.e. the formatting change touched no numerics.

Regression check of the package: `pytest -q tests -k "two_phase or surface_tension"`
→ **44 passed, 1 skipped, 1 xfailed** (`pytest-two-phase.txt`).

## 4. Numbers regenerated while validating (no hand-copying)

Static-drop Young–Laplace jump, exact `σ/R = 0.07/0.2 = 0.35 Pa`:

| n | computed Δp | exact | rel. error | <3 %? |
|---|---|---|---|---|
| 24 | 0.3559 Pa | 0.35 Pa | 1.68 % | yes |
| 32 | 0.3438 Pa | 0.35 Pa | 1.78 % | yes |

Figure: `artifacts/ci-lint-gate-fix/ruff-debt.png` — legacy violation histogram by rule
(generated from `ruff check --statistics`, log scale).

## 5. Caveats / honesty notes

* **The legacy backlog is not fixed** (9822 violations, 554 unformatted files). It is now
  advisory *and* visible in the job summary instead of blocking, and the ratchet stops new debt.
  Promoting lint to a hard gate still requires the backlog burn-down the CI comment describes.
* **`scripts/audit_two_phase_surface_tension_fix.py` crashes at runtime and did so before this
  change.** `python scripts/audit_two_phase_surface_tension_fix.py` raises
  `RuntimeError: The size of tensor a (22) must match the size of tensor b (24)` at
  `_projection_step_fixed` (line 118): it slices `f_u` with `[:, :-1]` against a full-size
  operand, so the script never reaches its "Young–Laplace < 3 %" claim. Reproduced with the
  pristine file from `HEAD` (`git show HEAD:scripts/... > /tmp/orig_fix.py`) to prove it is
  pre-existing and unrelated to the lint fix. It is **not** repaired here (out of scope for a
  CI-red fix; it needs a solver-API/shape audit and would change the numbers the PR advertises),
  so the "fix verifies the correction" claim of PR #104 remains unverified and should be treated
  as STUB-level evidence until someone reruns it.
* The defect script itself runs and reports `Defect confirmed: False` with Δp error ≈1.7 %, i.e.
  the baseline solver already satisfies Young–Laplace on this static case — the PR's "divergence
  defect" narrative is not reproduced by its own audit script. Recorded here rather than silently
  overwritten.
* `lint (advisory)` keeps its check name on purpose: renaming the job would drop the check the
  gate matches on.