# CI lint gate: why PR #114 was red, and what the gate now covers

Mission m4248 (custody round 2 on TensorFVM PR #114, branch `agent/m3679`).

## Symptom

PR #114 ("add analytical verification for transient scalar diffusion") added
three files and shipped a green `test` matrix, but its check-run set was

```
failing: lint (advisory)=failure,lint (advisory)=failure [strict: holding chain]
```

i.e. both ruff steps of the `lint` job were red, and the merge governor — which
requires *every* check-run conclusion to be `success` — held the PR.  The same
signature had kept TensorFEM #44 stuck for days.

## Two independent causes

1. **The PR's own files were not lint-clean.**  With the ruff version CI
   installs (`ruff>=0.6` → 0.15.0 at the time of writing):

   * `src/tensorfvm/verification/scalar_transient_diffusion.py:96` — `F841`
     unused local `x = mesh['x_centers']` (the solver really never reads it);
   * both new Python files were not `ruff format` clean (single quotes, trailing
     whitespace on docstring lines, `x[i-1]` spacing).

   Reproduced locally from the untouched branch:

   ```
   ruff check        → Found 1 error (F841)
   ruff format -q -   → 2 files would be reformatted
   ```

2. **The gate scanned the whole tree, which can never be clean.**  On the same
   commit `ruff check src tests scripts` reports 9875 findings and
   `ruff format --check` wants to rewrite 586 files:

   | rule | count | nature |
   |---|---|---|
   | E702 multiple statements on one line (semicolon) | 7634 | pre-existing style debt |
   | E701 multiple statements on one line (colon) | 1852 | pre-existing style debt |
   | F401 unused import | 111 | pre-existing (204 auto-fixable in total) |
   | E401 multiple imports on one line | 84 | pre-existing |
   | E731 lambda assignment | 71 | pre-existing |
   | F841 unused variable | 54 | pre-existing |
   | E402 module import not at top | 40 | pre-existing |
   | E741 ambiguous name | 20 | pre-existing |
   | F541 f-string without placeholders | 6 | pre-existing |
   | E703 useless semicolon | 3 | pre-existing |

   So cause 2 alone is enough to keep *every* PR in this repository red forever,
   whatever the PR contains.  Fixing it "properly" would mean rewriting ~9.5k
   statement lines in hundreds of files unrelated to any single mission — out of
   scope here, and exactly the kind of drive-by churn that the repository's own
   CI comment ("Fix the backlog, then drop this continue-on-error and promote it
   to a gate") reserves for a dedicated clean-up campaign.

## What changed

`.github/workflows/ci.yml`, `lint` job only (the `test` job is untouched):

* the job is now **regression-scoped**: it computes the event's own file set
  (`git merge-base origin/<base_ref> HEAD` for pull requests, `github.event.before`
  for pushes) and gates exactly the changed `src/`, `tests/`, `scripts/` Python
  files, with both `ruff check` and `ruff format --check`;
* the job-level `continue-on-error: true` is **gone**, so the check can now fail
  — it is a real gate, just a narrow one (a gate that cannot fail is not a gate);
* the tree-wide count is still executed and printed in a separate
  `continue-on-error` step, labelled "Pre-existing tree-wide backlog
  (advisory, never gates)", so the backlog stays measurable and burning it down
  remains a visible, trackable project;
* checkout uses `fetch-depth: 0` **with `filter: blob:none`**: the merge-base
  needs commit/tree history, but the archived field data in this history is
  ~1.2 GiB, and `git diff --name-only` never needs the blobs;
* the job *name* `lint (advisory)` is deliberately kept unchanged so that a
  branch-protection rule naming this check keeps matching.

## How it was verified

`artifacts/m4248/ci_lint_repro.py` extracts the `run:` blocks of the `lint` job
*out of* `ci.yml` (no duplicated logic — it cannot drift), substitutes the
`${{ … }}` expressions and executes them in a throw-away `git worktree`:

```
python artifacts/m4248/ci_lint_repro.py --negative
  step 2 (scope)        exit 0
  step 3 (ruff)         exit 0
  step 4 (backlog)      exit 0
  negative control: inject `import json` into a gated file
                      -> step 3 exit 1 (F401 + E402)  ⇒ gate is live
```

Full log with the before/after evidence: `artifacts/m4248/lint_repro.log`.

## Baseline of the numbers above

The counts in the two sections above were measured on the #114 head base
(`87d4083`), i.e. the tree the failing check ran on.  They move as the
repository grows; the advisory CI step reprints the current tree-wide
numbers on every run, and `artifacts/m4248/lint_repro.log` holds the
measurement taken on the commit that carries this note.

## Consequences / limits

* A PR that touches a legacy dirty file *now* owes that file's lint fixes.  That
  is intended (the file is being edited anyway) but it is a policy change worth
  knowing when planning a mission on a legacy module.
* The 9875 tree-wide findings are **not** fixed by this mission; they are only
  made non-blocking and continuously reported.
* Burn-down recipe if the repository wants the tree-wide gate back:
  `ruff check src tests scripts --fix` (204 auto-fixes) then
  `ruff format src tests scripts`, then hand-fix E731/E741/E402/F841 leftovers,
  and finally delete the two advisory steps and drop `continue-on-error`.