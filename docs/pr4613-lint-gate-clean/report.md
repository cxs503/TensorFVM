# main CI repair: the ruff lint gate (161 → 0 findings)

Mission m4613.  Branch `agent/m4613`, cut from `main` @ `d3ffecc9`.

## 1. What was red, and why

`main` CI at `d3ffecc9` was red on the `lint (advisory)` job — the only job that
was red; `test (3.11)` / `test (3.12)` were green.  The job runs the rule set
declared in `pyproject.toml` (`[tool.ruff.lint] select = ["E4", "E9", "F"]`),
pinned to `ruff>=0.15,<0.16`:

```
$ ruff check src tests scripts        # ruff 0.15.0, the pinned minor
Found 165 errors.
   99  F401  unused-import
   44  F841  unused-variable
   20  E401  multiple-imports-on-one-line
    1  E402  module-import-not-at-top-of-file
    1  F541  f-string-missing-placeholders
```

The job carries `continue-on-error: true`, which is exactly why the debt could
accumulate unnoticed: the gate never turned *itself* red locally, but the check
suite still reported failure on the branch, so every PR inherited a red main.

## 2. How it was repaired

Three passes, all mechanical, no behaviour change intended:

1. **Auto-fixable rules (120 findings).** `ruff check --fix src tests scripts`
   for F401 / E401 / F541.  Splitting `import argparse,json,sys` into three
   lines and dropping imports that no consumer uses are definition-preserving
   edits; the tree was byte-compiled afterwards (`python -m compileall -q src
   tests scripts`) and the full suite was run.
2. **Dead bindings (44 F841), hand-edited site by site.**  Ruff marks its F841
   fix "unsafe" because deleting the right-hand side could drop a side effect,
   so each of the 44 sites was inspected.  Two categories:
   * *side-effect-free RHS* (numpy/torch expressions, attribute reads, list
     comprehensions): the binding was deleted;
   * *call whose effect matters*: the call was kept and only the binding
     dropped — `gmsh=pytest.importorskip('gmsh')` → `pytest.importorskip('gmsh')`
     (the skip must still fire), and the duplicated
     `net = transport._net_flux(...)` in `fct_conservation_audit.py` (identical
     to the `_net_flux` call on the preceding line) was removed.
   Deleting one binding cascades (`o`, `f`, `ni`, `ids`, `oi`, `chi`, `fv1`,
   `ladder`, `finest`, `centres`, `math` became dead in turn); those second-order
   sites were resolved in a second, explicitly anchored pass, and ruff was rerun
   until clean.
   One edit was caught *by the suite*, not by lint: `richardson.py` had
   `r21, r32 = h[1]/h[0], h[2]/h[1]` with only `r32` unused — dropping the whole
   line removed `r21`, which the surrounding function still uses (F821 showed up
   on the next ruff run and the binding was restored as `r21 = h[1]/h[0]`).
4. **E402 in `tests/test_two_phase_projection_operator_matrix.py`** is
   deliberate: the module imports `tensorfvm.two_phase_solver` *below*
   `pytest.importorskip("scipy.sparse")` so a missing optional dependency
   degrades to a skip instead of a collection error.  Fixed the way the repo
   already handles that pattern (`tests/test_lbm_reference_audit.py`): a
   `[tool.ruff.lint.per-file-ignores]` entry with the reason inline, not a
   hoisted import.

## 3. Knock-on: the SIMPLEC evidence pin had to be regenerated

`tests/test_simplec_artifacts.py::test_provenance_pins_the_sources_that_produced_the_numbers`
pins the SHA-256 of the sources behind the shipped mobility evidence.  My only
edit to `src/tensorfvm/verification/simplec_channel.py` was deleting an unused
`import math`, which of course moved its hash and turned that pin red — by
design ("rerun the script, not the prose").  So the evidence was regenerated
with the documented command:

```
python scripts/verify_simplec_mobility.py --output scripts/artifacts/simplec-mobility
→ passed=true, guards 11/11, mutation-sensitive=true, total 99 869 B (< 200 kB cap)
```

Regeneration exposed a **reproducibility defect** in the generator: the
fixed-point comparison table was built from

```python
for label in {c["label"] for c in CONFIGURATIONS}:      # a set of strings
```

so the order of the shipped table depended on `PYTHONHASHSEED` — the same
numbers produced a different `report.md` on every run (HEAD's own table is in a
hash order: `simple-tuned, simple-default, simplec-a07-p10, simplec-a08-p10`).
The iteration now follows the declared `CONFIGURATIONS` order
(`dict.fromkeys(...)`), which is both deterministic and the physically
meaningful order (SIMPLE baselines, then SIMPLEC variants).

**Numbers unchanged.** A structural diff of HEAD's `summary.json` against the
regenerated one, excluding `provenance` / `elapsed` / guard timings, is empty
(0 differing leaves, programmatic walk), and the token-count ladders
(`41, 43, 32, 49` at ny=12 and `87, 53, 87, 53` at ny=24) reproduce exactly.

## 4. Verification

| check | command | result |
|---|---|---|
| lint gate | `ruff check src tests scripts` (ruff 0.15.0, pinned minor) | **All checks passed!** (0 findings, was 165) |
| byte-compile | `python -m compileall -q src tests scripts` | OK |
| full suite | `pytest -q --timeout=1800 -n 48` (1462 tests) | see `verification.txt` |
| artifact guards | `pytest -q tests/test_simplec_artifacts.py tests/test_simplec_mobility.py` | 19 passed |
| evidence regeneration | `verify_simplec_mobility.py` | passed, guards 11/11, mutation-sensitive |
| evidence fidelity | structural diff vs `HEAD:summary.json` | 0 numeric differences |

Environment: python 3.11.6, torch 2.7.1, numpy 2.4.6, scipy 1.16.0, ruff 0.15.0
(all matching the pin in `pyproject.toml`; CI additionally runs 3.12).

## 5. Scope / what was *not* done

* No rule was added to or removed from the gate, and `continue-on-error` was
  left in place for the lint job: promoting it to a hard gate is a separate
  decision for the owner (the workflow comment says "one release cycle").  With
  0 findings the job is now green either way.
* `ruff format --check` was not introduced — the repository has ~9.4k
  deliberately dense one-line statements (see `docs/pr109-lint-gate/report.md`)
  and reformatting would bury every real diff.
* No solver physics changed.  Every edit is an unused import, an unused
  binding, an import split, or a set→declared-order iteration; the full suite
  and the artifact guards are the evidence.