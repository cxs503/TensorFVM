# TensorFVM `main` CI repair — the `lint (advisory)` gate (55 -> 0)

Mission **m4694**.  Branch `agent/m4694`, cut from `main` @ `bd06c3fb`
(`m4613: matured mission deliverables（34 个文件被门禁滤除） (#187)`).

## 1. Case, goal and what was measured

`main` was reported red on three jobs: `test (3.11)`, `test (3.12)` and
`lint (advisory)`.  The goal of this mission is a green `main`.  This report
states exactly which of the three the worktree could reproduce and repair, and
which it could not — the distinction is the point of the evidence.

| job | what it runs (`.github/workflows/ci.yml`) | reproduced here? |
|---|---|---|
| `test (3.11)` / `test (3.12)` | `pytest -q --timeout=1800 --cov=tensorfvm ...` on CPython 3.11 / 3.12, torch>=2.7, optional `gmsh` installed | **not reproduced** (full suite green, see §3.2) |
| `lint (advisory)` | `ruff check src tests scripts`, ruff pinned `>=0.15,<0.16`, rule set `E4 + E9 + F` (`pyproject.toml`) | **reproduced and repaired** (§2, §3.1) |

Environment of this worktree: loongarch64, Python 3.11.6, torch 2.7.1,
numpy 2.4.6, scipy 1.16.0, ruff 0.15.0, pytest 9.2.0 (the CI pin is a lower
bound, `pytest>=8`).  No `python3.12` interpreter and no `gmsh` wheel exist on
this host (`pip install "gmsh>=4.12"` -> `from versions: none`), so the 3.12 leg
and the gmsh-gated tests cannot be exercised here.

## 2. Root cause and repair

`ruff check src tests scripts` on a clean `bd06c3fb` worktree: **55 findings**
(48 `F401`, 5 `F841`, 2 `E401`), all of them under `tests/`:

```
$ ruff check --statistics src tests scripts          # before
48  F401  unused-import
 5  F841  unused-variable
 2  E401  multiple-imports-on-one-line
Found 55 errors.
```

`src/` was already clean.  The reason the debt survived the "lint gate clean"
PR #187 is visible in that commit's own message: *34 files were filtered by the
push gate*, and the filtered set is what carried the `tests/` half of the
mechanical fix.  So PR #187 landed the `src/` + `scripts/` repairs (its local
`All checks passed!` was true for the files it could still push) while `main`
kept 55 findings — the advisory job therefore stayed red for every later PR.

Repair, mechanical, definition-preserving:

1. `ruff check --fix src tests scripts` — 50 findings (`F401`, `E401`), i.e.
   dropping imports no consumer uses and splitting `import a,b` into one import
   per line.  33 test files touched.
2. The 5 `F841` sites, hand-inspected one by one (ruff marks its fix unsafe
   because deleting the right-hand side can drop a side effect):
   * `test_native_hybrid_ns_gauge_consistency.py` — dead `p_exact = .4 + mesh.cell_centers @ g`
     (pure matmul, no side effect) → deleted;
   * `test_suboff_geometry.py` — `gmsh = pytest.importorskip('gmsh')`: the call
     *is* the skip, so the call is kept and only the binding dropped;
   * `test_suboff_validation_integration.py` — dead `neighbor = np.full(...)` → deleted;
   * `test_two_phase_vof.py` — dead `init_vol = ...` (the sibling sites that
     *use* `init_vol` are untouched) and dead `x = (torch.arange(nx)...)` → deleted.

No production code (`src/`), no assertion, no tolerance and no test semantics
changed; `git diff --stat`: 34 files, +13 / -53 lines, all inside `tests/`.

## 3. Results

### 3.1 Lint gate (the repaired red)

```
$ ruff check src tests scripts          # after
All checks passed!        rc=0
$ ruff check --statistics src tests scripts
                  (no output: zero findings)
$ python -m compileall -q src tests scripts     # ok
$ python -m pytest -q -n 8 <the 34 touched test files>
239 passed, 7 skipped, 1 xfailed in 71.67 s
```

### 3.2 Test jobs (not reproduced — full suite green)

```
$ python -m pytest -q --timeout=1800 -p no:cacheprovider -n 24
1426 passed, 37 skipped, 2 xfailed, 50 subtests passed in 509.90 s
```

i.e. **0 failed, 0 errors** on this host at `bd06c3fb` + the lint repair.  The
m4613 evidence records `1424 passed, 37 skipped, 2 xfailed` on the same suite; the
`+2` are the `test_smoke_m4626` cases added by PR #186 after that snapshot.  The
37 skips are dominated by the optional `gmsh` runtime (absent here) and by
evidence fixtures the sparse worktree cannot reach; CI installs `gmsh`, so **those
37 skipped cases are the part of the `test (3.11)/(3.12)` jobs this worktree
cannot vouch for.**  No failing test name could be retrieved for `bd06c3fb`:
`curl https://api.github.com/...` is blocked in this sandbox, so the CI log was
not available — this is stated as a limitation, not papered over.

## 4. Figures

`lint-and-suite-before-after.png` — left: findings per rule before/after the
repair (55 -> 0); right: outcome counts of the full suite on this branch.

## 5. Notes and pitfalls

* **Sparse checkout can invent a red test.**  This worktree is a cone-limited
  checkout (`src tests scripts docs pyproject.toml README.rst`); `pytest
  --collect-only` therefore died with `ModuleNotFoundError: smoke_marker_m4626`
  because `tests/test_smoke_m4626.py` imports a root-level module that the cone
  excludes.  That file *is* tracked at `bd06c3fb` (`git ls-tree HEAD`), so it is
  a worktree artefact, not a repository defect; materialising the file (identical
  to the blob) makes collection clean.  Any "missing fixture" red seen from a
  sparse worktree must be cross-checked with `git ls-tree` before it is believed.
* **The push gate, not the code, decides what a repair PR really fixes.**  Check
  the merged tree (`git show <merge>:<path>` / fresh `ruff check`) instead of the
  PR body's claim.
* The host is shared (load average ~600-900 on 128 cores), so wall-clock numbers
  here are not CI numbers: a serial run reached only 43 % in ~25 min while the
  24-worker run finished in 510 s.  Timing-sensitive assertions were not the
  subject of this repair.
* Scope kept to `tests/`; `pyproject.toml`'s lint config was not relaxed (no new
  `per-file-ignores`), so the gate keeps its meaning.

## 6. Error judgement against the baseline

| quantity | baseline | this branch | error | < 3 %? |
|---|---|---|---|---|
| `ruff check` findings (`src tests scripts`, 0.15.0) | 55 | **0** | 0 % residual (relative to the 0 target: satisfied exactly) | **yes (0 findings)** |
| failing tests in the full suite | CI reports failure on 3.11/3.12 | **0 failed / 0 error** locally (1426 passed) | — | **not verifiable for the CI-only legs** (no 3.12 interpreter, no `gmsh`, CI log unreachable) |
| the 34 touched test files | green | 239 passed, 7 skipped, 1 xfailed | 0 regressions | **yes** |
| `python -m compileall -q src tests scripts` | ok | ok | — | **yes** |

**Honest verdict**: the `lint (advisory)` red is fixed and proven (55 -> 0 with
the repo's own pinned ruff minor, plus a re-run of every touched test file).
The `test (3.11)` / `test (3.12)` red is **not** reproduced on this host — the
suite is green on 3.11/torch 2.7.1 — so this mission can close the lint leg with
evidence but cannot claim the test legs are green; the residual exposure is
exactly the 37 skipped cases (`gmsh`-gated) and the 3.12 interpreter, both of
which need a CI runner (or a gmsh-capable, 3.12-capable host) to settle.
