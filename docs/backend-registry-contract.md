# Backend registry contract: register / resolve / fallback

Status: **enforced by `tests/test_backend_registry_contract.py`** (18 tests,
32 sub-cases), mutation-proven by
`scripts/verify_backend_registry_contract_negatives.py` (11 / 11 injected
defects turn the suite red).

Scope: `src/tensorfvm/backend_registry.py` (`register_backend`, `get_backend`,
`registered_backends`, `build_mesh`) plus the two registry consumers that decide
which numerics actually run — `SimpleSolver.__new__` (factory hook / built-in
fallback) and `SolverConfig.reference_length` / `SolverConfig.residual_length`
(characteristic-length rules).

`tests/test_backend_registry.py` already covered the happy path (a downstream
package can register and be routed). This module pins the **boundaries**: every
rejection is typed, every failed registration is inert, illegal names never
resolve, and the built-in collocated path survives as the fallback.

## 1. Contract

### Register (`register_backend`)

| # | Rule | Rejection |
|---|---|---|
| R1 | Argument must be a `SolverBackend` | `TypeError: backend must be a SolverBackend instance` |
| R2 | Name must be a non-blank `str` | `ValueError: backend name must be nonempty` |
| R3 | `min_nx` / `min_ny` / `nx_multiple` must be positive `int` (not `bool`, not float) | `ValueError: backend grid-size constraints must be positive integers` |
| R4 | Capability flags must be real `bool` (not `0/1`, not `"yes"`) | `TypeError: backend capability ... must be bool` |
| R5 | Factories must be callable or `None` | `TypeError: ... must be callable or None` |
| R6 | Non-`cartesian` backends must supply both factories | `ValueError: non-cartesian backends must provide a solver_factory/mesh_factory` |
| R7 | A custom `cartesian` entry must supply **both** factories or **neither** | `ValueError: a custom cartesian backend must provide both solver and mesh factories` |
| R8 | `requires_mesh_file` implies a `mesh_factory` | `ValueError: a backend requiring mesh_file must provide a mesh_factory` |
| R9 | Cylinder constraints imply `uses_cylinder=True` | `ValueError: cylinder constraints require uses_cylinder=True` |
| R10 | `requires_cylinder` and `forbids_cylinder` are mutually exclusive | `ValueError: backend cannot both require and forbid a cylinder` |
| R11 | Length rules must be a known rule string or a callable | `TypeError` / `ValueError: unsupported characteristic-length rule` |
| R12 | `required_mesh_masks` must be a tuple of nonempty strings | `ValueError: required_mesh_masks must be a tuple of nonempty strings` |
| R13 | A duplicate name is rejected unless `replace=True` | `ValueError: solver backend ... is already registered` |
| R14 | Rejection is atomic: no entry, no shadowing, mapping unchanged | asserted by snapshot comparison in 18 sub-cases |
| R15 | Names are stored **verbatim** (no strip / case folding) | asserted: `" spaced "` does not answer to `"spaced"` |

### Resolve (`get_backend`, `registered_backends`)

| # | Rule |
|---|---|
| P1 | Resolution is identity-stable: the registered object comes back, twice. |
| P2 | Unknown names raise `ValueError("unknown mesh/solver backend: <name!r>")`; the original `KeyError` is chained as `__cause__`, and no registry internals appear in the message. |
| P3 | Illegal names (`None`, `""`, `"   "`, wrong case, padded, `list`, `int`) all raise that same `ValueError`; unhashable names chain a `TypeError` as `__cause__`. |
| P4 | `registered_backends()` returns a fresh, sorted `tuple[str, ...]`. |
| P5 | The error surfaces at `SolverConfig(mesh_type=...)` construction, i.e. before any tensor work. |

### Fallback (`SimpleSolver`, `build_mesh`, length rules)

| # | Rule |
|---|---|
| F1 | `mesh_type="cartesian"` has no factory ⇒ `SimpleSolver(config)` is a real `SimpleSolver` (built-in collocated path preserved). |
| F2 | A registered `solver_factory` takes over `SimpleSolver(config)`, and receives the very `config` object. |
| F3 | `SimpleSolver` **subclasses** bypass the hook (`cls is SimpleSolver` guard), so internal derived solvers cannot be hijacked by the registry. |
| F4 | `build_mesh` needs a `mesh_factory`: `cartesian` reports `does not provide a collocated mesh`; unknown names report the resolve error. |
| F5 | `build_mesh` routes to the registered factory and validates its output against the `Mesh2D` contract at the boundary (`TypeError: Mesh2D is missing required field ...`); factory exceptions propagate unchanged. |
| F6 | `residual_length=None` falls back to the backend's `reference_length` rule; an explicit `residual_length` (e.g. `"height"`) is independent of `reference_length` (e.g. `"length"`). |

Test isolation: the registry is process-global and has no public unregister API,
so the contract tests snapshot `backend_registry._BACKENDS` under its `RLock` in
`setUp` and restore it in `tearDown`; test order and other modules' registrations
cannot influence the result.

## 2. Negative self-proof (mutation testing)

Command:

```
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/verify_backend_registry_contract_negatives.py
```

Full transcript: `docs/backend-registry-contract/negative_injection.txt`
(11 mutations, each applied to the working tree, run, then restored).

| # | Injected defect | Observed |
|---|---|---|
| 1 | duplicate-name guard removed | RED `test_duplicate_registration_requires_explicit_replace` |
| 2 | duplicate guard ignores `replace=False` | RED same test |
| 3 | unknown-name `ValueError` downgraded to `KeyError` | RED `test_unknown_name_raises_configuration_friendly_valueerror` |
| 4 | blank-name guard weakened to a non-str check | RED `test_registration_rejects_blank_names_without_mutation` |
| 5 | `required_mesh_masks` guard removed | RED `test_invalid_specifications_are_rejected_and_never_registered` (2 failed) |
| 6 | cylinder/`uses_cylinder` consistency guard removed | RED same test (7 failed) |
| 7 | names silently `.strip()`ed on write | RED `test_names_are_stored_and_looked_up_verbatim` |
| 8 | `SimpleSolver` factory hook disabled | RED `test_registered_factory_takes_over_the_simple_solver_entry_point` |
| 9 | factory hook no longer excludes subclasses | RED `test_subclasses_bypass_the_registry_factory_hook` |
| 10 | `build_mesh` missing-factory guard removed | RED `test_build_mesh_reports_missing_and_unknown_mesh_backends` |
| 11 | `residual_length` fallback dropped | RED `test_reference_length_fallback_chain_uses_registry_rules` |

Baseline (unmutated tree): contract suite green. A mutation that stays green is
reported as `GREEN (mutation NOT caught)` and the script exits non-zero, so the
proof cannot silently degrade into a no-op.

## 3. Verification log

| Check | Command | Result |
|---|---|---|
| contract + legacy registry tests | `TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/test_backend_registry_contract.py tests/test_backend_registry.py -q` | 23 passed, 32 subtests passed |
| mutation proof | `python3 scripts/verify_backend_registry_contract_negatives.py` | 11/11 red, baseline green |
| lint | `ruff check .` | clean (0 errors; `origin/main` baseline was 55 pre-existing `F401/F841/E401`) |
| full collection | `pytest tests --collect-only -q` | 1480 tests collected |

### Lint debt paid down in this branch

`ruff check .` was **red on `origin/main`** (55 errors: 48 × `F401`, 5 × `F841`,
2 × `E401`, spread over 34 test modules) — the repo-wide gate in
`pyproject.toml` (`select = ["E4","E9","F"]`) could not pass on main. This branch
clears it: unused imports and dead local assignments only, verified by
`ruff check .` (0 errors), by full collection (1480 tests, no import errors) and
by running the touched modules (`test_two_phase_vof`,
`test_suboff_validation_integration`, `test_native_hybrid_ns_gauge_consistency`,
`test_mac_time_order`, `test_wall_traction`, `test_overset_poisson`,
`test_wall_cell_gradient_attribution`, `test_periodic_mac`,
`test_suboff_validation`, `test_porous_drag`, `test_fct_conservation_audit`,
`test_fourth_round_audit`, `test_gci_band_coverage`, `test_annular_verification`).
No assertion, tolerance, skip or test was removed or relaxed.

Known pre-existing breakage, **not** touched here: `tests/test_smoke_m4626.py`
fails collection on `ModuleNotFoundError: smoke_marker_m4626` (landed on main
with the m4626 deliverables). It is excluded from the collection count above and
left for its owner rather than deleted.

## 4. Reproduce

```
cd <checkout>
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 -m pytest tests/test_backend_registry_contract.py tests/test_backend_registry.py -q
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python3 scripts/verify_backend_registry_contract_negatives.py
ruff check .
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` keeps torch from autoloading out-of-tree
`PrivateUse1` device backends at import, so the registry contract is exercised
against a plain CPU build.