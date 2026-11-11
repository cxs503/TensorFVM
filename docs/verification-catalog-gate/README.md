# Verification-catalog gate: making the index of verified capabilities machine-checked

## 1. Case: the repository's own index had drifted

`docs/verification-catalog.md` is the document TensorFVM points at when it claims
"capability X is verified, capability Y is not". It is hand-written prose, and
nothing checked it. Two classes of silent lie were live in the tree at commit
`7b7418dc`:

| defect class | instances found (before) | example |
|---|---:|---|
| dead provenance pointer (path quoted as evidence does not exist) | 2 | §2.5 cites `tests/test_multi_element.py` (file is `test_multi_element_flow.py`); §5.1 cites `tests/test_mesh_quality.py` (file is `test_mesh_quality_metrics.py`) |
| PASS entry with no live test file behind it | 7 | §1.3 "Multiple channel tests across test suite", §7.1 "Extensive across test suite", §7.2 "Verification scripts", §9.1–9.3 (no test pointer at all) |
| *Current Gaps* item with no machine-readable status | 5 / 5 | §10.1 items 2 and 3 were **stale**: compressible flow and moving-mesh ALE *are* verified in this repository (`src/tensorfvm/compressible/verification.py`, `src/tensorfvm/moving_mesh.py` + `tests/test_moving_mesh_ale.py`, `tests/test_ale_transport_analytic.py`, `docs/pmm-ale-cfd/`), yet the index still declared them unverified |

The third row is the one that matters for evidence discipline: the index is
quoted to justify *not* doing work, so a stale "unverified" entry both
misreports the capability ledger and invites a duplicate build.

This is a documentation/verifier-infrastructure task: there is no physical
field, no mesh and no numerical parameter, hence no field-contour figure is
produced (the deliverable is a gate plus the repaired index; the "result" is a
before/after verdict table, not a field).

## 2. What was built

`src/tensorfvm/verification/catalog_gate.py` parses the catalog and turns three
statements into executable checks:

* **pointer** — every repository-relative path quoted inside an entry
  (`src/…`, `tests/…`, `docs/…`) must exist.
* **evidence** — an entry marked ✅ PASS must cite at least one existing
  `tests/test_*.py`. A PASS backed by the prose "across the test suite" is
  unsupported, because it names nothing that can go red.
* **gap** — every *Current Gaps* item must carry
  `[[gap-status: open]]` or `[[gap-status: closed-by <paths>]]`, and every
  `closed-by` pointer must exist. A gap can therefore only go stale by breaking
  a machine check.

The parser is intentionally a dumb regex over headings/backticks: the catalog
stays the single source of truth, nothing is duplicated into a second registry.

CLI: `PYTHONPATH=src python -m tensorfvm.verification.catalog_gate --json docs/verification-catalog-gate/artifacts/catalog_gate.json`
(exit code 1 on failure, so it drops straight into CI).

## 3. Result (programmatic, from `artifacts/catalog_gate.json`)

| check | before (7b7418dc) | after |
|---|---:|---:|
| entries parsed | 31 | 31 |
| entries marked PASS | 26 | 26 |
| quoted pointers | 70 | 84 |
| dangling pointers | **2** | **0** |
| PASS entries without a live test file | **7** | **0** |
| gaps without a status annotation | **5** | **0** |
| gaps open / closed-by | 0 / 0 | 3 / 2 |
| verdict | **FAIL** | **PASS** |

Gap ledger after the repair: `3D Lid-Driven Cavity`, `Combustion`, `Radiation`
remain `open`; `Compressible Flow` and `Moving Mesh` are `closed-by` paths that
the gate resolves.

Catalog repairs made (all are index hygiene, no physics changed): the two dead
test paths repointed at the files that exist; the prose-only "tests" citations
replaced by real test modules (`tests/test_native_developed_channel.py`,
`tests/test_sparse_simple.py`, `tests/test_native_collocated3d.py`,
`tests/test_fct_conservation_audit.py`,
`tests/test_expansion_momentum_closure.py`,
`tests/test_native_energy_passivity.py`, …); the five gap items annotated.

## 4. Tests

`tests/test_verification_catalog_gate.py` — 6 cases, all green:

* the committed catalog passes the gate (guards against future drift);
* the gate *fails* on a synthetic dead pointer, on a synthetic PASS-without-test
  entry, and on an un-annotated gap plus a dangling `closed-by` pointer
  (negative controls: the gate is not vacuous);
* gap-status parsing of the shipped catalog;
* CLI exit code 0 on the shipped catalog.

```
PYTHONPATH=src python3 -m pytest tests/test_verification_catalog_gate.py -q   # 6 passed
```

## 5. Caveats / scope

* The gate checks **provenance liveness, not truth**: it proves that the cited
  test file exists, not that the test asserts the number the entry quotes. It
  complements — it does not replace — `tensorfvm.verification.doc_claims`
  (which binds report numbers to artifact leaves) and the per-capability
  reference rechecks.
* Parsing is markdown-shaped: a future re-styling of the catalog (e.g. tables
  instead of `**Implementation**` lines) would silently reduce the pointer
  population. The entry count (≥30) and pointer count are asserted in the tests,
  so a collapse in coverage is visible.
* Whether the two "closed" gaps are *fully* closed is bounded by the cited
  evidence: compressible flow is closed for the inviscid/EOS verification
  surface and moving mesh for space-conservation + free-stream preservation; the
  open-boundary ALE *flow* solver is still missing and is recorded separately in
  `docs/pmm-ale-cfd/summary.json` (C1/C2 FAIL), so the annotation does not
  overstate it.

## 6. Error verdict

The quantity this task delivers is a **count of defects in the index**, not a
physical observable, so the "is the key number within 3%?" question is answered
against the primary evidence instead: before = 2 dangling pointers / 7 unsupported
PASS entries / 5 un-annotated gaps; after = 0 / 0 / 0, both numbers produced by
the gate itself from the committed catalog (`artifacts/catalog_gate.json`) and
pinned by 6 tests. No physical benchmark is claimed or implied here.