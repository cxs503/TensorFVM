# CASE-FVM-CHANNEL-KMM180 — artifact hash contract and byte reproducibility

Mission m18277 (anchor `CASE-FVM-CHANNEL-KMM180`, self-start). This note records
two defects found in the committed artifact set and what was changed, with the
before/after evidence. No graded number moved: every RMS/Cf/U_bulk value in
`benchmark.json`, `metrics.csv`, `profiles.csv`, `stations.csv` is identical to
the numbers already published (the regeneration reproduced them exactly).

## Defect 1 — the artifact gate was red on `main` (integrity, not physics)

`docs/case-fvm-channel-kmm180/REPORT.md` had a hand-appended section

```
## Verification (2026-11-13 re-open)
All 53 unit tests pass; artifact gate clean (sha256 consistent across
benchmark.json / metrics.csv / profiles.csv / stations.csv / manifest.json /
oss_solutions catalog).  The case is ready for PR re-submission.
```

`REPORT.md` is a *generated* artifact and is listed in `manifest.json -> sha256`.
The hand edit changed its bytes without regenerating the manifest, so

* `PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180.py --check` printed
  `ARTIFACT GATE: 1 problem(s) - REPORT.md does not match its manifest sha256`;
* `tests/test_channel_kmm180_artifacts.py::test_committed_artifacts_pass_the_gate`
  failed on `main` (1 failed, 11 passed).

The appended sentence was therefore itself false at the time it was written
("artifact gate clean"). Fix: `REPORT.md` is regenerated from the generator
(`tensorfvm.benchmark_channel_kmm1987`), which is the only writer of that file;
`manifest.json` is regenerated in the same run, so the gate is green again:

```
ARTIFACT GATE: clean (benchmark/metrics/profiles/stations/manifest/catalog consistent)
```

Verification command: `PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180.py --check`.

## Defect 2 — the pinned artifacts were not byte-reproducible

`manifest.json` pins `sha256` for `benchmark.json`, `metrics.csv`,
`profiles.csv`, `stations.csv`, `results.csv`, `REPORT.md`. Three of those files
carried values that change on every run:

| file | volatile content (before) |
|---|---|
| `benchmark.json` | `elapsed_s`, `matrix_elapsed_s`, `repro_cmd` (contains the `--out` path) |
| `results.json` | `elapsed_s` |
| `manifest.json` | `repro_cmd` (contains the `--out` path) |

Measured consequence (two full driver runs, same machine, same code):

```
docs/case-fvm-channel-kmm180/benchmark.json   vs  a fresh run: DIFF
  -  "elapsed_s": 53.36,          +  "elapsed_s": 55.63,
  -  "matrix_elapsed_s": 34.0,    +  "matrix_elapsed_s": 34.91,
  -  "repro_cmd": "... --out docs/case-fvm-channel-kmm180"
  +  "repro_cmd": "... --out /tmp/k180"
metrics.csv / profiles.csv / stations.csv / results.csv: identical
```

So the documented reproducer could never reproduce the pinned hash of the
richest artifact: the gate would fail after every legitimate regeneration, and
the only ways out were (a) exempt `benchmark.json` from the gate (which drops
tamper detection on the file that carries all the numbers) or (b) hand-edit the
manifest (which is what produced Defect 1). The self-contradictory state was
already visible in the code: `results.json` is hashed *and* documented as
"informational, not a byte-reproducibility target".

## The contract now enforced

1. Hashed artifacts contain no wall clock and no invocation path. The volatile
   values are published in `timings.json`, which `manifest.json` lists under
   `volatile_artifacts` and does **not** hash.
2. `manifest.json -> repro_cmd` is the documented, path-independent reproducer;
   the actual invocation (with its `--out`) is recorded in `timings.json ->
   invocation`.
3. The gate (`scripts/run_case_fvm_channel_kmm180.py --check`) fails if
   `timings.json` is missing (values must be published, not dropped), if it is
   hashed, or if `benchmark.json` / `results.json` carry any of
   `elapsed_s`, `matrix_elapsed_s`, `repro_cmd`.
4. `tests/test_channel_kmm180_reproducibility.py` pins the behaviour: two
   `--quick` driver runs into different `--out` directories must produce
   byte-identical `benchmark.json`, `results.json`, `results.csv`, `metrics.csv`,
   `profiles.csv`, `stations.csv`, `REPORT.md` and `manifest.json`.

Before/after, measured:

| check | before | after |
|---|---|---|
| `--check` on committed artifacts | 1 problem (REPORT.md sha256) | clean |
| `test_committed_artifacts_pass_the_gate` | FAIL | pass |
| two runs, different `--out` | pinned hashes differ for 2 files | all pinned bytes identical |
| volatile numbers still published | yes (inside hashed files) | yes (`timings.json`, unhashed) |

## Honest limitations

* Byte reproducibility is demonstrated on this machine, one Python/dependency
  set; cross-platform float formatting is not covered (values are written with
  fixed `repr`/`%.17g` formats, but no CI matrix runs here).
* The wall clock is still *measured*; it is only moved out of the pinned files.
  A run whose numbers legitimately change (a closure change) still legitimately
  changes `benchmark.json`, which is the intended behaviour: the hash pins the
  numerics, not the timing.
* This work does **not** change the anchor's scientific verdict. The case still
  reads QUALIFIED: SST passes the literal window (`y+ in [30, centreline]`)
  grid-robustly, and fails the physical log layer (`y+ in [30, 0.2 Re_tau]`),
  where the grid audit classifies `sst/log_layer` as *grid-decided* (the n=64
  PASS is a discretisation artefact). Both numbers remain published side by side.

## Reproduce

```bash
PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180.py            # regenerate, then gate
PYTHONPATH=src python scripts/audit_case_fvm_channel_kmm180_grid.py     # grid audit (reads benchmark.json)
PYTHONPATH=src python scripts/run_case_fvm_channel_kmm180.py --check     # gate only (offline, seconds)
PYTHONPATH=src python -m pytest tests/test_channel_kmm180_reproducibility.py \
       tests/test_channel_kmm180_artifacts.py tests/test_channel_kmm180_loglaw.py \
       tests/test_channel_kmm180_windows.py tests/test_channel_kmm1987.py -q
```