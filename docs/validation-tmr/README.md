# NASA TMR validation cases (2DZP flat plate, 2DBFS backward-facing step)

This directory documents the TensorFVM runs against the **NASA Turbulence
Modeling Resource**. The runnable package lives in `validation_tmr/`, the
artefacts in `validation_tmr/runs/` and `validation_tmr/figures/`, and the
paper-level report is generated into `validation_tmr/report.md`.

## What is real here

- The reference values and the grids are **files**, not transcribed numbers:
  `validation_tmr/reference/tmr/` holds the TMR case pages, the Kármán–Schoenherr
  skin-friction table, the Coles law-of-the-wall table and the Driver &
  Seegmiller experiment files, plus the official nested PLOT3D grids.
  `validation_tmr/fetch_reference.py` regenerates
  `validation_tmr/reference/tmr/provenance.json` with the upstream URL, byte
  count and SHA-256 of every file.
- Every grid level is a separate converged SIMPLE/Spalart–Allmaras run on an
  official TMR grid (`src/tensorfvm/tmr_grid.py` reads the multi-zone PLOT3D
  files and builds the collocated face list, registered as the `tmr-grid`
  backend).
- `validation_tmr/gate.py` recomputes every error from the stored cell fields
  and wall data, compares against the values written at run time (1e-9), applies
  the acceptance criteria and exits non-zero on failure. `--red-team` falsifies
  one converged field and proves the gate rejects it.

## One derived artefact

The TMR 2DBFS family starts at 33×33/13×33/49×57/17×57 nodes (4992 cells) and
doubles from there; the next level already needs ~80k cells for a case whose
pressure-correction system is strongly anisotropic. `derive_coarser_grid.py`
therefore applies TMR's own every-other-point rule to the coarsest published
level to obtain a sixth, still nested level (1248 cells). It is marked `derived`
in the provenance and never described as published.

## Reproduce

```bash
python3 validation_tmr/fetch_reference.py         # verify reference hashes
python3 validation_tmr/derive_coarser_grid.py     # 2DBFS coarsest level
python3 validation_tmr/run_flatplate.py --grid 35x25 --max-iterations 8000
python3 validation_tmr/run_flatplate.py --grid 69x49 --max-iterations 8000
python3 validation_tmr/run_flatplate.py --grid 137x97 --max-iterations 2500
python3 validation_tmr/run_backstep.py  --grid 5levdn --max-iterations 1500
python3 validation_tmr/run_backstep.py  --grid 4levdn --max-iterations 1500
python3 validation_tmr/run_backstep.py  --grid 3levdn --max-iterations 1600
python3 validation_tmr/gate.py --red-team         # recompute + gate + red team
python3 validation_tmr/figures.py                 # 7 figures
python3 validation_tmr/report.py --post           # build and publish the report
python3 -m pytest tests/test_tmr_validation.py    # mesh/linear/post-processing
```

## Layout

| path | content |
|---|---|
| `validation_tmr/tmr_grid.py` (in `src/tensorfvm/`) | PLOT3D → collocated face-list mesh, geometry-based patch labels |
| `validation_tmr/linear.py` | LU/ILU-preconditioned linear solve (same discretisation, ~50× faster than Jacobi-CG on stretched grids) |
| `validation_tmr/common.py` | solver construction, run loop with checkpoints, field/wall dump, restart |
| `validation_tmr/postprocess.py` | Cf(Re_θ), u⁺(y⁺), x_r/H, Cf/Cp/profiles — all conventions in one place |
| `validation_tmr/gate.py` | recomputation + acceptance gate (exit code) + red team |
| `validation_tmr/figures.py` | the seven report figures |
| `validation_tmr/report.py` | manuscript builder (numbers injected from artefacts) |
| `validation_tmr/runs/` | per-level settings, residual history, fields, wall data, metrics, `gate.json` |