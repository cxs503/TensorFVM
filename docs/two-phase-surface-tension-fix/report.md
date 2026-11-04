# Two-Phase Surface Tension: Verification Re-Assessment

## 1. Background

The test `test_static_drop_young_laplace_jump` in `tests/test_two_phase_surface_tension_verification.py`
is marked as `xfail(strict=True)` with the reason that "the explicit capillary source is added to the
transient predictor but the projection step does not hold it in balance, so a static drop diverges
within a few steps instead of carrying dp = σ/R".

The original audit (`docs/two-phase-surface-tension-audit/summary.json`) reported exponential
divergence: pressure jumps of ~10²³ Pa vs the exact 0.35 Pa after 10 steps.

## 2. Re-Assessment (2026-11-04)

Running the same case with the current solver code:

| Grid | Exact dp (Pa) | Computed dp (Pa) | Relative Error | Diverged? |
|------|---------------|-------------------|----------------|-----------|
| n=24 | 0.35 | 0.3559 | 1.68% | No |
| n=32 | 0.35 | 0.3438 | 1.78% | No |

**The defect appears to have been resolved.** The solver now produces a pressure jump
within 3% of the exact Young-Laplace value after 10 time steps, with no divergence.

## 3. Possible Explanations

1. **Solver update**: The projection step or CSF force computation may have been
   corrected since the original audit.
2. **Different parameters**: The original audit may have used different viscosity,
   density ratio, or time step.
3. **Longer time needed**: The divergence may manifest after more than 10 steps.

## 4. Recommendation

The strict xfail should be re-evaluated. A longer run (100+ steps) should be performed
to confirm stability. If the solver remains stable, the xfail should be removed and
replaced with a passing test.

## 5. Artifacts

- `scripts/audit_two_phase_surface_tension_defect.py`: Reproduction script
- `docs/two-phase-surface-tension-fix/outputs/defect_summary.json`: Quantified results
- `docs/two-phase-surface-tension-fix/outputs/divergence_history.png`: Divergence plot
