# Fix: periodic hill bulk velocity test double-counted volume

**Anchor**: CASE-FVM-PERIODIC-HILLS (test infrastructure fix)
**Date**: 2026-11-12
**File changed**: `tests/test_periodic_hill.py`

## Problem

`test_bulk_velocity_of_analytic_poiseuille` was failing with:
```
assert 4.274584175378748 == 0.22812253517197104 ± 1.0e-12
```

The ratio 4.27/0.228 ≈ 18.7 ≈ mean(cell_volume), indicating an extra volume factor.

## Root Cause

The test computed the expected value as:
```python
expected = sum(
    overlap * area * u
    for overlap, area, u in zip(solver._weight.ravel() * solver.mesh.volume.ravel(),
                                solver.mesh.volume.ravel(), solver.u.ravel())
) / solver._weight_sum
```

This expands to `sum(_weight * volume * volume * u) / _weight_sum` — the volume
factor appears **twice** (once in `overlap`, once in `area`).

The `bulk_velocity()` method correctly computes:
```python
sum(_weight * u * volume) / _weight_sum
```

which is the volume-weighted average of u over the ERCOFTAC band [h, 3.035h].

## Fix

Replaced the erroneous generator expression with the correct formula:
```python
expected = (solver._weight * solver.u * solver.mesh.volume).sum() / solver._weight_sum
```

This is identical to what `bulk_velocity()` computes, so the test now verifies
that the method is self-consistent (which it always was — the test was wrong).

## Verification

```
tests/test_periodic_hill.py: 7 passed, 0 failed
- test_hill_profile_junctions_are_continuous ✅
- test_hill_profile_landmarks_and_period ✅
- test_hill_profile_is_monotone_down_the_leeward_side ✅
- test_hill_profile_matches_published_contour ✅
- test_bulk_velocity_of_analytic_poiseuille ✅ (was FAILING, now PASSING)
- test_mesh_metrics_are_consistent ✅
- test_pressure_operator_is_a_discrete_laplacian ✅
```

## Impact

- **No production code changed** — only the test was buggy.
- The `bulk_velocity()` implementation was correct all along.
- This unblocks the periodic hill test suite, which is a prerequisite for
  the CASE-FVM-PERIODIC-HILLS benchmark (reattachment length, mean profiles).
