# Taylor-Green Vortex: Negative Control Qualification Fix

**Date:** 2026-11-04  
**Mission:** m3466  
**Status:** ✅ Completed

## Problem

The TGV verification study (`docs/periodic-tgv/study.json`) reported `negative_control_qualified: false`, meaning the verification gate was not discriminating — even a challenging high-Reynolds case passed all gates, making the 3% acceptance criterion vacuous.

### Root Cause

Two issues in `src/tensorfvm/verification/periodic_tgv.py`:

1. **Limit case used finest grid:** The negative control ran at `n=max(grids)=64`, where even Re=3142 is well-resolved and all errors are tiny.

2. **Only velocity checked:** The pass/fail logic only examined `velocity_error`, which is nearly exact on the staggered MAC grid (discretely divergence-free initial condition). Pressure and vorticity errors were ignored.

## Solution

### Code Changes

**File:** `src/tensorfvm/verification/periodic_tgv.py`

1. **Use coarsest grid for limit case:**
   ```python
   # Before: limit = run_case(max(grids), ...)
   # After:  limit = run_case(min(grids), ...)
   ```
   At coarse resolution (n=16), high-Re flows challenge the scheme's pressure and vorticity accuracy.

2. **Check multiple quantities:**
   ```python
   # Before: limit_passed = passed(limit.velocity_error) and limit_failure is None
   # After:  limit_all_passed = (passed(limit.velocity_error) 
   #                            and passed(limit.pressure_error)
   #                            and passed(limit.vorticity_error)
   #                            and passed(limit.energy_error)
   #                            and limit_failure is None)
   ```

3. **Record which quantities failed:**
   Added `pressure_3_percent_passed`, `vorticity_3_percent_passed`, `energy_3_percent_passed`, and `all_quantities_passed` to the limit case output.

### Test Coverage

**File:** `tests/test_periodic_tgv.py`

Added `test_negative_control_is_qualified()` which verifies:
- The negative control is qualified (`negative_control_qualified == True`)
- The limit case fails on at least one quantity (`all_quantities_passed == False`)
- Pressure or vorticity fails (not just velocity)
- The expected failure is observed

Updated `test_published_study_gates_are_consistent()` to check `all_quantities_passed` instead of just `velocity_3_percent_passed`.

## Results

### Updated Study (grids=16,32)

**Verification Status:**
- ✅ Physical accuracy qualified: `True`
- ✅ Negative control qualified: `True`

**Limit Case (n=16, Re=3142):**
| Quantity | Error | Passed (<3%) |
|----------|-------|--------------|
| Velocity | 0.0051% | ✅ Yes |
| Pressure | 3.794% | ❌ No |
| Vorticity | 5.816% | ❌ No |
| Energy | 0.0102% | ✅ Yes |

**Conclusion:** The gate is now discriminating. The limit case fails on pressure (3.8%) and vorticity (5.8%), proving the 3% criterion is not vacuous.

## Verification

All tests pass:
```bash
$ python3 -m pytest tests/test_periodic_tgv.py -v
========================= 9 passed in 29.98s =========================
```

## Impact

- **Before:** Verification was incomplete; the gate could not detect physical accuracy degradation.
- **After:** Verification is fully qualified; the gate discriminates between acceptable and unacceptable solutions.

This fix closes a critical verification gap and strengthens confidence in the periodic MAC kernel's physical accuracy claims.

## Files Modified

1. `src/tensorfvm/verification/periodic_tgv.py` — Core fix
2. `tests/test_periodic_tgv.py` — New test + updated assertion
3. `docs/periodic-tgv/study.json` — Regenerated with fix
4. `docs/periodic-tgv/NEGATIVE_CONTROL_FIX.md` — This document
