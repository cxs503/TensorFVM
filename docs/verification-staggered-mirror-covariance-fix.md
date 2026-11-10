# TensorFVM Staggered Grid Mirror Covariance Fix

**Date**: 2026-11-09  
**Task**: Self-start m7533 (tp-arch-1 strategy seat)  
**Nature**: Verification debt closure - solver defect fix

## 1. Problem Statement

The sudden expansion verification (m6320) identified spurious asymmetry in symmetric geometries:
- Re=5: mirror_rel = 5.5% (should be ~0 for steady symmetric flow)
- Re=100: mirror_rel = 17%

Root cause: `StaggeredSimple._u_on_v` had an off-by-one stencil error and asymmetric wall treatment, breaking mirror covariance of the v-momentum operator.

## 2. Fix Applied

**File**: `src/tensorfvm/staggered_simple.py`, method `_u_on_v`

**Before**:
```python
def _u_on_v(self, u):
    ua = 0.5 * (u + _shift(u, 0, -1))  # zeros at j=0
    west = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
    east = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
    west[:, 1:] = ua[: self.nx, :]     # j=0 left as zero
    east[:, 1:] = ua[1:, :]
    return west, east
```

**After**:
```python
def _u_on_v(self, u):
    ua = 0.5 * (u + _shift_clamped(u, 0, -1))  # clamped at boundaries
    west = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
    east = np.zeros((self.nx, self.ny + 1), dtype=DTYPE)
    west[:, : self.ny] = ua[: self.nx, :]       # j=0 gets value
    east[:, : self.ny] = ua[1:, :]
    west[:, self.ny] = u[: self.nx, -1]         # top wall: adjacent cell
    east[:, self.ny] = u[1:, -1]
    return west, east
```

**Key changes**:
1. `_shift` → `_shift_clamped`: replicates edge values instead of zeroing
2. `[:, 1:]` → `[:, :ny]`: assigns to j=0..ny-1 instead of j=1..ny
3. Explicit top wall treatment: uses adjacent cell value

This makes the stencil mirror-covariant: under j → ny-j, the pattern is preserved.

## 3. Verification Results

### 3.1 Operator-Level Mirror Covariance (synthetic symmetric field, D/6, Re=5)

| Coefficient | Before Fix | After Fix | Status |
|---|---|---|---|
| v_aE | 0.543 | **0.0** | ✅ PASS |
| v_aW | 0.541 | **0.0** | ✅ PASS |
| v_aN (vs v_aS mirror) | 0.0 | **0.0** | ✅ PASS |
| v_aP | 0.143 | **0.0** | ✅ PASS |
| u_aE, u_aW, u_aN, u_aP | 0.0 | **0.0** | ✅ PASS |

**Note**: Source terms `u_b` (0.579) and `v_b` (0.850) still show asymmetry due to the van Leer limiter boundary treatment (separate issue, not addressed in this fix).

### 3.2 Actual Flow Solution (sudden expansion, Re=5, D/10)

| Quantity | Before Fix | After Fix | Status |
|---|---|---|---|
| u asymmetry (max) | 5.5% | **9.4e-16** | ✅ Machine precision |
| v antisymmetry (max) | 5.5% | **2.0e-16** | ✅ Machine precision |

The solution is now symmetric to machine precision at Re=5, where the steady solution is unique and must be symmetric.

### 3.3 Test Suite Impact

- `tests/test_expansion_symmetry.py`: 7 passed, 1 xfailed
- The xfailed test (`test_v_momentum_operator_is_mirror_covariant`) checks both coefficients and source terms. Coefficients now pass; source terms still fail (van Leer limiter issue).

## 4. Error Assessment

| Criterion | Target | Measured | <3%? |
|---|---|---|---|
| v-momentum coefficients mirror covariance | <1e-12 | **0.0** | ✅ YES |
| u asymmetry (Re=5 sudden expansion) | <1e-10 | **9.4e-16** | ✅ YES |
| v antisymmetry (Re=5 sudden expansion) | <1e-10 | **2.0e-16** | ✅ YES |
| Source terms mirror covariance | <1e-12 | 0.58 / 0.85 | ❌ NO (separate issue) |

## 5. Limitations & Known Issues

1. **Source term asymmetry**: The van Leer limiter (`_vanleer_face`) uses `_shift` which zeros boundaries asymmetrically. This affects `u_b` and `v_b` but not the main coefficients. Fixing this requires careful redesign of the limiter boundary treatment and is left for future work.

2. **Baseline re-run required**: This fix changes the discrete operator, so all incompressible baselines (`docs/incompressible-baselines/baseline-runs/*.json`) need to be re-generated. This was not done in this task due to scope.

3. **Backward-facing step & natural convection**: These cases also use `_u_on_v` and will be affected by the fix. Their baselines also need re-generation.

## 6. Recommendations for Next Steps

1. **Re-run all incompressible baselines** with the fixed operator and update `docs/incompressible-baselines/`.
2. **Fix van Leer limiter** boundary treatment to achieve full mirror covariance (including source terms).
3. **Update test** `test_v_momentum_operator_is_mirror_covariant` to either:
   - Check only coefficients (remove `v_b` and `u_b` from assertions), OR
   - Keep as xfail until van Leer limiter is also fixed
4. **Document** the fix in the verification catalog and update capability台账.

## 7. Reproduction

```bash
# Run expansion symmetry tests
python3 -m pytest tests/test_expansion_symmetry.py -v

# Verify operator covariance manually
python3 -c "
import sys; sys.path.insert(0, 'src')
import numpy as np
from tensorfvm.incompressible_baselines import build_sudden_expansion

solver = build_sudden_expansion(6, re_D=5.0, inlet_window=(1.5, 2.5))
ny = solver.ny
rng = np.random.default_rng(0)
u = 0.5 * (rng.standard_normal((solver.nx+1, ny)) + rng.standard_normal((solver.nx+1, ny))[:, ::-1])
v = 0.5 * (rng.standard_normal((solver.nx, ny+1)) - rng.standard_normal((solver.nx, ny+1))[:, ::-1])
p = 0.5 * (rng.standard_normal((solver.nx, ny)) + rng.standard_normal((solver.nx, ny))[:, ::-1])
solver.u, solver.v, solver.p = u, v, p

vE, vW, vN, vS, vP, bv = solver._v_coefficients()
def rel(a, b):
    scale = max(float(np.abs(a).max()), float(np.abs(b).max()), 1e-30)
    return float(np.abs(a - b).max()) / scale

print('v_aE:', rel(vE, vE[:, ::-1]))
print('v_aW:', rel(vW, vW[:, ::-1]))
print('v_aP:', rel(vP, vP[:, ::-1]))
"
```

## 8. Conclusion

The primary defect (`_u_on_v` stencil error) has been successfully fixed. The v-momentum coefficients are now mirror-covariant, and the actual flow solution is symmetric to machine precision. The source term asymmetry (van Leer limiter) is a separate issue left for future work.

**Impact**: This fix improves the accuracy and reliability of all incompressible flow simulations using `StaggeredSimple` (sudden expansion, backward-facing step, natural convection cavity).
