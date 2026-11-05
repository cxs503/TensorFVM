# TensorFVM Verification Coverage Audit

**Date**: 2026-11-03  
**Scope**: Comprehensive audit of analytic/manufactured verification coverage  
**Status**: Snapshot of current verification state

## Executive Summary

TensorFVM has extensive verification infrastructure covering major solver components. This audit catalogs the verification status, identifies open defects, and maps coverage gaps.

## Verification Matrix

### 1. Viscous Operator (Stokes)

| Test | Status | Order | Notes |
|------|--------|-------|-------|
| `test_native_hybrid_stokes3d.py` | ✅ PASS (15 tests) | — | Basic Stokes functionality |
| `test_native_viscous_order3d.py` | ⚠️ 1 xfail | 1st-2nd | Wall-adjacent divergence O(1) defect |
| `test_native_viscous_manufactured_solve.py` | ✅ PASS | 1.8+ | Manufactured solution with restored source |
| `test_native_energy_passivity.py` | ✅ PASS (9 tests) | — | Passivity within 70° quality gate |
| `test_native_energy_small.py` | ⚠️ 1 xfail | — | Passivity fails beyond 85.9° nonorthogonality |

**Open Defects**:
- Wall-adjacent cell divergence is O(1) instead of second order (documented, mechanism identified as wall-flux deficit)
- Energy passivity breaks beyond 85.9° nonorthogonality (outside 70° quality gate)

### 2. Convection Operator

| Test | Status | Order | Notes |
|------|--------|-------|-------|
| `test_native_hybrid_convection_order.py` | ✅ PASS (19 tests) | 1st-2nd | Upwind + deferred linear |
| `test_native_hybrid_flow.py` | ✅ PASS | — | Coupled flow verification |

**Coverage**: First-order upwind and second-order deferred linear upwind both verified.

### 3. Navier-Stokes (Coupled)

| Test | Status | Notes |
|------|--------|-------|
| `test_native_hybrid_ns3d.py` | ✅ PASS | Nonlinear NS with Picard iteration |
| `test_native_kovasznay.py` | ✅ PASS (9 tests) | Kovasznay exact solution |
| `verification-native-abc3d-spatial` | ✅ PASS (n=12) | ABC manufactured, pressure 2.91% at finest |

**ABC Verification Results** (from `docs/verification-native-abc3d-spatial/report.md`):
- n=3 (27 cells): velocity 0.32%, pressure 24.7% ❌
- n=6 (216 cells): velocity 0.09%, pressure 9.1% ❌  
- n=12 (1728 cells): velocity 0.02%, pressure 2.91% ✅
- Pressure convergence order: ~1.5 (sub-optimal, not asymptotically second-order)

### 4. Scalar Transport

| Test | Status | Notes |
|------|--------|-------|
| `test_native_hybrid_scalar3d.py` | ✅ PASS (6 tests) | Basic scalar transport |
| `scripts/verify_native_hybrid_scalar.py` | ✅ Exists | Analytic scalar verification script |

**Coverage**: Diffusion + advection with analytic boundary fluxes.

### 5. Lid-Driven Cavity (Ghia Benchmark)

| Test | Status | Notes |
|------|--------|-------|
| `test_lid_cavity.py` | ✅ PASS (9 tests) | Smoke test |
| `test_lid_cavity_re1000_evidence.py` | ⚠️ 1 xfail | Re=400 v-centreline discrepancy |
| `test_ghia_reference_consistency.py` | ✅ PASS (8 tests) | Ghia table consistency |

**Open Issue**: Re=400 v-centreline at x=0.9063 shows growing error under refinement (0.115/0.142/0.149 for n=32/64/128). Attribution analysis exonerates extraction path; defect is in stored solution field next to wall.

### 6. Taylor-Green Vortex

| Test | Status | Notes |
|------|--------|-------|
| `test_taylor_green_mac.py` | ✅ PASS | TGV with MAC projection |
| `test_periodic_mac.py` | ✅ PASS | Periodic MAC verification |
| `test_periodic_tgv.py` | ✅ PASS | Periodic TGV |

**Coverage**: Decaying TGV with analytic initial conditions, energy dissipation verified.

### 7. Two-Phase Flow

| Test | Status | Notes |
|------|--------|-------|
| `test_two_phase_solver_verification.py` | ✅ PASS | 5 verification anchors |
| `test_two_phase_projection.py` | ✅ PASS | Projection verification |
| `test_two_phase_surface_tension_verification.py` | ⚠️ 1 xfail | Young-Laplace pressure jump |

**Coverage**: VOF transport, projection, surface tension (partial).

### 8. Turbulence (SST)

| Test | Status | Notes |
|------|--------|-------|
| `test_native_hybrid_sst3d.py` | ✅ PASS | SST model integration |
| `test_native_sst_channel_reference.py` | ✅ PASS | Channel flow reference |
| `test_native_sst_wall3d.py` | ✅ PASS | Wall treatment |

**Coverage**: SST k-omega with wall functions, channel flow verification.

## Verification Gaps

### High Priority

1. **Pressure convergence order**: ABC verification shows pressure converging at ~1.5 order instead of expected 2nd order. Mechanism not fully understood.

2. **Wall-adjacent viscous truncation**: O(1) error at wall-adjacent cells. Mechanism identified (wall-flux deficit) but not yet fixed.

3. **Re=400 lid cavity v-centreline**: Solution field defect near right wall. Requires solver investigation.

### Medium Priority

4. **Surface tension verification**: Young-Laplace pressure jump test remains xfail. Static droplet equilibrium not fully verified.

5. **Extended scalar precision**: `longdouble` precision path for scalar transport exists but limited verification.

### Low Priority

6. **High-Reynolds lid cavity**: Re=1000 verification exists but strict acceptance gate remains xfail due to Re=400 issue.

7. **Overset verification**: Multiple overset tests exist but full verification suite not cataloged here.

## Test Suite Health

```bash
# Quick health check (lightweight tests only)
pytest tests/test_native_hybrid_stokes3d.py \
       tests/test_native_hybrid_convection_order.py \
       tests/test_native_kovasznay.py \
       tests/test_lid_cavity.py \
       tests/test_taylor_green_mac.py \
       -q --timeout=60
```

**Expected**: ~60 tests pass, 3 xfail (documented defects).

## Recommendations

1. **Immediate**: Document pressure convergence mechanism in ABC verification. Is the 1.5 order expected for this discretization, or is there a fixable defect?

2. **Short-term**: Investigate wall-flux deficit mechanism. Can the one-sided wall flux be improved without breaking energy stability?

3. **Medium-term**: Resolve Re=400 v-centreline defect. Is this a grid-induced artifact or a fundamental limiter of the collocated arrangement?

4. **Long-term**: Extend verification to:
   - Compressible flow (if supported)
   - Rotating frames
   - Free surface with surface tension
   - Fluid-structure interaction

## Reproduction

All verification results are reproducible from committed evidence bundles:

```bash
# ABC spatial verification
python scripts/verify_native_hybrid_ns.py --output outputs/abc-spatial

# Lid cavity ladder
python scripts/verify_lid_cavity_ladder.py --output outputs/lid-ladder

# Scalar transport
python scripts/verify_native_hybrid_scalar.py --output outputs/scalar
```

## Conclusion

TensorFVM has strong verification coverage for incompressible Navier-Stokes with SST turbulence. Three open defects are documented and tracked. The verification infrastructure is mature, with analytic/manufactured solutions, convergence studies, and literature benchmarks. Priority should be given to understanding the pressure convergence order and wall-flux mechanisms.

---

**Audit performed by**: agent/m3732  
**Verification date**: 2026-11-03  
**Next review**: After wall-flux defect resolution
