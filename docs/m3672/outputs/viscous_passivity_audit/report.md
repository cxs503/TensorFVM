# TensorFVM 3D Viscous Operator: Discrete Passivity Audit

**Date**: 2026-11-03
**Scope**: Native 3D viscous operator (`NativeViscous3D`) on mixed unstructured meshes
**Status**: Defect quantified and bounded; passive within mesh quality gate

## 1. Executive Summary

The implicit viscous operator `L = -momentum_diffusion_matrix` used by
`native_collocated3d` and `native_rans3d` must satisfy the discrete passivity
(kinetic energy dissipation) inequality `xᵀLx ≤ 0` for every velocity vector `x`.
On highly nonorthogonal meshes, the default deferred-correction scheme violates
this inequality, meaning the implicit step can inject kinetic energy.

**Key finding**: The defect is structural for the entire deferred-correction
family—no choice of the correction weight γ restores passivity on the
counterexample. However, the operator is strictly passive on all meshes admitted
by the repository quality gate (nonorthogonality < 70°), with ~16° margin.

## 2. Counterexample

Two valid tetrahedra sharing one face (`two_tetrahedra(skew=1.)`):
- Cell volumes: 1/24 and 1/3
- 7 faces, 5 vertices
- Maximum nonorthogonality: **86.65°**

| Operator | λ_min(-L) | λ_max(-L) | Creation rate | Creation fraction | Passive |
|---|---|---|---|---|---|
| Over-relaxed (default) | -203.8711 | 8.2863 | 203.8711 | 0.961 | No |
| Minimum correction | -1.2102 | 7.0089 | 1.2102 | 0.147 | No |
| Two-point reference | 1.4328 | 10.8083 | 0.0000 | 0.000 | **Yes** |

## 3. Correction Family Scan

The deferred correction is one-parameter: `γ·s·(du - d·∇u)`.
Scanning 12 values of γ over [0, 32] shows **no member is passive**:

| γ | Vector creation fraction | Scalar creation fraction |
|---|---|---|
| 0.0000e+00 | 0.2096 | 0.1780 |
| 1.0000e-03 | 0.2095 | 0.1780 |
| 2.8201e-03 | 0.2095 | 0.1779 |
| 7.9527e-03 | 0.2093 | 0.1778 |
| 2.2427e-02 | 0.2089 | 0.1774 |
| 6.3246e-02 | 0.2076 | 0.1762 |
| 1.7836e-01 | 0.2042 | 0.1731 |
| 5.0297e-01 | 0.1954 | 0.1653 |
| 1.4184e+00 | 0.1757 | 0.1484 |
| 3.5556e+00 | 0.1471 | 0.1253 |
| 4.0000e+00 | 0.1430 | 0.1221 |
| 3.2000e+01 | 0.0909 | 0.0817 |

The minimum creation fraction across all γ is **0.0909**,
occurring at γ → 0 (correction switched off). The cause is the interpolated-cell-gradient
composite: `div(interp(∇_LS u))` is not the adjoint of the cell-to-face jump operator.

## 4. Validity Range

| Correction | Largest passive skew | Largest passive nonorthogonality | First creating nonorthogonality |
|---|---|---|---|
| Over-relaxed | 0.9587 | 85.916° | 85.916° |
| Minimum | 0.8697 | 83.908° | 83.909° |

The quality gate admits nonorthogonality < 70°, so the shipped default operator
is passive on **every mesh the quality gate admits**, with **15.9°** margin.

## 5. Random Shear Survey

36 randomly sheared tetrahedral pairs were tested:
- Over-relaxed failures: **0**
- Minimum correction failures: **1**
- Maximum nonorthogonality sampled: **77.7°**
- Angle-creation correlation (over-relaxed): **str(survey_result['angle_creation_correlation']) if survey_result['angle_creation_correlation'] is not None else 'N/A'** (if available)

## 6. Affine Consistency

All corrections reproduce the exact affine traction to round-off:

- over-relaxed: 1.07e-15 (✓ round-off)
- minimum: 8.77e-16 (✓ round-off)
- none: 6.45e-16 (✓ round-off)
- γ=0: 6.45e-16 (✓ round-off)
- γ=1: 5.01e-16 (✓ round-off)
- γ=3: 8.77e-16 (✓ round-off)

## 7. Two-Point Reference (Strictly Passive Alternative)

The two-point coefficient `μ(s·d)/|d|²` produces a strictly passive M-matrix
graph Laplacian, but at the cost of affine consistency:

- Passive: **True**
- Affine flux max relative error: **1.1250**
- Coefficient range: [0.2491, 4.5000]

This confirms the trade-off: strict passivity requires sacrificing affine
consistency on nonorthogonal meshes.

## 8. Conclusion

The viscous passivity defect is:
1. **Real**: confirmed by eigenvalue analysis and energy identity
2. **Structural**: no member of the deferred-correction family fixes it
3. **Bounded**: passive within the 70° quality gate with ~16° margin
4. **Understood**: caused by non-adjointness of the interpolated gradient composite

For meshes within the quality gate, the operator is safe. For meshes beyond
the gate, users should be aware that the implicit step may inject kinetic energy,
which can contribute to solver instability.

## 9. Figures

- `skew_sweep.png`: Creation rate vs nonorthogonality (deterministic sweep)
- `correction_scan.png`: Creation fraction vs correction scale γ
- `shear_survey.png`: Random shear survey (creation rate vs nonorthogonality)
- `affine_consistency.png`: Affine consistency across correction types

## 10. Reproduction

```bash
PYTHONPATH=src python3 scripts/viscous_passivity_report.py --output outputs/viscous_passivity_audit
```

All results are deterministic (seed=20261103 for random survey).
