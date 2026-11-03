# Native 3D viscous operator: discrete passivity audit

Status: **open defect, quantified and bounded**. `tests/test_native_energy_small.py::test_two_cell_skew_viscosity_must_dissipate` stays a strict `xfail`; the quantified
verdict lives in `tests/test_native_energy_passivity.py`.

## 1. Object and criterion

The implicit viscous operator consumed by `native_collocated3d` and `native_rans3d` is
`L = -momentum_diffusion_matrix`. A cell-centred discretisation of
`-div(mu (grad u + grad u^T))` with homogeneous Dirichlet data must satisfy the discrete
dissipation (passivity) inequality

    x^T L x <= 0   for every velocity vector x,   i.e. -L positive semidefinite.

Otherwise the implicit step can inject kinetic energy. There is no pressure, no
turbulence and no flow benchmark in the test: it is a pure operator property, so it also
covers the Rhie-Chow mobility reference built from the same scalar Laplacian.

Counterexample (`two_tetrahedra(1.)`): two valid tetrahedra sharing one face, cell
volumes 1/24 and 1/3, 7 faces, largest nonorthogonality 86.651 deg.

## 2. Measured result on the counterexample

| operator | scalar min eig | scalar max eig | vector min eig | vector max eig | creation rate | creation fraction | passive |
|---|---|---|---|---|---|---|---|
| deferred over-relaxed (default) | -5.1417 | 7.6369 | -7.1478 | 9.6604 | 7.1478 | 0.425 | **no** |
| deferred minimum correction | -0.3889 | 5.6030 | -0.7952 | 7.2815 | 0.7952 | 0.098 | **no** |
| two-point reference | - | - | 1.4328 | 10.8083 | 0.0000 | 0.000 | yes |

`creation rate = max(0, -min eig(-L))`, `creation fraction = creation / (creation + dissipation)`.
The assembled traction and the matrix quadratic form agree to 1.8e-15, and the prescribed
velocity (1,0,0) in cell 0 creates 4.8328 units of kinetic energy per unit time
(the energy identity is not the issue; the sign of the operator is).

## 3. Is it a weight bug or structural?

The deferred correction is one-parameter: `gamma*s*(du - d.grad)` with `gamma = 1/(s.d)` (over-relaxed, default) or `(s.d)/(|d|^2 |s|^2)` (minimum/orthogonal correction).
Scanning 12 weights over [0.0e+00, 32.0] gives a smallest creation eigenvalue of 5.3243 at
the smallest weight (`gamma -> 0`, i.e. the correction switched off): **no member of the
family is passive**. The cause is the interpolated-cell-gradient composite, not the
correction weight: `div(interp(grad_LS u))` is not the adjoint of the cell-to-face jump
operator, so the symmetric part is indefinite.

## 4. Validity range in mesh-quality terms

| correction | largest passive skew | largest passive nonorthogonality | first creating nonorthogonality |
|---|---|---|---|
| over-relaxed | 0.9587 | 85.916 deg | 85.916 deg |
| minimum | 0.8697 | 83.908 deg | 83.909 deg |

The repository mesh-quality gate (`tensorfvm.mesh_quality.quality_gate_passed`) admits
nonorthogonality < 70 deg, so the shipped default operator is passive on **every mesh the
quality gate admits, with 15.9 deg of margin**. Independently, 144 randomly sheared tetrahedral
pairs (`random_shear_survey`) gave 0 over-relaxed failures and 4 minimum-correction failures.

Deterministic sweep (vector operator, creation rate):

| skew | nonorthogonality [deg] | over-relaxed | minimum |
|---|---|---|---|
| 0.000 | 25.24 | 0.000e+00 | 0.000e+00 |
| 0.100 | 30.53 | 0.000e+00 | 0.000e+00 |
| 0.200 | 35.22 | 0.000e+00 | 0.000e+00 |
| 0.300 | 39.30 | 0.000e+00 | 0.000e+00 |
| 0.400 | 47.56 | 0.000e+00 | 0.000e+00 |
| 0.500 | 60.99 | 0.000e+00 | 0.000e+00 |
| 0.600 | 70.97 | 0.000e+00 | 0.000e+00 |
| 0.700 | 77.50 | 0.000e+00 | 0.000e+00 |
| 0.800 | 81.79 | 0.000e+00 | 0.000e+00 |
| 0.900 | 84.67 | 0.000e+00 | 2.321e-01 |
| 1.000 | 86.65 | 7.148e+00 | 7.952e-01 |
| 1.100 | 88.05 | 3.689e+01 | 1.087e+00 |
| 1.200 | 89.06 | 1.078e+02 | 1.198e+00 |
| 1.300 | 89.80 | 5.825e+02 | 1.204e+00 |

## 5. Consistency/passivity trade-off

| scheme | affine-traction relative error | creation rate at skew=1 | passive |
|---|---|---|---|
| deferred over-relaxed | 7.0e-15 | 7.1478 | no |
| deferred minimum | 9.4e-16 | 0.7952 | no |
| deferred none | 1.7e-15 | 1.4116 | no |
| two-point (`mu (s.d)/|d|^2`) | 1.125 | 0.0000 | yes |

Every deferred variant reproduces the exact affine traction to round-off, so re-weighting
`gamma` costs nothing in consistency - but it never buys passivity either. The strictly
passive two-point operator pays 112% flux error on the skewed face. This is the classical
monotonicity/accuracy barrier, so the honest verdict is a *bounded open defect*, not a bug
to be patched by tuning.

## 6. Recommendation

* Keep the default over-relaxed correction (widest passive range).
* The minimum correction is gentler past the boundary (168 x smaller creation at skew=1.25) but
  loses 2.0 deg of passive range; it is not a drop-in safety win.
* Reject meshes above the 70 deg gate as already done; do **not** relax it past
  86 deg without re-running this audit.
* A fully passive *and* affine-exact operator needs a symmetric scheme family
  (SUSHI/DDfV-style), which is a separate implementation item, not a weight change.

## 7. Reproduction

```
python scripts/audit_native_viscous_passivity.py   # writes audit.json + 4 figures
python -m pytest tests/test_native_energy_passivity.py tests/test_native_energy_small.py -q
```

Figures: `fig1_passivity_spectrum.png` (signed spectra), `fig2_creation_vs_angle.png`
(creation versus nonorthogonality with the gate and the bisected boundary),
`fig3_family_scan.png` (no gamma is passive), `fig4_tradeoff.png` (consistency versus
passivity).
