# Verification: hybrid tangential gradient-jump stabilisation

**Unit**: `tensorfvm.native_gradient_jump3d.NativeGradientJump3D`
**Audit script**: `scripts/verify_native_gradient_jump_algebra.py`
**Test suite**: `tests/test_native_gradient_jump_verification.py` (18 tests)
**Evidence**: `evidence.json` (57 checks, 57/57 pass), `gradient-jump-verification.png`

## 1. What is under test

Hybrid face-velocity discretisations constrain the *average* velocity on a face,
not its tangential variation.  The resulting nonconforming space needs a
stabilising term that measures the jump of the reconstructed gradient across
internal faces; for Crouzeix-Raviart elements on simplices this is the strain
(Korn) term of Hansbo & Larson (2003, doi:10.1051/m2an:2003020, eq. 26).  The
module extends that construction to polyhedral faces: per internal face it
assembles

```
E = sum_f  w_f * jump_f(i,j) * Cov_f(j,k) * jump_f(i,k)
w_f        = strength * harmonic(mu_owner, mu_neighbor) * |A_f| / h_f
Cov_f      = area-weighted centroid inertia tensor of the triangulated face
```

and adds `kron(scalar, I3)` to `NativeHybridViscous3D.full_matrix`.  Its energy
is booked as `gradient_jump_energy` in the traction ledger of
`NativeHybridStokes3D`, and its per-cell stress replay feeds
`interface_flux_defect_max`.  Neither had a reference verification of its own,
which is the gap this unit closes.

## 2. Method

All checks are exact algebra or hand-derived geometry on 2-216 cell meshes; no
solver iteration is involved, so the runtime is ~1 min and the tolerances can be
set at machine precision instead of at a numerical scheme's error level.

Mesh families (`h` = cell size, all closed with Dirichlet boundaries so the
operator is exercised in the reduced velocity space):

| family | cells | faces | internal faces | note |
|---|---|---|---|---|
| `hex_cube_n2` | 8 | 72 | 24 | uniform, h = 1/2, analytic reference case |
| `hex_sheared_n2` | 8 | 72 | 24 | affine shear 0.18, non-orthogonal faces |
| `hex_sheared_n3` | 27 | 216 | 108 | refinement + shear 0.10 |
| `tetra_pair` | 2 | 9 | 1 | simplex family, minimal internal face |

## 3. Results

### 3.1 Structural identities (all four families)

| check | quantity | measured | reference | rel. error | tol | verdict |
|---|---|---|---|---|---|---|
| `hex_cube_n2/symmetry` | hex_cube_n2: matrix symmetry | 0.000000e+00 | 0 (exact) | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/psd` | hex_cube_n2: lambda_min(M) | -8.021151e-17 | >= 0 (exact PSD) | 8.02e-17 | 1e-14 | PASS |
| `hex_cube_n2/affine_energy` | hex_cube_n2: energy of affine velocity | 2.165601e-32 | 0 (exact) | 2.17e-32 | 1e-24 | PASS |
| `hex_cube_n2/kernel` | hex_cube_n2: dim ker(M) | 186.000000 | >= 12 (affine space) | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/energy_identity` | hex_cube_n2: ledger energy vs u^T M u | 2.999383 | u^T M u | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/ledger_closure` | hex_cube_n2: stress replay vs M@u | 3.246500e-16 | 0 (exact) | 3.25e-16 | 1e-12 | PASS |
| `hex_cube_n2/energy_gradient_fd` | hex_cube_n2: dE/du = 2*M@u (FD) | 2.021828e-09 | 2*M@u | 2.02e-09 | 1e-06 | PASS |
| `hex_cube_n2/strength_0.5` | hex_cube_n2: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/strength_2.5` | hex_cube_n2: energy ratio at strength=2.5 | 2.500000 | 2.5 | 8.88e-16 | 1e-12 | PASS |
| `hex_cube_n2/mu_2` | hex_cube_n2: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/strength_zero` | hex_cube_n2: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/symmetry` | hex_sheared_n2: matrix symmetry | 0.000000e+00 | 0 (exact) | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/psd` | hex_sheared_n2: lambda_min(M) | -2.065556e-16 | >= 0 (exact PSD) | 2.07e-16 | 1e-14 | PASS |
| `hex_sheared_n2/affine_energy` | hex_sheared_n2: energy of affine velocity | 5.105207e-31 | 0 (exact) | 5.11e-31 | 1e-24 | PASS |
| `hex_sheared_n2/kernel` | hex_sheared_n2: dim ker(M) | 186.000000 | >= 12 (affine space) | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/energy_identity` | hex_sheared_n2: ledger energy vs u^T M u | 4.776024 | u^T M u | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/ledger_closure` | hex_sheared_n2: stress replay vs M@u | 2.356968e-16 | 0 (exact) | 2.36e-16 | 1e-12 | PASS |
| `hex_sheared_n2/energy_gradient_fd` | hex_sheared_n2: dE/du = 2*M@u (FD) | 2.372660e-09 | 2*M@u | 2.37e-09 | 1e-06 | PASS |
| `hex_sheared_n2/strength_0.5` | hex_sheared_n2: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/strength_2.5` | hex_sheared_n2: energy ratio at strength=2.5 | 2.500000 | 2.5 | 4.44e-16 | 1e-12 | PASS |
| `hex_sheared_n2/mu_2` | hex_sheared_n2: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/strength_zero` | hex_sheared_n2: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/symmetry` | hex_sheared_n3: matrix symmetry | 0.000000e+00 | 0 (exact) | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/psd` | hex_sheared_n3: lambda_min(M) | -1.406556e-16 | >= 0 (exact PSD) | 1.41e-16 | 1e-14 | PASS |
| `hex_sheared_n3/affine_energy` | hex_sheared_n3: energy of affine velocity | 1.335593e-30 | 0 (exact) | 1.34e-30 | 1e-24 | PASS |
| `hex_sheared_n3/kernel` | hex_sheared_n3: dim ker(M) | 513.000000 | >= 12 (affine space) | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/energy_identity` | hex_sheared_n3: ledger energy vs u^T M u | 11.426695 | u^T M u | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/ledger_closure` | hex_sheared_n3: stress replay vs M@u | 4.841903e-16 | 0 (exact) | 4.84e-16 | 1e-12 | PASS |
| `hex_sheared_n3/energy_gradient_fd` | hex_sheared_n3: dE/du = 2*M@u (FD) | 4.936369e-09 | 2*M@u | 4.94e-09 | 1e-06 | PASS |
| `hex_sheared_n3/strength_0.5` | hex_sheared_n3: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/strength_2.5` | hex_sheared_n3: energy ratio at strength=2.5 | 2.500000 | 2.5 | 1.78e-15 | 1e-12 | PASS |
| `hex_sheared_n3/mu_2` | hex_sheared_n3: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/strength_zero` | hex_sheared_n3: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/symmetry` | tetra_pair: matrix symmetry | 0.000000e+00 | 0 (exact) | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/psd` | tetra_pair: lambda_min(M) | -3.514729e-16 | >= 0 (exact PSD) | 3.51e-16 | 1e-14 | PASS |
| `tetra_pair/affine_energy` | tetra_pair: energy of affine velocity | 1.921650e-32 | 0 (exact) | 1.92e-32 | 1e-24 | PASS |
| `tetra_pair/kernel` | tetra_pair: dim ker(M) | 21.000000 | >= 12 (affine space) | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/energy_identity` | tetra_pair: ledger energy vs u^T M u | 9.103991 | u^T M u | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/ledger_closure` | tetra_pair: stress replay vs M@u | 1.412938e-16 | 0 (exact) | 1.41e-16 | 1e-12 | PASS |
| `tetra_pair/energy_gradient_fd` | tetra_pair: dE/du = 2*M@u (FD) | 1.578306e-10 | 2*M@u | 1.58e-10 | 1e-06 | PASS |
| `tetra_pair/strength_0.5` | tetra_pair: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/strength_2.5` | tetra_pair: energy ratio at strength=2.5 | 2.500000 | 2.5 | 4.44e-16 | 1e-12 | PASS |
| `tetra_pair/mu_2` | tetra_pair: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/strength_zero` | tetra_pair: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/weight` | weight = mu*area/height | 0.250000 | 0.25 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/cov_major` | covariance major in-plane eigenvalue | 0.020833 | 0.020833333333333332 | 1.67e-16 | 1e-12 | PASS |
| `hex_cube_n2/cov_minor` | covariance minor in-plane eigenvalue | 0.006944 | 0.006944444444444444 | 1.25e-16 | 1e-12 | PASS |
| `hex_cube_n2/cov_normal` | covariance normal eigenvalue | 0.000000e+00 | 0.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/cov_trace` | covariance trace | 0.027778 | 0.027777777777777776 | 1.25e-16 | 1e-12 | PASS |
| `hex_cube_n2/lambda_min_gain` | hex_cube_n2: lambda_min increase | 0.172687 | > 0 (monotone stiffening) | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/lambda_min_gain` | hex_sheared_n2: lambda_min increase | 0.174664 | > 0 (monotone stiffening) | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/lambda_min_gain` | hex_sheared_n3: lambda_min increase | 0.252429 | > 0 (monotone stiffening) | 0.00e+00 | 1e-12 | PASS |


Reading the table: the matrix is bitwise symmetric, positive semidefinite to
roundoff (`lambda_min ~ -1e-16`), and its energy is exactly zero for arbitrary
affine velocity fields (`~1e-31`, i.e. floating-point dust) - the classical
"vanishes on the approximation space" requirement.  The stress replay closes on
`M @ u` to `1e-16`, so the interface-flux audit sees no defect that the
assembled matrix does not contain.

### 3.2 Absolute scaling against hand-derived geometry

| check | quantity | measured | reference | rel. error | tol | verdict |
|---|---|---|---|---|---|---|
| `hex_cube_n2/weight` | weight = mu*area/height | 0.250000 | 0.25 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/cov_major` | covariance major in-plane eigenvalue | 0.020833 | 0.020833333333333332 | 1.67e-16 | 1e-12 | PASS |
| `hex_cube_n2/cov_minor` | covariance minor in-plane eigenvalue | 0.006944 | 0.006944444444444444 | 1.25e-16 | 1e-12 | PASS |
| `hex_cube_n2/cov_normal` | covariance normal eigenvalue | 0.000000e+00 | 0.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/cov_trace` | covariance trace | 0.027778 | 0.027777777777777776 | 1.25e-16 | 1e-12 | PASS |


For the uniform `h = 1/2` cube the internal faces are triangles of a quad split
(area `h^2/2`), the centroid distance is `h`, hence `w = mu*h/2 = 0.25` exactly;
a right triangle of leg `h` has centroid inertia eigenvalues `h^2/12`,
`h^2/36`, `0` (trace `h^2/9`).  Measured values reproduce these to `1e-16`, so
there is **no hidden constant** in the operator's absolute level.

### 3.3 Strength and viscosity scaling

| check | quantity | measured | reference | rel. error | tol | verdict |
|---|---|---|---|---|---|---|
| `hex_cube_n2/strength_0.5` | hex_cube_n2: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/strength_2.5` | hex_cube_n2: energy ratio at strength=2.5 | 2.500000 | 2.5 | 8.88e-16 | 1e-12 | PASS |
| `hex_cube_n2/mu_2` | hex_cube_n2: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/strength_zero` | hex_cube_n2: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/strength_0.5` | hex_sheared_n2: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/strength_2.5` | hex_sheared_n2: energy ratio at strength=2.5 | 2.500000 | 2.5 | 4.44e-16 | 1e-12 | PASS |
| `hex_sheared_n2/mu_2` | hex_sheared_n2: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n2/strength_zero` | hex_sheared_n2: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/strength_0.5` | hex_sheared_n3: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/strength_2.5` | hex_sheared_n3: energy ratio at strength=2.5 | 2.500000 | 2.5 | 1.78e-15 | 1e-12 | PASS |
| `hex_sheared_n3/mu_2` | hex_sheared_n3: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_sheared_n3/strength_zero` | hex_sheared_n3: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/strength_0.5` | tetra_pair: energy ratio at strength=0.5 | 0.500000 | 0.5 | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/strength_2.5` | tetra_pair: energy ratio at strength=2.5 | 2.500000 | 2.5 | 4.44e-16 | 1e-12 | PASS |
| `tetra_pair/mu_2` | tetra_pair: energy ratio at mu=2 | 2.000000 | 2.0 | 0.00e+00 | 1e-12 | PASS |
| `tetra_pair/strength_zero` | tetra_pair: term absent at strength=0 | 1.000000 | 1.0 | 0.00e+00 | 1e-12 | PASS |


The quadratic form is exactly linear in `strength` and in a uniform `mu` (the
harmonic mean of equal viscosities is that viscosity); `strength = 0` assembles
no term at all, and negative / NaN / inf strengths are rejected.

### 3.4 Refinement consistency

| n | h | energy (u = x^2 + yz) |
|---|---|---|
| 2 | 0.5000 | 4.166667e-02 |
| 3 | 0.3333 | 2.469136e-02 |
| 4 | 0.2500 | 1.562500e-02 |
| 5 | 0.2000 | 1.066667e-02 |

Local orders 1.290, 1.591, 1.711; the
sequence approaches 2 (mean of the two finest pairs
1.651).
The defect energy therefore vanishes under refinement with no `h`-independent
floor - the term is a consistent defect measure, not a hidden bulk viscosity.

### 3.5 Effect on the reduced velocity operator

| family | lambda_min (strength 0) | lambda_min (strength 1) | increase |
|---|---|---|---|
| `hex_cube_n2` | 2.897903e-01 | 3.398334e-01 | +17.3% |
| `hex_sheared_n2` | 2.937608e-01 | 3.450703e-01 | +17.5% |
| `hex_sheared_n3` | 1.149630e-01 | 1.439830e-01 | +25.2% |

## 4. Findings and limits

1. All 57 audit checks pass at machine precision; the test suite pins 18 of them.
2. **Coercivity is not reproduced as a failure mode.** On every family tested the
   bare hybrid operator (`strength = 0`) is already positive definite
   (`lambda_min = 0.115 ... 0.294`), so this audit can show that the jump term
   *stiffens* the operator (+17% ... +25% in `lambda_min`) while leaving affine
   modes untouched, but it cannot demonstrate a mesh family on which the term is
   *necessary*.  The module docstring makes the same disclaimer ("no generic
   coercivity theorem is claimed"); a counterexample family remains open work.
3. **Kernel is large by construction.** `dim ker(M) = 186/240` on `hex_cube_n2`:
   only internal faces contribute, and for hexahedra the affine modes already
   exhaust the reconstructions that agree on both sides of a face.  The dimension
   is reported, not asserted tight; the guarantee is only `ker >= 12`.
4. **Energy convention.** The codebase books energies as the full quadratic form
   `u^T M u` (no 1/2), consistently for the local, jump and Robin terms, and pairs
   it with `kinetic_energy_rate = -u^T (A u)`, so `rate + energy = 0` for a
   homogeneous Dirichlet problem.  Under that convention the booked reaction
   `M @ u` is half of `dE/du = 2 M u`; the audit pins both numbers explicitly
   (`ledger_closure`, `energy_gradient_fd`) so that a future change of either
   convention fails a test instead of silently cancelling.
5. This unit verifies algebraic and geometric consistency only.  It does not
   compare against a flow solution, a benchmark drag/lift value, or a published
   discrete energy; those are covered by the RANS/Stokes units elsewhere in
   `docs/`.
