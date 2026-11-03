# Verification table (generated from evidence.json)

## Structural identities

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

## Geometry

| check | quantity | measured | reference | rel. error | tol | verdict |
|---|---|---|---|---|---|---|
| `hex_cube_n2/weight` | weight = mu*area/height | 0.250000 | 0.25 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/cov_major` | covariance major in-plane eigenvalue | 0.020833 | 0.020833333333333332 | 1.67e-16 | 1e-12 | PASS |
| `hex_cube_n2/cov_minor` | covariance minor in-plane eigenvalue | 0.006944 | 0.006944444444444444 | 1.25e-16 | 1e-12 | PASS |
| `hex_cube_n2/cov_normal` | covariance normal eigenvalue | 0.000000e+00 | 0.0 | 0.00e+00 | 1e-12 | PASS |
| `hex_cube_n2/cov_trace` | covariance trace | 0.027778 | 0.027777777777777776 | 1.25e-16 | 1e-12 | PASS |

## Scaling

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
