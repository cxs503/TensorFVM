# Native 3D mesh and operator verification

Input: `docs/verification-suboff-mesh/coarse/suboff.msh`; SHA256 `254dc5060def74185896d97a93bb7d0e9e417796ec2c43749bb46264d4b9c025`.

The actual volume contains 6810 cells and 14384 unique shared faces. Native owner/neighbor flux integration uses one value per face with opposite signs in adjacent cells. No extrusion or spanwise periodicity is assumed.

Minimum cell volume: 5.668e-08. Maximum relative cell area closure: 1.45875e-16. Minimum owner-neighbor projection: 4.251e-08. Maximum nonorthogonality: 62.5508 degrees.

## Independent manufactured fields

Pressure p=1.7+0.7x−1.3y+2.1z has known constant gradient. Constant velocity (0.3,0.8,−0.4) has zero divergence, and velocity (x,2y,3z) has divergence 6. Boundary values come directly from these analytic fields. Error is the maximum absolute component error over the actual mesh.

| Response | Maximum absolute error |
| --- | ---: |
| least_squares_gradient_max_error | 1.17573e-13 |
| face_affine_max_error | 1.77636e-15 |
| gauss_gradient_max_error | 5.70766e-13 |
| constant_vector_divergence_max_error | 1.19553e-13 |
| affine_vector_divergence_max_error | 3.11751e-13 |
| global_flux_balance_error | 2.09353e-17 |

## Qualification scope

Geometry and affine operator contracts pass. This is not a validated SUBOFF flow result. Mesh nonorthogonality, near-wall resolution, pressure-velocity coupling and physical benchmark accuracy require separate gates. Only linear convex tetrahedra, planar-faced hexahedra, wedges and pyramids are supported; unsupported and untagged topology is rejected.

See [audit.json](audit.json) for exact values, patch counts and source hashes. Source snapshots are archived alongside this report.
