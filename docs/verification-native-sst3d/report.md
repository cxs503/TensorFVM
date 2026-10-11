# Native 3D SST-2003 closure and transport development

**Runnable model and transport components; not an integrated or validated RANS benchmark.** No SUBOFF drag/Cp accuracy, grid convergence, wall-function validation or turbulence physical qualification is claimed.

## Model definition and official provenance

The implemented variant is **SST-2003 with exact production**, as defined by the [NASA Turbulence Modeling Resource](https://tmbwg.github.io/turbmodels/sst.html#sst-2003). The old NASA URL now redirects to a migration announcement. The official page and 42 standard/SST-2003 equation images are archived here, with retrieval metadata and SHA256 in [source-provenance.json](source-provenance.json).

The closure uses the full three-dimensional strain invariant for the eddy-viscosity limiter. Both production equations use the factor-10 limiter; gamma is blended from 5/9 and 0.44, and the cross-diffusion floor is 1e-10. The full Reynolds stress includes the isotropic k term. Momentum integration must retain that term or explicitly define a modified pressure. It must not silently call an implementation omitting that term this exact variant. No constants were fitted to benchmark data.

## Software

`src/tensorfvm/native_sst3d.py` provides:

- `sst2003_closure(k, omega, wall_distance, nu, velocity_gradient, grad_k, grad_omega, rho=...)`: scalar blending, tensor invariants, kinematic/dynamic turbulent viscosity, exact/limited production, destruction and cross-diffusion sources, dynamic Reynolds stress.
- `wall_omega(nu, first_cell_distance)`: low-Re wall condition, not a wall function. k at the wall is prescribed zero by the caller.
- `inflow_from_intensity(speed, intensity, length_scale)`: an explicitly assumed isotropic inlet parameterization. It does not establish experimental turbulence values or implement sustaining-source variants.
- `NativeSSTTransport3D(mesh, scalar_dirichlet_mask).step(...)`: both conservative scalar transport updates using native shared faces, full nonorthogonal `NativeViscous3D` diffusion and first-order upwind convection.

`velocity_gradient[..., i, j]` means du_i/dx_j. Mesh lengths, nearest-wall distances and molecular kinematic viscosity must use consistent units. The transport face flux is u dot S (volume flux). Density must be constant. Local closure sources and transport equations are specific/per mass; returned Reynolds stress is dynamic and includes density.

For transport, callers supply physical k/omega boundary values, normal derivatives, an externally obtained velocity-gradient tensor and nearest-wall distances. Boundary inflow uses explicit scalar values even where the diffusive boundary is Neumann. A wall distance based only on nearest wall-face centroid is not the required nearest-surface distance.

Each update freezes gradients, viscosity and source coefficients from the old state. Negative source contributions are represented as implicit sinks; they are not discarded. Invalid updated states are rejected without silent positivity clipping. Returned ledgers balance actual boundary advective/diffusive flux, volume storage and the **frozen linearized source**. A small ledger is not a nonlinear SST convergence criterion.

## Verification performed

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src \
  /data/TensorFEM/.venv/bin/python -m pytest -q tests/test_native_sst3d.py
```

Eight tests passed:

1. Independent high-shear values verify eddy viscosity and the common production limiter in both equations.
2. Pure strain and solid rotation distinguish the SST-2003 strain limiter from older vorticity-based viscosity.
3. Exact production equals the full dynamic Reynolds stress contracted with velocity gradient and divided by density. Negative expansion production is retained.
4. Rotation of all tensor/vector inputs preserves scalar closure values and transforms stress correctly.
5. The k=0 analytic ratio avoids a zero-viscosity division; illegal negative/zero states are rejected.
6. Wall and assumed inlet parameterizations have the documented units.
7. A real two-tetrahedron native grid reproduces independent homogeneous implicit destruction decay and accounts for volume-integrated sources.
8. Shared internal convection, prescribed inflow and diffusion balance actual boundary/source/storage contributions.

The tests establish local formula consistency and discrete conservation on native 3D cells. They do not establish wall-resolved turbulent physical accuracy. See [verification.json](verification.json) for executable path, source hashes and test outcome.

## Outstanding qualification

- No momentum/pressure RANS coupling or nonlinear steady convergence test has been integrated.
- Nearest-surface distance, wall mesh y+, prism-layer adequacy and inlet turbulence provenance remain caller responsibilities.
- Full nonorthogonal diffusion need not be an M matrix; no unconditional positivity/stability guarantee is made.
- Eddy-viscosity diffusion from cell coefficients uses the existing geometric two-point interpolation; variable-coefficient interpolation refinement is unqualified.
- No official TMR physical test, original SUBOFF experimental drag/Cp comparison, below-3% error or GPU acceleration has passed.
