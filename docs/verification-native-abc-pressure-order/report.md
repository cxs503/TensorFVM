# ABC manufactured solution: attribution of the ~1.5 pressure order

The verification coverage audit (`docs/verification-coverage-audit.md`) lists the sub-second-order pressure convergence of the fixed-parameter 3D ABC MMS as its first open item, mechanism unknown. This note resolves that question from archived producer output and states where the remaining truncation actually lives.

## Case (physics, geometry, parameters)

Steady incompressible Navier-Stokes manufactured solution on the logical cube [0,1]^3 mapped by the shear `coordinate_transform` = [[1.0, 0.18, 0.12], [0.0, 1.0, 0.2], [0.0, 0.0, 1.0]]: amplitude U = 0.5, wavenumber k = 0.7, kinematic viscosity nu = 0.1, density rho = 1.0. Velocity is u = U(sin kz + cos ky, sin kx + cos kz, sin ky + cos kx) and pressure is p = rho U^2 sin kx cos ky sin kz. Every physical face carries analytic Dirichlet velocity and the exact triangle-integrated mass flux (order-8 Duffy quadrature); the body force is the exact differential acceleration evaluated at the actual cell centroids. Grids n = 3, 4, 6, 8, 12, 1728 cells at the finest level; hexahedra expose real planar triangular shared facets, so no Cartesian stencil is substituted.

## How it was done

Producer (unchanged, rerun here): `scripts/verify_native_abc_convergence3d.py`. Audit: `scripts/audit_native_abc_pressure_order3d.py`, which (1) recomputes the published pressure error from the stored fields, (2) groups cells by topological distance to the nearest face, (3) measures the truncation of the directional difference actually available at each cell against the analytically differentiated exact gradient, and (4) fits power laws on the coarser usable grid pairs and predicts the finest grid as a hold-out. Volume fractions and depths are exact geometry; the only fitted quantities are the two power-law exponents named in the model table.

Grids 3, 6, 12 and 4, 8 come from two invocations of the same producer with identical settings and are merged for the order fit; provenance is recorded in `outputs/abc-pressure-order/merged/config.json`.

## Results

| n | cells | pressure % published | pressure % recomputed | shell-0 volume frac | shell-0 local % | shell-0 share of squared error | shell-0 stencil trunc. | interior stencil trunc. |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 27 | 24.7230 | 24.7230 | 0.9630 | 25.1627 | 99.8% | 5.435e-01 | 2.029e-02 |
| 4 | 64 | 16.4797 | 16.4797 | 0.8750 | 17.5091 | 98.8% | 3.612e-01 | 1.207e-02 |
| 6 | 216 | 9.0798 | 9.0798 | 0.7037 | 10.4893 | 93.9% | 2.146e-01 | 5.677e-03 |
| 8 | 512 | 5.7613 | 5.7613 | 0.5781 | 7.1130 | 88.1% | 1.521e-01 | 3.294e-03 |
| 12 | 1728 | 2.9124 | 2.9124 | 0.4213 | 3.9806 | 78.7% | 9.614e-02 | 1.513e-03 |

Recomputation reproduces every published pressure error to better than 1e-9 relative: **True**.

Shell-resolved local pressure error (gauge-removed, volume-weighted RMS per shell):

| n | shell 0 | shell 1 | shell 2 | shell 3 | shell 4 | shell 5 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 25.1627% | 6.3967% |
| 4 | 17.5091% | 5.1645% |
| 6 | 10.4893% | 4.3510% | 1.7185% |
| 8 | 7.1130% | 3.4381% | 1.9657% | 0.8133% |
| 12 | 3.9806% | 2.1592% | 1.5055% | 1.0102% | 0.5961% | 0.2507% |

Depth of each shell centre at the finest grid: shell 0 = 0.042, shell 1 = 0.125, shell 2 = 0.208, shell 3 = 0.292, shell 4 = 0.375, shell 5 = 0.458.

Measured orders (log-log, per refinement step):

| refinement | global | shell-0 local | interior local | shell-0 stencil | interior stencil |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 -> 4 | 1.410 | 1.261 | n/a | 1.420 | n/a |
| 4 -> 6 | 1.470 | 1.264 | 0.560 | 1.284 | 1.860 |
| 6 -> 8 | 1.581 | 1.350 | 1.033 | 1.197 | 1.892 |
| 8 -> 12 | 1.682 | 1.432 | 1.352 | 1.132 | 1.919 |

`n/a` marks statistics withheld rather than zero: the grid has fewer than 8 interior cells, so its interior RMS is not stable (n = 3 has a single interior cell).

Composite hold-out model: the shell-0 and interior exponents are fitted on the two coarser usable pairs and the finest grid is predicted, not fitted.

| quantity | value |
| --- | ---: |
| fitted shell-0 order | 1.350 |
| fitted interior order | 1.033 |
| predicted shell-0 local RMS | 4.1143e-02 |
| predicted interior local RMS | 2.0106e-02 |
| predicted global error | 3.0775e-02 |
| observed global error | 2.9124e-02 |
| relative prediction error | 5.67% |
| predicted order (mid -> fine) | 1.546 |
| observed order (mid -> fine) | 1.682 |

## Error judgement against the benchmark definition

The benchmark quantity is the volume-weighted relative L2 pressure error with the gauge removed independently from the computed and exact fields. The audit reproduces the published value exactly, so the numbers above and the archived report are the same measurement.

| level | pressure % | below the 3% gate? |
| ---: | ---: | --- |
| 3 | 24.7230 | no |
| 4 | 16.4797 | no |
| 6 | 9.0798 | no |
| 8 | 5.7613 | no |
| 12 | 2.9124 | yes |

Explicit answer: the finest grid (n = {}) is at {:.2f}%, i.e. **below 3%**; every coarser level is above 3% and is reported as failing, unchanged from the archived evidence.

Auxiliary benchmark, not a physical validation: no SUBOFF drag, Cp, wall y+, turbulence or force accuracy claim is made here.

## Verdict

the global pressure order is inherited from boundary-adjacent cells: shell 0 carries 79% of the squared error at n=12 while the measured stencil available there is first order; the composite power-law model reproduces the finest grid error to 5.7%, so no additional interior defect is required. The residual is a boundary-closure truncation, i.e. fixable by a higher-order near-boundary gradient, not by interior bookkeeping.

* boundary-adjacent shell 0 carries the majority of the squared error: **True** (shell 0 plus shell 1: 94.2%)
* composite hold-out model within 10%: **True**
* additional interior defect required: **False**

## Notes, limits, open items

Shell-0 local order rises {:.2f} -> {:.2f} across the sequence and the global order {:.2f} -> {:.2f}; neither is asymptotic. The measured interior stencil is second order while the near-boundary stencil converges to first order, which is the mechanism that keeps the global order below two.
* The truncation column measures the stencil a cell centre can form, not the solver internal operator; it bounds, and does not reproduce, the discrete near-boundary gradient.
* A further nested level n = 16 did not finish inside the local single-job budget and the compute cluster checkout has no NumPy/PyTorch, so the sequence stops at n = 12. An n = 16/24 extension is the direct follow-up.
* Fixing the boundary closure to second order is a code change in the near-boundary pressure/gradient reconstruction and was NOT attempted here; this note only establishes that the fix must live there and that the far field is consistent with second-order truncation.
* Trap found while doing this: the first version of the truncation diagnostic mapped logical axis 0 to the x direction although the structured array stores k on axis 0, which produced rising instead of falling truncation. The convention is now asserted by a regression test.

## Reproduction

```bash
PYTHONPATH=src python scripts/verify_native_abc_convergence3d.py --grids 3 4 6 8 12 --output NEW_DIR
PYTHONPATH=src python scripts/audit_native_abc_pressure_order3d.py --run-dir NEW_DIR --output OUT \
    --fixture tests/data/abc-pressure-order/pressure-order-fixture.npz \
    --figure docs/verification-native-abc-pressure-order/abc-pressure-order.png \
    --report docs/verification-native-abc-pressure-order/report.md
```
