# SUBOFF native-mesh Hodge initialization diagnostic

**This is an inviscid initialization/mesh-operator diagnostic, not a viscous SUBOFF benchmark. The scalar is velocity potential, not physical pressure.**

## Problem and software

Actual bare SUBOFF tetrahedra are read from `docs/verification-suboff-mesh/medium/suboff.msh` (SHA256 `6a89eb3fe08ec5eea9d4e9d05642535bd8c91e9d36b3aaac8481b01a16076802`). The native 3D mesh has 13,977 cells and 29,148 unique shared faces. No extrusion, cylinder topology or Cartesian mask is used. Starting from uniform U=(1,0,0), solve the mixed-boundary velocity-potential Poisson problem, then correct native face flux. At the wall dphi/dn=U dot n removes penetration; inlet/farfield derivative zero preserves their normal flux, and outlet phi=0 anchors the system. The whole skew/nonorthogonal face response is assembled with mixed LS boundary rows and CPU sparse LU. One scalar face traction supplies a conservative Gauss cell correction.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/initialize_suboff_native.py --mesh docs/verification-suboff-mesh/medium/suboff.msh --output NEW_EMPTY_DIRECTORY
```

## Actual mesh and fields

![True z=0 tetrahedral cuts and computed velocity-potential / speed fields](initialization.png)

These polygons are true tetrahedral intersections, not projected edges or interpolated raster cells. Raw native fields and topology are in fields.npz. No pressure or Cp/drag curve is inferred from velocity potential. No physical time has advanced; the velocity represents a slip potential initialization, not a no-slip viscous solution.

## Native projection checks

| Metric | Actual | Limit | Passed |
| --- | ---: | ---: | --- |
| native_projected_divergence | 1.86934e-12 | 1e-07 | True |
| wall_native_normal_speed | 7.22985e-17 | 1e-12 | True |
| fixed_outer_flux_change | 0 | 1e-12 | True |
| conservative_potential_impulse_ledger | 2.94903e-17 | 1e-10 | True |

Independent mixed-BC affine manufacture phi=1.2+0.7x−0.3y+0.2z on this exact mesh: scalar error 2.55351e-14, gradient error 1.23346e-13, final flux remainder 3.65159e-15. Analytic boundary data are supplied on all physical faces.

## Unresolved cell/face consistency

Native projected divergence is 1.86934e-12 U/L. Re-interpolating the corrected conservative cell velocity gives divergence 80.1014 U/L and maximum interior normal speed difference 0.134717 U. These are exposed separately: successful face projection does not certify a coupled incompressible Navier--Stokes solver. Numerical flow and physical qualification remain false.

## Performance and next stage

Setup 16.3604 s; one solve 0.0259 s. The actual full nonorthogonal matrix has 206,955 entries and L/U factors have 11,717,670 entries (140.724 MB sparse arrays). These are measured CPU reference costs, not GPU or engineering-scale performance.

Next: conservative low-Re viscous momentum on the native mesh with matching wall pressure response; independent cell/face and stress verification; boundary-layer/wake refinement; then matched experimental Reynolds and original reference data. The present tetrahedral meshes have no prism boundary layer. geometry/operator/projection checks do not imply drag accuracy or mesh independence.
