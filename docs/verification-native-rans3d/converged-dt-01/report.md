# Native 3D NS/SST integration pilot

**This is a nonzero inlet/no-slip integration test, not physical turbulence validation or a SUBOFF benchmark.**

The actual unit-box native mesh has 27 hexahedral cells with planar native triangle facets. Inlet speed is 1, density 1, molecular viscosity .05 (Re=20). Two y walls have prescribed zero velocity, x=1 has physical pressure zero, and z boundaries have prescribed uniform streamwise velocity. Initial/inflow k=.001 and omega=1. These are test inputs, not measured turbulence. This low-Re setup is not a turbulent reference benchmark.

The solver assembles complete momentum/continuity with variable molecular/eddy stress and SST-2003 transport. Eddy viscosity is exactly zero at low-Re walls. The nearest actual wall-triangle distance is verified against the independent box-plane distance: maximum error 0.

## Actual result

- Cell streamwise velocity range: [0.9030382629621455, 1.194138077143862].
- Wall native normal flux maximum: 0.
- Native face divergence maximum: 1.7502e-10.
- Current reconstructed divergence maximum after updating SST viscosity: 1.31318e-09.
- Current nonlinear SST k residual: 8.72833e-08.
- Current nonlinear SST omega residual: 1.15974e-11.
- Total wall force: [0.5648014014986433, -0.0009681807169169129, 9.643320758148086e-05] in normalized rho U^2 L^2 units.
- Physical pressure/isotropic/anisotropic force decomposition error: 0.
- Numerical convergence: **True**. Physical accuracy qualification: **False**.

The run performed 57 outer updates with pseudo dt=0.1, turbulence relaxation=0.7, outer gate=1e-07 and inner momentum gate=1e-8. Numerical convergence is reported separately from physical accuracy. The current sources are recomputed from the current fields; the small frozen transport ledger is separately reported and is not used as a nonlinear convergence claim. There is no known drag/Cp target and no below-3% qualification.

## Pressure and boundary forces

The solved pressure is pi=p+2rho k/3. Physical pressure is p=pi−2rho k/3. Physical outlet pressure is converted with current boundary k, and all pressure source vectors are refreshed. Total force is invariant: pi S−tau_anis S = p S−(tau_anis−2rho k I/3)S. The same shared face quantities supply both momentum and reported boundary forces.

## Reproduction and evidence

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_native_rans_integration3d.py --n 3 --outer 100 --pseudo-dt 0.1 --turbulence-relaxation 0.7 --tolerance 1e-07 --output NEW_EMPTY_DIRECTORY
```

[result.json](result.json) contains actual fields' metrics and complete outer histories, including current nonlinear and frozen linearized ledgers. Raw fields and native geometry are in fields.npz; imported producer sources are archived at launch. This small integration test does not establish engineering mesh cost, turbulence accuracy, wall y+, spatial convergence or GPU performance.
