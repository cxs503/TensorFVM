# Native SUBOFF SST-2003 diagnostic

Cells: 9,068; Re_L: 1.2e+07; model: NASA TMR SST-2003 (exact production, isotropic Reynolds stress retained).
Numerically converged: **False**. Experimental validation passed: **false**.
Computed maximum wall y+: **193.841**.
Algorithm: **simple**. Wall treatment: **spalding**.
Computed y+ and mesh/wall-treatment sensitivity are required for qualification.

Physical pressure is converted from pi=p+2rho k/3, and the reported force
includes both anisotropic and isotropic Reynolds stress consistently.
The input, numerical sources, configuration, current nonlinear residuals,
raw fields, actual wall y+ and force components are frozen in this directory.
Experimental metrics and a mesh-refinement study remain outstanding.

Failure: `{'type': 'RuntimeError', 'message': 'SIMPLE linear solve failed true residual gate: momentum=2.25112e-16, pressure=5.46047e-08; accepted state preserved', 'rejected_linear_solve': {'momentum_relative_residual': 2.2511225324511483e-16, 'pressure_relative_residual': 5.460469464808807e-08, 'accepted_state_changed': False}}`. CPU wall time: 23.423s.
