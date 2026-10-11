# Native SUBOFF SST-2003 diagnostic

Cells: 9,068; Re_L: 1.2e+07; model: NASA TMR SST-2003 (exact production, isotropic Reynolds stress retained).
Numerically converged: **False**. Experimental validation passed: **false**.
Computed maximum wall y+: **393.961**.
Algorithm: **simple**. Wall treatment: **spalding**.
Computed y+ and mesh/wall-treatment sensitivity are required for qualification.

Physical pressure is converted from pi=p+2rho k/3, and the reported force
includes both anisotropic and isotropic Reynolds stress consistently.
The input, numerical sources, configuration, current nonlinear residuals,
raw fields, actual wall y+ and force components are frozen in this directory.
Experimental metrics and a mesh-refinement study remain outstanding.

Failure: `{'type': 'RuntimeError', 'message': 'Explicit divergence guard exceeded: max speed 28.0287 > 5; state retained without clipping'}`. CPU wall time: 24.713s.
