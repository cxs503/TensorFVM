# Native SUBOFF SST-2003 diagnostic

Cells: 9,068; Re_L: 1.2e+07; model: NASA TMR SST-2003 (exact production, isotropic Reynolds stress retained).
Numerically converged: **False**. Experimental validation passed: **false**.
Computed maximum wall y+: **0**.
This wall-resolved model requires a suitable first-cell distance; a set mesh
height alone is not a wall-resolution qualification. No wall function is used.

Physical pressure is converted from pi=p+2rho k/3, and the reported force
includes both anisotropic and isotropic Reynolds stress consistently.
The input, numerical sources, configuration, current nonlinear residuals,
raw fields, actual wall y+ and force components are frozen in this directory.
Experimental metrics and a mesh-refinement study remain outstanding.

Failure: `{'type': 'RuntimeError', 'message': 'nonpositive momentum diagonal; current mesh/operator is not admissible'}`. CPU wall time: 13.248s.
