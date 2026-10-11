# Native SUBOFF MMS reconstruction audit

This audits the frozen converged coarse field, not a new NS solve. Mesh: `/home/jsyc/tensor-suite-development/TensorFVM/docs/verification-native-suboff-mms/coarse/input.msh`; fields: `/home/jsyc/tensor-suite-development/TensorFVM/docs/verification-native-suboff-mms/coarse/fields.npz`. Source and input SHA256 bindings are in `audit.json`.

## Production field errors

Velocity volume-weighted relative L2: **0.197898%**. Pressure after volume-mean gauge removal: **5.499916%**. These are centroid point-value errors, not cell-average norms. MMS is mathematical verification; prescribed hull velocity is not physical no-slip SUBOFF.

## Independent continuum consistency

At each cell, e is an integrated momentum residual. The reported relative norm is sqrt(sum(|e_c|²/V_c))/sqrt(sum(|F_c|²/V_c)), where F is independent volume-quadrature analytic acceleration integrated over the cell. Contributions are not an error budget of the coupled pressure solution.

| Contribution | Relative volume-weighted acceleration L2 |
|---|---:|
| point_vs_volume_forcing | 0.224595% |
| pressure_face_reconstruction | 2.357561% |
| pressure_quadrature_vs_volume_gradient | 0.000229% |
| viscous_discretization | 11.528113% |
| convection_discretization | 4.911882% |
| total_exact_state_residual | 12.566742% |

Faces use order-5 Duffy Gauss integration. Each convex native cell is decomposed into centroid-to-triangle tetrahedra, with four-point degree-2 volume quadrature. Trigonometric integrals are approximate; pressure surface-versus-volume-gradient agreement is reported independently. Native tetra fan volume closure is 2.88e-15. No production residual defines the analytic force.

The continuum ABC definition is u=U(sin(kz)+cos(ky), sin(kx)+cos(kz), sin(ky)+cos(kx)), p=rho U² sin(kx)cos(ky)sin(kz). It obeys div u=0 and laplacian u=-k²u. Hence f=(u·grad)u+grad(p)/rho+nu k²u. This is a self-defined analytic verification, with no experimental reference.

## Read-only quadratic reconstruction probe

The probe fits the nine scaled polynomial terms x,y,z,x²,y²,z²,xy,xz,yz through actual mesh adjacency. Geometry/rank determine 2–4-ring stencils; analytic pressure samples do not select a stencil or tune parameters. A shared face value averages owner/neighbor predictions with native interpolation weights. Triangle quadratic means include covariance sum(vertex-centroid outer products)/12, so the same face force is added and subtracted for its two cells.

| Quantity | Production reconstruction | Quadratic consistency probe |
|---|---:|---:|
| Pressure gradient relative L2 | 6.046557% | 1.104129% |
| Pressure force consistency | 2.357561% | 0.189106% |
| Boundary face pressure absolute RMS | 0.0010062337 | 7.556187e-05 |
| Internal face pressure absolute RMS | 0.00033120999 | 2.4751354e-05 |

The maximum normalized design condition is 70.2624; the independently checked degree-2 coefficient reproduction defect is 6.46e-14. **The quadratic probe is not in the production solver and computes no new NS pressure solution. It does not establish pressure below 3%.**

## Interpretation and next checks

The cell-type and geometric-region energy distributions are in the JSON. Most domain volume lies in external tetrahedra; refining only the hull layer cannot establish global MMS convergence. Pressure reconstruction has a measurable consistency improvement under polynomial reconstruction. Production integration would require compatible skew/Hessian terms in Rhie–Chow compact and interpolated gradient fluxes, retained shared-face conservation, and physical velocity-wall free-pressure treatment. Do not remove RC or impose exact manufactured pressure as a production wall repair.

Independent source-volume integration removes a smaller measured residual, but full coupled ablations are needed to determine its pressure-solution impact. Viscous stress and advective reconstruction also require convergence checks. Keep fixed analytic parameters, solve on genuinely refined external meshes, and report native continuity and momentum residuals. This audit neither establishes grid convergence nor experimental drag/Cp acceptance.
