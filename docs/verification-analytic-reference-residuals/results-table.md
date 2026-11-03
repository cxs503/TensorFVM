| reference | claim | symbolic residual | numeric max rel. deviation | points |
|---|---|---|---|---|
| native_order3d.manufactured/stress/divergence_of_stress | grad u, deviatoric stress mu(grad u + grad u^T - 2/3 tr(grad u) I) and its divergence | 0 (exact) | 2.070e-15 | 4000 |
| annular.analytic_velocity / analytic_flow | annular Hagen-Poiseuille profile solves mu (1/r)(r u')' = dp/dz with dp/dz = -G and u(ri)=u(ro)=0 | 0 (exact) | 8.356e-15 | 3999 |
| extended_cases.exact_velocity/exact_pressure ('abc') | u = (sin z + cos y, sin x + cos z, sin y + cos x) e^{-nu t} with p = -|u|^2/2 is an exact 3-D Navier-Stokes solution | 0 (exact) | 0.000e+00 | 2000 |
| extended_cases.exact_velocity ('advected-shear') | u = (sin(y-0.7t) e^{-nu t}, 0.7, 0), p = 0 solves the incompressible Navier-Stokes equations | 0 (exact) | 0.000e+00 | 0 |
| overset_poisson.manufactured_value/manufactured_source, overset_scalar.exact_value | -Laplace(phi) equals the coded right-hand side; the overset scalar quadratic is the coded value | 0 (exact) | 2.761e-16 | 2000 |
| curved_channel.reference (mapped-coordinate manufactured NS solution) | u=(w, g' w), p=G(L-x) with w=6V eta(1-eta) and the coded body force is an exact solution of the incompressible 2-D Navier-Stokes equations in the sin^4-mapped channel | 0 (exact) | 3.318e-15 | 1500 |
| extended_cases torch implementations vs symbolic solution | the torch reference fields evaluate the audited closed forms (not a different formula) | 0 (exact) | 1.032e-18 | 1500 |

| h | max abs. derivative deviation |
|---|---|
| 0.1000 | 7.121e-02 |
| 0.0487 | 1.695e-02 |
| 0.0237 | 4.023e-03 |
| 0.0115 | 9.543e-04 |
| 0.0056 | 2.263e-04 |
| 0.0027 | 5.366e-05 |
| 0.0013 | 1.273e-05 |
| 0.0006 | 3.018e-06 |
| 0.0003 | 7.156e-07 |

observed slopes: 1.99, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00, 2.00
