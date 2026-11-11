# Rotational added inertia of an ellipsoid: closed form vs the panel solver

TensorFVM · generated programmatically from `summary.json` (no number here is hand-copied).

## 1. Case

Closed triaxial ellipsoids in an unbounded ideal fluid, `rho = 1.0`. Two independent
descriptions of the same physical quantity are compared:

* **closed form** (`tensorfvm.ellipsoid_added_inertia`): the degree-2 Lamé potential of the
  rotation problem, `phi = C_2 x y FF(lambda)`, with the amplitude fixed by the Neumann data
  and the Kirchhoff energy integrated in closed form;
* **panel solver** (`tensorfvm.panel_added_mass.rotational_added_inertia`): constant-strength
  source panels on a latitude/longitude mesh (faceting error included, nothing idealised).

Geometries: spheroids `a/b = 2, 4, 8` (panel meshes `24x14`, `48x26`, `64x34`) plus a
triaxial ellipsoid `1.6 / 1.2 / 0.9` (used for the
geometry-independent checks, which the panel meshes of this module cannot represent).

## 2. Method

For rotation with angular velocity `Omega` about a principal axis the perturbation potential
solves

    laplace(phi) = 0,     dphi/dn = (Omega x r) . n =: psi on S,     phi -> 0 at infinity,

and the added inertia is `A = -rho integral_S phi psi dS` (Green's identity orientation, so
that `A` is positive definite). The classical solution is `phi = C_2 x y FF(lambda)` with

    FF(lam) = int_lam^inf dm / ((A^2+m)(B^2+m) sqrt((A^2+m)(B^2+m)(C^2+m))),

which reproduces the `x y / |grad F|` structure of `psi` because
`x^2/A^4 + y^2/B^4 + z^2/C^4 = |grad F|^2/4`. Matching amplitudes gives
`C_2 = 2(1/B^2 - 1/A^2)/coef` with `coef = 2(1/A^2 + 1/B^2) FF(0) - 4/(A^2 B^2 A B C)`, and

    A_axis = -4 rho (1/B^2 - 1/A^2)^2 FF(0) (2 pi/15 A^3 B^3 C) / coef.

The only numerical ingredient is the one-dimensional integral `FF(0)` (quadrature at
`1e-13`), cross-checked against an independent closed form for prolate spheroids
(rel. diff `2.5e-15`), so the reference itself is not a discretisation.

The checks of §3 deliberately avoid validating the derivation with its own algebra:
`dphi/dn` is obtained by Richardson-extrapolated finite differences of the closed-form
potential, the Laplacian numerically, and the surface moment by brute-force Gauss×trapezoid
quadrature on the exact (unfaceted) surface.

## 3. Results

| criterion | statement | value | threshold | verdict |
|---|---|---|---|---|
| C1 | closed-form potential reproduces the rotation data: max|dphi/dn - psi|/max|psi| < 1e-6 | 1.7606306301874916e-10 | 1e-06 | PASS |
| C2 | I = (2 pi/15) a^3 b^3 c matches surface quadrature (triaxial): rel < 1e-10 | 7.48941558428992e-15 | 1e-10 | PASS |
| C3 | harmonicity: |laplace(phi)|/|phi| < 1e-6 (Richardson-extrapolated FD) | 4.919720293574282e-08 | 1e-06 | PASS |
| C4 | quadrupole decay phi(2r)/phi(r) = 1/8 within 1% | 0.0008035606804976669 | 0.01 | PASS |
| C5 | sphere and spheroid-roll added inertia are exactly zero | 0.0 | 0.0 | PASS |
| C6 | slender asymptote: monotone and within 2% of 1/5 at a/b = 100 | 0.0004953603523895622 | 0.02 | PASS |
| C7 | near-sphere: I_55 ~ (a^2-b^2)^2, spread over b/a = 0.99/0.999/0.9999 < 2% | 0.01230962399297316 | 0.02 | PASS |
| C8 | panel solver within 3% of the closed form at the finest mesh (a/b = 2, 4, 8) | 0.02385989314901129 | 0.03 | PASS |
| C9 | two-finest-mesh spread of the panel I_55 < 10% (mesh uncertainty, all ratios) | 0.010402695388162549 | 0.1 | PASS |

**C1 — does the closed form solve the stated Neumann problem?** (numerical `dphi/dn`)

| geometry | points | max abs data | max rel residual |
|---|---|---|---|
| spheroid_a_over_b_2 | 288 | 0.936281 | 1.76e-10 |
| triaxial | 288 | 0.389486 | 4.70e-11 |

The residual is at the level of the finite-difference truncation, including the triaxial
shape: the potential is the exact solution of the problem the closed form claims to solve,
so the physics it encodes is not a fit to the panels.

**C6 — slender asymptote** (`A_55/(rho V a^2)` against the strip-theory value `1/5`)

| a/b | I_55 | I_55/(rho V a^2) |
|---|---|---|
| 4 | 34.6327 | 0.129187 |
| 8 | 365.669 | 0.170502 |
| 20 | 6451.02 | 0.192508 |
| 50 | 103858 | 0.198354 |
| 100 | 835683 | 0.199505 |

**C7 — near-sphere vanishing** `A_55 ~ (a^2 - b^2)^2`

| b/a | I_55 | I_55 / (a^2-b^2)^2 |
|---|---|---|
| 0.6 | 0.0588702 | 0.143726 |
| 0.8 | 0.0272987 | 0.210638 |
| 0.99 | 0.000109218 | 0.275796 |
| 0.999 | 1.11451e-06 | 0.278907 |
| 0.9999 | 1.11676e-08 | 0.279218 |

The law is asymptotic (the ratio still drifts by 1.23% over the
three shapes closest to the sphere), which is why the criterion is scored on
`b/a = 0.99 / 0.999 / 0.9999` rather than on the whole family.

**C8 — panel solver against the closed form**

| a/b | mesh | panels | I_55 (panel) | rel. err vs closed form | |I_44-I_66|/I_66 |
|---|---|---|---|---|
| 2 | 24x14 | 336 | 1.841757 | 8.178% | 6.0e-16 |
| 2 | 48x26 | 1248 | 1.937567 | 3.401% | 2.3e-16 |
| 2 | 64x34 | 2176 | 1.957935 | 2.386% | 6.8e-16 |
| 4 | 24x14 | 336 | 31.967493 | 7.696% | 7.8e-16 |
| 4 | 48x26 | 1248 | 33.661395 | 2.805% | 1.3e-15 |
| 4 | 64x34 | 2176 | 33.987097 | 1.864% | 4.2e-16 |
| 8 | 24x14 | 336 | 355.997199 | 2.645% | 1.3e-15 |
| 8 | 48x26 | 1248 | 366.086158 | 0.114% | 7.8e-16 |
| 8 | 64x34 | 2176 | 367.188995 | 0.416% | 6.2e-16 |

Worst finest-mesh error `2.39%`. Three-point extrapolation of the
*panel* sequence alone (order fitted from the three meshes, closed form not used as input)
gives exponent `[2.121219204082406, 2.4160844499329186, 4.021354900328889]` and the relative error
`[0.002773169611445277, 0.0019905669836707777, 0.006880673363163785]`, i.e. the panel sequence is consistent with a
power-law convergence onto the closed-form value. This is reported as a *diagnostic*: the
latitude/longitude refinement is anisotropic, and at `a/b = 8` the error is already at the
0.1-0.4 % level where it stops being monotone, so no exponent can be fitted (None).

## 4. Figures

`fig_rotational_reference.png` — left: `A_55/(rho V a^2)` versus elongation `a/b` for the
closed form (line), the panel meshes (markers) and the strip-theory asymptote `1/5`
(dotted); right: panel relative error versus panel count on a log-log axis, one series per
elongation.

## 5. Notes and limits

* Unbounded fluid only: no free surface, no twin-body images (ledger entry D2 stays open).
* The panel meshes cannot represent a triaxial ellipsoid, so the triaxial comparison is
  structural only (C1/C2/C3 use the exact surface, not a mesh).
* The extrapolated numbers of §3 are a three-point fit on an anisotropic
  (latitude/longitude) refinement family, not a physical convergence rate -- which is why the
  acceptance is scored on the *finest mesh* (C8) and on the two-finest-mesh spread (C9), with
  the extrapolation kept as a diagnostic.
* Ledger entry D7 (wiring the rotational block into `maneuvering.dynamics.J_z`) is *not*
  part of this verification and remains open.

## 6. Error judgement

9/9 criteria PASS. Quantitative judgements against the closed
form: panel error 2.39% (a/b=2),
1.86% (a/b=4),
0.42% (a/b=8) at the finest meshes -- **all < 3 %**,
which closes the "analytic < 3 %" half of debt D1 that
`verification-rotational-added-mass` had to declare WITHHELD. The near-sphere spread
1.23% and the slender-limit error
0.05% are within their frozen thresholds.
