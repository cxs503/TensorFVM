# Rotational added mass M_44..M_66: verification against exact anchors

TensorFVM · generated programmatically from `summary.json` (no number here is hand-copied).

## 1. Case

Closed bodies in an unbounded ideal fluid, rho = 1.0. Added mass from the
constant-strength source-panel solver of `tensorfvm.panel_potential`; geometry is the
latitude/longitude panelisation of `tensorfvm.panel_added_mass` (faceting error included,
nothing is idealised).

| case | geometry | panels |
|---|---|---|
| sphere about a diameter | a = 1.0 | 576 |
| sphere, reference point offset | centre (0, 0, 1.5a) | 336 / 576 / 1248 |
| prolate spheroid, roll | a/b = 2.0 | 432 |
| prolate spheroid, pitch/yaw | a/b = 2.0 | 160 / 336 / 1248 / 2176 |
| near-sphere family | a/b = 1.25 / 1.5 / 2 | 432 each |

## 2. Method

Kirchhoff added-mass matrix about the coordinate origin (`added_mass_matrix_6dof`):

    A_ab = -rho * integral_S phi_a psi_b dS,      a, b = 0..5
    psi_a = n_a  (a < 3),   psi_{3+a} = (r x n)_a
    laplace(phi_a) = 0,     dphi_a/dn = -psi_a on S,     phi_a -> 0 at infinity

The six Neumann problems share one influence matrix (single LU). Translational block
`0..2`, rotational `3..5`, off-diagonal blocks = translation/rotation coupling.
Acceptance criteria were frozen in the module docstring before the run; the solver
change is `solve_sources(..., normal_velocity=...)` (uniform onset flow unchanged).

## 3. Results

| criterion | statement | value | threshold | verdict |
|---|---|---|---|---|
| C1 | sphere rotational block ~ 0: max|I_rot|/(m_a a^2) < 1e-2 | 1.1425192841032535e-05 | 0.01 | PASS |
| C2 | spheroid roll about the symmetry axis = 0: |I_44|/(rho V a^2) < 1e-10 | 5.135079859529735e-34 | 1e-10 | PASS |
| C3 | offset-axis sphere: I_xx, I_yy within 3% of m_a d^2, I_zz within 3% of 0, translational block within 3% | {'max_rel_err_rotation': 0.014597472134483544, 'zz_rel_err': 4.645702020267853e-33, 'translation_rel_err': np.float64(0.014598509643049854)} | 0.03 | PASS |
| C4 | spheroid transverse: |I_55-I_66|/I_55 < 1e-8 and the two-finest-mesh spread < 3% | {'max_axisym_rel': 6.804453105275055e-16, 'two_finest_spread': 0.010402695388162632, 'observed_order_diagnostic': 5.570381715107755} | {'axisym': 1e-08, 'spread': 0.03} | PASS |
| C5 | raw reciprocity max|A_ab-A_ba|/max|A| < 5e-2 (why the API symmetrises) | 1.6751691270641872e-16 | 0.05 | PASS |
| C6 | I_55/(rho b (a^2-b^2)^2) finite and of the same order for a/b=1.25,1.5,2 | [0.25706058840889845, 0.2353208997576359, 0.20784324506339646] | {'max_over_min': 3.0} | PASS |

**Sphere about a diameter** (exact value 0): panel diagonal
[2.392886793015634e-05, 2.3928867930156e-05, 1.6284910024083973e-32], residual scaled by `m_a a^2` = 1.14e-05
(polyhedral faceting). Translational block error 2.57%.

**Offset reference point** (exact `I_oo = m_a d^2 = 4.71239`):

| mesh | panels | I_xx | err I_xx | err I_yy | err I_zz | err translation |
|---|---|---|---|---|---|---|
| 24x14 | 336 | 4.52948 | 3.88% | 3.88% | 1.2e-33 | 3.88% |
| 32x18 | 576 | 4.59115 | 2.57% | 2.57% | 3.5e-33 | 2.57% |
| 48x26 | 1248 | 4.64360 | 1.46% | 1.46% | 4.6e-33 | 1.46% |

Observed order (offsets, two finest levels) p = 0.733; Richardson
`I_xx = 4.72282` -> relative error **0.221%**.
Coupling block max |A_tr| = 3.09573 vs exact `m_a d = 3.14159`.

**Spheroid roll about the symmetry axis** (exact 0): `I_44 = 1.69e-32`
against the scale `rho V a^2 = 32.8772`, i.e.
5.1e-34 -- the structural zero reproduces to machine precision,
which is only possible if the moment arm, the normal orientation and the sign of the
singular integral are all right.

**Spheroid pitch/yaw block** (no analytic value available):

| mesh | panels | V (panel) | I_55 | I_66 | |I55-I66|/I55 | I55/(rho V_exact a^2) | raw asymmetry |
|---|---|---|---|---|---|---|
| 16x10 | 160 | 7.96413 | 1.73231 | 1.73231 | 2.56e-16 | 0.05169 | 1.11e-16 |
| 24x14 | 336 | 8.17838 | 1.84176 | 1.84176 | 6.03e-16 | 0.05496 | 1.20e-16 |
| 48x26 | 1248 | 8.32322 | 1.93757 | 1.93757 | 2.29e-16 | 0.05782 | 1.18e-16 |
| 64x34 | 2176 | 8.34629 | 1.95793 | 1.95793 | 6.80e-16 | 0.05843 | 1.68e-16 |

Two-finest-mesh spread 1.04% (uncertainty of the converged value);
`I_55/(rho V_exact a^2)` at the finest mesh = 0.05843.
Order estimate p = 5.57 is **noise-dominated** at these panel
counts (latitude resolution dominates; see §5) and is therefore reported, not gated on.

**Near-sphere vanishing** (a moment arm or Neumann-data defect would break it):

| a/b | I_55 | I_55/(rho V a^2) | I_55/(rho b (a^2-b^2)^2) |
|---|---|---|---|
| 1.25 | 0.08134 | 0.01013 | 0.25706 |
| 1.50 | 0.36769 | 0.02651 | 0.23532 |
| 2.00 | 1.87059 | 0.05690 | 0.20784 |

**Slender limit** (strip theory `I = integral rho pi r(x)^2 x^2 dx` -> `(1/5) rho V a^2`):

| a/b | I_55 | I_55/(rho V a^2) |
|---|---|---|
| 2 | 1.8921 | 0.0565 |
| 4 | 32.8825 | 0.1227 |
| 8 | 362.0260 | 0.1688 |

## 4. Point of attention -- the ledger anchor was wrong

`docs/VERIFICATION_DEBT_LEDGER.md` D1 proposed `I_add(sphere) = (5/8) pi rho a^5`.
On a sphere `r x n = 0`, hence `(Omega x r) . n = 0`; with `phi -> 0` at infinity the
exterior Neumann problem has only `phi = 0`, so the fluid kinetic energy and the
pressure perturbation both vanish: the ideal-fluid added moment of inertia of a sphere
is **exactly zero** (Fossen: the sphere's added-mass matrix is `diag(m_a, m_a, m_a, 0, 0, 0)`).
The only non-zero rotational sphere anchor is the parallel-axis value `m_a d^2` used above.
Had the ledger anchor been used as the acceptance reference, a correct implementation
would have been scored 100 % wrong. The ledger entry has been corrected and closed.

## 5. Limits of this verification

* No elementary closed form for a spheroid transverse added inertia exists in this tree,
  so the pitch/yaw block is checked for *structure* (axisymmetry, reciprocity, positivity,
  correct near-sphere vanishing, slender-limit approach), not against an analytic value.
  Honest status: the "analytic < 3 %" half of the original D1 acceptance is **WITHHELD**;
  the exact-zero anchors and the parallel-axis anchor are quantitative and pass.
* Panel counts are capped at 2176 (load-average discipline of this shift), and the
  latitude/longitude mesh under-resolves the spheroid poles, which is why the order
  estimate is noisy. A uniform-area mesh is the natural follow-up.
* Unbounded fluid only (no free surface, no images): the free-surface / twin-body
  rotational problem stays open as ledger entry D2.

## 6. Error judgement

6/6 criteria PASS. The quantitative < 3 % tests against exact values
(sphere offset axis, Richardson-extrapolated 0.221%;
sphere about a diameter; spheroid roll) all pass; the un-anchored pitch/yaw value carries
a 1.04% mesh uncertainty and is declared WITHHELD for the
"analytic < 3 %" criterion rather than scored against an invented reference.
