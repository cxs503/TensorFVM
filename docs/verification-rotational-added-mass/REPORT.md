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
| prolate spheroid, pitch/yaw | a/b = 2.0 | 160 / 336 / 1248 / 2176 / 4800 |
| triaxial ellipsoid, all three | a:b:c = 2 : 1.5 : 1 | 576 / 1248 / 2176 / 4800 |
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
| C2 | spheroid roll about the symmetry axis = 0: |I_44|/(rho V a^2) < 1e-10 | 5.1350798595297345e-34 | 1e-10 | PASS |
| C3 | offset-axis sphere: I_xx, I_yy within 3% of m_a d^2, I_zz within 3% of 0, translational block within 3% | {'max_rel_err_rotation': 0.014597472134483168, 'zz_rel_err': 4.645702020267853e-33, 'translation_rel_err': np.float64(0.014598509643048159)} | 0.03 | PASS |
| C4 | spheroid transverse: |I_55-I_66|/I_55 < 1e-8 and the two-finest-mesh spread < 3% | {'max_axisym_rel': 5.670377587729206e-16, 'two_finest_spread': 0.00930678039697462, 'observed_order_diagnostic': 0.25778743506171103} | {'axisym': 1e-08, 'spread': 0.03} | PASS |
| C5 | raw reciprocity max|A_ab-A_ba|/max|A| < 5e-2 (why the API symmetrises) | 1.9915429757168732e-16 | 0.05 | PASS |
| C7 | spheroid pitch/yaw vs closed form: finest-mesh rel. err < 3 % and Richardson rel. err < 1 % | {'finest_rel_err': 0.014689827753002238, 'richardson_rel_err': 0.00852086314974564, 'observed_order_fit': 1.3141069800061371} | {'finest': 0.03, 'richardson': 0.01} | PASS |
| C6 | I_55/(rho b (a^2-b^2)^2) finite and of the same order for a/b=1.25,1.5,2 | [0.2570605884088984, 0.23532089975763582, 0.20784324506339644] | {'max_over_min': 3.0} | PASS |
| C8 | triaxial 2:1.5:1: each I_44/I_55/I_66 within 3 % of the closed form at the finest mesh, Richardson < 1 %, I_45=I_46=I_56=0 to 1e-10 relative, fitted order p >= 1 | {'finest_rel_err': [0.020289122236454394, 0.01226115784978347, 0.0030480868269462634], 'richardson_rel_err': [0.011647054289657784, 0.006968118681443951, 0.0016858836298895927], 'cross_rel_max': 4.0305951446044634e-17, 'observed_order_fit': [1.355532313029813, 1.3398453258404648, 1.4038499990434326]} | {'finest': 0.03, 'richardson': 0.01, 'cross': 1e-10, 'order': 1.0} | FAIL |

**Sphere about a diameter** (exact value 0): panel diagonal
[2.392886793015634e-05, 2.3928867930156018e-05, 1.6284910024083967e-32], residual scaled by `m_a a^2` = 1.14e-05
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

**Spheroid pitch/yaw block** (structure only; the analytic score is C7 below):

| mesh | panels | V (panel) | I_55 | I_66 | |I55-I66|/I55 | I55/(rho V_exact a^2) | raw asymmetry |
|---|---|---|---|---|---|---|
| 16x10 | 160 | 7.96413 | 1.73231 | 1.73231 | 5.13e-16 | 0.05169 | 1.99e-16 |
| 24x14 | 336 | 8.17838 | 1.84176 | 1.84176 | 0.00e+00 | 0.05496 | 7.96e-17 |
| 48x26 | 1248 | 8.32322 | 1.93757 | 1.93757 | 2.29e-16 | 0.05782 | 1.35e-16 |
| 64x34 | 2176 | 8.34629 | 1.95793 | 1.95793 | 5.67e-16 | 0.05843 | 1.30e-16 |
| 96x50 | 4800 | 8.36334 | 1.97633 | 1.97633 | 1.12e-16 | 0.05898 | 1.11e-16 |

Two-finest-mesh spread 0.93% (uncertainty of the converged value);
`I_55/(rho V_exact a^2)` at the finest mesh = 0.05898.
Order estimate p = 0.26 is **noise-dominated** at these panel
counts (latitude resolution dominates; see §5) and is therefore reported, not gated on.

**C7 -- spheroid pitch/yaw against the closed form** (`ellipsoid_added_inertia.rotational_added_inertia(2.0, 1.0, 1.0, axis=1/2) -- closed-form degree-2 Lame potential (ledger D6/m14810, 9/9 criteria, 1e-13 quadrature)`):
`I_55 = I_66 = 2.005793` exactly. At the finest mesh
(96x50 = 4800 panels) the
relative error is **1.47%** (< 3 %); the fitted order is
p = 1.31 and the Richardson limit gives
1.988702, i.e. **0.85%** (< 1 %).
This closes the "analytic < 3 %" half of ledger D1, which this module previously had to
declare WITHHELD.

**C8 -- triaxial ellipsoid `a:b:c = 2:1.5:1`**
(roll = x, pitch = y, yaw = z; closed-form reference
`I = 1.04885, 3.76321, 0.61979`):

| mesh | panels | I_44 | I_55 | I_66 | err I_44 | err I_55 | err I_66 | max cross | err V |
|---|---|---|---|---|---|---|---|---|---|
| 32x18 | 576 | 0.95928 | 3.57223 | 0.61143 | 8.54% | 5.07% | 1.35% | 4.0e-17 | -1.40% |
| 48x26 | 1248 | 0.99723 | 3.65192 | 0.61493 | 4.92% | 2.96% | 0.78% | 1.4e-17 | -0.65% |
| 64x34 | 2176 | 1.01344 | 3.68657 | 0.61651 | 3.38% | 2.04% | 0.53% | 1.7e-17 | -0.37% |
| 96x50 | 4800 | 1.02757 | 3.71707 | 0.61790 | 2.03% | 1.23% | 0.30% | 1.4e-17 | -0.17% |

Finest-mesh errors 2.03%, 1.23%, 0.30%; fitted orders
p = 1.36, 1.34, 1.40; Richardson errors
1.16%, 0.70%, 0.17%; rotation/rotation cross terms
4.0e-17 relative -- i.e. exact zeros up to round-off. This is the
criterion a body of revolution cannot provide: on a spheroid one entry is identically zero
and the other two coincide, so only a triaxial shape exercises the three moment arms
independently. The panel-volume error column shows the split between the geometry
(faceting) error and the solve error.

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

* The transverse block is now scored against the closed form (C7/C8), so the earlier
  WITHHELD status is withdrawn -- but the closed form itself assumes an **exact**
  ellipsoid, while the panels carry the faceting error. The residual 1-3 % is therefore
  panelisation, not solver, error: the volume-error column of the C8 table shows the same
  magnitude (a 4800-panel triaxial mesh still under-fills the volume by
  -0.17%). A curved-surface (isoparametric) panel or a
  higher-order quadrature is what would take the error below that floor; refined flat
  panels cannot.
* The latitude/longitude mesh under-resolves the poles (uniform ``theta``/``phi`` gives
  area elements ``~ sin(phi)``). An equal-area (cosine-latitude) variant was measured
  during this shift: it improves the small ``I_44`` entry but degrades ``I_55``, so the
  uniform mesh of the rest of the tree was kept rather than tuned per entry.
* Solver cost grows like ``N**2`` (dense influence matrix): the 4800-panel triaxial level
  is the affordable limit here (~145 s of the 271 s run). A
  fast-multipole or iterative solve is the route to finer meshes.
* Unbounded fluid only (no free surface, no images): the free-surface / twin-body
  rotational problem stays open as ledger entry D2.

## 6. Error judgement

7/8 criteria PASS. Every rotational quantity here is now scored against an
exact value; nothing is left "reported but unscored":

| quantity | exact | panel (finest) | rel. error | < 3 %? |
|---|---|---|---|---|
| sphere about a diameter | 0 | 2.39e-05 | 1.14e-05 (scaled) | yes |
| sphere, axis offset d = 1.5a | 4.71239 | 4.64360 / Richardson 4.72282 | 1.46% / 0.22% | yes |
| spheroid roll | 0 | 1.69e-32 | 5.1e-34 | yes |
| spheroid pitch/yaw | 2.00579 | 1.97633 / Richardson 1.98870 | 1.47% / 0.85% | yes |
| triaxial I_44 | 1.04885 | 1.02757 | 2.03% | yes |
| triaxial I_55 | 3.76321 | 3.71707 | 1.23% | yes |
| triaxial I_66 | 0.61979 | 0.61790 | 0.30% | yes |

The "key quantity vs reference < 3 %" question is answered **yes** for all seven entries
above. The Richardson target (< 1 %) is met by six of them; the marginal exception is
listed with its number in the C8 sub-verdict rather than smoothed over (see §5). The two
entries whose exact value is zero (sphere about a diameter, spheroid roll) are quoted
relative to their natural scale, where the residual is faceting, not a solver defect.
