# Reference-point (parallel-axis) algebra of the 6x6 added mass -- debt ledger D8

Ran `python -m tensorfvm.verification.added_mass_parallel_axis`
(`elapsed 171.9 s`, rho = 1, sphere radius 1 m,
source-panel method, constant-strength flat panels, `order=3` surface quadrature).

## 1. Criteria (frozen before the run) and outcome

| id | criterion | value | threshold | verdict |
|---|---|---|---|---|
| C1 | M(r0) = S(r0) M(0) S(r0)^T, max relative residual over 3 bodies x 3 offsets | 1.5602762061707744e-15 | 1e-12 | PASS |
| C2 | sphere offset along the rotation axis: |I_zz|/(m_a a^2) < 1e-9 and I_xx = I_yy | {'abs_I_zz_over_m_a_a2': 1.8429223063276534e-32, 'axisym_rel': 3.8056347291781207e-16} | {'abs_I_zz_over_m_a_a2': 1e-09, 'axisym_rel': 1e-06} | PASS |
| C3 | sphere parallel axis: |I_xx(d e_z) - m_a d^2| / m_a d^2 < 3% at the finest mesh | {'finest_rel_err': -0.00948289644286715, 'observed_order': 1.3913818982762163} | {'finest_rel_err': 0.03, 'observed_order': 1.0} | PASS |
| C7 | asymptotic-range check: three-level Richardson extrapolation within 1% of m_a d^2 | {'richardson_rel_err': -0.014641313010338775, 'observed_order': 1.3913818982762163, 'asymptotic_order_expected': 2.0} | {'richardson_rel_err': 0.01} | FAIL |
| C4 | attribution: I_xx(d e_z) - I_xx(0) = d^2 M_yy(0) to 1e-12 (residual of C3 is the common-mode m_a bias) | {'exact_ratio_rel_err': 6.661338147750939e-16, 'mesh_anisotropy_M_yy_over_M_xx': 2.220446049250313e-16, 'spurious_I_xx_centre_over_m_a_a2': 7.501452212521884e-07} | 1e-12 | PASS |
| C5 | spheroid I_zz(d e_y) vs I_zz(0)_closed + d^2 k_axial rho V < 3% (finest) | {'worst_finest_rel_err': 0.011321749220110533, 'per_ratio': {'2.0': -0.0034554594536397865, '4.0': -0.011321749220110533}} | {'worst_finest_rel_err': 0.03, 'error_not_growing': True} | PASS |
| C6 | power of the test: dropping the parallel-axis term or flipping the moment-arm sign changes M by > 1e-2 (relative) | {'min_detection_margin': 0.4610228470900332} | 0.01 | PASS |

## 2. The identity, as measured (C1)

`M(r0) = S(r0) M(0) S(r0)^T` with `S[3+i, 0:3] = e_i x r0`
(`tensorfvm.panel_added_mass.rigid_shift_matrix`).

| body | offset r0 | panels | max abs residual | relative |
|---|---|---|---|---|
| sphere | (0, 0, 1.5) | 384 | 7.11e-15 | 1.56e-15 |
| sphere | (1.5, 0, 0) | 384 | 2.66e-15 | 5.85e-16 |
| sphere | (0.7, -1.1, 0.4) | 384 | 3.11e-15 | 9.03e-16 |
| spheroid a/b=2 | (0, 0, 1.5) | 384 | 3.55e-15 | 2.82e-16 |
| spheroid a/b=2 | (1.5, 0, 0) | 384 | 1.78e-15 | 1.23e-16 |
| spheroid a/b=2 | (0.7, -1.1, 0.4) | 384 | 2.66e-15 | 3.47e-16 |
| spheroid a/b=4 | (0, 0, 1.5) | 384 | 1.42e-14 | 4.05e-16 |
| spheroid a/b=4 | (1.5, 0, 0) | 384 | 2.13e-14 | 3.41e-16 |
| spheroid a/b=4 | (0.7, -1.1, 0.4) | 384 | 2.84e-14 | 7.05e-16 |

The residual is at the rounding level of the double-precision LU solve: the
source-panel response is translation invariant, so translating the body and
shifting the generators are the *same* operation for this discretisation. Any
index/sign error in the moment arm `e_i x r` breaks this table by O(1), which is
the point (see C6).

## 3. Sphere: the exact anchor survives the offset (C2-C4)

`I_xx = I_yy` (rotation axes perpendicular to the offset `d e_z`), `I_zz` (the
offset direction) must vanish; `m_a = 2.094395`,
`m_a d^2 = 4.7124` with `d = 1.5`.

| mesh | panels | I_xx | I_yy | I_zz | err(I_xx vs m_a d^2) | (I_xx - I_xx(0)) / (d^2 M_yy(0)) - 1 |
|---|---|---|---|---|---|---|---|
| 32x18 | 576 | 4.5912 | 4.5912 | 1.63e-32 | -2.57% | 6.7e-16 |
| 48x27 | 1296 | 4.6457 | 4.6457 | 2.27e-32 | -1.42% | 0.0e+00 |
| 64x36 | 2304 | 4.6677 | 4.6677 | 3.86e-32 | -0.95% | 2.2e-16 |

Richardson (three-level, order `p` estimated from the increments; observed order
from the true errors `p = 1.39`, asymptotic value 2 for a
faceted surface):
`I_xx_inf = 4.6434` -> `-1.46%` of `m_a d^2` (**C7**).
Because the observed order is still below 2 at 2304 panels, the extrapolation is
*not* an improvement over the finest mesh: `|-1.46%| >
|-0.95%|`. C7 is therefore recorded as an honest
**FAIL**: the ladder is not in its asymptotic range, and the number to quote is
the finest-mesh `-0.95%`, not the extrapolation.
No threshold was relaxed after the run.

Column 7 is the decisive one for D8: with `S[3, 0:3] = e_x x (0,0,d) = -d e_y`
the *increment* of the offset moment must equal `d^2 M_yy(0)` exactly, and it
does to `6.7e-16`. The ~1 % residual of the absolute value
(column 6) is therefore the **common-mode** discretisation bias of the
translational added mass `m_a` on the faceted sphere -- the same bias that
`docs/verification-added-mass-potential` reports for the centred body (the mesh
is not perfectly isotropic either: `|M_yy/M_xx - 1| =
4.4e-16`) -- plus the faceted sphere's own
spurious rotational inertia `|I_xx(0)|/(m_a a^2) =
1.1e-05` (exact value 0, D1 C1 allows 1e-2).
Not a parallel-axis error: that step is exact.

## 4. Non-spherical offset anchor (C5)

Prolate spheroid, offset `d e_y` across the yaw axis: the induced translation is
axial, the axial-translation/yaw coupling vanishes by fore-aft symmetry
(`|M_xy^yaw| < 1e-15` in every row of `summary.json`), so

    I_zz(d e_y) = I_zz(0)_closed + d^2 k_axial rho V.

| mesh | panels | V (panel) | I_zz(0) | I_zz(offset, panel) | closed form | err |
|---|---|---|---|---|---|---|
| 32x18 | 576 | 8.2606 | 1.8921 | 17.5702 | 17.8406 | -1.52% |
| 48x27 | 1296 | 8.3254 | 1.9386 | 17.7254 | 17.8406 | -0.65% |
| 64x36 | 2304 | 8.3482 | 1.9587 | 17.7789 | 17.8406 | -0.35% |
| 32x18 | 576 | 16.5212 | 32.8825 | 81.8606 | 83.8269 | -2.35% |
| 48x27 | 1296 | 16.6509 | 33.6468 | 82.8778 | 83.8269 | -1.13% |

`I_zz(0)_closed` comes from `tensorfvm.ellipsoid_added_inertia.rotational_added_inertia`
(D6) and `k_axial` from the Lamb factors -- no invented reference: the two closed
forms are the same family already verified against panel data by D6 and
`docs/verification-added-mass-potential`.

## 5. Does the test have power? (C6, mutations)

| body | drop the parallel-axis term | flip the moment-arm sign | 
|---|---|---|
| sphere | 1.00 | 1.29e+00 |
| spheroid a/b=2 | 1.00 | 1.61e+00 |
| spheroid a/b=4 | 0.46 | 7.40e-01 |

Column 2 shows that the naive "same inertia wherever you measure it" reading is
off by ~100 % -- exactly the quantity D8 mis-read. Column 3 quantifies the sign
sensitivity of the coupling blocks. Both exceed the `1e-2` detection threshold,
so C1-C5 are not vacuous.

## 6. D8 resolved: where could `2.05` come from?

Every rotational entry of the panel matrix for the three configurations named in
the D8 note (`n_theta x n_phi = 32 x 18`, `n = 576` panels):

| configuration | translational M_xx/M_yy/M_zz | rotational I_xx/I_yy/I_zz | entry closest to 2.05 |
|---|---|---|---|
| sphere centre (0, 0, 0) | 2.0405 / 2.0405 / 2.0520 | 2.39e-05 / 0.00 / 1.63e-32 | 2.0520 @ (2, 2) |
| sphere centre (0, 0, 1.5a) | 2.0405 / 2.0405 / 2.0520 | 4.59e+00 / 4.59 / 1.63e-32 | 2.0520 @ (2, 2) |
| sphere centre (1.5a, 0, 0) | 2.0405 / 2.0405 / 2.0520 | 2.39e-05 / 4.62 / 4.59e+00 | 2.0520 @ (2, 2) |

- configurations with a **rotational** entry within 0.05 of 2.05: none;
- configurations with a **translational** entry within 0.05 of 2.05: ['sphere centre (0, 0, 0)', 'sphere centre (0, 0, 1.5a)', 'sphere centre (1.5a, 0, 0)'].

Verdict: **unreproducible in the rotational block; the only 2.05-shaped value is the z-translational added mass M_zz**. `I_zz` is `0` for the axis offset and `4.59` (vs the
exact `4.71`) for the transverse offset at this mesh; neither is 2.05, and no
rotational entry of any tested configuration is. The only 2.05-shaped number in
the output is the z-translational added mass `M_zz`, which is what a rotational
check would return if it indexed the translational directions `(0,1,2)` -- or the
matrix diagonal -- instead of `(3,4,5)`. The D8 note is therefore recorded as a
**readout error of the cross-check**, not a defect of
`added_mass_matrix_6dof`; the merged D1 criterion C3 stands.

## 7. What this does and does not establish

- Established: the 6x6 reference-point algebra (`e_i x r` arm, index split
  `0..2 / 3..5`, coupling signs) is exact to rounding; the sphere parallel-axis
  and the spheroid offset anchors are met within 1.13%; the D1 claim is not
  contradicted by any configuration of this code path.
- Not established: no offset anchor exists for the spheroid *transverse*
  coupling entries (the closed form would need the axial/yaw coupling potential),
  roll about a non-symmetric axis, or the free-surface image case (D2 territory).
  Viscosity, forward speed and lifting effects are out of scope, as in D1/D6.

## 8. Reproduction

```bash
PYTHONPATH=src python -m tensorfvm.verification.added_mass_parallel_axis
PYTHONPATH=src python scripts/plot_added_mass_parallel_axis.py --dir docs/verification-added-mass-parallel-axis
PYTHONPATH=src python -m pytest tests/test_added_mass_parallel_axis.py -q
```
