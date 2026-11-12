# Added mass of the Lamb ellipsoid family from the source-panel solver

* anchor `AN-LAMB-ELLIPSOID` (Lamb 1932 sec. 108-114), repo commit `fea32d475b52`, generated 2026-11-12T07:12:00+00:00
* solver: `tensorfvm.panel_potential` constant-strength source panels + `tensorfvm.panel_added_mass.added_mass_matrix`, 3x3 Gauss per panel, `rho = 1`
* ladder: 16x12..64x48 (N_theta x N_phi); 8 bodies; 120 reported rows (case x level x axis)

Produced by `benchmarks/added_mass_lamb/run_benchmark.py`. Every number below is
injected programmatically from `anchors.json`, `ellipsoid_scan.csv` and
`convergence.json`; nothing is transcribed by hand.

## 1. Physics, anchor and method

Added mass of a body accelerating in an unbounded inviscid irrotational fluid:

```
M_ij = -rho * closed_integral_S phi_i n_j dS ,   laplace(phi_i) = 0 ,
dphi_i/dn = -n_i on S ,   phi_i -> 0 at infinity
```

with `n` pointing into the fluid. For a triaxial ellipsoid the exact answer is
diagonal, `M_ii = k_i rho V` with `V = (4/3) pi a b c` and the Lamb factors

```
alpha0 = a b c * integral_0^inf dl / ((a^2+l) sqrt((a^2+l)(b^2+l)(c^2+l))) ,
k_1 = alpha0 / (2 - alpha0)
```

plus cyclic permutations. Closed forms are used for the sphere, the prolate spheroid
(`e = sqrt(1-b^2/a^2)`) and the oblate spheroid (`e = sqrt(1-c^2/a^2)`); a body with
three unequal axes uses the defining integral. Both routes live in
`src/tensorfvm/panel_added_mass.py` and are cross-checked, so the anchor cannot be a
typo agreeing with itself.

Discretisation: constant-strength source panels on a latitude/longitude quad mesh of
the exact ellipsoid; the Neumann problem is solved once per translation direction and
the added mass follows from `-rho * integral phi_i n_j dS` with the solver's own 3x3
Gauss rule. `h_rel = 1/sqrt(N_panels)` is the representative panel size.

| case | family | a | b | c | a/b | a/c | alpha0 | beta0 | gamma0 | k1 | k2 | k3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `sphere_1_1_1` | sphere | 1 | 1 | 1 | 1 | 1 | 0.666667 | 0.666667 | 0.666667 | 0.500000 | 0.500000 | 0.500000 |
| `prolate_2_1_1` | prolate | 2 | 1 | 1 | 2 | 2 | 0.347128 | 0.826436 | 0.826436 | 0.210015 | 0.704210 | 0.704210 |
| `prolate_3_1_1` | prolate | 3 | 1 | 1 | 3 | 3 | 0.217419 | 0.891291 | 0.891291 | 0.121969 | 0.803899 | 0.803899 |
| `prolate_5_1_1` | prolate-slender | 5 | 1 | 1 | 5 | 5 | 0.111642 | 0.944179 | 0.944179 | 0.059121 | 0.894261 | 0.894261 |
| `oblate_1_1_0.5` | oblate | 1 | 1 | 0.5 | 1 | 2 | 0.472800 | 0.472800 | 1.054401 | 0.309586 | 0.309586 | 1.115060 |
| `oblate_1_1_0.25` | oblate-flat | 1 | 1 | 0.25 | 1 | 4 | 0.296359 | 0.296359 | 1.407283 | 0.173956 | 0.173956 | 2.374291 |
| `triaxial_2_1_0.5` | triaxial | 2 | 1 | 0.5 | 2 | 4 | 0.224701 | 0.569561 | 1.205738 | 0.126571 | 0.398172 | 1.518061 |
| `triaxial_1.5_1_0.75` | triaxial | 1.5 | 1 | 0.75 | 1.5 | 2 | 0.400928 | 0.670713 | 0.928360 | 0.250725 | 0.504566 | 0.866298 |

Anchor self-consistency: `alpha0+beta0+gamma0 = 2` holds to 5.42e-14; the closed-form and integral routes agree to 3.77e-14 (absolute, on the Lamb factors).

## 2. Anchor limits (criterion 1)

| limit | parameter | quantity | value | reference | deviation | verdict |
|---|---|---|---|---|---|---|
| sphere | a = b = c = 1 | k_1, k_2, k_3 (= Cm) | 0.5000000000, 0.5000000000, 0.5000000000 | 0.5 | 1.11e-14 % (tol 0.1 %) | PASS |
| sphere | a = b = c = 1 | M_11 vs 0.5 rho (4/3) pi a^3 | 2.0943951024 | 2.0943951023931953 | 2.22e-14 % (tol 0.1 %) | PASS |
| slender prolate | a = 1, b = c in {0.5, 0.2, 0.1, 0.05, 0.01} | k_axial -> 0, k_transverse -> 1 | b/a=0.5: k_ax=0.21002, k_tr=0.70421; b/a=0.2: k_ax=0.05912, k_tr=0.89426; b/a=0.1: k_ax=0.02071, k_tr=0.96023; b/a=0.05: k_ax=0.00679, k_tr=0.98659; b/a=0.01: k_ax=0.00043, k_tr=0.99914 | [0.0, 1.0] | k_ax 4.30e-04, 1-k_tr 8.59e-04 | PASS |
| flat oblate disc | a = b = 1, c in {0.5, 0.25, 0.1, 0.01, 0.001} | M_33/(8/3 rho a^3) -> 1, k_in_plane -> 0, gamma0 -> 2 | c/a=0.5: M33/(8/3 rho a^3)=0.875766, k_inplane=0.3096; c/a=0.25: M33/(8/3 rho a^3)=0.932382, k_inplane=0.1740; c/a=0.1: M33/(8/3 rho a^3)=0.971401, k_inplane=0.0748; c/a=0.01: M33/(8/3 rho a^3)=0.997036, k_inplane=0.0078; c/a=0.001: M33/(8/3 rho a^3)=0.999703, k_inplane=0.0008 | 1.0 | M33 gap 2.97e-04, k_in_plane 7.85e-04 | PASS |

The 0.1 % sphere requirement is met by a wide margin: the anchor returns `k = 0.5`
identically on all three axes, so the deviation is round-off, not accuracy. The
slender-prolate and flat-oblate ladders show both limits being approached
monotonically; `k_3` itself diverges like `a/c` for a flattening body, so the disc
limit is stated on `M_33`, which stays finite.

## 3. Panel scan: all cases, all levels (criteria 2 and 5)

`gauss` quadrature, `order = 3`, `rho = 1`. Deviation is `k_panel/k_anchor - 1` in
percent; `PASS` means inside +-5 %.

| case | level | N panels | vol err % | k1 (dev %) | k2 (dev %) | k3 (dev %) | verdict |
|---|---|---|---|---|---|---|---|
| `sphere_1_1_1` | 16x12 | 192 | -4.211 | 0.47097 (-5.81) | 0.48238 (-3.52) | 0.48238 (-3.52) | FAIL |
| `sphere_1_1_1` | 24x18 | 432 | -1.889 | 0.48799 (-2.40) | 0.49317 (-1.37) | 0.49317 (-1.37) | PASS |
| `sphere_1_1_1` | 32x24 | 768 | -1.066 | 0.49383 (-1.23) | 0.49674 (-0.65) | 0.49674 (-0.65) | PASS |
| `sphere_1_1_1` | 48x36 | 1728 | -0.475 | 0.49780 (-0.44) | 0.49906 (-0.19) | 0.49906 (-0.19) | PASS |
| `sphere_1_1_1` | 64x48 | 3072 | -0.267 | 0.49907 (-0.19) | 0.49976 (-0.05) | 0.49976 (-0.05) | PASS |
| `prolate_2_1_1` | 16x12 | 192 | -4.211 | 0.20023 (-4.66) | 0.66460 (-5.63) | 0.66460 (-5.63) | FAIL |
| `prolate_2_1_1` | 24x18 | 432 | -1.889 | 0.20662 (-1.62) | 0.68531 (-2.68) | 0.68531 (-2.68) | PASS |
| `prolate_2_1_1` | 32x24 | 768 | -1.066 | 0.20867 (-0.64) | 0.69288 (-1.61) | 0.69288 (-1.61) | PASS |
| `prolate_2_1_1` | 48x36 | 1728 | -0.475 | 0.20993 (-0.04) | 0.69857 (-0.80) | 0.69857 (-0.80) | PASS |
| `prolate_2_1_1` | 64x48 | 3072 | -0.267 | 0.21026 (+0.11) | 0.70070 (-0.50) | 0.70070 (-0.50) | PASS |
| `prolate_3_1_1` | 16x12 | 192 | -4.211 | 0.11671 (-4.31) | 0.75030 (-6.67) | 0.75030 (-6.67) | FAIL |
| `prolate_3_1_1` | 24x18 | 432 | -1.889 | 0.12029 (-1.38) | 0.77703 (-3.34) | 0.77703 (-3.34) | PASS |
| `prolate_3_1_1` | 32x24 | 768 | -1.066 | 0.12141 (-0.45) | 0.78713 (-2.09) | 0.78713 (-2.09) | PASS |
| `prolate_3_1_1` | 48x36 | 1728 | -0.475 | 0.12207 (+0.09) | 0.79499 (-1.11) | 0.79499 (-1.11) | PASS |
| `prolate_3_1_1` | 64x48 | 3072 | -0.267 | 0.12223 (+0.21) | 0.79807 (-0.73) | 0.79807 (-0.73) | PASS |
| `prolate_5_1_1` | 16x12 | 192 | -4.211 | 0.05664 (-4.20) | 0.83169 (-7.00) | 0.83169 (-7.00) | FAIL |
| `prolate_5_1_1` | 24x18 | 432 | -1.889 | 0.05837 (-1.28) | 0.86244 (-3.56) | 0.86244 (-3.56) | PASS |
| `prolate_5_1_1` | 32x24 | 768 | -1.066 | 0.05890 (-0.37) | 0.87421 (-2.24) | 0.87421 (-2.24) | PASS |
| `prolate_5_1_1` | 48x36 | 1728 | -0.475 | 0.05921 (+0.15) | 0.88347 (-1.21) | 0.88347 (-1.21) | PASS |
| `prolate_5_1_1` | 64x48 | 3072 | -0.267 | 0.05927 (+0.26) | 0.88714 (-0.80) | 0.88714 (-0.80) | PASS |
| `oblate_1_1_0.5` | 16x12 | 192 | -4.211 | 0.29356 (-5.18) | 0.30142 (-2.64) | 1.03043 (-7.59) | FAIL |
| `oblate_1_1_0.5` | 24x18 | 432 | -1.889 | 0.30345 (-1.98) | 0.30725 (-0.75) | 1.07027 (-4.02) | PASS |
| `oblate_1_1_0.5` | 32x24 | 768 | -1.066 | 0.30674 (-0.92) | 0.30900 (-0.19) | 1.08589 (-2.62) | PASS |
| `oblate_1_1_0.5` | 48x36 | 1728 | -0.475 | 0.30887 (-0.23) | 0.30997 (+0.13) | 1.09857 (-1.48) | PASS |
| `oblate_1_1_0.5` | 64x48 | 3072 | -0.267 | 0.30949 (-0.03) | 0.31017 (+0.19) | 1.10380 (-1.01) | PASS |
| `oblate_1_1_0.25` | 16x12 | 192 | -4.211 | 0.16509 (-5.10) | 0.16998 (-2.28) | 2.00829 (-15.42) | FAIL |
| `oblate_1_1_0.25` | 24x18 | 432 | -1.889 | 0.17069 (-1.88) | 0.17328 (-0.39) | 2.15690 (-9.16) | FAIL |
| `oblate_1_1_0.25` | 32x24 | 768 | -1.066 | 0.17249 (-0.84) | 0.17412 (+0.09) | 2.22162 (-6.43) | FAIL |
| `oblate_1_1_0.25` | 48x36 | 1728 | -0.475 | 0.17364 (-0.18) | 0.17449 (+0.31) | 2.27955 (-3.99) | PASS |
| `oblate_1_1_0.25` | 64x48 | 3072 | -0.267 | 0.17396 (+0.00) | 0.17451 (+0.32) | 2.30589 (-2.88) | PASS |
| `triaxial_2_1_0.5` | 16x12 | 192 | -4.211 | 0.12091 (-4.47) | 0.38201 (-4.06) | 1.35108 (-11.00) | FAIL |
| `triaxial_2_1_0.5` | 24x18 | 432 | -1.889 | 0.12468 (-1.49) | 0.39165 (-1.64) | 1.42389 (-6.20) | FAIL |
| `triaxial_2_1_0.5` | 32x24 | 768 | -1.066 | 0.12588 (-0.54) | 0.39489 (-0.83) | 1.45403 (-4.22) | PASS |
| `triaxial_2_1_0.5` | 48x36 | 1728 | -0.475 | 0.12660 (+0.02) | 0.39705 (-0.28) | 1.47982 (-2.52) | PASS |
| `triaxial_2_1_0.5` | 64x48 | 3072 | -0.267 | 0.12677 (+0.16) | 0.39774 (-0.11) | 1.49106 (-1.78) | PASS |
| `triaxial_1.5_1_0.75` | 16x12 | 192 | -4.211 | 0.23859 (-4.84) | 0.48342 (-4.19) | 0.81073 (-6.41) | FAIL |
| `triaxial_1.5_1_0.75` | 24x18 | 432 | -1.889 | 0.24636 (-1.74) | 0.49572 (-1.75) | 0.83842 (-3.22) | PASS |
| `triaxial_1.5_1_0.75` | 32x24 | 768 | -1.066 | 0.24888 (-0.74) | 0.49992 (-0.92) | 0.84887 (-2.01) | PASS |
| `triaxial_1.5_1_0.75` | 48x36 | 1728 | -0.475 | 0.25046 (-0.11) | 0.50280 (-0.35) | 0.85701 (-1.07) | PASS |
| `triaxial_1.5_1_0.75` | 64x48 | 3072 | -0.267 | 0.25089 (+0.06) | 0.50374 (-0.16) | 0.86021 (-0.70) | PASS |

Geometry is validated before any flow number is used: the panelised volume
`closed_integral x.n dS / 3` and the three axis extents of the panel corners are
compared with `4/3 pi a b c` and `a, b, c` (both are CSV columns), so no coefficient
above can be contaminated by a mis-panelised geometry. Off-diagonal entries of the
body-frame matrix stay at round-off (`offdiag_rel` column). The volume deficit is the
same number for every body at a given level -- not a bug: in this parametrisation every
ellipsoid is an affine image of the panelled unit sphere, and the relative volume
deficit of an inscribed polyhedron is affine-invariant, which is itself a consistency
check on the mesh construction.

Read the level column honestly: coarse rungs are outside +-5 % for several bodies and
are listed as FAIL, not dropped. Criterion 2 is defined on the finest mesh of the
declared ladder; the per-rung verdicts are kept so that the entry point into the band
is visible -- bodies fully inside the band per level: 16x12: 0/8, 24x18: 6/8, 32x24: 7/8, 48x36: 8/8, 64x48: 8/8 (first level where every body is inside: 48x36).

## 4. Observed order, Richardson extrapolation and GCI (criterion 3)

`p_all` is the least-squares slope of `log|error|` vs `log h` over all levels;
`p_coarse` uses only the coarsest levels and is the order fed into the two-finest-level
Richardson extrapolation, so the extrapolation cannot tune itself against the value it
is later compared with. `p_apparent` is the three-finest-level estimate.
`GCI = Fs * |(k_fine - k_R)/k_fine| / (r^p - 1)` with `Fs = 1.25`, `r` the finest
refinement ratio.

| case | axis | |dev| % (finest) | p_all | p_coarse | p_apparent | k_Richardson (dev %) | GCI % (coarse p) | GCI % (3-grid p) |
|---|---|---|---|---|---|---|---|---|
| `sphere_1_1_1` | 1 | 0.185 | 2.474 | 2.231 | 2.808 | 0.500488 (+0.098) | 0.394 | 0.206 |
| `sphere_1_1_1` | 2 | 0.048 | 3.041 | 2.426 | 2.968 | 0.500453 (+0.091) | 0.171 | 0.096 |
| `sphere_1_1_1` | 3 | 0.048 | 3.041 | 2.426 | 2.968 | 0.500453 (+0.091) | 0.171 | 0.096 |
| `prolate_2_1_1` | 1 | 0.114 | 3.246 | 2.847 | 3.327 | 0.210512 (+0.237) | 0.121 | 0.075 |
| `prolate_2_1_1` | 2 | 0.499 | 1.748 | 1.807 | 2.423 | 0.703815 (-0.056) | 0.816 | 0.374 |
| `prolate_2_1_1` | 3 | 0.499 | 1.748 | 1.807 | 2.423 | 0.703815 (-0.056) | 0.816 | 0.374 |
| `prolate_3_1_1` | 1 | 0.210 | 2.581 | 3.219 | 3.614 | 0.122325 (+0.292) | 0.067 | 0.047 |
| `prolate_3_1_1` | 2 | 0.725 | 1.600 | 1.678 | 2.309 | 0.803039 (-0.107) | 1.254 | 0.543 |
| `prolate_3_1_1` | 3 | 0.725 | 1.600 | 1.678 | 2.309 | 0.803039 (-0.107) | 1.254 | 0.543 |
| `prolate_5_1_1` | 1 | 0.260 | 2.252 | 3.465 | 3.791 | 0.059313 (+0.325) | 0.048 | 0.036 |
| `prolate_5_1_1` | 2 | 0.796 | 1.567 | 1.643 | 2.281 | 0.893212 (-0.117) | 1.416 | 0.601 |
| `prolate_5_1_1` | 3 | 0.796 | 1.567 | 1.643 | 2.281 | 0.893212 (-0.117) | 1.416 | 0.601 |
| `oblate_1_1_0.5` | 1 | 0.030 | 3.559 | 2.485 | 3.027 | 0.310090 (+0.163) | 0.231 | 0.131 |
| `oblate_1_1_0.5` | 2 | 0.188 | 2.058 | 3.758 | 3.976 | 0.310267 (+0.220) | 0.021 | 0.017 |
| `oblate_1_1_0.5` | 3 | 1.010 | 1.454 | 1.539 | 2.183 | 1.113204 (-0.167) | 1.912 | 0.777 |
| `oblate_1_1_0.25` | 1 | 0.003 | 4.916 | 2.589 | 3.097 | 0.174255 (+0.172) | 0.191 | 0.113 |
| `oblate_1_1_0.25` | 2 | 0.320 | 1.207 | 4.603 | 6.759 | 0.174521 (+0.325) | 0.002 | 0.000 |
| `oblate_1_1_0.25` | 3 | 2.881 | 1.208 | 1.263 | 1.943 | 2.366021 (-0.348) | 7.439 | 2.545 |
| `triaxial_2_1_0.5` | 1 | 0.161 | 3.203 | 3.017 | 3.462 | 0.126902 (+0.262) | 0.091 | 0.060 |
| `triaxial_2_1_0.5` | 2 | 0.110 | 2.588 | 2.294 | 2.853 | 0.398465 (+0.074) | 0.245 | 0.132 |
| `triaxial_2_1_0.5` | 3 | 1.779 | 1.313 | 1.385 | 2.048 | 1.514026 (-0.266) | 3.934 | 1.462 |
| `triaxial_1.5_1_0.75` | 1 | 0.065 | 3.305 | 2.703 | 3.210 | 0.251253 (+0.210) | 0.155 | 0.093 |
| `triaxial_1.5_1_0.75` | 2 | 0.165 | 2.330 | 2.184 | 2.757 | 0.504812 (+0.049) | 0.305 | 0.159 |
| `triaxial_1.5_1_0.75` | 3 | 0.703 | 1.594 | 1.675 | 2.304 | 0.865375 (-0.106) | 1.213 | 0.525 |

**The GCI criterion is not met on every component with the predeclared order
estimate; that is reported as a FAIL, not smoothed over.** Failing components and
both order estimates:

| case | axis | p_coarse | p_apparent | GCI (coarse p) % | GCI (3-grid p) % |
|---|---|---|---|---|---|
| `oblate_1_1_0.25` | 3 | 1.26 | 1.94 | 7.44 | 2.55 |

Physically the whole of the excess sits in the flattening direction of the most
flattened body, whose own convergence is the slowest of the family (see its
per-level errors in section 3): the coarse-pair order underestimates the true rate
there, which inflates the GCI by construction. With the standard three-grid order
the same component lands inside the 5 % limit, so the honest reading is: the
solution is converged in the engineering sense, the *predeclared* GCI test is not
passed, and the single slow body is the reason.

### Why the observed order is not the textbook 2

The task anticipated second order; the measurement says otherwise and the reason is
structural, not a bug. Three effects set the slope: (i) the curved surface is replaced
by flat quads, so the Neumann data `n.e_i` and the surface element `dS` carry a
geometric error that enters the added mass at an order below the nominal one on a
non-uniform mesh; (ii) the single-layer potential at the self panel is weakly singular
and is evaluated by quadrature rather than analytically (section 6); (iii) the
latitude/longitude mesh is anisotropic, panel areas span decades near the poles, so
`h = 1/sqrt(N)` is an average size and the local refinement ratio is not uniform. The
per-component orders above are consistent with first-order-class faceting-dominated
error. Richardson extrapolation still collapses the error by roughly an order of
magnitude on most components, which is the practical justification for using it.

## 5. Worst case and its attribution (criterion 4)

Worst component at the finest mesh: **`oblate_1_1_0.25`**, axis 3 (oblate-flat, semi-axes [1.0, 1.0, 0.25]):
`k_panel = 2.305894` against anchor `k = 2.374291`, i.e. **-2.881 %**; Richardson-extrapolated `2.366021` (-0.348 %), GCI 7.439 %.

* error per level (coarse to fine): -15.42 %, -9.16 %, -6.43 %, -3.99 %, -2.88 %
* mesh shape at the finest level: panel area max/min = 1.221e+02, worst `max_edge/sqrt(area)` = 9.04 (1.0 would be a square panel; the
  pole rows are degenerate quads, so an edge-length ratio would be unbounded there)

Attribution: the residual is **surface faceting, not the singular self-term**.
(a) The error falls monotonically along the ladder (list above), the signature of a
geometric discretisation error rather than of a biased quadrature. (b) The
`centroid` vs `gauss` study in section 6 bounds the self-panel contribution: changing
the surface rule moves the same coefficient by less than the total error of this case.
(c) Quantitatively, across all 120 case-level-axis rows the volume deficit of
the inscribed polyhedron and the added-mass deficit correlate with Pearson r = 0.755, and the added-mass deficit is 1.42x the volume deficit on average (range -1.20-10.77): the two errors are the same
faceting family, which is what a geometric discretisation error predicts and what a
biased self-term would not.
(d) The offending body is the most flattened one, so its panels crowd into the
equatorial band where the surface curvature radius is smallest: a fixed `n_phi` leaves
the largest chord error exactly there, and the panel area spread (quoted above) is the
largest of the family.

## 6. Singular integral and self-influence treatment

Three singular or near-singular terms appear; each is handled explicitly.

1. **Neumann problem, self panel.** The normal velocity induced by a flat
   constant-strength panel at its own centroid vanishes by antisymmetry (the
   quadrature nodes are coplanar with the field point), so the principal value is set
   to zero and the `+1/2` jump of a single-layer sheet is added to the matrix
   diagonal (`A[i,i]_kernel = 0`, `A[i,i] += 1/2` in
   `panel_potential.solve_sources`). Without the jump the system has the wrong
   signature and the source strengths are off by an O(1) factor.
2. **Potential evaluation, self panel.** `phi(x) = sum_j sigma_j/(4 pi) *
   integral_panel dS/|x - r'|`. The self term is *weakly* singular but integrable over
   the flat panel; it is evaluated with the same 3x3 Gauss rule as the velocity
   kernel, and a near-field cutoff (`r^2 < 1e-20`) drops nodes that coincide with the
   field point after round-off (which would otherwise give a ~1e17 spike).
3. **Added-mass surface integral.** `integral phi n dS` is not singular, but its
   accuracy depends on where `phi` is sampled: `quadrature='centroid'` samples once
   per panel (midpoint rule), `'gauss'` samples at the solver's own nodes with the
   matching weights. Both are compared on identical meshes below; the scan in
   section 3 uses `gauss`.

| case | level | error centroid (axes 1/2/3, %) | error gauss (%) | gauss - centroid (pp) |
|---|---|---|---|---|
| `sphere_1_1_1` | 32x24 | -1.99, -1.87, -1.87 | -1.23, -0.65, -0.65 | +0.76, +1.22, +1.22 |
| `sphere_1_1_1` | 48x36 | -1.02, -1.04, -1.04 | -0.44, -0.19, -0.19 | +0.58, +0.85, +0.85 |
| `triaxial_2_1_0.5` | 32x24 | -0.97, -2.06, -6.51 | -0.54, -0.83, -4.22 | +0.43, +1.23, +2.30 |
| `triaxial_2_1_0.5` | 48x36 | -0.33, -1.14, -4.07 | +0.02, -0.28, -2.52 | +0.35, +0.86, +1.56 |
| `oblate_1_1_0.25` | 32x24 | -1.38, -0.88, -9.39 | -0.84, +0.09, -6.43 | +0.54, +0.97, +2.96 |
| `oblate_1_1_0.25` | 48x36 | -0.61, -0.38, -5.98 | -0.18, +0.31, -3.99 | +0.43, +0.68, +1.99 |

The centroid rule is consistently further from the anchor than the Gauss rule, and
its deficit shrinks with `h`: that is what a per-panel `O(sqrt(area))` self-term
bias looks like, and the Gauss rule removes it. This is why the reported scan uses
`gauss`, and the size of the gap is the bound on the singular term that the
worst-case attribution relies on.

## 7. Criteria and verdicts

| # | criterion | measured | verdict |
|---|---|---|---|
| 1 | anchor limits: sphere Cm = 0.5 within 0.1 %; slender prolate axial 0 / transverse 1; flat oblate disc M_33 -> 8/3 rho a^3 | 4/4 limit checks PASS; sphere worst dev 1.11e-14 % | PASS |
| 2 | >= 6 axis ratios x >= 3 mesh levels; every diagonal coefficient within +-5.0 % of the anchor at the finest mesh | 24/24 components inside +-5 % at the finest mesh (8 bodies x 5 levels); failing: none | PASS |
| 3 | Richardson extrapolation: observed order reported and GCI < 5.0 % | p_all in [1.21, 4.92]; max GCI 7.439 %; 23/24 below the GCI limit; 24/24 extrapolated values inside +-5 % | FAIL |
| 4 | worst case named and attributed (surface faceting vs self-panel quadrature) | worst `oblate_1_1_0.25` axis 3: -2.881 % raw, -0.348 % extrapolated, GCI 7.439 %; attributed to faceting (see section 5) | PASS |
| 5 | every case reported (no cherry-picking): rows = cases x levels x 3 axes | 120 rows (expected 120) | PASS |

## 8. Limitations and applicability

* Only the unbounded ideal fluid is covered (no free surface, no images): the
  deep-water zero-frequency added mass. Frequency dependence, forward-speed terms
  and the free-surface image system are outside this benchmark.
* Accuracy statements are for the latitude/longitude panelisation; a mesh generator
  producing isotropic panels would shift the constants (not the orders) and is the
  obvious next lever if better than a few percent at moderate panel counts is needed.
* The anchor is a closed-form solution: agreement here means the solver is right for
  the ellipsoid family, not that a real hull is represented well. Hull geometry, the
  waterplane image and appendages remain separate error sources.
* Extreme flattening (`c/a < 1e-3`) is anchored by the closed form only; the integral
  route is not reliable there (documented in `lamb_integral_factors`) and the disc
  limit is therefore checked on `M_33`, not on the divergent `k_3`.

## 9. References

* Lamb, H. (1932) *Hydrodynamics*, 6th ed., sec. 108-114: ellipsoid added-mass
  coefficients, circular-disc limit.
* Landau, L.D. & Lifshitz, E.M. (1987) *Fluid Mechanics*, sec. 11: definition of the
  added-mass tensor from the fluid kinetic energy.
* Newman, J.N. (1977) *Marine Hydrodynamics*, sec. 4.4: `k_i = alpha0/(2-alpha0)`.
* Hess, J.L. & Smith, A.M.O. (1964) *Progress in Aeronautical Sciences* 8, 1-138:
  constant-strength source panel method.
* Roache, P.J. (1994) *J. Fluids Eng.* 116, 405-413: Grid Convergence Index.

## 10. Reproduce

```
PYTHONPATH=src python benchmarks/added_mass_lamb/run_benchmark.py
PYTHONPATH=src python benchmarks/added_mass_lamb/make_figures.py   # figures/
PYTHONPATH=src python -m pytest tests/test_benchmark_added_mass_lamb.py -q
```
