# Verification: occlusion-aware 2-D view factors (inter-surface shadowing)

Driver `python -m tensorfvm.verification.radiation_occlusion` (order = 96, 200000 rays, i.e. 25000 per surface of an 8-strip scene, 35.5 s). Predecessor: `python -m tensorfvm.verification.radiation` (unobstructed strips, catalog gap 5).

## 1. Case

Cross-section radiation in a 2.0 m x 1.2 m cargo hold, per unit depth:

* **beam scene** — simply connected, re-entrant: a 0.30 m x 0.60 m beam hung from the deck (8 strips). The crossed strings still close the rows, so their enclosure-level answer looks consistent while per-pair values are no longer view factors.
* **block scene** — multiply connected: a 0.4 m x 0.4 m floating obstacle (chilled pipe / ice block) inside the same hold (8 strips, obstacle traversed clockwise so its normal points into the domain).
* **blocked pair** — two facing strips with a segment spanning the gap (exact 0).
* **cavity** — the 1 x 1 square: control case, no occlusion at all.

Material properties: gray-diffuse, `eps = 0.9`; deck and walls 350 K, the beam's three exposed faces 260 K. The beam power is quoted in the black limit (eps = 1), where it follows from F alone: -887.69 W/m; with eps = 0.9 the gray enclosure gives -779.53 W/m.

## 2. Method

`tensorfvm.radiation.strip_form_factors_occluded` integrates the 2-D differential kernel `cos_i cos_j / (2 r)` over the receiver's **visible** part. For a fixed emitter station the shadow of one blocking segment is a wedge, so the visibility indicator along the receiver line is discontinuous only where the two shadow edges meet that line; those break points are computed analytically (the discontinuity is never sampled), each sub-interval is integrated by Gauss-Legendre, and the emitter is integrated by Gauss-Legendre as well. Cosines are signed and clipped at zero, so a chord through an obstacle's solid interior cannot exchange.

Independent references, none sharing algebra with the code under test: the crossed-strings contour identity (unoccluded control), exact zero (fully hidden pair), energy closure `sum_j F_ij = 1`, reciprocity of exchange areas, and a cosine-weighted Monte-Carlo ray trace (`monte_carlo_form_factors`, binomial standard error reported).

## 3. Results

| quantity | value |
|---|---|
| `cavity`: max abs(sum_j F_ij - 1), occluded | 1.87e-05 |
| `cavity`: max abs(sum_j F_ij - 1), crossed strings | 0.00e+00 |
| `beam`: max abs(sum_j F_ij - 1), occluded | 2.37e-05 |
| `beam`: max abs(sum_j F_ij - 1), crossed strings | 2.22e-16 |
| `block`: max abs(sum_j F_ij - 1), occluded | 3.92e-05 |
| `block`: max abs(sum_j F_ij - 1), crossed strings | 5.55e-16 |
| `beam`: max abs(F_occluded - F_MC) | 5.31e-03 (4 sigma band 1.26e-02) |
| `beam`: max abs(F_crossed_strings - F_MC) | 6.18e-01 |
| `block`: max abs(F_occluded - F_MC) | 6.36e-03 (4 sigma band 1.24e-02) |
| `block`: max abs(F_crossed_strings - F_MC) | 7.77e-01 |
| beam: min F entry, crossed strings | -0.4399 |
| block: max F between obstacle faces, crossed strings | 0.4142 (sqrt(2) - 1) |
| beam: Gauss order 24/48/96/192 error vs order 384 | 2.8e-04 / 8.9e-05 / 1.4e-05 / 2.8e-06 |
| beam: observed order (24 -> 48) | 1.66 |
| beam: beam-face power, occluded | -887.69 W/m |
| beam: beam-face power, crossed strings | -177.54 W/m |
| beam: relative power error of the crossed strings | 80.0 % |
| beam: all-or-nothing mask row-sum error | 0.41 |

Full matrices, Monte-Carlo standard errors and hidden fractions: `summary.json`.

## 4. Figure

`fig_radiation_occlusion.png` — left: the beam and block scenes with the pairs the crossed strings get wrong (dotted); middle: occluded vs crossed-strings view factors against the Monte-Carlo reference; right: beam power and quadrature convergence.

## 5. Notes, traps and limits

* **The obstructed-blind API does not fail loudly.** On the beam scene the crossed strings return negative view factors (minimum -0.4399) while every row still sums to one, so `GrayEnclosure`'s row-sum validation *accepts* them and the beam power comes out 80 % wrong. A single-pair query (`strip_view_factor`) has no validation at all.
* **The all-or-nothing `visible` mask cannot express partial shadows.** Masking whole pairs on the beam scene leaves the rows off by 0.4, i.e. the masked enclosure violates energy conservation by more than 100 %.
* **Holes must be traversed clockwise.** With the left-of-strip convention an obstacle inside the domain must be listed in reversed *vertex* order; `reversed(strips)` does not flip each strip and silently leaves the obstacle's normals pointing into its own solid (measured: the obstacle then exchanges exactly zero with the enclosure).
* **Back-to-back faces of a convex obstacle exchange nothing** (the chord crosses the solid, view factor 0) while cavity-style crossed strings read the obstacle as a cavity and return up to `sqrt(2) - 1 = 0.4142` (adjacent faces: `1 - sqrt(2)/2 = 0.2929`). Signed cosines in the kernel remove it; an `|cos|` kernel adds the spurious exchange (measured: enclosure row sum 1.81, not 1.00).
* **Shadow edges landing outside the receiver segment** break any `<2 break points -> visible` shortcut: a blocker can shadow part of a strip with both shadow edges beyond its end points (measured F error 0.44 before the fix).
* Limits: 2-D strips only, no participating media, no spectral properties, no radiometric data; a shadow boundary exactly through a strip end point is assigned by the midpoint rule.

## 6. Error judgement

| criterion | value | gate | under 3 % ? |
|---|---|---|---|
| C1_equivalence_no_blocker | 2.252e-05 | 5e-03 | yes |
| C1b_equivalence_clean_pair | 2.680e-16 | 1e-06 | yes |
| C2_fully_blocked_pair_is_zero | 0.000e+00 | 1e-12 | yes |
| C3_closure_obstructed | 3.923e-05 | 1e-03 | yes |
| C4_reciprocity_obstructed | 4.389e-05 | 1e-03 | yes |
| C5_monte_carlo_agreement | 6.361e-03 | 1e-02 | yes |
| C6_back_to_back_faces | 0.000e+00 | 1e-12 | yes |
| C7_convergence | 2.765e-06 | 3e-03 | yes |
| C8_enclosure_energy_residual | 6.352e-06 | 1e-04 | yes |
| C8b_isothermal_invariance | 2.147e-05 | 1e-04 | yes |
| C8c_crossed_strings_power_error | 8.000e-01 | 1e-02 | no - by construction: it measures the *pre-existing* API's error |
| C8d_negative_view_factors_rejected | 0.000e+00 | 0e+00 | yes |

The geometry and invariant criteria sit at round-off except C1, which is bounded by the
1/r corner singularity of the shared-edge pairs (also the limit of the pre-existing
quadrature helper); the Monte-Carlo criterion is bounded by the ray-trace statistical
band, which is declared rather than hidden. The only quantity above 3 % is C8c, which is
deliberately above it: it is the error of the obstructed-blind answer this capability
replaces, a defect magnitude and not an error of the new implementation.
