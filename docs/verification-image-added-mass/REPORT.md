# Image (free-surface / double-body) added mass vs analytic anchors — debt D2

TensorFVM · every number below is generated programmatically from `summary.json`
(no hand-copied values). Reproduce with

```
python -m tensorfvm.verification.image_added_mass --out docs/verification-image-added-mass
```

## 1. What is verified, and what the images mean

`panel_potential.IMAGE_SPECS` builds the Rankine influence matrix as a sum of the body
kernel and its reflections: `single_hull` = body + waterline (`z`) mirror, i.e. a body
cut by the free surface; `twin_hull` = body + centreplane (`y`) + waterline (`z`)
mirrors, i.e. the quarter body of a double-body model. Both are already used in
production paths (`twin_hull_form_factor`, `maneuvering.added_inertia_panel`, the
KVLCC2 double body) but their added mass had never been scored against a reference
value — that is debt D2.

All image entries are **same-sign**, so the potential is *even* across the mirror plane
and the imposed condition is `dphi/dn = 0` (calm-water / zero-frequency rigid wall),
not `phi = 0` (infinite-frequency free surface). C5 measures both facts.

The reference values follow from the energy identity `KE_half = KE_double / 2` and the
reflected mirror velocity `V' = S V`:

| direction w.r.t. cut plane | double-body motion | half-body added mass |
|---|---|---|
| parallel (x axial, y transverse) | rigid translation of the full body | `m_full / 2` (Lamb closed form) |
| normal (z) | straining: the halves move apart, `w = U|cos theta|` | `m_stretch / 2`, `m_stretch = 4 pi rho a^3 sum c_n^2/((2n+1)(n+1))` |
| quarter body, axial | rigid axial translation of the full body | `m_full / 4` |

## 2. Results — half sphere (radius 1, rho 1; `pi/3 = 1.047198` tangential, `m_stretch/2 = 1.740340` normal)

| mesh | panels | M_xx | err | M_yy | err | M_zz | err | max off-diag |
|---|---|---|---|---|---|---|---|---|
| 8x6 | 48 | 0.952994 | 8.996% | 0.954683 | 8.834% | 1.661381 | 4.537% | 3.9e-17 |
| 12x9 | 108 | 1.013067 | 3.259% | 1.009350 | 3.614% | 1.720871 | 1.119% | 1.6e-16 |
| 16x12 | 192 | 1.032871 | 1.368% | 1.028016 | 1.832% | 1.738479 | 0.107% | 1.6e-16 |
| 24x18 | 432 | 1.045289 | 0.182% | 1.040618 | 0.628% | 1.747667 | 0.421% | 1.8e-17 |
| 32x24 | 768 | 1.048649 | 0.139% | 1.044603 | 0.248% | 1.749057 | 0.501% | 3.4e-16 |

Richardson (three finest levels):

| entry | anchor | Richardson | rel. error | observed order p |
|---|---|---|---|---|
| M_xx | 1.047198 | 1.049895 | 0.258% | 1.89 |
| M_yy | 1.047198 | 1.046446 | 0.072% | 1.66 |
| M_zz | 1.740340 | 1.749305 | 0.515% | 2.72 |

Anchor anisotropy: `M_zz / M_xx = 1.6619` — the normal
direction is **not** obtainable by halving the rigid added mass (that shortcut is
40 % low); only the straining-double-body
anchor reproduces it.

## 3. Results — half spheroid (anisotropy; anchors `k_axial rho V/2`, `k_transv rho V/2`)

| a/b | mesh | panels | M_xx | err | M_yy | err | M_zz (normal) |
|---|---|---|---|---|---|---|---|
| 2 | 24x18 | 432 | 0.882952 | 0.369% | 2.919149 | 1.039% | 5.253570 (WITHHELD) |
| 2 | 32x24 | 768 | 0.884554 | 0.551% | 2.933584 | 0.549% | 5.262740 (WITHHELD) |
| 4 | 24x18 | 432 | 0.686132 | 0.421% | 7.164653 | 0.528% | 14.573828 (WITHHELD) |
| 4 | 32x24 | 768 | 0.687317 | 0.595% | 7.190727 | 0.166% | 14.608210 (WITHHELD) |

## 4. Results — quarter spheroid (`twin_hull`; anchor `k_axial rho V/4`)

| a/b | mesh | panels | M_xx | anchor | err |
|---|---|---|---|---|---|
| 1 | 24x18 | 432 | 0.531014 | 0.523599 | 1.416% |
| 1 | 32x24 | 768 | 0.530188 | 0.523599 | 1.258% |
| 2 | 24x18 | 432 | 0.446673 | 0.439854 | 1.550% |
| 2 | 32x24 | 768 | 0.445839 | 0.439854 | 1.360% |
| 4 | 24x18 | 432 | 0.345476 | 0.341626 | 1.127% |
| 4 | 32x24 | 768 | 0.345212 | 0.341626 | 1.050% |

## 5. Image convention (C5)

| quantity | value |
|---|---|
| probe points on the mirror plane | 81 |
| `max |u_z| / U` (rigid-wall residual) | 0.000e+00 |
| `max |phi| / (U a)` (wall potential) | 5.112e-01 |

The wall-normal velocity is at round-off (the same-sign image pair cancels exactly)
while the potential on the plane is O(1): the implemented mirror is the **rigid-wall**
free-surface model. D2's original "`phi = 0` residual < 1e-8" sentence therefore
describes a model that is **not implemented** — recorded as NOT_SUPPORTED (§7)
instead of being scored.

Getting this residual to round-off required a real defect fix: `influence_matrix`
dropped the "self-panel" matrix entry on the diagonal unconditionally, which is the
flat-panel principal value *only* when the field points are the panel centroids. For
any other field-point set (wall probes, potential evaluation) it deleted one genuine
panel contribution per field point; the residual was 7.8e-3 before the fix and 0.0
after.

## 6. Discrete double-body identity (C6)

| quantity | value |
|---|---|
| half body `half 24x18 vs full 48x18` | 432 vs 864 panels |
| `M_xx` half / (full/2) | 1.045288978 / 1.045288978 (rel 2.02e-14) |
| `M_yy` half / (full/2) | 1.040618262 / 1.040618262 (rel 4.27e-16) |

This identity is exact in the continuum, so holding to 2.0e-14 shows the image
plumbing reproduces the doubled system exactly; being insensitive to mesh error it is a
regression guard for the image set itself rather than an accuracy statement.

## 7. Criteria table

| id | statement | value | threshold | verdict |
|---|---|---|---|---|
| C1 | half sphere tangential M_xx, M_yy vs pi rho a^3/3 < 3% (Richardson < 1%, p >= 1) | {"err_xx_finest": 0.0013857277069517684, "err_yy_finest": 0.002477262414861591, "richardson_rel_err_max": 0.005151136042350234, "observed_order_min": 1.6610405134535107} | {"panel": 0.03, "richardson": 0.01, "order": 1.0} | PASS |
| C2 | half sphere normal M_zz vs m_stretch/2 < 3% | {"err_zz_finest": 0.005008676616725898, "M_zz": 1.7490570984228662, "anchor": 1.7403402966737704} | {"panel": 0.03} | PASS |
| C3 | half spheroid a/b=2,4: M_xx vs k_axial rho V/2, M_yy vs k_transv rho V/2, each < 3% | {"worst_err_finest": 0.005949056801321039} | {"panel": 0.03} | PASS |
| C4 | quarter spheroid (twin_hull) M_xx vs k_axial rho V/4 < 3% | {"worst_err_finest": 0.013604529444359914} | {"panel": 0.03} | PASS |
| C5 | rigid-wall residual < 1e-8 and wall potential > 1e-3 (mirror is Neumann, not phi=0) | {"max_wall_normal_velocity": 0.0, "max_wall_potential_over_ua": 0.5111560176570542, "n_probe": 81} | {"wall_residual": 1e-08, "wall_potential_min": 0.001} | PASS |
| C6 | discrete identity m(half+image) = m(full)/2 for tangential directions, < 1e-9 | {"rel_xx": 2.0180292635330275e-14, "rel_yy": 4.267551572424406e-16} | {"identity": 1e-09} | PASS |

## 8. Honest scope

* **Implemented → verified**: same-sign image sets (`single_hull`, `twin_hull`) for
  translational added mass, all three directions, sphere and spheroid bodies.
* **NOT_SUPPORTED**: the `phi = 0` (infinite-frequency free surface) mirror. Adding it
  needs a per-specification sign multiplier in `influence_matrix` / `source_potential`
  (all image terms are currently accumulated with `+`) plus the corresponding
  odd-image anchors. Until then no seakeeping (`omega -> infinity`) added mass can be
  produced by this module family.
* **WITHHELD**: the *normal*-direction added mass of a half **spheroid** (no closed
  form here: the straining double body of a spheroid needs spheroidal harmonics). The
  panel value is reported but not scored, and no invented anchor is used.
* **Not covered**: rotational block with images; the waterline *lid* question (a deck
  closing the waterplane is a different model).
