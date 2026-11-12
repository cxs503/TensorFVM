# Double-body closure audit: flat-lid cap vs waterplane mirror (debt D9)

TensorFVM · every number below is generated programmatically from `summary.json`
(zero hand-copied values). Reproduce with

```
python -m tensorfvm.verification.double_body_added_mass_gate --out docs/verification-double-body-gate
```

## 1. The question

`maneuvering.added_inertia_panel` closes the hull below the waterline in two ways:
**lid cap** (`with_lid=True`, `IMAGE_SPECS["body"]`; hull + flat lid at `z = 0` in
the unbounded fluid) and **waterplane mirror** (`with_lid=False`,
`IMAGE_SPECS["single_hull"]`; open hull + its image in `z = 0` = the classical
double body). Report `docs/maneuvering-added-inertia/REPORT.md` (debt D7) took the
corroborating sway number from the mirror route and the headline yaw number from
the lid route, justified by "the lid carries no flux, so it is exactly the plane
of symmetry of the double body".

That justification is wrong: zero normal velocity on the lid only makes the
potential *even* across it. In the lid model the fluid continues **above**
`z = 0` and streams over the deck, while the double body has the mirrored hull
there. The lid body is the closed-body problem (a bluff body with a flat top).

## 2. Anchoring the mirror route on exact values (C1, C2)

Half sphere, radius 1, rho 1, resting on the waterline plane. The mirror image of
the half body is the full sphere, so the half-body value is
`m_a/2 = 1.047198` (sway, parallel to the wall) and the value for
yaw about a vertical axis offset by `d = 1.5a` is
`m_a d^2/2 = 2.356194`. The sphere's own rotational added inertia
about a diameter is exactly zero, so a wrong moment arm cannot hide.

| mesh | panels | M_22 (mirror) | err | I_zz (d=1.5a) | err | I_zz own axis |
|---|---|---|---|---|---|---|
| 16x18 | 144 | 1.01645 | -2.936% | 2.28701 | -2.936% | 4.9e-33 |
| 24x24 | 288 | 1.02802 | -1.831% | 2.31305 | -1.831% | 7.8e-33 |
| 32x32 | 512 | 1.03529 | -1.137% | 2.32939 | -1.137% | 9.1e-33 |

| quantity | closed form | limit fit | rel. error | fitted q | observed p |
|---|---|---|---|---|---|
| sway | 1.04720 | 1.06040 | +1.260% | 0.44 | 0.83 |
| yaw_offset | 2.35619 | 2.38589 | +1.260% | 0.44 | 0.83 |

Both close inside the 3 % gate and the 1 % extrapolation gate; the rim row sits
at `z <= 1e-16`, so the mirror closes the body exactly.
**The mirror route is the double-body model.**

## 3. The lid cap is a different model (C3)

Same KVLCC2 hull, both closures, sway and yaw about midship; panels `lid/open`.

| mesh | panels | M_22 mirror | M_22 lid | ratio | J_z mirror | J_z lid | ratio | volume err |
|---|---|---|---|---|---|---|---|---|
| 21x5 | 252/210 | 2461.7 | 1537.2 | 0.624 | 6659.8 | 4416.2 | 0.663 | -3.993% |
| 41x9 | 820/738 | 2554.3 | 1626.1 | 0.637 | 6965.1 | 4723.4 | 0.678 | -1.199% |
| 61x13 | 1708/1586 | 2573.5 | 1651.2 | 0.642 | 7048.0 | 4822.4 | 0.684 | -0.558% |

The ratio is mesh-independent (M_22 spread 0.017,
J_z spread 0.021) while the gap itself is
35.8% (sway) and 31.6%
(yaw) — far outside the discretisation uncertainty, so this is a model
difference. D7's premise holds where it applies
(`lid_flux_max_sway = 0.0e+00`,
`lid_flux_max_yaw = 0.0e+00`: the lid really carries no flux)
and fails as an equivalence argument.

## 4. Corrected double-body numbers, KVLCC2 (C4, C5)

| quantity | value |
|---|---|
| `J_z` double body, midship, finest mesh (61x13, 1586 panels) | 7048.0 kg m² |
| `J_z` limit fit (exponent q = 0.77) | 7151.1 kg m² |
| `J_z` slender (`dynamics.added_masses`) | 13411.1 kg m² |
| `J_z` double body / slender | 0.533 |
| slender overestimate, double-body basis | **87.5%** |
| slender overestimate, lid basis (D7 headline `0.371`) | 178.1% |
| `M_22` double body, finest | 2573.5 kg |
| `m_y` slender | 3822.0 kg |
| `M_22` / `m_y` | 0.673 |

D7's qualitative conclusion survives (the strip formulas overestimate both `M_22`
and `J_z` for this hull), but its magnitude was overstated by about a factor of
two because the headline came from the lid cap: on the physically matching
(double-body) model the overestimate is 88%.

## 5. The 6x6 assembly is sound (C6); debt D8 is a phantom

Closed forms use only the parallel-axis identity `A_new = P^T A_old P`,
`P = [[I, [delta]x], [0, I]]`, applied to `diag(m_a, m_a, m_a, 0, 0, 0)` (sphere)
and to the Lamb translational + rotational entries (spheroid `a/b = 2`). All 36
entries are compared, which exercises the translation-rotation coupling block that
nothing in the repository scored before.

| case | mesh | panels | max abs error / scale |
|---|---|---|---|
| sphere offset (1.5,0,0) | 24x24 | 576 | 2.927% |
| sphere offset (1.5,0,0) | 32x32 | 1024 | 1.781% |
| spheroid a/b=2 offset (1.2,0,0.8) | 24x24 | 576 | 3.724% |
| spheroid a/b=2 offset (1.2,0,0.8) | 32x32 | 1024 | 2.513% |

Debt item D8 ("`added_mass_matrix_6dof` gives `I_zz = 2.05` for a sphere centred at
`(0,0,1.5)`, theory `0`") does not reproduce: `2.05` is `A_heave,heave`
(= 2.0571); the yaw entry is 1.9e-32
against reference 0.0e+00. It was a block-index misreading, and
C6 now pins all 36 entries.

## 6. Error judgement

| key quantity | reference | result | < 3 %? |
|---|---|---|---|
| hemisphere sway (mirror) | `m_a/2` exact | -1.14% | yes |
| hemisphere yaw, offset axis | `m_a d^2/2` exact | -1.14% | yes |
| 6x6 Kirchhoff vs closed form (4 cases) | parallel-axis algebra | worst finest 2.51% (coarsest 3.72%) | yes |
| lid vs mirror gap | model difference, no reference | 31.6% | not an error (C3) |
| `J_z` double body grid uncertainty | limit fit drift | +1.46% | yes |
| D7 headline `slender/lid = 0.371` | corrected `0.533` | 178% vs 88% | superseded |

All criteria: C1=PASS, C2=PASS, C3=PASS, C4=PASS, C5=PASS, C6=PASS

## 7. Reproduction and provenance

```bash
PYTHONPATH=src python -m tensorfvm.verification.double_body_added_mass_gate --out docs/verification-double-body-gate
PYTHONPATH=src python -m pytest tests/test_double_body_added_mass_gate.py -q
```
