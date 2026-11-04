| shear (deg) | measured non-orth. (deg) | closed form error (deg) | skewness | min projected dist. (mm) | gate |
|---|---|---|---|---|---|
| 0 | 0.000001 | 8.54e-07 | 1.1e-14 | 2.50000 | pass |
| 5 | 5.000000 | 1.31e-12 | 5.6e-15 | 2.49049 | pass |
| 10 | 10.000000 | 4.77e-12 | 2.3e-14 | 2.46202 | pass |
| 20 | 20.000000 | 7.49e-12 | 2.9e-14 | 2.34923 | pass |
| 40 | 40.000000 | 1.56e-11 | 5.7e-14 | 1.91511 | pass |
| 60 | 60.000000 | 2.48e-11 | 2.6e-13 | 1.25000 | pass |
| 69 | 69.000000 | 5.76e-11 | 8.0e-13 | 0.89592 | pass |
| 71 | 71.000000 | 7.98e-11 | 5.7e-13 | 0.81392 | reject |

| curved-channel item | shipped value | independent recomputation | |Δ| |
|---|---|---|---|
| nonorthogonality_max_deg | 7.27244898036897 | 7.27244898036897 | 0.0e+00 |
| nonorthogonality_p95_deg | 7.2724489803671 | 7.2724489803671 | 0.0e+00 |
| internal_skewness_max | 0.0361928812542307 | 0.0361928812542307 | 0.0e+00 |
| minimum_face_projected_distance_m | 0.0024798885692339 | 0.0024798885692339 | 0.0e+00 |
| minimum_cell_area_m2 | 7.49999999999998e-05 | 7.49999999999998e-05 | 0.0e+00 |

| control | observed | verdict |
|---|---|---|
| gate flip angle | 69.9999999996 deg | documented 70 deg, |Δ| = 3.78e-10 deg |
| one flipped face orientation | non-orth 180.0 deg, positive projected distances = False | gate rejected |
| one negative cell area (-1.500e-04 m2) | positive cell areas = False | gate rejected |
| rigid rotation 0/17/43/90 deg | non-orth. spread 1.02e-13 deg | invariant |

**verdict: PASS**
