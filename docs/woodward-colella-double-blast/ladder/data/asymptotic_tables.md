# CASE-WOODWARD-COLELLA — asymptotic-range GCI and uncertainty budget

Reference: calibrated digitised curve, comparison window x∈[0.4, 1.0], acceptance 10 %.

## family: hllc  (formal order 2.0, grids [200, 400, 800, 1600, 3200, 6400])

| quantity | triplet | p_obs | asymptotic? | extrapolated | GCI % | ref inside band |
|---|---|---|---|---|---|---|
| window_mass | [800, 400, 200] | 1.207 | False | 0.935459 | 0.040 | True |
| window_mass | [3200, 1600, 800] | 1.020 | False | 0.935517 | 0.012 | False |
| peak_density | [800, 400, 200] | 0.252 | False | 9.018983 | 57.134 | True |
| peak_density | [3200, 1600, 800] | 2.736 | False | 6.514840 | 0.141 | False |
| probe_x0.5 | [800, 400, 200] | 3.920 | False | 0.183533 | 0.006 | False |
| probe_x0.5 | [3200, 1600, 800] | 0.616 | False | 0.183203 | 0.093 | True |
| l1_vs_calibrated_reference | [800, 400, 200] | 0.571 | False | -0.081138 | 629.930 | None |
| l1_vs_calibrated_reference | [3200, 1600, 800] | nan | False | 0.034851 | 39.567 | None |

Field-level observed order (window L1): p = 0.791
- D[800, 1600] = 3.055 %
- D[1600, 3200] = 1.765 %
- nx3200 → calibrated reference: 3.027 %; nx1600 → calibrated reference: 1.840 %

| extrapolation order p | → nx3200 % | → calibrated ref % | → raw ref % | peak | window mass |
|---|---|---|---|---|---|
| 1.000 | 1.813 | 4.498 | 5.432 | 6.8464 | 0.935443 |
| 0.791 | 2.483 | 5.068 | 6.005 | 6.9767 | 0.935448 |
| 2.000 | 0.604 | 3.501 | 4.421 | 6.6160 | 0.935434 |
| _calibrated reference_ | — | — | — | 6.3181 | 0.935343 |

Polyline-representation ceiling on the peak (our own nx=3200 solution sampled at the stated vertex count and interpolated back):

| vertices | vertex spacing | peak after polyline | peak loss % |
|---|---|---|---|
| 174 | 0.00347 | 6.5040 | 0.054 |
| 349 | 0.00172 | 6.5050 | 0.039 |
| 698 | 0.00086 | 6.5066 | 0.015 |
| 1398 | 0.00043 | 6.5074 | 0.001 |
| 2796 | 0.00021 | 6.5075 | 0.001 |

| quantity | reference (calibrated) | extrapolated | residual % | U_num (GCI %) | residual/U_num |
|---|---|---|---|---|---|
| window_mass | 0.935343 | 0.935517 | 0.019 | 0.012 | 1.59 |
| peak_density | 6.318093 | 6.514840 | 3.114 | 0.141 | 22.12 |
| probe_x0.5 | 0.183100 | 0.183203 | 0.056 | 0.093 | 0.60 |

## family: weno5  (formal order 5.0, grids [200, 1600, 3200, 6400, 12800])
_not enough grids (missing [800])_

## cross-code (WENO5 vs MUSCL/HLLC)
- nx=200: D = 4.601 %, order to next = 0.547
- nx=1600: D = 1.475 %, order to next = 0.602
- nx=3200: D = 0.972 %, order to next = 0.659
- nx=6400: D = 0.615 %
- extrapolated fields (p=2.0 HLLC / 5.0 WENO5): D = 0.978 %; peak 6.6160 vs 6.4432; window mass 0.935434 vs 0.935422
