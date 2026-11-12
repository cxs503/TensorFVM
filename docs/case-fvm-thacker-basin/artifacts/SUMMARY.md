# CASE-FVM-THACKER-BASIN — generated summary

| criterion | measured | threshold | verdict |
|---|---|---|---|
| C1 lake at rest max\|u\| | 5.7e-15 m/s (drift 1.22e-15 m) | <= 1.0e-11 | PASS |
| C2 closed-form residual | 7.75e-08 | <= 1.0e-06 | PASS |
| C3 volume vs exact | 0.0038 % (drift 1.1e-14) | <= 0.1 % / <= 1e-09 | PASS |
| C4 centre depth L2 | 0.726 % | <= 3 % | PASS |
| C5 shoreline mean | 2.354 % | <= 3 % | PASS |
| C6 symmetry defect | 0.000 % | <= 2 % | PASS |
| C7 convergence order | 0.43 (monotone: True) | >= 1.0 | FAIL |
| C8 naive source control | max\|u\| = 5.04e+10 m/s | must fail | PASS (fails as required) |
| C9 health | h_min = 0, clipped = 0, dry flips = 0, boundary h = 0 | h_min >= 0, clipped = 0, boundary dry | PASS |

Resolution n = 160 (160 x 160 cells, d = 0.01875 m), 186 steps, 2 periods.
