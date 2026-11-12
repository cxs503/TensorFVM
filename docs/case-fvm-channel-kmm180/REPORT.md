# CASE-FVM-CHANNEL-KMM180: turbulent channel flow vs KMM1987 DNS

Reference: Kim, Moin & Moser 1987 J. Fluid Mech. 177:133, DOI:10.1017/S0022112087000892

Numerical reference: Lee & Moser 2015 J. Fluid Mech. 774:395 (Re_tau = 182.088, same flow).

## Criterion

* RMS relative deviation of u+(y+) over y+>30 (to centreline)
* tolerance < 5 %
* graded closure: `sst`

## Reference vs computed

| quantity | ref (DNS) | got (FVM) | rel. error |
|---|---|---|---|
| RMS rel. deviation u+, y+>30 | 0.000e+00 | 0.0363 | 3.63% |
| C_f | 0.008123 | 0.008667 | 6.69% |
| U_bulk+ | 15.6910 | 15.1909 | 3.19% |
| u+ centreline | 18.2683 | 17.8497 | 2.29% |
| apparent kappa | 0.3780 | 0.3108 | - |
| apparent B | 4.9337 | 2.0331 | - |

Verdict of the graded closure: **PASS** (resolution n = 512).

## Station table (y+ > 30)

| y+ | u+ ref | u+ computed | abs. error | rel. error |
|---|---|---|---|---|
| 30 | 13.8551 | 12.9282 | 0.9268 | -6.69% |
| 40 | 14.7182 | 13.9204 | 0.7978 | -5.42% |
| 50 | 15.3033 | 14.6405 | 0.6628 | -4.33% |
| 75 | 16.3185 | 15.8807 | 0.4379 | -2.68% |
| 100 | 17.0674 | 16.7236 | 0.3438 | -2.01% |
| 125 | 17.6495 | 17.3167 | 0.3328 | -1.89% |
| 150 | 18.0610 | 17.6909 | 0.3701 | -2.05% |
| 175 | 18.2583 | 17.8453 | 0.4130 | -2.26% |

## Negative controls

| closure | n | log-region RMS | C_f error | U_bulk+ error | verdict |
|---|---|---|---|---|---|
| komega | 64 | 10.62% | 23.89% | 10.16% | FAIL |
| komega | 128 | 10.61% | 23.72% | 10.10% | FAIL |
| komega | 256 | 10.59% | 23.66% | 10.08% | FAIL |
| kepsilon | 64 | 6.88% | 15.00% | 6.75% | FAIL |
| kepsilon | 128 | 6.88% | 14.99% | 6.75% | FAIL |
| kepsilon | 256 | 6.88% | 14.99% | 6.74% | FAIL |

All negative controls rejected: **True**

## Grid ladder (graded closure)

| n | log-region RMS | C_f error | U_bulk+ error | verdict |
|---|---|---|---|---|
| 64 | 2.25% | 3.56% | 1.73% | PASS |
| 128 | 3.06% | 5.36% | 2.58% | PASS |
| 256 | 3.44% | 6.25% | 2.99% | PASS |
| 512 | 3.63% | 6.69% | 3.19% | PASS |

## Discretisation ladder (graded closure)

| n -> 2n | max abs. du+ at stations | RMS abs. du+ | graded RMS coarse -> fine |
|---|---|---|---|
| 64 -> 128 | 1.445e-01 | 1.392e-01 | 2.25% -> 3.06% |
| 128 -> 256 | 7.053e-02 | 6.823e-02 | 3.06% -> 3.44% |
| 256 -> 512 | 3.419e-02 | 3.312e-02 | 3.44% -> 3.63% |

## Graded-window sensitivity

| window | y+>30_to_0.2Re_tau | y+>30_to_0.5Re_tau | y+>30_to_centreline |
|---|---|---|---|
| kepsilon_n128 | 6.33 % | 6.92 % | 6.88 % |
| kepsilon_n256 | 6.35 % | 6.92 % | 6.88 % |
| kepsilon_n64 | 6.36 % | 6.93 % | 6.88 % |
| komega_n128 | 13.48 % | 11.52 % | 10.61 % |
| komega_n256 | 13.43 % | 11.50 % | 10.59 % |
| komega_n64 | 13.46 % | 11.58 % | 10.62 % |
| sst_n128 | 5.66 % | 3.77 % | 3.06 % |
| sst_n256 | 6.08 % | 4.19 % | 3.44 % |
| sst_n512 | 6.30 % | 4.39 % | 3.63 % |
| sst_n64 | 4.61 % | 2.91 % | 2.25 % |

## Figures

![fig_velocity_profiles.png](fig_velocity_profiles.png)

![fig_log_region_error.png](fig_log_region_error.png)

## Notes and limitations

* The solve is a 1-D equilibrium channel (finite-volume in the wall-normal direction): the momentum equation reduces to `(nu + nu_t) dU/dy = 1 - y`, so convection, pressure-velocity coupling and geometry are not exercised.  Agreement here is evidence that the closure coupling and the near-wall discretisation are consistent, not that the 3-D solvers are accurate.
* The graded closure SST-2003 is calibrated to the log law, so a PASS is expected a priori for a wall-resolved mesh; the negative controls are what make the comparison non-trivial.
* The reference data set is Lee & Moser (2015) at Re_tau = 182.088 because it is the digitised, provenance-tracked channel data set in this repository; KMM1987 is the case-pool citation for the same flow at Re_tau = 180.
* At Re_tau = 180 the log layer is only a few wall units thick, so the apparent kappa of the reference fit falls below 0.41; it is reported as a diagnostic, not as a criterion.
* **Window dependence, disclosed up front.** The verdict is taken over the whole y+ > 30 range, as the case-pool row states (no upper bound).  Restricted to the strict log window 30 < y+ < 0.2 Re_tau the graded closure gives 4.61-6.30 %, i.e. above the 5 % bar: the case therefore certifies the resolved near-wall/log layer of the FVM channel solve, not a log-law constant.
