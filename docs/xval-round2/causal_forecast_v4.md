# Causal stopping-iteration forecast v4 — calibrated intervals

* cases: 8 (6 decaying)
* acceptance (ledger): point-forecast error <= 15%
* protocol: strict causal prefix replay, error taken at one prefix per lead; coverage measured at that prefix and over all prefixes

## Point forecast by lead (pass = error <= 15 %)

| lead | v1 | v3 | v4 |
|---|---|---|---|
| 50% | 2/6 (worst 38.8%) | 2/6 (worst 34.9%) | 2/6 (worst 38.8%) |
| 25% | 4/6 (worst 70.8%) | 2/6 (worst 70.8%) | 4/6 (worst 70.8%) |
| 10% | 5/6 (worst 20.8%) | 6/6 (worst 6.2%) | 6/6 (worst 10.1%) |
| first publish | 1/6 (worst 307.0%) | 0/6 (worst 103.6%) | 0/6 (worst 87.5%) |

## Interval coverage and width

| scope | arm | nominal | measured coverage | median width / stop |
|---|---|---|---|---|
| lead 50% | v3 | 68% | 0.0% (n=5) | 3.2% |
| lead 50% | v3 | 90% | 0.0% (n=5) | 5.6% |
| lead 50% | v4 | 68% | 80.0% (n=5) | 272.3% |
| lead 50% | v4 | 90% | 100.0% (n=5) | 338.6% |
| lead 25% | v3 | 68% | 16.7% (n=6) | 3.8% |
| lead 25% | v3 | 90% | 16.7% (n=6) | 6.6% |
| lead 25% | v4 | 68% | 50.0% (n=6) | 37.4% |
| lead 25% | v4 | 90% | 83.3% (n=6) | 46.1% |
| lead 10% | v3 | 68% | 16.7% (n=6) | 1.8% |
| lead 10% | v3 | 90% | 16.7% (n=6) | 2.4% |
| lead 10% | v4 | 68% | 66.7% (n=6) | 11.8% |
| lead 10% | v4 | 90% | 83.3% (n=6) | 19.8% |
| all prefixes | v3 | 68% | 11.8% (n=2441) | 4.5% |
| all prefixes | v3 | 90% | 19.4% (n=2381) | 7.8% |
| all prefixes | v4 | 68% | 71.3% (n=2435) | 85.3% |
| all prefixes | v4 | 90% | 87.9% (n=2435) | 192.7% |

## Per case, lead 25 % and lead 10 %

| case | stop | v1@25 | v3@25 | v4@25 | v1@10 | v3@10 | v4@10 | v4 iv68@25 | v3 iv68@25 |
|---|---|---|---|---|---|---|---|---|---|
| cd-pe10-gmres | 584 | 3.8% | 18.8% | 3.8% | 3.3% | 3.3% | 3.3% | [527, 812]* | [472, 475] |
| cd-pe1000-gmres | 498 | n/p | 31.1% | 31.1% | 4.4% | 4.4% | 4.4% | [503, 1193] | [627, 685] |
| poisson-n32-reltol1e-2 | 48 | 70.8% | 70.8% | 70.8% | 20.8% | 6.2% | 6.2% | [53, 2000] | [67, 130] |
| poisson-n32-tol1e-8 | 188 | 9.6% | 9.6% | 9.6% | 10.1% | 1.6% | 10.1% | [162, 185] | [169, 171] |
| poisson-n64-tol1e-12 | 849 | 2.0% | 2.0% | 2.0% | 2.7% | 2.7% | 2.7% | [766, 987]* | [808, 864]* |
| poisson-n64-tol1e-8 | 541 | 0.7% | 15.3% | 0.7% | 0.7% | 0.7% | 0.7% | [498, 614]* | [456, 460] |

`*` marks an interval that contains the true stopping iteration.
