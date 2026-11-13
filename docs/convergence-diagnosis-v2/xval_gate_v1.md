# XVAL: TensorFVM convergence-diagnosis v2 vs OpenFOAM-10 rules

run_utc: 2026-11-13T06:45:06Z  git: 70fa3abef270b0a438c90890a0d5bf59f63afef1

## Parameters (dimensionless)

| case | operator | solver | n | size | Pe | Pe_cell | tol | relTol | alpha*2/lambda_max | kappa(A) est |
|---|---|---|---|---|---|---|---|---|---|---|
| poisson-n32-tol1e-8 | poisson | gmres | 32 | 1024 | 0 | 0 | 1e-08 | 0 | 0 | 4.405e+02 |
| poisson-n64-tol1e-8 | poisson | gmres | 64 | 4096 | 0 | 0 | 1e-08 | 0 | 0 | 1.707e+03 |
| poisson-n64-tol1e-12 | poisson | gmres | 64 | 4096 | 0 | 0 | 1e-12 | 0 | 0 | 1.707e+03 |
| poisson-n32-reltol1e-2 | poisson | gmres | 32 | 1024 | 0 | 0 | 1e-12 | 0.01 | 0 | 4.405e+02 |
| cd-pe10-gmres | cd | gmres | 64 | 4096 | 10 | 0.1587 | 1e-10 | 0 | 0 | 2.006e+03 |
| cd-pe1000-gmres | cd | gmres | 64 | 4096 | 1000 | 15.87 | 1e-10 | 0 | 0 | 1.024e+02 |
| cd-pe10-cg-misuse | cd | cg | 32 | 1024 | 10 | 0.3226 | 1e-10 | 0 | 0 | 5.158e+02 |
| poisson-n32-richardson | poisson | richardson | 32 | 1024 | 0 | 0 | 1e-08 | 0 | 1.2 | 4.405e+02 |

## Reference vs candidate

| case | ref nIter | ref converged | v2 verdict | v2 k_ready | rate ref (o.o.s.) | rate v2 | rate err | forecast | fc err | rate pass | fc pass | verdict agree |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| poisson-n32-tol1e-8 | 188 | True | converged | 6 | 0.9108 | 0.8177 | 10.23% | 91 | 51.60% | PASS | FAIL | PASS |
| poisson-n64-tol1e-8 | 541 | True | converged | 6 | 0.9688 | 0.8263 | 14.70% | 95 | 82.44% | PASS | FAIL | PASS |
| poisson-n64-tol1e-12 | 849 | True | converged | 6 | 0.9694 | 0.8263 | 14.76% | 144 | 83.04% | PASS | FAIL | PASS |
| poisson-n32-reltol1e-2 | 48 | True | converged | 6 | 0.9278 | 0.8177 | 11.87% | 22 | 54.17% | PASS | FAIL | PASS |
| cd-pe10-gmres | 584 | True | converged | 6 | 0.963 | 0.8477 | 11.98% | 138 | 76.37% | PASS | FAIL | PASS |
| cd-pe1000-gmres | 498 | True | converged | 5 | 0.9544 | 0.9898 | 3.71% | 2257 | 353.21% | PASS | FAIL | PASS |
| cd-pe10-cg-misuse | 2000 | False | diverging | 104 | 1.002 | 0.9999 | 0.18% | 438291 | 21814.55% | PASS | FAIL | PASS |
| poisson-n32-richardson | 300 | False | diverging | 6 | 1.37 | 0.879 | 35.84% | 141 | 53.00% | FAIL | FAIL | PASS |

## Summary

* cases: 8
* rate error <= 15 %: 7/8 (FAIL)
* forecast error <= 15 %: 0/8 (FAIL)
* verdict agreement: 8/8 (PASS)
* early flag: poisson-n64-tol1e-8@37, poisson-n64-tol1e-12@37, cd-pe10-gmres@25, cd-pe1000-gmres@5, cd-pe10-cg-misuse@6, poisson-n32-richardson@13
* **benchmark_pass**: False
