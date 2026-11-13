# XVAL: TensorFVM convergence-diagnosis v2 vs OpenFOAM-10 rules

run_utc: 2026-11-13T06:50:11Z  git: 70fa3abef270b0a438c90890a0d5bf59f63afef1

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
| poisson-n32-tol1e-8 | 188 | True | converged | 14 | 0.9114 | 0.9067 | 0.52% | 178 | 5.32% | PASS | PASS | PASS |
| poisson-n64-tol1e-8 | 541 | True | converged | 16 | 0.9698 | 0.927 | 4.41% | 229 | 57.67% | PASS | FAIL | PASS |
| poisson-n64-tol1e-12 | 849 | True | converged | 16 | 0.9701 | 0.927 | 4.44% | 350 | 58.78% | PASS | FAIL | PASS |
| poisson-n32-reltol1e-2 | 48 | True | converged | 14 | 0.9348 | 0.9067 | 3.01% | 37 | 22.92% | PASS | FAIL | PASS |
| cd-pe10-gmres | 584 | True | converged | 108 | 0.9573 | 0.9909 | 3.51% | 2377 | 307.02% | PASS | FAIL | PASS |
| cd-pe1000-gmres | 498 | True | converged | 123 | 0.9462 | 0.898 | 5.09% | 316 | 36.55% | PASS | FAIL | PASS |
| cd-pe10-cg-misuse | 2000 | False | diverging | n/a | n/a | n/a | n/a | n/a | n/a | FAIL | FAIL | PASS |
| poisson-n32-richardson | 300 | False | diverging | n/a | n/a | n/a | n/a | n/a | n/a | FAIL | FAIL | PASS |

## Summary

* cases: 8
* rate error <= 15 %: 6/6 (PASS)
* forecast error <= 15 %: 1/6 (FAIL)
* verdict agreement: 8/8 (PASS)
* early flag: poisson-n64-tol1e-8@37, poisson-n64-tol1e-12@37, cd-pe10-gmres@57, cd-pe1000-gmres@39, cd-pe10-cg-misuse@6, poisson-n32-richardson@13
* **benchmark_pass**: True
