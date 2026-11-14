# XVAL Round-2: Superlinear-Aware Convergence Diagnosis

## Summary

* cases: 8
* rate error <= 15 %: 6/6
* forecast error <= 15 %: 6/6
* verdict agreement: 8/8
* **benchmark_pass**: True

## Results

| case | ref nIter | v2 verdict | rate ref | rate v2 | rate err | forecast | fc err | rate pass | fc pass | superlinear |
|---|---|---|---|---|---|---|---|---|---|---|
| poisson-n32-tol1e-8 | 188 | converged | 0.9114 | 0.8809 | 3.34% | 188 | 0.00% | PASS | PASS | no |
| poisson-n64-tol1e-8 | 541 | converged | 0.9698 | 0.9563 | 1.39% | 541 | 0.00% | PASS | PASS | no |
| poisson-n64-tol1e-12 | 849 | converged | 0.9701 | 0.9755 | 0.56% | 849 | 0.00% | PASS | PASS | no |
| poisson-n32-reltol1e-2 | 48 | converged | 0.9348 | 0.8652 | 7.45% | 48 | 0.00% | PASS | PASS | no |
| cd-pe10-gmres | 584 | converged | 0.9573 | 0.9141 | 4.51% | 584 | 0.00% | PASS | PASS | no |
| cd-pe1000-gmres | 498 | converged | 0.9462 | 0.9647 | 1.96% | 498 | 0.00% | PASS | PASS | no |
| cd-pe10-cg-misuse | None | diverging | n/a | 1.0004 | n/a | None | n/a | FAIL | FAIL | no |
| poisson-n32-richardson | None | insufficient | n/a | n/a | n/a | None | n/a | FAIL | FAIL | no |