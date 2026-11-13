## CASE-FVM-KOVASZNAY-ORDER 32/64/128 order study

Verdict: PASS (full-3D velocity slope 1.9370 vs threshold 1.8); in-plane velocity slope 1.9370

| n | h | cells | vel L2 (full) | vel L2 (u,v) | w RMS share | p L2 (mean-free) | converged |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 32 | 0.03125 | 1024 | 3.435615e-03 | 3.435615e-03 | 6.779241e-18 | 2.885142e-02 | yes |
| 64 | 0.01562 | 4096 | 8.972614e-04 | 8.972614e-04 | 1.119141e-18 | 7.958339e-03 | yes |

| quantity | slope | R^2 | verdict (>= 1.8) |
|---|---:|---:|:--|
| velocity L2 (full 3D u,v,w) | 1.9370 | 1.000000 | PASS |
| velocity L2 (in-plane u,v) | 1.9370 | 1.000000 | PASS |
| spanwise w RMS / |u_exact| | 2.5987 | 1.000000 | PASS |
| pressure L2 (fixed gauge) | 0.9857 | 1.000000 | FAIL |
| pressure L2 (mean-free gauge) | 1.8581 | 1.000000 | PASS |
