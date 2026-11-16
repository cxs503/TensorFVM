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
| spanwise w RMS / |u_exact| | n/a (machine zero) | n/a | n/a as an order; absolute check PASS (max 6.78e-18 <= 1e-12) |
| pressure L2 (fixed gauge) | 0.9857 | 1.000000 | FAIL |
| pressure L2 (mean-free gauge) | 1.8581 | 1.000000 | PASS |

| quantity | pair orders | spread | band | both >= thr | consistent |
|---|---|---:|---:|:--|:--|
| velocity L2 (full 3D u,v,w) | 1.9370 | n/a | 0.20 | yes | n/a |
| velocity L2 (in-plane u,v) | 1.9370 | n/a | 0.20 | yes | n/a |
| spanwise w RMS / |u_exact| | n/a (machine zero) | n/a | 0.20 | n/a | n/a |
| pressure L2 (fixed gauge) | 0.9857 | n/a | 0.20 | NO | n/a |
| pressure L2 (mean-free gauge) | 1.8581 | n/a | 0.20 | yes | n/a |

| quantity | three-level order f(h)=L+C h^p | stderr | convergence | extrapolated L | GCI |
|---|---:|---:|:--|---:|---:|
| velocity L2 (full 3D u,v,w) | n/a (fewer than 3 usable levels) | | | | |
| velocity L2 (in-plane u,v) | n/a (fewer than 3 usable levels) | | | | |
| pressure L2 (mean-free gauge) | n/a (fewer than 3 usable levels) | | | | |

Round-off series are *not* graded on an order (their exact value is zero, so any fitted exponent is noise); they are graded on absolute size against the round-off band:

- spanwise w RMS / |u_exact|: n/a (machine zero: every level <= 1e-12 x |u_exact|; a fitted exponent here is round-off noise, not a convergence order) (max share 6.779e-18 <= limit 1e-12)
