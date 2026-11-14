## CASE-FVM-KOVASZNAY-ORDER pipeline selftest 4/8/16 (pre-asymptotic, expected FAIL)

Verdict: FAIL (full-3D velocity slope 1.5166 vs threshold 1.8); in-plane velocity slope 1.5401

| n | h | cells | vel L2 (full) | vel L2 (u,v) | w RMS share | p L2 (mean-free) | converged |
|---:|---:|---:|---:|---:|---:|---:|:--|
| 4 | 0.25000 | 16 | 1.029423e-01 | 1.016212e-01 | 1.643940e-02 | 4.177775e-01 | yes |
| 8 | 0.12500 | 64 | 3.710529e-02 | 3.620366e-02 | 8.130072e-03 | 2.678226e-01 | yes |
| 16 | 0.06250 | 256 | 1.257478e-02 | 1.201574e-02 | 3.707731e-03 | 9.732922e-02 | yes |

| quantity | slope | R^2 | verdict (>= 1.8) |
|---|---:|---:|:--|
| velocity L2 (full 3D u,v,w) | 1.5166 | 0.999713 | FAIL |
| velocity L2 (in-plane u,v) | 1.5401 | 0.999633 | FAIL |
| spanwise w RMS / |u_exact| | 1.0743 | 0.999014 | FAIL |
| pressure L2 (fixed gauge) | 0.5008 | 0.860420 | FAIL |
| pressure L2 (mean-free gauge) | 1.0509 | 0.951839 | FAIL |

| quantity | pair orders | spread | band | both >= thr | consistent |
|---|---|---:|---:|:--|:--|
| velocity L2 (full 3D u,v,w) | 1.4721/1.5611 | 0.0890 | 0.20 | NO | yes |
| velocity L2 (in-plane u,v) | 1.4890/1.5912 | 0.1022 | 0.20 | NO | yes |
| spanwise w RMS / |u_exact| | 1.0158/1.1327 | 0.1169 | 0.20 | NO | yes |
| pressure L2 (fixed gauge) | 0.1514/0.8502 | 0.6988 | 0.20 | NO | NO |
| pressure L2 (mean-free gauge) | 0.6415/1.4603 | 0.8189 | 0.20 | NO | NO |

| quantity | three-level order f(h)=L+C h^p | stderr | convergence | extrapolated L | GCI |
|---|---:|---:|:--|---:|---:|
| velocity L2 (full 3D u,v,w) | 1.4243 | nan (3 levels: not identifiable) | monotone | -1.993046e-03 | 1.448e+00 |
| velocity L2 (in-plane u,v) | 1.4354 | nan (3 levels: not identifiable) | monotone | -2.174447e-03 | 1.476e+00 |
| pressure L2 (mean-free gauge) | 0.0000 | nan (3 levels: not identifiable) | monotone | -1.022024e+05 | 1.397e+06 |
