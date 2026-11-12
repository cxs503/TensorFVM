### 3.1 ladder under the tied zero-gradient closure (raw payload)

| n | cells | faces | total L2 | in-plane L2 | spanwise L2 | max\|w\| | tie residual | Picard conv | s |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 8 | 64 | 544 | 4.0054e-02 | 4.0054e-02 | 0.0000e+00 | 0.0000e+00 | 0.0e+00 | True | 7.7 |
| 16 | 256 | 2112 | 1.3328e-02 | 1.3328e-02 | 0.0000e+00 | 0.0000e+00 | 0.0e+00 | True | 65.8 |

### 3.2 boundary-closure control at n = 8 (same mesh, same discretisation)

| closure | in-plane L2 | spanwise L2 | max\|w\| |
|---|---:|---:|---:|
| symmetry | 4.0092e-02 | 9.0032e-03 | 1.8955e-02 |
| dirichlet | 3.9458e-02 | 5.7157e-18 | 1.8617e-17 |
| zero_gradient | 4.0054e-02 | 0.0000e+00 | 0.0000e+00 |

### 3.3 observed orders (tied closure, in-plane; total == in-plane because w == 0)

| pair | in-plane (== total) order | vs 1.8 gate |
|---|---:|---|
| 8 -> 16 | 1.588 | FAIL |

