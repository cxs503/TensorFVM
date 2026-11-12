### 3.1 tied zero-gradient ladder (raw payload, one process per level)

| n | cells | faces | dofs | total L2 | in-plane L2 | spanwise L2 | max\|w\| | tie residual | Picard | s | peak RSS MB |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|
| 8 | 64 | 544 | 1824 | 4.005423e-02 | 4.005423e-02 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True | 11.2 | 505.7 |
| 16 | 256 | 2112 | 7104 | 1.332788e-02 | 1.332788e-02 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True | 66.7 | 540.9 |
| 32 | 1024 | 8320 | 28032 | 3.805383e-03 | 3.805383e-03 | 0.0e+00 | 0.0e+00 | 0.0e+00 | True | 1008.2 | 858.2 |

### 3.2 observed orders on every available pair

| pair | total | in-plane | pressure (mean-free) | vs 1.8 gate |
|---|---:|---:|---:|---|
| 8->16 | 1.5875 | 1.5875 | 1.4359 | FAIL |
| 16->32 | 1.8083 | 1.8083 | 1.7453 | PASS |

### 3.3 three-level fit ``f(h) = L + C h^p`` (levels 32/16/8)

| p | stderr | convergence | monotone | asymptotic | C extrapolated | GCI |
|---:|---:|---|---|---|---:|---:|
| 1.4889 | nan (0 dof) | monotone | True | False | -1.465403e-03 | 1.7314 |

### 3.4 machine verdict (pre-declared criteria)

| id | statement | value | verdict |
|---|---|---|---|
| C1 | constraint manifold max|u_face-u_cell| on tied dofs <= 1e-12 | 0 | **PASS** |
| C2 | spanwise L2 and max|w| exactly 0 at every level (w exact = 0) | 0 | **PASS** |
| C3 | in-plane L2 (tied) <= 1.02 x in-plane L2 (delivered free traction) at n=8 | 0.999058 | **PASS** |
| C4 | anchor: total velocity L2 order >= 1.8 on the finest pair 16->32 | 1.80833 | **PASS** |
| C5 | in-plane L2 order >= 1.8 on the finest pair | 1.80833 | **PASS** |
| C6 | Picard reported numerical convergence at every level | [True, True, True] | **PASS** |
| C7 | every level pair with n_coarse >= 16 reports total order >= 1.8 | 1.80833 | **PASS** |
| C8 | three-level f(h)=L+C h^p gives p >= 1.8 with stderr <= 0.1 and a consistent ladder | {'order': 1.4888517724055252, 'order_stderr': nan, 'convergence': 'monotone'} | **FAIL** |

anchor_closed: **False**; not_run: []
