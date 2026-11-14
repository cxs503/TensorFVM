| n | h | cells | vel L2 (full 3D) | vel L2 (in-plane u,v) | spanwise share | p L2 (fixed gauge) | p L2 (mean-free) | max\|w\| |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 6.25000e-02 | 256 | 1.257478e-02 | 1.201574e-02 | 3.708e-03 | 1.564750e-01 | 9.732922e-02 | 9.414e-03 |
| 32 | 3.12500e-02 | 1024 | 3.839693e-03 | 3.431614e-03 | 1.723e-03 | 7.812177e-02 | 2.895337e-02 | 4.526e-03 |
| 64 | 1.56250e-02 | 4096 | 1.217236e-03 | 8.964913e-04 | 8.234e-04 | 3.949543e-02 | 7.986706e-03 | 2.196e-03 |

| quantity | pair orders | least-squares slope | R² | residual dof |
|---|---|---:|---:|---:|
| `velocity_l2` | 16→32: 1.7115 / 32→64: 1.6574 | **1.6844** | 0.999914 | 1 |
| `velocity_l2_inplane` | 16→32: 1.8080 / 32→64: 1.9365 | **1.8722** | 0.999607 | 1 |
| `pressure_l2` | 16→32: 1.0021 / 32→64: 0.9840 | **0.9931** | 0.999972 | 1 |
| `pressure_mean_free_l2` | 16→32: 1.7491 / 32→64: 1.8581 | **1.8036** | 0.999696 | 1 |

All values above are recomputed from the stored cell-centred fields by `scripts/kovasznay_inplane_decomposition.py`; the table is generated, not transcribed.
