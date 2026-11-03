# Richardson/GCI tool: self-verification and annular application

## Annular Poiseuille refinement (closed-form reference)

| grid | h (m) | flow rate $Q$ (m3/s) | analytic $Q$ | relative $Q$ error (%) | field $L_2$ (%) | field $L_\infty$ (%) | linear its | true residual | single-grid 3% gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 256x64 | 0.00391 | 4.9486895999e-05 | 4.9473816620e-05 | 0.0264 | 0.0455 | 0.0715 | 77 | 7.96e-12 | True |
| 128x32 | 0.00781 | 4.9526059775e-05 | 4.9473816620e-05 | 0.1056 | 0.1819 | 0.2813 | 36 | 3.14e-12 | True |
| 64x16 | 0.01562 | 4.9681602214e-05 | 4.9473816620e-05 | 0.4200 | 0.7251 | 1.0845 | 17 | 7.31e-13 | True |
| 32x8 | 0.03125 | 5.0286140785e-05 | 4.9473816620e-05 | 1.6419 | 2.8608 | 4.0424 | 8 | 3.13e-14 | True |

| triple (fine first) | p observed | asymptotic | extrapolated $Q$ error (%) | GCI (%) | band covers analytic $Q$ | improvement over fine grid |
|---|---:|---|---:|---:|---|---:|
| 256x64 -> 128x32 -> 64x16 | 1.9897 | True | 0.000202 | 0.0333 | True | 130.8x |
| 128x32 -> 64x16 -> 32x8 | 1.9585 | True | 0.003315 | 0.1360 | True | 31.9x |
