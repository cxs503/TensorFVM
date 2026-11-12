# criteria 判定表（程序化注入，零手抄）

| 判据 | 实测 | 门 | PASS |
|---|---|---|---|
| A1/A2 psi/u/v psi_abs_max Ra=1000 n=32 | 0.7848 | 3 | PASS |
| A1/A2 psi/u/v u_max Ra=1000 n=32 | -0.1387 | 3 | PASS |
| A1/A2 psi/u/v v_max Ra=1000 n=32 | 0.1935 | 3 | PASS |
| A3 streamfunction quadrature gap Ra=1000 n=32 | 0.007316 | 0.01 | PASS |
| A4 kinematic |dpsi/dy-u|/max|u| Ra=1000 n=32 | 0.07599 | 0.1 | PASS |
| A5 mid-plane vs global extrema Ra=1000 n=32 | 2.398e-16 | 0.01 | PASS |
| A6 |psi| diagnostic Ra=1000 n=32 | 1.183 | — | PASS |
| A1/A2 psi/u/v psi_abs_max Ra=10000 n=32 | 0.6914 | 3 | PASS |
| A1/A2 psi/u/v u_max Ra=10000 n=32 | 0.3468 | 3 | PASS |
| A1/A2 psi/u/v v_max Ra=10000 n=32 | 0.08165 | 3 | PASS |
| A3 streamfunction quadrature gap Ra=10000 n=32 | 0.00786 | 0.01 | PASS |
| A4 kinematic |dpsi/dy-u|/max|u| Ra=10000 n=32 | 0.07905 | 0.1 | PASS |
| A5 mid-plane vs global extrema Ra=10000 n=32 | 0 | 0.01 | PASS |
| A6 |psi| diagnostic Ra=10000 n=32 | 5.106 | — | PASS |
| A1/A2 psi/u/v psi_abs_max Ra=10000 n=64 | 0.212 | 3 | PASS |
| A1/A2 psi/u/v u_max Ra=10000 n=64 | 0.1501 | 3 | PASS |
| A1/A2 psi/u/v v_max Ra=10000 n=64 | 0.1971 | 3 | PASS |
| A3 streamfunction quadrature gap Ra=10000 n=64 | 0.00231 | 0.01 | PASS |
| A4 kinematic |dpsi/dy-u|/max|u| Ra=10000 n=64 | 0.04633 | 0.1 | PASS |
| A5 mid-plane vs global extrema Ra=10000 n=64 | 0.0005609 | 0.01 | PASS |
| A6 |psi| diagnostic Ra=10000 n=64 | 5.082 | — | PASS |
| A1/A2 psi/u/v psi_abs_max Ra=100000 n=32 | 7.65 | 5 | FAIL |
| A1/A2 psi/u/v u_max Ra=100000 n=32 | 4.266 | 5 | PASS |
| A1/A2 psi/u/v v_max Ra=100000 n=32 | -2.385 | 5 | PASS |
| A3 streamfunction quadrature gap Ra=100000 n=32 | 0.01448 | 0.01 | FAIL |
| A4 kinematic |dpsi/dy-u|/max|u| Ra=100000 n=32 | 0.1024 | 0.1 | FAIL |
| A5 mid-plane vs global extrema Ra=100000 n=32 | 0.1835 | 0.2 | PASS |
| A6 |psi| diagnostic Ra=100000 n=32 | 9.808 | — | PASS |
| A7 registry hygiene (no superseded reference digits) | 14 | 0 | FAIL |

## 逐网格明细

| Ra | n | steps | |psi|max | err% acc | u_max | err% acc | v_max | err% acc | Nu | err% acc |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | 32 | 3800 | 1.1832 | +0.78 | 3.644 | -0.14 | 3.704 | +0.19 | 1.1203 | +0.20 |
| 10000 | 32 | 3600 | 5.1061 | +0.69 | 16.234 | +0.35 | 19.633 | +0.08 | 2.2716 | +1.28 |
| 10000 | 64 | 11200 | 5.0818 | +0.21 | 16.202 | +0.15 | 19.656 | +0.20 | 2.2510 | +0.36 |
| 100000 | 32 | 8000 | 9.8080 | +7.65 | 36.212 | +4.27 | 66.954 | -2.38 | 4.7445 | +4.99 |
