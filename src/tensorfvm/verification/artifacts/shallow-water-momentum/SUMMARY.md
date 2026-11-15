# Momentum identity + velocity budget — CASE-FVM-DAMBREAK-STOKER

All numbers below are injected from `momentum.json` / `velocity.json`.

## Momentum identity (exact: `M(t) = g/2 (h_L^2 - h_R^2) t`)

| case | rate [m^3/s^2] | M(t*) closed form | quadrature | quad rel err | guard |
|---|---:|---:|---:|---:|:--:|
| `ritter-dry` | 4.905000 | 3.132091953 | 3.132091953 | 3.47e-11 | ok |
| `stoker-wet` | 3.678750 | 2.349068965 | 2.349081617 | 5.39e-06 | ok |

| case | n | solver M | exact M | closure rel err |
|---|---:|---:|---:|---:|
| `ritter-dry` | 800 | 3.132091953 | 3.132091953 | 2.57e-12 |
| `ritter-dry` | 1600 | 3.132091953 | 3.132091953 | 2.96e-12 |
| `ritter-dry` | 3200 | 3.132091953 | 3.132091953 | 3.24e-12 |
| `stoker-wet` | 800 | 2.349068965 | 2.349068965 | 1.17e-14 |
| `stoker-wet` | 1600 | 2.349068965 | 2.349068965 | 2.53e-14 |
| `stoker-wet` | 3200 | 2.349068965 | 2.349068965 | 4.65e-14 |

## Velocity field (never graded by the anchor's G1-G6)

| case | n | u L1 | u L1 (mass) | hu L1 | depth L1 | missing-tip share | 5 % |
|---|---:|---:|---:|---:|---:|---:|:--:|
| `ritter-dry` | 800 | 5.0703% | 0.3669% | 0.5890% | 0.1542% | 54.9% | FAIL |
| `ritter-dry` | 1600 | 4.4351% | 0.1833% | 0.2951% | 0.0772% | 76.2% | PASS |
| `ritter-dry` | 3200 | 3.5497% | 0.0916% | 0.1477% | 0.0386% | 85.6% | PASS |
| `ritter-dry` | 6400 | 2.6727% | 0.0456% | 0.0739% | 0.0193% | 92.1% | PASS |
| `stoker-wet` | 800 | 0.6991% | 0.6783% | 0.6831% | 0.0851% | 0.0% | PASS |
| `stoker-wet` | 1600 | 0.3517% | 0.3380% | 0.3409% | 0.0427% | 0.0% | PASS |
| `stoker-wet` | 3200 | 0.1812% | 0.1706% | 0.1730% | 0.0219% | 0.0% | PASS |
| `stoker-wet` | 6400 | 0.0850% | 0.0848% | 0.0853% | 0.0105% | 0.0% | PASS |

Observed orders (last to first):
* `ritter-dry` u L1: ['0.19', '0.32', '0.41'], mass weighted: ['1.00', '1.00', '1.01'], depth: ['1.00', '1.00', '1.00']
* `stoker-wet` u L1: ['0.99', '0.96', '1.09'], mass weighted: ['1.00', '0.99', '1.01'], depth: ['0.99', '0.97', '1.06']

## Sensitivity control (Ritter, n = 800, window x in [-1.00, 0.00])

| eps | momentum predicted | momentum measured | gain | velocity L1 max delta | velocity predicted |
|---:|---:|---:|---:|---:|---:|
| 1e-03 | 2.752e-04 | 2.752e-04 | 1.000 | 1.667e-04 | 1.667e-04 |
| 1e-02 | 2.752e-03 | 2.752e-03 | 1.000 | 1.667e-03 | 1.667e-03 |
| 1e-01 | 2.752e-02 | 2.752e-02 | 1.000 | 1.667e-02 | 1.667e-02 |
