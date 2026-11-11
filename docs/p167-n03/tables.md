### Zalesak rotating disk (solid-body rotation, closed solution)

| grid | scheme | revs | steps | mass drift | relative L1 | sym. difference | peak | thickness median/p90 [cells] |
|---|---|---|---|---|---|---|---|---|
| 64x64 | cicsam | 1 | 396 | 0.00e+00 | 0.0866 | 0.1057 | 1.000 | 1.55 / 2.06 |
| 64x64 | cicsam | 2 | 792 | 0.00e+00 | 0.1378 | 0.1463 | 1.000 | 1.43 / 1.95 |
| 64x64 | plic | 1 | 396 | 4.17e-15 | 0.0481 | 0.0569 | 1.000 | 1.53 / 1.87 |
| 64x64 | plic | 2 | 792 | 2.63e-15 | 0.0758 | 0.0894 | 1.000 | 1.43 / 1.87 |
| 64x64 | fct | 1 | 396 | 0.00e+00 | 0.5812 | 0.2561 | 0.901 | 12.51 / 24.51 |
| 64x64 | fct | 2 | 792 | 0.00e+00 | 0.7016 | 0.3171 | 0.810 | 15.39 / 28.91 |
| 64x64 | cicsam | 1 | 396 | 0.00e+00 | 0.0866 | 0.1057 | 1.000 | 1.55 / 2.06 |
| 64x64 | cicsam | 2 | 792 | 0.00e+00 | 0.1378 | 0.1463 | 1.000 | 1.43 / 1.95 |
| 64x64 | cicsam | 4 | 1584 | 0.00e+00 | 0.2111 | 0.2358 | 1.000 | 1.46 / 1.94 |
| 64x64 | cicsam | 8 | 3167 | 0.00e+00 | 0.3079 | 0.3252 | 1.000 | 1.52 / 1.96 |
| 128x128 | cicsam | 1 | 798 | 2.20e-16 | 0.0398 | 0.0382 | 1.000 | 1.42 / 1.90 |
| 128x128 | cicsam | 2 | 1596 | 0.00e+00 | 0.0624 | 0.0612 | 1.000 | 1.52 / 1.94 |
| 128x128 | cicsam | 4 | 3192 | 2.20e-16 | 0.0926 | 0.0937 | 1.000 | 1.41 / 1.91 |
| 128x128 | cicsam | 8 | 6384 | 2.20e-16 | 0.1404 | 0.1396 | 1.000 | 1.44 / 1.89 |
| 128x128 | fct | 1 | 798 | 2.20e-16 | 0.4261 | 0.1405 | 0.986 | 16.35 / 39.20 |
| 128x128 | fct | 2 | 1596 | 2.20e-16 | 0.5255 | 0.1750 | 0.953 | 21.02 / 48.24 |
| 128x128 | fct | 4 | 3192 | 2.20e-16 | 0.6359 | 0.2457 | 0.891 | 27.45 / 54.48 |
| 128x128 | fct | 8 | 6384 | 2.20e-16 | 0.7555 | 0.3222 | 0.798 | 32.91 / 66.51 |
| 256x256 | cicsam | 1 | 1603 | 8.78e-16 | 0.0178 | 0.0156 | 1.000 | 1.49 / 1.89 |
| 256x256 | cicsam | 2 | 3205 | 1.10e-15 | 0.0281 | 0.0271 | 1.000 | 1.47 / 1.88 |

### Dam break vs Martin & Moyce (1952)

| run | t*=1 | 1.5 | 2.0 | 2.5 | 3.0 | worst | steps | mass drift |
|---|---|---|---|---|---|---|---|---|
| reference x* | 1.40 | 1.95 | 2.45 | 2.90 | 3.30 | — | — | — |
| cicsam 100x30 | 0.575 (-59.0 %) | 1.095 (-43.9 %) | 1.614 (-34.1 %) | 2.049 (-29.3 %) | 2.517 (-23.7 %) | 59.0 % | 86 | 8.9e-16 |
| cicsam 200x60 | 0.599 (-57.2 %) | 1.065 (-45.4 %) | 1.570 (-35.9 %) | 2.037 (-29.8 %) | 2.491 (-24.5 %) | 57.2 % | 170 | 8.1e-16 |
| fct 100x30 | 0.472 (-66.3 %) | 0.948 (-51.4 %) | 1.437 (-41.3 %) | 2.052 (-29.3 %) | 2.622 (-20.5 %) | 66.3 % | 92 | 0.0e+00 |

### Hydrostatic rest state

| grid | surface model | max pressure deviation | spurious velocity | mass drift |
|---|---|---|---|---|
| 32x16 | well-balanced | 0.000e+00 | 0.000e+00 | 0.0e+00 |
| 64x32 | well-balanced | 0.000e+00 | 0.000e+00 | 0.0e+00 |
| 128x64 | well-balanced | 0.000e+00 | 0.000e+00 | 0.0e+00 |
| 128x64 | control | 1.563e-02 | 6.968e-08 | 2.2e-16 |

### Machine verdict

| gate | criterion | measured | threshold | verdict |
|---|---|---|---|---|
| G1 Zalesak mass ledger | worst relative area drift over every grid, scheme and over 1 | 4.171e-15 | 0.001 | PASS |
| G2 CICSAM sharpness | CICSAM interface thickness, 90th percentile of 1/max|d alpha | 1.937 | 2.0 | PASS |
| G3 CICSAM vs PLIC (report only) | side-by-side quality table (no threshold: the gate reports t | — | — | PASS |
| G4 hydrostatic | max |p - rho g (H - y)| / (rho g H) over the well-balanced g | 0 | 0.01 | PASS |
| G5 dam-break front | worst surge-front error vs the digitised Martin & Moyce (195 | 0.6625 | 0.1 | FAIL |
