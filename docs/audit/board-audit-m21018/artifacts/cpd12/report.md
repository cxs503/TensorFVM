# Cylinder Re=100 (unbounded): vortex-shedding verification

Backend: **immersed**   Verdict: **FAIL: cd_mean = 1.5193 vs 1.3400 (13.38 %), st = 0.1441 vs 0.1640 (12.11 %)**

## 1. Physical problem

Two-dimensional incompressible flow past a circular cylinder, Re = U D / nu = 100: laminar vortex shedding. Domain 10.0D upstream, 20.0D downstream, +-10.0D laterally (blockage 5.0 %).

## 2. Literature anchor

| Quantity | Reference | Band | Source |
|---|---|---|---|
| Cd_mean | 1.340 | +/-0.070 (5.2 %) | Braza 1986; Schaefer-Turek 1996 |
| St | 0.164 | +/-0.005 (3.0 %) | Braza 1986; Schaefer-Turek 1996 |

Citation: Braza et al. JFM 165 (1986) 79-130, doi:10.1017/S0022112086003014; Schafer & Turek, NNFM 52 (1996), doi:10.1007/978-3-322-89849-4_39

## 3. Results

| Quantity | Computed | Reference | Rel. error | Status |
|---|---|---|---|---|
| Cd_mean | 1.5193 | 1.3400 | 13.38 % | FAIL |
| St | 0.1441 | 0.1640 | 12.11 % | FAIL |
| St (FFT cross-check) | 0.1441 | 0.1640 | 12.13 % | cross-check |
| Cl amplitude | 0.1128 | 0.3200 | 64.74 % | informational |

Record: 6001 force samples, t in [0.02, 120.02] tU/D; statistics window [50.74, 120.02] (10.00 periods).

## 4. Gates

- all gates passed

## 5. Figures

![force_time_series.png](force_time_series.png)
![lift_spectrum.png](lift_spectrum.png)
![error_bars.png](error_bars.png)
![phase_plot.png](phase_plot.png)
