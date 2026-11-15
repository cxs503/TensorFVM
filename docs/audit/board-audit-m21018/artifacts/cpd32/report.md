# Cylinder Re=100 (unbounded): vortex-shedding verification

Backend: **immersed**   Verdict: **GATED: no vortex shedding: lift half-amplitude nan < gate 0.05**

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
| Cd_mean | nan | 1.3400 | nan % | FAIL |
| St | not reported | 0.1640 | - | FAIL |
| Cl amplitude | nan | 0.3200 | nan % | informational |

Record: 6001 force samples, t in [0.02, 120.02] tU/D; statistics window [90.02, 120.02] (0.00 periods).

## 4. Gates

- no vortex shedding: lift half-amplitude nan < gate 0.05

## 5. Figures

![force_time_series.png](force_time_series.png)
![lift_spectrum.png](lift_spectrum.png)
![error_bars.png](error_bars.png)
![phase_plot.png](phase_plot.png)
