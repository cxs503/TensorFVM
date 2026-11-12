# Stokes' second problem (oscillating wall) — criteria (pre-declared)

- **A1** PASS: closed-form PDE residual (numerical derivatives) = 1.5854e-07 (< 1.000e-06) — max|residual|=7.927e-08 m/s^2
- **A2** PASS: omega->0 branch equals Couette = 2.5087e-10 (< 1.000e-03) — delta = 100 H
- **A3** PASS: delta<<H branch equals the semi-infinite Stokes layer = 2.7089e-14 (< 1.000e-09) — delta = H/200, y <= 20 delta
- **A4** PASS: closed-form wall shear vs central difference of the profile = 4.6357e-11 (< 1.000e-06) — 4 phases, W=5
- **A5** PASS: semi-infinite wall-shear amplitude rho U0 sqrt(nu omega) = 0.0000e+00 (< 1.000e-03) — ratio=1.000000000
- **A5b** PASS: semi-infinite wall-shear lead of 45 deg (plate convention) = 0.0000e+00 (< 5.000e-01) — mu du/dy phase=-135.0000 deg, plate stress phase=45.000000 deg
- **A6** PASS: steady moving wall equals Couette (both wall_flux variants) = 1.2408e-11 (< 1.000e-10) — half-cell: 1.241e-11 (314 steps); quadratic: 1.240e-11 (314 steps)
- **B1** PASS: solver vs closed form, W=2, ny=64, dt/T=5e-03 = 7.3502e-03 (< 3.000e-02) — 10 phases, settled after 3 periods (600 steps), delta/H=0.500, 4.1 s
- **B2** PASS: solver vs closed form, W=5, ny=64, dt/T=5e-03 = 6.8215e-03 (< 3.000e-02) — 10 phases, settled after 6 periods (1200 steps), delta/H=0.200, 7.1 s
- **B3** PASS: solver vs closed form, W=10, ny=64, dt/T=5e-03 = 7.6050e-03 (< 3.000e-02) — 10 phases, settled after 7 periods (1400 steps), delta/H=0.100, 8.1 s
- **B4a** PASS: bulk-velocity amplitude, W=5 = 6.6614e-04 (< 3.000e-02) — 1.409757e-01 vs 1.408818e-01 m/s
- **B4b** PASS: bulk-velocity phase, W=5 = 3.8513e-01 (< 5.000e+00) — -45.355 vs -45.740 deg
- **B5** PASS: wall-shear amplitude (3-point estimator), W=5 = 2.6407e-02 (< 3.000e-02) — 7.257243e-02 vs 7.070529e-02 Pa; scheme flux estimator (not graded) err=4.550e-05
- **B6** PASS: discrete unsteady momentum budget rho H dUbar/dt + tau_0 + tau_H = 3.7886e-15 (< 1.000e-08) — max|residual|=1.894e-15 Pa
- **B7a** PASS: no wall-normal velocity, W=5 = 0.0000e+00 (< 1.000e-12) — 
- **B7b** PASS: cell divergence after projection, W=5 = 0.0000e+00 (< 1.000e-10) — 
- **B8** PASS: temporal convergence order = 1.0247e+00 (> 7.000e-01) — dt/T=1e-02: 1.751e-02, 5e-03: 8.606e-03, 3e-03: 4.213e-03
- **B9** PASS: wall-normal convergence order = 6.6193e-01 (> 5.000e-01) — monotone ny=16: 8.213e-03, ny=32: 3.690e-03, ny=64: 2.332e-03
- **C1** PASS: oscillating vs steady-wall profile difference, W=5 = 2.8178e+00 (> 1.000e-01) — a solver without wall motion cannot pass B1..B3

**19/19 PASS — all_pass=True**
