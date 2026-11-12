# Pulsatile channel — criteria (pre-declared)

- **A1** PASS: closed-form PDE residual (numerical derivatives) = 2.6570e-09 (< 1.000e-06) — max|residual|=3.188e-11 m/s^2
- **A2** PASS: omega->0 limit equals the analytic parabola = 1.1245e-09 (< 1.000e-06) — centre 0.150000000 vs 0.150000000
- **A3** PASS: bulk / wall-gradient formulas vs numeric = 4.9802e-09 (< 1.000e-06)
- **A4** PASS: Stokes-layer decay slope vs -1/delta = 4.1293e-03 (< 2.000e-02) — W=40: slope*delta=-0.99587; W=80: slope*delta=-0.99926
- **A5** PASS: steady solver bulk velocity vs G0 H^2/(12 nu) = 4.8828e-04 (< 1.000e-03) — bulk_num=0.100048828 m/s
- **B1** PASS: solver vs closed form, W=2, ny=32, dt/T=1/200 = 1.5429e-02 (< 3.000e-02) — 8 phases, 400 steps, 8.2 s, delta/H=0.707
- **B2** PASS: solver vs closed form, W=5, ny=32, dt/T=1/200 = 6.0871e-03 (< 3.000e-02) — 8 phases, 400 steps, 8.2 s, delta/H=0.283
- **B3** PASS: solver vs closed form, W=10, ny=32, dt/T=1/200 = 3.9393e-03 (< 3.000e-02) — 8 phases, 400 steps, 8.2 s, delta/H=0.141
- **B4a** PASS: bulk-velocity amplitude, W=5 = 3.3207e-03 (< 3.000e-02) — 0.036894 vs 0.037017 m/s
- **B4b** PASS: bulk-velocity phase lag, W=5 = 1.0727e-01 (< 5.000e+00) — -66.803 vs -66.696 deg
- **B5** PASS: wall-shear amplitude, W=5, ny=64 (3-point fit) = 1.6520e-02 (< 3.000e-02) — 2.574568e-03 vs 2.532728e-03 Pa; scheme-consistent wall flux (not graded) err=8.191e-04
- **B6** PASS: temporal convergence order at W=5 = 8.7449e-01 (> 7.000e-01) — dt/T=1/100: 1.065e-02 (192 steps); dt/T=1/200: 6.087e-03 (400 steps); dt/T=1/400: 3.167e-03 (800 steps)
- **B7** PASS: wall-normal convergence ny/2 -> ny at W=5 = 1.0000e+00 (> 5.000e-01) — ny=16: 7.636e-03 -> ny=32: 6.087e-03
- **B8a** PASS: spanwise/normal velocity stays zero at W=5 = 0.0000e+00 (< 1.000e-12)
- **B8b** PASS: cell divergence after projection at W=5 = 0.0000e+00 (< 1.000e-10)
- **C1** PASS: oscillatory vs steady profile difference (W=5) = 1.3414e+00 (> 1.000e-01) — a steady solver cannot pass B1..B3
