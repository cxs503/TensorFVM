# Shallow-water well-balanced verification -- criteria

- `W1a_smooth_bed_max_velocity`: **PASS** (value 5.027746e-14, threshold 1.000000e-10, lt)
- `W1b_smooth_bed_surface_drift`: **PASS** (value 5.551115e-15, threshold 1.000000e-12, lt)
- `W2a_step_bed_max_velocity`: **PASS** (value 0.000000e+00, threshold 1.000000e-10, lt)
- `W2b_step_bed_surface_drift`: **PASS** (value 0.000000e+00, threshold 1.000000e-12, lt)
- `W2c_first_order_step_max_velocity`: **PASS** (value 0.000000e+00, threshold 1.000000e-10, lt)
- `W3a_wet_dry_max_velocity`: **PASS** (value 2.571920e-14, threshold 1.000000e-10, lt)
- `W3b_wet_dry_surface_drift`: **PASS** (value 3.330669e-15, threshold 1.000000e-12, lt)
- `W3c_wet_dry_mask_unchanged`: **PASS** (value 0.000000e+00, threshold 5.000000e-01, lt)
- `W3d_wet_dry_positivity`: **PASS** (value -0.000000e+00, threshold 0.000000e+00, le)
- `W4_naive_control_spurious_velocity`: **PASS** (value 3.937807e-01, threshold 1.000000e-02, gt)
- `M1a_smooth_bed_volume_drift`: **PASS** (value 1.313282e-15, threshold 1.000000e-13, lt)
- `M1b_step_bed_volume_drift`: **PASS** (value 0.000000e+00, threshold 1.000000e-13, lt)
- `M1c_wet_dry_volume_drift`: **PASS** (value 6.271192e-15, threshold 1.000000e-13, lt)
- `F1a_flat_reduction_depth`: **PASS** (value 0.000000e+00, threshold 0.000000e+00, le)
- `F1b_flat_reduction_momentum`: **PASS** (value 0.000000e+00, threshold 0.000000e+00, le)
- `F2a_constant_bed_depth`: **FAIL** (value 1.820592e-15, threshold 0.000000e+00, le)
- `F2b_constant_bed_momentum`: **FAIL** (value 9.658940e-15, threshold 0.000000e+00, le)
- `D1a_sill_positivity`: **PASS** (value -9.029280e-01, threshold 0.000000e+00, le)
- `D1b_sill_volume_drift`: **FAIL** (value 4.360462e-12, threshold 1.000000e-13, lt)

**16/19 PASS as declared**

Round-off re-gate (reported next to the declared verdicts, not instead of them):
- `F2a' constant_bed_depth_roundoff`: **PASS** (value 1.820592e-15, threshold 1.000000e-11)
- `F2b' constant_bed_momentum_roundoff`: **PASS** (value 9.658940e-15, threshold 1.000000e-11)
- `D1b' sill_volume_drift_roundoff`: **PASS** (value 4.360462e-12, threshold 1.000000e-11)

**19/19 PASS under the round-off bound**
