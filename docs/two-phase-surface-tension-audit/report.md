# Two-phase CSF surface-tension audit

parameters: {"radius_m": 0.2, "sigma_n_m": 0.07, "mu_pa_s": 0.01, "dt_s": 1e-05, "steps": 10, "grids": [24, 32], "rho_liquid": 1000.0, "rho_gas": 1.0}

| n | kappa rel err | net/gross CSF | dP (Pa) | exact | rel err | |u|max | Ca | div_final |
|---|---|---|---|---|---|---|---|---|
| 24 | +0.0032 | 0.0000 | -1.18714e+22 | 0.35 | -33918221948708261462016.0000 | 4.47e+20 | 6.39e+19 | 4.26e+22 |
| 32 | +0.1005 | 0.0000 | 8.49545e+23 | 0.35 | +2427270233001129123250176.0000 | 1.45e+23 | 2.07e+22 | 1.85e+25 |

criteria: {
  "C1_curvature_within_3pct": {
    "24": true,
    "32": false
  },
  "C2_net_csf_force_within_3pct": {
    "24": true,
    "32": true
  },
  "C3_young_laplace_within_3pct": {
    "24": false,
    "32": false
  },
  "C4_capillary_number_below_0p01": {
    "24": false,
    "32": false
  },
  "C5_finite_iterates": {
    "24": true,
    "32": true
  }
}

all criteria passed: False
