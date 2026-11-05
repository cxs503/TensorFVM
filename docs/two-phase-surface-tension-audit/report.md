> **读数警告（m3646, 2026-11-04，第 2 轮）**：本文件记录的 `dP = -1.19e+22 Pa` 等数字在当前源码上**不可复现**。重生成后 n=24/32/48/64 的 Δp 相对误差为 +1.86 % / +1.39 % / +0.72 % / +0.31 %（全部 ≤3 % 且随加密单调收缩），Ca ≤ 3.6e-8、迭代全部有界（无发散）；离散曲率仍在 n=32/48 失守（+10.05 % / +30.61 %）。旧记录保留为历史 FAIL 证据不动；最新证据与分类结论见 `docs/two-phase-capillary-evidence-refresh/report.md`（平台报告 3790）。

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
