## Kovasznay order estimator: negative control

Verdict: PASS -- reference slope 1.9623, control slope 1.2456, margin 0.7167

| series | convection order | closure | velocity slope | p mean-free slope | reaches >= 1.8 |
|---|:--:|:--:|---:|---:|:--|
| second order (convection_order=2) | 2 | zero_gradient | 1.9623 | 1.8777 | yes |
| first order (convection_order=1) | 1 | zero_gradient | 1.2456 | 1.0138 | no |

| series | n | h | vel L2 (full) | p L2 (mean-free) | converged |
|---|---:|---:|---:|---:|:--|
| second order (convection_order=2) | 32 | 0.031250 | 3.435615e-03 | 2.885142e-02 | yes |
| second order (convection_order=2) | 64 | 0.015625 | 8.972614e-04 | 7.958339e-03 | yes |
| second order (convection_order=2) | 128 | 0.007812 | 2.262325e-04 | 2.136311e-03 | yes |
| first order (convection_order=1) | 32 | 0.031250 | 4.997776e-03 | 2.046736e-02 | yes |
| first order (convection_order=1) | 64 | 0.015625 | 1.939693e-03 | 8.948565e-03 | yes |
| first order (convection_order=1) | 128 | 0.007812 | 8.888359e-04 | 5.019535e-03 | yes |

Gates: {"control_premise_ok": true, "all_levels_converged": true, "reference_passes_threshold": true, "control_rejected_by_threshold": true, "control_below_ceiling": true, "margin_reached": true}
