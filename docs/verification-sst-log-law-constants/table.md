# SST-2003 constant algebra: what log-law slope do the constants imply?

`k` budget of the log-layer ansatz closes identically:
  0

omega budget solved for $\kappa^2$:
  $\kappa^2 = \frac{3.33333333333333 \beta - 0.3 \gamma}{\sigma_{\omega}}$

| constant family | beta | gamma | sigma_omega | implied kappa | vs 0.41 |
|---|---|---|---|---|---|
| inner_k_omega | 0.0750 | 0.55556 | 0.500 | 0.40825 | -0.427% |
| outer_k_epsilon | 0.0828 | 0.44000 | 0.856 | 0.41015 | +0.037% |

Numeric root through `sst2003_closure` (no closed form used): 0.408242707869
|closed form - root|: 5.58e-06
closure eddy viscosity vs kappa*u_tau*y, max rel error: 2.22e-16
the gap is finite-difference truncation: it shrinks with sampling, see ladder

| samples | numeric root | gap to closed form |
|---|---|---|
| 81 | 0.407966298 | -2.82e-04 |
| 161 | 0.408167374 | -8.09e-05 |
| 321 | 0.408226675 | -2.16e-05 |
| 641 | 0.408242708 | -5.58e-06 |

F1 across the log layer: [1.000e+00, 1.000e+00]

| y+ | u+ (model kappa) | u+ (wall kappa) | du+ | du+/u+ [%] | cf bias [%] |
|---|---|---|---|---|---|
| 30 | 13.8980 | 13.8624 | +0.03559 | +0.2568 | +0.5135 |
| 100 | 16.8471 | 16.7989 | +0.04819 | +0.2869 | +0.5738 |
| 300 | 19.5381 | 19.4785 | +0.05969 | +0.3065 | +0.6129 |
| 1000 | 22.4873 | 22.4150 | +0.07229 | +0.3225 | +0.6450 |

Negative controls (1% constant edits):

| edit | implied kappa | change [%] |
|---|---|---|
| sigma_omega1 +1% | 0.406222 | -0.4963 |
| sigma_omega1 -1% | 0.410305 | +0.5038 |
| beta1 +1% | 0.414327 | +1.4889 |
| gamma1 +1% | 0.404145 | -1.0051 |
| beta_star +1% | 0.403141 | -1.2510 |
