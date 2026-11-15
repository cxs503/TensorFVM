# CASE-FVM-DAMBREAK-STOKER · long-time (self-similar) verification

Programmatically generated from `study.json` -- do not edit by hand.
dx = 0.005 m, CFL = 0.4, order = 2, t* = 1.0, 2.0, 5.0, 10.0
tolerances: position 5%, speed 5%, collapse L1 2.0% h0, step drift 0.50%

## ritter-dry

| t* | n | steps | x_f exact | x_f measured | position err | x_f/t err | speed (fit) err | depth L1 | extrema | mass drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 800 | 931 | 2.0000 | 2.0006 | 0.6393 % | 0.0297 % | 0.0391 % | 0.1645 % | 0 | 2.84e-14 |
| 2 | 1600 | 1904 | 4.0000 | 4.0005 | 0.3177 % | 0.0135 % | 0.0125 % | 0.0823 % | 0 | 5.33e-14 |
| 5 | 4000 | 4854 | 10.0000 | 10.0005 | 0.1158 % | 0.0052 % | 0.0025 % | 0.0329 % | 0 | 1.32e-13 |
| 10 | 8000 | 9804 | 20.0000 | 20.0005 | 0.0374 % | 0.0026 % | 0.0003 % | 0.0165 % | 0 | 2.58e-13 |

| t* | collapse L1/h0 vs t*=2 (numerical) | collapse L1/h0 (exact, noise floor) |
|---:|---:|---:|
| 1 | 3.893e-05 | 0.000e+00 |
| 2 | 0.000e+00 | 0.000e+00 |
| 5 | 2.390e-05 | 0.000e+00 |
| 10 | 3.204e-05 | 0.000e+00 |

**verdict: PASS**

| criterion | value | limit | pass |
|---|---:|---:|:--:|
| front_position_max | 0.0064 | 0.05 | yes |
| front_speed_x_over_t_max | 2.966e-04 | 0.05 | yes |
| front_speed_fit | 3.908e-04 | 0.05 | yes |
| similarity_collapse_l1 | 3.893e-05 | 0.02 | yes |
| spurious_extrema_total | 0.000e+00 | 0 | yes |
| mass_drift_max | 2.583e-13 | 1e-10 | yes |
| negative_depth | 0.000e+00 | 0 | yes |

## stoker-wet

| t* | n | steps | x_f exact | x_f measured | position err | x_f/t err | speed (fit) err | depth L1 | extrema | mass drift |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 589 | 574 | 0.9444 | 0.9444 | 0.1748 % | 0.0048 % | 0.0098 % | 0.1053 % | 0 | 1.48e-14 |
| 2 | 1178 | 1148 | 1.8888 | 1.8885 | 0.1120 % | 0.0131 % | 0.0076 % | 0.0525 % | 0 | 2.56e-14 |
| 5 | 2944 | 2869 | 4.7220 | 4.7233 | 0.3467 % | 0.0280 % | 0.0009 % | 0.0226 % | 0 | 5.56e-14 |
| 10 | 5889 | 5738 | 9.4439 | 9.4437 | 0.0298 % | 0.0023 % | 0.0006 % | 0.0105 % | 0 | 1.11e-13 |

| t* | h2 exact | h2 measured | h2 err | h1 err | step h2/h1 err | bore position err | plateau cells |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.7269 | 0.7269 | 0.0000 % | 0.0000 % | 0.0000 % | 0.0048 % | 150 |
| 2 | 0.7269 | 0.7269 | 0.0001 % | 0.0000 % | 0.0001 % | 0.0131 % | 301 |
| 5 | 0.7269 | 0.7269 | 0.0000 % | 0.0000 % | 0.0000 % | 0.0280 % | 751 |
| 10 | 0.7269 | 0.7269 | 0.0000 % | 0.0000 % | 0.0000 % | 0.0023 % | 1502 |

| t* | collapse L1/h0 vs t*=2 (numerical) | collapse L1/h0 (exact, noise floor) |
|---:|---:|---:|
| 1 | 6.822e-05 | 0.000e+00 |
| 2 | 0.000e+00 | 0.000e+00 |
| 5 | 4.825e-05 | 0.000e+00 |
| 10 | 5.613e-05 | 0.000e+00 |

**verdict: PASS**

| criterion | value | limit | pass |
|---|---:|---:|:--:|
| front_position_max | 0.0035 | 0.05 | yes |
| front_speed_x_over_t_max | 2.803e-04 | 0.05 | yes |
| front_speed_fit | 9.844e-05 | 0.05 | yes |
| similarity_collapse_l1 | 6.822e-05 | 0.02 | yes |
| spurious_extrema_total | 0.000e+00 | 0 | yes |
| mass_drift_max | 1.110e-13 | 1e-10 | yes |
| negative_depth | 0.5000 | 0 | yes |
| still_water_step_drift | 5.697e-07 | 0.005 | yes |

## controls (t* = 5)

### CFL refinement at fixed dx (temporal vs spatial error)

| CFL | case | position err | x_f/t err | speed fit err | depth L1 |
|---:|---|---:|---:|---:|---:|
| 0.4 | ritter-dry | 0.0052 % | 0.0052 % | 0.0025 % | 0.0329 % |
| 0.4 | stoker-wet | 0.0280 % | 0.0280 % | 0.0009 % | 0.0226 % |
| 0.2 | ritter-dry | 0.0053 % | 0.0053 % | 0.0023 % | 0.0329 % |
| 0.2 | stoker-wet | 0.0281 % | 0.0281 % | 0.0014 % | 0.0226 % |

### scheme order (falsifiability: the metric must detect a coarser scheme)

| order | case | position err | x_f/t err | speed fit err | depth L1 | extrema |
|---:|---|---:|---:|---:|---:|---:|
| 2 | ritter-dry | 0.0052 % | 0.0052 % | 0.0025 % | 0.0329 % | 0 |
| 2 | stoker-wet | 0.0280 % | 0.0280 % | 0.0009 % | 0.0226 % | 0 |
| 1 | ritter-dry | 0.1364 % | 0.1364 % | 0.0978 % | 0.2603 % | 0 |
| 1 | stoker-wet | 0.0177 % | 0.0177 % | 0.0330 % | 0.1395 % | 0 |

wall time: 0.0 s
