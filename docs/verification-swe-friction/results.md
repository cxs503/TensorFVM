# Verification results (generated from ``summary.json``)

Module: `tensorfvm.verification.shallow_water_friction`  
Verdict: **16/16 PASS** (all_pass = True, runtime 8.9 s)

## Criteria

| id | criterion | value | gate | unit | verdict | note |
|---|---|---|---|---|---|---|
| C1 | friction-only decay u(t) vs u0/(1+C u0 t) (production CFL) | 3.614365e-09 | < 1.000e-02 | rel. max | **PASS** | u(60 s)=0.835836 vs 0.835836 m/s |
| C2 | observed time order of the friction source (fixed-dt ladder) | 3.003817e+00 | > 2.700e+00 | p | **PASS** | dt=0.4,0.2,0.1 err=7.40e-09,9.21e-10,1.15e-10 |
| C3 | normal-flow relaxation u(t) vs u_n tanh(g S0 t/u_n) | 8.090026e-10 | < 3.000e-02 | rel. max | **PASS** | t=20,60,120 |
| C4 | steady velocity vs Manning u_n = h^(2/3) sqrt(S0)/n | 3.987741e-10 | < 3.000e-02 | rel. | **PASS** | u_num=1.054093 vs u_n=1.054093 m/s (tanh residual 1.05e-14) |
| C5 | uniform depth preserved while friction acts (mass consistency) | 0.000000e+00 | < 1.000e-12 | m | **PASS** | a non-uniform answer would mean the source leaked into the mass flux |
| C6 | negative control: no source -> no decay (drift of u) | 0.000000e+00 | < 1.000e-12 | rel. | **PASS** |  |
| C6b | bit-identical inviscid residual when sources are off | 0.000000e+00 | <= 0.000e+00 | abs. | **PASS** | max |rhs_with_sources_off - rhs_flux_only| |
| C7 | lake at rest over a sloped bed with friction: max|u| | 1.643933e-13 | < 1.000e-12 | m/s | **PASS** | surface drift 1.688e-14 m |
| C8 | stiff drag at CFL 0.95: velocity never reverses | -1.384702e-01 | < 0.000e+00 | -u_min | **PASS** | finite=True, min u=0.138470 m/s, decay error 3.69% (n=0.15: accuracy limited by dt) |
| C8b | reversal guard actually fires in the stiff-drag run | 1.600000e+01 | > 1.000e+00 | count | **PASS** | a guard that never binds is untested code |
| C8c | guarded drag at dt far beyond CFL (dt=10 s): no reversal | -1.841205e-01 | < 0.000e+00 | -u_min | **PASS** | min u over 3 steps = 0.184120 m/s |
| C8d | negative control: the same run unguarded does reverse the flow | 1.793578e+300 | > 0.000e+00 | -u_min_unguarded | **PASS** | min u = -1.79358e+300 m/s (diverged) -> the guard is not vacuous |
| C9 | mechanical energy non-increasing in a friction-only run | -1.666966e-01 | < 0.000e+00 | J/m step | **PASS** | max forward difference of E(t) (must be <= 0) |
| C9b | energy left after 40 s vs exact decay | 2.591871e-11 | < 1.000e-02 | rel. | **PASS** |  |
| C10 | wet/dry dam break: friction slows the front | 2.590000e+00 | < 3.670e+00 | m | **PASS** | front no-friction 3.6700 m, with n=0.03 2.5900 m |
| C11 | wet/dry dam break with friction: volume conservation | 1.054712e-14 | < 1.000e-10 | rel. | **PASS** | finite=True, max|u|=3.559 m/s |

## Friction-only decay in a uniform channel (n = 0.03, h = 1 m, u0 = 1.5 m/s)

| t / s | u solver / (m/s) | u closed form / (m/s) | rel. difference |
|---|---|---|---|
| 10 | 1.324579333615 | 1.324579335679 | 1.559e-09 |
| 30 | 1.073495046130 | 1.073495049399 | 3.045e-09 |
| 60 | 0.835836195373 | 0.835836198394 | 3.614e-09 |

depth drift of the uniform state over the whole run: 0.000e+00 m

## Relaxation to Manning normal flow (n = 0.03, S0 = 1e-3, from rest)

| t / s | u solver / (m/s) | u_n tanh(g S0 t/u_n) / (m/s) | rel. difference |
|---|---|---|---|
| 20 | 0.193965182266 | 0.193965182261 | 2.751e-11 |
| 60 | 0.534198938008 | 0.534198937823 | 3.450e-10 |
| 120 | 0.850072380461 | 0.850072379773 | 8.090e-10 |
| 1200 | 1.054092552969 | 1.054092552969 | 1.055e-14 |

analytic normal velocity u_n = 1.054092553389 m/s; friction slope at the end of the run S_f = 9.999999992025e-04 vs the imposed S0 = 1.000e-03 (rel. 7.98e-10)

## Wet/dry dam break (Ritter geometry, t* = 2) with and without friction

| quantity | inviscid | with n = 0.03 |
|---|---|---|
| front position / m | 3.6700 | 2.5900 |
| volume drift (rel.) | 1.210e-14 | 1.055e-14 |
| max abs u / (m/s) | 5.6431 | 3.5590 |

## Stiff-drag robustness

| quantity | value |
|---|---|
| min u at CFL 0.95, n = 0.15 / (m/s) | 0.138470 |
| decay error vs closed form (dt-limited) | 3.69% |
| reversal-guard activations | 16 |
| min u, dt = 10 s, guarded / (m/s) | 0.184120 |
| min u, dt = 10 s, unguarded / (m/s) | -1.79358e+300 (diverged) |
