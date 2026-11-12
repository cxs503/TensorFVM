# Pre-declared criteria — shallow-water bottom friction / slope sources

Declared **before** the first run of `python -m tensorfvm.verification.shallow_water_friction`
(thresholds were fixed from the closed forms, not tuned to observed output).
Observed values: `summary.json` / `results.md` (machine-generated).

Setup: 1-D shallow water, flat bed, uniform grid, MUSCL + HLL + SSP-RK3,
`g = 9.81 m/s^2`, `h = 1 m`, `u0 = 1.5 m/s`, `n = 0.03 s m^(-1/3)`, `S0 = 1e-3`,
propagation length 200 m, 64 cells (production CFL 0.4).

| id | statement | gate | rationale |
|---|---|---|---|
| C1 | Friction-only decay `u(t)` matches `u0/(1 + C u0 t)`, `C = g n^2 / h^(4/3)` | rel. max < 1 % | the uniform state has zero flux divergence, so the semi-discrete system *is* this ODE; a factor of `h`, or a missing `|u|`, changes `C` by orders of magnitude |
| C2 | Observed temporal order of the source term on a fixed-`dt` ladder (0.4/0.2/0.1 s) | p > 2.7 | SSP-RK3 must stay third order with the new source; a splitting/limiter hack would drop it |
| C3 | Normal-flow relaxation `u(t)` matches `u_n tanh(g S0 t / u_n)` | rel. max < 3 % | fixes the *magnitude* of the drag: `u_n` carries `n`, `h^(2/3)` and `sqrt(S0)` |
| C4 | Steady velocity after 1200 s vs Manning `u_n = h^(2/3) sqrt(S0)/n` | rel. < 3 % | end state of the same anchor, judged on the textbook normal-flow formula |
| C5 | Depth stays uniform while friction acts | max drift < 1e-12 m | friction is a momentum sink; leakage into the mass flux would break conservation |
| C6 | Negative control: `n = 0`, `S0 = 0` -> no decay | rel. drift of `u` < 1e-12 | a gate that cannot fail proves nothing |
| C6b | Residual with sources off is bit-identical to the flux-only residual | max abs diff <= 0 | the default path of a merged solver must not move |
| C7 | Lake at rest on a constant-slope bed with friction on | `max abs u < 1e-12 m/s` | `S_f(u=0) = 0`: friction must not create a force at rest (well-balanced property of the Audusse solver survives) |
| C8 | Stiff drag (`n = 0.15`) at CFL 0.95: velocity never reverses | `u_min >= 0` | drag must be a sink, never a source |
| C8b | Reversal guard fires in that run | activations > 0 | untested safety code is not safety code |
| C8c | `dt = 10 s` (CFL ~ 3.7, 3 steps): guarded run never reverses | `u_min >= 0` | isolate the guard: uniform state -> only drag acts |
| C8d | Same run **unguarded** does reverse / diverge | `-u_min_unguarded > 0` | negative control for the guard |
| C9 | Mechanical energy non-increasing in a friction-only run | `max dE <= 0` | friction dissipates and nothing else does |
| C9b | Energy left after 40 s vs exact decay | rel. < 1 % | ties the dissipation to the closed-form velocity |
| C10 | Wet/dry dam break with `n = 0.03` reaches a *shorter* front than inviscid | `x_f(friction) < x_f(inviscid)` | friction acts in the wet/dry front, not only in the uniform limit |
| C11 | Same dam break conserves volume | rel. < 1e-10 | the source must not leak mass through the dry front |

Not covered here (see `REPORT.md` §5): inflow/outflow boundaries (the 1-D solver has
transmissive ends only, so a *real* bed slope with steady discharge cannot be posed),
2-D, Coriolis/rotation, turbulent friction (Chéron/Manning-Strickler variants), grain
roughness of ice (Skinner/Beltaos forms).
