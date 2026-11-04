# SST-2003 constant algebra: what log-law slope do the constants actually imply?

**Verification-debt repayment.** The shipped SST-2003 closure
(`src/tensorfvm/native_sst3d.py`) carries ten published constants, and the wall
functions (`src/tensorfvm/wall_functions.py`) assume `kappa = 0.41`. Nothing in
the repository connected the two: the model's constants fix the slope of its own
equilibrium logarithmic layer, and a transcription slip in `beta`, `gamma` or
`sigma_omega` would move that slope away from `0.41` — biasing skin friction and
wall shear — while every channel/plate regression test stays green, because the
tests grade the *solver*, not the constant set.

Artifacts in this directory: `audit.json` (machine-readable, generated),
`table.md` (generated tables), `fig1..3` (figures). Reproduce with

```
python3 scripts/audit_sst_log_law_constants.py
python3 -m pytest tests/test_sst_log_law_constants.py -q      # 14 passed
```

## 1. Case: background, fields, parameters

Equilibrium logarithmic layer of a k-omega type eddy-viscosity model, wall
distance `y`, friction velocity `u_tau`, `nu -> 0`:

| quantity | log-layer value |
|---|---|
| `k` | `u_tau^2 / sqrt(beta*)` |
| `omega` | `u_tau / (sqrt(beta*) kappa y)` |
| `nu_t` | `kappa u_tau y` |
| `dU/dy` | `u_tau / (kappa y)` |
| `P/rho` | `u_tau^3 / (kappa y)` |

Constants are **read from the module**, never re-typed here (beta*=0.09,
inner {beta1=0.075, gamma1=5/9, sigma_w1=0.5}, outer {beta2=0.0828, gamma2=0.44,
sigma_w2=0.856}, a1=0.31); the wall law is the shipped Spalding law with
kappa=0.41, E=9.8 (B=ln E/kappa=5.567).

## 2. Method

Three legs, no solver, no reference table copied from a paper:

1. **Symbolic** — substitute the ansatz into the k and omega budget with SymPy.
   The k budget closes *identically* (residual 0), so `kappa` is fixed by the
   omega budget alone; solving it gives the closed form
   `kappa^2 = (beta/sqrt(beta*) - gamma sqrt(beta*)) / sigma_omega`.
2. **Numeric** — build the log-layer state, call the repository closure
   `sst2003_closure` (the object under audit), assemble the per-mass omega
   budget (production − destruction + cross diffusion + d/dy[(nu + sigma nu_t)
   domega/dy]) and bisect for the `kappa` that zeroes it. The root finder knows
   nothing about the closed form. The molecular `nu`-part is excluded from the
   gate and reported separately (`molecular_share_at_root`) because the identity
   is the `nu -> 0` statement.
3. **Consequence** — convert the implied `kappa` into a log-law offset and a
   skin-friction bias against the wall-function `kappa`.

Negative controls: each constant is perturbed by 1 % and the implied slope is
recomputed; an audit whose answer cannot move proves nothing.

## 3. Results

| constant family | beta | gamma | sigma_omega | implied kappa | vs 0.41 |
|---|---|---|---|---|---|
| inner k-omega | 0.0750 | 0.55556 | 0.500 | **0.408248** | −0.427 % |
| outer k-epsilon | 0.0828 | 0.44000 | 0.856 | **0.410151** | +0.037 % |

* Numeric root through the repository closure: `kappa = 0.4082427`, i.e.
  `|closed form − root| = 5.6e-6`; the gap halves at second order under
  refinement (81→641 samples: −2.8e-4, −8.1e-5, −2.2e-5, −5.6e-6; fitted order
  ≈ 2.0), so the agreement is the identity, not a coincidence.
* In the ideal log layer `grad(k) = 0`, so the cross-diffusion term falls to its
  floor and `F1 = 1` everywhere (checked: `F1 in [1.000, 1.000]`). **The inner
  (near-wall) family therefore governs the log layer**: the operative implied
  slope is 0.40825, 0.43 % below the wall law's 0.41.
* Forcing `kappa = 0.41` in the same budget raises the relative omega residual
  from 2.7e-4 (at the root) to 1.8e-3, a 6.8× violation — the mismatch is real
  and detected by the budget, it is simply small.
* Consistency the other way: `sigma_w1` would have to be 0.49574 (not 0.5, a
  0.85 % change) for the inner family to imply exactly 0.41; the outer family is
  already within 0.07 % of that calibration point.
* `a1 = 0.31` is only 3.33 % above `sqrt(beta*) = 0.3`, the threshold at which
  `nu_t = a1 k / max(a1 omega, Omega F2)` stops selecting the `a1 omega` branch;
  below it `nu_t` is biased low by exactly `a1/sqrt(beta*)`.
* Consequence for the wall law: `u+` shifts by +0.036/+0.048/+0.060/+0.072 at
  `y+ = 30/100/300/1000` (0.26–0.32 %), implying a `c_f` bias of **+0.51 % to
  +0.65 %**.

## 4. Figures

* `fig1_omega_budget_vs_kappa.png` — omega budget residual vs the assumed slope:
  a clean minimum at 0.40825, with the wall law's 0.41 and the outer family's
  0.41015 marked.
* `fig2_sigma_to_kappa.png` — implied slope vs `sigma_w1` with the implemented
  value and the value that would make `kappa = 0.41` exactly.
* `fig3_wall_law_consequence.png` — `Δu+/u+` and the implied `c_f` bias vs `y+`,
  against the ±3 % acceptance band.

## 5. Caveats

* The identity is the *equilibrium log-layer* statement: it assumes zero
  convection, zero pressure gradient and `nu -> 0`. The molecular part of the
  diffusion is therefore excluded from the root; it is 0.65 % of the omega
  budget at `y+ = 300` and is reported in `audit.json`.
* `F1 = 1` in the ideal log layer only because `grad(k) = 0` there; the outer
  family's 0.41015 is the value that would govern a region with strong `grad(k)`
  and is reported as a bracket, not as an operating point.
* `cf` bias is a first-order propagation of the `u+` offset (2 Δu+/u+); it is a
  sensitivity statement about the constant set, not a prediction of any
  particular simulation's error.
* The `a1` limiter margin is an algebraic branch margin; it says nothing about
  how the model behaves away from the ideal log layer.
* This module does not make the SST model physically accurate — it only rules
  out constant-level transcription as a source of the >3 % errors seen in
  near-wall RANS results.

## 6. Error verdict

| quantity | reference | value | error | < 3 %? |
|---|---|---|---|---|
| symbolic omega-budget identity | SymPy re-derivation | residual 0 | exact | yes |
| k-budget of the ansatz | analytic | residual 0 | exact | yes |
| numeric root (closure) | closed form 0.408248 | 0.4082427 | 1.4e-5 rel. | yes |
| `nu_t` from closure | `kappa u_tau y` | — | 2.2e-16 | yes |
| inner-family implied kappa | wall law 0.41 | 0.408248 | −0.427 % | yes |
| outer-family implied kappa | wall law 0.41 | 0.410151 | +0.037 % | yes |
| family spread | — | 0.464 % | — | yes |
| implied `c_f` bias | wall law | 0.51–0.65 % | — | yes |

Conclusion: the implemented SST-2003 constant set is internally consistent to
≤0.5 % in the log-law slope and consistent with the Spalding wall law to
≤0.65 % in skin friction. No transcription defect found; the constant set cannot
explain the ≥3 % discrepancies reported for near-wall RANS runs, and the 1 %
perturbation controls show this audit would catch an edit of that size
(implied slope moves 0.50–1.49 % per 1 % constant change).