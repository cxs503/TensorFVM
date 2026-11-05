# TwoPhaseSolver transient projection: audit and repair (m2766)

Scope: `src/tensorfvm/two_phase_solver.py`, the `tp.enabled=True` path
(`_projection_step`), which the merged VOF solver uses for every two-phase run.
The SIMPLE path (`tp.enabled=False`) is untouched and stays bit-identical.

## 1. Defects found (all reproduced before the repair)

| # | Defect | Evidence before |
|---|--------|-----------------|
| D1 | Pressure-Poisson operator built with negated off-diagonals *and* subtracted in `matvec`, i.e. `-A` instead of `A`: CG on an anti-elliptic operator | resting column → `|v| = 9.2e49` after 10 steps |
| D2 | Operator coefficients missed `dt`, right-hand side missed the cell volume (off by `dx*dy*dt`) | same blow-up, projection never removed divergence |
| D3 | Predictor omitted the pressure gradient of the previous iterate (non-incremental Chorin splitting); a uniform body force is divergence-free, so the correction could never balance gravity | resting column drift |
| D4 | Hydrostatic initialisation was discontinuous at the interface (`rho_g g (H - h_i)` jump) and its gauge was pinned at the wall, not at the pinched top cell (half-cell offset `rho_g g dy / 2`) | interior force residual `max = 157.0`, `rms = 27.8` |
| D5 | `sigma` was a dead parameter: `csf_surface_tension` was never assembled into the momentum predictor | static drop `dp = 0.0000` for `sigma/R = 0.29 … 0.44` |
| D6 | Sealed domain (`top_atmosphere=False`, `side_outlet=False`) gave an all-Neumann pressure problem with no datum and no error | silent divergence |

## 2. Repairs

* `_projection_pressure`: coefficients `dt (1/rho)_face (A_f / d_f)`, symmetric
  positive coefficients in `matvec`, right-hand side `V * div(u*)`; CG diagnostics
  exposed as `solver.cg_info`.
* `_projection_step`: predictor now carries `-grad(p^n)/rho` (incremental
  projection), the correction updates `u` and `p <- p + dphi`.
* `initialize_hydrostatic_pressure`: continuous piecewise-linear profile with the
  gauge at the pinned top-cell centre; matches the arithmetic-mean face density
  used by the momentum and pressure equations, so hydrostatic rest is an exact
  fixed point.
* `hydrostatic_balance_residual`: evaluates interior faces only (the two boundary
  rows have no defined gradient and reported a spurious `rho g`).
* `_csf_face`: density-weighted Brackbill source on the staggered faces, gated by
  `surface_tension` (exactly zero when off), with a compact interface weight
  `4 a (1-a)` because the curvature stencil also returns a finite `kappa` in the
  pure-phase cell next to the interface.
* `_check_pressure_datum`: sealed domains fail fast with `ValueError`.

## 3. Verification status

* Resting stratified column: `|u|max = 1.1e-14`, `|v|max = 1.4e-14` after 10 steps
  (was `9.2e49`); hydrostatic force residual `max < 1e-9` on interior faces
  (was `157.0`). Both promoted to plain assertions.
* Capillary source: exactly zero with the switch off (bit-identical guarantee),
  non-zero at a curved interface when on.
* Sealed domain: `ValueError` raised (promoted to a plain assertion).
* **Young-Laplace `dp = sigma/R`: still FAILING (xfail, honest).** With `sigma`
  now reaching the predictor, a static drop develops parasitic currents
  (`|v|` grows by ~3.4x per step) instead of a pressure jump.

## 4. Formerly open defect: CLOSED, kept for the record

The defect that used to be handed over here — "the assembled operator is a
factor ~2 from the operator the velocity correction applies", measured as
`max|div|` growing from `2.07e-1` to `4.15e-1` — no longer reproduces. Measured
through the production correction routine on the same random field
(`n = 32`, `dt = 1e-5`):

| quantity | as handed over | measured now |
|---|---|---|
| `max |div|` before -> after | `2.07e-1 -> 4.15e-1` (2.0x amplification) | `2.07e-1 -> 2.16e-11` (1.0e-10) |
| median `|div|` | `8.7e-2`, essentially unchanged | `4.39e-2 -> 3.67e-12` |

Residual causes of the old numbers: the audit drove `_projection_pressure`
directly and re-implemented the correction inline, leaving the atmospheric top
face uncorrected (that alone leaves 31-34% of the initial divergence), and the
script read `solver.cg_info`, an attribute that no longer exists, so the
published `reproduce` command crashed before printing anything.

The operator is now verified independently at matrix level —
`tests/test_two_phase_projection_operator_matrix.py` (12 checks) and
`scripts/two_phase_projection_operator_verify.py`, reported in
`docs/two-phase-projection-operator-verification/report.md`. Three
configurations (isotropic, anisotropic `dx != dy`, anisotropic + 1000:1
density) satisfy `||A dp - b||_inf / ||b||_inf < 1e-8`, and two mutation
controls (swapped `dx/dy` face geometry, dropped top datum) are rejected with
residuals of 0.20-0.34, so the gate is not vacuous.

What is **not** closed: the 2D Young-Laplace jump `dp = sigma/R` of a static
drop (section 3) still misses the 3% band (+4.9% / -12.2% / -7.9% for
`R = 0.16 / 0.20 / 0.24 m` at 20 steps, tiny but time-growing parasitic
currents). That belongs to the CSF source term, not to the projection operator,
and is being handled on the balanced-force CSF lane.

## 5. Reproduce

```
PYTHONPATH=src python3 scripts/two_phase_projection_audit.py   # metrics.json + png
PYTHONPATH=src python3 scripts/two_phase_projection_operator_verify.py
PYTHONPATH=src python3 -m pytest tests/test_two_phase_solver_verification.py tests/test_two_phase_projection_operator_matrix.py -q
```
