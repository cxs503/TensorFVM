# Two-phase CSF: balanced-force surface tension

Unit m3376.  Verification-debt closure for the static-drop chain of
`tensorfvm.two_phase_solver`: the 2D Young-Laplace jump `dp = sigma/R` now lands
inside the 3 % gate instead of being carried as a strict `xfail`.

## What was wrong

The CSF source `sigma kappa grad(alpha)` was evaluated at **cell centres** and
then averaged onto the MAC faces.  The pressure correction, however, is driven
by the **two-point face difference** `(p_R - p_L)/dx`.  A cell-force average has
no representation of that form: its discrete curl does not vanish, so the
projection cannot cancel it and the residual pumps velocity every step.  Measured
on a static drop (n = 32, six steps): `max|u| = 4.9e-5 m/s`, i.e. 7e-6 of the
capillary scale `sigma/mu = 7 m/s`, growing with resolution, and `dp` scattered
between +0.6 % and -30 % over the grid ladder (`legacy` columns of
[`report.md`](report.md)).

## What changed

1. `balanced_csf_face_forces` (`src/tensorfvm/two_phase_vof.py`): the force on a
   face is `sigma * kappa_face * (alpha_R - alpha_L)/dx` -- the **same**
   two-point stencil the pressure gradient uses (Francois et al. 2006,
   *J. Comput. Phys.* 213:141, balanced-force formulation).  For a uniform
   curvature the discrete pressure `p = sigma*kappa*alpha` cancels it exactly.
2. `smooth_alpha` (Brackbill--Kothe--Zemach 1992, *J. Comput. Phys.* 100:335):
   the colour field is convolved with the separable `[1/4,1/2,1/4]` kernel
   before the normals are differentiated, which removes the grid-scale noise
   that `kappa = div n` otherwise amplifies.
3. Both are the solver defaults (`TwoPhaseConfig.csf_balance="balanced"`,
   `csf_smoothing=3`); `csf_balance="legacy"` keeps the retired discretisation
   executable so the improvement stays falsifiable
   (`tests/test_two_phase_csf_balanced.py`).

## Measured (every number written by `scripts/audit_two_phase_csf_balance.py`)

[`report.md`](report.md) carries the full tables, [`summary.json`](summary.json)
the machine-readable form, `csf-balance.png` the figure.  Summary of the verdict
block:

| series | worst &#124;dp/(sigma/R)-1&#124; | passes 3 % gate |
|---|---|---|
| ramp / legacy | 30.2 % | NO (erratic, non-monotone in n) |
| ramp / balanced | 1.9 % | yes |
| tanh / legacy | 5.5 % | NO |
| tanh / balanced | 5.9 % (n=24), 1.4 % (n=48), 0.8 % (n=64) | yes for n >= 48 |

Spurious currents drop from `4.9e-5` to `5.2e-7 m/s` at n = 32 (94x) and the
maximum velocity divergence from `2.4e-13` to `6.6e-16`.

## Honest limits

* The 3 % gate needs `n >= 48` once the interface is *resolved* (`tanh`
  profile); at n = 24 the curvature is still 5.9 % off and the gate fails.  That
  limitation is pinned by `test_resolved_interface_reaches_the_gate_on_fine_grids`
  rather than hidden.
* Curvature is a **second**-order quantity of a first-order interface
  description: `kappa` from a grid-scale ramp keeps a small residual band error,
  which is exactly why smoothing is part of the fix and not cosmetics.
* No convergence order is claimed for `dp`: the series is monotone but the
  asymptotic range was not demonstrated, and no order fit is reported.
* Only 2D, uniform square grids, matched viscosity (`mu_l` = 1e-2 == `mu_g`),
  matched-viscosity static drop, `gravity = 0`.  Bubble rise, capillary-wave
  damping and the VOF advection of a moving interface are out of scope.

## Reproduce

```bash
PYTHONPATH=src python scripts/audit_two_phase_csf_balance.py
PYTHONPATH=src python -m pytest tests/test_two_phase_csf_balanced.py \
    tests/test_two_phase_surface_tension_verification.py -q
```
