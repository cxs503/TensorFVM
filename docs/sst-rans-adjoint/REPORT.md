# k-omega SST RANS Adjoint: Gradient Verification, Grid Convergence and Frozen-Turbulence Bias

**TensorFVM — mission p97w3-n06 (m9478), branch `agent/m9478`**

Module: `src/tensorfvm/adjoint_shape/sst_rans.py`,
`src/tensorfvm/adjoint_shape/sst_study.py`,
`src/tensorfvm/adjoint_shape/sst_figures.py`; tests:
`tests/test_sst_rans_adjoint.py`. All numbers below are injected from
`docs/sst-rans-adjoint/result.json` by `sst_report.py`.

---

## 0. Verdict first (honest status)

| criterion | target | measured | status |
|---|---|---|---|
| C1 adjoint vs central difference (8 params, **SST-`nu_t`** closure) | < 1e-3 | 6.26e-11 | PASS |
| C1 adjoint vs central difference (laminar reference) | < 1e-3 | 1.30e-10 | PASS |
| C2 adjoint-solve residual | < 1e-10 | 2.21e-16 | PASS |
| C3 frozen-turbulence bias | < 5 % | 62.06% | FAIL |
| C4 grid-convergence order ($C_D$ / gradient norm) | >= 1.5 | 0.40 / -1.08 | FAIL |
| C5 pytest | >= 10 cases green | 15 cases | see `tests/test_sst_rans_adjoint.py` |
| fully coupled k-omega SST transport steady state | required by the node | rel. residual plateau 2.6e-06, not converged | **FAIL (honest)** |

The **fully coupled** k-omega SST transport solve (k and omega transported
together with momentum) did **not** reach a steady state on any configuration
tried (Re = 1e3 and 1e4; Newton and segregated Gauss-Seidel).  The turbulence
variables collapse onto the trivial laminar fixed point (k -> 0, omega -> 0),
which is an exact solution of the homogeneous part of the model; the coupled
steady state is therefore *not available* here and every SST statement below is
made for the **prescribed-`nu_t(theta)` closure built from the SST formula**, not
for the transported model.  This is reported as an honest FAIL, not worked
around.

---

## 1. Case introduction (physics, geometry, parameters)

Steady, incompressible, streamwise-periodic channel (`[0, 6.0] x [0, 2.0]`) with
a body that is **not meshed**: it enters as a Brinkman volume penalisation
`+ alpha/eta u` with a compact-support smoothstep indicator `alpha(x; theta)`.
The body is a star-shaped radial Fourier curve

    R(phi; theta) = r0 (1 + sum_q theta_q mode_q(phi)),  q = 1..8,
    mode_q in {cos k phi, sin k phi},  k = 2,3,4,5,

so there are **eight** shape parameters.  k >= 2 modes are used on purpose: the
k = 1 pair is a rigid translation, whose drag derivative is flat for a
symmetric channel and would make a relative-error test ill conditioned.

| parameter | value |
|---|---|
| grid (main case) | 10 x 6 MAC (staggered) |
| domain | 6.0 x 2.0 |
| body centre / radius | [2.0, 0.9] / r0 = 0.25 |
| Brinkman time scale eta | 0.001 |
| Reynolds number (D = 2 r0, bulk velocity) | 1000.0 |
| kinematics viscosity nu | 5.000e-04 |
| driving body force | 1.500e-03 |
| shape parameters | 8 (k = 2..5, cos/sin), base = +0.0001, +0.0149, -0.0137, -0.0445, -0.0227, -0.0496, +0.0030, +0.0670 |

Two closures are run:

* **laminar** (`nu_t = 0`) — the reference that validates the machinery chain;
* **SST-`nu_t`** — the k-omega SST eddy viscosity evaluated on a *prescribed*
  (k, omega) profile anchored to the body surface,

      nu_t = a1 k / max(a1 omega, F2 S),   F2 = tanh(arg2^2),
      arg2 = max(2 sqrt(k)/(beta* omega d), 500 nu/(d^2 omega)),

  with `d = min(y_wall, |g(theta)|)` the distance to the nearest solid.  Because
  `d` depends on `theta`, `nu_t = nu_t(theta)` and `d nu_t / d theta != 0` —
  the term the frozen-turbulence approximation drops.  (Field: figure 5,
  `nu_t/nu` max = 0.9.)

---

## 2. How it was done

### 2.1 Discretisation
Identical staggered MAC layout, smooth-upwind convection (`a tanh(a/eps)`
instead of `|a|`, so the flux is C-infinity where the normal velocity vanishes),
central diffusion with a face-averaged `nu + nu_t`, half-cell wall treatment, and
`J = C_D = F_x / (0.5 rho U_m^2 D)` with `F_x` the penalisation reaction.  The
SST min/max switches are smoothed with a 1e-6 relative width so the Jacobian is
continuous (a tie in `max()` is otherwise a kink where a derivative and a
central difference disagree by O(1)).

### 2.2 Exact discrete Jacobian
The residual is written once as `R(U, theta)` in `torch`; the discrete Jacobian
`A = dR/dU` is obtained by **coloured reverse-mode AD**: a colour is
`(field, i mod 5, j mod 5)`, one reverse pass per colour (125 colours for the
5-field layout, 75 for the 3-field one) gives `J^T g_c` for the whole grid, and
because the structural support (radius-2 box over every field pair) is symmetric
and two cells sharing a colour are five apart, `J[r, k]` is read off without
aliasing.  The assembly is verified against a dense central-difference Jacobian
in the test suite (relative error < 1e-8 laminar, < 1e-4 coupled SST, the latter
limited by the difference truncation), and `nx` must be a multiple of the colour
period (the periodic wrap otherwise aliases cell 0 with cell nx-1 — this bug was
caught by exactly that test).

### 2.3 Discrete adjoint and its modes
`A^T lambda = dJ/dU`, then `dJ/dtheta = dJ/dtheta|_U - lambda^T dR/dtheta|_U`.
Three modes select what is differentiated:

* `full` — everything, including `d nu_t / d theta`;
* `frozen_nut` — `nu_t` detached (classic frozen turbulence);
* `frozen_nut_diss` — additionally detaching the upwind dissipation coefficient
  `a tanh(a/eps)`, i.e. the term a *continuous* adjoint does not see.

The frozen modes are a one-line `detach()` because the closure is written once.

### 2.4 Reference gradient
Central differences of the **re-converged** solve at `eps = 0.0001 / 5e-05` and `eps/2`,
combined by Richardson `(4 g(eps/2) - g(eps))/3`; each solve is driven to
`|R|_2/|R0|_2 < 1e-12` by damped Newton with the exact Jacobian.

---

## 3. Results

### 3.1 Adjoint vs central differences (figure 1)

| q | full-diff adjoint | FD + Richardson | rel. err |
|---|---|---|---|
| q | adjoint | FD+Richardson | rel. err |
|---|---|---|---|
| 1 | -9.686760e-01 | -9.686760e-01 | 1.65e-11 |
| 2 | -6.234583e-01 | -6.234583e-01 | 4.95e-11 |
| 3 | +6.771523e-01 | +6.771523e-01 | 7.30e-12 |
| 4 | +6.891279e-01 | +6.891279e-01 | 2.51e-11 |
| 5 | -5.235815e-01 | -5.235815e-01 | 6.26e-11 |
| 6 | -8.142610e-01 | -8.142610e-01 | 4.86e-11 |
| 7 | +1.079104e-01 | +1.079104e-01 | 3.91e-11 |
| 8 | +1.024912e+00 | +1.024912e+00 | 1.56e-12 |

laminar reference: `C_D = 5.61103976`, `|R|/|R0| = 5.9e-18`,
adjoint-solve residual `1.40e-16`, max rel. err vs Richardson
**1.30e-10**.

SST-`nu_t` closure: `C_D = 8.32688108`, `|R|/|R0| = 4.3e-17`,
adjoint residual (full) `2.21e-16`, adjoint residual (frozen)
`2.21e-16`, max rel. err **6.26e-11**.

The two-step FD change for the frozen problem is 4.8e-07, i.e. the
Richardson extrapolation is resolving the gradient to well below the adjoint
comparison threshold.

### 3.2 Frozen-turbulence bias (figure 2)

Frozen vs full-differential adjoint, per parameter:

| q | full | frozen | bias |
|---|---|---|---|
| q | full | frozen | bias |
|---|---|---|---|
| 1 | -9.686760e-01 | -9.686799e-01 | 0.00% |
| 2 | -6.234583e-01 | -5.745666e-01 | 7.84% |
| 3 | +6.771523e-01 | +6.836208e-01 | 0.96% |
| 4 | +6.891279e-01 | +6.380796e-01 | 7.41% |
| 5 | -5.235815e-01 | -5.599985e-01 | 6.96% |
| 6 | -8.142610e-01 | -8.172887e-01 | 0.37% |
| 7 | +1.079104e-01 | +4.094090e-02 | 62.06% |
| 8 | +1.024912e+00 | +9.597097e-01 | 6.36% |

max bias **62.06%**, mean **11.49%**.

Three independent numbers bracket this:

* adjoint-frozen vs adjoint-full (the model difference): **62.06%**;
* adjoint-frozen vs the central difference of the re-converged system
  (an FD-based estimate of the same difference): **62.06%**;
* the frozen *operator* itself is exact: with `nu_t` held as a constant field the
  adjoint reproduces the central difference to **7.8e-10** (test
  `test_frozen_adjoint_matches_fd_with_fixed_nut`, adjoint residual
  2.2e-16).

So the frozen gradient is a correct gradient of the *wrong* objective: dropping
`d nu_t / d theta` moves the drag gradient by up to 62.1% on this closure.

### 3.3 Grid convergence (figure 3)

| nx x ny | h | C_D | \|dC_D/dtheta\| | adjoint residual |
|---|---|---|---|---|
| grid | h | C_D | grad norm | adj. residual |
|---|---|---|---|---|
| 10 x 6 | 0.6000 | 8.32688108 | 2.063907e+00 | 2.2e-16 |
| 20 x 10 | 0.3000 | 10.16851788 | 2.113004e+01 | 1.6e-16 |
| 40 x 20 | 0.1500 | 4.76911553 | 9.200190e+00 | 4.9e-16 |

Monotone series: **False**.  The apparent orders (**0.40** for `C_D`,
**-1.08** for the gradient norm) are therefore *meaningless* when the flag is
False: the three grids are not in the asymptotic range, because the prescribed
`nu_t` profile has a fixed physical length scale (0.25) while the Brinkman
interface width scales with the mesh, so the discrete problems differ by more
than `h`.  (A `nu_t` length tied to `h`, or a body-fitted mesh, is required for a
meaningful order study.)  Richardson estimates, for reference, are
`C_D^(ex) ~ 26.970067` and `|grad|^(ex) ~ -1.5347e+00`.
`C_D` changes by 53.10% between the two finest grids; the gradient norm
by 56.46%.

### 3.4 Coupled SST transport solve (figure 4, honest failure)

| Re | solver | converged | rel. residual | k_max |
|---|---|---|---|---|
| 1e+03 | newton | False | 8.52e-07 | 7.7e-04 |
| 1e+04 | segregated | False | 2.58e-06 | 1.1e-04 |

Both solvers drive the turbulence to the trivial state (`k_max ~ 7.7e-04`,
`omega` -> 0) which is an exact solution of the uncoupled parts of the model;
the residual therefore stalls instead of converging.  Removing the SST
cross-diffusion term from the omega equation (kept in the `F1` blending
argument) did not change the outcome.  Consequently `dC_D/dtheta` for the
**transported** model, and therefore the frozen-vs-full bias *of the coupled
model*, cannot be certified here.

---

## 4. Caveats, traps, scope

1. **Scope of the SST claim.** All SST numbers use the SST *closure* on a
   prescribed profile; the transported k/omega system is not solved. The bias
   measured in 3.2 is the bias of that closure, not of a converged RANS field.
2. **Trivial fixed point.** In a streamwise-periodic domain there is no inflow
   boundary condition to keep `k` and `omega` at a finite level; the laminar
   (`k = 0`) state is an exact solution of the homogeneous turbulence system, and
   both solvers fall into it.  A defect-correction/continuation formulation or an
   inflow patch is needed before the coupled model can be certified.
3. **Colouring precondition.** `nx` must be a multiple of the colour period (5);
   the periodic wrap silently aliases otherwise, and the dense-FD Jacobian test
   is the guard.
4. **Brinkman smearing.** `C_D` is mesh dependent at O(h^0.4) because the
   interface width scales with `h`; the gradient inherits that.
5. **Smoothing of the SST switches** introduces a 1e-6 relative model
   perturbation; without it the `max()` kinks would make an FD reference
   meaningless.

## 5. Error judgement (explicit)

* adjoint vs central difference, 8 parameters, SST-`nu_t` closure:
  **6.26e-11** -> below the 1e-3 acceptance line;
* adjoint vs central difference, laminar: **1.30e-10** -> below;
* `C_D` vs a Richardson-extrapolated grid limit: 53.10% between the two
  finest grids, i.e. **above** 3 % -- but the series is not monotone
  (False=False means the order study itself is not valid);
* frozen-turbulence bias: **62.06%** -> above 5 %;
* coupled SST steady state: **not available**, so no `C_D` accuracy statement
  can be made for the transported model (honest FAIL).

## 6. Reproduction

```
python -m pytest tests/test_sst_rans_adjoint.py -q      # C5
PYTHONPATH=src python -c "from tensorfvm.adjoint_shape.sst_study import main; main()"
PYTHONPATH=src python -c "from tensorfvm.adjoint_shape.sst_figures import main; main()"
PYTHONPATH=src python -c "from tensorfvm.adjoint_shape.sst_report import main; main()"
```
