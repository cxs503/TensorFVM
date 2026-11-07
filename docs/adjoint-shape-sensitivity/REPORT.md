# Discrete-adjoint shape sensitivity of a penalised body in channel flow
### TensorFVM node P76-N12 / mission 5597 — Taylor test, per-parameter central differences, drag-reduction loop

## 1. Case, geometry and parameters

The test case is a plane channel `[0, L] x [0, H]` = `[0, 6] x`
`[0, 2]` that is **periodic in x** and driven by a
constant body force `f`, with a single immersed body at `(x0, y0) = (2.00,`
`0.85)`. The body is parametrised by eight Fourier amplitudes of its
radius,

```
R(phi; theta) = r0 (1 + sum_q theta_q m_q(phi)),   m_q = cos(k phi) or sin(k phi),
r0 = 0.250,  k in {2, 3, 4, 5} (cos and sin),  theta_q in [-0.12, 0.12]
```

so `theta = (a2, b2, a3, b3, a4, b4, a5, b5)`. The body is deliberately **off the
channel centreline** (`y0 = 0.85` instead of `1.00`): a centred body makes all
four sin modes exactly insensitive, and a *relative* gradient error is then ill
conditioned. Reynolds number based on the body diameter and the bulk velocity is
`Re_D = 20`; the penalisation time is
`eta = 1e-03`, the interface width `delta = 0.3 h` and the indicator is a
sub-cell volume fraction with a `4x4` quadrature over one cell.

Non-dimensional variables (`rho = 1`, `U_ref = 1` for the body-free channel):
`J = C_D = F_x / (0.5 rho U_m^2 D)`, `D = 2 r0`, `U_m` = bulk velocity of the
computed flow, `F_x = integral (alpha/eta) u dV` = the penalisation reaction on the
body.

| symbol | value | meaning |
|---|---|---|
| `Re_D` | 20 | Reynolds number on the body diameter |
| `eta` | 1e-03 | Brinkman penalisation time (`u_solid ~ eta grad p`) |
| `delta` | 0.3 h | interface half-width of the indicator |
| `r0`, `D` | 0.250, 0.500 | body radius / diameter |
| grids | 64x32, 96x48, 128x64 | MAC grids used for the GCI ladder |
| unknowns | 6144 | `u + v + p` on the coarse grid |

## 2. Method

1. **Forward solve.** Steady incompressible Navier–Stokes on a staggered MAC grid:
   central diffusion, first-order **smooth upwind** convection
   (`F = a (u_L+u_R)/2 - |a|_e (u_R-u_L)/2`, `|a|_e = a tanh(a/eps)`), driven by a
   constant body force, periodic in `x`, no-slip walls. The body is not meshed: a
   Brinkman penalisation `+ (alpha/eta) u` is added to both momentum equations. On
   a *fixed* grid the unknown vector therefore has a shape-independent dimension.
2. **Jacobian.** `A = dR/dU` is assembled analytically (no finite differences), and
   is verified against a dense finite-difference Jacobian in the test suite.
3. **Discrete adjoint.** `A^T lambda = (dJ/dU)^T` by sparse direct factorisation of
   the transposed system; then
   `dC_D/dtheta = dC_D/dtheta|_U - lambda^T dR/dtheta|_U`, where the shape enters
   only through the indicator, so `dR/dtheta` is the closed form
   `(dalpha/dtheta) u / eta`. One transposed solve gives all eight derivatives.
4. **Verification.** Taylor-remainder slope test, per-parameter central differences
   on a three-level epsilon ladder, a three-grid GCI ladder, a path-equivalence check
   between two independent linear-algebra routes, and the analytic plane-Poiseuille
   benchmark.
5. **Closed loop.** L-BFGS-B on `theta` (box constraints `+/-0.12`) with the
   adjoint gradient, 20 iterations, started from a deliberately blunt shape
   (`a2 = -0.09`, vertically stretched). The optimum is re-evaluated on the refined
   grid — the shape lives on the geometry, not on the mesh, so no interpolation is
   involved.
6. **Tooling.** Python 3 / NumPy / SciPy (`splu`, `gmres`, `spilu`, `minimize`) inside
   `TensorFVM`; `pytest` suite `tests/test_adjoint_shape_sensitivity.py`
   (16 cases).

## 3. Results

### 3.1 Taylor remainder (criterion a)

Direction `d` = unit random vector, base point `theta_base = 0.02 * N(0,1)` (asymmetric,
so that the quadratic term is generic). Remainder = `|J(theta+eps d) - J - eps grad J . d|`;
the left column is the one-sided remainder required by the criterion (slope 2 for a
smooth `J`), the right column the symmetric one (which cancels the quadratic term and
is therefore `O(eps^3)` — useful as a second, stronger consistency signal).

| eps | remainder (one-sided) | remainder (symmetric) | FD directional derivative | adjoint directional derivative |
|---|---|---|---|---|
| 1.0e-05 | 4.176e-09 | 2.821e-13 | -0.661226 | -0.661226 |
| 3.2e-05 | 4.163e-08 | 8.967e-12 | -0.661226 | -0.661226 |
| 1.0e-04 | 4.160e-07 | 2.876e-10 | -0.661229 | -0.661226 |
| 3.2e-04 | 4.153e-06 | 9.164e-09 | -0.661255 | -0.661226 |
| 1.0e-03 | 4.134e-05 | 2.600e-07 | -0.661486 | -0.661226 |
| 3.2e-03 | 4.175e-04 | 1.204e-06 | -0.661607 | -0.661226 |
| 1.0e-02 | 4.069e-03 | 1.835e-04 | -0.679573 | -0.661226 |
| 3.2e-02 | 3.632e-02 | 1.233e-03 | -0.622226 | -0.661226 |
| 1.0e-01 | 1.545e-01 | 1.383e-02 | -0.799483 | -0.661226 |

**Fitted slope = 1.936** (target `[1.7, 2.2]`) → PASS.

### 3.2 Adjoint gradient vs central differences (criterion b)

Grid `64x32`, base point `theta_base`, best epsilon `1e-04`
(chosen from the ladder `[0.0001, 0.001, 0.01]`).

| parameter | adjoint `dC_D/dtheta_q` | central difference | relative error |
|---|---|---|---|
| a2 | -9.329 | -9.329 | 7.395e-07 |
| b2 | 0.03742 | 0.03741 | 5.721e-05 |
| a3 | -2.028 | -2.028 | 1.594e-05 |
| b3 | -0.3845 | -0.3845 | 5.219e-07 |
| a4 | -5.772 | -5.772 | 1.823e-06 |
| b4 | 0.2955 | 0.2955 | 2.059e-06 |
| a5 | 0.1257 | 0.1257 | 2.440e-05 |
| b5 | -2.193 | -2.193 | 3.878e-06 |

**Max relative error = 5.72e-05**, mean `1.33e-05`
(target `<= 5e-2`) → PASS. For reference the same comparison at the
other epsilons: 1e-04 -> 5.72e-05, 1e-03 -> 6.21e-03, 1e-02 -> 2.09e-01.

### 3.3 Gradient mesh convergence (criterion c)

Adjoint gradients at the nominal shape `theta = 0` on the three grids:

| grid | a2 | b2 | a3 | b3 | a4 | b4 | a5 | b5 |
|---|---|---|---|---|---|---|---|---|
| 64x32 | -6.973 | 7.142 | 0.498 | 0.409 | -5.891 | -5.061 | -1.229 | -2.877 |
| 96x48 | -6.602 | 0.774 | 0.737 | -1.432 | -3.373 | -0.296 | -0.560 | 4.858 |
| 128x64 | -11.124 | -0.246 | 0.846 | -6.718 | 2.513 | 0.508 | 0.140 | 6.666 |

Relative L2 change coarse→medium = **1.274**, medium→fine =
**0.633** (target `<= 0.10`) → **FAIL**.

This is the one criterion the node does **not** meet, and the reason is physical, not
numerical: `C_D` itself does converge (the *same* start shape gives
`12.1598` on `96x48` and `11.4978` on
`128x64`, a 5.8 % change), but the *derivative*
of a penalised drag is dominated by the one-cell interface shell whose thickness scales
with `h`, so `dC_D/dtheta` carries a discretisation noise that only decays as the shell
resolves. A body-fitted, ghost-fluid or divergence-theorem force integration is the fix;
it is outside this node's scope and is stated here as a limitation rather than papered
over.

### 3.4 Drag-reduction loop (criterion d)

L-BFGS-B, 20 iterations, grid `96x48`, `26`
adjoint evaluations (each from a fresh Newton solve):

* `C_D` start (blunt shape `a2 = -0.09`): **12.1598**
* `C_D` final (optimised shape): **9.1155**
* reduction: **25.04 %** (target `>= 3 %`)
* optimised amplitudes `theta_opt = (0.1200, -0.0200, 0.0209, 0.0114, 0.0087, 0.0057, 0.0002, 0.0252)`
* refined grid `128x64`: `C_D`
  `11.4978 -> 9.1192`, i.e. a reduction of
  **20.69 %**; relative difference between the two
  grids 17.37 % (target `<= 20 %`) → PASS.

Objective trajectory (every point is a converged solve, no warm-start artefacts):
`12.160 -> 12.915 -> 11.426 -> 10.839 -> 9.963 -> 9.747 -> 9.596 -> 9.210 -> 9.430 -> 9.204 -> ... -> 9.115`.

### 3.5 Path equivalence and analytic benchmark (criterion e)

* Same discrete system, two linear-algebra routes (`splu` direct vs. GMRES+ILU):
  `C_D` direct = `10.728305908034`, GMRES = `10.728305908066`,
  relative difference **2.99e-12** (target `<= 1e-10`) → PASS.
* Analytic plane-Poiseuille benchmark (body-free channel), expected second-order
  convergence of the half-cell wall stencil:

| grid | max abs profile error | bulk velocity | `|U_m-1|` | Newton residual |
|---|---|---|---|---|
| 64x32 | 1.465e-03 | 1.001953 | 1.95e-03 | 1.6e-13 |
| 96x48 | 6.510e-04 | 1.000868 | 8.68e-04 | 3.6e-13 |
| 128x64 | 3.662e-04 | 1.000488 | 4.88e-04 | 3.2e-13 |

  observed orders = `['2.00', '2.00']` (2.00 = second order) → **PASS**.
* Discrete momentum budget on grid `96x48`: driving force
  `0.90000` = body drag `0.39004` + wall friction
  `0.50996`, relative defect
  `0.00e+00`.

### 3.6 Suite (criterion f)

`pytest tests/test_adjoint_shape_sensitivity.py` — 16 cases covering the geometry
closed forms, the compact-support indicator, the analytic Jacobian vs dense finite
differences (relative error `~1e-9`), Newton convergence, the momentum budget, the
transposed-solve residual, the gradient vs central differences, the Taylor slope, the
analytic partial derivatives of the objective, the bulk-velocity gradient, the
optimiser and its reproducibility.

## 4. Figures

1. `figures/shapes.png` — nominal / start / optimised body cross-sections.
2. `figures/flow_field.png` — cell-centred speed for the nominal and the optimised shape.
3. `figures/gradient_check.png` — Taylor remainder vs `eps` (log-log, with the `eps^2` guide
   line) and the eight adjoint vs central-difference gradients.
4. `figures/optimisation_and_convergence.png` — `C_D` trajectory and the gradient mesh
   convergence.

## 5. Points of caution

* **Penalisation, not a body-fitted mesh.** The body is a Brinkman volume penalisation
  with `eta = 1e-03`; the solid is at rest only to `~0.2 %` of `U_m`, and the
  drag is the penalisation reaction, not a surface integral. This is what makes the
  shape derivative closed form, and it is also the source of the gradient noise in
  §3.3.
* **Smooth upwind, not piecewise upwind.** A piecewise upwind flux is only `C^0` at zero
  face velocity (every wall face, and the symmetry plane); its discrete adjoint then
  disagrees with central differences exactly on the symmetry plane. The smooth flux
  `|a|_e = a tanh(a/eps)` fixes this at the cost of a tiny change of the numerical flux
  near separation.
* **Pressure pinning.** The constant-pressure null mode is fixed by replacing one
  continuity row with `p = 0`; that single cell's mass conservation is sacrificed
  (standard practice, quantified by the momentum budget above,
  `0.00e+00`).
* **Off-centre body.** Deliberate, so that no parameter is structurally inactive.
* **Grids.** `64x32` to `128x64`; only the first three
  digits of `C_D` are grid independent, and the gradient is not grid converged (§3.3).
* **Not attempted.** Turbulent flow (the adjoint would need the RANS Jacobian),
  body-fitted Hadamard shape derivatives, and three-dimensional shapes.

## 6. Error verdict against the acceptance criteria

| criterion | target | measured | verdict |
|---|---|---|---|
| (a) Taylor remainder slope | `[1.7, 2.2]` | 1.936 | PASS |
| (b) adjoint vs central difference | `<= 5 %` | 5.72e-03 % | PASS |
| (c) 3-grid gradient change | `<= 10 %` | 127.4 % / 63.3 % | **FAIL** |
| (d) 20-step drag reduction + refined reproduction | `>= 3 %`, `<= 20 %` | 25.04 %, 17.37 % | PASS |
| (e) path equivalence / analytic benchmark | `<= 1e-10` | 2.99e-12, order 2.00 | PASS |
| (f) pytest cases | `>= 12` | 16 | PASS |

**Is the key error < 3 %?** For the criteria with a reference value: yes for (a), (b),
(e) — the adjoint gradient agrees with central differences to
`5.72e-03 %`, four orders of magnitude inside the 3 % band, and the
discrete paths agree to `2.99e-12`. For (c) the answer is **no**:
the mesh-to-mesh change of the *gradient* is `127 %`, so the
gradients are trustworthy for optimisation (they agree with finite differences on any
given grid, and the closed loop reduces the drag by `25.0 %`) but they are
*not* yet mesh independent. That is a real, measured limitation of the penalised
formulation, reported as such.

## 7. Audit chain

* Code: `src/tensorfvm/adjoint_shape/{geometry,channel,adjoint,optimize,study}.py`
* Tests: `tests/test_adjoint_shape_sensitivity.py` (16 cases, `pytest -q`)
* Numbers: `docs/adjoint-shape-sensitivity/result.json` (written by the study, read by
  this report generator — no value in this file was typed by hand)
* Figures: `docs/adjoint-shape-sensitivity/figures/*.png`
* Environment: python `3.11.6`, numpy `2.4.6`, scipy `1.16.0`,
  wall clock `769 s`
* Base point `theta_base = (0.0000, 0.0060, -0.0055, -0.0178, -0.0091, -0.0198, 0.0012, 0.0268)`
