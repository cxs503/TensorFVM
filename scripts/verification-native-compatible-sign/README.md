# Verification: sign convention, absolute scaling and defect boundary of the viscous momentum assemblies

Scope: audit of two merged assemblies that had never been compared numerically.

* `NativeViscous3D.momentum_diffusion_matrix` = `D A G` — the production viscous
  slot used by every native solver (`src/tensorfvm/native_viscous3d.py`).
* `compatible_momentum_operators(...)['matrix']` = `K = G^T 2 mu V G` — the
  dissipative-by-construction assembly added by `native_compatible_viscous3d.py`.

The merged audit `docs/verification-native-compatible-viscous/` (dissipativity
sweep) and `tests/test_native_compatible_viscous3d.py` explicitly reserved three
questions: *absolute scaling*, *how to adopt the operator*, and *where the legacy
defect starts*. This directory answers them with machine-checkable evidence.

Reproduce (no solver run, seconds of CPU):

```
PYTHONPATH=src python3 scripts/audit_compatible_viscous_sign.py \
    --out scripts/verification-native-compatible-sign --refinements 2,4,8
python3 -m pytest tests/test_native_compatible_sign_convention.py -q
```

Outputs: `sweep.json`, `sweep.csv`, `report-table.md` (programmatically injected
tables), `sign-sweep.png`, `energy-identity.png`, `residual-order.png`.

## 1. Physical setting

Two mesh families, both with viscosity `mu`, Dirichlet (no-slip/analytic) data and
no pressure or turbulence:

| family | cells | distortion | use |
|---|---|---|---|
| `two_tetrahedra(skew)` | 2 tetrahedra | skew 0 → 1 forces max non-orthogonality 25.2° → 86.7° | the open-defect counterexample |
| `sheared_cube(n, shear, tilt)` | `n^3` hexahedra | affine shear/tilt, triangulated quad facets | well-shaped control family |

All matrices are dense for the 2-cell case and sparse otherwise; every diagnosis
below is a linear-algebra statement about the assembled operators, so no time
integration and no iterative solver is involved.

## 2. Findings

### F1 — the two assemblies use *opposite* sign conventions (doc defect)

`u^T (D A G) u <= 0` is the dissipative sign in the production slot;
`u^T K u >= 0` for the compatible matrix (PSD by construction). Over 256 random
probes (`sign_probes` in `sweep.json`):

| mesh | probes with `legacy ≤ 0 ≤ compatible` | probes with `compatible ≤ 0 ≤ legacy` |
|---|---|---|
| `two_tetrahedra` skew 0.0 | 64/64 | 0/64 |
| `two_tetrahedra` skew 1.0 | 39/64 (sym M indefinite here — the known defect) | 0/64 |
| `sheared_cube(n=2)` | 64/64 | 0/64 |
| `sheared_cube(n=4, shear .35)` | 64/64 | 0/64 |

So swapping the assembly into a solver's viscous slot *without* a sign change
turns the viscous term into an anti-diffusive term. The merged docstring claimed
the returned matrix has the "same sign convention as
`NativeViscous3D.momentum_diffusion_matrix`"; that claim is wrong and is corrected
here, together with a drop-in helper:

```python
from tensorfvm.native_compatible_viscous3d import compatible_momentum_matrix_in_legacy_sign
M = compatible_momentum_matrix_in_legacy_sign(operators)   # = -K, dissipative in that slot
```

Pinned by `test_compatible_matrix_is_ppsd_where_the_production_slot_is_negative`
and `test_legacy_sign_helper_makes_the_compatible_operator_a_dissipative_drop_in`.

### F2 — absolute scaling is certified (the reserved follow-up): **PASS**

For every mesh/viscosity pair tested, the quadratic form of the compatible
operator equals the volume-weighted discrete strain energy to round-off,
`u^T K u = 2 mu sum_c V_c dev(S_h(c)) : dev(S_h(c))`, i.e. no hidden artificial
viscosity:

| mesh | mu | `u^T K u` | `2 mu sum V dev(S):dev(S)` | relative error |
|---|---|---|---|---|
| `two_tetrahedra` skew 0.0 | 1.0 | +1.42636e+01 | +1.42636e+01 | 0.00e+00 |
| `two_tetrahedra` skew 1.0 | 1.0 | +3.01193e+01 | +3.01193e+01 | ≤ 1e-16 |
| `sheared_cube` n=2 | 1.0 | +1.38254e+01 | +1.38254e+01 | 0.00e+00 |
| `sheared_cube` n=4 shear .35 | 1.0 | +6.53393e+01 | +6.53393e+01 | 2.17e-16 |
| above four | 2.5 | (×2.5 exactly) | (×2.5 exactly) | ≤ 1.8e-16 |

The full table (8 rows) is in `report-table.md`; the identity is plotted in
`energy-identity.png`. Linearity in `mu` is checked separately.

### F3 — where the legacy defect starts: it needs an indefinite symmetric part

`lambda_max(sym(D A G)) > 0` is the necessary condition for the discrete viscous
term to create kinetic energy. Sweeping the two-tetrahedron family
(`sign-sweep.png`, left panel):

| skew | max non-orthogonality | `lambda_max(sym M)` legacy | `lambda_min(K)` compatible |
|---|---|---|---|
| 0.00 | 25.2° | −8.91 | +0.535 |
| 0.75 | 79.9° | −8.09 | +2.857 |
| 0.95 | 85.8° | −1.11 | +3.353 |
| 1.00 | 86.7° | **+7.15** | +3.090 |

The production slot stays negative (semi)definite on the whole sheared-hexahedron
family (`lambda_max` between −0.48 and −5.03 for `shear ≤ 0.5`, `n = 2, 4`), so the
open defect is confined to strongly skewed *tetrahedral* geometry
(non-orthogonality ≳ 85.8° in this family), not to sheared hexahedra. That bounds,
but does not remove, the risk to the production path.

### F4 — residual order against `-div(2 mu S)` (with a scalar control)

Smooth solenoidal field, `h = 1/n` on the sheared cube, residual normalised by cell
volume (`residual-order.png`):

| cells | h | legacy momentum | scalar diffusion (control) | compatible (sign corrected) |
|---|---|---|---|---|
| 8 | 0.500 | 0.1771 | 0.3362 | 0.7305 |
| 64 | 0.250 | 0.1254 | 0.1636 | 0.4274 |
| 512 | 0.125 | 0.1162 | 0.0959 | 0.3076 |

Observed orders are 0.50 / 0.11 (momentum), 1.04 / 0.77 (scalar). The scalar control
shows that this *global* residual is dominated by the one-sided Dirichlet
face-gradient closure (first-order face flux concentrated on the `O(h)` boundary
layer ⇒ `O(h^{1/2})` in L2), so the global residual is not a sharp discriminator
between the two assemblies. What it does establish is that the sign-corrected
compatible operator is *convergent* (0.731 → 0.308) and of the same accuracy class
as the production operator at these resolutions. Pointwise/interior orders are not
resolved at these resolutions and are listed as follow-up.

## 3. Error judgement (vs reference values)

| key quantity | reference | measured | relative error | < 3 % ? |
|---|---|---|---|---|
| `u^T K u` vs `2 mu sum V dev(S):dev(S)` | discrete strain energy (exact algebra) | identical up to round-off | ≤ 2.2e-16 | yes |
| scaling in `mu` (1, 2.5) | linear | linear | ≤ 1.8e-16 | yes |
| sign of `u^T M u` / `u^T K u` | dissipative slot sign | opposite signs | exact, 256/256 probes | yes (exact) |
| legacy `lambda_max` on sheared hexahedra | ≤ 0 (dissipative) | −0.484 … −5.03 | n/a (sign test) | yes |
| momentum residual at n=8 | `-div(2 mu S)` (analytic) | 0.1162 | 11.6 % | **no** — global residual, dominated by boundary closure; not a convergence claim |

## 4. Notes, limitations, follow-ups

* Everything here is a statement about *assembled matrices*; no time stepping, no
  pressure coupling. Solver-level stability was not re-measured.
* The `two_tetrahedra` geometry is not incompressible, so `F2` is asserted for an
  arbitrary field on purpose: the identity is an algebraic property of `K`.
* The legacy defect (`u^T M u > 0` at skew 1) remains open and its strict `xfail`
  in `tests/test_native_energy_small.py` is untouched; this audit only bounds it.
* Follow-up worth a separate ticket: an interior-stencil order study (interior
  cells only, ≥2 cells away from the wall) to determine whether the slow global
  rate is purely boundary closure; and a solver-level swap test of
  `compatible_momentum_matrix_in_legacy_sign` in `native_simple3d`.