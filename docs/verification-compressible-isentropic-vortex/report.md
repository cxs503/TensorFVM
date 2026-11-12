# Isentropic vortex (Shu 1998): closed-form envelope calibration — verification-debt closure

**Mission**: m16364 (selfstart · tp-mod-1 models seat) · **Repo**: TensorFVM · 2026-11-12
**Anchor**: `NEW:` closure of the skipped gate `test_exact_solution_is_isentropic_and_bounded`
(no open feature-ledger anchor existed for it; the catalog item "Compressible: Isentropic
vortex (exact solution)" was still `[[gap-status: open]]`).

## 1. Case (physics / geometry / parameters)

The isentropic vortex is an *exact, smooth* solution of the 2-D Euler equations that
translates with the uniform free stream on a periodic box:

| symbol | value | note |
|---|---|---|
| box | 10 x 10, periodic in x and y | ≫ 2 core radii, core at the box centre |
| `gamma` | 1.4 | ideal gas, two identical materials (`Y = 0`) |
| `beta` | 5.0 (Shu value) and 1.0 (gentle control) | swirl strength |
| free stream | `u_inf = 1`, `v_inf = 0` | exact solution = initial field advected |
| grid (envelope check) | 41 x 41 nodes, `dx = 0.25` | core sampled **exactly** (x = 5.0 is on the grid) |
| grid (order study, committed) | 32 / 64 / 128 cells | `study.json`, second order enforced |

Closed form (`src/tensorfvm/compressible/isentropic_vortex.py`):

```
T(r)   = 1 - K e^{1-r^2},      K = (gamma-1) beta^2 / (8 gamma pi^2)
rho(r) = T^{1/(gamma-1)},      p = rho^gamma = T^{gamma/(gamma-1)}      (isentropic)
delta u = -(beta/2pi) Y e^{(1-r^2)/2},  delta v = +(beta/2pi) X e^{(1-r^2)/2}
```

## 2. Method

1. Read the skipped gate: `pytest.skip("analytic envelope assertion not yet calibrated")`,
   removed rather than weakened.
2. Derived the envelope **in closed form** (no fitted table, no hand-typed reference):
   extrema at the core `r = 0` where `e^{1-r^2} = e`; positivity condition `K e < 1`;
   swirl-speed maximum `beta/(2 pi)` from `d/dd[d e^{(1-d^2)/2}] = 0` at `d = 1`.
3. Regenerated every number programmatically:
   `PYTHONPATH=src python docs/verification-compressible-isentropic-vortex/envelope_calibration.py`
   → `envelope.json` (analytic block, measured block, periodic-image block).
4. Replaced the skip with three machine-checked invariants + a new periodic-image
   bound gate; then re-ran the file.

## 3. Results (programmatic, from `envelope.json`)

| beta | K = (γ−1)β²/(8γπ²) | T_min = 1−K·e | rho_min (analytic) | rho_min (measured 41²) | p_min (analytic) | p_min (measured) |
|---|---|---|---|---|---|---|
| 5.0 | 0.090469 | 0.754091 | 0.493807 | 0.493807 | 0.372375 | 0.372375 |
| 1.0 | 0.003619 | 0.990164 | 0.975590 | 0.975590 | 0.965994 | 0.965994 |

| invariant | measured | verdict |
|---|---|---|
| isentropic residual `max|p − rho^1.4|` | 0.0 (bit-exact) | PASS |
| radial monotonicity `min diff(rho)` along sorted r² | 0.0 (non-decreasing) | PASS |
| far-field bounds `rho_max`, `p_max` (both beta) | 1.0, 1.0 | PASS |
| swirl-speed envelope `max u = u_inf + beta/(2π)` | attained (≥ 0.9·envelope) and never exceeded | PASS |
| **neglected periodic-image amplitude** (nearest image, r = 5) | **6.144212e-06** | bounded, < 1e-4 |

Figure: `vortex_field.png` (field) and `order_convergence.png` (observed order) are the
committed artefacts of the parent verification face — the change here adds no new physics
run, so no new field plot is claimed.

## 4. Error judgement vs baseline

| quantity | baseline (closed form) | measured | error | <3%? |
|---|---|---|---|---|
| `rho_min`, beta=5 | 0.493807324 | 0.493807324 | 0.0 (rel.) | **yes** |
| `p_min`, beta=5 | 0.372375018 | 0.372375018 | 0.0 (rel.) | **yes** |
| `rho_min`, beta=1 | 0.975590088 | 0.975590088 | 0.0 (rel.) | **yes** |
| `p_min`, beta=1 | 0.965993782 | 0.965993782 | 0.0 (rel.) | **yes** |
| isentropic identity | `p = rho^1.4` | residual 0.0 | 0.0 (abs.) | **yes** |
| image truncation vs N=32 truncation error (~1e-2) | — | 6.1e-6 / 1e-2 ≈ 6e-4 | — | **yes** (three orders below) |

The extrema match to the last bit because the 41-point grid samples the core exactly; the
envelope is a property of the closed form, so "0.0 %" is the expected result and is stated
as such, not as a solver accuracy claim.

## 5. Notes / limitations (this is the §5 the old gate pointed at)

* The module docstring claimed the single-image representation is "exact to double
  precision" and that the amplitude is "below 1e-11 at r = 5". **Both were wrong**: the
  exact amplitude at r = 5 is `e^{-12} = 6.14e-6`. Corrected in the module docstring and
  now pinned by `test_single_image_representation_is_bounded_by_the_nearest_image`.
  The practical consequence is nil for the order study (the image error is dt-independent
  and 3–4 orders below the truncation error, so it shifts the error *floor* by ~6e-4
  relative, far below the 1.8 order floor), but the claim as written was overstated.
* The same docstring stated `T_min = 0.8825` for beta = 5; the closed form gives
  `T_min = 0.7541` (`rho_min = 0.4938`). The old number corresponds to no term of the
  formula as implemented — erased, not grandfathered.
* Scope: this envelope says nothing about the scheme's accuracy on the grid (that is
  `study.json`), only that the *reference* it is measured against is smooth, bounded,
  positive and single-valued on the sampled grid.
* Still open on this face (unchanged, honestly): the gate measures a periodic, smooth,
  laminar flow — no wall, no shock, no limiter-active regime.

## 6. Reproduction

```bash
PYTHONPATH=src python docs/verification-compressible-isentropic-vortex/envelope_calibration.py
PYTHONPATH=src python -m pytest tests/test_compressible_isentropic_vortex.py -q
PYTHONPATH=src python -m pytest tests/test_verification_catalog_gate.py -q
```