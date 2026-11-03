# Periodic MAC kernel: temporal order of accuracy (verification debt closure)

Status: **closed for the smooth periodic regime** — the two open points of the
published `docs/periodic-mac/study.json` are now measured.

| quantity | published before | measured here |
|---|---|---|
| face-velocity temporal order | 3-level ratio against a fixed reference (floor-contaminated) | **2.0002** (n=16), **2.0000** (n=32) |
| projection-pressure temporal order | *never measured* | **1.0125** (n=16), **1.0063** (n=32) |
| physical accuracy of the TG decay | `physical_accuracy_qualified = False` | velocity **2.56e-3** (n=16) → **6.42e-4** (n=32); energy **5.13e-3** → **1.28e-3**, both < 3% |

## Why a new measurement was needed

`scripts/run_periodic_mac.py` compares the discrete solution with a reference
field that is *itself* wrong by a `dt`-independent amount: the O(h²) mismatch
between the discrete convective divergence and a discrete gradient of the
Taylor-Green mode. Once that floor dominates, `log2(e(dt)/e(dt/2))` is a floor
ratio, not a temporal order. `estimator_analytic_check` pins the algebra: for
`E(dt) = dt^p + floor` the naive ratio drifts to zero slope (1.996 → 1.933 in
the synthetic example), while differencing two solutions on the *same* grid
cancels the floor exactly (`E_k - E_{k+1} = dt^p (1 - 2^-p)`) and returns `p`
(2.000 here).

## Method

* Exact solution: Taylor-Green vortex, an exact Navier-Stokes solution on
  `[0,2π)³`, `u = (A sin x cos y, -A cos x sin y, 0) e^{-2νt}`,
  `p = A²/4 (cos 2x + cos 2y) e^{-4νt}`. `A = 1`, `ν = 0.1 Pa·s`, `T = 1 s`.
* Self-convergence on a fixed grid, four `dt` levels per grid, Richardson fit
  (least squares on `log(difference)` vs level index, slope divided by `log 2`).
* The FACES and the CENTRES need different abscissae: `face_coordinates(axis)`
  shifts exactly one axis by `h/2`, so `u_x` is sampled on x-faces with y at
  centres and `u_y` on y-faces with x at centres. Mixing them breaks the
  discrete divergence-free property and `step` raises (found the hard way).
* Pressure is compared after removing its spatial mean (the projection
  potential has a gauge constant).
* Negative control: the production projection is kept and only the trapezoidal
  momentum averaging is replaced by forward Euler. A tool that cannot see this
  planted first-order defect cannot certify anything.
* `dt` is halved per grid doubling so `dt/h` and the convective CFL number are
  fixed across grids. At n=32 with `dt = 0.2 s` the CFL number is ≈1 and the
  Picard iteration of the midpoint step fails (residual NaN) — that run is
  reported as a rejection, not as a result.

## Measured

| n | dt series [s] | velocity order | pressure order | Euler control | `u` floor vs exact | `K` floor vs exact |
|---|---|---|---|---|---|---|
| 16 | 0.2, 0.1, 0.05, 0.025 | 2.0002 | 1.0125 | 1.0212 | 2.560e-3 | 5.126e-3 |
| 32 | 0.1, 0.05, 0.025, 0.0125 | 2.0000 | 1.0063 | 1.0106 | 6.418e-4 | 1.284e-3 |

The error floors fall by 4.0× per grid doubling (`h²`), which is what makes them
spatial rather than temporal; the two-level orders per grid are
`2.0003 / 2.0001` and `2.0001 / 2.0000` for velocity, `1.0176 / 1.0087` and
`1.0084 / 1.0042` for pressure.

## Finding and limitation

The implicit-midpoint momentum update really is second order, but the
**projection pressure is first order** — about one order below the velocity, as
is standard for a pressure-correction step whose pressure carries no time
extrapolation. Consequences to state plainly:

* any quantity obtained from `∇p` (e.g. the discrete momentum residual, a
  pressure-derived force) is only O(`dt`) accurate at fixed grid;
* upgrading to second-order pressure requires a pressure extrapolation term
  (second-order pressure-correction), which is *not* implemented here;
* the pressure error is also bounded below by an O(h²) floor (≈2.8e-2 at n=16,
  ≈5.9e-3 at n=32) from the same convective imbalance.

Scope: smooth, laminar, single-mode, uniform-periodic box, single rank, CPU.
Nothing here certifies wall-bounded, turbulent or sharp-gradient regimes, and
the cylinder/FSI backends are explicitly out of scope for this prototype.

## Reproduce

```bash
TENSORFVM_EVIDENCE_ROOT=. PYTHONPATH=src python scripts/run_mac_time_order.py --figures
PYTHONPATH=src python -m pytest tests/test_mac_time_order.py -q
```

Artefacts: `study.json`, `temporal-order.png`, `error-floor.png`, `fields.png`.