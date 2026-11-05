# Overset as a fluid solver, first unit: coupled Stokes against an exact solution

## 1. The debt this note repays

`docs/fvm-accuracy-development/README.md` acceptance item 4 says, in the
repository's own words: *"Current 3D overset results verify interpolation, not a
fluid solver."* `src/tensorfvm/overset_poisson.py` states the same limit inside
its own metrics as `navier_stokes_verified: False`, and
`docs/verification-analytic-reference-residuals/report.md` §4 lists the
un-audited reference fields. So the composite mesh had been certified as
(i) an interpolation device and (ii) a scalar Laplacian — never as a **coupled
velocity–pressure operator**.

This note adds the smallest honest unit of that missing capability in 2D: the
steady **Stokes** saddle-point system on the same composite (Cartesian
background + body-fitted annular component + one fringe layer of donor
interpolation), with the pressure coupled across the overlap.

## 2. Method

Assembled unknowns, three per cell (`u`, `v`, `p`) for every cell of every grid:

| block | equation |
|---|---|
| active cell, `u`/`v` | `-mu*lap(u) + grad(p) = f`, face conductances `|S|/dist` (same operator as the certified scalar path) |
| active cell, `p` | continuity `sum_f (u_f . S_f) = 0`, `u_f = (u_P+u_N)/2` internally, exact normal velocity on physical faces |
| fringe cell | donor-interpolation rows for **u, v and p** |
| hole cell | identity row, nothing claimed |

One pressure gauge row (replaces one continuity row) removes the constant-shift
null space. The face flux additionally carries the compact Rhie–Chow
correction with `d_f = V/aP` and the cell pressure gradient rebuilt from the same
face lists; without it the collocated system is **exactly singular** (measured,
see §5).

Reference field, generated in SymPy with analytic derivatives (never
hand-differentiated):

```
psi = A (x^2-a^2)(y^2-a^2)(x^2+y^2-R^2),  u = dpsi/dy, v = -dpsi/dx, p = B x y
f = -mu*lap(u) + grad(p)
```

`psi` vanishes on the outer square *and* on the inner circle, so the exact
velocity has zero normal component on every physical boundary — the discrete
continuity rows are then compatible with the prescribed data instead of fighting
it. Two identities are proved at import and asserted in the test suite:
`div(u) == 0` and the Stokes residual `== 0`.

Reproduce:

```bash
PYTHONPATH=src python -m pytest tests/test_overset_stokes.py -q
PYTHONPATH=src python scripts/verify_overset_stokes.py     # writes runs/overset_stokes/
```

## 3. Results (program-generated, zero hand-copied numbers)

| n | unknowns | `u` rel. L2 | `v` rel. L2 | velocity rel. L2 | `p` rel. L2 | Σ|div| defect (rel.) | algebraic residual | donor residual |
|---|---|---|---|---|---|---|---|---|
| 32 | 4,608 | 7.61e-03 | 7.34e-03 | 7.48e-03 | 4.78e-01 | 1.93e-03 | 2.0e-15 | 3.3e-14 |
| 48 | 10,368 | 3.59e-03 | 3.48e-03 | 3.53e-03 | 4.13e-01 | — | 2.8e-15 | 7.6e-14 |
| 64 | 18,432 | 2.19e-03 | 2.15e-03 | 2.17e-03 | 3.45e-01 | 1.27e-03 | 4.1e-15 | 1.3e-13 |

Observed orders (n = 32→48→64): `u` 1.08, 0.72; `v` 1.08, 0.69; `p` 0.21, 0.26.

## 4. Error judgement (per quantity, against the exact solution)

| quantity | n=32 | n=64 | < 3 % ? |
|---|---|---|---|
| velocity `u` | 0.76 % | 0.22 % | **yes** |
| velocity `v` | 0.73 % | 0.21 % | **yes** |
| pressure `p` | 47.8 % | 34.5 % | **NO — honest FAIL** |
| discrete continuity (rel.) | 0.19 % | 0.13 % | yes |
| donor interpolation residual | 3.3e-14 | 1.3e-13 | yes (exact rows) |

## 5. What this does and does not establish

* **The collocated route is not free.** A first assembly without the Rhie–Chow
  term produced `MatrixRankWarning: Matrix is exactly singular` and a non-finite
  solution at every level: the checkerboard mode of the collocated
  gradient/divergence pair is in the null space, so the system is rank deficient
  *and* inconsistent (`navier_stokes_verified` could never have been earned by
  simply adding momentum rows). The Rhie–Chow face flux reproduces a solvable
  system — that is a measured requirement, not a preference.
* **A real assembly defect was found and fixed.** `_faces` registers each face
  once (single owner), so a cell that is never a face owner — the box-diagonal
  corner cell — got an *empty* continuity row in the first version. The fix
  books every face from both sides. `test_overset_stokes.py` now fails if any
  active cell loses its continuity row.
* **Velocity passes, pressure does not.** `u`, `v` and the discrete continuity
  defect are inside the 3 % band and shrink, but the pressure L2 error stays at
  35–48 % with observed order ≈ 0.2. That is recorded as
  `test_pressure_error_stays_an_open_defect_not_a_pass`, not smoothed over. The
  likely cause is the first-order-like pressure coupling of the compact
  Rhie–Chow weight used here (`d_f = V/aP` with a face-list gradient) rather than
  a pressure equation consistent with the momentum diagonal.
* **Scope.** Linear Stokes, 2D, manufactured data on *all* boundaries including
  the polygonal inner wall, point-value donor constraints. Not convection, not
  3D, not turbulence, not a no-slip physical case; `strict_local_conservation`
  stays `False`.
