# Spalding all-y-plus wall law: literature audit + two repaired defects

Module under test: `src/tensorfvm/wall_functions.py` (constitutive core also used by
`native_spalding3d.py`, which imports `wall_traction_linearization`).
Reproduce: `PYTHONPATH=src python3 scripts/audit_wall_law_literature.py`
Artifacts: `src/tensorfvm/verification/artifacts/wall_law/wall-law-literature.{json,png}`
Tests: `tests/test_wall_law_literature.py`

## Why this audit
Every pre-existing test of this module is a self-consistency check (exact shear on a
flat wall, manufactured circular shear, Jacobian vs finite differences, torch/NumPy
parity). None compares the **law itself** with externally published wall laws or
friction lines, so the docstring claim "independently verified" rested on internal
consistency only. Shipped constants: kappa = 0.41, E = 9.8, hence B = ln(E)/kappa = 5.5668.

## 1. Inverse law vs a 50-digit reference
`u_plus_at_y_plus` (bisection) against an mpmath 50-digit Newton solve of the same
implicit law, 25 points over y+ in [1e-6, 1e6]: **max relative error 2.68e-10**
(worst point y+ = 1e-6, i.e. the float64 series threshold, not the iteration).

## 2. Region characterisation (module law, deviation measured relative to u+)
| criterion | value |
|---|---|
| leaves u+ = y+ by 1% | y+ = 3.93 |
| within 1% of ln(E y+)/kappa | y+ = 118.0 (kappa*y+ = 48.4) |
| within 0.5% of ln(E y+)/kappa | y+ = 198.6 |

## 3. Reichardt (1951) composite law (independent anchor)
Reichardt: u+ = ln(1+kappa y+)/kappa + 7.8[1 - exp(-y+/11) - (y+/11)exp(-y+/3)]
| quantity | value |
|---|---|
| max abs delta u+ on y+ in [1,1000] | 0.574 |
| rms delta u+ on [30,1000] | 0.319 |
| implied u_tau, U = 10 m/s, y = 1 mm, water (y+ = 485) | module 0.48495 m/s vs Reichardt 0.48286 m/s = **+0.43%** |

## 4. Channel friction vs smooth-pipe Prandtl-von Karman
Law-implied profile integral: U_b/u_tau(Re_tau) -> Cf = 2 (u_tau/U_b)^2 with
Re_b = 2h U_b/nu. Matched pipe comparison needs D_h = 4h, so f_D = 4 Cf at Re_Dh = 2 Re_b.

| Re_b | Cf (law) | f_D/4 (PvK) | err | Dean 0.073 Re_b^-0.25 |
|---|---|---|---|---|
| 1e4 | 0.007210 | 0.006472 | +11.4% | -1.2% |
| 3e4 | 0.005421 | 0.005017 | +8.1% | -2.3% |
| 1e5 | 0.004160 | 0.003910 | +6.4% | +1.3% |
| 3e5 | 0.003363 | 0.003184 | +5.6% | +7.8% |
| 1e6 | 0.002727 | 0.002594 | +5.1% | +18.1% |
| 1e7 | 0.001920 | 0.001836 | +4.6% | +47.9% |

Fitted exponent of the law-implied line: n = 0.190. The +4.6...+11.4% band against the
pipe law is partly physical (channel vs pipe, no wake component) and stays inside the
15% acceptance band. Dean's -0.25 power fit diverges monotonically (+48% at Re_b = 1e7),
so **bulk friction scaling must not be read off this constitutive law**; only the local
traction at a given (U, y) state is qualified.

## 5. Constant sensitivity (Cf at Re_b = 1e5)
| constants | B | Cf | deviation vs shipped |
|---|---|---|---|
| kappa = 0.40, E = 9.0 | 5.493 | 0.004047 | -2.7% |
| kappa = 0.41, E = 9.8 (shipped) | 5.567 | 0.004160 | 0 |
| kappa = 0.4187, E = 9.793 | 5.449 | 0.004320 | +3.8% |

## 6. Two defects found and repaired
1. **Silent overflow in the series branch.** `np.where` evaluated z^4 for *all* inputs, so
   `kappa*u+ > ~1e103` raised `RuntimeWarning: overflow` (fatal under `-W error`). Fixed by
   evaluating the series on a masked copy (`scaled = where(small, z, 0)`); values unchanged.
2. **Inverse of the law returned the bracket width, not the root, past y+ ~ 1e20.** The
   former fixed 90 halvings of [0, y+] resolve y+ * 2^-90; at y+ = 1e30 it returned
   **403.90 instead of 174.05** with no diagnostic. Fixed with a rigorous upper bound
   u+ <= (ln(E y+) + 10 kappa)/kappa (the exponential term alone exceeds y+ by exp(10 kappa)),
   plus a relative convergence break. Residual of the recovered root now <= 2.5e-14 up to
   y+ = 1e60.
3. **Declared saturation limit** (not a bug): for kappa u+ > 700 the law saturates at
   y+ ~ 1.03e303 in float64; the forward law is then non-decreasing but not strictly
   increasing. Documented and regression-tested.

## Verification
`pytest tests/test_wall_law_literature.py tests/test_wall_traction.py
tests/test_spalding_linearization.py tests/test_native_spalding3d.py
tests/test_native_sst_wall3d.py tests/test_engineering_benchmarks.py` -> all green
(47 passed, 3 skipped), so the repairs do not disturb the shipped coupling paths.
