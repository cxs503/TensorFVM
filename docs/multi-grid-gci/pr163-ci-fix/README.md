# PR #163 CI fix: `reference_agreement` could not read an N-level study

## Symptom (CI, run on 37ee48b4, `test (3.11)`)

```
FAILED tests/test_multi_grid_gci.py::test_reference_agreement_accepts_a_multi_grid_study
E   AttributeError: 'MultiGridStudy' object has no attribute 'fallback_reason'
src/tensorfvm/verification/richardson.py:242
1 failed, 1381 passed, 18 skipped, 2 xfailed, 50 subtests passed in 680.62s
```

## Root cause

`reference_agreement(study, reference)` was widened to echo the confidence
qualifiers of a study into its verdict:

```python
band=study.band, asymptotic=bool(study.asymptotic),
fallback_reason=study.fallback_reason,          # richardson.py:242
safety_factor=study.safety_factor)
```

`richardson.RefinementStudy` (three grids) carries `fallback_reason`, but the
N-level result `multi_grid.MultiGridStudy` never did: it exposed `convergence`
(`'monotone'`/`'inconsistent'`) and `asymptotic`, and only implied the fall-back
by the safety factor (1.25 vs 3.0).  Passing a `MultiGridStudy` - which the API
documentation explicitly supports - therefore raised `AttributeError`.

## Fix (minimal, one module)

* `MultiGridStudy` gains `fallback_reason: str`, documented next to the other
  reported quantities.
* `multi_grid_order` fills it with the same vocabulary as the three-grid
  routine: `'non-monotone'` when the series has no monotone one-term decay (or
  the fit is not finite) and the band falls back to `theory_order` with
  `Fs = 3.0`, `'none'` for a resolved fit.
  The N-level estimator has no `'implausible-order'` branch: the order is
  minimised inside `ORDER_BOUNDS`, so a fitted order cannot leave the bracket.
* Removed a dead `res = np.full(...)` assignment in the fall-back branch (the
  value is never read; ruff `F841`).
* Regression test `test_fallback_reason_is_quoted_for_multi_grid_studies`
  (positive case + non-monotone negative control).

No number moves: `order`, `gci`, `band`, `safety_factor` are untouched.

## Evidence

Generated with `tensorfvm.verification.multi_grid.multi_grid_order` on a
4-level ladder `h = 1/64, 1/32, 1/16, 1/8` (fine -> coarse); see
`figure.png` and `evidence.json`.

| case | convergence | fallback_reason | order_used | extrapolated | GCI | Fs |
|---|---|---|---|---|---|---|
| `f = 1 + 0.5 h^2` (resolved) | monotone | `none` | 2.0000 | 1.000000 | 1.526e-4 | 1.25 |
| `(1, .8, .9, .7)` (negative control) | inconsistent | `non-monotone` | 1.0 | 1.200000 | 0.600 | 3.00 |

`reference_agreement(resolved, L)` now returns
`{'band_covered': True, 'efficiency_index': 0.0, 'fallback_reason': 'none'}`;
the negative control returns `fallback_reason='non-monotone'`.

Error of the resolved study against the closed-form limit `L = 1`:
`|1.000000 - 1| / 1 = 0.0%` (< 3%).

## Local verification

```
pytest -q tests/test_multi_grid_gci.py tests/test_richardson_gci.py \
          tests/test_gci_band_coverage.py tests/test_gci_fallback_band.py
# 70 passed
ruff check src tests scripts   # All checks passed
```
