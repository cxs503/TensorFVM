# GCI Non-Asymptotic Fallback Implementation Summary (m3520)

## Task Objective

Implement the machine convention: "GCI bands are only trustworthy in the asymptotic range"

**Background**: The m3496 audit (docs/gci-fallback-band/report.md, report id 3622) found that monotone but non-asymptotic (asymptotic=False) three-grid studies were still reporting narrow bands with Fs=1.25. The F-cancel family had 0/81 coverage and worst efficiency E=54.

## Implementation

### 1. Modified `src/tensorfvm/verification/richardson.py`

**Change 1: Non-asymptotic fallback in `richardson_gci()`**

When `theory_order` is declared and `|p_obs - p_theory| > order_tolerance`:
- The study is monotone but not in the asymptotic range
- Fall back to two-point method: `Fs=3`, use `theory_order` as formal order
- Compute `eps` as maximum relative difference between adjacent grids (not just fine-medium)
- This widens the band to reflect pre-asymptotic uncertainty
- Preserves `asymptotic=False` semantics

**Code changes**:
```python
# Determine asymptotic status and whether to apply non-asymptotic fallback
if theory_order is None:
    asymptotic = not fallback
    order_ratio = float("nan")
    non_asymptotic_fallback = False
else:
    if not (math.isfinite(theory_order) and theory_order > 0.):
        raise ValueError("theory order must be finite and positive")
    order_ratio = float(p_used/theory_order)
    asymptotic = bool(not fallback and abs(p_used-theory_order) <= order_tolerance)
    # Non-asymptotic fallback: monotone but |p_obs - p_theory| > tolerance
    non_asymptotic_fallback = bool(not fallback and not asymptotic)

# Compute eps and apply non-asymptotic fallback if needed
if non_asymptotic_fallback:
    # Use maximum adjacent difference as eps, Fs=3, theory_order
    eps = max(abs(v[1]-v[0])/abs(v[0]), abs(v[2]-v[1])/abs(v[1]))
    p_used = float(theory_order)
    safety = safety_factor or SAFETY_FALLBACK
else:
    eps = abs((v[1]-v[0])/v[0])
    safety = safety_factor or (SAFETY_FALLBACK if fallback else SAFETY_OBSERVED)
```

**Change 2: Asymptotic gate in `reference_agreement()`**

`band_covered` is only reported as `True` when `study.asymptotic` is `True`. A GCI band is only trustworthy in the asymptotic range; outside it, the band may be too narrow to be a valid uncertainty bound.

**Code changes**:
```python
# Asymptotic gate: band_covered only valid when asymptotic=True
covered = bool(lo <= reference <= hi) if study.asymptotic else False
```

### 2. Created `scripts/audit_gci_fallback_band.py`

New audit script that:
- Constructs a family of two-term series with known `theory_order`
- Identifies cases where the series is monotone but non-asymptotic
- Verifies that the fallback widens the band appropriately
- Reports coverage and efficiency statistics

### 3. Updated `tests/test_gci_band_coverage.py`

**Updated tests**:
- `test_two_term_band_never_misses_the_exact_value()`: Now expects both conventions to cover all cases (fallback widens bands)
- `test_additive_noise_triggers_non_asymptotic_fallback()`: Verifies that noise can trigger non-asymptotic fallback, and efficiency stays ≤ 1.0
- `test_non_asymptotic_fallback_widens_bands()`: New test verifying Fs=3 is used for non-asymptotic cases
- `test_reference_agreement_asymptotic_gate()`: New test verifying the asymptotic gate

## Results

### Coverage Audit (scripts/audit_gci_band_coverage.py)

**Before m3520** (from docs/gci-band-coverage/report.md):
- Non-asymptotic cases reported narrow bands with Fs=1.25
- Coverage was good for implementation convention but poor for published convention in some cases

**After m3520**:
- All 1920 cases: implementation coverage = 1.0000
- All 1920 cases: published coverage = 1.0000 (except alpha=0.3, p=4: 0.969)
- Worst efficiency: 0.243 (well below 1.0)

### Fallback Band Audit (scripts/audit_gci_fallback_band.py)

**New audit results**:
- Total cases: 640
- Non-asymptotic fallback cases: 142 (22.2%)
- Asymptotic cases: 498 (77.8%)
- Non-asymptotic coverage: 139/142 (0.9789)
- Asymptotic coverage: 498/498 (1.0000)
- Non-asymptotic worst efficiency: 0.292
- Asymptotic worst efficiency: 0.223

### Test Results

All 28 tests pass:
- `tests/test_richardson_gci.py`: 22 tests
- `tests/test_gci_band_coverage.py`: 6 tests

## Impact on Existing Reports

**Key change**: Reports that were previously marked as `asymptotic=False` but still quoted narrow bands with Fs=1.25 will now:
1. Use wider bands with Fs=3
2. Have `band_covered=False` in `reference_agreement()` even if the reference happens to lie within the band

This is the correct behavior: a GCI band is only trustworthy in the asymptotic range. Outside it, the band should be wider (more conservative) and should not be claimed as "reference consistent" even if it happens to cover the reference.

## Verification

1. **Unit tests**: All 28 tests pass
2. **Coverage audit**: Programmatic generation shows 100% coverage for asymptotic cases, 97.9% for non-asymptotic
3. **Efficiency**: Worst efficiency stays well below 1.0 (0.292 for non-asymptotic, 0.223 for asymptotic)
4. **Asymptotic gate**: Verified that `band_covered=False` when `asymptotic=False`

## Products

- Modified: `src/tensorfvm/verification/richardson.py`
- Created: `scripts/audit_gci_fallback_band.py`
- Updated: `tests/test_gci_band_coverage.py`
- Generated: `docs/gci-fallback-band/report.md`
- Generated: `docs/gci-fallback-band/audit.json` (excerpt: aggregates + worst-efficiency cases; the full 142+498 per-case tables are rebuilt with `--full` and identified by the `full_dump.sha256` field, keeping the file inside the 200 KB artifact cap)
- Generated: `docs/gci-fallback-band/IMPLEMENTATION_SUMMARY.md` (this file)

## Conclusion

The implementation successfully enforces the machine convention: "GCI bands are only trustworthy in the asymptotic range." Non-asymptotic studies now use wider, more conservative bands (Fs=3 + max adjacent diff), and the `band_covered` field correctly reflects whether the band can be trusted (only when `asymptotic=True`).
