# GCI Non-Asymptotic Fallback Band Audit (m3520)

## 1. Background

The m3496 audit found that monotone but non-asymptotic three-grid studies
were reporting narrow bands with Fs=1.25, leading to poor coverage (0/81)
and worst efficiency E=54 in the F-cancel family.  This audit verifies the
fix implemented in m3520: when `theory_order` is declared and
`|p_obs-p_theory|>order_tolerance`, the band falls back to the two-point
method (Fs=3 + formal order + max adjacent difference eps).

## 2. Method

Constructed a family of two-term series `f = f_exact + C*h^p + alpha*C*h^s`
with known `theory_order=p`.  Identified cases where the series is monotone
but non-asymptotic (|p_obs-p_theory| > 0.2).  Verified that the fallback
widens the band appropriately.

## 3. Results

Total cases: 640

- Non-asymptotic fallback cases: 142
- Asymptotic cases: 498
- Inconsistent cases: 0

### Coverage

- Non-asymptotic fallback coverage: 139/142 (0.9789)
- Asymptotic coverage: 498/498 (1.0000)

### Efficiency

- Non-asymptotic worst efficiency: 0.292
- Asymptotic worst efficiency: 0.223

## 4. Conclusion

The non-asymptotic fallback successfully widens the band when the study is
monotone but not in the asymptotic range.  The safety factor increases from
1.25 to 3.0, and eps is computed as the maximum adjacent difference rather
than just the fine-medium difference.  This provides a more conservative
band that better reflects the pre-asymptotic uncertainty.

## 5. Products

- This report: `docs/gci-fallback-band/report.md`
- Audit data (excerpt: aggregates + worst cases; full tables rebuilt with `--full`, SHA-256 in `full_dump`): `docs/gci-fallback-band/audit.json`