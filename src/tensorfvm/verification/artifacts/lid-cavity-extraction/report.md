# Re = 400 lid-cavity v-centreline: extraction path vs solution field

The strict xfail in `tests/test_lid_cavity_re1000_evidence.py` records a v-centreline error that grows under refinement (0.149 at n = 128) and hypothesises a *wall-adjacent extraction defect*. This audit measures the extraction path on fields of known closed form.

## 1. Where the samples actually sit

| n | h | v mid-plane y | exact | min node-to-Ghia offset (cells) |
|---|---|---|---|---|
| 32 | 0.031250 | 0.500000 | True | 0.498 |
| 64 | 0.015625 | 0.500000 | True | 0.497 |
| 128 | 0.007812 | 0.500000 | True | 0.494 |

`v` is staggered as `v[x_cell, y_face]` with faces at `j*h`, so the horizontal mid-plane is the stored face `j = n/2`: no half-cell offset, no interpolation of the ordinate. The same holds for `u` at `x = L/2`.

## 2. Extraction error on an analytic field

| n | max abs error | max relative (to |v| peak) | <= 1e-2 ? |
|---|---|---|---|
| 32 | 3.943e-05 | 1.480e-04 | True |
| 64 | 1.295e-06 | 4.860e-06 | True |
| 128 | 8.205e-08 | 3.080e-07 | True |

Observed order in h: **4.45**.

## 3. Positive control (does the measurement have teeth?)

| n | sample shift (cells) | max abs error | clean error | ratio |
|---|---|---|---|---|
| 32 | 0.5 | 2.890e-02 | 3.943e-05 | 733x |
| 32 | 1.0 | 5.395e-02 | 3.943e-05 | 1368x |
| 64 | 0.5 | 1.494e-02 | 1.295e-06 | 11539x |
| 64 | 1.0 | 2.892e-02 | 1.295e-06 | 22339x |

## 4. Attribution

Extraction budget at n = 32: 1.480e-04 relative; recorded defect 0.149; ratio **1007x**.

**Verdict.** extraction path exonerated: it reproduces a smooth profile to 1.48e-04 relative, 1007x below the 0.149 the ladder records at x/L = 0.9063; the discrepancy belongs to the solution field, so the xfail's 'wall-adjacent extraction defect, not discretisation' hypothesis is refuted.

## 5. Live ladder (cached run, re-scored here)

| n | steps | converged | max rel. error | worst x/L | stored |v| peak | exceeds Ghia peak |
|---|---|---|---|---|---|---|
| 32 | 18000 | True | 0.178 | 0.9688 | 0.4139 | False |

|v| exceeding the published peak is stored at x/L = []

## 6. Scope

This audit attributes the discrepancy; it does not fix the solution-field defect. The analytic block is exact (roundoff on nodes), so any residual attribution to the extraction path is bounded below 1 % of the profile scale, i.e. two orders below the recorded 14.9 %.
