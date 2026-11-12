# D1/D2 resolution audit -- is the dam-break error a model error or a measurement limit?

Auto-rendered from `audit_d1d2.json` by `python -m tensorfvm.twophase_gap.dambreak_audit`.  **No gate changes here**: D1/D2 keep their verdicts; this document only decomposes the misfit.

Reference: Martin & Moyce (1952) digitised anchors, t* = t sqrt(g/a), x* = (x_f - a)/a, h* = h/h0.  Comparison window `0.3 < t* <= 4.0`.  Gates: D1 front `0.15`, D2 height `0.15`.

## 1. Gates as declared (unchanged)

| grid | steps | D1 front | verdict | D2 height | verdict | volume drift |
|---|---|---|---|---|---|---|
| 60x20 | 82 | 0.839 | FAIL | 0.798 | FAIL | 2.63e-16 |
| 120x40 | 169 | 0.988 | FAIL | 0.845 | FAIL | 4.11e-16 |

## 2. Decomposition of the misfit

| grid | front best time shift [t*] | RMS x* (no shift -> shifted) | front late slope got/ref (ratio) | height best shift [t*] | height slope ratio | verdict (front / height) |
|---|---|---|---|---|---|---|
| 60x20 | 0.87 | 0.771 -> 0.132 | 0.986 / 0.754 (1.307) | 0.77 | 2.437 | phase_lag / phase_lag |
| 120x40 | 0.82 | 0.766 -> 0.120 | 0.990 / 0.756 (1.310) | 0.96 | 3.141 | phase_lag / phase_lag |

Time-*scale* falsification control (would fire if the digitised anchor were normalised with `sqrt(2g/a)` instead of `sqrt(g/a)`):

| grid | best pure time scale | RMS at best scale | RMS at best shift |
|---|---|---|---|
| 60x20 | 1.540 | 0.289 | 0.132 |
| 120x40 | 1.530 | 0.301 | 0.120 |

## 3. Cell-quantisation floor under the 15 % gate

| grid | front quantum [x*] | front floor at window start | window samples where the floor alone violates D1 | height quantum [h*/h0] | samples where the floor alone violates D2 |
|---|---|---|---|---|---|
| 60x20 | 0.175 | 0.376 | 0.16 | 0.0875 | 0.84 |
| 120x40 | 0.087 | 0.192 | 0.03 | 0.0437 | 0.47 |

## 4. Conclusions (machine-derived)

* front phase lag is **0.82 - 0.87 t\*** across the refinement sequence, i.e. grid independent;
* late-window front slope ratio is **1.31 - 1.31** (> 1: the computed front is *steeper* than the reference once it is moving -- the defect is a delayed release, not a slow wave);
* D1 is monotone in resolution: **False**;
* the largest quantisation floor at the window start is **0.38**, so on the coarser grids the 15 % gate is decided by cell quantisation at early `t*` rather than by model accuracy.

