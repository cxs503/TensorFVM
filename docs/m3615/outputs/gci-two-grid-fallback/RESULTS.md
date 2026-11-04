# GCI two-grid fall-back audit

cases=636 (fall-back=468, informative=112, consistent controls=168)

Informative subset (exact value inside the span of the three computed values):

| convention | coverage | worst efficiency |
|---|---|---|
| legacy (fine pair, \|f_ext\| scaling) | 0.2321 | 90.168 |
| fixed (worst pair, data scaling) | 1.0000 | 0.870 |

| h*/h_fine | cases | legacy coverage | fixed coverage | legacy worst E | fixed worst E | median width ratio |
|---|---|---|---|---|---|---|
| 1.3 | 9 | 0.000 | 1.000 | 16.093 | 0.870 | 48.632 |
| 1.6 | 15 | 0.000 | 1.000 | 12.680 | 0.626 | 44.726 |
| 2 | 6 | 0.000 | 1.000 | 3.200 | 0.667 | 7.520 |

| r21 | r32 | cases | legacy coverage | fixed coverage | legacy worst E | fixed worst E |
|---|---|---|---|---|---|---|
| 2 | 2 | 15 | 0.000 | 1.000 | 12.680 | 0.667 |
| 2 | 1.5 | 3 | 0.000 | 1.000 | 9.885 | 0.238 |
| 1.5 | 2 | 12 | 0.000 | 1.000 | 16.093 | 0.870 |
