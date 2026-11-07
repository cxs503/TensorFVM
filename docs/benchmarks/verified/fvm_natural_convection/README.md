# fvm_natural_convection: differentially heated square cavity

**Verdict: FAIL** (12/15 criteria)

Boussinesq cavity, Pr = 0.71, Ra = 1e4/1e5/1e6, 32/64/96 squared grids. Criterion: mean Nu within 3 % of the de Vahl Davis values, streamfunction amplitude reported as a secondary quantity.

| criterion | value | reference | error | verdict |
|---|---|---|---|---|
| cavity.Nu.1e+04.finest | 2.2156316497654687 | 2.243 | 0.012201671972595281 | PASS |
| cavity.Nu.1e+04.monotone | 0.012201671972595281 | None | None | PASS |
| cavity.Nu.1e+04.order | None | 1.5 | None | FAIL |
| cavity.psi.1e+04 | 5.10534280206384 | 5.071 | 0.006772392440118297 | PASS |
| cavity.energy.1e+04 | 2.2156316497654687 | 2.212032520805994 | 0.0016244256845922656 | PASS |
| cavity.Nu.1e+05.finest | 4.441884535833445 | 4.519 | 0.01706471877994136 | PASS |
| cavity.Nu.1e+05.monotone | 0.01706471877994136 | None | None | FAIL |
| cavity.Nu.1e+05.order | None | 1.5 | None | FAIL |
| cavity.psi.1e+05 | 10.079484962239736 | 9.111 | 0.10629842632419438 | PASS |
| cavity.energy.1e+05 | 4.441884535833445 | 4.429344745971751 | 0.0028230787541938668 | PASS |
| cavity.Nu.1e+06.finest | 8.735297896186518 | 8.8 | 0.007352511796986674 | PASS |
| cavity.Nu.1e+06.monotone | 0.007352511796986674 | None | None | PASS |
| cavity.Nu.1e+06.order | 2.812400652141748 | 1.5 | None | PASS |
| cavity.psi.1e+06 | 18.624587636585037 | 16.32 | 0.14121247773192627 | PASS |
| cavity.energy.1e+06 | 8.735297896186518 | 8.673063064000763 | 0.007124523161702858 | PASS |

Programmatic source: `scripts/report_fvm_incompressible.py`;
raw runs: `docs/incompressible-baselines/baseline-runs/cavity_n032_ra1e04.json`, `docs/incompressible-baselines/baseline-runs/cavity_n032_ra1e05.json`, `docs/incompressible-baselines/baseline-runs/cavity_n032_ra1e06.json`, `docs/incompressible-baselines/baseline-runs/cavity_n064_ra1e04.json`, `docs/incompressible-baselines/baseline-runs/cavity_n064_ra1e05.json`, `docs/incompressible-baselines/baseline-runs/cavity_n064_ra1e06.json`, `docs/incompressible-baselines/baseline-runs/cavity_n096_ra1e04.json`, `docs/incompressible-baselines/baseline-runs/cavity_n096_ra1e05.json`, `docs/incompressible-baselines/baseline-runs/cavity_n096_ra1e06.json`
