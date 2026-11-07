# fvm_sudden_expansion: planar sudden expansion, AR = 4, Re_D = 100

**Verdict: FAIL** (2/7 criteria)

Uniform inlet over D, expansion to 4D, 12D long. The Borda-Carnot value K_BC = (1 - 1/AR)^2 = 0.5625 is an *inviscid* momentum balance; the measured laminar loss is larger, and the Reynolds sweep documents that the gap is physical rather than numerical.

| criterion | value | reference | error | verdict |
|---|---|---|---|---|
| expansion.K.finest | 1.1312543698053101 | 0.5625 | 1.0111188796538846 | FAIL |
| expansion.grid.monotone | 1.0111188796538846 | None | None | FAIL |
| expansion.grid.order | 1.0771143651041442 | 1.5 | None | FAIL |
| expansion.area_ratio | 4.0 | 4.0 | None | PASS |
| expansion.identity | 1.1312543698053101 | 1.1312543698053101 | 0.0 | PASS |
| expansion.K.domain_length | 0.8885330625954506 | 0.5625 | 0.5796143335030233 | FAIL |
| expansion.K.Re_sweep | 1.5189007365033147 | 0.5625 | 1.7002679760058927 | FAIL |

Programmatic source: `scripts/report_fvm_incompressible.py`;
raw runs: `runs/incompressible-baselines/expansion_n010.json`, `runs/incompressible-baselines/expansion_n020.json`, `runs/incompressible-baselines/expansion_n030.json`, `runs/incompressible-baselines/expansion_re0000100.json`, `runs/incompressible-baselines/expansion_re0000200.json`, `runs/incompressible-baselines/expansion_re0000500.json`, `runs/incompressible-baselines/expansion_re0001000.json`, `runs/incompressible-baselines/expansion_re0002000.json`, `runs/incompressible-baselines/expansion_re0005000.json`
