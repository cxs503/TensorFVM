# fvm_backward_step: backward-facing step at Re_H = 800

**Verdict: FAIL** (8/12 criteria)

Staircase step, expansion ratio 1.9423, parabolic inlet, Gartling/Armaly convention (Re based on the downstream height and bulk velocity). Criterion: L_r/H = 6.0 +- 0.2 with a 10 % relative gate; the friction coefficient is checked for the correct sign trend and against the analytic downstream value 12 nu/(U_b H).

| criterion | value | reference | error | verdict |
|---|---|---|---|---|
| bfs.reattachment.finest | 6.963501677785078 | 6.0 | 0.16058361296417964 | FAIL |
| bfs.reattachment.band | 6.963501677785078 | 6.0 | 0.16058361296417964 | FAIL |
| bfs.grid.monotone | 0.16058361296417964 | None | None | PASS |
| bfs.grid.order | 0.8636454983009949 | 1.5 | 0.6363545016990051 | FAIL |
| bfs.friction.convergence | 0.01576579744917914 | None | None | FAIL |
| bfs.friction.developed_comparison | -0.11667692195202231 | 0.015 | 8.778461463468153 | PASS |
| bfs.friction.trend | -0.15345683312355415 | 0.0 | None | PASS |
| bfs.mass.continuity | 2.4423173997437098e-06 | 0.0 | 2.4423173997437098e-06 | PASS |
| bfs.friction.domain_length | 0.01840107656453514 | 0.015 | 0.22673843763567603 | PASS |
| bfs.convergence.H/10 | 9.285154210858211 | 9.287967803441472 | 0.0003029287614689938 | PASS |
| bfs.convergence.H/20 | 7.802756524894603 | 7.787888472501673 | 0.0019091249759710374 | PASS |
| bfs.convergence.H/40 | 7.012605515762292 | 6.963501677785078 | 0.007051601370882868 | PASS |

Programmatic source: `scripts/report_fvm_incompressible.py`;
raw runs: `docs/incompressible-baselines/baseline-runs/bfs_n010.json`, `docs/incompressible-baselines/baseline-runs/bfs_n020.json`, `docs/incompressible-baselines/baseline-runs/bfs_n040.json`
