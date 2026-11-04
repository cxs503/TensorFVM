# Two-phase transient projection audit

| 项 | 判据 | 实测 | 判定 |
|---|---|---|---|
| A 静水锚点连续 | 界面跳变 = ρ_face·g·dy = -76.72 Pa | -71.81 Pa (缺 4.91 Pa) | FAIL |
| B 静止分层柱保持静止 | max\|v\|<1e-9 | n=16: 1.19e+38, n=32: 9.23e+49, n=64: 4.57e+61 | FAIL |
| C Young-Laplace 压差 σ/R | 误差 <3% (σ/R=0.350 Pa) | 0.000e+00 Pa (100.00%) | FAIL |
| D 全封闭域 fail-fast | 抛 ValueError | 抛 ValueError | PASS |
