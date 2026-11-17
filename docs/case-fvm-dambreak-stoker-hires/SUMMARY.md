# High-resolution dam-break extension (auto-generated)

generated: 2026-11-17T16:05:06Z  commit: `86db1ca7`  tolerance: 5%

## ritter-dry — headline = front_fan (fan-shape estimator at t*=2)

| n | dx [m] | x_f(t) series | graded | headline | depth L1 | verdict failing | wall [s] |
|---|---|---|---|---|---|---|---|
| 800 | 0.0125 | 0.532% | 23/25 | 0.039% | 0.1542% | — | 1.0 |
| 1600 | 0.00625 | 0.425% | 24/25 | 0.017% | 0.0772% | — | 2.5 |
| 3200 | 0.003125 | 0.380% | 24/25 | 0.008% | 0.0386% | — | 6.9 |
| 6400 | 0.0015625 | 0.267% | 24/25 | 0.004% | 0.0193% | — | 21.7 |

* observed order (depth L1): 0.9992248819539148
* observed order (x_f(t) series): 0.31333007366267634
* observed order (headline): 1.090296434500497
* series decay: finest ratio 0.7035582806276643 → at floor = False
* series max over accepted grids (n>=800): 0.5317% (all pass: True)

## stoker-wet — headline = shock_position (oracle-windowed, AUDIT F4)

| n | dx [m] | x_f(t) series | graded | headline | depth L1 | verdict failing | wall [s] |
|---|---|---|---|---|---|---|---|
| 800 | 0.0125 | 0.795% | 24/25 | 0.043% | 0.0851% | — | 0.6 |
| 1600 | 0.00625 | 0.844% | 24/25 | 0.021% | 0.0427% | — | 1.5 |
| 3200 | 0.003125 | 0.429% | 24/25 | 0.016% | 0.0219% | — | 4.2 |
| 6400 | 0.0015625 | 0.205% | 24/25 | 0.006% | 0.0105% | — | 13.2 |

* observed order (depth L1): 1.0041914164574552
* observed order (x_f(t) series): 0.6840435098665504
* observed order (headline): 0.8953160877402851
* series decay: finest ratio 0.4787147850093342 → at floor = False
* series max over accepted grids (n>=800): 0.8443% (all pass: True)

| n | bore x sub-cell | bore x cell centre (neg. control) | bore h2 | bore step ratio |
|---|---|---|---|---|
| 800 | 0.0433% | 0.3987% | 1.836e-06 | 0.0006% |
| 1600 | 0.0210% | 0.2333% | 3.716e-05 | 0.0035% |
| 3200 | 0.0160% | 0.0149% | 1.084e-04 | 0.0107% |
| 6400 | 0.0060% | 0.0265% | 9.691e-07 | 0.0001% |

| n | fan max\|dh/dx\| [m/m] | jump max\|dh/dx\| [m/m] | fan/jump | below cut 0.01 | cut used | plateau flatness (flag > 0.05) |
|---|---|---|---|---|---|---|
| 800 | 0.3131 | 5.838 | 5.364e-02 | no | 0.01 | 0.867% |
| 1600 | 0.3158 | 11.22 | 2.816e-02 | no | 0.01 | 1.669% |
| 3200 | 0.3195 | 23.63 | 1.352e-02 | no | 0.01 | 3.961% |
| 6400 | 0.3226 | 50.19 | 6.428e-03 | yes | 0.001 | 0.175% |
| ↳ 800 ladder | 0.01: bounded n=229 | | | | | |
| ↳ 1600 ladder | 0.01: bounded n=472 | | | | | |
| ↳ 3200 ladder | 0.01: bounded n=964 | | | | | |
| ↳ 6400 ladder | 0.01: unbounded, 0.001: bounded n=1902 | | | | | |

* F9: fan/jump first below the cut at n = 6400
* observed order (bore sub-cell): 0.8956532016310292
* bore sub-cell decay: finest ratio 0.37295518707535635 → at floor = False
* worst cell-centre error over range: 0.3987%
