# FVM internal-flow verification face

Independent of the LBM chain (`src/tensorfvm/verification`): every number below comes from the staggered MAC / projection solver `src/tensorfvm/internal_flow/mac.py`.

| case | physics | finest mesh | verdict |
|---|---|---|---|
| [fvm_backward_step](fvm_backward_step/REPORT.md) | backward-facing step, expansion ratio 2 | H/20 | $L_r/H_{out}$ = 5.916 (1.4% off 6.0, pass) <br> $p$ = None |
| [fvm_sudden_expansion](fvm_sudden_expansion/REPORT.md) | planar sudden expansion, area ratio 4 | D/40 | $L_r/D$ = 3.356 (pass), $K$ = 0.4412 (21.6% off Borda-Carnot, FAIL) <br> $p$ = None |
| [fvm_natural_convection](fvm_natural_convection/REPORT.md) | differentially heated square cavity | N/128 | $\overline{Nu}$ within 1.15% of de Vahl Davis (pass) <br> orders 2.14, 2.32, 2.31 |

Reproduce: `python -m tensorfvm.internal_flow.sweep --case bfs|expansion|cavity --param N --out DIR` per mesh, then `python scripts/assemble_internal_flow_benchmarks.py`.
