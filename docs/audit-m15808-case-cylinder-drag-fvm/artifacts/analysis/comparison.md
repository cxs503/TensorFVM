# CASE-CYLINDER-DRAG-FVM - reference vs computed

Reference: Turek & Schaefer 1996, Benchmark computations of laminar flow around a cylinder, NNFM 52, 547-566

Source page: https://wwwold.mathematik.tu-dortmund.de/~featflow/en/benchmarks/cfdbenchmarking/flow/dfg_benchmark1_re20.html (sha256 unbound, see pro...)

| grid | C_D | err % | C_L | err % | p_diff (Pa) | err % | verdict |
|---|---|---|---|---|---|---|---|
| 32x12 | 6.670193 | +19.5475 | +0.051804 | +387.848 | 0.129441 | +10.1435 | C_D FAIL / C_L FAIL / p_diff FAIL |
| 64x24 | 6.324792 | +13.3570 | +0.015102 | +42.215 | 0.130991 | +11.4630 | C_D FAIL / C_L FAIL / p_diff FAIL |

Case tolerance: 5%; case verdict on finest grid: **FAIL**

## Convergence

- drag: not available (needs three grids)
- lift: not available (needs three grids)
- pressure_difference_pa: not available (needs three grids)

## Sensitivity

- p_diff via crude inverse-distance cell-centre probing on finest grid: 0.081171 Pa (-30.9298% vs ref)
- drag split (pressure/viscous) on finest grid: 4.2077 / 2.1171
- iterations to converge: {'32x12': [200, False], '64x24': [976, True]}

Figures: cd_refinement.png, error_bars.png, drag_decomposition.png, field_pressure.png, field_speed.png, wall_pressure.png, residuals.png
