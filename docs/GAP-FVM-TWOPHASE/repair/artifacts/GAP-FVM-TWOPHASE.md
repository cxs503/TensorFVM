# GAP-FVM-TWOPHASE -- VOF interface capture (Zalesak + dam break)

Verdict: **FAIL** (2/10 gates)

## Gates

| id | case | metric | limit | got | verdict |
|---|---|---|---|---|---|
| Z1 | zalesak | geometric_error | <= 0.05 | not run | FAIL (not_run) |
| Z2 | zalesak | mass_drift_relative | <= 0.001 | not run | FAIL (not_run) |
| Z3 | zalesak | perimeter_error | <= 0.1 | not run | FAIL (not_run) |
| Z4 | zalesak | bound_violation | <= 1e-12 | not run | FAIL (not_run) |
| Z5 | zalesak | convergence_order | >= 0.8 | not run | FAIL (not_run) |
| D1 | dambreak | front_max_rel_error | <= 0.15 | 0.8392 | FAIL (measured) |
| D2 | dambreak | height_max_rel_error | <= 0.15 | 0.4419 | FAIL (measured) |
| D3 | dambreak | volume_drift | <= 0.01 | 1.314e-16 | PASS (measured) |
| D4 | dambreak | bound_violation | <= 1e-10 | not run | FAIL (not_measured) |
| D5 | dambreak | control_max_abs_change | <= 1e-14 | 0 | PASS (measured) |

## Zalesak slotted disk (one rigid revolution)

| scheme | N | geometric err | mass drift | perimeter err | peak retention | interface width [cells] |
|---|---|---|---|---|---|---|
| plic | 64 | 0.0930 | 8.76e-15 | 0.1129 | 1.0000 | 6.000 |

Exact solution of one revolution = the initial field; the analytic slotted-disk area/perimeter are in each leg's `reference` block.

## Dam break vs Martin & Moyce (1952)

| scheme | grid | dx/a | front err | height err | volume drift | control |
|---|---|---|---|---|---|---|
| plic | 60x20 | 0.175 | 0.8392 | 0.4419 | 1.31e-16 | 0 |

Front error uses the half-cell depth convention over `0.3 < t* <= t_end`; the leading-edge convention is reported per leg in `front_max_rel_error_leading_edge`.

