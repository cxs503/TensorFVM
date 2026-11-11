# GAP-FVM-TWOPHASE -- VOF interface capture (Zalesak + dam break)

Verdict: **FAIL** (8/10 gates)

## Gates

| id | case | metric | limit | got | verdict |
|---|---|---|---|---|---|
| Z1 | zalesak | geometric_error | <= 0.05 | 0.03955 | PASS (measured) |
| Z2 | zalesak | mass_drift_relative | <= 0.001 | 6.423e-16 | PASS (measured) |
| Z3 | zalesak | perimeter_error | <= 0.1 | 0.0643 | PASS (measured) |
| Z4 | zalesak | bound_violation | <= 1e-12 | 0 | PASS (measured) |
| Z5 | zalesak | convergence_order | >= 0.8 | 1.234 | PASS (measured) |
| D1 | dambreak | front_max_rel_error | <= 0.15 | 0.9882 | FAIL (measured) |
| D2 | dambreak | height_max_rel_error | <= 0.15 | 0.845 | FAIL (measured) |
| D3 | dambreak | volume_drift | <= 0.01 | 4.114e-16 | PASS (measured) |
| D4 | dambreak | bound_violation | <= 1e-10 | 0 | PASS (measured) |
| D5 | dambreak | control_max_abs_change | <= 1e-14 | 0 | PASS (measured) |

## Zalesak slotted disk (one rigid revolution)

| scheme | N | geometric err | mass drift | perimeter err | peak retention | interface width [cells] |
|---|---|---|---|---|---|---|
| cicsam | 128 | 0.0438 | 4.82e-16 | 0.0704 | 1.0000 | 12.000 |
| cicsam | 64 | 0.1105 | 5.42e-14 | 0.1202 | 1.0000 | 6.000 |
| plic | 128 | 0.0395 | 6.42e-16 | 0.0643 | 1.0000 | 12.000 |
| plic | 64 | 0.0930 | 8.76e-15 | 0.1129 | 1.0000 | 6.000 |

Exact solution of one revolution = the initial field; the analytic slotted-disk area/perimeter are in each leg's `reference` block.

## Dam break vs Martin & Moyce (1952)

| scheme | grid | dx/a | front err | height err | volume drift | control |
|---|---|---|---|---|---|---|
| plic | 120x40 | 0.087 | 0.9882 | 0.8450 | 4.11e-16 | 0 |
| plic | 240x80 | 0.044 | 0.8477 | 0.8265 | 1.31e-16 | 0 |

Front error uses the half-cell depth convention over `0.3 < t* <= t_end`; the leading-edge convention is reported per leg in `front_max_rel_error_leading_edge`.

