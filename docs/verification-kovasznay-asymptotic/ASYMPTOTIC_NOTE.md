# CASE-FVM-KOVASZNAY-ORDER — asymptotic-pair addendum (n = 32 adopted)

Every number below is injected by `asymptotic_note.py` from `runs/summary.json` / `runs/level-n32-zero_gradient-reused_lu.json`; nothing is hand copied.

## 1. What changed

- Ladder levels now available: 8, 16, 32.
- `n = 32` was **adopted** (not re-solved) from `docs/verification-kovasznay-order-closure/runs/level_32/summary.json` (1008 s of solver time already spent there); the record carries `source` / `adopted_by` / `provenance` fields.
- Same configuration as the ladder: closure `zero_gradient`, Re = 40, second-order deferred correction, Picard to 1e-8, volume-weighted L2 norms.

## 2. Pair orders

| pair | regime | total L2 order | in-plane | pressure (mean-free) |
|---|---|---:|---:|---:|
| 8->16 | pre-asymptotic (n_coarse = 8) | 1.5875 | 1.5875 | 1.4359 |
| 16->32 | **asymptotic (n_coarse >= 16)** | 1.8083 | 1.8083 | 1.7453 |

## 3. Verdicts

| id | criterion | value | verdict |
|---|---|---|---|
| C1 | tied-dof residual max|u_face-u_cell| <= 1e-12 at every level | 0.0000 | **PASS** |
| C2 | spanwise L2 and max|w| exactly 0 at every level (w exact = 0) | 0.0000 | **PASS** |
| C3 | ANCHOR: total velocity L2 order >= 1.8 on the finest pair 16->32 | 1.8083 | **PASS** |
| C4 | in-plane L2 order >= 1.8 on the finest pair | 1.8083 | **PASS** |
| C5 | least-squares total L2 slope over all levels >= 1.8 | 1.6979 | **FAIL** |
| C6 | Picard reported numerical convergence at every level | True, True, True | **PASS** |
| C7 | informational: pressure (mean-free) order on the finest pair | 1.7453 | **INFO** |
| C8 | informational: linear-configuration independence of total L2 <= 1e-06 | n/a | **INFO** |

- Pre-declared ladder criterion C3/C4 on the finest available pair (16->32): 1.8083 vs gate 1.8.
- C5 (least-squares over *all* levels, including the pre-asymptotic n = 8 point): 1.6979 -> FAIL (the n = 8 level is not in the asymptotic range and drags the global slope down).
- Asymptotic-only least-squares slope (n >= 16, 2 levels): 1.8083.

## 4. Honest bookkeeping / limits

- The ledger criterion literally asks for a slope on **32/64/128**. This note delivers the **16->32** asymptotic pair only; `n = 64` (cluster job cj1373, ~83 min single-core estimate) had not been dispatched when this note was written, so the literal 32/64/128 ladder is **still open** and no PASS is claimed for it.
- C5 is left as FAIL rather than re-defining the gate after seeing the data: the asymptotic-only slope is reported next to it, not instead of it.
- `zero_gradient` ties the free tangential dofs, so the spanwise component is exactly zero at every level and total L2 == in-plane L2 (values identical above, not a coincidence).
- Convergence is a flag from the campaign record; its momentum residual was not stored, so the residual column is n/a for n = 32 rather than invented.

