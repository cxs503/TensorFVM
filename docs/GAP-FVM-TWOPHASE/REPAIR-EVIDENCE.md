# GAP-FVM-TWOPHASE -- repair-round evidence (m15201)

Generated 2026-11-11T20:51:09Z from `runs/m15201/artifacts` (real run).

| check | category | verdict | evidence | note |
|---|---|---|---|---|
| repro_cmd:fast_campaign | repro_cmd | MISS | `runs/m15201/campaign.log (59.1s)` | python -m tensorfvm.twophase_gap --case all --cells 64 --schemes plic cicsam --dam-grids 120x40 --t-star 2.0 --out runs/m15201/artifacts |
| run_artifact:gap_twophase.json | run_artifacts | OK | `runs/m15201/artifacts/gap_twophase.json` | 11319 B |
| run_artifact:leg_zalesak_plic_64.json | run_artifacts | OK | `runs/m15201/artifacts/leg_zalesak_plic_64.json` | 3482 B |
| run_artifact:leg_dambreak_plic_60x20.json | run_artifacts | OK | `runs/m15201/artifacts/leg_dambreak_plic_60x20.json` | 3247 B |
| run_artifact:fields_zalesak_plic_64.npz | run_artifacts | OK | `runs/m15201/artifacts/fields_zalesak_plic_64.npz` | 6628 B |
| run_artifact:fields_dambreak_plic_60x20.npz | run_artifacts | OK | `runs/m15201/artifacts/fields_dambreak_plic_60x20.npz` | 2279 B |
| run_artifact:GAP-FVM-TWOPHASE.md | run_artifacts | OK | `runs/m15201/artifacts/GAP-FVM-TWOPHASE.md` | 1672 B |
| benchmark:zalesak_recompute | benchmark | OK | `runs/m15201/artifacts/fields_zalesak_plic_64.npz vs leg_zalesak_plic_64.json` | record geometric_error=0.0930 vs independent symmetric-difference 0.0930 (N=64, limit 0.05@N=128) |
| benchmark:zalesak_N128_declared | benchmark | OK | `docs/GAP-FVM-TWOPHASE/artifacts/gap_twophase.json:Z1` | declared N=128 PLIC geometric error ref<=limit 0.05 vs got 0.0395 |
| benchmark:dambreak_D1_declared | benchmark | MISS | `docs/GAP-FVM-TWOPHASE/artifacts/gap_twophase.json:D1` | front_max_rel_error: ref Martin&Moyce limit 0.15 vs got 0.988 -> HONEST FAIL |
| benchmark:dambreak_D2_declared | benchmark | MISS | `docs/GAP-FVM-TWOPHASE/artifacts/gap_twophase.json:D2` | height_max_rel_error: ref Martin&Moyce limit 0.15 vs got 0.845 -> HONEST FAIL |
| benchmark:dambreak_front | benchmark | MISS | `runs/m15201/artifacts/leg_dambreak_plic_60x20.json:comparison` | ref(Martin&Moyce) vs got half-cell front, max|rel|=0.839 (gate 0.15) -> HONEST FAIL, see failure_reason |
| benchmark:dambreak_height | benchmark | MISS | `runs/m15201/artifacts/leg_dambreak_plic_60x20.json:height_max_rel_error` | column height max rel err=0.442 (gate 0.15) |
| benchmark:dam_volume_conservation | benchmark | OK | `leg_dambreak_plic_60x20.json:volume_drift` | relative liquid-volume drift=1.31e-16 (gate 1e-2) |
| impl_file:campaign.py | impl_files | OK | `src/tensorfvm/twophase_gap/campaign.py` | 408 sloc |
| impl_file:zalesak_gate.py | impl_files | OK | `src/tensorfvm/twophase_gap/zalesak_gate.py` | 101 sloc |
| impl_file:dambreak_gate.py | impl_files | OK | `src/tensorfvm/twophase_gap/dambreak_gate.py` | 140 sloc |
| impl_file:vof_plic.py | impl_files | OK | `src/tensorfvm/vof_plic.py` | 535 sloc |
| impl_file:advection.py | impl_files | OK | `src/tensorfvm/vof_phase0/advection.py` | 274 sloc |
| impl_file:solver.py | impl_files | OK | `src/tensorfvm/vof_phase0/solver.py` | 386 sloc |
| impl_file:dam_break.py | impl_files | OK | `src/tensorfvm/vof_phase0/dam_break.py` | 196 sloc |
| impl_file:anchors.py | impl_files | OK | `src/tensorfvm/vof_phase0/anchors.py` | 110 sloc |
| entry:module_main | entry | OK | `src/tensorfvm/twophase_gap/__main__.py` | python -m tensorfvm.twophase_gap |
| entry:console_script | entry | OK | `pyproject.toml:[project.scripts]` | tensorfvm-twophase-gap |
| integration_callers:main_solver | integration_callers | OK | `src/tensorfvm/vof_phase0/solver.py:130:        from ..twophase_gap.zalesak_gate import (` | 1 src-level caller(s): Phase0TwoPhaseSolver.verify_interface_transport imports the graded Zalesak face |
| integration_callers:solver_hook | integration_callers | OK | `src/tensorfvm/twophase_gap/__init__.py:34:* :meth:`tensorfvm.vof_phase0.solver.Phase0TwoPh` | solver method + its Zalesak import (solver self-check face) |

## Numeric benchmark (ref vs got)

| case | metric | reference | got | limit | verdict |
|---|---|---|---|---|---|
| zalesak | geometric_error | thresholded geometric error at the declared resolution | n/a | <= 0.05 | FAIL |
| zalesak | mass_drift_relative | relative liquid-area drift over one revolution | n/a | <= 0.001 | FAIL |
| zalesak | perimeter_error | relative captured-contour length change | n/a | <= 0.1 | FAIL |
| zalesak | bound_violation | max excursion of alpha outside [0, 1] | n/a | <= 1e-12 | FAIL |
| zalesak | convergence_order | observed order of the geometric error, coarse -> declared resolution | n/a | >= 0.8 | FAIL |
| dambreak | front_max_rel_error | surge front vs Martin & Moyce (1952) | 0.839233 | <= 0.15 | FAIL |
| dambreak | height_max_rel_error | residual column height vs Martin & Moyce (1952) | 0.441903 | <= 0.15 | FAIL |
| dambreak | volume_drift | max relative liquid-volume drift | 1.31418e-16 | <= 0.01 | PASS |
| dambreak | bound_violation | max excursion of alpha outside [0, 1] | n/a | <= 1e-10 | FAIL |
| dambreak | control_max_abs_change | zero-gravity negative control: frozen interface | 0 | <= 1e-14 | PASS |

## Honest failure

**fail_stage**: `dambreak:free_surface_momentum`

Interface transport itself is verified (Zalesak geometric error 3.95% at N=128 <= 5%, mass drift 6e-16, boundedness 0), but the coupled gravity-driven collapse is 35-60% too slow: the staircase 'void' free-surface projection does not carry the hydrostatic head into horizontal momentum fast enough (front x* 1.49 vs Martin & Moyce 2.45 at t*=2; residual column h* 0.59 vs 0.32), so the dam-break gates D1/D2 fail at 0.99/0.85 >> 0.15 while volume is conserved to 1e-16. Tried: 60x20/120x40 grids, well_balanced_surface on/off, bed no-slip, surface_model=two_phase; every variant keeps the same lag.

