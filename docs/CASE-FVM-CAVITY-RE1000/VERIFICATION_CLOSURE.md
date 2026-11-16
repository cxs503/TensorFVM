# CASE-FVM-CAVITY-RE1000 Verification Closure (m22651)

> Anchor: `CASE-FVM-CAVITY-RE1000` (算例池·方腔流 Re=1000 基准)
> Acceptance: <5% tolerance vs Ghia et al. 1982
> Date: 2026-11-16

## Verification Summary

The lid-driven cavity at Re=1000 has been verified against Ghia, Ghia & Shin (1982).
All 7 acceptance criteria PASS at the judge grid n=128:

| Criterion | Measured | Threshold | Pass |
|---|---|---|---|
| C1 max|Δu|/U | 0.0043 | 0.03 | ✓ |
| C2 max|Δv|/U | 0.0112 | 0.03 | ✓ |
| C3 u_min relative | 0.65% | 3% | ✓ |
| C4 v_max relative | 0.78% | 3% | ✓ |
| C5 vortex centre | 0.0040L | 0.01L | ✓ |
| C6 max|div u| | 1.2e-13 | 1e-10 | ✓ |
| C7 steady residual | 9.9e-08 | 1e-06 | ✓ |

## Evidence Chain

1. **Run artefacts**: `docs/CASE-FVM-CAVITY-RE1000/runs/` (summary.json, benchmark.json, fields.npz, per-resolution JSON)
2. **Machine gate tests**: `tests/test_case_cavity_re1000.py` (7 passed) — re-derives all criteria from stored artefacts
3. **Registered gate tests**: `tests/test_case_cavity_re1000_gate_registered.py` (11 passed) — asserts code matches DB row
4. **DB registration**: `tensor_benchmark_cases` id=146, anchor_slug=CASE-FVM-CAVITY-RE1000, status=passed
5. **Comprehensive report**: `docs/CASE-FVM-CAVITY-RE1000/REPORT.md` with figures
6. **Platform report**: id=20359, published via API

## Grid Convergence

| n | u_min rel | v_max rel | max|Δu| | max|Δv| | Verdict |
|---|---|---|---|---|---|
| 32 | 17.36% | 14.96% | 0.2033 | 0.0754 | FAIL |
| 64 | 2.80% | 2.34% | 0.0293 | 0.0197 | PASS |
| 128 | 0.65% | 0.78% | 0.0043 | 0.0112 | PASS |

Observed convergence order (32→64→128): p(Linf u)=2.57, p(Linf v)=1.51

## GCI Uncertainty

| Functional | Observed p | GCI(fine) | Fine vs Ghia | Ref in GCI band |
|---|---|---|---|---|
| u_min | 1.90 | 1.45% | 0.63% | ✓ |
| v_max | 1.98 | 1.29% | 0.79% | ✓ |

## Feature Ledger Update

Status updated from `open` to `done` in `feature_ledger` table.

## Outstanding

- n=256 cluster job (cj4278) submitted for extended grid convergence; not required for acceptance.
