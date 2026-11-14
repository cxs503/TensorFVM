# CASE-CYLINDER-RE100: Unbounded Cylinder Vortex Shedding at Re=100

## Anchor

`GAP-DIFFLBM-EXT-LIT-CYL` — JOUR-DIFFLBM 第3个文献锚：圆柱绕流 Re=100 的 Cd/St

## Acceptance Criteria

| Quantity | Reference | Tolerance | Source |
|----------|-----------|-----------|--------|
| Cd_mean  | 1.34      | <5%       | Braza 1986; Schafer-Turek 1996 |
| St       | 0.164     | <3%       | Braza 1986; Schafer-Turek 1996 |

## Physical Problem

2D incompressible flow past a circular cylinder at Re=100 (based on diameter D and free-stream velocity U_inf). The flow is unsteady with periodic von Karman vortex shedding.

- Domain: 20D x 20D, cylinder centered at (10D, 10D), D=1.0 m
- Boundary conditions: uniform inlet U=1.0 m/s, far-field outer boundary, no-slip cylinder
- Fluid: rho=1.0 kg/m3, nu=0.01 m2/s (Re=UD/nu=100)
- Solver: body-fitted O-grid SIMPLE/URANS, implicit Euler time stepping

## Literature References

1. **Braza, P., Chassaing, P. & Ha Minh, H. (1986).** "Numerical study and physical analysis of the pressure and velocity fields in the near wake of a circular cylinder." *J. Fluid Mech.* 165, 79-130. doi:10.1017/S0022112086003014

2. **Schafer, M. & Turek, S. (1996).** "Benchmark Computations of Laminar Flow Around a Cylinder." In: Hirschel, E.H. et al. (eds) *Flow Simulation with High-Performance Computers II*. Notes on Numerical Fluid Mechanics, vol 52. Vieweg+Teubner. doi:10.1007/978-3-322-89849-4_39 (DFG 2D-2 benchmark)

## Reproduction

```bash
# Short test (verifies code path)
PYTHONPATH=src python -m pytest tests/test_case_cylinder_re100.py -v

# Production run (cluster)
PYTHONPATH=src python -m tensorfvm.verification.cylinder_re100 \
    --output results/cylinder-re100 \
    --grid 128x48 --steps 3000 --dt 0.005 --warmup 500
```

## Files

- `src/tensorfvm/verification/cylinder_re100.py` — verification module
- `artifacts/case-cylinder-re100/reference/reference.json` — literature reference with provenance
- `tests/test_case_cylinder_re100.py` — contract tests
- `docs/case-cylinder-re100/README.md` — this file

## Status

- [x] Verification module created
- [x] Reference file with DOI provenance
- [x] Contract tests (6/6 passing)
- [ ] Production run (cluster submission pending)
- [ ] Final report with figures
